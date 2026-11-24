"""Stochastic edge-of-stability experiment based on Cohen et al.'s src/ files.

USAGE: Copy these sgdm_eos*.py files into the ORIGINAL repository src/ folder
and run `python src/sgdm_eos.py ...` from the repository root.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset

from archs import load_architecture       # ORIGINAL Cohen et al. src/archs.py
from data import load_dataset           # ORIGINAL Cohen et al. src/data.py
from utilities import get_loss_and_acc  # ORIGINAL Cohen et al. src/utilities.py
from sgdm_eos_diagnostics import diagnostics, sharpness_only_diagnostics


def args_parser():
    ap = argparse.ArgumentParser(description='CIFAR10 5k EoS: SGD / effective-LR SGD / Polyak SGDM')
    ap.add_argument('--output', default='sgdm_eos_results')
    ap.add_argument('--arch', default='fc-tanh', help='Original src/archs.py architecture name')
    ap.add_argument('--loss', choices=('mse', 'ce'), default='mse')
    ap.add_argument('--eta', type=float, default=0.01, help='Common raw learning rate for SGD and SGDM')
    ap.add_argument('--mus', type=float, nargs='+', default=[0.0, 0.5, 0.8, 0.9])
    ap.add_argument('--kinds', nargs='+', choices=('sgd', 'scaled_sgd', 'sgdm'),
                    default=['sgd', 'scaled_sgd', 'sgdm'],
                    help='Optimizers to run. Use --kinds sgdm for SGDM-only sweeps.')
    ap.add_argument('--batch-size', type=int, default=200)
    ap.add_argument('--steps', type=int, default=15000, help='One step = one mini-batch update')
    ap.add_argument('--subset-seed', type=int, default=2026)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--eval-every', type=int, default=100)
    ap.add_argument('--diag-every', type=int, default=500)
    ap.add_argument('--physical-batch-size', type=int, default=250)
    ap.add_argument('--noise-batches', type=int, default=50)
    ap.add_argument('--eig-tol', type=float, default=1e-3)
    ap.add_argument('--eig-maxiter', type=int, default=120)
    ap.add_argument('--theory-order', choices=(1, 2, 0), default=1, type=int,
                    help='1=leading gap; 2=+ next Taylor term; 0=full sqrt closure')
    # To enable the additional term from the boxed Taylor expansion on draft p.57:
    # set --theory-order 2 (or uncomment the next line to change the default).
    # ap.set_defaults(theory_order=2)
    ap.add_argument('--device', default='auto', help='auto / cpu / cuda / cuda:0')
    ap.add_argument('--no-diagnostics', action='store_true', help='Loss-only quick smoke test')
    ap.add_argument('--sharpness-only', action='store_true',
                    help='Compute only the top Hessian eigenvalue for every optimizer')
    ap.add_argument('--full-diagnostics-all-kinds', action='store_true',
                    help='Also compute alpha, beta and projected noise for scaled_sgd')
    ap.add_argument('--save-checkpoints', action='store_true')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--deterministic', action='store_true')
    return ap


def check_args(a):
    if a.eta <= 0 or a.batch_size < 1 or a.steps < 1 or a.eval_every < 1 or a.diag_every < 1:
        raise ValueError('eta, batch-size, steps, eval-every, diag-every must all be positive')
    if a.physical_batch_size <= 0 or a.noise_batches < 2:
        raise ValueError('physical-batch-size>0 and noise-batches>=2 are required')
    if not a.mus or any(not 0 <= m < 1 for m in a.mus):
        raise ValueError('mus must be in [0,1)')
    if len(set(a.mus)) != len(a.mus) or len(set(a.seeds)) != len(a.seeds):
        raise ValueError('mus and seeds must not contain duplicates')


def select_random_5k(train, seed):
    if len(train) < 5000:
        raise ValueError('Expected full 50k CIFAR-10 training dataset; load_dataset("cifar10",loss)')
    idx = np.random.default_rng(seed).choice(len(train), size=5000, replace=False)
    assert len(set(idx.tolist())) == 5000
    return TensorDataset(*(t[idx].contiguous() for t in train.tensors)), idx


def eval_objective(model, dataset, loss_fn, accuracy_fn, physical_batch_size, device):
    """Compute exact mean-per-example loss and accuracy over a FIXED train/test set."""
    model.eval()
    loss_total = 0.
    acc_total = 0.
    with torch.no_grad():
        x, y = dataset.tensors
        for start in range(0, len(dataset), physical_batch_size):
            xb = x[start:start + physical_batch_size].to(device)
            yb = y[start:start + physical_batch_size].to(device)
            outputs = model(xb)
            loss_total += loss_fn(outputs, yb).item()
            acc_total += accuracy_fn(outputs, yb).item()
    return loss_total / len(dataset), acc_total / len(dataset)


def seed_all(seed, deterministic):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _tag(x):
    return format(float(x), '.6g').replace('.', 'p')


def experiment_specs(a):
    # Each comparison has the SAME model initialization and SAME mini-batch stream.
    # Explicit mu=0 SGDM is retained as an implementation-equivalence control.
    if 'sgd' in a.kinds:
        yield ('sgd', 0.0, a.eta)
    for mu in a.mus:
        if 'scaled_sgd' in a.kinds:
            yield ('scaled_sgd', mu, a.eta / (1 - mu))
        if 'sgdm' in a.kinds:
            yield ('sgdm', mu, a.eta)


def run_one(a, train, test, indices_digest, kind, mu, lr, seed, device):
    # Include batch size because batch-size sweeps share one output directory.
    run_name = f'{kind}_mu{_tag(mu)}_eta{_tag(lr)}_b{a.batch_size}_seed{seed}'
    path = Path(a.output) / run_name
    complete = path / '_COMPLETE'
    if complete.exists() and not a.overwrite:
        print(f'SKIP completed: {path}', flush=True)
        return
    path.mkdir(parents=True, exist_ok=True)
    seed_all(seed, a.deterministic)
    model = load_architecture(a.arch, 'cifar10-5k').to(device)
    model.train()
    loss_fn, acc_fn = get_loss_and_acc(a.loss)
    actual_mu = mu if kind == 'sgdm' else 0.0
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=actual_mu,
                                nesterov=False, weight_decay=0, dampening=0)
    # Independent data-stream RNG. Reset to same seed for ALL methods within seed.
    minibatch_rng = np.random.default_rng(seed + 197052)
    X, y = train.tensors
    configs = {'kind': kind, 'mu_parameter': mu, 'optimizer_momentum': actual_mu,
               'lr': lr, 'reference_eta': a.eta, 'batch_size': a.batch_size,
               'steps': a.steps, 'seed': seed, 'subset_seed': a.subset_seed,
               'subset_sha256': indices_digest, 'arch': a.arch, 'loss': a.loss,
               'eval_every': a.eval_every, 'diag_every': a.diag_every,
               'physical_batch_size': a.physical_batch_size,
               'noise_batches': a.noise_batches, 'theory_order': a.theory_order,
               'sharpness_only': a.sharpness_only,
               'full_diagnostics_all_kinds': a.full_diagnostics_all_kinds,
               'torch_version': torch.__version__, 'device': str(device)}
    (path / 'config.json').write_text(json.dumps(configs, indent=2))
    print(f'RUN {run_name} ({device})', flush=True)

    # Evaluate initial state. Scaled SGD gets sharpness-only diagnostics.
    trajectory = []
    diag_rows = []
    warm_u = None
    rng_diag_seed = 3413 + 100000 * seed
    start_wall = time.time()

    def record(step):
        nonlocal warm_u
        train_loss, train_acc = eval_objective(model, train, loss_fn, acc_fn,
                                               a.physical_batch_size, device)
        test_loss, test_acc = eval_objective(model, test, loss_fn, acc_fn,
                                             a.physical_batch_size, device)
        trajectory.append({'step': step, 'train_loss': train_loss, 'test_loss': test_loss,
                           'train_acc': train_acc, 'test_acc': test_acc,
                           'elapsed_seconds': time.time() - start_wall})
        if not all(math.isfinite(v) for v in (train_loss, train_acc, test_loss, test_acc)):
            raise FloatingPointError(f'Nonfinite full loss/accuracy at step {step}, {run_name}')
        if not a.no_diagnostics and kind in ('sgd', 'sgdm', 'scaled_sgd') and (step == 0 or step % a.diag_every == 0 or step == a.steps):
            if a.sharpness_only or (kind == 'scaled_sgd' and not a.full_diagnostics_all_kinds):
                # This optimizer has zero momentum.  Pass its actual learning
                # rate so the recorded boundary is 2/lr = 2(1-mu)/eta.
                d, warm_u = sharpness_only_diagnostics(
                    model, loss_fn, train, eta=lr,
                    physical_batch_size=a.physical_batch_size,
                    seed=rng_diag_seed + step, device=device,
                    eig_tol=a.eig_tol, eig_maxiter=a.eig_maxiter,
                    warm_start=warm_u)
            else:
                d, warm_u = diagnostics(model, loss_fn, train, eta=lr, mu=actual_mu,
                                        batch_size=min(a.batch_size, len(train)),
                                        physical_batch_size=a.physical_batch_size,
                                        n_noise_batches=a.noise_batches,
                                        seed=rng_diag_seed + step, device=device,
                                        eig_tol=a.eig_tol, eig_maxiter=a.eig_maxiter,
                                        warm_start=warm_u, theory_order=a.theory_order)
            diag_rows.append({'step': step, **d})
            if a.sharpness_only or kind == 'scaled_sgd':
                print(f'  step={step} train={train_loss:.5g} test={test_loss:.5g} '
                      f'S={d["sharpness"]:.4g} boundary={d["boundary"]:.4g}', flush=True)
            else:
                print(f'  step={step} train={train_loss:.5g} test={test_loss:.5g} '
                      f'S={d["sharpness"]:.4g} pred={d["predicted"]:.4g} '
                      f'alpha={d["alpha"]:.3g} q={d["q"]:.3g}', flush=True)
        elif step % max(a.eval_every, 100) == 0:
            print(f'  step={step} train={train_loss:.5g} test={test_loss:.5g}', flush=True)
        write_csv(path / 'losses.csv', trajectory)
        if diag_rows:
            write_csv(path / 'diagnostics.csv', diag_rows)

    record(0)
    for step in range(1, a.steps + 1):
        # Per-step independent uniformly random subset WITHOUT replacement inside minibatch.
        ids = minibatch_rng.choice(len(train), size=min(a.batch_size, len(train)), replace=False)
        xb = X[ids].to(device)
        yb = y[ids].to(device)
        optimizer.zero_grad(set_to_none=True)
        f = loss_fn(model(xb), yb) / len(ids)
        if not bool(torch.isfinite(f)):
            raise FloatingPointError(f'Nonfinite mini-batch loss at step {step} ({run_name})')
        f.backward()
        optimizer.step()
        if step % a.eval_every == 0 or (not a.no_diagnostics and step % a.diag_every == 0) or step == a.steps:
            record(step)
    if a.save_checkpoints:
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'step': a.steps, 'config': configs}, path / 'final.pt')
    complete.write_text('finished\n')
    del model, optimizer
    if device.type == 'cuda':
        torch.cuda.empty_cache()


def write_csv(filename, records):
    if not records:
        return
    tmp = filename.with_suffix('.tmp')
    with tmp.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    os.replace(tmp, filename)


def main():
    a = args_parser().parse_args()
    check_args(a)
    device = torch.device('cuda' if a.device == 'auto' and torch.cuda.is_available()
                          else 'cpu' if a.device == 'auto' else a.device)
    Path(a.output).mkdir(parents=True, exist_ok=True)
    print('Loading ORIGINAL Cohen dataset with full-train preprocessing...', flush=True)
    original_train, test = load_dataset('cifar10', a.loss)
    train, idx = select_random_5k(original_train, a.subset_seed)
    digest = hashlib.sha256(idx.astype(np.int64).tobytes()).hexdigest()
    np.save(Path(a.output) / 'train_5k_indices.npy', idx)
    (Path(a.output) / 'manifest.json').write_text(json.dumps({
        'source': 'locuslab/edge-of-stability, default branch github',
        'sampling': 'uniform 5000 from all original CIFAR10 train images, without replacement',
        'preprocessing': 'original load_dataset("cifar10", loss): full 50k train-based normalisation',
        'subset_seed': a.subset_seed, 'indices_sha256': digest,
        'seed_list': a.seeds, 'mu_list': a.mus, 'eta': a.eta,
        'batch_size': a.batch_size, 'requested_steps': a.steps,
        'scaled_sgd_definition': 'standard SGD with lr=eta/(1-mu), zero momentum',
        'theory_order': a.theory_order,
    }, indent=2))
    del original_train
    print(f'Random subset fingerprint SHA256: {digest}', flush=True)
    for seed in a.seeds:
        for kind, mu, lr in experiment_specs(a):
            run_one(a, train, test, digest, kind, mu, lr, seed, device)
    print('Finished. Run: python src/sgdm_eos_plot.py --input ' + str(a.output))


if __name__ == '__main__':
    main()
