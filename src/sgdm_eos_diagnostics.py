"""Cohen-style full-objective Hessian diagnostics plus the SGDM sharpness gap.

Drop alongside the original edge-of-stability src/{gd,data,archs,utilities}.py.
No backpropagation graph is retained across minibatches or optimisation steps.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
from scipy.sparse.linalg import LinearOperator, eigsh
from torch.nn.utils import parameters_to_vector
from torch.utils.data import TensorDataset


def _parameters(model):
    return tuple(p for p in model.parameters() if p.requires_grad)


def _flat(tensors):
    return torch.cat([t.reshape(-1) for t in tensors])


def _parts(vector, parameters):
    return tuple(v.reshape_as(p) for v, p in zip(torch.split(vector, [p.numel() for p in parameters]), parameters))


def _batches(ds, batch_size):
    X, y = ds.tensors
    for begin in range(0, len(ds), batch_size):
        yield X[begin:begin + batch_size], y[begin:begin + batch_size]


def full_gradient(model, loss_fn, dataset, physical_batch_size, device):
    """Gradient of N^-1 sum_i ell_i, not of the sum or a typical minibatch."""
    params = _parameters(model)
    output = torch.zeros(sum(p.numel() for p in params), device=device)
    for x, y in _batches(dataset, physical_batch_size):
        f = loss_fn(model(x.to(device)), y.to(device)) / len(dataset)
        grads = torch.autograd.grad(f, params)
        output += _flat(grads).detach()
    return output


def hessian_vector_product(model, loss_fn, dataset, vector, physical_batch_size, device):
    """Exact autograd Hessian-vector product on a fixed objective, using chunks."""
    params = _parameters(model)
    vector = vector.to(device=device, dtype=params[0].dtype).detach()
    chunks = _parts(vector, params)
    out = torch.zeros_like(vector)
    for x, y in _batches(dataset, physical_batch_size):
        loss = loss_fn(model(x.to(device)), y.to(device)) / len(dataset)
        g = torch.autograd.grad(loss, params, create_graph=True)
        gv = sum((gi * vi).sum() for gi, vi in zip(g, chunks))
        hv = torch.autograd.grad(gv, params)
        out += _flat(hv).detach()
    return out


def top_eigenpair(model, loss_fn, dataset, physical_batch_size, device, tol=1e-3,
                  maxiter=120, seed=0, warm_start=None):
    """Largest *algebraic* eigenpair (not largest absolute eigenvalue), via SciPy Lanczos."""
    params = _parameters(model)
    n = sum(p.numel() for p in params)
    rng = np.random.default_rng(seed)
    v0 = (rng.normal(size=n) if warm_start is None else np.asarray(warm_start)).astype(np.float64)
    v0 /= np.linalg.norm(v0)

    def mv(arr):
        v = torch.from_numpy(np.asarray(arr, dtype=np.float32).copy()).to(device)
        h = hessian_vector_product(model, loss_fn, dataset, v, physical_batch_size, device)
        return h.detach().cpu().numpy().astype(np.float64)

    op = LinearOperator((n, n), matvec=mv, dtype=np.float64)
    vals, vecs = eigsh(op, k=1, which='LA', tol=tol, maxiter=maxiter, v0=v0)
    u = torch.from_numpy(vecs[:, 0].copy()).to(device=device, dtype=params[0].dtype)
    u = u / u.norm()
    top = float(vals[0])
    residual = (hessian_vector_product(model, loss_fn, dataset, u, physical_batch_size, device)
                - top * u).norm().item() / max(abs(top), 1.0)
    return top, u.detach(), float(residual)


def sharpness_gradient_and_alpha_beta(model, loss_fn, dataset, u, physical_batch_size, device):
    """Use stop-gradient eigenvector u to form dS = ∇_theta(u^T H(theta)u).

    For a simple eigenvalue, this is the genuine eigenvalue derivative, provided
    the supplied eigenvector is accurate. Beta uses its component orthogonal to u,
    as in the revised definition in the user's draft (p.56).
    """
    params = _parameters(model)
    uc = _parts(u.detach().to(device), params)
    ds = torch.zeros_like(u)
    full_g = torch.zeros_like(u)
    for x, y in _batches(dataset, physical_batch_size):
        loss = loss_fn(model(x.to(device)), y.to(device)) / len(dataset)
        first = torch.autograd.grad(loss, params, create_graph=True)
        full_g += _flat(first).detach()
        directional = sum((g * v).sum() for g, v in zip(first, uc))
        second = torch.autograd.grad(directional, params, create_graph=True)
        rayleigh = sum((h * v).sum() for h, v in zip(second, uc))
        third = torch.autograd.grad(rayleigh, params)
        ds += _flat(third).detach()
    ds_perp = ds - torch.dot(ds, u) * u
    alpha = -torch.dot(full_g, ds).item()
    beta = torch.dot(ds_perp, ds_perp).item()
    return float(alpha), float(beta), float(torch.dot(ds, u).item()), float(full_g.norm().item())


def projected_minibatch_variance(model, loss_fn, dataset, u, batch_size, n_batches,
                                 physical_batch_size, seed, device):
    """Unbiased sample variance of <g_B,u> over independent random size-b subsets.

    Individual batches sample without replacement; samples across separate draws
    are independent. This captures the finite-population correction automatically.
    """
    if batch_size >= len(dataset):
        return 0.0
    if n_batches < 2:
        raise ValueError('noise_batches must be >= 2')
    params = _parameters(model)
    rng = np.random.default_rng(seed)
    X, y = dataset.tensors
    uc = _parts(u.detach(), params)
    scores = []
    for _ in range(n_batches):
        chosen = rng.choice(len(dataset), size=batch_size, replace=False)
        v = 0.0
        for idx in np.array_split(chosen, max(1, int(np.ceil(batch_size / physical_batch_size)))):
            inputs = X[idx].to(device)
            targets = y[idx].to(device)
            loss = loss_fn(model(inputs), targets) / batch_size
            g = torch.autograd.grad(loss, params)
            v += sum((gi.detach() * ui).sum().item() for gi, ui in zip(g, uc))
        scores.append(v)
    return float(np.var(scores, ddof=1))


def predict_sharpness(eta: float, mu: float, alpha: float, beta: float,
                      sigma_u2: float, order: int = 1) -> Dict[str, float]:
    """Frozen-coefficient equilibrium prediction of draft page 57.

    order=1: leading Taylor term (default).
    order=2: adds the quadratic term in q to sqrt(1-q).
    order=0: exact square-root mean-field closure, when q<=1.
    Invalid alpha, beta or expansion domain returns NaNs, never silent clipping.
    """
    if eta <= 0 or not 0 <= mu < 1 or order not in (0, 1, 2):
        raise ValueError('Need eta>0, 0<=mu<1, order in {0,1,2}')
    boundary = 2 * (1 + mu) / eta
    nan = {'boundary': boundary, 'gap': np.nan, 'predicted': np.nan,
           'q': np.nan, 'second_term': np.nan, 'exact_predicted': np.nan}
    if not all(np.isfinite(z) for z in (alpha, beta, sigma_u2)) or alpha <= 0 or beta <= 0 or sigma_u2 < 0:
        return nan
    delta2 = 2 * alpha / beta
    q = eta**2 * sigma_u2 / ((1 - mu**2) * delta2)
    first = eta * beta * sigma_u2 / (4 * alpha * (1 - mu))
    second = eta**3 * beta**2 * sigma_u2**2 / (32 * alpha**2 * (1-mu)**2 * (1+mu))
    exact = (boundary + (1+mu)/eta * (np.sqrt(1 - q) - 1)) if q <= 1 else np.nan
    if order == 0:
        gap = boundary - exact
    elif order == 1:
        gap = first
    else:
        gap = first + second
    return {'boundary': boundary, 'gap': float(gap), 'predicted': float(boundary-gap),
            'q': float(q), 'second_term': float(second), 'exact_predicted': float(exact)}


def diagnostics(model, loss_fn, dataset, eta, mu, batch_size, physical_batch_size,
                n_noise_batches, seed, device, eig_tol=1e-3, eig_maxiter=120, warm_start=None,
                theory_order=1):
    top, u, residual = top_eigenpair(model, loss_fn, dataset, physical_batch_size,
                                    device, tol=eig_tol, maxiter=eig_maxiter,
                                    seed=seed, warm_start=warm_start)
    alpha, beta, kappa, grad_norm = sharpness_gradient_and_alpha_beta(
        model, loss_fn, dataset, u, physical_batch_size, device)
    var = projected_minibatch_variance(
        model, loss_fn, dataset, u, batch_size, n_noise_batches,
        physical_batch_size, seed + 173, device)
    prediction = predict_sharpness(eta, mu, alpha, beta, var, order=theory_order)
    actual_gap = prediction['boundary'] - top
    data = dict(sharpness=top, eig_residual=residual, alpha=alpha, beta=beta,
                kappa=kappa, grad_norm=grad_norm, sigma_u2=var,
                observed_gap=actual_gap, theory_valid=int(np.isfinite(prediction['predicted'])),
                q_warning=int(np.isfinite(prediction['q']) and prediction['q'] > 0.1))
    data.update(prediction)
    return data, u.cpu().numpy()


def sharpness_only_diagnostics(model, loss_fn, dataset, eta, physical_batch_size,
                               seed, device, eig_tol=1e-3, eig_maxiter=120,
                               warm_start=None):
    """Top full-objective Hessian eigenvalue for ordinary/scaled SGD.

    ``eta`` is the learning rate actually used by the optimizer.  Consequently
    the appropriate ordinary-SGD stability boundary is 2 / eta.  The NaN
    fields keep diagnostics.csv compatible with the common plotting loader
    without paying for third derivatives or gradient-noise estimation.
    """
    top, u, residual = top_eigenpair(
        model, loss_fn, dataset, physical_batch_size, device,
        tol=eig_tol, maxiter=eig_maxiter, seed=seed,
        warm_start=warm_start)
    boundary = 2.0 / eta
    nan = float('nan')
    data = dict(
        sharpness=top, eig_residual=residual,
        alpha=nan, beta=nan, kappa=nan, grad_norm=nan,
        sigma_u2=nan, observed_gap=boundary - top,
        theory_valid=0, q_warning=0, boundary=boundary,
        gap=nan, predicted=nan, q=nan, second_term=nan,
        exact_predicted=nan,
    )
    return data, u.cpu().numpy()
