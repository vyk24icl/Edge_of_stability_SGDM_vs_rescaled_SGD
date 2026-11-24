"""Plot the four SGDM comparison studies from saved experiment CSV files.

The studies intentionally plot measured losses and measured full-objective
sharpness only. No predicted sharpness or threshold-minus-theory gap appears.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def csv_rows(file):
    if not file.exists():
        return []
    with file.open(newline='') as f:
        return [
            {k: (float(v) if k != 'step' else int(v)) for k, v in row.items()}
            for row in csv.DictReader(f)
        ]


def find_runs(root):
    runs = []
    for file in sorted(root.glob('*/config.json')):
        cfg = json.loads(file.read_text())
        path = file.parent
        losses = csv_rows(path / 'losses.csv')
        if losses:
            runs.append({
                'config': cfg,
                'path': path,
                'losses': losses,
                'diagnostics': csv_rows(path / 'diagnostics.csv'),
                'complete': (path / '_COMPLETE').exists(),
            })
    return runs


def _matches(value, expected):
    if isinstance(expected, (float, int)) and isinstance(value, (float, int)):
        return np.isclose(float(value), float(expected))
    return value == expected


def select_runs(runs, **filters):
    return [
        run for run in runs
        if all(_matches(run['config'].get(key), value) for key, value in filters.items())
    ]


def grouped_mean(runs, field, section, **filters):
    subset = [run for run in select_runs(runs, **filters) if run[section]]
    if not subset:
        return None
    common = sorted(set.intersection(*(
        {int(row['step']) for row in run[section]} for run in subset
    )))
    if not common:
        return None
    values = []
    for run in subset:
        mapping = {int(row['step']): row.get(field, np.nan) for row in run[section]}
        values.append([mapping[step] for step in common])
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values).sum(axis=0)
    mean = np.asarray([
        np.nanmean(values[:, j]) if valid[j] else np.nan
        for j in range(len(common))
    ])
    sd = np.asarray([
        np.nanstd(values[:, j], ddof=1) if valid[j] > 1 else 0.0
        for j in range(len(common))
    ])
    return np.asarray(common), mean, sd, len(subset)


def add_line(ax, result, label, color, linestyle='-', show_sd=True, lw=1.8):
    if result is None:
        return
    x, y, sd, n = result
    ax.plot(x, y, linestyle=linestyle, label=label, color=color, linewidth=lw)
    if show_sd and n > 1:
        ax.fill_between(x, y - sd, y + sd, color=color, alpha=.13, linewidth=0)


def unique_config(runs, key, kind=None):
    subset = runs if kind is None else select_runs(runs, kind=kind)
    return sorted({float(run['config'][key]) for run in subset})


def require_one(values, label, study):
    if len(values) != 1:
        raise ValueError(
            f'Study {study} requires one fixed {label}, found {values}. '
            'Use a separate --input directory for each study.'
        )
    return values[0]


def style_loss_axis(ax, title):
    ax.set(
        xlabel='Minibatch parameter updates',
        ylabel='Mean loss per example',
        title=title,
    )
    ax.set_yscale('log')
    ax.grid(alpha=.22)
    ax.legend(fontsize=9)


def style_accuracy_axis(ax, title):
    ax.set(
        xlabel='Minibatch parameter updates',
        ylabel='Training accuracy',
        title=title,
    )
    ax.set_ylim(-.02, 1.02)
    ax.grid(alpha=.22)
    ax.legend(fontsize=9)


def style_sharpness_axis(ax, title):
    ax.set(
        xlabel='Minibatch parameter updates',
        ylabel=r'Full-objective sharpness $\lambda_{\max}(\nabla^2 L)$',
        title=title,
    )
    ax.grid(alpha=.22)
    ax.legend(fontsize=9)


def plot_study_1(runs, figures):
    """SGD, rescaled SGD and SGDM losses at fixed eta/batch, varying mu."""
    eta = require_one(unique_config(runs, 'reference_eta'), 'reference eta', 1)
    batch = require_one(unique_config(runs, 'batch_size'), 'batch size', 1)
    sgdm_mus = set(unique_config(runs, 'mu_parameter', kind='sgdm'))
    scaled_mus = set(unique_config(runs, 'mu_parameter', kind='scaled_sgd'))
    mus = sorted(sgdm_mus & scaled_mus)
    if not mus:
        raise ValueError('Study 1 needs paired SGDM and scaled_sgd runs.')
    colors = plt.cm.viridis(np.linspace(.12, .88, len(mus)))

    for metric, name, title in (
        ('train_loss', '01_train_loss_optimizer_mu.png', 'Training loss'),
        ('test_loss', '01_test_loss_optimizer_mu.png', 'Test loss'),
    ):
        fig, ax = plt.subplots(figsize=(9.2, 5.4))
        add_line(
            ax, grouped_mean(runs, metric, 'losses', kind='sgd'),
            'SGD', 'black', lw=2.4,
        )
        for mu, color in zip(mus, colors):
            add_line(
                ax,
                grouped_mean(runs, metric, 'losses', kind='sgdm', mu_parameter=mu),
                rf'SGDM, $\mu={mu:g}$', color, '-', lw=2.0,
            )
            add_line(
                ax,
                grouped_mean(runs, metric, 'losses', kind='scaled_sgd', mu_parameter=mu),
                rf'Rescaled SGD, $\mu={mu:g}$', color, '--', lw=1.8,
            )
        style_loss_axis(ax, rf'{title}: $\eta={eta:g}$, $b={batch:g}$')
        fig.tight_layout()
        fig.savefig(figures / name, dpi=200, bbox_inches='tight')
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    add_line(
        ax, grouped_mean(runs, 'train_acc', 'losses', kind='sgd'),
        'SGD', 'black', lw=2.4,
    )
    for mu, color in zip(mus, colors):
        add_line(
            ax,
            grouped_mean(runs, 'train_acc', 'losses', kind='sgdm', mu_parameter=mu),
            rf'SGDM, $\mu={mu:g}$', color, '-', lw=2.0,
        )
        add_line(
            ax,
            grouped_mean(runs, 'train_acc', 'losses', kind='scaled_sgd', mu_parameter=mu),
            rf'Rescaled SGD, $\mu={mu:g}$', color, '--', lw=1.8,
        )
    style_accuracy_axis(ax, rf'Training accuracy: $\eta={eta:g}$, $b={batch:g}$')
    fig.tight_layout()
    fig.savefig(figures / '01_train_accuracy_optimizer_mu.png', dpi=200,
                bbox_inches='tight')
    plt.close(fig)


def plot_study_2(runs, figures):
    """SGDM training loss and sharpness at fixed mu/batch, varying eta."""
    sgdm = select_runs(runs, kind='sgdm')
    mu = require_one(unique_config(sgdm, 'mu_parameter'), 'momentum', 2)
    batch = require_one(unique_config(sgdm, 'batch_size'), 'batch size', 2)
    etas = unique_config(sgdm, 'reference_eta')
    if len(etas) < 2:
        raise ValueError('Study 2 needs at least two SGDM step sizes.')
    colors = plt.cm.plasma(np.linspace(.08, .82, len(etas)))

    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    for eta, color in zip(etas, colors):
        add_line(
            ax,
            grouped_mean(sgdm, 'train_loss', 'losses', reference_eta=eta),
            rf'$\eta={eta:g}$', color,
        )
    style_loss_axis(ax, rf'SGDM training loss: $\mu={mu:g}$, $b={batch:g}$')
    fig.tight_layout()
    fig.savefig(figures / '02_train_loss_sgdm_vary_eta.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    for eta, color in zip(etas, colors):
        add_line(
            ax,
            grouped_mean(sgdm, 'train_acc', 'losses', reference_eta=eta),
            rf'$\eta={eta:g}$', color,
        )
    style_accuracy_axis(ax, rf'SGDM training accuracy: $\mu={mu:g}$, $b={batch:g}$')
    fig.tight_layout()
    fig.savefig(figures / '02_train_accuracy_sgdm_vary_eta.png', dpi=200,
                bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    for eta, color in zip(etas, colors):
        add_line(
            ax,
            grouped_mean(sgdm, 'sharpness', 'diagnostics', reference_eta=eta),
            rf'Sharpness, $\eta={eta:g}$', color,
        )
        threshold = 2 * (1 + mu) / eta
        ax.axhline(
            threshold, color=color, linestyle=':', linewidth=1.35,
            label=rf'$2(1+\mu)/\eta={threshold:.1f}$ for $\eta={eta:g}$',
        )
    style_sharpness_axis(ax, rf'SGDM sharpness: $\mu={mu:g}$, $b={batch:g}$')
    fig.tight_layout()
    fig.savefig(figures / '02_sharpness_sgdm_vary_eta.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_study_3(runs, figures):
    """SGDM training loss and sharpness at fixed eta/batch, varying mu."""
    sgdm = select_runs(runs, kind='sgdm')
    eta = require_one(unique_config(sgdm, 'reference_eta'), 'reference eta', 3)
    batch = require_one(unique_config(sgdm, 'batch_size'), 'batch size', 3)
    mus = unique_config(sgdm, 'mu_parameter')
    if len(mus) < 2:
        raise ValueError('Study 3 needs at least two SGDM momentum values.')
    colors = plt.cm.viridis(np.linspace(.12, .88, len(mus)))

    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    for mu, color in zip(mus, colors):
        add_line(
            ax,
            grouped_mean(sgdm, 'train_loss', 'losses', mu_parameter=mu),
            rf'$\mu={mu:g}$', color,
        )
    style_loss_axis(ax, rf'SGDM training loss: $\eta={eta:g}$, $b={batch:g}$')
    fig.tight_layout()
    fig.savefig(figures / '03_train_loss_sgdm_vary_mu.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    for mu, color in zip(mus, colors):
        add_line(
            ax,
            grouped_mean(sgdm, 'train_acc', 'losses', mu_parameter=mu),
            rf'$\mu={mu:g}$', color,
        )
    style_accuracy_axis(ax, rf'SGDM training accuracy: $\eta={eta:g}$, $b={batch:g}$')
    fig.tight_layout()
    fig.savefig(figures / '03_train_accuracy_sgdm_vary_mu.png', dpi=200,
                bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    for mu, color in zip(mus, colors):
        add_line(
            ax,
            grouped_mean(sgdm, 'sharpness', 'diagnostics', mu_parameter=mu),
            rf'Sharpness, $\mu={mu:g}$', color,
        )
        threshold = 2 * (1 + mu) / eta
        ax.axhline(
            threshold, color=color, linestyle=':', linewidth=1.35,
            label=rf'$2(1+\mu)/\eta={threshold:.1f}$ for $\mu={mu:g}$',
        )
    style_sharpness_axis(ax, rf'SGDM sharpness: $\eta={eta:g}$, $b={batch:g}$')
    fig.tight_layout()
    fig.savefig(figures / '03_sharpness_sgdm_vary_mu.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_study_4(runs, figures):
    """SGDM sharpness at fixed eta/mu, varying batch size."""
    sgdm = select_runs(runs, kind='sgdm')
    eta = require_one(unique_config(sgdm, 'reference_eta'), 'reference eta', 4)
    mu = require_one(unique_config(sgdm, 'mu_parameter'), 'momentum', 4)
    batches = unique_config(sgdm, 'batch_size')
    if len(batches) < 2:
        raise ValueError('Study 4 needs at least two SGDM batch sizes.')
    colors = plt.cm.cividis(np.linspace(.08, .9, len(batches)))

    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    for batch, color in zip(batches, colors):
        add_line(
            ax,
            grouped_mean(sgdm, 'sharpness', 'diagnostics', batch_size=batch),
            rf'$b={batch:g}$', color,
        )
    threshold = 2 * (1 + mu) / eta
    ax.axhline(
        threshold, color='black', linestyle='--', linewidth=1.5,
        label=rf'$2(1+\mu)/\eta={threshold:.1f}$',
    )
    style_sharpness_axis(ax, rf'SGDM sharpness: $\eta={eta:g}$, $\mu={mu:g}$')
    fig.tight_layout()
    fig.savefig(figures / '04_sharpness_sgdm_vary_batch.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


def equilibrium_mean(rows, field, last):
    values = np.asarray([row.get(field, np.nan) for row in rows], dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan
    return float(np.mean(values[-last:]))


def _float_key(value):
    return round(float(value), 12)


def gap_records(runs, last, full_batch_size):
    """Return paired deterministic-baseline/minibatch equilibrium summaries."""
    relevant = [
        run for run in runs
        if run['config']['kind'] in ('sgdm', 'scaled_sgd') and run['diagnostics']
    ]
    baselines = {}
    for run in relevant:
        cfg = run['config']
        if int(cfg['batch_size']) != full_batch_size:
            continue
        key = (
            cfg['kind'], _float_key(cfg['mu_parameter']),
            _float_key(cfg['reference_eta']), int(cfg['seed']),
        )
        baselines[key] = equilibrium_mean(run['diagnostics'], 'sharpness', last)

    records = []
    for run in relevant:
        cfg = run['config']
        batch = int(cfg['batch_size'])
        if batch >= full_batch_size:
            continue
        key = (
            cfg['kind'], _float_key(cfg['mu_parameter']),
            _float_key(cfg['reference_eta']), int(cfg['seed']),
        )
        baseline = baselines.get(key, np.nan)
        sharpness = equilibrium_mean(run['diagnostics'], 'sharpness', last)
        beta = equilibrium_mean(run['diagnostics'], 'beta', last)
        sigma = equilibrium_mean(run['diagnostics'], 'sigma_u2', last)
        records.append({
            'kind': cfg['kind'],
            'mu': float(cfg['mu_parameter']),
            'eta': float(cfg['reference_eta']),
            'batch_size': batch,
            'seed': int(cfg['seed']),
            'baseline_sharpness': baseline,
            'sharpness': sharpness,
            'gap': baseline - sharpness,
            'beta': beta,
            'sigma_u2': sigma,
        })
    return records


def aggregate_records(records, xfield, field, kind):
    groups = {}
    for record in records:
        if record['kind'] != kind or not np.isfinite(record[field]):
            continue
        groups.setdefault(record[xfield], []).append(record[field])
    xs = sorted(groups)
    means = []
    sds = []
    ns = []
    for x in xs:
        values = np.asarray(groups[x], dtype=float)
        means.append(float(np.mean(values)))
        sds.append(float(np.std(values, ddof=1)) if len(values) > 1 else 0.0)
        ns.append(len(values))
    return np.asarray(xs, dtype=float), np.asarray(means), np.asarray(sds), np.asarray(ns)


def paired_gap_difference(records, xfield):
    index = {
        (record['kind'], record[xfield], record['seed']): record['gap']
        for record in records if np.isfinite(record['gap'])
    }
    groups = {}
    xs = sorted({record[xfield] for record in records})
    seeds = sorted({record['seed'] for record in records})
    for x in xs:
        for seed in seeds:
            left = index.get(('sgdm', x, seed), np.nan)
            right = index.get(('scaled_sgd', x, seed), np.nan)
            if np.isfinite(left) and np.isfinite(right):
                groups.setdefault(x, []).append(left - right)
    out_x, means, sds = [], [], []
    for x in sorted(groups):
        values = np.asarray(groups[x], dtype=float)
        out_x.append(x)
        means.append(float(np.mean(values)))
        sds.append(float(np.std(values, ddof=1)) if len(values) > 1 else 0.0)
    return np.asarray(out_x, dtype=float), np.asarray(means), np.asarray(sds)


def power_law_fit(x, y):
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if valid.sum() < 2:
        return None
    slope, intercept = np.polyfit(np.log10(x[valid]), np.log10(y[valid]), 1)
    return float(slope), float(intercept)


def _optimizer_label(kind):
    return 'SGDM' if kind == 'sgdm' else 'Rescaled SGD'


def plot_gap_vs_batch(records, figures, fixed_mu):
    selected = [record for record in records if np.isclose(record['mu'], fixed_mu)]
    fig, (ax, ax_diff) = plt.subplots(
        2, 1, figsize=(8.3, 7.0), sharex=True,
        gridspec_kw={'height_ratios': [3, 1], 'hspace': .08},
    )
    colors = {'sgdm': '#17639B', 'scaled_sgd': '#D47A19'}
    anchors = []
    for kind in ('sgdm', 'scaled_sgd'):
        x, mean, sd, _ = aggregate_records(selected, 'batch_size', 'gap', kind)
        positive = np.isfinite(mean) & (mean > 0)
        if not positive.any():
            continue
        ax.errorbar(
            x[positive], mean[positive], yerr=sd[positive], marker='o',
            color=colors[kind], capsize=3, linewidth=1.5,
            label=_optimizer_label(kind),
        )
        fit = power_law_fit(x, mean)
        if fit is not None:
            slope, intercept = fit
            fit_x = np.logspace(np.log10(x[positive].min()), np.log10(x[positive].max()), 100)
            fit_y = 10 ** (intercept + slope * np.log10(fit_x))
            ax.plot(fit_x, fit_y, '--', color=colors[kind], alpha=.65,
                    label=rf'{_optimizer_label(kind)} fit: $b^{{{slope:.2f}}}$')
        anchors.append((x[positive][0], mean[positive][0]))
    if anchors:
        anchor_x = min(value[0] for value in anchors)
        anchor_y = float(np.exp(np.mean(np.log([value[1] for value in anchors]))))
        all_batches = np.asarray([record['batch_size'] for record in selected], dtype=float)
        ref_x = np.logspace(np.log10(all_batches.min()), np.log10(all_batches.max()), 100)
        ref_y = anchor_y * (ref_x / anchor_x) ** -1
        ax.plot(ref_x, ref_y, ':', color='black', linewidth=1.3,
                label=r'Theory reference: $\propto b^{-1}$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel('Measured sharpness gap')
    ax.set_title(rf'Deterministic-baseline gap versus batch size: $\mu={fixed_mu:g}$')
    ax.grid(alpha=.2, which='both')
    ax.legend(fontsize=8)

    x, mean, sd = paired_gap_difference(selected, 'batch_size')
    ax_diff.errorbar(x, mean, yerr=sd, marker='o', color='#5B3A8A', capsize=3)
    ax_diff.axhline(0, color='black', linewidth=.8)
    ax_diff.set_xscale('log')
    ax_diff.set_xlabel('Batch size $b$')
    ax_diff.set_ylabel(r'$\Delta S_M-\Delta S_R$')
    ax_diff.grid(alpha=.2, which='both')
    fig.subplots_adjust(left=.13, right=.98, bottom=.10, top=.92, hspace=.08)
    fig.savefig(figures / '05a_gap_vs_batch.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_gap_vs_mu(records, figures, fixed_batch):
    selected = [record for record in records if record['batch_size'] == fixed_batch]
    fig, (ax, ax_diff) = plt.subplots(
        2, 1, figsize=(8.3, 7.0), sharex=True,
        gridspec_kw={'height_ratios': [3, 1], 'hspace': .08},
    )
    colors = {'sgdm': '#17639B', 'scaled_sgd': '#D47A19'}
    for kind in ('sgdm', 'scaled_sgd'):
        x, mean, sd, _ = aggregate_records(selected, 'mu', 'gap', kind)
        ax.errorbar(x, mean, yerr=sd, marker='o', color=colors[kind], capsize=3,
                    linewidth=1.5, label=_optimizer_label(kind))
    ax.set_ylabel('Measured sharpness gap')
    ax.set_title(rf'Deterministic-baseline gap versus momentum: $b={fixed_batch}$')
    ax.grid(alpha=.2)
    ax.legend(fontsize=9)

    x, mean, sd = paired_gap_difference(selected, 'mu')
    ax_diff.errorbar(x, mean, yerr=sd, marker='o', color='#5B3A8A', capsize=3)
    ax_diff.axhline(0, color='black', linewidth=.8)
    ax_diff.set_xlabel(r'Momentum label $\mu$')
    ax_diff.set_ylabel(r'$\Delta S_M-\Delta S_R$')
    ax_diff.grid(alpha=.2)
    fig.subplots_adjust(left=.13, right=.98, bottom=.10, top=.92, hspace=.08)
    fig.savefig(figures / '05b_gap_vs_mu.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_noise_vs_batch(records, figures, fixed_mu):
    selected = [record for record in records if np.isclose(record['mu'], fixed_mu)]
    fig, ax = plt.subplots(figsize=(8.3, 5.4))
    colors = {'sgdm': '#17639B', 'scaled_sgd': '#D47A19'}
    anchors = []
    for kind in ('sgdm', 'scaled_sgd'):
        x, mean, sd, _ = aggregate_records(selected, 'batch_size', 'sigma_u2', kind)
        positive = np.isfinite(mean) & (mean > 0)
        if not positive.any():
            continue
        ax.errorbar(x[positive], mean[positive], yerr=sd[positive], marker='o',
                    color=colors[kind], capsize=3, linewidth=1.5,
                    label=_optimizer_label(kind))
        fit = power_law_fit(x, mean)
        if fit is not None:
            slope, intercept = fit
            fit_x = np.logspace(np.log10(x[positive].min()), np.log10(x[positive].max()), 100)
            fit_y = 10 ** (intercept + slope * np.log10(fit_x))
            ax.plot(fit_x, fit_y, '--', color=colors[kind], alpha=.65,
                    label=rf'{_optimizer_label(kind)} fit: $b^{{{slope:.2f}}}$')
        anchors.append((x[positive][0], mean[positive][0]))
    if anchors:
        anchor_x = min(value[0] for value in anchors)
        anchor_y = float(np.exp(np.mean(np.log([value[1] for value in anchors]))))
        batches = np.asarray([record['batch_size'] for record in selected], dtype=float)
        ref_x = np.logspace(np.log10(batches.min()), np.log10(batches.max()), 100)
        ax.plot(ref_x, anchor_y * (ref_x / anchor_x) ** -1, ':', color='black',
                linewidth=1.3, label=r'Theory reference: $\propto b^{-1}$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Batch size $b$')
    ax.set_ylabel(r'$\sigma_u^2=\mathrm{Var}_B[\langle g_B,u\rangle]$')
    ax.set_title(rf'Projected gradient-noise variance: $\mu={fixed_mu:g}$')
    ax.grid(alpha=.2, which='both')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / '05c_projected_noise_vs_batch.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_gap_vs_beta_sigma(records, figures, fixed_mu):
    selected = [record for record in records if np.isclose(record['mu'], fixed_mu)]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), squeeze=False)
    colors = {'sgdm': '#17639B', 'scaled_sgd': '#D47A19'}
    for ax, kind in zip(axes.ravel(), ('sgdm', 'scaled_sgd')):
        x_gap, gap_mean, gap_sd, _ = aggregate_records(selected, 'batch_size', 'gap', kind)
        x_beta, beta_mean, _, _ = aggregate_records(selected, 'batch_size', 'beta', kind)
        x_sigma, sigma_mean, _, _ = aggregate_records(selected, 'batch_size', 'sigma_u2', kind)
        common = sorted(set(x_gap.tolist()) & set(x_beta.tolist()) & set(x_sigma.tolist()))
        if not common:
            continue
        gap_map = dict(zip(x_gap, gap_mean))
        gap_sd_map = dict(zip(x_gap, gap_sd))
        beta_map = dict(zip(x_beta, beta_mean))
        sigma_map = dict(zip(x_sigma, sigma_mean))
        gaps = np.asarray([gap_map[x] for x in common], dtype=float)
        gap_sds = np.asarray([gap_sd_map[x] for x in common], dtype=float)
        products = np.asarray([beta_map[x] * sigma_map[x] for x in common], dtype=float)
        valid_anchor = np.flatnonzero((gaps > 0) & (products > 0))
        if not len(valid_anchor):
            continue
        anchor = valid_anchor[0]
        scale = gaps[anchor] / products[anchor]
        positions = np.arange(len(common))
        ax.bar(positions, gaps, yerr=gap_sds, capsize=3, alpha=.55,
               color=colors[kind], label='Measured gap')
        ax.plot(positions, scale * products, 's--', color='#222222', linewidth=1.5,
                label=r'Scaled $\bar{\beta}\,\bar{\sigma}_u^2$')
        ax.set_xticks(positions)
        ax.set_xticklabels([f'{int(x)}' for x in common])
        ax.set_xlabel('Batch size $b$')
        ax.set_ylabel('Sharpness gap / scaled mechanism')
        ax.set_title(_optimizer_label(kind))
        ax.grid(alpha=.18, axis='y')
        ax.legend(fontsize=8)
        ax.text(.03, .87, rf'$c={scale:.3g}$ anchored at $b={int(common[anchor])}$',
                transform=ax.transAxes, ha='left', va='top', fontsize=8,
                bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .8})
    fig.suptitle(rf'Measured gap versus Liao-style $\bar{{\beta}}\,\bar{{\sigma}}_u^2$: $\mu={fixed_mu:g}$')
    fig.tight_layout()
    fig.savefig(figures / '05d_gap_vs_beta_sigma_by_batch.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_study_5(runs, figures, last, fixed_eta, fixed_mu, fixed_batch,
                 full_batch_size, selected_plots):
    records = gap_records(runs, last, full_batch_size)
    if fixed_eta is not None:
        records = [record for record in records
                   if np.isclose(record['eta'], fixed_eta)]
    if not records:
        raise ValueError('Study 5 needs minibatch runs and matched full-batch baselines.')
    if 'a' in selected_plots:
        plot_gap_vs_batch(records, figures, fixed_mu)
    if 'b' in selected_plots:
        plot_gap_vs_mu(records, figures, fixed_batch)
    if 'c' in selected_plots:
        plot_noise_vs_batch(records, figures, fixed_mu)
    if 'd' in selected_plots:
        plot_gap_vs_beta_sigma(records, figures, fixed_mu)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='One study result directory')
    parser.add_argument('--study', type=int, choices=(1, 2, 3, 4, 5), required=True)
    parser.add_argument('--last', type=int, default=10,
                        help='Equilibrium window length for study 5')
    parser.add_argument('--study5-mu', type=float, default=.7)
    parser.add_argument('--study5-eta', type=float,
                        help='Retain only this reference eta in Study 5')
    parser.add_argument('--study5-batch', type=int, default=200)
    parser.add_argument('--full-batch-size', type=int, default=5000)
    parser.add_argument('--study5-plots', nargs='+', choices=('a', 'b', 'c', 'd'),
                        default=('a', 'b', 'c', 'd'),
                        help='Select which Study-5 panels to generate')
    parser.add_argument('--plot-eta', type=float,
                        help='Before plotting Studies 1-4, retain only this reference eta')
    parser.add_argument('--plot-mu', type=float,
                        help='Before plotting Studies 1-4, retain only this momentum label')
    parser.add_argument('--plot-batch', type=int,
                        help='Before plotting Studies 1-4, retain only this batch size')
    parser.add_argument('--plot-batches', type=int, nargs='+',
                        help='Before plotting Studies 1-4, retain these batch sizes')
    args = parser.parse_args()

    root = Path(args.input)
    runs = find_runs(root)
    if not runs:
        raise FileNotFoundError(f'No saved run CSVs found under {root}/')
    figures = root / 'figures'
    figures.mkdir(parents=True, exist_ok=True)
    if args.study == 5:
        plot_study_5(
            runs, figures, args.last, args.study5_eta, args.study5_mu,
            args.study5_batch, args.full_batch_size, args.study5_plots,
        )
    else:
        if args.plot_eta is not None:
            runs = [run for run in runs if
                    np.isclose(run['config']['reference_eta'], args.plot_eta)]
        if args.plot_mu is not None:
            runs = [run for run in runs if
                    np.isclose(run['config']['mu_parameter'], args.plot_mu)]
        if args.plot_batch is not None:
            runs = [run for run in runs if
                    int(run['config']['batch_size']) == args.plot_batch]
        if args.plot_batches is not None:
            allowed_batches = set(args.plot_batches)
            runs = [run for run in runs if
                    int(run['config']['batch_size']) in allowed_batches]
        plotter = {
            1: plot_study_1,
            2: plot_study_2,
            3: plot_study_3,
            4: plot_study_4,
        }[args.study]
        plotter(runs, figures)
    print('Plots available in:', figures)


if __name__ == '__main__':
    main()
