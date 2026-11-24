param(
    [int[]] $Seeds = @(0),
    [int] $Steps = 10000
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path ".\src\sgdm_eos.py")) {
    throw "Run this script from the edge-of-stability repository root."
}

if (-not (Test-Path ".\src\sgdm_eos_plot.py")) {
    throw "Could not find src\sgdm_eos_plot.py."
}

if (-not $env:DATASETS) {
    $env:DATASETS = Join-Path (Get-Location) "datasets"
}
New-Item -ItemType Directory -Force -Path $env:DATASETS | Out-Null

function Invoke-PythonChecked {
    param([object[]] $Arguments)

    & python @Arguments

    if ($LASTEXITCODE -ne 0) {
        throw "Python failed with exit code $LASTEXITCODE"
    }
}

$Output = "final_sgdm_vs_rescaled_sharpness_10k"

$Common = @(
    "src/sgdm_eos.py",
    "--device", "auto",
    "--arch", "fc-tanh",
    "--loss", "mse",
    "--steps", "$Steps",
    "--subset-seed", "2026",
    "--eval-every", "100",
    "--diag-every", "200",
    "--physical-batch-size", "250",
    "--eig-tol", "0.001",
    "--eig-maxiter", "120",
    "--seeds"
) + ($Seeds | ForEach-Object { "$_" })


# Run SGDM and rescaled SGD.
# --sharpness-only avoids computing beta and noise diagnostics.
Invoke-PythonChecked ($Common + @(
    "--kinds", "scaled_sgd", "sgdm",
    "--eta", "0.01",
    "--mus", "0.3", "0.5", "0.7",
    "--batch-size", "500",
    "--sharpness-only",
    "--output", $Output
))


# Create a dedicated plotting helper.
$PlotterPath = ".\src\plot_final_sharpness_comparison.py"

$PlotterCode = @'
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sgdm_eos_plot import find_runs, grouped_mean, add_line


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    root = Path(args.input)
    runs = find_runs(root)

    if not runs:
        raise FileNotFoundError(f"No runs found under {root}")

    runs = [
        run for run in runs
        if run["config"]["kind"] in ("sgdm", "scaled_sgd")
    ]

    etas = sorted({
        float(run["config"]["reference_eta"])
        for run in runs
    })
    batches = sorted({
        int(run["config"]["batch_size"])
        for run in runs
    })

    if len(etas) != 1:
        raise ValueError(f"Expected one reference eta, found {etas}")
    if len(batches) != 1:
        raise ValueError(f"Expected one batch size, found {batches}")

    eta = etas[0]
    batch = batches[0]

    sgdm_mus = {
        float(run["config"]["mu_parameter"])
        for run in runs
        if run["config"]["kind"] == "sgdm"
    }
    scaled_mus = {
        float(run["config"]["mu_parameter"])
        for run in runs
        if run["config"]["kind"] == "scaled_sgd"
    }
    mus = sorted(sgdm_mus & scaled_mus)

    if not mus:
        raise ValueError("No paired SGDM/rescaled-SGD runs found.")

    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    colors = plt.cm.viridis(np.linspace(0.12, 0.88, len(mus)))


    # ---------------------------------------------------------------
    # Plot 1: training loss
    # ---------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(9.4, 5.5))

    for mu, color in zip(mus, colors):
        add_line(
            ax,
            grouped_mean(
                runs,
                "train_loss",
                "losses",
                kind="sgdm",
                mu_parameter=mu,
            ),
            rf"SGDM, $\mu={mu:g}$",
            color,
            linestyle="-",
            lw=2.0,
        )

        effective_eta = eta / (1.0 - mu)

        add_line(
            ax,
            grouped_mean(
                runs,
                "train_loss",
                "losses",
                kind="scaled_sgd",
                mu_parameter=mu,
            ),
            rf"Rescaled SGD, $\mu={mu:g}$ "
            rf"$(\eta_{{\rm eff}}={effective_eta:.4g})$",
            color,
            linestyle="--",
            lw=1.8,
        )

    ax.set_xlabel("Minibatch parameter updates")
    ax.set_ylabel("Mean training loss per example")
    ax.set_yscale("log")
    ax.set_title(
        rf"SGDM versus rescaled SGD: $\eta={eta:g}$, $b={batch}$"
    )
    ax.grid(alpha=0.22)
    ax.legend(fontsize=8)
    fig.tight_layout()

    fig.savefig(
        figures / "final_train_loss_sgdm_vs_rescaled.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


    # ---------------------------------------------------------------
    # Plot 2: sharpness, using two panels inside one figure
    # ---------------------------------------------------------------

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

    configurations = (
        ("sgdm", "SGDM"),
        ("scaled_sgd", "Rescaled SGD"),
    )

    for ax, (kind, title) in zip(axes, configurations):
        for mu, color in zip(mus, colors):
            add_line(
                ax,
                grouped_mean(
                    runs,
                    "sharpness",
                    "diagnostics",
                    kind=kind,
                    mu_parameter=mu,
                ),
                rf"Sharpness, $\mu={mu:g}$",
                color,
                linestyle="-",
                lw=2.0,
            )

            if kind == "sgdm":
                threshold = 2.0 * (1.0 + mu) / eta
            else:
                # Rescaled SGD uses eta_eff = eta/(1-mu).
                threshold = 2.0 * (1.0 - mu) / eta

            ax.axhline(
                threshold,
                color=color,
                linestyle=":",
                linewidth=1.4,
                label=rf"Threshold, $\mu={mu:g}$: {threshold:.1f}",
            )

        ax.set_xlabel("Minibatch parameter updates")
        ax.set_ylabel(
            r"Full-objective sharpness "
            r"$\lambda_{\max}(\nabla^2 L)$"
        )
        ax.set_title(title)
        ax.grid(alpha=0.22)
        ax.legend(fontsize=7)

    fig.suptitle(
        rf"Sharpness comparison: $\eta={eta:g}$, $b={batch}$"
    )
    fig.tight_layout()

    fig.savefig(
        figures / "final_sharpness_sgdm_vs_rescaled.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    print("Plots available in:", figures)


if __name__ == "__main__":
    main()
'@

Set-Content -Path $PlotterPath -Value $PlotterCode -Encoding UTF8


# Generate the two requested plots.
Invoke-PythonChecked @(
    $PlotterPath,
    "--input", $Output
)

Write-Host ""
Write-Host "Final comparison completed." -ForegroundColor Green
Write-Host "Plots are available in: $Output\figures"