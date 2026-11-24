param(
    [int[]] $Seeds = @(0)
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

$Common = @(
    "src/sgdm_eos.py",
    "--device", "auto",
    "--arch", "fc-tanh",
    "--loss", "mse",
    "--steps", "15000",
    "--subset-seed", "2026",
    "--eval-every", "100",
    "--diag-every", "500",
    "--physical-batch-size", "250",
    "--eig-tol", "0.001",
    "--eig-maxiter", "120",
    "--seeds"
) + ($Seeds | ForEach-Object { "$_" })


# ---------------------------------------------------------------------------
# Make an independent test-accuracy plotting file.
# The original sgdm_eos_plot.py remains unchanged.
# ---------------------------------------------------------------------------

$ExtraPlotter = ".\src\sgdm_eos_plot_testacc.py"
$PlotText = Get-Content ".\src\sgdm_eos_plot.py" -Raw

$PlotText = $PlotText.Replace("'train_acc'", "'test_acc'")
$PlotText = $PlotText.Replace("Training accuracy", "Test accuracy")
$PlotText = $PlotText.Replace("training accuracy", "test accuracy")
$PlotText = $PlotText.Replace("train_accuracy", "test_accuracy")

Set-Content -Path $ExtraPlotter -Value $PlotText -Encoding UTF8


# ---------------------------------------------------------------------------
# Shared core output:
#
# Study 1:
#   eta=0.02, b=500, mu={0.1, 0.3, 0.5}
#
# Study 4:
#   eta=0.02, mu=0.5, b={100, 200, 300, 500, 1000}
#
# Study 4 is run first so its b=500 SGDM trajectory contains sharpness.
# Study 1 then reuses that trajectory.
# ---------------------------------------------------------------------------

$CoreOutput = "part1_additional_testacc_study4"

foreach ($BatchSize in @(100, 200, 300, 500, 1000)) {
    Invoke-PythonChecked ($Common + @(
        "--kinds", "sgdm",
        "--eta", "0.02",
        "--mus", "0.5",
        "--batch-size", "$BatchSize",
        "--sharpness-only",
        "--output", $CoreOutput
    ))
}

Invoke-PythonChecked ($Common + @(
    "--kinds", "sgd", "scaled_sgd", "sgdm",
    "--eta", "0.02",
    "--mus", "0.1", "0.3", "0.5",
    "--batch-size", "500",
    "--no-diagnostics",
    "--output", $CoreOutput
))


# ---------------------------------------------------------------------------
# Study 2:
# eta={0.01, 0.0125, 0.015}, mu=0.5, but b=200 instead of b=500.
# ---------------------------------------------------------------------------

$Study2Output = "part1_additional_study2_b200"

foreach ($Eta in @("0.01", "0.0125", "0.015")) {
    Invoke-PythonChecked ($Common + @(
        "--kinds", "sgdm",
        "--eta", $Eta,
        "--mus", "0.5",
        "--batch-size", "200",
        "--sharpness-only",
        "--output", $Study2Output
    ))
}


# ---------------------------------------------------------------------------
# Study 3:
# eta=0.02, b=500, mu={0.7, 0.8, 0.9}.
# Only SGDM is run, so there is no rescaled-SGD eta/(1-mu) instability here.
# ---------------------------------------------------------------------------

$Study3Output = "part1_additional_study3_mu07_08_09"

Invoke-PythonChecked ($Common + @(
    "--kinds", "sgdm",
    "--eta", "0.02",
    "--mus", "0.7", "0.8", "0.9",
    "--batch-size", "500",
    "--sharpness-only",
    "--output", $Study3Output
))


# ---------------------------------------------------------------------------
# Plot test accuracy and the requested sharpness figures.
# ---------------------------------------------------------------------------

# Study 1 test accuracy.
Invoke-PythonChecked @(
    $ExtraPlotter,
    "--study", "1",
    "--input", $CoreOutput,
    "--plot-eta", "0.02",
    "--plot-batch", "500"
)

# Study 2 test accuracy and sharpness versus eta.
Invoke-PythonChecked @(
    $ExtraPlotter,
    "--study", "2",
    "--input", $Study2Output
)

# Study 3 test accuracy and sharpness versus mu.
Invoke-PythonChecked @(
    $ExtraPlotter,
    "--study", "3",
    "--input", $Study3Output
)

# Study 4 sharpness versus batch size.
Invoke-PythonChecked @(
    $ExtraPlotter,
    "--study", "4",
    "--input", $CoreOutput,
    "--plot-eta", "0.02",
    "--plot-mu", "0.5",
    "--plot-batches", "100", "200", "300", "500", "1000"
)

Write-Host ""
Write-Host "Additional Part 1 experiment completed." -ForegroundColor Green
Write-Host "Study 1 and Study 4: $CoreOutput\figures"
Write-Host "Study 2: $Study2Output\figures"
Write-Host "Study 3: $Study3Output\figures"