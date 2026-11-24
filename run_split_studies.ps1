param(
    [ValidateSet("1", "2", "Both")]
    [string] $Part = "Both",
    [int[]] $Seeds = @(0)
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path ".\src\sgdm_eos.py")) {
    throw "Run this script from the edge-of-stability repository root."
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

function New-CommonArguments {
    param(
        [int] $Steps,
        [int] $DiagEvery
    )
    return @(
        "src/sgdm_eos.py",
        "--device", "auto",
        "--arch", "fc-tanh",
        "--loss", "mse",
        "--steps", "$Steps",
        "--subset-seed", "2026",
        "--eval-every", "100",
        "--diag-every", "$DiagEvery",
        "--physical-batch-size", "250",
        "--eig-tol", "0.001",
        "--eig-maxiter", "120",
        "--seeds"
    ) + ($Seeds | ForEach-Object { "$_" })
}


# ---------------------------------------------------------------------------
# PART 1
# Studies 1-3 and panels 05a-05c.
# Core setting matches the previous 15k Study-5 experiment:
# eta=0.02, b=500, mu in {0.1, 0.3, 0.5}; batch sweep at mu=0.5.
# ---------------------------------------------------------------------------
if ($Part -eq "1" -or $Part -eq "Both") {
    $Output = "part1_15k_studies123_05abc"
    $Common = New-CommonArguments -Steps 15000 -DiagEvery 500

    $Study2Output = "part1_15k_study2_eta001_00125_0015"
    $Study3Output = "part1_15k_study3_mu05_06_08"

    # Batch sweep for 05a and 05c. The b=500, mu=0.5 runs are also reused by
    # Studies 1 and 3 and by the momentum sweep in 05b.
    foreach ($BatchSize in @(100, 200, 300, 500, 1000)) {
        Invoke-PythonChecked ($Common + @(
            "--kinds", "scaled_sgd", "sgdm",
            "--eta", "0.02",
            "--mus", "0.5",
            "--batch-size", "$BatchSize",
            "--noise-batches", "50",
            "--full-diagnostics-all-kinds",
            "--output", $Output
        ))
    }

    # Shared optimizer/momentum sweep for Study 1, Study 3 and 05b.
    # Already-completed b=500, mu=0.5 trajectories are skipped automatically.
    Invoke-PythonChecked ($Common + @(
        "--kinds", "sgd", "scaled_sgd", "sgdm",
        "--eta", "0.02",
        "--mus", "0.1", "0.3", "0.5",
        "--batch-size", "500",
        "--noise-batches", "50",
        "--full-diagnostics-all-kinds",
        "--output", $Output
    ))

    # Study 2 uses eta=0.01, 0.0125 and 0.015.
    # The main Study-5 data are untouched.
    foreach ($Eta in @("0.01", "0.0125", "0.015")) {
        Invoke-PythonChecked ($Common + @(
            "--kinds", "sgdm",
            "--eta", $Eta,
            "--mus", "0.5",
            "--batch-size", "500",
            "--sharpness-only",
            "--output", $Study2Output
        ))
    }

    # Alternative Study 3 sweep:
    # only these momentum values will appear in the Study 3 figures.
    Invoke-PythonChecked ($Common + @(
        "--kinds", "sgdm",
        "--eta", "0.02",
        "--mus", "0.5", "0.6", "0.8",
        "--batch-size", "500",
        "--sharpness-only",
        "--output", $Study3Output
    ))

    # Deterministic references needed for the measured gaps in 05a and 05b.
    Invoke-PythonChecked ($Common + @(
        "--kinds", "scaled_sgd", "sgdm",
        "--eta", "0.02",
        "--mus", "0.1", "0.3", "0.5",
        "--batch-size", "5000",
        "--sharpness-only",
        "--output", $Output
    ))

    # Study 1: optimizer comparison at the common eta and batch size.
    Invoke-PythonChecked @(
        "src/sgdm_eos_plot.py", "--study", "1",
        "--input", $Output,
        "--plot-eta", "0.02",
        "--plot-batch", "500"
    )

    Invoke-PythonChecked @(
        "src/sgdm_eos_plot.py", "--study", "2",
        "--input", $Study2Output
    )

    Invoke-PythonChecked @(
        "src/sgdm_eos_plot.py", "--study", "3",
        "--input", $Study3Output
    )

    # Generate only 05a, 05b and 05c from this part.
    Invoke-PythonChecked @(
        "src/sgdm_eos_plot.py", "--study", "5",
        "--input", $Output,
        "--last", "10",
        "--study5-eta", "0.02",
        "--study5-mu", "0.5",
        "--study5-batch", "500",
        "--full-batch-size", "5000",
        "--study5-plots", "a", "b", "c"
    )

    Write-Host "Part 1 completed: $Output\figures" -ForegroundColor Green
}


# ---------------------------------------------------------------------------
# PART 2
# Study 4 and panel 05d.
# Setting follows the earlier 10k experiments: eta=0.01, mu=0.7 and the
# original smaller batch-size sweep. The same minibatch trajectories feed
# both Study 4 and 05d.
# ---------------------------------------------------------------------------
if ($Part -eq "2" -or $Part -eq "Both") {
    $Output = "part2_17p5k_study4_05d_larger_batches"
    $Common = New-CommonArguments -Steps 17500 -DiagEvery 200
    $StudyBatches = @(100, 200, 250, 300, 500)

    foreach ($BatchSize in $StudyBatches) {
        Invoke-PythonChecked ($Common + @(
            "--kinds", "scaled_sgd", "sgdm",
            "--eta", "0.01",
            "--mus", "0.7",
            "--batch-size", "$BatchSize",
            "--noise-batches", "50",
            "--full-diagnostics-all-kinds",
            "--output", $Output
        ))
    }

    # Matched deterministic references for the two measured gaps in 05d.
    Invoke-PythonChecked ($Common + @(
        "--kinds", "scaled_sgd", "sgdm",
        "--eta", "0.01",
        "--mus", "0.7",
        "--batch-size", "5000",
        "--sharpness-only",
        "--output", $Output
    ))

    # Study 4 uses only the requested minibatch trajectories, not b=5000.
    Invoke-PythonChecked @(
        "src/sgdm_eos_plot.py", "--study", "4",
        "--input", $Output,
        "--plot-eta", "0.01",
        "--plot-mu", "0.7",
        "--plot-batches", "100", "200", "250", "300", "500"
    )

    # Generate only 05d. With diagnostics every 200 updates, last=20 averages
    # approximately the final 4,000 updates.
    Invoke-PythonChecked @(
        "src/sgdm_eos_plot.py", "--study", "5",
        "--input", $Output,
        "--last", "20",
        "--study5-eta", "0.01",
        "--study5-mu", "0.7",
        "--study5-batch", "200",
        "--full-batch-size", "5000",
        "--study5-plots", "d"
    )

    Write-Host "Part 2 completed: $Output\figures" -ForegroundColor Green
}

Write-Host "Requested experiment part(s) completed." -ForegroundColor Green
