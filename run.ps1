<#
  Launcher for vocalsplit that does not need `conda activate`.

  This machine has conda installed but not initialised for PowerShell, so
  `conda activate vocalsplit` fails in a fresh shell. Rather than require
  `conda init powershell` (which edits your profile), this calls the
  environment's python.exe directly.

    .\run.ps1 ui                       browser UI at http://127.0.0.1:7860
    .\run.ps1 lab                      JupyterLab, pick the "Python (vocalsplit)" kernel
    .\run.ps1 test                     run the test suite
    .\run.ps1 detect "C:\Music\a.mp3"  classify only, write nothing
    .\run.ps1 split  "C:\Music\a.mp3"  separate into .\output as mp3

  Deliberately NOT an advanced script: no [CmdletBinding()] and no [Parameter()]
  attributes. Either one would add PowerShell's common parameters, and then a
  passthrough flag like -o collides with -OutVariable and the call fails before
  it ever reaches python.
#>

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot

if ($args.Count -ge 1) { $Command = [string]$args[0] } else { $Command = "help" }
if ($args.Count -gt 1) { $Rest = @($args[1..($args.Count - 1)]) } else { $Rest = @() }

# --- locate the environment -------------------------------------------------
$candidates = @(
    $env:VOCALSPLIT_PYTHON,
    "$env:USERPROFILE\anaconda3__\envs\vocalsplit\python.exe",
    "$env:USERPROFILE\anaconda3\envs\vocalsplit\python.exe",
    "$env:USERPROFILE\miniconda3\envs\vocalsplit\python.exe"
)
$py = $candidates | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1

if (-not $py) {
    Write-Host "Could not find the vocalsplit environment." -ForegroundColor Red
    Write-Host "Create it with:  powershell -ExecutionPolicy Bypass -File .\setup_env.ps1"
    Write-Host "Or point at it:  `$env:VOCALSPLIT_PYTHON = 'C:\path\to\env\python.exe'"
    exit 1
}

switch ($Command.ToLower()) {

    "ui" {
        Write-Host "Starting the web UI -- Ctrl+C to stop." -ForegroundColor Cyan
        & $py "$root\app.py" @Rest
    }

    "lab" {
        # JupyterLab lives in the base conda install, not this env. That is fine:
        # the "Python (vocalsplit)" kernel is registered globally, so pick it from
        # the kernel menu once the notebook opens.
        $lab = "$env:USERPROFILE\anaconda3__\Scripts\jupyter-lab.exe"
        if (-not (Test-Path $lab)) {
            Write-Host "JupyterLab not found at $lab" -ForegroundColor Red
            Write-Host "Open your usual Jupyter and select the 'Python (vocalsplit)' kernel."
            exit 1
        }
        Write-Host "Kernel menu -> Python (vocalsplit). Any other kernel fails on import." -ForegroundColor Yellow
        & $lab --notebook-dir="$root" "notebooks/separate_song.ipynb"
    }

    "test" { & $py -m pytest "$root\tests" -v @Rest }

    "detect" {
        if ($Rest.Count -eq 0) { Write-Host "Give me a file or folder." -ForegroundColor Red; exit 1 }
        & $py -m vocalsplit detect @Rest
    }

    "split" {
        if ($Rest.Count -eq 0) { Write-Host "Give me a file or folder." -ForegroundColor Red; exit 1 }
        # Defaults to mp3. Anything else in $Rest (-o, --model, --stems) passes
        # straight through, and a later --format on the command line wins.
        & $py -m vocalsplit separate --format mp3 --bitrate 320 @Rest
    }

    "shell" {
        Write-Host "Environment python: $py" -ForegroundColor Cyan
        & $py @Rest
    }

    default {
        Write-Host ""
        Write-Host "  vocalsplit" -ForegroundColor Cyan
        Write-Host "  environment: $py"
        Write-Host ""
        Write-Host "  .\run.ps1 ui                          browser UI at http://127.0.0.1:7860"
        Write-Host "  .\run.ps1 lab                         JupyterLab on the notebook"
        Write-Host "  .\run.ps1 test                        run the test suite"
        Write-Host "  .\run.ps1 detect <file|folder>        classify only, write nothing"
        Write-Host "  .\run.ps1 split  <file|folder> [args] separate to mp3"
        Write-Host ""
        Write-Host "  extra args pass through, e.g."
        Write-Host "  .\run.ps1 split ""C:\Music"" -o ""C:\Music\split"" --model htdemucs_ft"
        Write-Host ""
        exit 0
    }
}

exit $LASTEXITCODE
