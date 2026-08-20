<#
  One-shot setup for vocalsplit on Windows.

    powershell -ExecutionPolicy Bypass -File .\setup_env.ps1

  Finds conda even when it is not on PATH, picks the CUDA or CPU build of torch
  by looking for an NVIDIA GPU, installs everything, registers the Jupyter
  kernel, and verifies the result actually imports.

  Options:
    -EnvName  <name>   environment name           (default: vocalsplit)
    -NoConda           use a plain venv instead of conda
    -Cpu               force the CPU build of torch even if a GPU is present

  Not an advanced script on purpose -- see the note in run.ps1.
#>

param(
    [string]$EnvName = "vocalsplit",
    [switch]$NoConda,
    [switch]$Cpu
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$CUDA_INDEX = "https://download.pytorch.org/whl/cu124"

function Say($msg, $colour = "Cyan") { Write-Host "`n== $msg" -ForegroundColor $colour }

# --- 1. GPU? ----------------------------------------------------------------
$useCuda = $false
if (-not $Cpu) {
    $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($smi) {
        # Capture the whole output first. Piping a native command straight into
        # `Select-Object -First 1` stops the process early and leaves
        # $LASTEXITCODE = -1 even on success, which silently downgrades a GPU
        # machine to the CPU build.
        $out = & nvidia-smi --query-gpu=name --format=csv,noheader
        $ok = ($LASTEXITCODE -eq 0)
        $name = @($out) | Where-Object { $_ -and $_.Trim() } | Select-Object -First 1
        if ($ok -and $name) {
            $useCuda = $true
            Say "NVIDIA GPU found: $($name.Trim()) -- installing the CUDA build of torch"
        }
    }
}
if (-not $useCuda) {
    Say "No NVIDIA GPU (or -Cpu given) -- installing the CPU build. Separation will be ~10x slower." "Yellow"
}

# --- 2. build the environment -----------------------------------------------
if ($NoConda) {
    Say "Creating a venv at .\.venv"
    $base = Get-Command python -ErrorAction SilentlyContinue
    if (-not $base) { Write-Host "python not on PATH." -ForegroundColor Red; exit 1 }
    & python -m venv "$root\.venv"
    $py = "$root\.venv\Scripts\python.exe"
}
else {
    # conda is frequently installed but not on PATH, which is why this looks around.
    $conda = (Get-Command conda -ErrorAction SilentlyContinue).Source
    if (-not $conda) {
        $conda = Get-ChildItem "$env:USERPROFILE\*conda*\Scripts\conda.exe",
                               "$env:LOCALAPPDATA\*conda*\Scripts\conda.exe",
                               "C:\ProgramData\*conda*\Scripts\conda.exe" `
                               -ErrorAction SilentlyContinue |
                 Select-Object -First 1 -ExpandProperty FullName
    }
    if (-not $conda) {
        Write-Host "No conda found. Re-run with -NoConda to use a plain venv instead." -ForegroundColor Red
        exit 1
    }

    Say "Creating conda env '$EnvName' (python 3.11) using $conda"
    & $conda create -y -n $EnvName python=3.11
    if ($LASTEXITCODE -ne 0) { Write-Host "conda create failed." -ForegroundColor Red; exit 1 }

    # Ask conda where it put the env rather than guessing the prefix.
    $envsDir = Split-Path (Split-Path $conda -Parent) -Parent
    $py = Join-Path $envsDir "envs\$EnvName\python.exe"
    if (-not (Test-Path $py)) {
        Write-Host "Created the env but cannot find $py" -ForegroundColor Red
        exit 1
    }
}

# --- 3. install --------------------------------------------------------------
Say "Upgrading pip"
& $py -m pip install --upgrade pip --quiet

Say "Installing torch (this is ~2.5 GB on CUDA and takes a while)"
if ($useCuda) { & $py -m pip install torch torchaudio --index-url $CUDA_INDEX }
else          { & $py -m pip install torch torchaudio }
if ($LASTEXITCODE -ne 0) { Write-Host "torch install failed." -ForegroundColor Red; exit 1 }

Say "Installing vocalsplit and its extras"
& $py -m pip install -e "$root[dev,web,notebook]"
if ($LASTEXITCODE -ne 0) { Write-Host "vocalsplit install failed." -ForegroundColor Red; exit 1 }

Say "Registering the Jupyter kernel"
& $py -m ipykernel install --user --name $EnvName --display-name "Python ($EnvName)"

# --- 4. verify ---------------------------------------------------------------
Say "Verifying"
# A real file, not `python -c`: PowerShell mangles multi-line strings passed to a
# native command, and the resulting SyntaxError looks like a bug in the check.
& $py "$root/scripts/verify_install.py"
if ($LASTEXITCODE -ne 0) {
    Write-Host "`nInstalled, but the verification failed above." -ForegroundColor Red
    exit 1
}

Say "Done" "Green"
Write-Host ""
Write-Host "  Use it without touching conda:"
Write-Host "    .\run.ps1 ui                              browser UI"
Write-Host "    .\run.ps1 split ""C:\Music\song.mp3"" -o ""C:\Music\split"""
Write-Host "    .\run.ps1 test"
Write-Host ""
if ($EnvName -ne "vocalsplit") {
    Write-Host "  Non-default env name, so point run.ps1 at it:" -ForegroundColor Yellow
    Write-Host "    `$env:VOCALSPLIT_PYTHON = '$py'"
    Write-Host ""
}
