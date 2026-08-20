# Creates a conda env with CUDA-enabled PyTorch and installs vocalsplit.
# Usage:  powershell -ExecutionPolicy Bypass -File .\setup_env.ps1
$ErrorActionPreference = "Stop"
$env_name = "vocalsplit"

conda create -y -n $env_name python=3.11
$py = (conda run -n $env_name python -c "import sys; print(sys.executable)")

# CUDA build of torch (Windows wheels on PyPI are CPU-only).
# Drop the --index-url line if you have no NVIDIA GPU.
& $py -m pip install --upgrade pip
& $py -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
& $py -m pip install -e .

Write-Host ""
Write-Host "Done. Activate with:  conda activate $env_name"
Write-Host "Then:                 vocalsplit detect `"C:\path\to\song.mp3`""
