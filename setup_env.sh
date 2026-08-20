#!/usr/bin/env bash
# One-shot setup for vocalsplit on macOS and Linux.
#
#   bash setup_env.sh              # conda if present, otherwise a venv
#   bash setup_env.sh --venv       # force a plain venv at ./.venv
#   bash setup_env.sh --cpu        # skip the CUDA build even if a GPU is present
#
# On Linux the PyPI torch wheels already bundle CUDA, so there is no index-url
# dance -- unlike Windows, where the default wheels are CPU-only.
# On macOS torch uses Metal (MPS) automatically where available.

set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
env_name="vocalsplit"
use_venv=0
force_cpu=0

for arg in "$@"; do
  case "$arg" in
    --venv) use_venv=1 ;;
    --cpu)  force_cpu=1 ;;
    --name=*) env_name="${arg#*=}" ;;
    *) echo "unknown option: $arg" >&2; exit 1 ;;
  esac
done

say() { printf '\n== %s\n' "$1"; }

# --- 1. environment ---------------------------------------------------------
if [[ $use_venv -eq 0 ]] && command -v conda >/dev/null 2>&1; then
  say "Creating conda env '$env_name' (python 3.11)"
  conda create -y -n "$env_name" python=3.11
  py="$(conda run -n "$env_name" python -c 'import sys; print(sys.executable)' | tail -1)"
else
  say "Creating a venv at $root/.venv"
  python3 -m venv "$root/.venv"
  py="$root/.venv/bin/python"
fi

[[ -x "$py" ]] || { echo "could not locate the new python: $py" >&2; exit 1; }

# --- 2. install -------------------------------------------------------------
say "Upgrading pip"
"$py" -m pip install --upgrade pip --quiet

say "Installing torch (large download, be patient)"
if [[ $force_cpu -eq 1 && "$(uname -s)" == "Linux" ]]; then
  "$py" -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
else
  "$py" -m pip install torch torchaudio
fi

say "Installing vocalsplit and its extras"
"$py" -m pip install -e "$root[dev,web,notebook]"

say "Registering the Jupyter kernel"
"$py" -m ipykernel install --user --name "$env_name" --display-name "Python ($env_name)"

# --- 3. verify --------------------------------------------------------------
say "Verifying"
"$py" "$root/scripts/verify_install.py"

say "Done"
cat <<EOF

  Activate, then use it:

    $( [[ $use_venv -eq 1 ]] && echo "source .venv/bin/activate" || echo "conda activate $env_name" )

    vocalsplit detect  "song.mp3"
    vocalsplit separate "song.mp3" -o output --format mp3
    python app.py                       # browser UI on http://127.0.0.1:7860
    pytest tests/

  Or skip activation entirely:

    $py -m vocalsplit separate "song.mp3" -o output --format mp3

EOF
