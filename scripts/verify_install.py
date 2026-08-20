"""Check that an install is actually usable, and say which device it will use.

    python scripts/verify_install.py

Called at the end of setup_env.ps1 / setup_env.sh. It lives in a file rather than
being passed to `python -c` from the shell: PowerShell mangles multi-line strings
handed to a native command, which turns a broken import into a confusing
SyntaxError from inside the check itself.

Exit code 0 = ready to use, 1 = something is missing.
"""
from __future__ import annotations

import importlib
import sys

# (import name, human name) -- the ones whose absence actually breaks something
REQUIRED = [
    ("torch", "torch"),
    ("torchaudio", "torchaudio"),
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("librosa", "librosa"),
    ("soundfile", "soundfile"),
    ("soxr", "soxr"),
    ("demucs", "demucs"),
    ("vocalsplit", "vocalsplit"),
]
OPTIONAL = [
    ("lameenc", "mp3 output"),
    ("gradio", "browser UI (app.py)"),
    ("matplotlib", "notebook plot"),
    ("ipykernel", "jupyter kernel"),
]


def _version(mod) -> str:
    return str(getattr(mod, "__version__", "?"))


def main() -> int:
    missing: list[str] = []
    mods: dict[str, object] = {}

    for name, label in REQUIRED:
        try:
            mods[name] = importlib.import_module(name)
        except Exception as e:
            missing.append(f"{label}: {type(e).__name__}: {e}")

    if missing:
        print("MISSING or broken:", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)
        return 1

    torch = mods["torch"]
    if torch.cuda.is_available():
        device = f"cuda -- {torch.cuda.get_device_name(0)}"
        speed = "a 4-minute song takes roughly 30 seconds"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        # Demucs runs on MPS but has historically been patchy there; the CPU
        # fallback is automatic if a kernel is unsupported.
        device = "mps -- Apple Silicon"
        speed = "a 4-minute song takes roughly 1-2 minutes"
    else:
        device = "cpu"
        speed = "a 4-minute song takes roughly 4 minutes"

    print(f"  python      {sys.version.split()[0]}")
    for name, _ in REQUIRED:
        print(f"  {name:<11} {_version(mods[name])}")

    for name, why in OPTIONAL:
        try:
            m = importlib.import_module(name)
            print(f"  {name:<11} {_version(m)}")
        except Exception:
            print(f"  {name:<11} not installed -- no {why}")

    print(f"\n  device      {device}")
    print(f"  expect      {speed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
