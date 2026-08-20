"""Audio I/O: decode to float32 [channels, samples] at a target sample rate."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

AUDIO_EXTS = {".wav", ".flac", ".ogg", ".opus", ".mp3", ".m4a", ".aac", ".wma", ".aiff", ".aif"}


def _resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Resample [channels, samples]."""
    if sr_in == sr_out:
        return x
    try:
        import soxr

        return soxr.resample(x.T, sr_in, sr_out, quality="HQ").T.astype(np.float32)
    except ImportError:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(sr_in, sr_out)
        return resample_poly(x, sr_out // g, sr_in // g, axis=1).astype(np.float32)


def _decode_via_ffmpeg(path: Path, sr: int) -> np.ndarray:
    """Fallback decoder for formats libsndfile cannot open (m4a/aac/wma...)."""
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise RuntimeError(
            f"Cannot decode {path.name}: libsndfile failed and ffmpeg is not on PATH. "
            "Install ffmpeg, or convert the file to wav/flac/mp3 first."
        )
    cmd = [exe, "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "2", "-ar", str(sr), "-"]
    raw = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).T.copy()


def load_audio(path: str | Path, sr: int = 44100) -> tuple[np.ndarray, int]:
    """Return (audio [2, n] float32, sr). Mono input is duplicated to stereo."""
    path = Path(path)
    try:
        data, sr_in = sf.read(str(path), dtype="float32", always_2d=True)
        x = data.T
    except Exception:
        x = _decode_via_ffmpeg(path, sr)
        sr_in = sr

    if x.shape[0] == 1:
        x = np.repeat(x, 2, axis=0)
    elif x.shape[0] > 2:
        x = x[:2]

    x = _resample(np.ascontiguousarray(x), sr_in, sr)
    return np.ascontiguousarray(x, dtype=np.float32), sr


def _encode_mp3(path: Path, x: np.ndarray, sr: int, bitrate: int) -> None:
    """MP3 via lameenc (a demucs dependency, so it is already installed)."""
    try:
        import lameenc
    except ImportError:
        exe = shutil.which("ffmpeg")
        if exe is None:
            raise RuntimeError("mp3 output needs either lameenc or ffmpeg installed")
        raw = (np.clip(x.T, -1.0, 1.0) * 32767).astype("<i2").tobytes()
        subprocess.run(
            [exe, "-v", "error", "-y", "-f", "s16le", "-ar", str(sr), "-ac",
             str(x.shape[0]), "-i", "-", "-b:a", f"{bitrate}k", str(path)],
            input=raw, check=True,
        )
        return

    enc = lameenc.Encoder()
    enc.set_bit_rate(bitrate)
    enc.set_in_sample_rate(sr)
    enc.set_channels(x.shape[0])
    enc.set_quality(2)  # 0 = best/slowest, 9 = worst/fastest
    pcm = (np.clip(x.T, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    path.write_bytes(enc.encode(pcm) + enc.flush())


def save_audio(
    path: str | Path,
    x: np.ndarray,
    sr: int,
    subtype: str = "PCM_24",
    bitrate: int = 320,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".mp3":
        _encode_mp3(path, x, sr, bitrate)
        return
    if path.suffix.lower() == ".flac" and subtype == "PCM_32":
        subtype = "PCM_24"  # flac maxes out at 24-bit
    sf.write(str(path), x.T, sr, subtype=subtype)


def find_audio_files(target: str | Path) -> list[Path]:
    """Expand a file or directory into a sorted list of audio files."""
    p = Path(target)
    if p.is_file():
        return [p]
    if p.is_dir():
        return sorted(f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in AUDIO_EXTS)
    return []
