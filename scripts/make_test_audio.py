"""Generate two synthetic clips to smoke-test the detector without a music library.

  instrumental.wav  kick + bass + arp + hats, no voice at all
  vocal.wav         the same bed with a formant-synthesised sung melody on top

This is a smoke test, not a benchmark. The singing is a source-filter model
(glottal pulse train + vibrato through vowel formants), which is voice-shaped
enough for the pipeline to route and score, but real music is the real test.

  python scripts/make_test_audio.py testdata
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import lfilter

SR = 44100
DUR = 30.0
BPM = 124.0


def _t(n: int) -> np.ndarray:
    return np.arange(n) / SR


def _env(n: int, attack: float, decay: float) -> np.ndarray:
    t = _t(n)
    return np.minimum(t / max(attack, 1e-4), 1.0) * np.exp(-t / decay)


def _place(dst: np.ndarray, src: np.ndarray, at: int) -> None:
    end = min(at + src.size, dst.size)
    if end > at:
        dst[at:end] += src[: end - at]


def kick(n: int) -> np.ndarray:
    t = _t(n)
    f = 120.0 * np.exp(-t / 0.03) + 45.0
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * _env(n, 0.001, 0.12)


def hat(n: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    x = rng.standard_normal(n)
    return lfilter([1, -0.92], [1], x) * _env(n, 0.0005, 0.02) * 0.35


def bass_note(f0: float, n: int) -> np.ndarray:
    t = _t(n)
    saw = 2 * (t * f0 - np.floor(0.5 + t * f0))
    # one-pole lowpass with a falling cutoff -- a plain synth bass
    y, a = np.empty_like(saw), 0.06
    prev = 0.0
    for i, s in enumerate(saw):
        prev += a * (s - prev)
        y[i] = prev
    return y * _env(n, 0.005, 0.18)


def arp_note(f0: float, n: int) -> np.ndarray:
    t = _t(n)
    x = sum(np.sin(2 * np.pi * f0 * h * t) / h for h in (1, 2, 3, 5))
    return x * _env(n, 0.003, 0.10) * 0.25


def formant(x: np.ndarray, freq: float, bw: float, gain: float) -> np.ndarray:
    r = np.exp(-np.pi * bw / SR)
    theta = 2 * np.pi * freq / SR
    return gain * lfilter([1 - r], [1, -2 * r * np.cos(theta), r * r], x)


def sung_note(f0: float, n: int, vowel: tuple) -> np.ndarray:
    """Glottal pulse train with vibrato + jitter, filtered by vowel formants."""
    t = _t(n)
    rng = np.random.default_rng(int(f0))
    vib = 1.0 + 0.022 * np.sin(2 * np.pi * 5.4 * t) * np.minimum(t / 0.25, 1.0)
    jitter = 1.0 + 0.004 * np.cumsum(rng.standard_normal(n)) / np.sqrt(n)
    phase = np.cumsum(f0 * vib * jitter) / SR

    # bandlimited-ish pulse: harmonics with a -12 dB/oct glottal rolloff
    src = sum(np.sin(2 * np.pi * h * phase) / (h**1.6) for h in range(1, 45))
    src += 0.012 * rng.standard_normal(n)  # breath

    out = sum(formant(src, f, bw, g) for f, bw, g in vowel)
    return out * _env(n, 0.06, 1.6)


VOWEL_A = ((700, 90, 1.0), (1220, 110, 0.55), (2600, 160, 0.22), (3400, 220, 0.10))
VOWEL_I = ((300, 70, 1.0), (2300, 130, 0.60), (3000, 170, 0.25), (3600, 240, 0.10))


def build_bed(n: int) -> np.ndarray:
    beat = 60.0 / BPM
    x = np.zeros(n)
    scale = [110.00, 130.81, 146.83, 164.81]  # A2 C3 D3 E3
    for i in range(int(DUR / beat)):
        at = int(i * beat * SR)
        _place(x, kick(int(0.30 * SR)) * 0.9, at)
        _place(x, hat(int(0.06 * SR)), at + int(0.5 * beat * SR))
        _place(x, bass_note(scale[(i // 4) % 4], int(0.45 * beat * SR)) * 0.5, at)
        for j in range(4):
            f = scale[(i + j) % 4] * 4
            _place(x, arp_note(f, int(0.30 * beat * SR)), at + int(j * 0.25 * beat * SR))
    return x


def build_melody(n: int) -> np.ndarray:
    """Four sung phrases with rests between them, like an actual vocal line."""
    beat = 60.0 / BPM
    x = np.zeros(n)
    midi = [69, 71, 72, 74, 72, 71, 69, 67]  # A4-ish
    for phrase in range(4):
        start = phrase * 7.0 * beat * 4 / 4 + 2.0
        if start > DUR - 3:
            break
        for k, m in enumerate(midi):
            at = int((start + k * beat * 0.75) * SR)
            if at >= n:
                break
            f0 = 440.0 * 2 ** ((m - 69) / 12)
            vowel = VOWEL_A if k % 2 == 0 else VOWEL_I
            _place(x, sung_note(f0, int(0.70 * beat * SR), vowel), at)
    return x


def stereo(x: np.ndarray, width: float = 0.12) -> np.ndarray:
    d = int(width * 0.001 * SR) + 1
    return np.stack([x, np.concatenate([np.zeros(d), x[:-d]])])


def norm(x: np.ndarray, peak_db: float = -1.0) -> np.ndarray:
    p = float(np.abs(x).max()) or 1.0
    return x * (10 ** (peak_db / 20) / p)


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "testdata")
    out.mkdir(parents=True, exist_ok=True)
    n = int(DUR * SR)

    bed = build_bed(n)
    voice = build_melody(n)

    sf.write(out / "instrumental.wav", norm(stereo(bed)).T, SR, subtype="PCM_24")
    sf.write(out / "vocal.wav", norm(stereo(bed * 0.75 + voice * 1.5)).T, SR, subtype="PCM_24")
    print(f"wrote {out / 'instrumental.wav'}")
    print(f"wrote {out / 'vocal.wav'}")


if __name__ == "__main__":
    main()
