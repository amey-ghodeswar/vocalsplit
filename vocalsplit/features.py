"""Measure how much real singing is in a separated vocal stem.

The premise: every separator will hand you a "vocals" stem for an instrumental
track too -- it just contains bleed (synth lead, reverb tail, hi-hat wash) at a
low level, scattered in time, and not pitched like a voice. So instead of asking
a classifier about the mixture, we separate first and then interrogate the stem
along four axes that bleed and singing differ on:

    level       how loud the stem is *relative to the mixture, per frame*
    activity    what fraction of the music is vocal-active
    continuity  whether activity forms multi-second phrases or scattered dust
    voicedness  whether the active parts are periodic like a voice
"""
from __future__ import annotations

import numpy as np

EPS = 1e-10
HOP_S = 0.05  # 50 ms frames
WIN_S = 0.10


def _mono(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=0) if x.ndim == 2 else x


def _db(x: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(x, EPS))


def _frame_rms(y: np.ndarray, frame: int, hop: int) -> np.ndarray:
    import librosa

    return librosa.feature.rms(y=y, frame_length=frame, hop_length=hop, center=True)[0]


def _longest_run(mask: np.ndarray, hop_s: float, bridge_frames: int = 6) -> float:
    """Longest contiguous True run in seconds, bridging gaps up to `bridge_frames`."""
    if not mask.any():
        return 0.0
    m = mask.copy()
    gap = 0
    for i in range(1, len(m)):
        if not m[i]:
            gap += 1
        else:
            if 0 < gap <= bridge_frames:
                m[i - gap : i] = True
            gap = 0
    best = run = 0
    for v in m:
        run = run + 1 if v else 0
        best = max(best, run)
    return best * hop_s


def _active_regions(active: np.ndarray, min_frames: int = 6) -> list[tuple[int, int]]:
    """Contiguous [start, end) frame runs of `active`, dropping very short ones."""
    runs, start = [], None
    for i, v in enumerate(active):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_frames:
                runs.append((start, i))
            start = None
    if start is not None and len(active) - start >= min_frames:
        runs.append((start, len(active)))
    return runs


def _voicedness(voc_mono: np.ndarray, sr: int, active: np.ndarray, hop: int,
                budget_s: float = 15.0) -> tuple[float, float]:
    """Run pYIN on the loudest contiguous vocal-active regions.

    Whole-track pYIN is slow and pointless: if the *most* voice-like regions
    aren't periodic, nothing else will be. Regions rather than individual
    frames, because splicing 50 ms fragments would fabricate discontinuities
    that pYIN would read as unvoiced.
    """
    import librosa

    regions = _active_regions(active)
    if not regions:
        return 0.0, 0.0

    scored = sorted(
        regions,
        key=lambda r: -float(np.sqrt(np.mean(voc_mono[r[0] * hop : r[1] * hop] ** 2) + EPS)),
    )

    chunks, total, budget = [], 0, int(budget_s * sr)
    for s0, s1 in scored:
        seg = voc_mono[s0 * hop : min(s1 * hop, voc_mono.size)]
        if seg.size == 0:
            continue
        chunks.append(seg)
        total += seg.size
        if total >= budget:
            break
    if not chunks:
        return 0.0, 0.0

    y = np.concatenate(chunks)
    if y.size < 4096 or float(np.abs(y).max()) < 1e-6:
        return 0.0, 0.0

    y16 = librosa.resample(y, orig_sr=sr, target_sr=16000)
    try:
        _, voiced_flag, voiced_prob = librosa.pyin(
            y16, fmin=70.0, fmax=1100.0, sr=16000, frame_length=2048, hop_length=256
        )
    except Exception:
        return 0.0, 0.0

    voiced_flag = np.nan_to_num(np.asarray(voiced_flag, dtype=float))
    voiced_prob = np.nan_to_num(np.asarray(voiced_prob, dtype=float))
    return float(voiced_flag.mean()), float(np.median(voiced_prob))


def frame_analysis(mix: np.ndarray, voc: np.ndarray, sr: int) -> dict:
    """Per-frame view of the stem against the mixture.

    Split out from `stem_metrics` so a notebook can plot exactly what the
    detector saw -- where it thinks the vocals are, and how sure it is.
    """
    hop = max(1, int(round(HOP_S * sr)))
    frame = max(hop * 2, int(round(WIN_S * sr)))

    m, v = _mono(mix), _mono(voc)
    n = min(m.size, v.size)
    m, v = m[:n], v[:n]

    m_rms, v_rms = _frame_rms(m, frame, hop), _frame_rms(v, frame, hop)
    k = min(m_rms.size, v_rms.size)
    m_rms, v_rms = m_rms[:k], v_rms[:k]

    m_db, v_db = _db(m_rms), _db(v_rms)
    rel_db = v_db - m_db

    # "Music is playing here" -- excludes lead-in silence and fade-outs, and
    # keeps a quiet master from being judged against an absolute floor.
    peak = float(np.percentile(m_db, 95)) if k else -120.0
    playing = (m_db > peak - 35.0) & (m_db > -90.0)  # -90 dBFS = digital silence
    if playing.sum() < 4:
        playing = m_db > -90.0

    # A frame counts as vocal-active if the stem is a meaningful share of the
    # mixture there, and is above the track's own noise floor. Both gates are
    # relative to the mixture, so a quiet master scores like a loud one.
    active = playing & (rel_db > -20.0) & (v_db > peak - 60.0)

    return {
        "times": np.arange(k) * HOP_S,
        "mix_db": m_db,
        "vocal_db": v_db,
        "rel_db": rel_db,
        "playing": playing,
        "active": active,
        "peak_db": peak,
        "hop": hop,
        "mono_mix": m,
        "mono_vocal": v,
        "sr": sr,
    }


def stem_metrics(mix: np.ndarray, voc: np.ndarray, sr: int) -> dict:
    """Compare a vocals stem against the mixture it came from."""
    f = frame_analysis(mix, voc, sr)
    m, v, active, playing = f["mono_mix"], f["mono_vocal"], f["active"], f["playing"]
    rel_db = f["rel_db"]

    n_playing = int(playing.sum())
    activity_ratio = float(active.sum() / n_playing) if n_playing else 0.0
    rel_playing = rel_db[playing] if n_playing else np.array([-120.0])

    voiced_ratio, voiced_prob = _voicedness(v, sr, active, f["hop"])

    return {
        "duration_s": round(m.size / sr, 2),
        "global_rel_db": round(
            20.0 * float(np.log10((np.sqrt(np.mean(v**2)) + EPS) / (np.sqrt(np.mean(m**2)) + EPS))), 2
        ),
        "p95_rel_db": round(float(np.percentile(rel_playing, 95)), 2),
        "median_rel_db": round(float(np.median(rel_playing)), 2),
        "activity_ratio": round(activity_ratio, 4),
        "max_run_s": round(_longest_run(active, HOP_S), 2),
        "voiced_ratio": round(voiced_ratio, 4),
        "voiced_prob": round(voiced_prob, 4),
        "vocal_peak_dbfs": round(float(np.max(f["vocal_db"])) if f["vocal_db"].size else -120.0, 2),
    }
