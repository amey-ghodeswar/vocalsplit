"""Tests for the detector logic, with no separator in the loop.

We hand `stem_metrics` a mixture and a hand-built "vocals stem" that stands in
for what a separator would return -- real singing in one case, plausible bleed
in the other -- and check the verdict. This isolates the part of the system
that is actually ours; Demucs has its own test suite.

    pytest tests/ -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from make_test_audio import SR, build_bed, build_melody, stereo  # noqa: E402

from vocalsplit.decide import Thresholds, classify  # noqa: E402
from vocalsplit.features import _longest_run, stem_metrics  # noqa: E402

DUR = 20.0
N = int(DUR * SR)


def _bleed(bed: np.ndarray, level_db: float = -32.0) -> np.ndarray:
    """What a separator hands back for an instrumental: a quiet, filtered,
    aperiodic ghost of the mix that follows its envelope."""
    rng = np.random.default_rng(7)
    hf = np.diff(bed, prepend=0.0)  # crude high-pass -> hats/transient wash
    noise = rng.standard_normal(bed.size) * 0.15
    env = np.abs(bed) / (np.abs(bed).max() + 1e-9)
    x = (hf * 0.6 + noise * env) * 10 ** (level_db / 20)
    return x / (np.abs(x).max() + 1e-9) * 10 ** (level_db / 20)


def _mix_and_stem() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bed, voice = build_bed(N), build_melody(N)
    bed /= np.abs(bed).max() + 1e-9
    voice /= np.abs(voice).max() + 1e-9
    return bed, voice, _bleed(bed)


def test_sung_stem_reads_as_vocal():
    bed, voice, _ = _mix_and_stem()
    mix = stereo(bed * 0.6 + voice * 0.7)
    m = stem_metrics(mix, stereo(voice * 0.7), SR)
    v = classify(m, Thresholds())
    assert v["label"] == "vocal", (v, m)
    assert m["max_run_s"] > 1.0
    assert m["voiced_ratio"] > 0.3


def test_bleed_stem_reads_as_instrumental():
    bed, _, bleed = _mix_and_stem()
    mix = stereo(bed)
    m = stem_metrics(mix, stereo(bleed), SR)
    v = classify(m, Thresholds())
    assert v["label"] == "instrumental", (v, m)
    assert m["p95_rel_db"] < -20


def test_silent_stem_is_instrumental():
    bed, _, _ = _mix_and_stem()
    m = stem_metrics(stereo(bed), stereo(np.zeros(N)), SR)
    v = classify(m, Thresholds())
    assert v["label"] == "instrumental"
    assert m["activity_ratio"] == 0.0


def test_level_is_relative_not_absolute():
    """A quiet master must not be judged differently from a loud one."""
    bed, voice, _ = _mix_and_stem()
    loud = stem_metrics(stereo(bed * 0.6 + voice * 0.7), stereo(voice * 0.7), SR)
    quiet = stem_metrics(
        stereo((bed * 0.6 + voice * 0.7) * 0.02), stereo(voice * 0.7 * 0.02), SR
    )
    assert abs(loud["p95_rel_db"] - quiet["p95_rel_db"]) < 1.5
    assert classify(loud)["label"] == classify(quiet)["label"]


def test_longest_run_bridges_short_gaps():
    mask = np.array([1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1], bool)
    assert _longest_run(mask, 0.05, bridge_frames=6) == 8 * 0.05  # gap of 2 bridged
    assert _longest_run(mask, 0.05, bridge_frames=1) == 3 * 0.05  # gap of 2 not bridged


def test_hard_override_beats_the_soft_vote():
    """Even with perfect continuity and voicedness, a stem 40 dB down is bleed."""
    m = {
        "p95_rel_db": -40.0, "activity_ratio": 0.9, "max_run_s": 30.0,
        "voiced_ratio": 0.9, "voiced_prob": 0.9,
    }
    v = classify(m, Thresholds())
    assert v["label"] == "instrumental"
    assert v["p_vocal"] <= 0.05


def test_uncertain_band_is_reachable():
    t = Thresholds()
    m = {
        "p95_rel_db": t.p95_rel_db, "activity_ratio": t.activity_ratio,
        "max_run_s": t.max_run_s, "voiced_ratio": t.voiced_ratio, "voiced_prob": 0.5,
    }
    v = classify(m, t)
    assert v["label"] == "uncertain"
    assert 0.4 < v["p_vocal"] < 0.6
