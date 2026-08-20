"""Turn stem metrics into a vocal / instrumental verdict."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class Thresholds:
    """Midpoints of each soft vote, plus the widths that set how sharply it flips.

    Defaults were set from htdemucs behaviour and are deliberately conservative
    on the instrumental side: calling a vocal track instrumental loses a stem,
    while the reverse just wastes a little compute. Retune with `vocalsplit
    calibrate` against your own library.
    """

    p95_rel_db: float = -24.0
    p95_rel_db_w: float = 3.0

    activity_ratio: float = 0.08
    activity_ratio_w: float = 0.7  # in natural-log units

    max_run_s: float = 1.5
    max_run_s_w: float = 1.0

    voiced_ratio: float = 0.35
    voiced_ratio_w: float = 0.12

    # weights must sum to 1
    w_level: float = 0.30
    w_activity: float = 0.25
    w_continuity: float = 0.20
    w_voiced: float = 0.25

    # decision bands on p_vocal
    vocal_at: float = 0.60
    instrumental_at: float = 0.40

    # hard overrides, applied before the soft vote
    hard_instrumental_p95_rel_db: float = -34.0
    hard_instrumental_activity: float = 0.005

    @classmethod
    def load(cls, path: str | Path) -> "Thresholds":
        return cls(**json.loads(Path(path).read_text()))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))


def _sigmoid(z: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(z, -30, 30))))


def classify(metrics: dict, t: Thresholds | None = None) -> dict:
    """Return {label, p_vocal, votes, reasons}. label in vocal/instrumental/uncertain."""
    t = t or Thresholds()

    votes = {
        "level": _sigmoid((metrics["p95_rel_db"] - t.p95_rel_db) / t.p95_rel_db_w),
        "activity": _sigmoid(
            (np.log(metrics["activity_ratio"] + 1e-4) - np.log(t.activity_ratio))
            / t.activity_ratio_w
        ),
        "continuity": _sigmoid((metrics["max_run_s"] - t.max_run_s) / t.max_run_s_w),
        "voiced": _sigmoid((metrics["voiced_ratio"] - t.voiced_ratio) / t.voiced_ratio_w),
    }
    p = (
        t.w_level * votes["level"]
        + t.w_activity * votes["activity"]
        + t.w_continuity * votes["continuity"]
        + t.w_voiced * votes["voiced"]
    )

    reasons: list[str] = []
    if metrics["p95_rel_db"] < t.hard_instrumental_p95_rel_db:
        p = min(p, 0.02)
        reasons.append(
            f"vocal stem peaks {metrics['p95_rel_db']:.1f} dB below the mixture "
            "-- that is bleed, not a voice"
        )
    if metrics["activity_ratio"] < t.hard_instrumental_activity:
        p = min(p, 0.02)
        reasons.append(f"vocal-active in only {metrics['activity_ratio']*100:.2f}% of the music")

    if p >= t.vocal_at:
        label = "vocal"
    elif p <= t.instrumental_at:
        label = "instrumental"
    else:
        label = "uncertain"

    if not reasons:
        if label == "vocal":
            reasons.append(
                f"{metrics['activity_ratio']*100:.1f}% vocal-active, longest phrase "
                f"{metrics['max_run_s']:.1f}s, {metrics['voiced_ratio']*100:.0f}% voiced"
            )
        elif label == "instrumental":
            reasons.append(
                f"only {metrics['activity_ratio']*100:.1f}% vocal-active and "
                f"{metrics['voiced_ratio']*100:.0f}% voiced -- consistent with instrument bleed"
            )
        else:
            reasons.append("evidence is mixed; treating as vocal but flagging it")

    return {
        "label": label,
        "p_vocal": round(float(p), 4),
        "votes": {k: round(v, 4) for k, v in votes.items()},
        "reasons": reasons,
    }
