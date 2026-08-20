"""Tune thresholds against your own labelled audio.

    vocalsplit calibrate --vocal ./has_vocals --instrumental ./instrumentals

Prints the metric distribution for each class, sweeps each threshold for the
split that separates them best, and can write the result to a JSON file that
`--thresholds` will load.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .audio import find_audio_files, load_audio
from .decide import Thresholds, classify
from .features import stem_metrics
from .separator import DemucsSeparator

METRIC_KEYS = ("p95_rel_db", "activity_ratio", "max_run_s", "voiced_ratio")


def collect(dirs: dict[str, Path], cfg, log=print) -> list[dict]:
    sep = DemucsSeparator(cfg.model, cfg.device, cfg.shifts, cfg.overlap, cfg.segment)
    rows = []
    for label, d in dirs.items():
        files = find_audio_files(d)
        log(f"{label}: {len(files)} files")
        for f in files:
            x, sr = load_audio(f, sr=sep.samplerate)
            m = stem_metrics(x, sep.separate(x, sr)["vocals"], sr)
            rows.append({"file": str(f), "truth": label, **m})
            log(f"  {f.name[:48]:48s} " + "  ".join(f"{k}={m[k]}" for k in METRIC_KEYS))
    return rows


def _best_split(vals_v: np.ndarray, vals_i: np.ndarray) -> tuple[float, float]:
    """Threshold maximising balanced accuracy for 'vocal is the larger side'."""
    cands = np.unique(np.concatenate([vals_v, vals_i]))
    if cands.size < 2:
        return float(cands[0]) if cands.size else 0.0, 0.0
    mids = (cands[:-1] + cands[1:]) / 2
    best, best_acc = float(mids[0]), 0.0
    for t in mids:
        acc = 0.5 * ((vals_v > t).mean() + (vals_i <= t).mean())
        if acc > best_acc:
            best, best_acc = float(t), float(acc)
    return best, best_acc


def summarise(rows: list[dict], base: Thresholds, out: Path | None, log=print) -> Thresholds:
    v = [r for r in rows if r["truth"] == "vocal"]
    i = [r for r in rows if r["truth"] == "instrumental"]
    if not v or not i:
        log("need examples of BOTH classes to calibrate")
        return base

    tuned = Thresholds(**{**base.__dict__})
    log("\nmetric            vocal (med/p10)      instrumental (med/p90)   split   bal-acc")
    for key in METRIC_KEYS:
        av = np.array([r[key] for r in v], float)
        ai = np.array([r[key] for r in i], float)
        t, acc = _best_split(av, ai)
        setattr(tuned, key, round(t, 4))
        log(
            f"{key:16s}  {np.median(av):8.3f}/{np.percentile(av,10):8.3f}   "
            f"{np.median(ai):8.3f}/{np.percentile(ai,90):8.3f}   "
            f"{t:8.3f}  {acc*100:5.1f}%"
        )

    for name, th in (("default", base), ("tuned", tuned)):
        wrong = [
            (r["file"], r["truth"], classify(r, th)["label"])
            for r in rows
            if classify(r, th)["label"] != r["truth"]
        ]
        log(f"\n{name} thresholds: {len(rows)-len(wrong)}/{len(rows)} correct")
        for f, truth, got in wrong:
            log(f"  MISS {Path(f).name[:56]:56s} truth={truth} got={got}")

    if out:
        tuned.save(out)
        log(f"\nwrote {out}  --  use it with:  vocalsplit separate --thresholds {out} ...")
    return tuned


def dump_rows(rows: list[dict], path: Path) -> None:
    path.write_text(json.dumps(rows, indent=2))
