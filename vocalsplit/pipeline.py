"""End-to-end: load -> probe -> separate -> verify -> write."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .audio import load_audio, save_audio
from .decide import Thresholds, classify
from .features import stem_metrics
from .separator import DemucsSeparator


@dataclass
class Config:
    model: str = "htdemucs"
    device: str | None = None
    shifts: int = 0
    overlap: float = 0.25
    segment: float | None = None

    # Pre-screen: separate a few short excerpts before committing to the whole
    # track. Only ever used to *skip* work, never to claim a track has vocals.
    probe: bool = True
    probe_windows: int = 6
    probe_window_s: float = 10.0
    probe_skip_below: float = 0.20  # skip full separation when p_vocal <= this

    force: bool = False           # separate even when judged instrumental
    keep_stems: bool = False      # also write drums/bass/other
    emit_passthrough: bool = False  # for instrumentals, copy input as instrumental.wav
    out_format: str = "wav"
    subtype: str = "PCM_24"
    bitrate: int = 320  # mp3 only
    thresholds: Thresholds = field(default_factory=Thresholds)


def _probe_excerpts(x: np.ndarray, sr: int, n: int, win_s: float) -> np.ndarray | None:
    """Concatenate `n` evenly spaced windows into one pseudo-track."""
    total = x.shape[1]
    win = int(win_s * sr)
    if total <= win * 2:
        return None  # short enough that probing saves nothing

    n = max(1, min(n, total // win))
    starts = np.linspace(0, total - win, n).astype(int)
    return np.concatenate([x[:, s : s + win] for s in starts], axis=1)


def analyze(
    path: str | Path,
    cfg: Config,
    separator: DemucsSeparator | None = None,
    log=print,
) -> dict:
    """Classify (and, unless instrumental, separate) one file. Returns a report."""
    path = Path(path)
    t0 = time.perf_counter()

    sep = separator or DemucsSeparator(
        cfg.model, cfg.device, cfg.shifts, cfg.overlap, cfg.segment
    )
    x, sr = load_audio(path, sr=sep.samplerate)
    log(f"  loaded {x.shape[1] / sr:6.1f}s  [{sep.device}]")

    report: dict = {
        "file": str(path),
        "model": cfg.model,
        "device": sep.device,
        "duration_s": round(x.shape[1] / sr, 2),
        "probe": None,
        "separated": False,
    }

    # --- stage 1: cheap probe -------------------------------------------------
    if cfg.probe and not cfg.force:
        exc = _probe_excerpts(x, sr, cfg.probe_windows, cfg.probe_window_s)
        if exc is not None:
            pm = stem_metrics(exc, sep.separate(exc, sr)["vocals"], sr)
            pv = classify(pm, cfg.thresholds)
            report["probe"] = {"metrics": pm, **pv}
            log(f"  probe   p_vocal={pv['p_vocal']:.2f} ({pv['label']})")
            if pv["p_vocal"] <= cfg.probe_skip_below:
                report.update(
                    label="instrumental",
                    p_vocal=pv["p_vocal"],
                    votes=pv["votes"],
                    metrics=pm,
                    reasons=pv["reasons"] + ["decided from probe; full separation skipped"],
                    elapsed_s=round(time.perf_counter() - t0, 2),
                )
                return report

    # --- stage 2: full separation + verdict ----------------------------------
    stems = sep.separate(x, sr)
    voc = stems["vocals"]
    metrics = stem_metrics(x, voc, sr)
    verdict = classify(metrics, cfg.thresholds)
    report.update(metrics=metrics, **verdict)
    log(f"  verdict p_vocal={verdict['p_vocal']:.2f} -> {verdict['label']}")

    report["stems"] = stems
    report["mixture"] = x
    report["sr"] = sr
    report["elapsed_s"] = round(time.perf_counter() - t0, 2)
    return report


def write_outputs(report: dict, out_dir: Path, cfg: Config, log=print) -> dict:
    """Write audio + report.json. Strips the in-memory arrays from the report."""
    src = Path(report["file"])
    dest = out_dir / src.stem
    dest.mkdir(parents=True, exist_ok=True)
    ext = "." + cfg.out_format.lstrip(".")
    written: list[str] = []

    stems = report.pop("stems", None)
    x = report.pop("mixture", None)
    sr = report.pop("sr", None)

    instrumental_only = report["label"] == "instrumental" and not cfg.force

    if instrumental_only:
        # The whole point: do not manufacture a vocal stem that is really bleed.
        if cfg.emit_passthrough:
            if x is None:
                x, sr = load_audio(src, sr=44100)
            save_audio(dest / f"instrumental{ext}", x, sr, cfg.subtype, cfg.bitrate)
            written.append(f"instrumental{ext}")
        log(f"  instrumental -- no separation written ({report['reasons'][0]})")
    elif stems is not None:
        voc = stems["vocals"]
        # Exact complement rather than summing the other stems: whatever the model
        # failed to attribute still ends up in the instrumental instead of vanishing.
        n = min(x.shape[1], voc.shape[1])
        voc, inst = voc[:, :n], x[:, :n] - voc[:, :n]

        # A separated stem routinely peaks above the mixture, and the complement
        # can too, so either can exceed full scale. Integer wav and mp3 would clip
        # that away silently. One gain across every stem keeps them summable.
        outs = {"vocals": voc, "instrumental": inst}
        if cfg.keep_stems:
            outs.update({k: v[:, :n] for k, v in stems.items() if k != "vocals"})
        peak = max(float(np.abs(a).max()) for a in outs.values())
        gain = min(1.0, 0.999 / peak) if peak > 0 else 1.0
        if gain < 1.0:
            log(f"  peak {20 * np.log10(peak):+.2f} dBFS -> scaling all stems by "
                f"{20 * np.log10(gain):.2f} dB to avoid clipping")
            report["output_gain_db"] = round(float(20 * np.log10(gain)), 3)

        for name, a in outs.items():
            save_audio(dest / f"{name}{ext}", a * gain, sr, cfg.subtype, cfg.bitrate)
            written.append(f"{name}{ext}")
        report["separated"] = True
        log(f"  wrote {', '.join(written)}")

    report["output_dir"] = str(dest)
    report["written"] = written
    (dest / "report.json").write_text(json.dumps(report, indent=2))
    return report
