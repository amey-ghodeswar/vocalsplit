"""Command line interface."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .audio import find_audio_files
from .decide import Thresholds
from .pipeline import Config, analyze, write_outputs
from .separator import KNOWN_MODELS, DemucsSeparator


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("inputs", nargs="+", help="audio files and/or directories")
    p.add_argument("-m", "--model", default="htdemucs", choices=KNOWN_MODELS)
    p.add_argument("-d", "--device", default="auto", help="auto | cuda | cpu")
    p.add_argument("--shifts", type=int, default=0,
                   help="test-time augmentation passes; 2-5 is slower but cleaner")
    p.add_argument("--overlap", type=float, default=0.25)
    p.add_argument("--segment", type=float, default=None,
                   help="seconds per chunk; lower it if you hit VRAM limits")
    p.add_argument("--no-probe", action="store_true",
                   help="analyse the whole track instead of pre-screening excerpts")
    p.add_argument("--thresholds", type=Path, default=None,
                   help="JSON from `vocalsplit calibrate`")
    p.add_argument("-q", "--quiet", action="store_true")


def _config(a: argparse.Namespace) -> Config:
    return Config(
        model=a.model,
        device=a.device,
        shifts=a.shifts,
        overlap=a.overlap,
        segment=a.segment,
        probe=not a.no_probe,
        force=getattr(a, "force", False),
        keep_stems=getattr(a, "stems", False),
        emit_passthrough=getattr(a, "emit_passthrough", False),
        out_format=getattr(a, "format", "wav"),
        subtype=getattr(a, "subtype", "PCM_24"),
        bitrate=getattr(a, "bitrate", 320),
        thresholds=Thresholds.load(a.thresholds) if a.thresholds else Thresholds(),
    )


def _gather(inputs: list[str]) -> list[Path]:
    files: list[Path] = []
    for i in inputs:
        found = find_audio_files(i)
        if not found:
            print(f"warning: no audio found at {i}", file=sys.stderr)
        files += found
    return files


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vocalsplit",
        description="Split vocals from instruments, and recognise instrumental tracks "
                    "instead of inventing a vocal stem for them.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("separate", help="classify, then split when there are vocals")
    _common(sp)
    sp.add_argument("-o", "--out", type=Path, default=Path("output"))
    sp.add_argument("--force", action="store_true",
                    help="separate even if the track is judged instrumental")
    sp.add_argument("--stems", action="store_true", help="also write drums/bass/other")
    sp.add_argument("--emit-passthrough", action="store_true",
                    help="for instrumentals, still write instrumental.wav (a copy of the input)")
    sp.add_argument("--format", default="wav", choices=["wav", "flac", "mp3"])
    sp.add_argument("--subtype", default="PCM_24", choices=["PCM_16", "PCM_24", "FLOAT"])
    sp.add_argument("--bitrate", type=int, default=320, help="mp3 kbps")

    dp = sub.add_parser("detect", help="classify only, write no audio")
    _common(dp)
    dp.add_argument("--json", action="store_true", help="emit one JSON array on stdout")

    cp = sub.add_parser("calibrate", help="tune thresholds on labelled folders")
    cp.add_argument("--vocal", type=Path, required=True)
    cp.add_argument("--instrumental", type=Path, required=True)
    cp.add_argument("-m", "--model", default="htdemucs", choices=KNOWN_MODELS)
    cp.add_argument("-d", "--device", default="auto")
    cp.add_argument("--shifts", type=int, default=0)
    cp.add_argument("--overlap", type=float, default=0.25)
    cp.add_argument("--segment", type=float, default=None)
    cp.add_argument("-o", "--out", type=Path, default=Path("thresholds.json"))
    cp.add_argument("--dump", type=Path, default=None, help="write raw per-file metrics")
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)

    if a.cmd == "calibrate":
        from .calibrate import collect, dump_rows, summarise

        cfg = _config(argparse.Namespace(**vars(a), no_probe=True, thresholds=None))
        rows = collect({"vocal": a.vocal, "instrumental": a.instrumental}, cfg)
        if a.dump:
            dump_rows(rows, a.dump)
        summarise(rows, Thresholds(), a.out)
        return 0

    cfg = _config(a)
    files = _gather(a.inputs)
    if not files:
        print("no input files", file=sys.stderr)
        return 1

    log = (lambda *_: None) if a.quiet else print
    sep = DemucsSeparator(cfg.model, cfg.device, cfg.shifts, cfg.overlap, cfg.segment)

    reports = []
    for n, f in enumerate(files, 1):
        log(f"[{n}/{len(files)}] {f.name}")
        try:
            r = analyze(f, cfg, separator=sep, log=log)
        except Exception as e:  # keep a bad file from killing a batch
            print(f"  ERROR {f.name}: {e}", file=sys.stderr)
            reports.append({"file": str(f), "error": str(e)})
            continue
        if a.cmd == "separate":
            r = write_outputs(r, a.out, cfg, log=log)
        else:
            for k in ("stems", "mixture", "sr"):
                r.pop(k, None)
        reports.append(r)

    if a.cmd == "detect":
        if a.json:
            print(json.dumps(reports, indent=2))
        else:
            log("")
            for r in reports:
                if "error" in r:
                    print(f"{Path(r['file']).name:50s} ERROR")
                else:
                    print(f"{Path(r['file']).name:50s} {r['label']:12s} "
                          f"p_vocal={r['p_vocal']:.2f}  {r['reasons'][0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
