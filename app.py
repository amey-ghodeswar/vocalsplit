"""Web UI for vocalsplit -- the same file runs locally and on Hugging Face Spaces.

    python app.py                     -> http://127.0.0.1:7860
    python app.py --share             -> plus a temporary public link
    git push to a Space               -> permanent public URL

On Spaces this picks up whatever hardware the Space has; on CPU basic a 4-minute
track takes ~4 minutes instead of ~30 seconds, so set the Space to a GPU tier if
you expect anyone else to use it.
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np

from vocalsplit.audio import load_audio, save_audio
from vocalsplit.decide import Thresholds, classify
from vocalsplit.features import stem_metrics
from vocalsplit.separator import DemucsSeparator

BITRATE = 320
_CACHE: dict[str, DemucsSeparator] = {}


def get_separator(model: str, shifts: int) -> DemucsSeparator:
    """One separator per (model, shifts). Reloading weights per request is waste."""
    key = f"{model}:{shifts}"
    if key not in _CACHE:
        _CACHE[key] = DemucsSeparator(model, device="auto", shifts=shifts)
    return _CACHE[key]


def _verdict_markdown(v: dict, m: dict) -> str:
    icon = {"vocal": "🎤", "instrumental": "🎹", "uncertain": "❓"}[v["label"]]
    head = {
        "vocal": "Vocals found — split into two tracks below.",
        "instrumental": "No vocals in this track. Nothing was separated.",
        "uncertain": "Ambiguous. Split anyway, but check the result.",
    }[v["label"]]

    bars = "\n".join(
        f"| {axis} | `{'█' * int(val * 24):<24}` | {val:.2f} |"
        for axis, val in v["votes"].items()
    )
    return f"""### {icon} {v['label'].upper()} &nbsp;&nbsp; `p_vocal = {v['p_vocal']:.3f}`

{head}

*{v['reasons'][0]}*

| axis | | vote |
|---|---|---|
{bars}

<sub>level = how loud the vocal stem is against the mixture ·
activity = share of the music that is vocal-active ·
continuity = longest unbroken phrase ({m['max_run_s']:.1f}s) ·
voiced = how periodic the loud parts are</sub>"""


def separate(audio_path, model, quality, progress=gr.Progress()):
    if not audio_path:
        raise gr.Error("Upload an audio file first.")

    shifts = {"fast": 0, "better": 2, "best": 5}[quality]
    progress(0.05, desc=f"loading {model}…")
    sep = get_separator(model, shifts)

    progress(0.15, desc="decoding audio…")
    mix, sr = load_audio(audio_path, sr=sep.samplerate)
    minutes = mix.shape[1] / sr / 60
    if minutes > 15:
        raise gr.Error(f"Track is {minutes:.1f} min; the limit here is 15.")

    progress(0.25, desc=f"separating {minutes:.1f} min on {sep.device}…")
    voc = sep.separate(mix, sr)["vocals"]

    progress(0.80, desc="checking whether that is really a voice…")
    metrics = stem_metrics(mix, voc, sr)
    verdict = classify(metrics, Thresholds())
    md = _verdict_markdown(verdict, metrics)

    # The whole point: an instrumental track gets no vocal file. Demucs will
    # always hand back a "vocals" stem, but here it is synth partials and
    # reverb tails, and shipping it as a vocal would be a lie.
    if verdict["label"] == "instrumental":
        return md, None, None, metrics

    progress(0.90, desc="encoding mp3…")
    n = min(mix.shape[1], voc.shape[1])
    voc, inst = voc[:, :n], mix[:, :n] - voc[:, :n]

    # One shared gain across both stems: either can exceed full scale, and
    # scaling them independently would break `vocals + instrumental == input`.
    peak = max(float(np.abs(a).max()) for a in (voc, inst))
    gain = min(1.0, 0.999 / peak) if peak > 0 else 1.0

    stem = Path(audio_path).stem[:60]
    out = Path(tempfile.mkdtemp(prefix="vocalsplit-"))
    v_path, i_path = out / f"{stem} (vocals).mp3", out / f"{stem} (instrumental).mp3"
    save_audio(v_path, voc * gain, sr, bitrate=BITRATE)
    save_audio(i_path, inst * gain, sr, bitrate=BITRATE)

    return md, str(v_path), str(i_path), metrics


CSS = """
.gradio-container {max-width: 980px !important}
footer {display: none !important}
"""

# Gradio 6 moved theme and css from the Blocks constructor to launch().
with gr.Blocks(title="vocalsplit") as demo:
    gr.Markdown(
        """# vocalsplit
Split a song into **vocals** and **instrumental** — and, if the track has no
singing in it, say so instead of inventing a vocal stem out of bleed.

Demucs (MIT) does the separation; the instrumental check is four measurements
on the separated stem, not a guess about the mixture."""
    )

    with gr.Row():
        with gr.Column(scale=3):
            audio_in = gr.Audio(type="filepath", label="Song", sources=["upload"])
        with gr.Column(scale=2):
            model = gr.Dropdown(
                ["htdemucs", "htdemucs_ft", "mdx_extra"],
                value="htdemucs",
                label="Model",
                info="htdemucs_ft is ~4× slower for ~0.5 dB",
            )
            quality = gr.Radio(
                ["fast", "better", "best"],
                value="fast",
                label="Quality",
                info="test-time augmentation: 1×, 3×, 6× the compute",
            )
            go = gr.Button("Separate", variant="primary", size="lg")

    verdict_md = gr.Markdown()
    with gr.Row():
        vocals_out = gr.Audio(label="Vocals", type="filepath", interactive=False)
        inst_out = gr.Audio(label="Instrumental", type="filepath", interactive=False)
    with gr.Accordion("Raw metrics", open=False):
        metrics_out = gr.JSON()

    go.click(
        separate,
        inputs=[audio_in, model, quality],
        outputs=[verdict_md, vocals_out, inst_out, metrics_out],
    )

    gr.Markdown(
        """<sub>`instrumental` is computed as `mixture − vocals`, so the two files sum
back to exactly what you uploaded. Separation is not perfect — expect audible bleed on
dense mixes. Backing vocals ride with the lead.</sub>"""
    )

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--share", action="store_true", help="temporary public link")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--host", default="127.0.0.1")
    a = p.parse_args()
    demo.queue(max_size=8).launch(
        server_name=a.host,
        server_port=a.port,
        share=a.share,
        theme=gr.themes.Soft(),
        css=CSS,
        max_file_size="60mb",
    )
