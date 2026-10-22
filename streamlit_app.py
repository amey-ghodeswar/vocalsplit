"""Web UI for vocalsplit on Streamlit Community Cloud (free, CPU only).

    streamlit run streamlit_app.py          -> http://localhost:8501

Same behaviour as app.py (the Gradio version): upload a song, get a verdict and,
if there are vocals, two mp3s. This file exists because Hugging Face now charges
for Gradio Spaces while Streamlit Community Cloud is still free with a GitHub
login. Its limits shape everything below:

  * ~2.7 GB RAM.  A 4-minute track peaks near 2 GB in this pipeline, so uploads
    are capped at MAX_MINUTES and only one separation runs at a time (the lock).
  * up to 2 shared vCPUs.  Expect 1-3 minutes per minute of audio.  htdemucs_ft
    and test-time shifts are not offered here; they would take 4-6x longer.
  * the app sleeps after inactivity; the first visitor waits for it to wake.
"""
from __future__ import annotations

import hmac
import io
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import streamlit as st
import torch

from vocalsplit.audio import load_audio, save_audio
from vocalsplit.decide import Thresholds, classify
from vocalsplit.features import stem_metrics
from vocalsplit.separator import DemucsSeparator

MODEL = "htdemucs"
BITRATE = 192          # web delivery; the CLI defaults to 320
MAX_MINUTES = 6.0
MAX_UPLOAD_MB = 60
THREADS = 2

# One separation at a time, whatever the number of browser sessions. Two at once
# would double the working set and get the container killed.
_RUN_LOCK = threading.Lock()


@st.cache_resource(show_spinner="Loading the separation model (first run only)...")
def get_separator() -> DemucsSeparator:
    torch.set_num_threads(THREADS)
    return DemucsSeparator(MODEL, device="cpu", shifts=0)


def _decode(upload) -> tuple[np.ndarray, int, Path]:
    suffix = Path(upload.name).suffix.lower() or ".mp3"
    tmp = Path(tempfile.mkdtemp(prefix="vocalsplit-")) / f"input{suffix}"
    tmp.write_bytes(upload.getvalue())
    mix, sr = load_audio(tmp, sr=get_separator().samplerate)
    return mix, sr, tmp


def _encode_mp3(x: np.ndarray, sr: int) -> bytes:
    with tempfile.TemporaryDirectory(prefix="vocalsplit-") as d:
        p = Path(d) / "stem.mp3"
        save_audio(p, x, sr, bitrate=BITRATE)
        return p.read_bytes()


def separate(mix: np.ndarray, sr: int, status) -> dict:
    """The same steps as app.py's separate(), reporting progress into `status`."""
    sep = get_separator()
    minutes = mix.shape[1] / sr / 60

    status.update(label=f"Separating {minutes:.1f} min of audio on CPU. "
                        f"This takes a few minutes here, a good moment for a coffee ☕. "
                        f"Please keep the tab open.")
    t0 = time.time()
    voc = sep.separate(mix, sr)["vocals"]
    t_sep = time.time() - t0

    status.update(label="Checking whether that is really a voice...")
    metrics = stem_metrics(mix, voc, sr)
    verdict = classify(metrics, Thresholds())
    result = {"verdict": verdict, "metrics": metrics, "seconds": t_sep,
              "vocals": None, "instrumental": None}

    # The whole point: an instrumental track gets no vocal file. Demucs always
    # hands back a "vocals" stem, but here it is synth partials and reverb
    # tails, and shipping it as a vocal would be a lie.
    if verdict["label"] == "instrumental":
        return result

    status.update(label="Encoding mp3...")
    n = min(mix.shape[1], voc.shape[1])
    voc, inst = voc[:, :n], mix[:, :n] - voc[:, :n]
    # One shared gain across both stems: either can exceed full scale, and
    # scaling them independently would break `vocals + instrumental == input`.
    peak = max(float(np.abs(a).max()) for a in (voc, inst))
    gain = min(1.0, 0.999 / peak) if peak > 0 else 1.0
    result["vocals"] = _encode_mp3(voc * gain, sr)
    del voc
    result["instrumental"] = _encode_mp3(inst * gain, sr)
    return result


def show_verdict(v: dict, m: dict) -> None:
    icon = {"vocal": "🎤", "instrumental": "🎹", "uncertain": "❓"}[v["label"]]
    head = {
        "vocal": "Vocals found. Split into two tracks below.",
        "instrumental": "No vocals in this track. Nothing was separated.",
        "uncertain": "Ambiguous. Split anyway, but check the result.",
    }[v["label"]]
    st.subheader(f"{icon} {v['label'].upper()}  ·  p_vocal = {v['p_vocal']:.3f}")
    st.write(head)
    st.caption(v["reasons"][0])

    cols = st.columns(4)
    hints = {
        "level": "how loud the vocal stem is against the mixture",
        "activity": "share of the music that is vocal-active",
        "continuity": f"longest unbroken phrase ({m['max_run_s']:.1f} s)",
        "voiced": "how periodic the loud parts are",
    }
    for col, (axis, val) in zip(cols, v["votes"].items()):
        with col:
            st.metric(axis, f"{val:.2f}", help=hints[axis])
            st.progress(float(val))


def _configured_key() -> str:
    """The shared access key, or "" when none is configured.

    On Streamlit Community Cloud: app menu -> Settings -> Secrets ->
        ACCESS_KEY = "whatever-you-hand-out"
    Locally: the same line in .streamlit/secrets.toml (git-ignored).
    With no key set anywhere the app runs open, so `streamlit run` needs no setup.
    """
    try:
        return str(st.secrets.get("ACCESS_KEY", "")).strip()
    except Exception:  # noqa: BLE001  (no secrets file at all)
        return ""


def require_key() -> None:
    """Show nothing but a key prompt until the visitor enters the configured key.

    This is a gate against passers-by, not real authentication: one shared key,
    remembered for the browser session only (a page reload asks again). Good
    enough to keep a free CPU server for the people you actually sent the link to.
    """
    key = _configured_key()
    if not key or st.session_state.get("authed"):
        return

    hero()
    st.write("This is a private preview. Enter the access key you were given.")
    with st.form("gate"):
        given = st.text_input("🎟️ Access key", type="password")
        submitted = st.form_submit_button("Enter", type="primary")
    if submitted:
        if hmac.compare_digest(given.strip(), key):
            st.session_state["authed"] = True
            st.rerun()
        st.error("That key is not right.")
    st.stop()


# ----------------------------------------------------------------------------- look

# A painted room (static/cafe.webp, the owner's own illustration) fills the page.
# The app sits on a sheet of warm paper in the middle, where the painting has its
# empty wall, so the guitar, gramophone and singer stay visible on the left and the
# girl at the window on the right. Widget colours come from .streamlit/config.toml;
# this sheet does the canvas, the paper, the fonts and the hero.
#
# The image is served by Streamlit's static file support (server.enableStaticServing
# in config.toml), which is why the URLs below are relative "app/static/..." paths.

THEME_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght,SOFT@9..144,400..700,50..100&family=Figtree:wght@400;500;600&display=swap');

:root {
  --cream: #F7EBDC;
  --cream-soft: rgba(247, 235, 220, .78);
  --lamp: #F6C87A;
  --caramel: #C9843A;
  --ink-shadow: 0 1px 2px rgba(0, 0, 0, .7), 0 2px 10px rgba(10, 4, 14, .6), 0 0 26px rgba(10, 4, 14, .5);
  --glass: rgba(20, 12, 24, .16);   /* the lightest tint that still shows an edge */
  --glass-edge: rgba(247, 235, 220, .28);
}

/* Text only. Streamlit's icon glyphs are text in an icon font, so no broad
   [class*=...] selectors here or the icons turn into their own names. */
.stApp, .stApp p, .stApp label, .stApp button, .stApp input, .stApp summary, .stApp small {
  font-family: 'Figtree', ui-sans-serif, system-ui, sans-serif;
}
.stApp [data-testid="stIconMaterial"], .stApp .material-symbols-rounded {
  font-family: 'Material Symbols Rounded' !important;
}
h1, h2, h3, [data-testid="stMetricValue"], [data-testid="stMetricValue"] *, .vs-title {
  font-family: 'Fraunces', Georgia, serif;
  font-variation-settings: 'SOFT' 100, 'opsz' 72;
}

/* the room: a fixed layer behind everything, Streamlit's own containers transparent */
body { background: #3F3048 !important; }
.stApp, [data-testid="stAppViewContainer"] { background: transparent !important; }
[data-testid="stHeader"] { background: transparent; }
.vs-canvas {
  position: fixed; inset: 0; z-index: -1; pointer-events: none;
  background-color: #3F3048;
  background-image: url("app/static/cafe.jpg");
  background-image: image-set(url("app/static/cafe.webp") type("image/webp"),
                              url("app/static/cafe.jpg") type("image/jpeg"));
  background-size: cover; background-position: center center; background-repeat: no-repeat;
}
/* Nothing is painted over the picture. Readability comes from the cream text and
   its shadow alone, so the words sit in the air. */

/* the app itself: no paper, just text in the air. Every layer between the
   canvas and the words is forced transparent so nothing can tint the column. */
[data-testid="stMainBlockContainer"] { padding: 3rem 1.5rem 3rem !important; }
[data-testid="stMain"], [data-testid="stMainBlockContainer"], .stMainBlockContainer, .block-container,
[data-testid="stVerticalBlock"], [data-testid="stVerticalBlockBorderWrapper"], [data-testid="stElementContainer"],
[data-testid="stMarkdown"], [data-testid="stMarkdownContainer"], [data-testid="stHorizontalBlock"],
[data-testid="stColumn"], [data-testid="stFileUploader"], [data-testid="stMetric"], .stApp section {
  background: transparent !important; background-color: transparent !important; box-shadow: none;
}
.stApp, .stApp p, .stApp label, .stApp summary, [data-testid="stMarkdownContainer"] p { color: var(--cream); }
[data-testid="stMarkdownContainer"] p { line-height: 1.6; text-shadow: var(--ink-shadow); }
[data-testid="stMarkdownContainer"] strong { color: var(--lamp); }
[data-testid="stMarkdownContainer"] a { color: var(--lamp); }
h1, h2, h3 { color: var(--cream); text-shadow: var(--ink-shadow); }
@media (max-width: 900px) {
  [data-testid="stMainBlockContainer"] { padding: 1.6rem 1rem 2rem !important; }
  .vs-canvas { background-position: 82% center; }
}

/* hero: the title with a slow equaliser beside it */
.vs-hero { display: flex; align-items: flex-end; gap: 18px; margin: .2rem 0 .2rem; }
.vs-title {
  margin: 0; line-height: 1; font-size: 3.4rem; font-weight: 600;
  letter-spacing: -.02em; color: var(--cream); text-shadow: var(--ink-shadow), 0 0 40px rgba(246, 200, 122, .35);
}
.vs-eq { display: flex; align-items: flex-end; gap: 5px; height: 38px; padding-bottom: 6px; }
.vs-eq i {
  display: block; width: 7px; height: 38px; border-radius: 4px 4px 2px 2px;
  background: linear-gradient(180deg, var(--lamp), var(--caramel));
  box-shadow: 0 0 12px rgba(246, 200, 122, .45);
  transform-origin: bottom; animation: vs-pulse 2.6s ease-in-out infinite alternate;
}
.vs-eq i:nth-child(1) { animation-duration: 2.9s; transform: scaleY(.45); }
.vs-eq i:nth-child(2) { animation-duration: 2.3s; animation-delay: -.8s; transform: scaleY(.8); }
.vs-eq i:nth-child(3) { animation-duration: 3.3s; animation-delay: -1.6s; transform: scaleY(.6); }
.vs-eq i:nth-child(4) { animation-duration: 2.5s; animation-delay: -.4s; transform: scaleY(.95); }
.vs-eq i:nth-child(5) { animation-duration: 3.1s; animation-delay: -2.1s; transform: scaleY(.5); }
@keyframes vs-pulse { from { transform: scaleY(.28); } to { transform: scaleY(1); } }
@media (prefers-reduced-motion: reduce) { .vs-eq i { animation: none; } }

/* the song lands here: a large drop area made of tinted glass */
[data-testid="stFileUploader"] label, [data-testid="stFileUploader"] label p { color: var(--cream); text-shadow: var(--ink-shadow); }
[data-testid="stFileUploaderDropzone"] {
  background: var(--glass); border: 2px dashed rgba(247, 235, 220, .55);
  border-radius: 18px; min-height: 150px; padding: 1.4rem 1.6rem; align-items: center;
  color: var(--cream);
}
[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--lamp); }
[data-testid="stFileUploaderDropzone"] small, [data-testid="stFileUploaderDropzone"] span { font-size: 1rem; color: var(--cream-soft); }
[data-testid="stFileUploaderDropzone"] button {
  font-size: 1.08rem; padding: .75rem 1.5rem; border-radius: 12px;
  background: rgba(247, 235, 220, .14); border: 1.5px solid rgba(247, 235, 220, .6); color: var(--cream);
}
[data-testid="stFileUploaderDropzone"] button:hover { background: rgba(247, 235, 220, .24); border-color: var(--lamp); color: var(--cream); }
[data-testid="stFileUploaderFile"], [data-testid="stFileUploaderFile"] * { color: var(--cream); }

/* buttons: caramel for the one action, glass for the rest */
[data-testid="stBaseButton-primary"] {
  background: linear-gradient(180deg, #D08F42, #AE6C2C); border: 0; color: #FFF7EA;
  font-weight: 600; letter-spacing: .01em; box-shadow: 0 12px 28px -12px rgba(0, 0, 0, .7);
}
.stButton [data-testid="stBaseButton-primary"] { min-height: 3.6rem; font-size: 1.22rem; border-radius: 14px; }
[data-testid="stBaseButton-primary"]:hover { background: linear-gradient(180deg, #DB9B4C, #B97530); color: #FFF7EA; }
[data-testid="stBaseButton-primary"]:disabled { background: rgba(247, 235, 220, .18); color: rgba(247, 235, 220, .55); box-shadow: none; }
[data-testid="stBaseButton-secondary"] {
  background: var(--glass); border: 1px solid var(--glass-edge); color: var(--cream);
}
[data-testid="stBaseButton-secondary"]:hover { border-color: var(--lamp); color: var(--lamp); }

/* status, result expander, form, alerts: tinted glass */
[data-testid="stExpander"] details, [data-testid="stForm"] {
  background: var(--glass); border: 1px solid var(--glass-edge); border-radius: 16px;
}
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary * { color: var(--cream); }
[data-testid="stAlert"] { border-radius: 14px; }
[data-testid="stTextInput"] input { background: rgba(20, 12, 24, .45); color: var(--cream); border-color: var(--glass-edge); }

/* verdict */
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * { color: var(--cream-soft); }
[data-testid="stMetricValue"], [data-testid="stMetricValue"] * { color: var(--cream); font-size: 2.3rem; text-shadow: var(--ink-shadow); }
[data-testid="stProgress"] [role="progressbar"] {
  background: rgba(247, 235, 220, .22); border-radius: 999px; overflow: hidden; height: 7px;
}
[data-testid="stProgress"] [role="progressbar"] > div {
  background: linear-gradient(90deg, var(--caramel), var(--lamp)); border-radius: 999px;
}

/* players and small print */
audio { width: 100%; border-radius: 999px; filter: sepia(.45) saturate(1.05) hue-rotate(-6deg); }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color: var(--cream-soft); text-shadow: var(--ink-shadow); }
[data-testid="stCaptionContainer"] code, [data-testid="stMarkdownContainer"] code {
  background: rgba(247, 235, 220, .14); color: var(--lamp); border-radius: 6px; padding: .1em .35em;
}
"""


def inject_style() -> None:
    st.markdown(f"<style>{THEME_CSS}</style><div class=\"vs-canvas\" aria-hidden=\"true\"></div>",
                unsafe_allow_html=True)


def hero() -> None:
    st.markdown(
        '<div class="vs-hero">'
        '<div class="vs-eq" aria-hidden="true"><i></i><i></i><i></i><i></i><i></i></div>'
        '<h1 class="vs-title">vocalsplit</h1>'
        "</div>",
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------- UI

st.set_page_config(page_title="vocalsplit", page_icon="🎤", layout="centered")
inject_style()
require_key()

hero()
st.markdown(
    "Split a song into **vocals** and **instrumental**, and if the track has no "
    "singing in it, say so instead of inventing a vocal stem out of bleed.\n\n"
    "Demucs (MIT) does the separation; the instrumental check is four measurements "
    "on the separated stem, not a guess about the mixture."
)

upload = st.file_uploader(
    "🎵 Song",
    type=["mp3", "wav", "flac", "ogg", "m4a", "aac", "aiff"],
    help=f"Up to {MAX_UPLOAD_MB} MB and {MAX_MINUTES:.0f} minutes. "
         "Longer tracks exceed this free server's memory.",
)

run = st.button("🎶 Separate", type="primary", disabled=upload is None, use_container_width=True)

if run and upload is not None:
    if upload.size > MAX_UPLOAD_MB * 1024 * 1024:
        st.error(f"File is {upload.size / 1e6:.0f} MB; the limit here is {MAX_UPLOAD_MB} MB.")
        st.stop()

    with st.status("Decoding audio...", expanded=False) as status:
        try:
            mix, sr, tmp = _decode(upload)
        except Exception as e:  # noqa: BLE001
            status.update(label="Could not decode that file", state="error")
            st.error(f"{type(e).__name__}: {e}")
            st.stop()

        minutes = mix.shape[1] / sr / 60
        if minutes > MAX_MINUTES:
            status.update(label="Track too long for this server", state="error")
            st.error(f"Track is {minutes:.1f} min; the limit on this free server is "
                     f"{MAX_MINUTES:.0f} min. Run it locally for longer tracks.")
            st.stop()

        if not _RUN_LOCK.acquire(blocking=False):
            status.update(label="Someone else's song is being separated. Waiting for a free slot...")
            _RUN_LOCK.acquire()
        try:
            result = separate(mix, sr, status)
        except Exception as e:  # noqa: BLE001
            status.update(label="Separation failed", state="error")
            st.error(f"{type(e).__name__}: {e}")
            st.stop()
        finally:
            _RUN_LOCK.release()
            del mix
            try:
                tmp.unlink()
            except OSError:
                pass
        status.update(label=f"Done in {result['seconds']:.0f} s of separation.", state="complete")

    st.session_state["result"] = result
    st.session_state["stem"] = Path(upload.name).stem[:60]

if "result" in st.session_state:
    r = st.session_state["result"]
    stem = st.session_state["stem"]
    show_verdict(r["verdict"], r["metrics"])

    if r["vocals"] is not None:
        left, right = st.columns(2)
        icons = {"vocals": "🎤", "instrumental": "🎸"}
        for col, name, data in ((left, "vocals", r["vocals"]),
                                (right, "instrumental", r["instrumental"])):
            with col:
                st.markdown(f"**{icons[name]} {name}**")
                st.audio(io.BytesIO(data), format="audio/mpeg")
                st.download_button(
                    f"Download {name}.mp3", data=data,
                    file_name=f"{stem} ({name}).mp3", mime="audio/mpeg",
                    use_container_width=True, key=f"dl-{name}",
                )
    else:
        st.info("No files were produced. This track has no vocals to separate, "
                "so writing a \"vocals\" file would just be handing you bleed.")

    with st.expander("🎚️ Raw metrics"):
        st.json(r["metrics"])

st.caption(
    "`instrumental` is computed as `mixture − vocals`, so the two files sum back to "
    "exactly what you uploaded. Separation is not perfect: expect audible bleed on "
    "dense mixes, and backing vocals ride with the lead. This free server runs on "
    "CPU, so a 4-minute song takes a few minutes; the same code runs ~10× faster "
    "on a laptop GPU."
)
