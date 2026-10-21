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

# A small café at dusk: lamplight in one corner, oat-coloured paper, espresso
# text, and a quiet equaliser beside the title as if a record is playing low.
# Colours live in .streamlit/config.toml too (widgets read them from there);
# this sheet adds what the theme file cannot: the background, the fonts, the
# hero, and the warm shadows.
_NOISE = (
    "url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='180' "
    "height='180'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' "
    "baseFrequency='0.85' numOctaves='2' stitchTiles='stitch'/%3E%3CfeColorMatrix "
    "values='0 0 0 0 0.32 0 0 0 0 0.19 0 0 0 0 0.08 0 0 0 0.09 0'/%3E%3C/filter%3E"
    "%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E\")"
)

THEME_CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght,SOFT@9..144,400..700,50..100&family=Figtree:wght@400;500;600&display=swap');

:root {{
  --espresso: #2C1B12;
  --walnut: #6B4A36;
  --caramel: #B06E2C;
  --lamp: #F0B45A;
  --oat: #F6EBD9;
  --linen: #E9D5B8;
  --shadow: 0 18px 40px -22px rgba(76, 42, 20, .45);
}}

/* Text only. Streamlit's icon glyphs are text in an icon font, so no broad
   [class*=...] selectors here or the icons turn into their own names. */
.stApp, .stApp p, .stApp label, .stApp button, .stApp input, .stApp summary, .stApp small {{
  font-family: 'Figtree', ui-sans-serif, system-ui, sans-serif;
}}
.stApp [data-testid="stIconMaterial"], .stApp .material-symbols-rounded {{
  font-family: 'Material Symbols Rounded' !important;
}}
h1, h2, h3, [data-testid="stMetricValue"], [data-testid="stMetricValue"] *, .vs-title {{
  font-family: 'Fraunces', Georgia, serif;
  font-variation-settings: 'SOFT' 100, 'opsz' 72;
}}

/* The room is painted on the body (the page canvas), the furniture is a fixed
   layer at z-index -1, and Streamlit's own container is left transparent. That
   order is what keeps every illustration behind the text, never over it. */
body {{
  background:
    {_NOISE} repeat,
    radial-gradient(1100px 650px at 8% -10%, rgba(240,180,90,.62) 0%, rgba(240,180,90,0) 62%),
    radial-gradient(820px 560px at 104% 106%, rgba(176,110,44,.32) 0%, rgba(176,110,44,0) 60%),
    linear-gradient(180deg, #EDDAC0 0%, #E5CDAB 100%) !important;
  /* Streamlit keeps <html> at zero height, so size the layers against the
     viewport or the gradients collapse to nothing. */
  background-attachment: fixed !important;
}}
.stApp {{ color: var(--espresso); background: transparent !important; }}
[data-testid="stAppViewContainer"] {{ background: transparent !important; }}
[data-testid="stHeader"] {{ background: transparent; }}

/* café wall: line-drawn instruments and a coffee, walnut ink, well behind */
.vs-scene {{
  position: fixed; inset: 0; z-index: -1; pointer-events: none; overflow: hidden;
  color: var(--walnut);
}}
.vs-scene > * {{ position: absolute; }}
.vs-scene svg {{ display: block; width: 100%; height: 100%; fill: none; stroke: currentColor;
  stroke-width: 2.2; stroke-linecap: round; stroke-linejoin: round; }}
.vs-guitar {{ left: 2.5vw; bottom: -44px; height: min(58vh, 540px); aspect-ratio: 200 / 540; opacity: .20;
  transform: rotate(-11deg); transform-origin: bottom center; }}
.vs-vinyl {{ right: -96px; top: -84px; width: 330px; aspect-ratio: 1; opacity: .17; }}
.vs-vinyl svg {{ animation: vs-spin 64s linear infinite; }}
.vs-cup {{ right: 4.5vw; bottom: 34px; width: 190px; aspect-ratio: 230 / 210; opacity: .24; }}
.vs-cup .steam {{ opacity: 0; animation: vs-steam 4.6s ease-in-out infinite; }}
.vs-cup .steam:nth-child(2) {{ animation-delay: -1.5s; }}
.vs-cup .steam:nth-child(3) {{ animation-delay: -3s; }}
.vs-phones {{ left: 5vw; top: 14vh; width: 120px; aspect-ratio: 140 / 130; opacity: .17; transform: rotate(8deg); }}
.vs-wave {{ left: 0; right: 0; bottom: 8px; height: 60px; width: 100%; opacity: .13; }}
.vs-note {{ font-family: 'Fraunces', Georgia, serif; font-size: 28px; line-height: 1;
  color: var(--caramel); opacity: 0; animation: vs-drift 18s linear infinite; }}
.vs-note.a {{ left: 11vw; top: 58vh; }}
.vs-note.b {{ left: 17vw; top: 30vh; font-size: 22px; animation-duration: 23s; animation-delay: -9s; }}
.vs-note.c {{ right: 13vw; top: 44vh; font-size: 34px; animation-duration: 20s; animation-delay: -4s; }}
.vs-note.d {{ right: 24vw; top: 72vh; font-size: 20px; animation-duration: 26s; animation-delay: -15s; }}
.vs-note.e {{ left: 29vw; top: 88vh; font-size: 24px; animation-duration: 21s; animation-delay: -12s; }}
.vs-note.f {{ right: 7vw; top: 24vh; font-size: 18px; animation-duration: 25s; animation-delay: -19s; }}
@keyframes vs-spin {{ to {{ transform: rotate(360deg); }} }}
@keyframes vs-steam {{
  0% {{ opacity: 0; transform: translateY(6px); }}
  35% {{ opacity: .9; }}
  100% {{ opacity: 0; transform: translateY(-26px); }}
}}
@keyframes vs-drift {{
  0% {{ opacity: 0; transform: translateY(0) rotate(-6deg); }}
  12% {{ opacity: .42; }}
  85% {{ opacity: .42; }}
  100% {{ opacity: 0; transform: translateY(-160px) rotate(8deg); }}
}}
@media (max-width: 900px) {{
  .vs-guitar, .vs-phones, .vs-note.b, .vs-note.e {{ display: none; }}
  .vs-vinyl {{ width: 210px; right: -70px; top: -60px; }}
  .vs-cup {{ width: 120px; right: 10px; bottom: 14px; }}
}}
@media (prefers-reduced-motion: reduce) {{
  .vs-vinyl svg, .vs-cup .steam, .vs-note {{ animation: none; }}
  .vs-cup .steam {{ opacity: .6; }} .vs-note {{ opacity: .35; }}
}}
[data-testid="stMainBlockContainer"], .block-container {{ padding-top: 3.2rem; }}

/* hero: the title with a slow equaliser beside it */
.vs-hero {{ display: flex; align-items: flex-end; gap: 18px; margin: .4rem 0 .2rem; }}
.vs-title {{
  margin: 0; line-height: 1; font-size: 3.4rem; font-weight: 600;
  letter-spacing: -.02em; color: var(--espresso);
}}
.vs-eq {{ display: flex; align-items: flex-end; gap: 5px; height: 38px; padding-bottom: 6px; }}
.vs-eq i {{
  display: block; width: 7px; height: 38px; border-radius: 4px 4px 2px 2px;
  background: linear-gradient(180deg, var(--lamp), var(--caramel));
  transform-origin: bottom; animation: vs-pulse 2.6s ease-in-out infinite alternate;
}}
.vs-eq i:nth-child(1) {{ animation-duration: 2.9s; transform: scaleY(.45); }}
.vs-eq i:nth-child(2) {{ animation-duration: 2.3s; animation-delay: -.8s; transform: scaleY(.8); }}
.vs-eq i:nth-child(3) {{ animation-duration: 3.3s; animation-delay: -1.6s; transform: scaleY(.6); }}
.vs-eq i:nth-child(4) {{ animation-duration: 2.5s; animation-delay: -.4s; transform: scaleY(.95); }}
.vs-eq i:nth-child(5) {{ animation-duration: 3.1s; animation-delay: -2.1s; transform: scaleY(.5); }}
@keyframes vs-pulse {{ from {{ transform: scaleY(.28); }} to {{ transform: scaleY(1); }} }}
@media (prefers-reduced-motion: reduce) {{ .vs-eq i {{ animation: none; }} }}

[data-testid="stMarkdownContainer"] p {{ color: #3E2A1E; line-height: 1.6; }}
[data-testid="stMarkdownContainer"] strong {{ color: var(--espresso); }}

/* the paper the song lands on */
[data-testid="stFileUploaderDropzone"] {{
  background: var(--oat); border: 1.5px dashed rgba(107,74,54,.45);
  border-radius: 18px; box-shadow: var(--shadow);
}}
[data-testid="stFileUploaderDropzone"]:hover {{ border-color: var(--caramel); }}
[data-testid="stFileUploaderFile"] {{ color: var(--espresso); }}

/* buttons: caramel for the one action, oat for the rest */
[data-testid="stBaseButton-primary"] {{
  background: linear-gradient(180deg, #BD7A33, #A4642A); border: 0; color: #FFF7EA;
  font-weight: 600; letter-spacing: .01em; box-shadow: 0 10px 22px -12px rgba(120,70,30,.6);
}}
[data-testid="stBaseButton-primary"]:hover {{ background: linear-gradient(180deg, #C8853B, #AD6B2D); color: #FFF7EA; }}
[data-testid="stBaseButton-primary"]:disabled {{ background: #D7BE9C; color: #8A6E55; box-shadow: none; }}
[data-testid="stBaseButton-secondary"] {{
  background: var(--oat); border: 1px solid rgba(107,74,54,.35); color: var(--espresso);
}}
[data-testid="stBaseButton-secondary"]:hover {{ border-color: var(--caramel); color: var(--caramel); }}

/* status, result expander, form, alerts: the same warm card */
[data-testid="stExpander"] details, [data-testid="stForm"] {{
  background: var(--oat); border: 1px solid rgba(107,74,54,.22); border-radius: 16px;
  box-shadow: var(--shadow);
}}
[data-testid="stExpander"] summary {{ color: var(--espresso); }}
[data-testid="stAlert"] {{ border-radius: 14px; }}

/* verdict */
[data-testid="stMetricLabel"] {{ color: var(--walnut); }}
[data-testid="stMetricValue"] {{ color: var(--espresso); font-size: 2.3rem; }}
[data-testid="stProgress"] [role="progressbar"] {{
  background: rgba(107,74,54,.16); border-radius: 999px; overflow: hidden; height: 7px;
}}
[data-testid="stProgress"] [role="progressbar"] > div {{
  background: linear-gradient(90deg, var(--caramel), var(--lamp)); border-radius: 999px;
}}

/* players and small print */
audio {{ width: 100%; border-radius: 999px; filter: sepia(.45) saturate(1.05) hue-rotate(-6deg); }}
[data-testid="stCaptionContainer"] {{ color: var(--walnut); }}
[data-testid="stCaptionContainer"] code, [data-testid="stMarkdownContainer"] code {{
  background: rgba(176,110,44,.12); color: #7A4A1C; border-radius: 6px; padding: .1em .35em;
}}
"""


# Line drawings for the wall. Plain paths, one stroke colour, no fills except the
# record label, so they read as sketches on a café wall rather than clip art.
_GUITAR = """
<svg viewBox="0 0 200 540">
  <path d="M86 14h28l4 44H82z"/>
  <circle cx="76" cy="26" r="4"/><circle cx="76" cy="40" r="4"/><circle cx="124" cy="26" r="4"/><circle cx="124" cy="40" r="4"/>
  <path d="M90 58h20v182H90z"/>
  <path d="M90 84h20M90 108h20M90 130h20M90 150h20M90 168h20M90 184h20M90 198h20"/>
  <path d="M100 240C44 240 32 302 62 332 26 362 20 472 100 512c80-40 74-150 38-180 30-30 18-92-38-92z"/>
  <circle cx="100" cy="380" r="30"/>
  <path d="M72 432h56v10H72z"/>
  <path d="M92 58v374M95.4 58v374M98.6 58v374M101.6 58v374M104.8 58v374M108 58v374" stroke-width="1"/>
</svg>"""

_VINYL = """
<svg viewBox="0 0 320 320">
  <circle cx="160" cy="160" r="150"/>
  <circle cx="160" cy="160" r="132" stroke-dasharray="3 7"/>
  <circle cx="160" cy="160" r="114" stroke-dasharray="2 6"/>
  <circle cx="160" cy="160" r="96" stroke-dasharray="3 7"/>
  <circle cx="160" cy="160" r="78" stroke-dasharray="2 6"/>
  <circle cx="160" cy="160" r="50" fill="#F6EBD9"/>
  <path d="M132 150h56M138 164h44M146 178h28" stroke-width="1.6"/>
  <circle cx="160" cy="160" r="5"/>
</svg>"""

_CUP = """
<svg viewBox="0 0 230 210">
  <ellipse cx="112" cy="180" rx="88" ry="13"/>
  <path d="M42 86l12 70c3 12 12 20 26 20h58c14 0 23-8 26-20l12-70z"/>
  <path d="M178 100c34-2 44 14 38 34-5 18-22 26-40 22"/>
  <path d="M70 94c22 6 60 6 84 0" stroke-width="1.6"/>
  <path class="steam" d="M84 70c-8-10 8-18 0-30 8-6 0-14 4-22"/>
  <path class="steam" d="M112 66c-8-10 8-18 0-30 8-6 0-14 4-22"/>
  <path class="steam" d="M140 70c-8-10 8-18 0-30 8-6 0-14 4-22"/>
</svg>"""

_PHONES = """
<svg viewBox="0 0 140 130">
  <path d="M20 86V62C20 34 42 12 70 12s50 22 50 50v24"/>
  <path d="M20 76h18v38H24c-6 0-10-4-10-10V86c0-6 2-10 6-10z"/>
  <path d="M120 76h-18v38h14c6 0 10-4 10-10V86c0-6-2-10-6-10z"/>
</svg>"""

_WAVE = """
<svg viewBox="0 0 1600 60" preserveAspectRatio="none">
  <path d="M0 30c40 0 40-18 80-18s40 36 80 36 40-28 80-28 40 20 80 20 40-34 80-34 40 44 80 44 40-30 80-30 40 14 80 14 40-24 80-24 40 32 80 32 40-20 80-20 40 10 80 10 40-26 80-26 40 36 80 36 40-18 80-18 40 8 80 8 40-30 80-30 40 24 80 24 40-14 80-14 40 20 80 20" stroke-width="1.8"/>
</svg>"""

SCENE_HTML = (
    '<div class="vs-scene" aria-hidden="true">'
    f'<div class="vs-guitar">{_GUITAR}</div>'
    f'<div class="vs-vinyl">{_VINYL}</div>'
    f'<div class="vs-cup">{_CUP}</div>'
    f'<div class="vs-phones">{_PHONES}</div>'
    f'<div class="vs-wave">{_WAVE}</div>'
    '<span class="vs-note a">♪</span><span class="vs-note b">♫</span>'
    '<span class="vs-note c">♩</span><span class="vs-note d">♬</span>'
    '<span class="vs-note e">♪</span><span class="vs-note f">♫</span>'
    "</div>"
)


def inject_style() -> None:
    st.markdown(f"<style>{THEME_CSS}</style>{SCENE_HTML}", unsafe_allow_html=True)


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
