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
                        f"This takes a few minutes here, please keep the tab open.")
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

    st.title("vocalsplit")
    st.write("This is a private preview. Enter the access key you were given.")
    with st.form("gate"):
        given = st.text_input("Access key", type="password")
        submitted = st.form_submit_button("Enter", type="primary")
    if submitted:
        if hmac.compare_digest(given.strip(), key):
            st.session_state["authed"] = True
            st.rerun()
        st.error("That key is not right.")
    st.stop()


# ----------------------------------------------------------------------------- UI

st.set_page_config(page_title="vocalsplit", page_icon="🎤", layout="centered")
require_key()

st.title("vocalsplit")
st.markdown(
    "Split a song into **vocals** and **instrumental**, and if the track has no "
    "singing in it, say so instead of inventing a vocal stem out of bleed.\n\n"
    "Demucs (MIT) does the separation; the instrumental check is four measurements "
    "on the separated stem, not a guess about the mixture."
)

upload = st.file_uploader(
    "Song",
    type=["mp3", "wav", "flac", "ogg", "m4a", "aac", "aiff"],
    help=f"Up to {MAX_UPLOAD_MB} MB and {MAX_MINUTES:.0f} minutes. "
         "Longer tracks exceed this free server's memory.",
)

run = st.button("Separate", type="primary", disabled=upload is None, use_container_width=True)

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
        for col, name, data in ((left, "vocals", r["vocals"]),
                                (right, "instrumental", r["instrumental"])):
            with col:
                st.markdown(f"**{name}**")
                st.audio(io.BytesIO(data), format="audio/mpeg")
                st.download_button(
                    f"Download {name}.mp3", data=data,
                    file_name=f"{stem} ({name}).mp3", mime="audio/mpeg",
                    use_container_width=True, key=f"dl-{name}",
                )
    else:
        st.info("No files were produced. This track has no vocals to separate, "
                "so writing a \"vocals\" file would just be handing you bleed.")

    with st.expander("Raw metrics"):
        st.json(r["metrics"])

st.caption(
    "`instrumental` is computed as `mixture − vocals`, so the two files sum back to "
    "exactly what you uploaded. Separation is not perfect: expect audible bleed on "
    "dense mixes, and backing vocals ride with the lead. This free server runs on "
    "CPU, so a 4-minute song takes a few minutes; the same code runs ~10× faster "
    "on a laptop GPU."
)
