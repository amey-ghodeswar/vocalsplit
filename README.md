---
title: vocalsplit
emoji: 🎤
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 6.25.0
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
short_description: Split vocals from instruments, and know when not to
---

# vocalsplit

Separates a song into **vocals** and **instrumental** — and, when handed a track
that has no singing in it (electronic, post-rock, a film cue, a backing track),
recognises that and refuses to manufacture a vocal stem out of bleed.

Everything here is open source: [Demucs](https://github.com/facebookresearch/demucs)
(MIT) for separation, [librosa](https://librosa.org) (ISC) for the analysis,
PyTorch, soundfile/libsndfile.

## Why the second half is the harder half

A separator is a function, not a detector. Feed `htdemucs` a pure techno track and
it will still hand you four stems, one of them labelled `vocals`. That stem is not
empty — it holds the lead synth's upper partials, reverb tails, hat wash, whatever
the model found most voice-shaped. Naively saving it produces a confident-looking
file full of garbage.

So the detection does not run on the mixture. It runs on the separated vocal stem,
and asks four questions that bleed and singing answer differently:

| axis | metric | singing | bleed |
|---|---|---|---|
| **level** | 95th pct. of per-frame stem-vs-mix level | −6 … −18 dB | below −28 dB |
| **activity** | fraction of playing frames that are vocal-active | 0.25 … 0.8 | < 0.05 |
| **continuity** | longest contiguous vocal run | multi-second phrases | scattered dust |
| **voicedness** | pYIN voiced ratio on the loudest active regions | > 0.5 | < 0.2 |

Each becomes a soft vote through a sigmoid; the weighted mean is `p_vocal`.
Above 0.6 → `vocal`, below 0.4 → `instrumental`, in between → `uncertain`
(reported honestly, and treated as vocal so you don't silently lose a stem).
Two hard overrides catch the unambiguous cases before the vote runs at all.

Everything is measured **relative to the mixture per frame**, not in absolute dBFS,
so a quiet master and a loud one are judged the same way.

### The probe

Full separation of a 4-minute track is the expensive step. By default the tool first
separates **6 × 10 s windows spread across the timeline** and scores those. If that
excerpt is confidently instrumental (`p_vocal ≤ 0.20`) it stops there — roughly 4×
faster than separating the whole thing to learn the same fact. The probe can only
ever *skip* work; it is never allowed to assert that a track has vocals, so a false
"yes" costs nothing but time. Disable it with `--no-probe`.

The tradeoff to know about: a 6-minute instrumental with one 8-bar vocal at 3:20 can
slip through the probe windows. `--no-probe` removes that risk at 4× the compute.

## Quick start

**First time on this machine?** Install once — one command, ~10 minutes, mostly
downloading torch:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_env.ps1     # Windows
```
```bash
bash setup_env.sh                                            # macOS / Linux
```

You need Python 3.9+ or Anaconda already present, an internet connection, and about
8 GB of disk. The script figures out the rest — see [Install](#install) for what it
does and how to steer it.

Then, every time after that: `conda activate` silently fails on a machine where
conda was never initialised for your shell, so there is a launcher that sidesteps
it entirely:

```powershell
cd D:\claude\vocal-splitter

.\run.ps1                                   # list what it can do
.\run.ps1 ui                                # browser UI at http://127.0.0.1:7860
.\run.ps1 lab                               # JupyterLab on the notebook
.\run.ps1 test                              # test suite
.\run.ps1 detect "C:\Music\song.mp3"        # classify only, write nothing
.\run.ps1 split  "C:\Music\song.mp3" -o "C:\Music\split"
.\run.ps1 split  "C:\Music" -o "C:\Music\split" --model htdemucs_ft
```

It locates the environment itself, or honours `$env:VOCALSPLIT_PYTHON`.

If you would rather use conda the normal way: run `conda init powershell` once (it
edits your PowerShell profile), reopen the terminal, and `conda activate vocalsplit`
will work from then on.

## Install

```powershell
conda create -y -n vocalsplit python=3.11
conda activate vocalsplit
# CUDA build -- Windows wheels on PyPI are CPU-only. Skip the index-url if no NVIDIA GPU.
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -e .
```

or just `powershell -ExecutionPolicy Bypass -File .\setup_env.ps1`.

`ffmpeg` is optional — libsndfile handles wav/flac/ogg/mp3 directly, and ffmpeg is
only used as a fallback for m4a/aac/wma.

## Use

```bash
# classify only, no audio written
vocalsplit detect "song.mp3"
vocalsplit detect ./library --json > verdicts.json

# classify, then split when there are vocals
vocalsplit separate "song.mp3" -o output
vocalsplit separate ./album -o output --stems      # also drums/bass/other
vocalsplit separate "techno.wav" --force           # split anyway, I know what I'm doing
vocalsplit separate "song.mp3" --format mp3 --bitrate 320
```

Output per track lands in `output/<track name>/`:

```
vocals.wav          only when the track was judged vocal
instrumental.wav
report.json         label, p_vocal, every metric, per-axis votes, reasons
```

`instrumental.wav` is computed as **mixture − vocals**, not as the sum of the other
stems. The stems don't sum back to the input exactly, and this way anything the
model failed to attribute stays in the instrumental instead of disappearing.

### Quality knobs

| flag | effect |
|---|---|
| `-m htdemucs_ft` | per-source fine-tuned model, ~4× slower, ~0.5 dB better |
| `--shifts 5` | test-time augmentation, meaningfully cleaner, N× slower |
| `--overlap 0.5` | more chunk overlap, fewer seam artifacts |
| `--segment 5` | shorter chunks if 6 GB of VRAM isn't enough |
| `--no-probe` | analyse the whole track instead of excerpts |

## Notebook

`notebooks/separate_song.ipynb` — set `SONG`, run all cells, get two mp3s plus a plot
of exactly where the detector heard vocals. Pick the **Python (vocalsplit)** kernel;
any other kernel fails on the first import.

```powershell
conda activate vocalsplit
python -m ipykernel install --user --name vocalsplit --display-name "Python (vocalsplit)"
jupyter lab
```

## Web UI

Two front-ends, same behaviour: upload a song, get a verdict and two players. On an
instrumental both produce no files at all and say why.

`app.py` (Gradio) is the one to run on your own machine; it uses the GPU if there
is one and lets you pick the model and quality.

```powershell
conda activate vocalsplit
python app.py              # http://127.0.0.1:7860
python app.py --share      # plus a temporary public link, valid 72 h
```

`streamlit_app.py` (Streamlit) is the one that runs on the free public website, see
[Deploy](#deploy). It is CPU-only, fixed to `htdemucs`, caps uploads at 6 minutes
and runs one separation at a time, because the free host has ~2.7 GB of RAM.

```powershell
pip install -e ".[streamlit]"
streamlit run streamlit_app.py     # http://localhost:8501
```

## Deploy

Three paths. The first is the only one that is free without a card on file.

### Streamlit Community Cloud — recommended, free

Free with a GitHub login, no payment method, one permanent URL. The limits are
~2.7 GB RAM and up to 2 shared vCPUs, which is why `streamlit_app.py` exists as a
separate, leaner front-end: a 4-minute track peaks near 2 GB and takes roughly 4 to
10 minutes there. The app sleeps after a period of no visitors and wakes on the next
one. Measured before deploying: 4 min of audio, 2 threads, 191 s, peak RSS 1.96 GB.

```bash
# 1. put the repo on GitHub (public repo; Community Cloud reads it from there)
git remote add origin https://github.com/<you>/vocalsplit.git
git push -u origin main --tags

# 2. share.streamlit.io -> sign in with GitHub -> New app
#      repository: <you>/vocalsplit   branch: main   main file: streamlit_app.py
#      Advanced settings -> Python version 3.11
#    Deploy. First build takes several minutes (torch); later ones are quick.
```

Four files make this work:

- **`requirements.txt`** starts with `--extra-index-url .../whl/cpu`, so pip installs
  the 200 MB CPU torch instead of the 2.5 GB CUDA one. Without that line the build
  still succeeds but burns memory for nothing.
- **`packages.txt`** lists `ffmpeg`, the apt package, so m4a/aac uploads decode.
- **`.streamlit/config.toml`** sets the 60 MB upload cap and the theme.
- **`streamlit_app.py`** holds a global lock so two visitors cannot run Demucs at
  once; the second one waits, which is better than the container being killed.

Python 3.11 matters: `numpy<2` has no wheels past 3.12 and `demucs 4.0.1` needs it.

### Hugging Face Spaces — paid since 2026

This repo is still shaped like a Space (the YAML header at the top of this README is
the Space config, `app.py` the entry point), but Gradio and Docker Spaces now require
a PRO subscription (US$9/month at the time of writing). Static Spaces stay free and
cannot run the model. If you do pay, PRO also unlocks ZeroGPU, which would make a
song take ~30 s instead of ~4 min; that needs a `@spaces.GPU` decorator in `app.py`.

```bash
# 1. create an empty Space at huggingface.co/new-space
#      SDK: Gradio        hardware: CPU basic (free) or a T4 tier
# 2. push this repo to it
git remote add space https://huggingface.co/spaces/<you>/vocalsplit
git push space main
```

Three things that make this work, each of which fails quietly if you change it:

- **`sdk_version` is pinned to 6.25.0** in the header, the version this was verified
  against locally, so the Space build is reproducible. `app.py` still detects at import
  whether `theme`/`css` belong on `Blocks` (Gradio 5) or on `launch()` (Gradio 6), so
  bumping the pin later is safe. `python_version` is pinned to 3.11 for the same reason.
- **`gradio` is deliberately absent** from `requirements.txt` for the same reason —
  listing it there fights the SDK version Spaces installs. Locally you get it from
  `pip install -e ".[web]"`.
- **`app.py` binds `0.0.0.0` when `SPACE_ID` is set.** Binding localhost builds
  cleanly and then serves nothing, which is an annoying half hour to debug.

Spaces installs the root `requirements.txt` automatically; on Linux the PyPI torch
wheels bundle CUDA, so the same file works on CPU and GPU tiers.

On CPU basic a 4-minute track takes ~4 minutes rather than ~30 seconds, which is why
`MAX_MINUTES` in `app.py` caps CPU uploads at 6 minutes. Fine for a demo; pick a GPU
tier if you expect other people to use it.

### Modal + Vercel — if you want an API

Modal gives US$30/month of free compute but requires a payment method on file.
Vercel cannot run the separation: no GPU, a 250 MB unzipped bundle cap (torch alone
is ~2.5 GB), and a 60 s function timeout on Hobby. So Modal runs the GPU work and
Vercel serves a static page that talks to it. Audio goes browser → Modal directly and
never passes through Vercel.

```bash
pip install modal
modal setup                        # one-time browser auth
modal deploy deploy/modal_app.py   # prints your endpoint URL

cd web && vercel --prod            # static, no build step
```

Paste the Modal URL into the field on the deployed page (stored in localStorage).
Then narrow `allow_origins` in `deploy/modal_app.py` from `"*"` to your Vercel domain
— otherwise anyone who finds the endpoint can spend your GPU budget.

Uploads spawn a background call and the browser polls, because a 4-minute track takes
longer than a web request should block for. Results live in a Modal Volume and a
scheduled job deletes them after 6 hours.

**`deploy/modal_app.py` and `web/index.html` have never been run against a live
deployment** — both need accounts that are yours to create. Their syntax is checked
and everything they import from `vocalsplit` is tested, but treat the first deploy as
debugging, and expect Modal's API to have drifted if you read this much later.

## Sharing it with someone else

Depends entirely on how much you want them to have to do.

### They just want their song split — send them a link

Deploy to Streamlit Community Cloud (above) and send the URL. They open it in a
browser, drag a song in, wait a few minutes, download two files. Nothing to install,
nothing to explain, works on a phone. This is the right answer for almost everyone;
the only cost is patience, since the free server is CPU-only.

For a one-off with GPU speed, `python app.py --share` on this laptop prints a public
link that works for 72 hours while the laptop is on.

### They want to run it themselves — send them the repo

A GitHub URL, or a zip of the working tree (`git archive --format=zip -o vocalsplit.zip HEAD`
gives you a clean one — tracked files only, no environments or generated audio).

Then they run **one command**:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_env.ps1     # Windows
```
```bash
bash setup_env.sh                                            # macOS / Linux
```

That finds conda even when it is not on PATH, falls back to a plain venv with
`-NoConda` / `--venv`, detects whether there is an NVIDIA GPU and picks the matching
torch build, installs everything, registers the Jupyter kernel, and verifies the
result imports before claiming success. Then:

```powershell
.\run.ps1 ui                     # Windows, no conda activate needed
```
```bash
conda activate vocalsplit && python app.py     # macOS / Linux
```

What they need first: **Python 3.9+**, about **8 GB of disk** (torch is most of it),
and patience on the first run — pip pulls ~2.5 GB for the CUDA build, and Demucs
downloads its 80 MB checkpoint the first time you separate anything. No ffmpeg
required unless they feed it m4a or wma.

Without an NVIDIA GPU it still works, roughly 10× slower: a 4-minute song takes
about 4 minutes instead of 30 seconds. Apple Silicon uses Metal automatically.

### They just want the audio — send the mp3s

`vocals.mp3` and `instrumental.mp3` are ordinary files. Nothing about them needs
this project to play.

## Tuning the detector to your library

The default thresholds are set from htdemucs behaviour and lean conservative on the
instrumental side. If your material is unusual — heavy vocoder, spoken word over
beats, sample-chopped house — point the calibrator at two labelled folders:

```bash
vocalsplit calibrate --vocal ./has_vocals --instrumental ./instrumentals -o thresholds.json
vocalsplit separate ./library --thresholds thresholds.json -o output
```

It prints each metric's distribution per class, the split that best separates them,
and which files the default and tuned thresholds get wrong.

## Python API

```python
from vocalsplit import Config, DemucsSeparator, analyze

sep = DemucsSeparator("htdemucs")             # load the model once for a batch
r = analyze("song.mp3", Config(), separator=sep)
print(r["label"], r["p_vocal"], r["metrics"])
if r["label"] == "vocal":
    vocals = r["stems"]["vocals"]             # numpy [2, n] at 44100 Hz
```

## Verified on this machine

`RTX 4050 Laptop (6 GB) / torch 2.6.0+cu124 / demucs 4.0.1 / Python 3.11 / gradio 6.25`

- `pytest tests/` — 7 passed.
- **CLI**, both classes, synthetic clips: `instrumental.wav` → **instrumental**
  (`p_vocal=0.02`, decided from the probe, full separation skipped);
  `vocal.wav` → **vocal** (`p_vocal=0.97`). Against ground truth the split scores
  **23.3 dB SDR on vocals** (13.2 dB if you did nothing) and **10.1 dB on the
  instrumental** (−13.2 dB baseline); `vocals + instrumental` reconstructs the input
  to 1.2e−7, i.e. exactly at 24-bit.
- **mp3 output** — 320 kbps, decodes back at 44.1 kHz stereo, correct duration.
- **Notebook** — executed end to end via nbconvert, 0 cell errors, plot renders.
- **Gradio app** — instrumental input returns `None` for both audio outputs;
  vocal input returns two mp3s; server answers HTTP 200.
- **Real songs**, two Hindi film tracks (4:36 and 7:53): both correctly **vocal** at
  `p_vocal=0.99`, 45 s each on the GPU.

Read the synthetic numbers as proof the plumbing works, **not** as a benchmark.
htdemucs scores ~9 dB on MUSDB18 and that is what to expect on real music. The
detector has still only been tested against one instrumental — a synthetic one.
That is the weakest claim here.

Both real songs peaked **above** full scale after separation (+1.01 and +1.40 dBFS),
which is why every stem is written through one shared gain. Without it those mp3s
would have clipped silently.

## Honest limits

- **"Completely" is not on the menu.** htdemucs scores ~9 dB SDR on vocals. Expect
  audible bleed on dense mixes, and a watery quality on isolated vocals. If you need
  better, the current open-source ceiling is a BS-Roformer checkpoint run through
  [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator);
  swapping `separator.py` for it is a contained change — the detector doesn't care
  which model produced the stem.
- **Backing vocals ride with the lead.** htdemucs has one vocal source. Separating
  lead from harmonies needs a different model.
- **Genuinely ambiguous cases exist.** A heavily vocoded lead or a sustained "aah"
  pad is voice-shaped by every measure here. That's what `uncertain` is for — the
  tool tells you rather than guessing quietly.
- **A cappella input** inverts the assumption: the *instrumental* stem is the bleed.
  Detecting that is the mirror image of this logic and isn't implemented.
