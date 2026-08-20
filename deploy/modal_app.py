"""GPU inference on Modal.

    pip install modal
    modal setup                       # one-time browser auth
    modal serve deploy/modal_app.py   # dev, hot-reloads, temporary URL
    modal deploy deploy/modal_app.py  # permanent URL -> paste into the web UI

Why Modal and not Vercel: Vercel functions have no GPU, cap the unzipped bundle
at 250 MB (torch alone is ~2.5 GB) and time out at 60 s on Hobby. Vercel serves
the static UI in ../web; every byte of audio is processed here.

Separation of a 4-minute track is ~30 s on a T4, which is longer than a web
request should block for, so uploads spawn a background call and the browser
polls. Results land in a Volume and are served back as ordinary file downloads.

NOTE: this file has not been run -- it needs a Modal account, which is yours to
create. Modal's API does move; if something below is renamed, the current docs
are at modal.com/docs. Everything it imports from `vocalsplit` is tested.
"""
from __future__ import annotations

import modal

app = modal.App("vocalsplit")
vol = modal.Volume.from_name("vocalsplit-results", create_if_missing=True)
RESULTS = "/results"

MODEL = "htdemucs"
BITRATE = 192          # web delivery; the CLI defaults to 320
RESULT_TTL_H = 6       # how long finished jobs stay downloadable


def _prefetch_weights() -> None:
    """Bake the ~80 MB checkpoint into the image so cold starts don't download it."""
    from demucs.pretrained import get_model

    get_model(MODEL)


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.6.0", "torchaudio==2.6.0",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "numpy<2",            # demucs 4.0.1 still wants the numpy 1.x ABI
        "demucs==4.0.1",
        "librosa>=0.10.1",
        "soundfile>=0.12.1",
        "soxr>=0.3.7",
        "lameenc>=1.7",
        "fastapi[standard]",
    )
    .run_function(_prefetch_weights)
    .add_local_python_source("vocalsplit")
)


@app.cls(
    image=image,
    gpu="T4",              # htdemucs runs ~10x realtime here; A10G if you want faster
    volumes={RESULTS: vol},
    scaledown_window=300,  # keep the model warm between requests
    timeout=1800,
)
class Separator:
    @modal.enter()
    def load(self) -> None:
        from vocalsplit import DemucsSeparator

        self.sep = DemucsSeparator(MODEL, device="cuda")

    @modal.method()
    def run(self, job_id: str, data: bytes, filename: str) -> dict:
        import tempfile
        from pathlib import Path

        import numpy as np

        from vocalsplit.audio import load_audio, save_audio
        from vocalsplit.decide import Thresholds, classify
        from vocalsplit.features import stem_metrics

        suffix = Path(filename).suffix or ".mp3"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as fh:
            fh.write(data)
            tmp = fh.name

        mix, sr = load_audio(tmp, sr=self.sep.samplerate)
        voc = self.sep.separate(mix, sr)["vocals"]
        metrics = stem_metrics(mix, voc, sr)
        verdict = classify(metrics, Thresholds())

        out = Path(RESULTS) / job_id
        out.mkdir(parents=True, exist_ok=True)
        written: list[str] = []

        if verdict["label"] != "instrumental":
            n = min(mix.shape[1], voc.shape[1])
            voc, inst = voc[:, :n], mix[:, :n] - voc[:, :n]
            peak = max(float(np.abs(a).max()) for a in (voc, inst))
            gain = min(1.0, 0.999 / peak) if peak > 0 else 1.0
            save_audio(out / "vocals.mp3", voc * gain, sr, bitrate=BITRATE)
            save_audio(out / "instrumental.mp3", inst * gain, sr, bitrate=BITRATE)
            written = ["vocals.mp3", "instrumental.mp3"]

        vol.commit()
        return {
            "job_id": job_id,
            "filename": filename,
            "duration_s": metrics["duration_s"],
            "metrics": metrics,
            "written": written,
            **verdict,
        }


@app.function(image=image, volumes={RESULTS: vol}, timeout=600)
@modal.asgi_app()
def web():
    import time
    import uuid
    from pathlib import Path

    from fastapi import FastAPI, File, HTTPException, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse

    api = FastAPI(title="vocalsplit")

    # The browser calls this straight from the Vercel origin, so the audio never
    # passes through Vercel at all. Lock allow_origins down to your own domain
    # once you know it -- "*" here means anyone can spend your GPU budget.
    api.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @api.post("/jobs")
    async def create_job(file: UploadFile = File(...)):
        data = await file.read()
        if not data:
            raise HTTPException(400, "empty upload")
        if len(data) > 60 * 1024 * 1024:
            raise HTTPException(413, "file larger than 60 MB")
        job_id = uuid.uuid4().hex[:16]
        call = Separator().run.spawn(job_id, data, file.filename or "upload.mp3")
        return {"job_id": job_id, "call_id": call.object_id}

    @api.get("/jobs/{call_id}")
    def poll(call_id: str):
        call = modal.FunctionCall.from_id(call_id)
        try:
            return {"status": "done", "result": call.get(timeout=0)}
        except TimeoutError:
            return {"status": "running"}

    @api.get("/files/{job_id}/{name}")
    def download(job_id: str, name: str):
        if name not in ("vocals.mp3", "instrumental.mp3"):
            raise HTTPException(404, "no such stem")
        vol.reload()
        path = Path(RESULTS) / job_id / name
        if not path.exists():
            raise HTTPException(404, "expired or never produced")
        return FileResponse(path, media_type="audio/mpeg", filename=f"{job_id}-{name}")

    @api.get("/health")
    def health():
        return {"ok": True, "model": MODEL}

    return api


@app.function(image=image, volumes={RESULTS: vol}, schedule=modal.Period(hours=6))
def cleanup() -> None:
    """Results are throwaway; without this the Volume grows without bound."""
    import shutil
    import time
    from pathlib import Path

    cutoff = time.time() - RESULT_TTL_H * 3600
    for d in Path(RESULTS).iterdir():
        if d.is_dir() and d.stat().st_mtime < cutoff:
            shutil.rmtree(d, ignore_errors=True)
    vol.commit()
