"""Demucs wrapper. Loads a pretrained model once and applies it to numpy audio."""
from __future__ import annotations

import numpy as np
import torch

# Models worth knowing about, roughly cheapest -> best.
#   htdemucs      : Hybrid Transformer Demucs v4, the sane default
#   htdemucs_ft   : same, fine-tuned per source. ~4x slower, ~0.5 dB better
#   htdemucs_6s   : adds guitar + piano stems (vocals quality unchanged)
#   mdx_extra     : Demucs v3 MDX winner, different artifact character
KNOWN_MODELS = ("htdemucs", "htdemucs_ft", "htdemucs_6s", "mdx_extra", "mdx_extra_q")


def pick_device(requested: str | None = None) -> str:
    if requested and requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


class DemucsSeparator:
    """Thin, reusable wrapper around demucs.apply.apply_model."""

    def __init__(
        self,
        model_name: str = "htdemucs",
        device: str | None = None,
        shifts: int = 0,
        overlap: float = 0.25,
        segment: float | None = None,
        progress: bool = False,
    ) -> None:
        from demucs.pretrained import get_model

        self.model_name = model_name
        self.device = pick_device(device)
        self.shifts = shifts
        self.overlap = overlap
        self.segment = segment
        self.progress = progress

        self.model = get_model(model_name)
        self.model.eval()
        self.sources: list[str] = list(self.model.sources)
        self.samplerate: int = int(self.model.samplerate)
        self.audio_channels: int = int(self.model.audio_channels)

    def _apply(self, wav: np.ndarray, device: str) -> np.ndarray:
        from demucs.apply import apply_model

        tensor = torch.from_numpy(np.ascontiguousarray(wav))
        # Demucs' own normalisation (see demucs.separate): standardise on the
        # mono reference so quiet and loud masters behave identically.
        ref = tensor.mean(0)
        mean, std = ref.mean(), ref.std().clamp(min=1e-8)
        tensor = (tensor - mean) / std

        with torch.no_grad():
            out = apply_model(
                self.model,
                tensor[None],
                device=device,
                shifts=self.shifts,
                split=True,
                overlap=self.overlap,
                segment=self.segment,
                progress=self.progress,
            )[0]

        # In place: `out * std + mean` followed by astype() makes three full copies
        # of the [sources, 2, n] output, ~1 GB transient on a 4-minute track. On a
        # 2.7 GB host that is the difference between fitting and being killed.
        out = out.cpu()
        out.mul_(std.cpu()).add_(mean.cpu())
        return np.ascontiguousarray(out.numpy(), dtype=np.float32)

    def separate(self, wav: np.ndarray, sr: int) -> dict[str, np.ndarray]:
        """wav: [2, n] float32 at `sr`. Returns {source_name: [2, n]}."""
        if sr != self.samplerate:
            raise ValueError(f"{self.model_name} expects {self.samplerate} Hz, got {sr}")
        if wav.shape[0] != self.audio_channels:
            raise ValueError(f"{self.model_name} expects {self.audio_channels} channels")

        try:
            stems = self._apply(wav, self.device)
        except torch.cuda.OutOfMemoryError:
            # 6 GB cards can trip on long segments. Halve the window, then give up to CPU.
            torch.cuda.empty_cache()
            old = self.segment
            self.segment = (old or 7.8) / 2
            try:
                stems = self._apply(wav, self.device)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                self.segment = old
                stems = self._apply(wav, "cpu")

        return {name: stems[i] for i, name in enumerate(self.sources)}
