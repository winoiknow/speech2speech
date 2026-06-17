# Copyright 2026 winoiknow (Eric Alborn, Anteon Group)
# Licensed under the Apache License, Version 2.0 (the "License").

"""Speaker-embedding models behind a pluggable interface.

``Embedder`` is the contract the rest of the service codes against; swapping the
model (ECAPA now, pyannote/embedding later for the Phase-4 diarization ecosystem)
is just a different implementation — nothing else changes.

  * EcapaEmbedder — SpeechBrain ECAPA-TDNN (spkrec-ecapa-voxceleb), 192-d,
    non-gated, strong text-independent recognition. The Phase-1 default.
  * StubEmbedder — deterministic hash-based vector, no torch. Lets the service
    run for contract/CI tests without the model.

All embedders return an **L2-normalized** vector, so cosine similarity is a plain
dot product downstream.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger("speaker_id.embedding")


@runtime_checkable
class Embedder(Protocol):
    model_id: str
    dim: int

    def embed(self, audio_f32_16k: np.ndarray) -> np.ndarray: ...

    def warmup(self) -> None: ...


def _l2(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return (v / n).astype(np.float32) if n > 0 else v.astype(np.float32)


class StubEmbedder:
    """Deterministic, torch-free embedder for tests (same audio → same vector)."""

    def __init__(self, dim: int = 192) -> None:
        self.model_id = "stub"
        self.dim = dim

    def embed(self, audio_f32_16k: np.ndarray) -> np.ndarray:
        h = hashlib.sha256(np.ascontiguousarray(audio_f32_16k, dtype=np.float32).tobytes()).digest()
        rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
        return _l2(rng.standard_normal(self.dim).astype(np.float32))

    def warmup(self) -> None:
        return


class EcapaEmbedder:
    """SpeechBrain ECAPA-TDNN speaker encoder (lazy-loaded)."""

    def __init__(
        self,
        source: str = "speechbrain/spkrec-ecapa-voxceleb",
        savedir: str = "/models/ecapa",
        device: str = "cpu",
    ) -> None:
        self.model_id = source
        self.dim = 192
        self._source = source
        self._savedir = savedir
        self._device = device
        self._clf = None

    def _ensure(self) -> None:
        if self._clf is not None:
            return
        try:  # SpeechBrain ≥1.0
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:  # older layout
            from speechbrain.pretrained import EncoderClassifier
        logger.info("Loading ECAPA encoder %s → %s (%s)", self._source, self._savedir, self._device)
        self._clf = EncoderClassifier.from_hparams(
            source=self._source, savedir=self._savedir, run_opts={"device": self._device}
        )

    def embed(self, audio_f32_16k: np.ndarray) -> np.ndarray:
        import torch

        if audio_f32_16k.size == 0:
            raise ValueError("empty audio")
        self._ensure()
        wav = torch.from_numpy(np.ascontiguousarray(audio_f32_16k, dtype=np.float32)).unsqueeze(0)
        with torch.no_grad():
            emb = self._clf.encode_batch(wav).reshape(-1).cpu().numpy().astype(np.float32)
        return _l2(emb)

    def warmup(self) -> None:
        self._ensure()
        # one tiny forward pass so the first real call isn't cold
        try:
            self.embed(np.zeros(self._device and 16000 or 16000, dtype=np.float32) + 1e-4)
        except Exception as e:
            logger.warning("ECAPA warmup pass failed (non-fatal): %s", e)


def make_embedder(model_id: str, savedir: str = "/models/ecapa", device: str = "cpu") -> Embedder:
    if not model_id or model_id == "stub":
        return StubEmbedder()
    # only ECAPA today; pyannote/wespeaker land here as additional branches.
    return EcapaEmbedder(source=model_id, savedir=savedir, device=device)
