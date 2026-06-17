# Copyright 2026 winoiknow (Eric Alborn, Anteon Group)
# Licensed under the Apache License, Version 2.0 (the "License").

"""Speaker diarization behind a pluggable interface (Phase 4, Tier 2).

``Diarizer`` is the contract /v1/diarize codes against; swapping the engine is
just a different implementation — nothing else changes:

  * StubDiarizer      — torch-free; returns the whole clip as ONE span. Lets the
                        /v1/diarize contract + the s2s corrective-event path land
                        and be tested with no heavy deps (degenerate but valid:
                        one span == the Tier-1 recognition result, in segment shape).
  * PyannoteDiarizer  — slot for `pyannote/speaker-diarization-community-1`
                        (gated weights, HF_TOKEN). Splits mixed audio into
                        per-speaker spans. Wired in a follow-up step.

A diarizer only finds *where* each speaker talks (spans + a diarizer-local label
like SPEAKER_00). It does NOT decide identity — /v1/diarize embeds each span and
runs it through the same recognition store/decision as /v1/identify, then clusters
the *unenrolled* spans into ephemeral per-call labels (see cluster.py).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger("speaker_id.diarization")

TARGET_SR = 16000


@dataclass
class Span:
    """A contiguous region attributed to one diarizer-local speaker."""

    start: float  # seconds from clip start
    end: float  # seconds from clip start
    speaker: str  # diarizer-local label (e.g. "SPEAKER_00"); NOT an identity


@runtime_checkable
class Diarizer(Protocol):
    model_id: str

    def diarize(self, audio_f32_16k: np.ndarray) -> list[Span]: ...

    def warmup(self) -> None: ...


class StubDiarizer:
    """Torch-free no-split diarizer: the whole clip is one span.

    Keeps /v1/diarize honest end-to-end without pyannote — the single span is
    embedded + identified exactly like /v1/identify, so the segment-shaped
    response and the s2s corrective path can be exercised in CI.
    """

    def __init__(self) -> None:
        self.model_id = "stub"

    def diarize(self, audio_f32_16k: np.ndarray) -> list[Span]:
        dur = float(len(audio_f32_16k)) / TARGET_SR if audio_f32_16k.size else 0.0
        return [Span(start=0.0, end=dur, speaker="SPEAKER_00")]

    def warmup(self) -> None:
        return


class PyannoteDiarizer:
    """pyannote community-1 diarization (lazy-loaded, gated weights).

    Stubbed slot for the stub-first phase — instantiating is fine, but diarize()
    raises until the real pipeline is wired (needs HF_TOKEN + the gated weights).
    make_diarizer() only returns this when explicitly selected, so the default
    stub path never trips it.
    """

    def __init__(
        self,
        source: str = "pyannote/speaker-diarization-community-1",
        device: str = "cpu",
        hf_token: str | None = None,
    ) -> None:
        self.model_id = source
        self._source = source
        self._device = device
        self._hf_token = hf_token
        self._pipeline = None

    def _ensure(self) -> None:
        if self._pipeline is not None:
            return
        import torch
        from pyannote.audio import Pipeline

        logger.info("Loading pyannote diarizer %s (%s)", self._source, self._device)
        # The token kwarg was renamed across pyannote.audio / huggingface_hub
        # versions (use_auth_token → token). Try the modern name, fall back.
        try:
            self._pipeline = Pipeline.from_pretrained(self._source, token=self._hf_token)
        except TypeError:
            self._pipeline = Pipeline.from_pretrained(self._source, use_auth_token=self._hf_token)
        if self._pipeline is None:
            # from_pretrained returns None (not raises) on gated-access/auth failure.
            raise RuntimeError(
                f"pyannote returned no pipeline for {self._source} — check HF_TOKEN and "
                "that you've accepted the model's gated-access terms on huggingface.co"
            )
        self._pipeline.to(torch.device(self._device))

    @staticmethod
    def _to_annotation(result):
        """Extract a pyannote Annotation (has .itertracks) from the pipeline output.

        Classic pipelines (3.1) return the Annotation directly; community-1
        (pyannote.audio 4.x) wraps it in a ``DiarizeOutput`` whose Annotation lives
        on one of these attributes. Tolerate both so a pyannote version bump can't
        silently empty the result.
        """
        if hasattr(result, "itertracks"):
            return result
        for attr in ("speaker_diarization", "diarization", "annotation"):
            ann = getattr(result, attr, None)
            if ann is not None and hasattr(ann, "itertracks"):
                return ann
        raise AttributeError(
            f"unrecognized diarization output {type(result).__name__}; "
            f"attrs={[a for a in dir(result) if not a.startswith('_')]}"
        )

    def diarize(self, audio_f32_16k: np.ndarray) -> list[Span]:
        import torch

        self._ensure()
        wav = torch.from_numpy(np.ascontiguousarray(audio_f32_16k, dtype=np.float32)).unsqueeze(0)
        result = self._pipeline({"waveform": wav, "sample_rate": TARGET_SR})
        ann = self._to_annotation(result)
        spans = [
            Span(start=float(seg.start), end=float(seg.end), speaker=str(label))
            for seg, _, label in ann.itertracks(yield_label=True)
        ]
        spans.sort(key=lambda s: s.start)
        return spans

    def warmup(self) -> None:
        try:
            self._ensure()
        except Exception as e:  # don't block startup on a slow/gated load
            logger.warning("pyannote warmup failed (non-fatal): %s", e)


def make_diarizer(model_id: str, device: str = "cpu", hf_token: str | None = None) -> Diarizer:
    if not model_id or model_id == "stub":
        return StubDiarizer()
    return PyannoteDiarizer(source=model_id, device=device, hf_token=hf_token)
