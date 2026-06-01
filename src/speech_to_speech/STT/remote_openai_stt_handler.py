# Copyright 2026 winoiknow (Eric Alborn, Anteon Group)
# Licensed under the Apache License, Version 2.0 (the "License").
# See the LICENSE file in the repository root for the full license text.

from __future__ import annotations

import io
import logging
import struct
from typing import Any, Iterator

import httpx
import numpy as np
from rich.console import Console

from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.pipeline.handler_types import STTIn, STTOut
from speech_to_speech.pipeline.messages import Transcription

logger = logging.getLogger(__name__)
console = Console()

SAMPLE_RATE = 16000


def _pcm_to_wav(pcm_bytes: bytes, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Wrap raw int16 mono PCM in a minimal RIFF/WAV container (44-byte header)."""
    num_channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = len(pcm_bytes)
    riff_size = 36 + data_size

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        riff_size,
        b"WAVE",
        b"fmt ",
        16,              # PCM fmt chunk size
        1,               # PCM format
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size,
    )
    return header + pcm_bytes


class RemoteOpenAISTTHandler(BaseHandler[STTIn, STTOut]):
    """
    Sends VAD-segmented audio to a remote OpenAI-compatible /v1/audio/transcriptions endpoint
    (e.g. faster-whisper-server) and returns the transcript.  No local model is loaded.
    """

    def setup(
        self,
        base_url: str = "http://localhost:8000",
        api_key: str = "sk-unused",
        model: str = "Systran/faster-whisper-large-v3",
        language: str = "en",
        timeout: float = 30.0,
        gen_kwargs: dict | None = None,  # accepted for pipeline compatibility, unused
    ) -> None:
        self.model = model
        self.language = language
        self.endpoint = base_url.rstrip("/") + "/v1/audio/transcriptions"
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.timeout = timeout
        self._client = httpx.Client(timeout=self.timeout)
        logger.info("RemoteOpenAISTTHandler ready → %s (model=%s)", self.endpoint, self.model)

    def process(self, vad_audio: STTIn) -> Iterator[STTOut]:
        # The VAD emits the growing buffer in "progressive" mode (for live partials) and
        # once in "final" mode at speech end. This is a remote HTTP endpoint, so a full
        # transcription per progressive chunk means ~10 calls per utterance — and, worse,
        # each would yield a final Transcription that the LLM treats as a new user turn
        # (one utterance → ~10 replies, the runaway loop). Local incremental engines
        # (parakeet) emit cheap PartialTranscriptions here; over HTTP we just skip
        # progressive and transcribe only the final segment: one call, one turn.
        if getattr(vad_audio, "mode", None) == "progressive":
            return

        # vad_audio.audio is float32 at 16 kHz; convert to int16 then wrap in WAV container
        audio_float32: np.ndarray = vad_audio.audio
        audio_int16 = (audio_float32 * 32768).clip(-32768, 32767).astype(np.int16)
        wav_bytes = _pcm_to_wav(audio_int16.tobytes())

        files: dict[str, Any] = {
            "file": ("audio.wav", io.BytesIO(wav_bytes), "audio/wav"),
            "model": (None, self.model),
            "response_format": (None, "verbose_json"),
        }
        if self.language and self.language != "auto":
            files["language"] = (None, self.language)

        logger.debug("RemoteOpenAISTT: posting %d bytes to %s", len(wav_bytes), self.endpoint)
        try:
            response = self._client.post(self.endpoint, headers=self.headers, files=files)
            response.raise_for_status()
            result = response.json()
            pred_text = result.get("text", "").strip()
            language_code = result.get("language") or None
        except httpx.HTTPError as exc:
            logger.error("RemoteOpenAISTT request failed: %s", exc)
            return

        if not pred_text:
            logger.debug("RemoteOpenAISTT: empty transcript, skipping")
            return

        console.print(f"[yellow]USER: {pred_text}")
        yield Transcription(text=pred_text, language_code=language_code)

    def cleanup(self) -> None:
        self._client.close()
