# Copyright 2026 winoiknow (Eric Alborn, Anteon Group)
# Licensed under the Apache License, Version 2.0 (the "License").
# See the LICENSE file in the repository root for the full license text.

from __future__ import annotations

import io
import logging
from typing import Iterator

import httpx
import numpy as np
from rich.console import Console

from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.pipeline.handler_types import STTIn, STTOut
from speech_to_speech.pipeline.messages import Transcription

logger = logging.getLogger(__name__)
console = Console()

SAMPLE_RATE = 16000


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
        # single shared client; closed in cleanup()
        self._client = httpx.Client(timeout=self.timeout)
        logger.info("RemoteOpenAISTTHandler ready → %s (model=%s)", self.endpoint, self.model)

    def process(self, vad_audio: STTIn) -> Iterator[STTOut]:
        # vad_audio.audio is float32 at 16 kHz; convert to int16 PCM for multipart upload
        audio_float32: np.ndarray = vad_audio.audio
        audio_int16 = (audio_float32 * 32768).clip(-32768, 32767).astype(np.int16)
        pcm_bytes = audio_int16.tobytes()

        files = {
            "file": ("audio.pcm", io.BytesIO(pcm_bytes), "application/octet-stream"),
            "model": (None, self.model),
        }
        if self.language and self.language != "auto":
            files["language"] = (None, self.language)

        logger.debug("RemoteOpenAISTT: posting %d bytes to %s", len(pcm_bytes), self.endpoint)
        try:
            response = self._client.post(self.endpoint, headers=self.headers, files=files)
            response.raise_for_status()
            pred_text = response.json().get("text", "").strip()
        except httpx.HTTPError as exc:
            logger.error("RemoteOpenAISTT request failed: %s", exc)
            return

        if not pred_text:
            logger.debug("RemoteOpenAISTT: empty transcript, skipping")
            return

        console.print(f"[yellow]USER: {pred_text}")
        yield Transcription(text=pred_text)

    def cleanup(self) -> None:
        self._client.close()
