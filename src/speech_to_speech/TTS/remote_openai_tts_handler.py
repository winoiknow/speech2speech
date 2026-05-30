# Copyright 2026 winoiknow (Eric Alborn, Anteon Group)
# Licensed under the Apache License, Version 2.0 (the "License").
# See the LICENSE file in the repository root for the full license text.

from __future__ import annotations

import logging
from math import gcd
from threading import Event
from time import perf_counter
from typing import Iterator

import httpx
import numpy as np
from rich.console import Console
from scipy.signal import resample_poly

from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.handler_types import TTSIn, TTSOut
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, EndOfResponse, TTSInput

logger = logging.getLogger(__name__)
console = Console()

SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2  # int16
CHUNK_SAMPLES = 512
CHUNK_BYTES = CHUNK_SAMPLES * BYTES_PER_SAMPLE


class RemoteOpenAITTSHandler(BaseHandler[TTSIn, TTSOut]):
    """
    Streams int16 mono PCM from a remote OpenAI-compatible TTS endpoint
    (e.g. winoiknow/openai-f5-tts with /v1/audio/speech/stream) and resamples it
    from the endpoint's native rate (``source_sample_rate``, F5-TTS is 24000) to
    the 16 kHz pipeline rate. Skipping this resample makes the audio play ~1.5x
    too slow and low-pitched downstream.

    The upstream HTTP connection is closed immediately when a barge-in is detected
    (cancel_scope.is_stale), preventing wasted bandwidth and latency.
    """

    def setup(
        self,
        should_listen: Event,
        base_url: str = "http://localhost:8880",
        api_key: str = "sk-unused",
        voice: str = "default",
        model: str = "tts-1",
        timeout: float = 60.0,
        source_sample_rate: int = 24000,
        gen_kwargs: dict | None = None,  # accepted for pipeline compatibility, unused
        cancel_scope: CancelScope | None = None,
    ) -> None:
        self.should_listen = should_listen
        self.cancel_scope = cancel_scope
        self.voice = voice
        self.model = model
        self.source_sample_rate = source_sample_rate
        self.stream_endpoint = base_url.rstrip("/") + "/v1/audio/speech/stream"
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.timeout = timeout
        self._client = httpx.Client(timeout=self.timeout)
        logger.info(
            "RemoteOpenAITTSHandler ready → %s (voice=%s, model=%s, source_rate=%d→%d Hz)",
            self.stream_endpoint, self.voice, self.model, self.source_sample_rate, SAMPLE_RATE,
        )

    def process(self, tts_input: TTSIn) -> Iterator[TTSOut]:
        if isinstance(tts_input, EndOfResponse):
            yield AUDIO_RESPONSE_DONE
            return

        assert isinstance(tts_input, TTSInput)

        gen = self.cancel_scope.generation if self.cancel_scope else None
        text = tts_input.text.strip()
        if not text:
            return

        console.print(f"[green]ASSISTANT: {text}")
        logger.debug("RemoteOpenAITTS: synthesising %d chars", len(text))

        payload = {
            "model": self.model,
            "input": text,
            "voice": self.voice,
        }
        pipeline_start = perf_counter()
        first_chunk = True
        body = bytearray()  # full clip; F5-TTS returns it in one shot
        raw_total = 0  # DEBUG: total bytes read off the socket

        try:
            with self._client.stream(
                "POST",
                self.stream_endpoint,
                headers={**self.headers, "Content-Type": "application/json"},
                json=payload,
            ) as response:
                response.raise_for_status()
                logger.info(
                    "RemoteOpenAITTS: stream opened (gen=%s, scope_gen=%s, entering iter_bytes loop)",
                    gen,
                    self.cancel_scope.generation if self.cancel_scope else None,
                )

                for raw in response.iter_bytes():
                    raw_total += len(raw)
                    # Check cancellation before processing each received chunk
                    if gen is not None and self.cancel_scope is not None and self.cancel_scope.is_stale(gen):
                        logger.info(
                            "RemoteOpenAITTS: ABORT on is_stale (captured gen=%s, current scope_gen=%s) "
                            "after %d bytes read — barge-in, closing upstream stream",
                            gen,
                            self.cancel_scope.generation,
                            raw_total,
                        )
                        return

                    if first_chunk:
                        logger.info(
                            "RemoteOpenAITTS: time-to-first-byte %.3fs (first %d bytes, header=%r) "
                            "— %r means WAV/RIFF container (NOT raw int16 PCM); frombuffer will "
                            "mis-parse the 44-byte header as samples",
                            perf_counter() - pipeline_start,
                            len(raw),
                            raw[:4],
                            b"RIFF",
                        )
                        first_chunk = False

                    body += raw

                logger.info("RemoteOpenAITTS: iter_bytes loop done — %d bytes read", raw_total)

        except httpx.HTTPError as exc:
            logger.error("RemoteOpenAITTS request failed: %s", exc)
            return

        usable = bytes(body[: (len(body) // BYTES_PER_SAMPLE) * BYTES_PER_SAMPLE])
        samples = np.frombuffer(usable, dtype=np.int16)
        if samples.size == 0:
            return

        # Resample from the endpoint's native rate to the 16 kHz pipeline rate.
        # Done once over the whole clip to avoid per-chunk boundary artifacts.
        if self.source_sample_rate != SAMPLE_RATE:
            g = gcd(SAMPLE_RATE, self.source_sample_rate)
            resampled = resample_poly(
                samples.astype(np.float32), SAMPLE_RATE // g, self.source_sample_rate // g
            )
            samples = np.clip(np.round(resampled), -32768, 32767).astype(np.int16)

        logger.info(
            "RemoteOpenAITTS: resampled %d samples @ %d Hz → %d samples @ %d Hz (%.2fs audio)",
            len(usable) // BYTES_PER_SAMPLE,
            self.source_sample_rate,
            samples.size,
            SAMPLE_RATE,
            samples.size / SAMPLE_RATE,
        )

        # Yield fixed CHUNK_SAMPLES chunks, zero-padding the last for downstream alignment.
        for start in range(0, samples.size, CHUNK_SAMPLES):
            chunk = samples[start : start + CHUNK_SAMPLES]
            if chunk.size < CHUNK_SAMPLES:
                chunk = np.pad(chunk, (0, CHUNK_SAMPLES - chunk.size))
            yield chunk.copy()

    def cleanup(self) -> None:
        self._client.close()
