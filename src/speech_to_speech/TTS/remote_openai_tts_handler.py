# Copyright 2026 winoiknow (Eric Alborn, Anteon Group)
# Licensed under the Apache License, Version 2.0 (the "License").
# See the LICENSE file in the repository root for the full license text.

from __future__ import annotations

import logging
from threading import Event
from time import perf_counter
from typing import Iterator

import httpx
import numpy as np
from rich.console import Console

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
    Streams 16 kHz int16 mono PCM from a remote OpenAI-compatible TTS endpoint
    (e.g. winoiknow/openai-f5-tts with /v1/audio/speech/stream).

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
        gen_kwargs: dict | None = None,  # accepted for pipeline compatibility, unused
        cancel_scope: CancelScope | None = None,
    ) -> None:
        self.should_listen = should_listen
        self.cancel_scope = cancel_scope
        self.voice = voice
        self.model = model
        self.stream_endpoint = base_url.rstrip("/") + "/v1/audio/speech/stream"
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.timeout = timeout
        self._client = httpx.Client(timeout=self.timeout)
        logger.info(
            "RemoteOpenAITTSHandler ready → %s (voice=%s, model=%s)",
            self.stream_endpoint, self.voice, self.model,
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
        remainder = b""
        raw_total = 0  # DEBUG: total bytes read off the socket
        chunks_yielded = 0  # DEBUG: 512-sample chunks emitted downstream

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
                            "after %d bytes read, %d chunks yielded — barge-in, closing upstream stream",
                            gen,
                            self.cancel_scope.generation,
                            raw_total,
                            chunks_yielded,
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

                    remainder += raw

                    # Yield complete 512-sample chunks
                    while len(remainder) >= CHUNK_BYTES:
                        chunk_bytes = remainder[:CHUNK_BYTES]
                        remainder = remainder[CHUNK_BYTES:]
                        chunks_yielded += 1
                        yield np.frombuffer(chunk_bytes, dtype=np.int16).copy()

                logger.info(
                    "RemoteOpenAITTS: iter_bytes loop done — %d bytes read, %d chunks yielded",
                    raw_total,
                    chunks_yielded,
                )

        except httpx.HTTPError as exc:
            logger.error("RemoteOpenAITTS request failed: %s", exc)
            return

        # Yield any trailing bytes, zero-padded to exactly CHUNK_SAMPLES for downstream alignment
        if len(remainder) >= BYTES_PER_SAMPLE:
            n_samples = len(remainder) // BYTES_PER_SAMPLE
            usable = remainder[: n_samples * BYTES_PER_SAMPLE]
            chunk = np.frombuffer(usable, dtype=np.int16).copy()
            if len(chunk) < CHUNK_SAMPLES:
                chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))
            yield chunk

    def cleanup(self) -> None:
        self._client.close()
