# Copyright 2026 winoiknow (Eric Alborn, Anteon Group)
# Licensed under the Apache License, Version 2.0 (the "License").
# See the LICENSE file in the repository root for the full license text.

from __future__ import annotations

import audioop
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
from speech_to_speech.debug import DEBUG_MODE
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.handler_types import TTSIn, TTSOut
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, EndOfResponse, TTSInput

logger = logging.getLogger(__name__)
console = Console()

SAMPLE_RATE = 16000          # pipeline rate
BYTES_PER_SAMPLE = 2         # int16
CHUNK_SAMPLES = 320          # 20 ms @ 16 kHz — small chunks keep outbound pacing smooth


def _decode_params(output_format: str) -> tuple[int, bool]:
    """Map an ElevenLabs output_format to (source_rate_hz, is_ulaw).

    Supports ``pcm_<rate>`` (raw int16 LE mono) and ``ulaw_8000`` (8-bit µ-law,
    free-tier friendly). mp3 is intentionally unsupported — it would need a heavy
    decoder, and PCM/µ-law cover the realtime path cleanly.
    """
    fmt = output_format.strip().lower()
    if fmt.startswith("pcm_"):
        return int(fmt.split("_", 1)[1]), False
    if fmt == "ulaw_8000":
        return 8000, True
    raise ValueError(
        f"Unsupported ELEVENLABS_OUTPUT_FORMAT {output_format!r}; use 'pcm_<rate>' or 'ulaw_8000' (mp3 not supported)"
    )


class ElevenLabsTTSHandler(BaseHandler[TTSIn, TTSOut]):
    """
    Streams audio from the ElevenLabs text-to-speech ``/stream`` endpoint and
    yields int16 mono PCM at the 16 kHz pipeline rate. Selected via
    ``TTS_SOURCE=elevenlabs`` (``--tts elevenlabs``); a sibling to
    RemoteOpenAITTSHandler with the same barge-in and chunk-output contract.

    Output format is taken from ``output_format`` (default ``pcm_16000`` → no
    resample). The upstream HTTP connection is closed immediately on barge-in
    (``cancel_scope.is_stale``), like the OpenAI-remote handler.
    """

    def setup(
        self,
        should_listen: Event,
        base_url: str = "https://api.elevenlabs.io",
        api_key: str = "",
        voice_id: str = "",
        model_id: str = "eleven_flash_v2_5",
        output_format: str = "pcm_16000",
        stability: float = 0.5,
        similarity_boost: float = 0.75,
        timeout: float = 60.0,
        gen_kwargs: dict | None = None,  # accepted for pipeline compatibility, unused
        cancel_scope: CancelScope | None = None,
    ) -> None:
        self.should_listen = should_listen
        self.cancel_scope = cancel_scope
        self.api_key = api_key
        self.voice_id = voice_id
        self.model_id = model_id
        self.output_format = output_format
        self.stability = stability
        self.similarity_boost = similarity_boost
        self.src_rate, self.is_ulaw = _decode_params(output_format)
        self.stream_endpoint = (
            base_url.rstrip("/") + f"/v1/text-to-speech/{voice_id}/stream?output_format={output_format}"
        )
        self.headers = {"xi-api-key": api_key, "Content-Type": "application/json"}
        self.timeout = timeout
        self._client = httpx.Client(timeout=self.timeout)
        if not api_key or not voice_id:
            logger.warning(
                "ElevenLabsTTSHandler: ELEVENLABS_API_KEY and/or ELEVENLABS_VOICE_ID is unset — requests will fail"
            )
        logger.info(
            "ElevenLabsTTSHandler ready → %s (model=%s, format=%s, src_rate=%d→%d Hz%s)",
            self.stream_endpoint.split("?")[0],
            self.model_id,
            self.output_format,
            self.src_rate,
            SAMPLE_RATE,
            ", µ-law" if self.is_ulaw else "",
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
        logger.debug("ElevenLabsTTS: synthesising %d chars", len(text))

        payload = {
            "text": text,
            "model_id": self.model_id,
            "voice_settings": {
                "stability": self.stability,
                "similarity_boost": self.similarity_boost,
            },
        }
        pipeline_start = perf_counter()
        first_chunk = True
        body = bytearray()
        raw_total = 0

        try:
            with self._client.stream(
                "POST",
                self.stream_endpoint,
                headers=self.headers,
                json=payload,
            ) as response:
                response.raise_for_status()
                if DEBUG_MODE:
                    logger.info("ElevenLabsTTS: stream opened (gen=%s, format=%s)", gen, self.output_format)
                for raw in response.iter_bytes():
                    raw_total += len(raw)
                    # Abort promptly on barge-in: closing the context manager tears
                    # down the upstream connection.
                    if gen is not None and self.cancel_scope is not None and self.cancel_scope.is_stale(gen):
                        if DEBUG_MODE:
                            logger.info(
                                "ElevenLabsTTS: ABORT on is_stale (captured gen=%s, current scope_gen=%s) "
                                "after %d bytes read — barge-in, closing upstream stream",
                                gen,
                                self.cancel_scope.generation,
                                raw_total,
                            )
                        return

                    if first_chunk and DEBUG_MODE:
                        logger.info(
                            "ElevenLabsTTS: time-to-first-byte %.3fs (first %d bytes)",
                            perf_counter() - pipeline_start,
                            len(raw),
                        )
                    first_chunk = False
                    body += raw

                if DEBUG_MODE:
                    logger.info("ElevenLabsTTS: stream done — %d bytes read", raw_total)

        except httpx.HTTPStatusError as exc:
            # Surface ElevenLabs' JSON error body (quota/voice/format issues) at ERROR.
            detail = ""
            try:
                detail = exc.response.read().decode("utf-8", "replace")[:500]
            except Exception:
                pass
            logger.error("ElevenLabsTTS request failed: %s %s", exc, detail)
            return
        except httpx.HTTPError as exc:
            logger.error("ElevenLabsTTS request failed: %s", exc)
            return

        if not body:
            return

        # Decode to int16 mono at src_rate.
        if self.is_ulaw:
            pcm = audioop.ulaw2lin(bytes(body), 2)
            samples = np.frombuffer(pcm, dtype=np.int16)
        else:
            usable = bytes(body[: (len(body) // BYTES_PER_SAMPLE) * BYTES_PER_SAMPLE])
            samples = np.frombuffer(usable, dtype=np.int16)
        if samples.size == 0:
            return

        # Resample once over the whole clip to the pipeline rate (avoids per-chunk
        # boundary artifacts), only when the source rate differs.
        if self.src_rate != SAMPLE_RATE:
            g = gcd(SAMPLE_RATE, self.src_rate)
            resampled = resample_poly(samples.astype(np.float32), SAMPLE_RATE // g, self.src_rate // g)
            samples = np.clip(np.round(resampled), -32768, 32767).astype(np.int16)

        if DEBUG_MODE:
            logger.info(
                "ElevenLabsTTS: %d samples @ %d Hz → %d samples @ %d Hz (%.2fs audio, resample=%s)",
                samples.size if self.src_rate == SAMPLE_RATE else -1,
                self.src_rate,
                samples.size,
                SAMPLE_RATE,
                samples.size / SAMPLE_RATE,
                self.src_rate != SAMPLE_RATE,
            )

        # Yield fixed CHUNK_SAMPLES chunks, zero-padding the last for alignment.
        for start in range(0, samples.size, CHUNK_SAMPLES):
            chunk = samples[start : start + CHUNK_SAMPLES]
            if chunk.size < CHUNK_SAMPLES:
                chunk = np.pad(chunk, (0, CHUNK_SAMPLES - chunk.size))
            yield chunk.copy()

    def cleanup(self) -> None:
        self._client.close()
