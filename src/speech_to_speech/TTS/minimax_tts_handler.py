# Copyright 2026 winoiknow (Eric Alborn, Anteon Group)
# Licensed under the Apache License, Version 2.0 (the "License").
# See the LICENSE file in the repository root for the full license text.

from __future__ import annotations

import json
import logging
from threading import Event
from time import perf_counter
from typing import Iterator

import numpy as np
from rich.console import Console

from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.debug import DEBUG_MODE
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.handler_types import TTSIn, TTSOut
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, EndOfResponse, TTSInput
from speech_to_speech.utils.ws_client import WSClient

logger = logging.getLogger(__name__)
console = Console()

SAMPLE_RATE = 16000          # MiniMax pcm @ 16000 == pipeline rate (no resample)
BYTES_PER_SAMPLE = 2         # int16
CHUNK_SAMPLES = 320          # 20 ms @ 16 kHz
CHUNK_BYTES = CHUNK_SAMPLES * BYTES_PER_SAMPLE


def _extract_audio(ev: dict) -> bytes:
    """MiniMax T2A v2 returns hex-encoded pcm in data.audio (base64 fallback)."""
    a = (ev.get("data") or {}).get("audio")
    if not a:
        return b""
    try:
        return bytes.fromhex(a)
    except ValueError:
        import base64
        try:
            return base64.b64decode(a)
        except Exception:
            return b""


class MiniMaxTTSHandler(BaseHandler[TTSIn, TTSOut]):
    """
    Streaming TTS over MiniMax's T2A v2 WebSocket. Selected with
    ``TTS_SOURCE=minimax`` (``--tts minimax``). Requests ``pcm`` at 16 kHz so the
    output drops straight onto the pipeline rate (no resample), and **yields each
    frame as it arrives** (streaming-first — never buffers the whole clip), which
    is what keeps time-to-first-audio ~0.5 s.

    A sibling to the F5/ElevenLabs handlers with the same barge-in contract: on
    ``cancel_scope.is_stale`` the WebSocket is closed immediately, stopping
    upstream synthesis. v1 connects per utterance (~0.5 s cold TTFB); a warm
    persistent connection (~0.27 s) is a future optimization.
    """

    def setup(
        self,
        should_listen: Event,
        api_key: str = "",
        voice_id: str = "",
        model: str = "speech-02-turbo",
        ws_url: str = "wss://api.minimax.io/ws/v1/t2a_v2",
        group_id: str = "",
        speed: float = 1.0,
        timeout: float = 20.0,
        gen_kwargs: dict | None = None,  # accepted for pipeline compatibility, unused
        cancel_scope: CancelScope | None = None,
    ) -> None:
        self.should_listen = should_listen
        self.cancel_scope = cancel_scope
        self.api_key = api_key
        self.voice_id = voice_id
        self.model = model
        self.speed = speed
        self.timeout = timeout
        self.connect_url = ws_url
        if group_id:
            sep = "&" if "?" in ws_url else "?"
            self.connect_url = f"{ws_url}{sep}GroupId={group_id}"
        if not api_key or not voice_id:
            logger.warning(
                "MiniMaxTTSHandler: MINIMAX_API_KEY and/or MINIMAX_VOICE_ID is unset — requests will fail"
            )
        logger.info(
            "MiniMaxTTSHandler ready → %s (model=%s, voice=%s, pcm@%d Hz)",
            ws_url, self.model, self.voice_id, SAMPLE_RATE,
        )

    def _emit(self, samples: np.ndarray) -> Iterator[np.ndarray]:
        for start in range(0, samples.size, CHUNK_SAMPLES):
            chunk = samples[start : start + CHUNK_SAMPLES]
            if chunk.size < CHUNK_SAMPLES:
                chunk = np.pad(chunk, (0, CHUNK_SAMPLES - chunk.size))
            yield chunk.copy()

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
        logger.debug("MiniMaxTTS: synthesising %d chars", len(text))

        ws: WSClient | None = None
        pipeline_start = perf_counter()
        remainder = b""        # leftover bytes that don't fill a CHUNK
        first_audio = True
        total = 0
        try:
            ws = WSClient(self.connect_url, {"Authorization": f"Bearer {self.api_key}"},
                          timeout=self.timeout)
            ws.recv()  # connected_success
            if DEBUG_MODE:
                logger.info("MiniMaxTTS: stream opened (gen=%s, model=%s)", gen, self.model)

            ws.send_text(json.dumps({
                "event": "task_start",
                "model": self.model,
                "voice_setting": {"voice_id": self.voice_id, "speed": self.speed,
                                  "vol": 1.0, "pitch": 0},
                "audio_setting": {"sample_rate": SAMPLE_RATE, "format": "pcm", "channel": 1},
            }))
            kind, msg = ws.recv()  # task_started (or error)
            ev = json.loads(msg)
            if ev.get("event") == "task_failed" or ev.get("base_resp", {}).get("status_code") not in (0, None):
                logger.error("MiniMaxTTS task_start rejected: %s", msg[:300])
                return

            ws.send_text(json.dumps({"event": "task_continue", "text": text}))

            while True:
                # Barge-in: abort before each frame read; closing the socket stops synthesis.
                if gen is not None and self.cancel_scope is not None and self.cancel_scope.is_stale(gen):
                    if DEBUG_MODE:
                        logger.info(
                            "MiniMaxTTS: ABORT on is_stale (captured gen=%s, current scope_gen=%s) "
                            "after %d bytes — barge-in, closing WS",
                            gen, self.cancel_scope.generation, total,
                        )
                    return

                kind, msg = ws.recv()
                if kind == "close":
                    break
                ev = json.loads(msg)
                chunk = _extract_audio(ev)
                if chunk:
                    total += len(chunk)
                    if first_audio:
                        if DEBUG_MODE:
                            logger.info("MiniMaxTTS: time-to-first-byte %.3fs (first audio chunk)",
                                        perf_counter() - pipeline_start)
                        first_audio = False
                    buf = remainder + chunk
                    usable = (len(buf) // CHUNK_BYTES) * CHUNK_BYTES
                    if usable:
                        samples = np.frombuffer(buf[:usable], dtype=np.int16)
                        yield from self._emit(samples)
                    remainder = buf[usable:]
                if ev.get("is_final") or ev.get("event") in ("task_finished", "task_failed"):
                    break

            # Flush any trailing partial chunk (zero-padded for alignment).
            if remainder:
                tail = np.frombuffer(remainder, dtype=np.int16) if len(remainder) % BYTES_PER_SAMPLE == 0 \
                    else np.frombuffer(remainder[:-1], dtype=np.int16)
                if tail.size:
                    yield from self._emit(tail)

            if DEBUG_MODE:
                logger.info("MiniMaxTTS: stream done — %d bytes (%.2fs audio)",
                            total, (total / BYTES_PER_SAMPLE) / SAMPLE_RATE)
        except Exception as exc:
            logger.error("MiniMaxTTS request failed: %s", exc)
            return
        finally:
            if ws is not None:
                ws.close()

    def cleanup(self) -> None:
        pass
