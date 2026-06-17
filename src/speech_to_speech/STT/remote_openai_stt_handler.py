# Copyright 2026 winoiknow (Eric Alborn, Anteon Group)
# Licensed under the Apache License, Version 2.0 (the "License").
# See the LICENSE file in the repository root for the full license text.

from __future__ import annotations

import io
import logging
import struct
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterator

import httpx
import numpy as np
from rich.console import Console

from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.pipeline.handler_types import STTIn, STTOut
from speech_to_speech.pipeline.messages import Transcription
from speech_to_speech.utils.concurrency import STT_LIMITER

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
        speaker_client: Any | None = None,  # RemoteSpeakerClient when SPEAKER_ID_ENABLED
        speaker_timeout: float = 0.8,
        diarize_enabled: bool = False,  # SPEAKER_DIARIZE_ENABLED — carry audio forward
        gen_kwargs: dict | None = None,  # accepted for pipeline compatibility, unused
    ) -> None:
        self.model = model
        self.language = language
        self.endpoint = base_url.rstrip("/") + "/v1/audio/transcriptions"
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.timeout = timeout
        self._client = httpx.Client(timeout=self.timeout)
        # Speaker identification runs concurrently with the transcribe round-trip
        # (both network-bound → overlap → ~0 added latency). None → disabled.
        self.speaker_client = speaker_client
        self.speaker_timeout = speaker_timeout
        # When diarization is on, attach the raw turn WAV to the Transcription so
        # the service layer can run the off-hot-path diarize + emit a correction.
        self.diarize_enabled = diarize_enabled
        self._spk_pool = ThreadPoolExecutor(max_workers=1) if speaker_client is not None else None
        logger.info("RemoteOpenAISTTHandler ready → %s (model=%s, speaker_id=%s)",
                    self.endpoint, self.model, "on" if speaker_client is not None else "off")

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

        # Fire speaker identify CONCURRENTLY (same wav_bytes = raw user voice) so it
        # overlaps the transcribe round-trip; joined below with a hard bound.
        spk_future = None
        if self.speaker_client is not None and self._spk_pool is not None:
            spk_future = self._spk_pool.submit(self.speaker_client.identify, wav_bytes)

        files: dict[str, Any] = {
            "file": ("audio.wav", io.BytesIO(wav_bytes), "audio/wav"),
            "model": (None, self.model),
            "response_format": (None, "verbose_json"),
        }
        if self.language and self.language != "auto":
            files["language"] = (None, self.language)

        logger.debug("RemoteOpenAISTT: posting %d bytes to %s", len(wav_bytes), self.endpoint)
        try:
            # Cap concurrent in-flight STT requests across all sessions (no-op
            # unless STT_MAX_CONCURRENCY is set); the slot covers the blocking POST.
            with STT_LIMITER.slot():
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

        # Join identify (already overlapped with transcribe → usually instant).
        # Bounded by the client timeout + a small grace; any failure → unknown.
        speaker = None
        if spk_future is not None:
            try:
                speaker = spk_future.result(timeout=self.speaker_timeout + 0.2)
            except Exception:
                speaker = None
            if speaker is not None:
                logger.debug("speaker: decision=%s id=%s score=%.3f", speaker.decision, speaker.speaker_id, speaker.score)

        console.print(f"[yellow]USER: {pred_text}")
        # Carry the raw turn WAV forward only when diarization is on, so the service
        # layer can diarize off the hot path and emit a correction. None otherwise.
        audio_wav = wav_bytes if self.diarize_enabled else None
        yield Transcription(text=pred_text, language_code=language_code, speaker=speaker, audio_wav=audio_wav)

    def cleanup(self) -> None:
        self._client.close()  # this handler's own transcription client — safe to close
        if self._spk_pool is not None:
            self._spk_pool.shutdown(wait=False)
        # Do NOT close self.speaker_client here: it is the SHARED RemoteSpeakerClient
        # built once by HandlerFactory and handed to every session. Closing it on one
        # session's teardown closes its httpx client process-wide, so every later
        # identify fails with "Cannot send a request, as the client has been closed"
        # until restart. The factory owns it; it's closed once at server shutdown.
