# Copyright 2026 winoiknow (Eric Alborn, Anteon Group)
# Licensed under the Apache License, Version 2.0 (the "License").
# See the LICENSE file in the repository root for the full license text.

"""Acoustic echo cancellation for the realtime input path.

s2s is the one component that holds both signals AEC needs: the **near-end**
(the caller's mic, arriving as input_audio_buffer.append) and the **far-end**
(the TTS it just sent out). This canceller removes the agent's own audio from
the mic *before the VAD sees it*, so the VAD stops tripping on echo — which is
what caused both the phantom barge-ins and the deaf-during-playback churn.

Backend: speexdsp's adaptive echo canceller. The class is **fail-safe** — if
speexdsp isn't installed, AEC is disabled, or any frame errors, it returns the
mic unchanged so the pipeline never breaks.

Alignment: far-end (sent) and near-end (received) are fed in lockstep; the
round-trip delay (s2s → client → speaker → mic → s2s) is absorbed by the
adaptive filter as long as it's within ``filter_length``. On a LAN/browser that
delay is small (tens of ms); a large client jitter buffer (e.g. telephony) would
need a correspondingly long filter or a pre-delay — tune AEC_FILTER_LENGTH_MS.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    from speexdsp import EchoCanceller as _SpeexEchoCanceller  # type: ignore
    _HAS_SPEEX = True
except Exception:  # pragma: no cover - optional dependency
    _SpeexEchoCanceller = None
    _HAS_SPEEX = False

FRAME_SAMPLES = 256          # 16 ms @ 16 kHz; 512-sample VAD chunks = 2 frames
BYTES_PER_SAMPLE = 2


class EchoCanceller:
    """Per-session AEC. ``add_far_end`` buffers outbound TTS; ``process`` cleans
    inbound mic bytes (int16 mono @ ``sample_rate``) and returns the same length."""

    def __init__(self, sample_rate: int = 16000, filter_length_ms: int = 250,
                 enabled: bool = True) -> None:
        self.sample_rate = sample_rate
        self.frame = FRAME_SAMPLES
        self._fb = self.frame * BYTES_PER_SAMPLE          # frame size in bytes
        self._silence = b"\x00" * self._fb
        # filter (tail) length, rounded to a whole number of frames
        fl = int(sample_rate * filter_length_ms / 1000)
        self.filter_length = max(self.frame, (fl // self.frame) * self.frame)
        self._far = bytearray()                            # far-end FIFO (int16 bytes)
        self._far_cap = sample_rate * BYTES_PER_SAMPLE * 2  # cap ~2 s of far-end
        self._ec = None

        self.enabled = bool(enabled) and _HAS_SPEEX
        if enabled and not _HAS_SPEEX:
            logger.warning("AEC requested but speexdsp is not installed — mic passes through unchanged")
        if self.enabled:
            try:
                self._ec = _SpeexEchoCanceller.create(self.frame, self.filter_length, sample_rate)
                logger.info(
                    "EchoCanceller ready (frame=%d, filter=%d samples / %d ms @ %d Hz)",
                    self.frame, self.filter_length,
                    int(self.filter_length / sample_rate * 1000), sample_rate,
                )
            except Exception as e:  # pragma: no cover
                logger.error("AEC init failed (%s) — mic passes through unchanged", e)
                self.enabled = False

    def add_far_end(self, pcm_int16: bytes) -> None:
        """Buffer outbound TTS (int16 mono @ sample_rate) as the far-end reference."""
        if not self.enabled or not pcm_int16:
            return
        self._far += pcm_int16
        if len(self._far) > self._far_cap:                 # bound memory if near-end stalls
            del self._far[: len(self._far) - self._far_cap]

    def process(self, near_int16: bytes) -> bytes:
        """Return echo-cancelled mic bytes (same length as input)."""
        if not self.enabled or self._ec is None:
            return near_int16
        n = len(near_int16)
        out = bytearray()
        off = 0
        while off + self._fb <= n:
            rec = near_int16[off : off + self._fb]
            if len(self._far) >= self._fb:
                play = bytes(self._far[: self._fb]); del self._far[: self._fb]
            else:
                play = self._silence                       # no far-end → nothing to cancel
            try:
                out += self._ec.process(rec, play)
            except Exception:
                out += rec                                 # fail-safe per frame
            off += self._fb
        if off < n:                                        # trailing partial frame → pass through
            out += near_int16[off:]
        return bytes(out)

    def reset(self) -> None:
        """Clear far-end buffer (e.g. on a new session). Filter state is kept."""
        self._far.clear()
