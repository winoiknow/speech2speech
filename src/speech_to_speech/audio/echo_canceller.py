# Copyright 2026 winoiknow (Eric Alborn, Anteon Group)
# Licensed under the Apache License, Version 2.0 (the "License").
# See the LICENSE file in the repository root for the full license text.

"""Acoustic echo cancellation for the realtime input path.

s2s holds both signals AEC needs: the **near-end** (caller mic, via
input_audio_buffer.append) and the **far-end** (the TTS it just sent). This
canceller subtracts the agent's own audio from the mic *before the VAD sees it*,
so the VAD stops tripping on echo (the root of both the phantom barge-ins and the
echo-clears-should_listen deafness).

Backend: **libspeexdsp via ctypes** — the echo canceller plus the preprocessor's
residual-echo suppressor, called directly against the system shared library. No
build step / Cython package (that pip package doesn't build on modern Python);
only the runtime lib (apt ``libspeexdsp1``) is needed. The class is **fail-safe**:
if the lib can't be loaded, AEC is disabled, or a frame errors, it returns the
mic unchanged so the pipeline never breaks.

Alignment: far-end (sent) and near-end (received) are fed in lockstep; the echo
round-trip delay is absorbed by the adaptive filter as long as it's within
``filter_length`` (tens of ms on LAN/browser; a client jitter buffer would need a
much longer filter or a pre-delay). For variable/large delay, WebRTC AEC3 (with
its built-in delay estimator) is the stronger backend — a planned alternative.
"""

from __future__ import annotations

import ctypes
import logging
import time

from speech_to_speech.debug import DEBUG_MODE

logger = logging.getLogger(__name__)

FRAME_SAMPLES = 256          # 16 ms @ 16 kHz; 512-sample VAD chunks = 2 frames
BYTES_PER_SAMPLE = 2

SPEEX_ECHO_SET_SAMPLING_RATE = 24
SPEEX_PREPROCESS_SET_ECHO_STATE = 24


def _load_lib():
    for name in ("libspeexdsp.so.1", "libspeexdsp.so", "libspeexdsp.so.1.5.0"):
        try:
            lib = ctypes.CDLL(name)
        except OSError:
            continue
        vp, ci = ctypes.c_void_p, ctypes.c_int
        lib.speex_echo_state_init.restype = vp
        lib.speex_echo_state_init.argtypes = [ci, ci]
        lib.speex_echo_cancellation.argtypes = [vp, vp, vp, vp]
        lib.speex_echo_ctl.argtypes = [vp, ci, vp]
        lib.speex_echo_state_destroy.argtypes = [vp]
        lib.speex_preprocess_state_init.restype = vp
        lib.speex_preprocess_state_init.argtypes = [ci, ci]
        lib.speex_preprocess_ctl.argtypes = [vp, ci, vp]
        lib.speex_preprocess_run.argtypes = [vp, vp]
        lib.speex_preprocess_state_destroy.argtypes = [vp]
        return lib
    return None


_LIB = _load_lib()


class EchoCanceller:
    """Per-session AEC. ``add_far_end`` buffers outbound TTS; ``process`` cleans
    inbound mic bytes (int16 mono @ ``sample_rate``) and returns the same length."""

    def __init__(self, sample_rate: int = 16000, filter_length_ms: int = 250,
                 enabled: bool = True) -> None:
        self.sample_rate = sample_rate
        self.frame = FRAME_SAMPLES
        self._fb = self.frame * BYTES_PER_SAMPLE
        self._silence = b"\x00" * self._fb
        self._out = ctypes.create_string_buffer(self._fb)   # reusable per-frame output
        fl = int(sample_rate * filter_length_ms / 1000)
        self.filter_length = max(self.frame, (fl // self.frame) * self.frame)
        self._far = bytearray()
        self._far_cap = sample_rate * BYTES_PER_SAMPLE * 2   # ~2 s cap
        self._echo = None
        self._pre = None
        self._dbg_t = 0.0          # diagnostic (DEBUG_MODE): echo-reduction meter
        self._dbg_far = 0
        self._dbg_tot = 0

        self.enabled = bool(enabled) and _LIB is not None
        if enabled and _LIB is None:
            logger.warning("AEC requested but libspeexdsp not found — mic passes through unchanged")
        if self.enabled:
            try:
                self._echo = _LIB.speex_echo_state_init(self.frame, self.filter_length)
                if not self._echo:
                    raise RuntimeError("speex_echo_state_init returned NULL")
                rate = ctypes.c_int(sample_rate)
                _LIB.speex_echo_ctl(self._echo, SPEEX_ECHO_SET_SAMPLING_RATE, ctypes.byref(rate))
                # Preprocessor linked to the echo state → residual-echo suppression.
                self._pre = _LIB.speex_preprocess_state_init(self.frame, sample_rate)
                if self._pre:
                    _LIB.speex_preprocess_ctl(self._pre, SPEEX_PREPROCESS_SET_ECHO_STATE,
                                              ctypes.c_void_p(self._echo))
                logger.info(
                    "EchoCanceller ready (libspeexdsp/ctypes; frame=%d, filter=%d samples / %d ms @ %d Hz, residual=%s)",
                    self.frame, self.filter_length,
                    int(self.filter_length / sample_rate * 1000), sample_rate, bool(self._pre),
                )
            except Exception as e:  # pragma: no cover
                logger.error("AEC init failed (%s) — mic passes through unchanged", e)
                self.enabled = False

    def add_far_end(self, pcm_int16: bytes) -> None:
        """Buffer outbound TTS (int16 mono @ sample_rate) as the far-end reference."""
        if not self.enabled or not pcm_int16:
            return
        self._far += pcm_int16
        if len(self._far) > self._far_cap:
            del self._far[: len(self._far) - self._far_cap]

    def process(self, near_int16: bytes) -> bytes:
        """Return echo-cancelled mic bytes (same length as input)."""
        if not self.enabled:
            return near_int16
        n = len(near_int16)
        out = bytearray()
        off = 0
        far_frames = total_frames = 0
        while off + self._fb <= n:
            rec = near_int16[off : off + self._fb]
            if len(self._far) >= self._fb:
                play = bytes(self._far[: self._fb]); del self._far[: self._fb]
                far_frames += 1
            else:
                play = self._silence                         # no far-end → nothing to cancel
            total_frames += 1
            try:
                _LIB.speex_echo_cancellation(self._echo, rec, play, self._out)
                if self._pre:
                    _LIB.speex_preprocess_run(self._pre, self._out)
                out += self._out.raw
            except Exception:
                out += rec                                   # fail-safe per frame
            off += self._fb
        if off < n:                                          # trailing partial frame
            out += near_int16[off:]
        result = bytes(out)
        if DEBUG_MODE and total_frames:
            self._diag(near_int16, result, far_frames, total_frames)
        return result

    def _diag(self, near_b: bytes, out_b: bytes, far_frames: int, total_frames: int) -> None:
        """Once/sec under DEBUG_MODE: how much echo energy is actually removed, and
        whether the far-end reference was present. reduction≈0 dB while far_present
        is high ⇒ the canceller isn't locking (cross-clock / convergence) → AEC3."""
        self._dbg_far += far_frames
        self._dbg_tot += total_frames
        now = time.time()
        if now - self._dbg_t < 1.0:
            return
        import numpy as np
        nin = np.frombuffer(near_b, dtype=np.int16).astype(np.float64)
        m = (len(out_b) // 2) * 2
        nout = np.frombuffer(out_b[:m], dtype=np.int16).astype(np.float64)
        rin = float(np.sqrt(np.mean(nin ** 2))) if nin.size else 0.0
        rout = float(np.sqrt(np.mean(nout ** 2))) if nout.size else 0.0
        red = 20.0 * np.log10(rin / rout) if (rin > 1 and rout > 1) else 0.0
        far_pct = 100.0 * self._dbg_far / max(1, self._dbg_tot)
        logger.info("AEC: in_rms=%.0f out_rms=%.0f reduction=%.1f dB | far_present=%.0f%%",
                    rin, rout, red, far_pct)
        self._dbg_t = now
        self._dbg_far = 0
        self._dbg_tot = 0

    def reset(self) -> None:
        """Clear the far-end buffer (e.g. on a new session)."""
        self._far.clear()
