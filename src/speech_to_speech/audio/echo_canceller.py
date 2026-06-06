# Copyright 2026 winoiknow (Eric Alborn, Anteon Group)
# Licensed under the Apache License, Version 2.0 (the "License").
# See the LICENSE file in the repository root for the full license text.

"""Acoustic echo cancellation for the realtime input path.

s2s holds both signals AEC needs: the **near-end** (caller mic, via
input_audio_buffer.append) and the **far-end** (the TTS it just sent). This
canceller subtracts the agent's own audio from the mic *before the VAD sees it*,
so the VAD stops tripping on echo (the root of the phantom barge-ins and the
echo-clears-should_listen deafness).

Two backends, picked by ``AEC_BACKEND``:
  * **aec3** (default): WebRTC AEC3 via the pure-Rust ``aec3`` crate (PyO3 wheel
    ``aec3_py``). Has built-in delay estimation + clock-drift handling — the
    thing speex lacked on the networked, cross-clock echo path. Works on 10 ms
    (160-sample @ 16 kHz) frames; AEC3 aligns far/near internally, so we just
    feed render + capture.
  * **speex** (fallback): libspeexdsp via ctypes. Adaptive filter only — slow to
    converge on a cross-clock path; kept for comparison.

Fail-safe throughout: if the backend can't load or a frame errors, the mic
passes through unchanged so the pipeline never breaks.
"""

from __future__ import annotations

import ctypes
import logging
import time

from speech_to_speech.debug import DEBUG_MODE

logger = logging.getLogger(__name__)

BYTES_PER_SAMPLE = 2

# ── speex backend (libspeexdsp via ctypes) ──────────────────────────────────
SPEEX_FRAME_SAMPLES = 256
SPEEX_ECHO_SET_SAMPLING_RATE = 24
SPEEX_PREPROCESS_SET_ECHO_STATE = 24


def _load_speex():
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
        lib.speex_preprocess_state_init.restype = vp
        lib.speex_preprocess_state_init.argtypes = [ci, ci]
        lib.speex_preprocess_ctl.argtypes = [vp, ci, vp]
        lib.speex_preprocess_run.argtypes = [vp, vp]
        return lib
    return None


_SPEEX = _load_speex()


class EchoCanceller:
    def __init__(self, sample_rate: int = 16000, filter_length_ms: int = 250,
                 enabled: bool = True, backend: str = "aec3") -> None:
        self.sample_rate = sample_rate
        self.backend = (backend or "aec3").strip().lower()
        self.enabled = bool(enabled)
        self.frame = 0
        self._fb = 0
        # diagnostic (DEBUG_MODE)
        self._dbg_t = 0.0
        self._dbg_far = 0
        self._dbg_tot = 0
        # buffers (aec3 reframes 512-sample chunks to 160; output kept length-matched)
        self._near_buf = bytearray()
        self._far_buf = bytearray()
        self._clean_buf = bytearray()
        # aec3 handles
        self._aec3_mod = None
        self._aec3 = None
        # speex handles
        self._echo = None
        self._pre = None
        self._silence = b""
        self._far = bytearray()
        self._far_cap = sample_rate * BYTES_PER_SAMPLE * 2
        # far-activity tracking (for the VAD's far-aware gate)
        self._last_far_t = 0.0
        self._far_active_tail = 0.5  # keep far_active this long after the last far chunk

        if not self.enabled:
            return
        if self.backend == "aec3":
            self._init_aec3()
        elif self.backend == "speex":
            self._init_speex(filter_length_ms)
        else:
            logger.warning("AEC_BACKEND=%r unknown — mic passes through unchanged", self.backend)
            self.enabled = False

    # ── aec3 backend ────────────────────────────────────────────────────────
    def _init_aec3(self) -> None:
        try:
            import aec3_py  # the PyO3 wheel
            self._aec3_mod = aec3_py
            self.frame = self.sample_rate // 100   # 10 ms frame; confirmed on create
            self._fb = self.frame * BYTES_PER_SAMPLE
            logger.info("EchoCanceller backend=aec3 (WebRTC AEC3, pure-Rust) — lazy init on first frame")
        except Exception as e:
            logger.warning("AEC backend=aec3 unavailable (%s) — mic passes through unchanged", e)
            self.enabled = False

    def _ensure_aec3(self) -> bool:
        # aec3_py.Aec3 is `unsendable` → must be created on the thread that uses it
        # (the asyncio thread, where add_far_end/process run).
        if self._aec3 is None:
            try:
                self._aec3 = self._aec3_mod.Aec3(self.sample_rate)
                self.frame = self._aec3.frame_samples
                self._fb = self.frame * BYTES_PER_SAMPLE
            except Exception as e:
                logger.error("AEC aec3 init failed (%s) — passing through", e)
                self.enabled = False
                return False
        return True

    # ── speex backend ───────────────────────────────────────────────────────
    def _init_speex(self, filter_length_ms: int) -> None:
        self.frame = SPEEX_FRAME_SAMPLES
        self._fb = self.frame * BYTES_PER_SAMPLE
        self._silence = b"\x00" * self._fb
        self._spx_out = ctypes.create_string_buffer(self._fb)  # reusable per-frame output
        fl = int(self.sample_rate * filter_length_ms / 1000)
        self.filter_length = max(self.frame, (fl // self.frame) * self.frame)
        if _SPEEX is None:
            logger.warning("AEC backend=speex but libspeexdsp not found — mic passes through unchanged")
            self.enabled = False
            return
        try:
            self._echo = _SPEEX.speex_echo_state_init(self.frame, self.filter_length)
            if not self._echo:
                raise RuntimeError("speex_echo_state_init returned NULL")
            rate = ctypes.c_int(self.sample_rate)
            _SPEEX.speex_echo_ctl(self._echo, SPEEX_ECHO_SET_SAMPLING_RATE, ctypes.byref(rate))
            self._pre = _SPEEX.speex_preprocess_state_init(self.frame, self.sample_rate)
            if self._pre:
                _SPEEX.speex_preprocess_ctl(self._pre, SPEEX_PREPROCESS_SET_ECHO_STATE,
                                            ctypes.c_void_p(self._echo))
            logger.info("EchoCanceller backend=speex (libspeexdsp/ctypes; filter=%d/%d ms, residual=%s)",
                        self.filter_length, int(self.filter_length / self.sample_rate * 1000), bool(self._pre))
        except Exception as e:
            logger.error("AEC speex init failed (%s) — passing through", e)
            self.enabled = False

    # ── public API ──────────────────────────────────────────────────────────
    @property
    def far_active(self) -> bool:
        """True when the agent's audio is (or just was) echoing into the mic: there
        is far-end still buffered to 'play', or far-end arrived within the tail
        window. The VAD uses this to gate on the AEC residual during playback —
        raw echo can exceed the raw gate at high volume, but the residual can't."""
        if not self.enabled:
            return False
        buffered = (len(self._far_buf) > 0) if self.backend == "aec3" else (len(self._far) > 0)
        return buffered or (time.time() - self._last_far_t) < self._far_active_tail

    def add_far_end(self, pcm_int16: bytes) -> None:
        """Buffer outbound TTS (int16 mono @ sample_rate) as the far-end reference."""
        if not self.enabled or not pcm_int16:
            return
        self._last_far_t = time.time()
        if self.backend == "aec3":
            if not self._ensure_aec3():
                return
            # Buffer only — do NOT feed process_render here. TTS arrives in bursts
            # (e.g. 1.6 s of audio in ~0.5 s wall-time), seconds ahead of when the
            # client actually plays it and the echo reaches the mic. AEC3's render
            # buffer is bounded and expects render fed ~one frame per capture tick,
            # so the far-end is released in lockstep with the near-end in process()
            # (paced by the realtime mic clock). Cap so a cancelled/stale burst
            # can't grow unbounded (~2 s).
            self._far_buf += pcm_int16
            cap = self.sample_rate * BYTES_PER_SAMPLE * 2
            if len(self._far_buf) > cap:
                del self._far_buf[: len(self._far_buf) - cap]
        else:
            self._far += pcm_int16
            if len(self._far) > self._far_cap:
                del self._far[: len(self._far) - self._far_cap]

    def process(self, near_int16: bytes) -> bytes:
        """Return echo-cancelled mic bytes (same length as input)."""
        if not self.enabled:
            return near_int16
        return self._process_aec3(near_int16) if self.backend == "aec3" else self._process_speex(near_int16)

    def _process_aec3(self, near: bytes) -> bytes:
        if not self._ensure_aec3():
            return near
        self._near_buf += near
        silence = b"\x00" * self._fb
        # Lockstep: one render frame per capture frame, paced by the realtime mic
        # clock. The far-end leads the acoustic echo by the (bounded, ~constant)
        # playback+network delay, which AEC3's delay estimator aligns internally.
        while len(self._near_buf) >= self._fb:
            nframe = bytes(self._near_buf[:self._fb]); del self._near_buf[:self._fb]
            if len(self._far_buf) >= self._fb:
                rframe = bytes(self._far_buf[:self._fb]); del self._far_buf[:self._fb]
                if DEBUG_MODE:
                    self._dbg_far += 1
            else:
                rframe = silence
            try:
                self._aec3.process_render(rframe)
                self._clean_buf += self._aec3.process_capture(nframe)
            except Exception:
                self._clean_buf += nframe  # fail-safe: pass through
            if DEBUG_MODE:
                self._dbg_tot += 1
        want = len(near)
        if len(self._clean_buf) >= want:
            out = bytes(self._clean_buf[:want]); del self._clean_buf[:want]
        else:  # startup priming: emit what we have + a little silence to keep lengths matched
            out = bytes(self._clean_buf) + b"\x00" * (want - len(self._clean_buf))
            self._clean_buf.clear()
        if DEBUG_MODE:
            self._diag(near, out)
        return out

    def _process_speex(self, near: bytes) -> bytes:
        n = len(near)
        out = bytearray()
        off = 0
        far_frames = total_frames = 0
        while off + self._fb <= n:
            rec = near[off : off + self._fb]
            if len(self._far) >= self._fb:
                play = bytes(self._far[: self._fb]); del self._far[: self._fb]
                far_frames += 1
            else:
                play = self._silence
            total_frames += 1
            try:
                _SPEEX.speex_echo_cancellation(self._echo, rec, play, self._spx_out)
                if self._pre:
                    _SPEEX.speex_preprocess_run(self._pre, self._spx_out)
                out += self._spx_out.raw
            except Exception:
                out += rec
            off += self._fb
        if off < n:
            out += near[off:]
        result = bytes(out)
        if DEBUG_MODE:
            self._dbg_far += far_frames
            self._dbg_tot += total_frames
            self._diag(near, result)
        return result

    def flush_far(self) -> None:
        """Drop queued far-end (call on barge-in cancel: the client stops playing
        the queued TTS, so feeding it as render would make AEC3 expect echo that
        never arrives and subtract it from real user speech)."""
        self._far.clear()
        self._far_buf.clear()

    def reset(self) -> None:
        """Clear buffers (e.g. on a new session). Backend filter state is kept."""
        self._far.clear()
        self._near_buf.clear()
        self._far_buf.clear()
        self._clean_buf.clear()

    # ── diagnostic ──────────────────────────────────────────────────────────
    def _diag(self, near_b: bytes, out_b: bytes) -> None:
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
        logger.info("AEC[%s]: in_rms=%.0f out_rms=%.0f reduction=%.1f dB | far=%.0f%%",
                    self.backend, rin, rout, red, far_pct)
        self._dbg_t = now
        self._dbg_far = 0
        self._dbg_tot = 0
