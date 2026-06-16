from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Iterator
from queue import Queue
from threading import Event
from typing import Any, TypeAlias

import numpy as np
import torch

from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.debug import DEBUG_MODE
from speech_to_speech.pipeline.events import SpeechStartedEvent, SpeechStoppedEvent
from speech_to_speech.pipeline.handler_types import VADIn, VADOut
from speech_to_speech.pipeline.messages import VADAudio
from speech_to_speech.pipeline.queue_types import TextEventItem
from speech_to_speech.utils.utils import int2float
from speech_to_speech.VAD.vad_iterator import VADIterator

logger = logging.getLogger(__name__)

VADInput: TypeAlias = bytes | tuple[bytes, RuntimeConfig] | tuple[bytes, bytes, RuntimeConfig]

# Optional import for audio enhancement
try:
    from df.enhance import enhance, init_df

    HAS_DF = True
except (ImportError, ModuleNotFoundError) as e:
    HAS_DF = False
    logger.warning(f"DeepFilterNet not available for audio enhancement: {e}")


class VADHandler(BaseHandler[VADIn, VADOut]):
    """
    Handles voice activity detection. When voice activity is detected, audio will be accumulated until the end of speech is detected and then passed
    to the following part.
    """

    def setup(
        self,
        should_listen: Event,
        thresh: float = 0.6,
        sample_rate: int = 16000,
        min_silence_ms: int = 1000,
        min_speech_ms: int = 500,
        max_speech_ms: float = float("inf"),
        speech_pad_ms: int = 30,
        audio_enhancement: bool = False,
        enable_realtime_transcription: bool = False,
        realtime_processing_pause: float = 0.25,
        input_rms_gate: float = 0.0,
        input_rms_gate_far: float = 400.0,
        far_sustain_min: int = 6,
        far_sustain_window: int = 12,
        turn_detection: str = "vad",
        turn_min_silence_ms: int = 300,
        turn_max_s: float = 30.0,
        turn_threshold: float = 0.5,
        turn_hold_grace_ms: int = 1200,
        smart_turn_model_path: str = "",
        text_output_queue: Queue[TextEventItem] | None = None,
        vad_model: Any = None,
        turn_detector: Any = None,
    ) -> None:
        self.should_listen = should_listen
        self.sample_rate = sample_rate
        self.input_rms_gate = float(input_rms_gate)
        self.input_rms_gate_far = float(input_rms_gate_far)
        # sustained-energy filter for the far (residual) gate: reject sparse
        # transient echo spikes, pass dense real speech.
        self._far_sustain_min = max(1, int(far_sustain_min))
        self._far_run: deque[int] = deque(maxlen=max(self._far_sustain_min, int(far_sustain_window)))
        self.min_silence_ms = min_silence_ms
        self.min_speech_ms = min_speech_ms
        self.max_speech_ms = max_speech_ms
        self.enable_realtime_transcription = enable_realtime_transcription
        self.realtime_processing_pause = realtime_processing_pause
        self.text_output_queue = text_output_queue
        self._last_turn_detection: dict | None = None
        # A pre-warmed Silero model (a per-session deepcopy made by the factory at
        # connect) lets us skip torch.hub.load on the hot connect path. Each copy
        # is independent — Silero rebinds its RNN state functionally per call, so
        # concurrent sessions never bleed through it. Fall back to loading here if
        # none was supplied (single-build / tests / pre-warm failure).
        if vad_model is not None:
            self.model = vad_model
        else:
            self.model, _ = torch.hub.load(
                "snakers4/silero-vad",
                "silero_vad",
                trust_repo=True,
                skip_validation=True,
            )
        self.iterator = VADIterator(
            self.model,
            threshold=thresh,
            sampling_rate=sample_rate,
            min_silence_duration_ms=min_silence_ms,
            speech_pad_ms=speech_pad_ms,
        )
        # ── Semantic end-of-turn (Smart Turn v3) ──────────────────────────────
        # When enabled, silero just detects the *pause* (use a short silence so we
        # ask quickly); Smart Turn decides whether the pause is really end-of-turn
        # or just a breath. _turn_carry accumulates segments across held pauses so
        # the detector always sees the whole turn-so-far.
        self.turn_detector = None
        self.turn_max_s = float(turn_max_s)
        # Grace fallback: after an "incomplete" verdict, if the user goes silent
        # this long they're done — finalize so a held turn can never hang waiting
        # for speech that never comes. _hold_deadline is a wall-clock time set when
        # we hold; force-finalize once it passes with no resumed speech.
        self.turn_hold_grace_s = max(0.0, turn_hold_grace_ms / 1000.0)
        self._hold_deadline: float | None = None
        self._turn_carry: list[np.ndarray] = []
        if (turn_detection or "vad").strip().lower() == "smart_turn":
            # A shared, pre-warmed detector (loaded once by the factory) avoids the
            # per-session ONNX load on the connect path. It's stateless — is_complete()
            # only reads the session/feature-extractor — so all sessions can share one.
            detector = turn_detector
            if detector is None:
                from speech_to_speech.VAD.smart_turn import SmartTurnDetector

                detector = SmartTurnDetector(smart_turn_model_path, threshold=turn_threshold, sample_rate=sample_rate)
            if detector.available:
                self.turn_detector = detector
                # Finalize on a short pause; Smart Turn makes the real end-of-turn call.
                self.iterator.min_silence_samples = int(sample_rate * turn_min_silence_ms / 1000)
                logger.info(
                    "Semantic end-of-turn ENABLED (Smart Turn v3): pause=%dms, hold_grace=%.0fms, "
                    "max_turn=%.0fs, threshold=%.2f",
                    turn_min_silence_ms, self.turn_hold_grace_s * 1000, self.turn_max_s, turn_threshold,
                )
            else:
                logger.warning("turn_detection=smart_turn requested but model unavailable — using VAD silence timing")

        self.audio_enhancement = audio_enhancement
        if audio_enhancement:
            if not HAS_DF:
                logger.error(
                    "Audio enhancement requested but DeepFilterNet is not available. Disabling audio enhancement."
                )
                self.audio_enhancement = False
            else:
                self.enhanced_model, self.df_state, _ = init_df()

        # State for progressive audio release
        self.last_process_time: float = 0.0

        # Cumulative sample counter for audio_start_ms / audio_end_ms
        self._total_samples: int = 0

        # Throttled logging state (summary once per second)
        self._last_log_time = 0.0
        self._log_chunks = 0
        # Diagnostic heartbeat: count EVERY chunk arriving at the VAD (before the
        # should_listen gate) so we can tell, during a TTS response, whether the
        # client is still streaming mic audio and whether listening is enabled.
        # This distinguishes "no input reached us" (AVA gated/stopped forwarding)
        # from "input arrived but should_listen was clear" (s2s-side gate bug).
        self._hb_chunks_received = 0
        self._hb_gated = 0  # chunks suppressed by the RMS gate this heartbeat window
        self._hb_far = 0  # chunks seen while the agent was speaking (residual gate active)
        self._hb_pass_max = 0.0  # loudest gated-signal RMS that cleared the gate this window
        self._hb_last_time = 0.0
        self._log_speech_starts = 0
        self._log_speech_ends = 0
        self._log_progressive_yields = 0
        self._speech_started_emitted = False

    @property
    def _audio_ms(self) -> int:
        """Cumulative audio received so far, in milliseconds."""
        return int(self._total_samples / self.sample_rate * 1000)

    def _apply_runtime_turn_detection(self, runtime_config: RuntimeConfig | None = None) -> None:
        """Check RuntimeConfig for turn_detection changes and apply them."""
        audio = runtime_config.session.audio if runtime_config else None
        audio_input = audio.input if audio is not None else None
        if not runtime_config or not audio_input or not audio_input.turn_detection:
            return
        td_raw = audio_input.turn_detection

        # Convert Pydantic models (e.g. OpenAI SDK ServerVad) to dict;
        # plain dicts pass through unchanged.
        if hasattr(td_raw, "model_dump"):
            td = td_raw.model_dump(exclude_none=True)
        elif isinstance(td_raw, dict):
            td = td_raw
        else:
            logger.warning(f"Unexpected turn_detection type: {type(td_raw)}")
            return

        # Compare normalized snapshot (identity on td_raw vs stored dict was wrong after first apply).
        if td == self._last_turn_detection:
            return

        self._last_turn_detection = dict(td)

        if "threshold" in td:
            self.iterator.threshold = td["threshold"]
            logger.info(f"VAD threshold updated to {td['threshold']}")
        if "silence_duration_ms" in td:
            if self.turn_detector is not None:
                # Smart Turn owns end-of-turn; keep our short pause so the client's
                # (often long) silence_duration_ms doesn't make turn-taking laggy.
                logger.debug(
                    "Ignoring client silence_duration_ms=%sms (Smart Turn semantic end-of-turn active)",
                    td["silence_duration_ms"],
                )
            else:
                self.iterator.min_silence_samples = self.sample_rate * td["silence_duration_ms"] / 1000
                logger.info(f"VAD silence duration updated to {td['silence_duration_ms']}ms")

    def process(self, audio_chunk: VADIn) -> Iterator[VADOut]:
        runtime_config = None
        detect_chunk = None  # echo-cancelled audio for the VAD decision (may differ from raw)
        far_active = False   # agent currently producing audio (echo present) → residual gate
        if isinstance(audio_chunk, tuple):
            if len(audio_chunk) == 4:
                # (raw, cleaned, far_active, runtime_config)
                audio_chunk, detect_chunk, far_active, runtime_config = audio_chunk
            elif len(audio_chunk) == 3:
                # (raw_bytes, cleaned_bytes, runtime_config): detect on cleaned,
                # but buffer/transcribe raw (AEC over-suppresses the user during
                # double-talk, so cleaned is bad for STT).
                audio_chunk, detect_chunk, runtime_config = audio_chunk
            else:
                audio_chunk, runtime_config = audio_chunk
        if detect_chunk is None:
            detect_chunk = audio_chunk
        self._apply_runtime_turn_detection(runtime_config)

        # Heartbeat BEFORE the gate: every chunk that reaches the VAD is counted,
        # regardless of should_listen. If chunks/s stays >0 during a TTS response,
        # the client is still forwarding mic audio (barge-in input is arriving);
        # if it drops to 0, the client gated it. should_listen tells us whether
        # the VAD would even act on it. Gated behind DEBUG_MODE (off by default).
        if DEBUG_MODE:
            self._hb_chunks_received += 1
            if far_active:
                self._hb_far += 1
            hb_now = time.time()
            if hb_now - self._hb_last_time >= 1.0:
                # far>0 means maxpass reflects the RESIDUAL (tune VAD_INPUT_RMS_GATE_FAR);
                # far=0 means maxpass reflects RAW (tune VAD_INPUT_RMS_GATE).
                logger.info(
                    "VAD heartbeat: %d chunks/s in | gated=%d far=%d maxpass=%.0f | should_listen=%s | triggered=%s",
                    self._hb_chunks_received,
                    self._hb_gated,
                    self._hb_far,
                    self._hb_pass_max,
                    self.should_listen.is_set(),
                    self.iterator.triggered,
                )
                self._hb_chunks_received = 0
                self._hb_gated = 0
                self._hb_far = 0
                self._hb_pass_max = 0.0
                self._hb_last_time = hb_now

        if not self.should_listen.is_set():
            return

        # Smart Turn hold fallback: we're holding an "incomplete" turn. If the user
        # resumed speaking, cancel the fallback (the normal pause path will re-check).
        # If they've stayed silent past the grace window, they're done — finalize the
        # held turn now so it can't hang waiting for a pause that never comes.
        if self._turn_carry and self._hold_deadline is not None:
            if self.iterator.triggered:
                self._hold_deadline = None
            elif time.time() >= self._hold_deadline:
                yield from self._force_finalize_turn()
                return

        # Normal listening mode
        self._log_chunks += 1
        raw_int16 = np.frombuffer(audio_chunk, dtype=np.int16)        # buffered for STT
        detect_int16 = np.frombuffer(detect_chunk, dtype=np.int16)    # scored by silero
        self._total_samples += len(raw_int16)
        detect_float32 = int2float(detect_int16)
        raw_float32 = int2float(raw_int16)

        # Far-aware RMS gate. The signal we gate on depends on whether the agent
        # is currently producing audio (echo present):
        #  • far INACTIVE (agent silent): gate on RAW mic energy — it cleanly
        #    separates speech from silence, and there's no echo to confuse it.
        #  • far ACTIVE (agent speaking): raw echo can exceed the raw gate at high
        #    speaker volume, so gate on the AEC *residual* instead. Echo cancels to
        #    a low residual, but the user's voice survives AEC (it isn't in the far
        #    reference), so a real barge-in leaves a much higher residual than echo.
        # Below the gate we present silence to silero (zero the cleaned chunk); the
        # raw audio stored for STT is never touched. `maxpass` logs the loudest
        # above-threshold chunk for tuning.
        if far_active:
            gate_rms = float(np.sqrt(np.mean(detect_int16.astype(np.float32) ** 2))) if detect_int16.size else 0.0
            gate_threshold = self.input_rms_gate_far
        else:
            self._far_run.clear()
            gate_rms = float(np.sqrt(np.mean(raw_int16.astype(np.float32) ** 2))) if raw_int16.size else 0.0
            gate_threshold = self.input_rms_gate
        above = (gate_threshold <= 0) or (gate_rms >= gate_threshold)

        # Sustained-energy requirement during playback: echo-residual leaks are
        # sparse TRANSIENT spikes (a level gate alone can't tell a 533 echo spike
        # from a 533 speech onset), but real speech is DENSE — most chunks above
        # the gate. So while far is active, a chunk only passes if the energy has
        # been sustained (>= _far_sustain_min above-gate chunks in the recent
        # window). A brief spike that clears the level gate but isn't sustained is
        # rejected as echo. Observed: phantom ~5/32 chunks (sparse) vs real
        # barge-in ~8/9 (dense).
        if far_active and gate_threshold > 0:
            self._far_run.append(1 if above else 0)
            passed = above and (sum(self._far_run) >= self._far_sustain_min)
        else:
            passed = above

        if not passed:
            detect_float32 = np.zeros_like(detect_float32)
            if DEBUG_MODE:
                self._hb_gated += 1
        if DEBUG_MODE and above:
            # ceiling of above-threshold chunks (residual when far, raw otherwise),
            # so the level gate stays tunable even when a spike is sustain-rejected.
            self._hb_pass_max = max(self._hb_pass_max, gate_rms)

        # Score on the (gated) cleaned signal; buffer the raw signal for STT.
        vad_output = self.iterator(torch.from_numpy(detect_float32), store=torch.from_numpy(raw_float32))

        # Deferred speech_started: only emit once buffer >= min_speech_ms
        is_triggered_now = self.iterator.triggered
        if is_triggered_now and not self._speech_started_emitted:
            buffer_samples = sum(len(t) for t in self.iterator.buffer)
            buffer_duration_ms = buffer_samples / self.sample_rate * 1000
            if buffer_duration_ms >= self.min_speech_ms:
                self._speech_started_emitted = True
                self._log_speech_starts += 1
                start_ms = max(0, self._audio_ms - int(buffer_duration_ms))
                logger.info("Speech started (confirmed, %.0fms buffered)", buffer_duration_ms)
                if self.text_output_queue:
                    self.text_output_queue.put(SpeechStartedEvent(audio_start_ms=start_ms))

        # Log a summary once per second instead of every chunk
        now = time.time()
        if now - self._last_log_time >= 1.0:
            state = "SPEAKING" if is_triggered_now else "silent"
            logger.debug(
                f"VAD: {self._log_chunks} chunks/s | {state} | "
                f"starts={self._log_speech_starts} ends={self._log_speech_ends} progressive={self._log_progressive_yields}"
            )
            self._log_chunks = 0
            self._log_speech_starts = 0
            self._log_speech_ends = 0
            self._log_progressive_yields = 0
            self._last_log_time = now

        if self.enable_realtime_transcription:
            # Progressive mode: yield audio chunks while speaking
            yield from self._process_realtime(vad_output)
        else:
            # Original mode: yield only when speech ends
            yield from self._process_normal(vad_output)

    def _should_end_turn(self, array: np.ndarray) -> tuple[bool, np.ndarray]:
        """Decide whether a silero-detected pause is really end-of-turn.

        Returns (end_turn, audio_to_emit). With no detector this is a pass-through
        (always end, original audio = legacy VAD behaviour). With Smart Turn, the
        segment is accumulated into the running turn and the detector scores the
        whole turn-so-far: complete (or max-turn ceiling) → end and emit the full
        turn; incomplete → keep listening (caller holds the mic open).
        """
        if self.turn_detector is None:
            return True, array

        self._turn_carry.append(array)
        combined = np.concatenate(self._turn_carry) if len(self._turn_carry) > 1 else self._turn_carry[0]
        dur_s = len(combined) / self.sample_rate
        if dur_s >= self.turn_max_s:
            logger.info("Smart Turn: max turn length %.1fs reached — ending turn", dur_s)
            self._turn_carry = []
            return True, combined

        complete, prob = self.turn_detector.is_complete(combined)
        if complete:
            if DEBUG_MODE:
                logger.info("Smart Turn: complete (p=%.2f, %.1fs) — ending turn", prob, dur_s)
            self._turn_carry = []
            return True, combined
        logger.info("Smart Turn: incomplete (p=%.2f, %.1fs) — pause, keep listening", prob, dur_s)
        return False, combined

    def _finalize_emit(self, array: np.ndarray) -> Iterator[VADOut]:
        """Common end-of-turn side effects + emit: SpeechStarted (if not already),
        stop listening, SpeechStopped, optional enhancement, yield the utterance."""
        duration_ms = len(array) / self.sample_rate * 1000
        end_ms = self._audio_ms
        if not self._speech_started_emitted and self.text_output_queue:
            self.text_output_queue.put(SpeechStartedEvent(audio_start_ms=max(0, end_ms - int(duration_ms))))
        self._log_speech_ends += 1
        self.should_listen.clear()
        self._hold_deadline = None
        logger.info(f"Speech ended ({duration_ms:.0f}ms), stop listening")
        if self.text_output_queue:
            self.text_output_queue.put(SpeechStoppedEvent(duration_s=duration_ms / 1000.0, audio_end_ms=end_ms))
        if self.audio_enhancement:
            array = self._apply_audio_enhancement(array)
        if self.enable_realtime_transcription:
            yield VADAudio(audio=array, mode="final")
        else:
            yield VADAudio(audio=array)
        self.last_process_time = 0.0
        self._speech_started_emitted = False

    def _force_finalize_turn(self) -> Iterator[VADOut]:
        """End a held (incomplete) turn after the grace window — the user went
        silent, so emit the accumulated turn rather than wait forever."""
        if not self._turn_carry:
            self._hold_deadline = None
            return
        array = np.concatenate(self._turn_carry) if len(self._turn_carry) > 1 else self._turn_carry[0]
        self._turn_carry = []
        logger.info(
            "Smart Turn: %.0fms silence after 'incomplete' — user done, finalizing held turn (%.1fs)",
            self.turn_hold_grace_s * 1000, len(array) / self.sample_rate,
        )
        self.iterator.reset_states()
        yield from self._finalize_emit(array)

    def _process_realtime(self, vad_output: list[torch.Tensor] | None) -> Iterator[VADOut]:
        """Process with real-time progressive audio release."""
        # Check if we're currently in a speech segment
        if hasattr(self.iterator, "buffer") and len(self.iterator.buffer) > 0:
            current_time = time.time()

            # Yield accumulated audio periodically while speaking
            if (current_time - self.last_process_time) >= self.realtime_processing_pause:
                array = torch.cat(self.iterator.speech_buffer()).cpu().numpy()
                duration_ms = len(array) / self.sample_rate * 1000

                if duration_ms >= self.min_speech_ms:
                    self._log_progressive_yields += 1
                    logger.debug(f"VAD: yielding progressive audio ({duration_ms:.0f}ms)")
                    yield VADAudio(audio=array, mode="progressive")
                    self.last_process_time = current_time

        # Handle end of speech
        if vad_output is not None:
            if len(vad_output) == 0:
                logger.info("VAD: phantom trigger (empty buffer), closing speech pair")
                if self._speech_started_emitted and self.text_output_queue:
                    self.text_output_queue.put(SpeechStoppedEvent(audio_end_ms=self._audio_ms))
                self._speech_started_emitted = False
                return

            array = torch.cat(vad_output).cpu().numpy()
            duration_ms = len(array) / self.sample_rate * 1000

            if duration_ms < self.min_speech_ms or duration_ms > self.max_speech_ms:
                logger.info(
                    f"VAD: discarding {duration_ms:.0f}ms segment (bounds: {self.min_speech_ms}-{self.max_speech_ms}ms)"
                )
                if self._speech_started_emitted and self.text_output_queue:
                    self.text_output_queue.put(SpeechStoppedEvent(audio_end_ms=self._audio_ms))
                self._speech_started_emitted = False
            else:
                end_turn, array = self._should_end_turn(array)
                if not end_turn:
                    # Semantic detector: the user only paused — hold the mic open.
                    # Arm the grace fallback so the turn can't hang if they stop, and
                    # keep _speech_started_emitted so the whole held turn is one
                    # Start/Stop bracket (no duplicate SpeechStarted per segment).
                    self._hold_deadline = time.time() + self.turn_hold_grace_s
                    self.last_process_time = 0.0
                    return
                yield from self._finalize_emit(array)

    def _process_normal(self, vad_output: list[torch.Tensor] | None) -> Iterator[VADOut]:
        """Original processing: yield only when speech ends."""
        if vad_output is not None:
            if len(vad_output) == 0:
                logger.info("VAD: phantom trigger (empty buffer), closing speech pair")
                if self._speech_started_emitted and self.text_output_queue:
                    self.text_output_queue.put(SpeechStoppedEvent(audio_end_ms=self._audio_ms))
                self._speech_started_emitted = False
                return

            array = torch.cat(vad_output).cpu().numpy()
            duration_ms = len(array) / self.sample_rate * 1000
            if duration_ms < self.min_speech_ms or duration_ms > self.max_speech_ms:
                logger.info(
                    f"VAD: discarding {duration_ms:.0f}ms segment (bounds: {self.min_speech_ms}-{self.max_speech_ms}ms)"
                )
                if self._speech_started_emitted and self.text_output_queue:
                    self.text_output_queue.put(SpeechStoppedEvent(audio_end_ms=self._audio_ms))
                self._speech_started_emitted = False
            else:
                end_turn, array = self._should_end_turn(array)
                if not end_turn:
                    # Semantic detector: the user only paused — hold the mic open
                    # and arm the grace fallback so a held turn can't hang.
                    self._hold_deadline = time.time() + self.turn_hold_grace_s
                    return
                yield from self._finalize_emit(array)

    def _apply_audio_enhancement(self, array: np.ndarray) -> np.ndarray:
        """Apply audio enhancement if enabled."""
        import torchaudio

        if self.sample_rate != self.df_state.sr():
            audio_float32 = torchaudio.functional.resample(
                torch.from_numpy(array),
                orig_freq=self.sample_rate,
                new_freq=self.df_state.sr(),
            )
            enhanced = enhance(
                self.enhanced_model,
                self.df_state,
                audio_float32.unsqueeze(0),
            )
            enhanced = torchaudio.functional.resample(
                enhanced,
                orig_freq=self.df_state.sr(),
                new_freq=self.sample_rate,
            )
        else:
            enhanced = enhance(self.enhanced_model, self.df_state, torch.from_numpy(array))
        return enhanced.numpy().squeeze()

    def on_session_end(self):
        self.iterator.reset_states()
        self.iterator.buffer = []
        self.last_process_time = 0.0
        self._total_samples = 0
        self._speech_started_emitted = False
        self._turn_carry = []
        self._hold_deadline = None
        self._far_run.clear()
        self.should_listen.set()
        logger.debug("VAD session state reset")

    @property
    def min_time_to_debug(self) -> float:
        return 0.00001
