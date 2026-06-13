# Copyright 2026 winoiknow (Eric Alborn, Anteon Group)
# Licensed under the Apache License, Version 2.0 (the "License").
# See the LICENSE file in the repository root for the full license text.
#
# Multi-session Phase A: extract the per-connection pipeline into a single
# owning object. This is a pure refactor — `HandlerFactory.build()` reproduces
# the old `build_pipeline()` byte-for-byte. The point is that everything
# dangerous to share across sessions (queues, events, cancel scope, the six
# handler threads) now lives on one `SessionPipeline` instance, so later phases
# can build/tear one down per WebSocket connection without disturbing this code.

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from queue import Queue
from threading import Event
from typing import Any

from openai.types.realtime import RealtimeSessionCreateRequest

from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.arguments_classes.speaker_id_arguments import SpeakerIdHandlerArguments
from speech_to_speech.audio.echo_canceller import EchoCanceller
from speech_to_speech.LLM.chat import Chat
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.control import SESSION_END
from speech_to_speech.pipeline.messages import PIPELINE_END
from speech_to_speech.pipeline.queue_types import (
    AudioInItem,
    AudioOutItem,
    LMOutItem,
    STTOutItem,
    TextEventItem,
    TextPromptItem,
    TTSInItem,
    VADOutItem,
)
from speech_to_speech.s2s_pipeline import (
    ParsedArguments,
    get_llm_handler,
    get_stt_handler,
    get_tts_handler,
    initialize_queues_and_events,
)
from speech_to_speech.STT.transcription_notifier import TranscriptionNotifier
from speech_to_speech.utils.thread_manager import ThreadManager
from speech_to_speech.utils.utils import short_sid
from speech_to_speech.VAD.vad_handler import VADHandler

logger = logging.getLogger(__name__)

# Pipeline-internal PCM rate the EchoCanceller operates at (mirrors the router).
PIPELINE_SAMPLE_RATE = 16000
# Acoustic echo cancellation on the input path, per session. Off by default.
# AEC_FILTER_LENGTH_MS should cover the echo round-trip delay (small on
# LAN/browser; larger with a client jitter buffer); it applies to the speex
# backend only. Backend: "aec3" (WebRTC AEC3, delay-estimating, default) or
# "speex" (libspeexdsp adaptive filter).
AEC_ENABLED = os.getenv("AEC_ENABLED", "0").lower() in ("1", "true", "on", "yes")
AEC_BACKEND = os.getenv("AEC_BACKEND", "aec3")
AEC_FILTER_LENGTH_MS = int(os.getenv("AEC_FILTER_LENGTH_MS", "250"))

# Concurrent realtime session cap. 1 ⇒ exactly today's single-session semantics.
# Raise deliberately after load-testing the shared STT/TTS/LLM endpoints. Forced
# to 1 with a warning when an in-process (local) model is selected — multi-session
# with local models means putting the model behind a serving endpoint instead.
S2S_MAX_SESSIONS = max(1, int(os.getenv("S2S_MAX_SESSIONS", "1")))

# Backends with no in-process model state (safe to run N sessions in one process).
# Anything else loads a model into this process and is pinned to one session.
_REMOTE_STT = {"openai-remote"}
_REMOTE_TTS = {"openai-remote", "elevenlabs", "minimax"}
_REMOTE_LLM = {"responses-api"}


def local_model_selected(module_kwargs: Any) -> str | None:
    """Return a human description of the first in-process model selected, or None
    if STT, TTS and the LLM are all remote/cloud (i.e. multi-session is safe)."""
    if module_kwargs.stt not in _REMOTE_STT:
        return f"STT={module_kwargs.stt}"
    if module_kwargs.tts not in _REMOTE_TTS:
        return f"TTS={module_kwargs.tts}"
    if module_kwargs.llm_backend not in _REMOTE_LLM:
        return f"LLM={module_kwargs.llm_backend}"
    return None


@dataclass
class SessionPipeline:
    """Everything one connection owns.

    Phase A builds exactly one of these at startup and runs it for the whole
    process lifetime (single-session semantics, unchanged). Phase B moves
    construction to WS-connect and teardown to disconnect; the field set here is
    deliberately the per-session unit of isolation called out in
    MULTI_SESSION_PLAN.md §3.1.
    """

    session_id: str

    # ── Queues (all fresh per session) ───────────────────────────────
    recv_audio: Queue[AudioInItem]
    spoken_prompt: Queue[VADOutItem]
    stt_output: Queue[STTOutItem]
    text_prompt: Queue[TextPromptItem]
    lm_response: Queue[LMOutItem]
    lm_processed: Queue[TTSInItem]
    send_audio: Queue[AudioOutItem]
    # Only populated in websocket/realtime modes; None elsewhere to avoid
    # unbounded growth (no consumer drains it in local/socket modes).
    text_output: Queue[TextEventItem] | None

    # ── Control primitives (all fresh per session) ───────────────────
    stop_event: Event
    should_listen: Event
    response_playing: Event
    cancel_scope: CancelScope
    # Far-end (agent TTS) reference buffer for AEC; near-end mic is cleaned
    # against it before the VAD decision. One per session (one call).
    echo_canceller: EchoCanceller

    # ── Handler threads ──────────────────────────────────────────────
    handlers: list[Any]
    threads: ThreadManager

    # ── Per-session asyncio send task (set by the router at connect) ──
    send_task: asyncio.Task | None = field(default=None)

    def start(self) -> None:
        self.threads.start()

    def wait(self) -> None:
        self.threads.wait()

    def stop(self) -> None:
        self.threads.stop()

    def shutdown(self, timeout: float = 5.0) -> None:
        """Tear down all handler threads. Safe to call from a worker thread.

        The handler chain is strictly linear (recv_audio → vad → … → tts →
        send_audio), so a single ``PIPELINE_END`` on ``recv_audio`` breaks each
        thread's loop and is forwarded downstream on cleanup, cascading through
        the whole chain. ``SESSION_END`` first runs each handler's
        ``on_session_end`` (closing per-session resources); ``stop_event`` is a
        belt-and-suspenders wake for any handler idling on its 0.1 s get-timeout.
        """
        self.recv_audio.put(SESSION_END)
        self.recv_audio.put(PIPELINE_END)
        self.stop_event.set()
        for thread in self.threads.threads:
            if thread.is_alive():
                thread.join(timeout=timeout)
                if thread.is_alive():
                    logger.warning(
                        "session %s: thread %s did not terminate within %.1fs",
                        self.session_id,
                        thread.name,
                        timeout,
                    )


class HandlerFactory:
    """Parsed CLI/env config captured once at startup, plus the read-only state
    shared across sessions (speaker-id client).

    ``build()`` produces a fully wired :class:`SessionPipeline`. In Phase A it is
    called exactly once; the per-session handler dispatchers it calls
    (``get_stt_handler`` etc.) already take queues as parameters, so making them
    callable per-session is plumbing, not logic.
    """

    def __init__(self, args: ParsedArguments) -> None:
        self.args = args

        # ── Speaker-id client (Phase 3 identify + Phase 4 diarize), env-backed,
        # off by default. Built once and shared (read-only HTTP client) across
        # both the STT handler and the realtime service. Flag(s) off → None →
        # byte-for-byte today's pipeline.
        self.speaker_args = SpeakerIdHandlerArguments()
        self.speaker_client = None
        if self.speaker_args.speaker_id_enabled or self.speaker_args.speaker_diarize_enabled:
            from speech_to_speech.speaker_id.remote_speaker_client import RemoteSpeakerClient

            self.speaker_client = RemoteSpeakerClient(
                self.speaker_args.speaker_id_base_url,
                self.speaker_args.speaker_id_api_key,
                self.speaker_args.speaker_id_timeout,
                diarize_timeout=self.speaker_args.speaker_diarize_timeout,
            )
            logger.info(
                "Speaker-id client → %s (identify=%s, diarize=%s)",
                self.speaker_args.speaker_id_base_url,
                self.speaker_args.speaker_id_enabled,
                self.speaker_args.speaker_diarize_enabled,
            )

    def build(self, session_id: str = "default") -> SessionPipeline:
        args = self.args
        module_kwargs = args.module_kwargs
        speaker_args = self.speaker_args
        speaker_client = self.speaker_client

        qe = initialize_queues_and_events()
        stop_event: Event = qe["stop_event"]
        should_listen: Event = qe["should_listen"]
        response_playing: Event = qe["response_playing"]
        cancel_scope: CancelScope = qe["cancel_scope"]
        recv_audio_chunks_queue: Queue[AudioInItem] = qe["recv_audio_chunks_queue"]
        send_audio_chunks_queue: Queue[AudioOutItem] = qe["send_audio_chunks_queue"]
        spoken_prompt_queue: Queue[VADOutItem] = qe["spoken_prompt_queue"]
        stt_output_queue: Queue[STTOutItem] = qe["stt_output_queue"]
        text_prompt_queue: Queue[TextPromptItem] = qe["text_prompt_queue"]
        lm_response_queue: Queue[LMOutItem] = qe["lm_response_queue"]
        lm_processed_queue: Queue[TTSInItem] = qe["lm_processed_queue"]
        # Only set for websocket/realtime modes; kept None otherwise to avoid
        # unbounded queue growth.
        text_output_queue: Queue[TextEventItem] | None = None

        comms_handlers: list[Any] = []
        if module_kwargs.mode == "local":
            from speech_to_speech.connections.local_audio_streamer import LocalAudioStreamer

            local_audio_streamer = LocalAudioStreamer(
                input_queue=recv_audio_chunks_queue,
                output_queue=send_audio_chunks_queue,
                should_listen=should_listen,
            )
            comms_handlers = [local_audio_streamer]
            should_listen.set()
        elif module_kwargs.mode == "websocket":
            from speech_to_speech.connections.websocket_streamer import WebSocketStreamer

            text_output_queue = qe["text_output_queue"]
            websocket_streamer = WebSocketStreamer(
                stop_event,
                input_queue=recv_audio_chunks_queue,
                output_queue=send_audio_chunks_queue,
                should_listen=should_listen,
                text_output_queue=text_output_queue,
                host=args.websocket_streamer_kwargs.ws_host,
                port=args.websocket_streamer_kwargs.ws_port,
            )
            comms_handlers = [websocket_streamer]
        elif module_kwargs.mode == "realtime":
            from speech_to_speech.api.openai_realtime.server import RealtimeServer

            text_output_queue = qe["text_output_queue"]

            vars(args.vad_handler_kwargs)["text_output_queue"] = text_output_queue

            for kw in (
                args.language_model_handler_kwargs,
                args.responses_api_language_model_handler_kwargs,
                args.kokoro_tts_handler_kwargs,
                args.qwen3_tts_handler_kwargs,
                args.pocket_tts_handler_kwargs,
                args.chat_tts_handler_kwargs,
                args.facebook_mms_tts_handler_kwargs,
                args.remote_openai_tts_handler_kwargs,
                args.elevenlabs_tts_handler_kwargs,
                args.minimax_tts_handler_kwargs,
            ):
                vars(kw)["cancel_scope"] = cancel_scope

            if module_kwargs.llm_backend == "responses-api":
                chat_size = vars(args.responses_api_language_model_handler_kwargs).get("chat_size", 10)
            else:
                chat_size = vars(args.language_model_handler_kwargs).get("chat_size", 10)

            realtime_conn = RealtimeServer(
                stop_event,
                input_queue=recv_audio_chunks_queue,
                output_queue=send_audio_chunks_queue,
                should_listen=should_listen,
                response_playing=response_playing,
                cancel_scope=cancel_scope,
                text_output_queue=text_output_queue,
                text_prompt_queue=text_prompt_queue,
                host=args.websocket_streamer_kwargs.ws_host,
                port=args.websocket_streamer_kwargs.ws_port,
                chat_size=chat_size,
                server_api_key=module_kwargs.server_api_key,
                speaker_client=speaker_client,
                speaker_diarize_enabled=speaker_args.speaker_diarize_enabled,
            )
            comms_handlers = [realtime_conn]
        else:
            from speech_to_speech.connections.socket_receiver import SocketReceiver
            from speech_to_speech.connections.socket_sender import SocketSender

            comms_handlers = [
                SocketReceiver(
                    stop_event,
                    recv_audio_chunks_queue,
                    should_listen,
                    host=args.socket_receiver_kwargs.recv_host,
                    port=args.socket_receiver_kwargs.recv_port,
                    chunk_size=args.socket_receiver_kwargs.chunk_size,
                ),
                SocketSender(
                    stop_event,
                    send_audio_chunks_queue,
                    should_listen,
                    host=args.socket_sender_kwargs.send_host,
                    port=args.socket_sender_kwargs.send_port,
                ),
            ]

        # Set VAD realtime transcription parameters from module_kwargs
        if module_kwargs.enable_live_transcription:
            args.vad_handler_kwargs.enable_realtime_transcription = True
            args.vad_handler_kwargs.realtime_processing_pause = module_kwargs.live_transcription_update_interval

        vad = VADHandler(
            stop_event,
            queue_in=recv_audio_chunks_queue,
            queue_out=spoken_prompt_queue,
            setup_args=(should_listen,),
            setup_kwargs=vars(args.vad_handler_kwargs),
        )

        # ── Speaker-id notifier wiring (client built in __init__). Inline
        # identify labels only when SPEAKER_ID_ENABLED; diarize-only deployments
        # add no inline prefix.
        transcription_notifier_kwargs: dict[str, Any] = {
            "text_output_queue": text_output_queue,
            "should_listen": should_listen,
            "label_format": speaker_args.speaker_id_label_format if speaker_args.speaker_id_enabled else "",
        }
        if module_kwargs.mode != "realtime":
            if module_kwargs.llm_backend == "responses-api":
                _lm_vars = vars(args.responses_api_language_model_handler_kwargs)
            else:
                _lm_vars = vars(args.language_model_handler_kwargs)
            transcription_notifier_kwargs["runtime_config"] = RuntimeConfig(
                chat=Chat(_lm_vars.get("chat_size", 30)),
                session=RealtimeSessionCreateRequest(
                    type="realtime",
                    instructions=_lm_vars.get("init_chat_prompt"),
                ),
            )

        transcription_notifier = TranscriptionNotifier(
            stop_event,
            queue_in=stt_output_queue,
            queue_out=text_prompt_queue,
            setup_kwargs=transcription_notifier_kwargs,
        )

        stt = get_stt_handler(
            module_kwargs,
            stop_event,
            spoken_prompt_queue,
            stt_output_queue,
            args.whisper_stt_handler_kwargs,
            args.faster_whisper_stt_handler_kwargs,
            args.paraformer_stt_handler_kwargs,
            args.mlx_audio_whisper_stt_handler_kwargs,
            args.parakeet_tdt_stt_handler_kwargs,
            args.remote_openai_stt_handler_kwargs,
            # Concurrent identify only when SPEAKER_ID_ENABLED; a diarize-only
            # deploy still attaches the turn audio (diarize_enabled) but runs no
            # per-turn identify.
            speaker_client=speaker_client if speaker_args.speaker_id_enabled else None,
            speaker_timeout=speaker_args.speaker_id_timeout,
            diarize_enabled=speaker_args.speaker_diarize_enabled,
        )

        lm = get_llm_handler(
            module_kwargs,
            stop_event,
            text_prompt_queue,
            lm_response_queue,
            args.language_model_handler_kwargs,
            args.responses_api_language_model_handler_kwargs,
        )

        # LM output processor extracts tools and forwards clean text to TTS.
        from speech_to_speech.LLM.lm_output_processor import LMOutputProcessor

        lm_processor = LMOutputProcessor(
            stop_event,
            queue_in=lm_response_queue,
            queue_out=lm_processed_queue,
            setup_kwargs={"text_output_queue": text_output_queue},
        )

        tts = get_tts_handler(
            module_kwargs,
            stop_event,
            lm_processed_queue,
            send_audio_chunks_queue,
            should_listen,
            args.chat_tts_handler_kwargs,
            args.facebook_mms_tts_handler_kwargs,
            args.pocket_tts_handler_kwargs,
            args.kokoro_tts_handler_kwargs,
            args.qwen3_tts_handler_kwargs,
            args.remote_openai_tts_handler_kwargs,
            args.elevenlabs_tts_handler_kwargs,
            args.minimax_tts_handler_kwargs,
        )

        pipeline_handlers: list[Any] = [*comms_handlers, vad, stt, transcription_notifier, lm, lm_processor, tts]

        return SessionPipeline(
            session_id=session_id,
            recv_audio=recv_audio_chunks_queue,
            spoken_prompt=spoken_prompt_queue,
            stt_output=stt_output_queue,
            text_prompt=text_prompt_queue,
            lm_response=lm_response_queue,
            lm_processed=lm_processed_queue,
            send_audio=send_audio_chunks_queue,
            text_output=text_output_queue,
            stop_event=stop_event,
            should_listen=should_listen,
            response_playing=response_playing,
            cancel_scope=cancel_scope,
            # Non-realtime modes don't run the AEC input path; a disabled
            # canceller is a no-op passthrough that just satisfies the field.
            echo_canceller=EchoCanceller(sample_rate=PIPELINE_SAMPLE_RATE, enabled=False),
            handlers=pipeline_handlers,
            threads=ThreadManager(pipeline_handlers),
        )

    def build_session_pipeline(self, session_id: str) -> SessionPipeline:
        """Build one realtime session's pipeline: fresh queues/events/cancel
        scope/echo-canceller + the six core handlers, with **no** comms handler
        (the server is shared infra, built once by :meth:`build_realtime_server`).

        Called per WebSocket connect. The shared handler-arg objects are mutated
        with this session's ``cancel_scope`` / ``text_output`` queue right before
        the handlers are constructed; with the single-session guard still in
        place (Phase B) only one pipeline exists at a time, so this is safe.
        Per-session arg isolation lands with the guard lift in Phase C.
        """
        args = self.args
        module_kwargs = args.module_kwargs
        speaker_args = self.speaker_args
        speaker_client = self.speaker_client

        qe = initialize_queues_and_events()
        stop_event: Event = qe["stop_event"]
        should_listen: Event = qe["should_listen"]
        response_playing: Event = qe["response_playing"]
        cancel_scope: CancelScope = qe["cancel_scope"]
        recv_audio_chunks_queue: Queue[AudioInItem] = qe["recv_audio_chunks_queue"]
        send_audio_chunks_queue: Queue[AudioOutItem] = qe["send_audio_chunks_queue"]
        spoken_prompt_queue: Queue[VADOutItem] = qe["spoken_prompt_queue"]
        stt_output_queue: Queue[STTOutItem] = qe["stt_output_queue"]
        text_prompt_queue: Queue[TextPromptItem] = qe["text_prompt_queue"]
        lm_response_queue: Queue[LMOutItem] = qe["lm_response_queue"]
        lm_processed_queue: Queue[TTSInItem] = qe["lm_processed_queue"]
        text_output_queue: Queue[TextEventItem] = qe["text_output_queue"]

        # Realtime arg wiring: each TTS/LM handler reads cancel_scope from its
        # setup_kwargs; the VAD emits protocol events onto text_output.
        vars(args.vad_handler_kwargs)["text_output_queue"] = text_output_queue
        for kw in (
            args.language_model_handler_kwargs,
            args.responses_api_language_model_handler_kwargs,
            args.kokoro_tts_handler_kwargs,
            args.qwen3_tts_handler_kwargs,
            args.pocket_tts_handler_kwargs,
            args.chat_tts_handler_kwargs,
            args.facebook_mms_tts_handler_kwargs,
            args.remote_openai_tts_handler_kwargs,
            args.elevenlabs_tts_handler_kwargs,
            args.minimax_tts_handler_kwargs,
        ):
            vars(kw)["cancel_scope"] = cancel_scope

        if module_kwargs.enable_live_transcription:
            args.vad_handler_kwargs.enable_realtime_transcription = True
            args.vad_handler_kwargs.realtime_processing_pause = module_kwargs.live_transcription_update_interval

        vad = VADHandler(
            stop_event,
            queue_in=recv_audio_chunks_queue,
            queue_out=spoken_prompt_queue,
            setup_args=(should_listen,),
            setup_kwargs=vars(args.vad_handler_kwargs),
        )

        # Realtime path: inline identify labels only when SPEAKER_ID_ENABLED; no
        # runtime_config on the notifier (the service owns each session's chat).
        transcription_notifier = TranscriptionNotifier(
            stop_event,
            queue_in=stt_output_queue,
            queue_out=text_prompt_queue,
            setup_kwargs={
                "text_output_queue": text_output_queue,
                "should_listen": should_listen,
                "label_format": speaker_args.speaker_id_label_format if speaker_args.speaker_id_enabled else "",
            },
        )

        stt = get_stt_handler(
            module_kwargs,
            stop_event,
            spoken_prompt_queue,
            stt_output_queue,
            args.whisper_stt_handler_kwargs,
            args.faster_whisper_stt_handler_kwargs,
            args.paraformer_stt_handler_kwargs,
            args.mlx_audio_whisper_stt_handler_kwargs,
            args.parakeet_tdt_stt_handler_kwargs,
            args.remote_openai_stt_handler_kwargs,
            speaker_client=speaker_client if speaker_args.speaker_id_enabled else None,
            speaker_timeout=speaker_args.speaker_id_timeout,
            diarize_enabled=speaker_args.speaker_diarize_enabled,
        )

        lm = get_llm_handler(
            module_kwargs,
            stop_event,
            text_prompt_queue,
            lm_response_queue,
            args.language_model_handler_kwargs,
            args.responses_api_language_model_handler_kwargs,
        )

        from speech_to_speech.LLM.lm_output_processor import LMOutputProcessor

        lm_processor = LMOutputProcessor(
            stop_event,
            queue_in=lm_response_queue,
            queue_out=lm_processed_queue,
            setup_kwargs={"text_output_queue": text_output_queue},
        )

        tts = get_tts_handler(
            module_kwargs,
            stop_event,
            lm_processed_queue,
            send_audio_chunks_queue,
            should_listen,
            args.chat_tts_handler_kwargs,
            args.facebook_mms_tts_handler_kwargs,
            args.pocket_tts_handler_kwargs,
            args.kokoro_tts_handler_kwargs,
            args.qwen3_tts_handler_kwargs,
            args.remote_openai_tts_handler_kwargs,
            args.elevenlabs_tts_handler_kwargs,
            args.minimax_tts_handler_kwargs,
        )

        core_handlers: list[Any] = [vad, stt, transcription_notifier, lm, lm_processor, tts]

        return SessionPipeline(
            session_id=session_id,
            recv_audio=recv_audio_chunks_queue,
            spoken_prompt=spoken_prompt_queue,
            stt_output=stt_output_queue,
            text_prompt=text_prompt_queue,
            lm_response=lm_response_queue,
            lm_processed=lm_processed_queue,
            send_audio=send_audio_chunks_queue,
            text_output=text_output_queue,
            stop_event=stop_event,
            should_listen=should_listen,
            response_playing=response_playing,
            cancel_scope=cancel_scope,
            echo_canceller=EchoCanceller(
                sample_rate=PIPELINE_SAMPLE_RATE,
                filter_length_ms=AEC_FILTER_LENGTH_MS,
                enabled=AEC_ENABLED,
                backend=AEC_BACKEND,
            ),
            handlers=core_handlers,
            # daemon=True: a per-session handler is torn down at disconnect; if one
            # is briefly stuck in a blocking call it must never hold up process exit.
            # name_prefix tags every handler thread with this session's short id so
            # stack dumps / logs are attributable to the right session.
            threads=ThreadManager(core_handlers, daemon=True, name_prefix=short_sid(session_id)),
        )

    def effective_max_sessions(self) -> int:
        """The concurrent-session cap to enforce, forcing 1 when an in-process
        model is selected (multi-session needs the model behind a serving
        endpoint, not in this process)."""
        local = local_model_selected(self.args.module_kwargs)
        if local is not None and S2S_MAX_SESSIONS > 1:
            logger.warning(
                "S2S_MAX_SESSIONS=%d ignored: in-process model selected (%s). Forcing 1 — run the "
                "model behind a serving endpoint (openai-remote/elevenlabs/minimax/responses-api) "
                "to enable multi-session.",
                S2S_MAX_SESSIONS,
                local,
            )
            return 1
        return S2S_MAX_SESSIONS

    def build_realtime_server(self) -> Any:
        """Build the shared realtime server (uvicorn + FastAPI app). Per-session
        pipelines are built on connect by the app via ``self`` as the session
        factory, so this carries no queues — only server + service config."""
        from speech_to_speech.api.openai_realtime.server import RealtimeServer

        args = self.args
        module_kwargs = args.module_kwargs
        if module_kwargs.llm_backend == "responses-api":
            chat_size = vars(args.responses_api_language_model_handler_kwargs).get("chat_size", 10)
        else:
            chat_size = vars(args.language_model_handler_kwargs).get("chat_size", 10)

        max_sessions = self.effective_max_sessions()
        if max_sessions > 1:
            logger.info("Multi-session enabled: up to %d concurrent realtime sessions", max_sessions)

        return RealtimeServer(
            stop_event=Event(),
            session_factory=self,
            host=args.websocket_streamer_kwargs.ws_host,
            port=args.websocket_streamer_kwargs.ws_port,
            chat_size=chat_size,
            server_api_key=module_kwargs.server_api_key,
            speaker_client=self.speaker_client,
            speaker_diarize_enabled=self.speaker_args.speaker_diarize_enabled,
            max_sessions=max_sessions,
        )
