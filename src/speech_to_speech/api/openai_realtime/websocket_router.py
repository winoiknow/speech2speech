import asyncio
import base64
import logging
import os
import threading
import time
import wave
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from queue import Empty, Queue
from threading import Event as ThreadingEvent
from typing import TYPE_CHECKING, Any, Callable, Optional, TypeVar

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from openai.types.realtime import (
    ConversationItemCreateEvent,
    InputAudioBufferAppendEvent,
    InputAudioBufferCommitEvent,
    ResponseCancelEvent,
    ResponseCreateEvent,
    SessionUpdateEvent,
)

from speech_to_speech.api.openai_realtime.service import RealtimeService, ServerEvent
from speech_to_speech.debug import DEBUG_MODE
from speech_to_speech.pipeline.control import SESSION_END, PipelineControlMessage, is_control_message
from speech_to_speech.pipeline.events import (
    AssistantTextEvent,
    PipelineEvent,
    SpeechStartedEvent,
    SpeechStoppedEvent,
    TokenUsageEvent,
)
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, PIPELINE_END
from speech_to_speech.utils.utils import _generate_id

if TYPE_CHECKING:
    from speech_to_speech.pipeline.session_pipeline import HandlerFactory, SessionPipeline

logger = logging.getLogger(__name__)
QItem = TypeVar("QItem")

# Outbound audio on the queue is int16 PCM at the pipeline rate. Used to pace
# delta emission to wall-clock so we behave like the real OpenAI Realtime server.
PIPELINE_SAMPLE_RATE = 16000
PIPELINE_BYTES_PER_SAMPLE = 2
# Emit small deltas (~20 ms of pipeline audio) rather than big slugs. With the
# real-time pacer below, "send a batch then sleep its duration" leaves the client
# holding ~one batch of lead, which it reads as a faster-than-real provider rate
# (AVA measured 26.9 kHz vs the assumed 24 kHz with 192 ms batches → garbled
# playout). Small batches keep the measured rate ≈ the true rate, matching how the
# real OpenAI server streams audio.
MAX_AUDIO_BATCH_BYTES = PIPELINE_SAMPLE_RATE * PIPELINE_BYTES_PER_SAMPLE * 20 // 1000  # 20 ms = 640 bytes
# Real-time pacing of outbound audio. F5-TTS returns a whole clip at once, so
# without this s2s dumps the entire response in ~150 ms then sends response.done
# immediately; real-time consumers (e.g. AVA over AudioSocket) play ~one frame,
# then finalize on response.done and drop the rest. Pace so response.done lands
# after the audio. Disable with S2S_REALTIME_PACING=0.
REALTIME_PACING_ENABLED = os.getenv("S2S_REALTIME_PACING", "1").lower() not in ("0", "false", "no")
# Sleep granularity while pacing, so a client-initiated cancel (discarding) breaks out promptly.
PACING_SLICE_S = 0.01
# Hold response.done this long after the last audio so a real-time consumer can drain its
# playout buffer before finalizing (otherwise it drops the buffered tail). Env: S2S_RESPONSE_DONE_TAIL_MS.
RESPONSE_DONE_TAIL_S = int(os.getenv("S2S_RESPONSE_DONE_TAIL_MS", "400")) / 1000
# Keepalive during the silent "thinking" gap: while a response is in_progress and
# nothing has been sent to the client for this long (LLM thinking, agent tool loop,
# TTS synth), emit a {"type": "s2s.keepalive"} event so any client can distinguish
# a slow turn from a dead connection and refresh its turn watchdog. Clients ignore
# unknown event types per Realtime convention; set 0 to disable for strict clients.
HEARTBEAT_S = float(os.getenv("S2S_HEARTBEAT_S", "5"))
# Reap a session this many seconds after its last inbound client traffic. 0 (the
# default) never reaps — warm connections (smart speakers) hold idle sessions
# indefinitely. Deployments that want to reclaim abandoned sessions set a value.
S2S_IDLE_TIMEOUT_S = float(os.getenv("S2S_IDLE_TIMEOUT_S", "0"))
# How often the send loop checks that this session's handler threads are still
# alive (dead-thread supervisor). A crashed handler fails the session fast with a
# server_error rather than serving a live socket from a half-dead pipeline. 0
# disables the check.
S2S_THREAD_SUPERVISOR_S = float(os.getenv("S2S_THREAD_SUPERVISOR_S", "2"))
# Debug: if set, write each response's exact outbound audio (the base64 PCM deltas the client
# receives, decoded) to a WAV in this dir, so the real on-wire audio can be auditioned directly.
TTS_DUMP_DIR = os.getenv("S2S_TTS_DUMP_DIR") or None
# Sample rate of the dumped WAV — the negotiated client output rate (GA pcm = 24000).
TTS_DUMP_RATE = int(os.getenv("S2S_TTS_DUMP_RATE", "24000"))


async def _send_event(ws: WebSocket, event: ServerEvent) -> None:
    try:
        await ws.send_json(event.model_dump())
    except (WebSocketDisconnect, RuntimeError) as e:
        # Client already gone (socket closed before/while we sent). Starlette
        # raises RuntimeError once the connection is closed; either way the
        # receive loop will exit on the same disconnect. Benign — don't log an
        # error/traceback for a normal hangup.
        logger.debug("Client disconnected before event could be sent: %s", e)
    except Exception as e:
        logger.error(f"Failed to send event to client: {e}")


async def _send_events(ws: WebSocket, events: list[ServerEvent]) -> None:
    for event in events:
        await _send_event(ws, event)


def _keep_audio_sentinel(item: Any) -> bool:
    return isinstance(item, bytes) and item == AUDIO_RESPONSE_DONE


def _keep_user_text_event(item: Any) -> bool:
    return isinstance(item, (SpeechStoppedEvent, TokenUsageEvent))


def _shutdown_pipeline_async(session_id: str, pipeline: "SessionPipeline") -> threading.Thread:
    """Tear a session's pipeline down in a fire-and-forget daemon worker.

    Deliberately NOT ``loop.run_in_executor``: awaiting the executor future from
    the disconnect handler can hang if the event loop stops servicing the result
    callback mid-teardown. A plain daemon thread sidesteps asyncio entirely — it
    sets ``stop_event`` and drains the chain so the handler threads exit promptly.
    Returns the worker so callers (server shutdown) can join it if they want to
    wait. The handler threads are daemonised too, so nothing can wedge exit.
    """
    worker = threading.Thread(
        target=pipeline.shutdown, name=f"shutdown-{session_id[:8]}", daemon=True
    )
    worker.start()
    return worker


def create_app(
    service: RealtimeService,
    session_factory: "HandlerFactory",
    stop_event: ThreadingEvent,
    server_api_key: Optional[str] = None,
    max_sessions: int = 1,
) -> FastAPI:
    """Build the realtime FastAPI app.

    Holds no pipeline queues: each WebSocket connection builds its own
    :class:`SessionPipeline` via ``session_factory`` at connect time and tears it
    down at disconnect. ``session_factory`` only needs a
    ``build_session_pipeline(session_id) -> SessionPipeline`` method, so tests can
    pass a lightweight stub. Up to ``max_sessions`` connections are accepted
    concurrently; each is fully isolated (its own queues/events/cancel scope/AEC/
    send loop), so they share nothing mutable beyond the read-only service config.
    """

    def _flush_queue(q: Queue[QItem], *, preserve: Callable[[QItem], bool] | None = None) -> None:
        """Drain a queue, optionally preserving items matching *preserve*.

        Preserved items are re-inserted at the **front** of the queue
        (atomically under the queue's mutex) so they are processed before
        anything a pipeline thread may have enqueued during the drain.
        """
        preserved: list[QItem] = []
        while True:
            try:
                item = q.get_nowait()
                if preserve and preserve(item):
                    preserved.append(item)
            except Empty:
                break
        if preserved:
            with q.mutex:
                for item in reversed(preserved):
                    q.queue.appendleft(item)
                q.not_empty.notify(len(preserved))

    def clean_session(pipeline: "SessionPipeline", preserve: Callable[[Any], bool] | None = None) -> None:
        # Invalidate in-flight LLM/TTS work (cooperative cancel via is_stale), then
        # flush this session's queues. reset() clears discarding only; generation
        # stays bumped. Blocking HTTP reads are not interrupted here; see
        # ResponsesApiModelHandler.process.
        pipeline.cancel_scope.cancel()
        _flush_queue(pipeline.send_audio, preserve=preserve)
        if pipeline.text_output is not None:
            _flush_queue(pipeline.text_output, preserve=preserve)
        pipeline.response_playing.clear()
        pipeline.cancel_scope.reset()
        pipeline.should_listen.set()

    def _to_audio_bytes(chunk: Any) -> bytes:
        if isinstance(chunk, PipelineControlMessage):
            raise TypeError(f"unexpected control message on audio output queue: {chunk!r}")
        if isinstance(chunk, np.ndarray) or hasattr(chunk, "tobytes"):
            return chunk.tobytes()
        return chunk

    async def _idle_reaper() -> None:
        """Close sessions with no inbound client traffic for > S2S_IDLE_TIMEOUT_S.

        Closing the socket makes the connection handler's receive raise
        WebSocketDisconnect, so the normal disconnect path tears the pipeline
        down. Reads the module global each tick so it can be tuned (or monkey-
        patched in tests) without rebuilding the app."""
        while not stop_event.is_set():
            timeout = S2S_IDLE_TIMEOUT_S
            await asyncio.sleep(max(0.05, min(timeout, 5.0)) if timeout > 0 else 1.0)
            if timeout <= 0:
                continue
            for sid in service.idle_session_ids(timeout):
                ws = app.state.websockets.get(sid)
                if ws is None:
                    continue
                logger.info("Reaping idle session %s (no client traffic for >%.0fs)", sid, timeout)
                try:
                    await ws.close(code=1001, reason="idle timeout")
                except Exception:
                    pass

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # session_id → SessionPipeline / WebSocket. Each session is independent;
        # there is no "active" session — the send loop and dispatch are keyed by
        # session_id throughout.
        app.state.sessions = {}
        app.state.websockets = {}
        reaper_task = asyncio.create_task(_idle_reaper()) if S2S_IDLE_TIMEOUT_S > 0 else None
        yield
        # Server shutdown: stop the reaper, then stop each session's send task,
        # close its socket, and tear down its pipeline threads. Shutdowns run in
        # parallel daemon workers (joined with a bound) so one slow teardown can't
        # stall the others.
        if reaper_task is not None:
            reaper_task.cancel()
            try:
                await reaper_task
            except asyncio.CancelledError:
                pass
        workers = []
        for session_id, pipeline in list(app.state.sessions.items()):
            if pipeline.send_task is not None:
                pipeline.send_task.cancel()
                try:
                    await pipeline.send_task
                except asyncio.CancelledError:
                    pass
            ws = app.state.websockets.get(session_id)
            if ws is not None:
                try:
                    await ws.close(code=1001)
                except Exception:
                    pass
            workers.append(_shutdown_pipeline_async(session_id, pipeline))
        for worker in workers:
            worker.join(timeout=6.0)

    app = FastAPI(lifespan=lifespan)

    @app.websocket("/v1/realtime")
    async def realtime_endpoint(ws: WebSocket) -> None:
        await ws.accept()

        if server_api_key:
            auth_header = ws.headers.get("authorization", "")
            scheme, _, token = auth_header.partition(" ")
            if scheme.lower() != "bearer" or token != server_api_key:
                logger.warning("Rejected connection: invalid or missing Bearer token")
                await ws.close(code=4001, reason="Unauthorized")
                return

        session_id = service.register()

        # Capacity guard: accept up to max_sessions concurrent connections. Count
        # registered connections — a synchronous reservation done by register()
        # above — rather than attached pipelines: the build below now yields the
        # event loop (asyncio.to_thread), so a pipeline-keyed count could over-
        # admit under simultaneous connects. register()→this check has no await
        # between them, so the count is race-free.
        if len(service.connection_ids) > max_sessions:
            logger.warning(
                "Rejected connection: at session capacity (%d/%d)", max_sessions, max_sessions
            )
            service.unregister(session_id)
            await _send_event(
                ws,
                service.make_error(
                    f"Session limit reached ({max_sessions} concurrent "
                    f"{'session' if max_sessions == 1 else 'sessions'}). Disconnect an existing client first.",
                    _type="session_limit_reached",
                ),
            )
            await ws.close(code=1008, reason=f"Session limit reached ({max_sessions})")
            return

        pipeline: Optional["SessionPipeline"] = None
        try:
            # Build this connection's pipeline OFF the event loop. Each handler's
            # __init__ eagerly runs setup(): the VAD handler loads the Silero model
            # and (when TURN_DETECTION=smart_turn) the Smart-Turn ONNX session,
            # which costs seconds. Doing that inline would freeze every other
            # session's send/receive loop for the whole build — and a client whose
            # connect timeout is shorter than the build would drop mid-build.
            # asyncio.to_thread keeps the loop live (other sessions stay served and
            # this socket's keepalive is answered). Instrumented: warm-connection
            # clients depend on a low connect cost.
            build_start = time.perf_counter()
            pipeline = await asyncio.to_thread(session_factory.build_session_pipeline, session_id)
            pipeline.start()
            build_ms = (time.perf_counter() - build_start) * 1000
            logger.info("Session %s pipeline built + started in %.0f ms", session_id, build_ms)

            service.attach_pipeline(session_id, pipeline)
            app.state.sessions[session_id] = pipeline
            app.state.websockets[session_id] = ws
            pipeline.echo_canceller.reset()  # fresh far-end buffer for a new call
            logger.info(f"Client connected (session {session_id})")

            # Per-session send loop: drains this pipeline's output queues to this ws.
            pipeline.send_task = asyncio.create_task(_session_send_loop(session_id, pipeline, ws))

            # Defensive: drain edge queues and reset events so nothing leaks into the
            # first turn.
            clean_session(pipeline)

            await _send_event(ws, service.build_session_created(session_id))

            while not stop_event.is_set():
                try:
                    raw = await asyncio.wait_for(ws.receive_json(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue

                service.touch(session_id)
                event = service.parse_client_event(raw)
                if event is None:
                    await _send_event(
                        ws,
                        service.make_error(f"Unknown or invalid event: {raw.get('type')}", "unknown_or_invalid_event"),
                    )
                    continue

                if isinstance(event, InputAudioBufferAppendEvent):
                    chunks = service.handle_audio_append(session_id, event)
                    rt_cfg = service._state(session_id).runtime_config
                    for chunk in chunks:
                        # Cancel the agent's echo (far-end) out of the mic for the
                        # VAD *decision*, but keep the raw chunk for STT: AEC
                        # over-suppresses the user during double-talk, so the
                        # cleaned signal is good for detection yet bad for Whisper.
                        # far_active drives the VAD's far-aware gate (raw gate when
                        # the agent is silent, residual gate while it's speaking).
                        # Payload: (raw, cleaned, far_active, runtime_config).
                        cleaned = pipeline.echo_canceller.process(chunk)
                        pipeline.recv_audio.put(
                            (bytes(chunk), cleaned, pipeline.echo_canceller.far_active, rt_cfg)
                        )

                elif isinstance(event, InputAudioBufferCommitEvent):
                    err = service.handle_audio_commit(session_id)
                    if err:
                        await _send_event(ws, err)

                elif isinstance(event, SessionUpdateEvent):
                    reply = service.handle_session_update(session_id, event)
                    await _send_event(ws, reply)

                elif isinstance(event, ConversationItemCreateEvent):
                    events = service.handle_conversation_item_create(session_id, event)
                    if events:
                        await _send_events(ws, events)

                elif isinstance(event, ResponseCreateEvent):
                    result = service.handle_response_create(session_id, event)
                    if result:
                        if result.type != "error":
                            pipeline.cancel_scope.new_response()
                        await _send_event(ws, result)

                elif isinstance(event, ResponseCancelEvent):
                    was_active = service._state(session_id).in_response
                    if was_active:
                        pipeline.cancel_scope.cancel()
                    _flush_queue(pipeline.send_audio, preserve=_keep_audio_sentinel)
                    if pipeline.text_output is not None:
                        _flush_queue(pipeline.text_output, preserve=_keep_user_text_event)
                    events = service.handle_response_cancel(session_id)
                    if events:
                        await _send_events(ws, events)
                    pipeline.response_playing.clear()

        except WebSocketDisconnect:
            logger.info(f"Client {session_id} disconnected")
        except RuntimeError as e:
            # Starlette raises RuntimeError("WebSocket is not connected...") when
            # the peer has already closed — e.g. it timed out and hung up while the
            # pipeline was still building. That's a normal disconnect, not a server
            # error, so don't log a scary traceback for it.
            if "not connected" in str(e).lower():
                logger.info(f"Client {session_id} disconnected before the session was ready")
            else:
                logger.error(f"Client {session_id} error: {type(e).__name__}: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Client {session_id} error: {type(e).__name__}: {e}", exc_info=True)
        finally:
            # pipeline may be None if the build (now off-loop) was cancelled before
            # it attached — e.g. server shutdown during connect. Tear down only
            # what exists; the registration is always cleaned up below.
            # Stop sending to this socket before tearing the pipeline down so a
            # PIPELINE_END draining through the send loop can't race the close.
            if pipeline is not None and pipeline.send_task is not None:
                pipeline.send_task.cancel()
                try:
                    await pipeline.send_task
                except asyncio.CancelledError:
                    pass
            # Bookkeeping first so the session is observably gone (and a session
            # slot freed) immediately — the handler-thread join below must not
            # gate that.
            service.detach_pipeline(session_id)
            if pipeline is not None:
                clean_session(pipeline)
            service.unregister(session_id)
            app.state.sessions.pop(session_id, None)
            app.state.websockets.pop(session_id, None)
            # Join the handler threads in a fire-and-forget daemon worker, NOT via
            # loop.run_in_executor: awaiting the executor future here can wedge the
            # connection handler if the event loop stops servicing the result
            # callback during teardown. The worker sets stop_event + drains the
            # chain so the (non-daemon) handler threads exit promptly on their own.
            if pipeline is not None:
                _shutdown_pipeline_async(session_id, pipeline)
            logger.info(f"Client {session_id} removed")

    @app.get("/v1/usage")
    async def usage_endpoint() -> dict[str, Any]:
        return service.get_usage()

    @app.get("/v1/sessions")
    async def sessions_endpoint() -> dict[str, Any]:
        """Live session roster: one entry per connected session with its coarse
        state, age, idle time, turn count and per-session usage."""
        sessions = service.get_sessions()
        return {"count": len(sessions), "max_sessions": max_sessions, "sessions": sessions}

    async def _session_send_loop(session_id: str, pipeline: "SessionPipeline", ws: WebSocket) -> None:
        """Poll one session's output queues and send to its client."""
        output_queue = pipeline.send_audio
        text_output_queue = pipeline.text_output
        cancel_scope = pipeline.cancel_scope
        response_playing = pipeline.response_playing
        should_listen = pipeline.should_listen
        echo_canceller = pipeline.echo_canceller
        sid = session_id

        pending_output_item = None
        deltas_sent = 0  # DEBUG: response.output_audio.delta events emitted this response
        delta_pcm_bytes = 0  # DEBUG: pre-resample PCM bytes encoded into deltas this response
        loop = asyncio.get_running_loop()
        audio_play_deadline: float | None = None  # wall-clock time current audio should finish playing
        last_client_send_t = loop.time()  # last time anything was sent to the client (drives keepalive)
        last_supervise_t = loop.time()  # last dead-thread supervisor check
        dump_pcm = bytearray() if TTS_DUMP_DIR else None  # DEBUG: exact outbound PCM for this response
        dump_seq = 0
        while not stop_event.is_set():
            try:
                # Process text events first (speech_started cancels active response)
                if text_output_queue:
                    try:
                        text_msg = text_output_queue.get_nowait()
                        is_speech_start = isinstance(text_msg, SpeechStartedEvent)

                        was_in_response = False
                        if is_speech_start:
                            was_in_response = service._state(sid).in_response

                        if cancel_scope.discarding and isinstance(text_msg, AssistantTextEvent):
                            pass
                        elif isinstance(text_msg, PipelineEvent):
                            events = service.dispatch_pipeline_event(sid, text_msg)
                            if events:
                                await _send_events(ws, events)
                                last_client_send_t = loop.time()

                        if is_speech_start and was_in_response:
                            active_cfg = service._state(sid).runtime_config
                            if active_cfg is None or active_cfg.interrupt_response_enabled:
                                cancel_scope.cancel()
                                echo_canceller.flush_far()  # queued TTS won't be played → don't feed it as render
                                _flush_queue(output_queue, preserve=_keep_audio_sentinel)
                                _flush_queue(text_output_queue, preserve=_keep_user_text_event)
                                if response_playing.is_set():
                                    response_playing.clear()
                                logger.warning(
                                    "BARGE-IN FIRED: SpeechStartedEvent while in_response → cancel_scope.cancel() "
                                    "(scope_gen now %s, discarding=True). If this fires spuriously every response, "
                                    "the VAD is likely tripping on echo/tail and this is why audio is dropped.",
                                    cancel_scope.generation,
                                )
                            else:
                                logger.info("Speech during response: interrupt_response disabled, ignoring")
                    except Empty:
                        pass

                try:
                    if pending_output_item is not None:
                        audio_chunk = pending_output_item
                        pending_output_item = None
                    else:
                        audio_chunk = output_queue.get_nowait()

                    if isinstance(audio_chunk, bytes) and audio_chunk == PIPELINE_END:
                        await _send_events(ws, service.finish_audio_response(sid))
                        break

                    if isinstance(audio_chunk, bytes) and audio_chunk == AUDIO_RESPONSE_DONE:
                        if dump_pcm is not None and dump_pcm:
                            try:
                                os.makedirs(TTS_DUMP_DIR, exist_ok=True)
                                path = os.path.join(TTS_DUMP_DIR, f"out_{dump_seq:03d}.wav")
                                with wave.open(path, "wb") as w:
                                    w.setnchannels(1)
                                    w.setsampwidth(2)
                                    w.setframerate(TTS_DUMP_RATE)
                                    w.writeframes(bytes(dump_pcm))
                                logger.info(
                                    "Wrote outbound audio dump %s (%d bytes, %.2fs @ %d Hz)",
                                    path, len(dump_pcm), len(dump_pcm) / 2 / TTS_DUMP_RATE, TTS_DUMP_RATE,
                                )
                            except Exception as e:
                                logger.error("TTS dump failed: %s", e)
                            dump_pcm.clear()
                            dump_seq += 1
                        # A done sentinel arriving while discarding belongs to a CANCELLED
                        # generation: its response.done(cancelled) was already emitted at
                        # barge-in/cancel time. The response lifecycle now opens at turn
                        # start, so a NEW response may already be in_progress here —
                        # finishing on the stale sentinel would close it before it produced
                        # any audio. Skip the finish events; still clear the discard guard
                        # and playback state below.
                        stale_done = bool(cancel_scope.discarding)
                        if not stale_done:
                            # Hold response.done briefly so the client can drain its playout
                            # buffer. It plays inbound audio at real-time and ends up ~0.3 s
                            # behind, so a response.done that lands right after the last delta
                            # makes it finalize and drop the still-buffered tail (a real-time
                            # client played 1.06 s of a 1.42 s greeting). Skip the hold on
                            # barge-in/cancel.
                            if REALTIME_PACING_ENABLED:
                                held = 0.0
                                while held < RESPONSE_DONE_TAIL_S:
                                    if cancel_scope.discarding:
                                        break
                                    await asyncio.sleep(PACING_SLICE_S)
                                    held += PACING_SLICE_S
                            await _send_events(ws, service.finish_audio_response(sid))
                            last_client_send_t = loop.time()
                        response_playing.clear()
                        cancel_scope.response_done()
                        should_listen.set()
                        logger.info(
                            "Response complete, listening re-enabled — sent %d audio delta event(s) "
                            "carrying %d PCM bytes to client this response",
                            deltas_sent,
                            delta_pcm_bytes,
                        )
                        deltas_sent = 0
                        delta_pcm_bytes = 0
                        audio_play_deadline = None
                        continue

                    if is_control_message(audio_chunk):
                        continue

                    if cancel_scope.discarding:
                        # A barge-in mid-long-response leaves a big pre-rendered backlog that
                        # drains here one chunk at a time; log per-chunk only under DEBUG_MODE
                        # so production doesn't get ~1000 identical lines per interruption.
                        if DEBUG_MODE:
                            logger.info(
                                "DROP audio chunk in send loop: cancel_scope.discarding=True "
                                "(scope_gen=%s). Audio will NOT reach client until __RESPONSE_DONE__ "
                                "clears the discard guard via response_done().",
                                cancel_scope.generation,
                            )
                        audio_play_deadline = None
                        continue

                    audio_chunk = _to_audio_bytes(audio_chunk)

                    audio_batch = bytearray(audio_chunk)
                    while len(audio_batch) < MAX_AUDIO_BATCH_BYTES:
                        try:
                            next_chunk = output_queue.get_nowait()
                        except Empty:
                            break

                        if (
                            isinstance(next_chunk, bytes) and next_chunk in {PIPELINE_END, AUDIO_RESPONSE_DONE}
                        ) or is_control_message(next_chunk, SESSION_END.kind):
                            pending_output_item = next_chunk
                            break

                        next_audio = _to_audio_bytes(next_chunk)
                        if len(audio_batch) + len(next_audio) > MAX_AUDIO_BATCH_BYTES:
                            pending_output_item = next_chunk
                            break
                        audio_batch.extend(next_audio)

                    if not response_playing.is_set():
                        response_playing.set()
                        should_listen.set()

                    out_events = service.encode_audio_chunk(sid, bytes(audio_batch))
                    await _send_events(ws, out_events)
                    deltas_sent += 1
                    delta_pcm_bytes += len(audio_batch)
                    last_client_send_t = loop.time()
                    if dump_pcm is not None:
                        for ev in out_events:
                            if getattr(ev, "type", "") == "response.output_audio.delta":
                                dump_pcm += base64.b64decode(ev.delta)

                    # Far-end reference for AEC: the 16 kHz pipeline PCM, fed as it is
                    # sent (≈ when it plays), so the canceller can subtract its echo
                    # from the inbound mic.
                    echo_canceller.add_far_end(bytes(audio_batch))

                    # Pace to wall-clock: hold the next dequeue (and the trailing
                    # __RESPONSE_DONE__) until this batch would have finished playing,
                    # so response.done lands after the audio instead of ~2 s early.
                    if REALTIME_PACING_ENABLED and audio_batch:
                        batch_seconds = len(audio_batch) / (PIPELINE_BYTES_PER_SAMPLE * PIPELINE_SAMPLE_RATE)
                        now = loop.time()
                        # Keep the deadline ABSOLUTE across batches so per-iteration overhead
                        # (the 10 ms tail sleep, send/encode time) is absorbed by a shorter next
                        # sleep rather than accumulating — re-anchoring every batch made delivery
                        # ~0.67x real-time (client measured 15 kHz vs 24 kHz). Only re-anchor on a
                        # real stall so we don't burst a large backlog to catch up.
                        if audio_play_deadline is None or now - audio_play_deadline > 0.5:
                            audio_play_deadline = now
                        audio_play_deadline += batch_seconds
                        while True:
                            remaining = audio_play_deadline - loop.time()
                            if remaining <= 0:
                                break
                            # A client-initiated cancel sets discarding from another task; bail out fast.
                            if cancel_scope.discarding:
                                audio_play_deadline = None
                                break
                            await asyncio.sleep(min(remaining, PACING_SLICE_S))
                except Empty:
                    pass

                # Dead-thread supervisor: if a handler crashed out of its loop,
                # fail this session fast (server_error + close) instead of serving
                # a live socket from a half-dead pipeline. Empty pipelines (test
                # stubs) have no threads, so this never fires for them.
                if S2S_THREAD_SUPERVISOR_S > 0 and loop.time() - last_supervise_t >= S2S_THREAD_SUPERVISOR_S:
                    last_supervise_t = loop.time()
                    dead = pipeline.dead_threads()
                    if dead:
                        logger.error(
                            "Session %s: handler thread(s) exited unexpectedly (%s) — failing session",
                            sid, ", ".join(dead),
                        )
                        try:
                            await _send_event(
                                ws,
                                service.make_error(
                                    f"Session pipeline failed: {', '.join(dead)} exited", "server_error"
                                ),
                            )
                            await ws.close(code=1011, reason="pipeline failure")
                        except Exception as e:
                            logger.debug("supervisor close failed: %s", e)
                        break

                # Keepalive: a response is in_progress but the wire has been silent
                # (LLM thinking, agent-side tool loop, TTS synth). Tell the client
                # we're alive so its turn watchdog doesn't false-fire on a slow turn.
                if HEARTBEAT_S > 0:
                    st = service._conns.get(sid)
                    if st is not None and st.in_response and loop.time() - last_client_send_t >= HEARTBEAT_S:
                        try:
                            await ws.send_json(
                                {
                                    "type": "s2s.keepalive",
                                    "event_id": _generate_id("event"),
                                    "response_id": st.current_response_id,
                                }
                            )
                        except Exception as e:
                            logger.debug("keepalive send failed: %s", e)
                        last_client_send_t = loop.time()

                await asyncio.sleep(0.01)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Send loop error: {e}")

    return app
