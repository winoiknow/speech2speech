import asyncio
import base64
import logging
import os
import wave
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from queue import Empty, Queue
from threading import Event as ThreadingEvent
from typing import Any, Callable, Optional, TypeVar

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
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.control import SESSION_END, PipelineControlMessage, is_control_message
from speech_to_speech.pipeline.events import (
    AssistantTextEvent,
    PipelineEvent,
    SpeechStartedEvent,
    SpeechStoppedEvent,
    TokenUsageEvent,
)
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, PIPELINE_END
from speech_to_speech.pipeline.queue_types import AudioInItem, AudioOutItem, TextEventItem

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
# Debug: if set, write each response's exact outbound audio (the base64 PCM deltas the client
# receives, decoded) to a WAV in this dir, so the real on-wire audio can be auditioned directly.
TTS_DUMP_DIR = os.getenv("S2S_TTS_DUMP_DIR") or None
# Sample rate of the dumped WAV — the negotiated client output rate (GA pcm = 24000).
TTS_DUMP_RATE = int(os.getenv("S2S_TTS_DUMP_RATE", "24000"))


async def _send_event(ws: WebSocket, event: ServerEvent) -> None:
    try:
        await ws.send_json(event.model_dump())
    except Exception as e:
        logger.error(f"Failed to send event to client: {e}")


async def _send_events(ws: WebSocket, events: list[ServerEvent]) -> None:
    for event in events:
        await _send_event(ws, event)


def _keep_audio_sentinel(item: Any) -> bool:
    return isinstance(item, bytes) and item == AUDIO_RESPONSE_DONE


def _keep_user_text_event(item: Any) -> bool:
    return isinstance(item, (SpeechStoppedEvent, TokenUsageEvent))


def create_app(
    service: RealtimeService,
    input_queue: Queue[AudioInItem],
    output_queue: Queue[AudioOutItem],
    text_output_queue: Queue[TextEventItem],
    should_listen: ThreadingEvent,
    response_playing: ThreadingEvent | None,
    cancel_scope: CancelScope | None,
    stop_event: ThreadingEvent,
    server_api_key: Optional[str] = None,
) -> FastAPI:

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

    def clean_session(preserve: Callable[[Any], bool] | None = None) -> None:
        # Invalidate in-flight LLM/TTS work (cooperative cancel via is_stale), then
        # flush queues. reset() clears discarding only; generation stays bumped.
        # Blocking HTTP reads are not interrupted here; see ResponsesApiModelHandler.process.
        if cancel_scope:
            cancel_scope.cancel()
        _flush_queue(output_queue, preserve=preserve)
        _flush_queue(text_output_queue, preserve=preserve)
        if response_playing:
            response_playing.clear()
        if cancel_scope:
            cancel_scope.reset()
        should_listen.set()

    def _to_audio_bytes(chunk: AudioOutItem) -> bytes:
        if isinstance(chunk, PipelineControlMessage):
            raise TypeError(f"unexpected control message on audio output queue: {chunk!r}")
        if isinstance(chunk, np.ndarray) or hasattr(chunk, "tobytes"):
            return chunk.tobytes()
        return chunk

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.websockets = {}
        app.state.active_session: Optional[str] = None  # type: ignore[misc]
        app.state.send_task = asyncio.create_task(_send_loop())
        yield
        app.state.send_task.cancel()
        try:
            await app.state.send_task
        except asyncio.CancelledError:
            pass
        for ws in list(app.state.websockets.values()):
            try:
                await ws.close()
            except Exception:
                pass

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

        if app.state.websockets:
            logger.warning("Rejected connection: a session is already active")
            await _send_event(
                ws,
                service.make_error(
                    "Only one concurrent session is supported. Disconnect the existing client first.",
                    _type="session_limit_reached",
                ),
            )
            await ws.close(code=1008, reason="Only one concurrent session is supported")
            return

        session_id = service.register()
        app.state.websockets[session_id] = ws
        app.state.active_session = session_id
        logger.info(f"Client connected (session {session_id})")

        # Defensive: drain edge queues and reset events so stale data from a
        # previous session that survived SESSION_END propagation doesn't leak.
        clean_session()

        try:
            await _send_event(ws, service.build_session_created(session_id))

            while not stop_event.is_set():
                try:
                    raw = await asyncio.wait_for(ws.receive_json(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue

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
                        input_queue.put((chunk, rt_cfg))

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
                        if result.type != "error" and cancel_scope:
                            cancel_scope.new_response()
                        await _send_event(ws, result)

                elif isinstance(event, ResponseCancelEvent):
                    was_active = service._state(session_id).in_response
                    if was_active and cancel_scope:
                        cancel_scope.cancel()
                    _flush_queue(output_queue, preserve=_keep_audio_sentinel)
                    _flush_queue(text_output_queue, preserve=_keep_user_text_event)
                    events = service.handle_response_cancel(session_id)
                    if events:
                        await _send_events(ws, events)
                    if response_playing:
                        response_playing.clear()

        except WebSocketDisconnect:
            logger.info(f"Client {session_id} disconnected")
        except Exception as e:
            logger.error(f"Client {session_id} error: {type(e).__name__}: {e}", exc_info=True)
        finally:
            clean_session()
            service.unregister(session_id)
            if not service._conns:
                input_queue.put(SESSION_END)
                logger.info("Last client disconnected, sent SESSION_END")
            app.state.websockets.pop(session_id, None)
            app.state.active_session = None
            logger.info(f"Client {session_id} removed")

    @app.get("/v1/usage")
    async def usage_endpoint() -> dict[str, Any]:
        return service.get_usage()

    async def _send_loop() -> None:
        """Poll pipeline output queues and send to each connected client."""
        pending_output_item = None
        deltas_sent = 0  # DEBUG: response.output_audio.delta events emitted this response
        delta_pcm_bytes = 0  # DEBUG: pre-resample PCM bytes encoded into deltas this response
        loop = asyncio.get_running_loop()
        audio_play_deadline: float | None = None  # wall-clock time current audio should finish playing
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
                        if is_speech_start and app.state.active_session:
                            was_in_response = service._state(app.state.active_session).in_response

                        if cancel_scope and cancel_scope.discarding and isinstance(text_msg, AssistantTextEvent):
                            pass
                        else:
                            for cid in service.connection_ids:
                                ws = app.state.websockets.get(cid)
                                if ws and isinstance(text_msg, PipelineEvent):
                                    events = service.dispatch_pipeline_event(cid, text_msg)
                                    if events:
                                        await _send_events(ws, events)

                        if is_speech_start and was_in_response:
                            active_cfg = (
                                service._state(app.state.active_session).runtime_config
                                if app.state.active_session
                                else None
                            )
                            if active_cfg is None or active_cfg.interrupt_response_enabled:
                                if cancel_scope:
                                    cancel_scope.cancel()
                                _flush_queue(output_queue, preserve=_keep_audio_sentinel)
                                _flush_queue(text_output_queue, preserve=_keep_user_text_event)
                                if response_playing and response_playing.is_set():
                                    response_playing.clear()
                                logger.warning(
                                    "BARGE-IN FIRED: SpeechStartedEvent while in_response → cancel_scope.cancel() "
                                    "(scope_gen now %s, discarding=True). If this fires spuriously every response, "
                                    "the VAD is likely tripping on echo/tail and this is why audio is dropped.",
                                    cancel_scope.generation if cancel_scope else None,
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
                        for cid in service.connection_ids:
                            ws = app.state.websockets.get(cid)
                            if ws:
                                await _send_events(ws, service.finish_audio_response(cid))
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
                        # Hold response.done briefly so the client can drain its playout
                        # buffer. It plays inbound audio at real-time and ends up ~0.3 s
                        # behind, so a response.done that lands right after the last delta
                        # makes it finalize and drop the still-buffered tail (AVA played
                        # 1.06 s of a 1.42 s greeting). Skip the hold on barge-in/cancel.
                        if REALTIME_PACING_ENABLED and not (cancel_scope and cancel_scope.discarding):
                            held = 0.0
                            while held < RESPONSE_DONE_TAIL_S:
                                if cancel_scope and cancel_scope.discarding:
                                    break
                                await asyncio.sleep(PACING_SLICE_S)
                                held += PACING_SLICE_S
                        for cid in service.connection_ids:
                            ws = app.state.websockets.get(cid)
                            if ws:
                                await _send_events(ws, service.finish_audio_response(cid))
                        if response_playing:
                            response_playing.clear()
                        if cancel_scope:
                            cancel_scope.response_done()
                        should_listen.set()
                        logger.info(
                            "Response complete, listening re-enabled — sent %d audio delta event(s) "
                            "carrying %d PCM bytes to %d client(s) this response",
                            deltas_sent,
                            delta_pcm_bytes,
                            len(service.connection_ids),
                        )
                        deltas_sent = 0
                        delta_pcm_bytes = 0
                        audio_play_deadline = None
                        continue

                    if is_control_message(audio_chunk):
                        continue

                    if cancel_scope and cancel_scope.discarding:
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

                    if response_playing and not response_playing.is_set():
                        response_playing.set()
                        should_listen.set()

                    for idx, cid in enumerate(service.connection_ids):
                        ws = app.state.websockets.get(cid)
                        if ws:
                            out_events = service.encode_audio_chunk(cid, bytes(audio_batch))
                            await _send_events(ws, out_events)
                            deltas_sent += 1
                            delta_pcm_bytes += len(audio_batch)
                            if dump_pcm is not None and idx == 0:
                                for ev in out_events:
                                    if getattr(ev, "type", "") == "response.output_audio.delta":
                                        dump_pcm += base64.b64decode(ev.delta)

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
                            if cancel_scope and cancel_scope.discarding:
                                audio_play_deadline = None
                                break
                            await asyncio.sleep(min(remaining, PACING_SLICE_S))
                except Empty:
                    pass

                await asyncio.sleep(0.01)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Send loop error: {e}")

    return app
