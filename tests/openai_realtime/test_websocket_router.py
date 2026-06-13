"""Integration tests for api.openai_realtime.websocket_router.

Uses Starlette's synchronous TestClient with WebSocket support to exercise
the full FastAPI app produced by ``create_app``.  Each test gets a fresh
app, service, and set of queues so there is no cross-test state.
"""

import base64
import time
from queue import Queue
from threading import Event as ThreadingEvent

import pytest
from starlette.testclient import TestClient

from speech_to_speech.api.openai_realtime.service import CHUNK_SIZE_BYTES, RealtimeService
from speech_to_speech.api.openai_realtime.websocket_router import create_app
from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.pipeline.cancel_scope import CancelScope

from .conftest import FixedQueueSessionFactory, IndependentSessionStubFactory
from speech_to_speech.pipeline.control import SESSION_END, is_control_message
from speech_to_speech.pipeline.events import (
    AssistantTextEvent,
    SpeechStartedEvent,
    SpeechStoppedEvent,
    TranscriptionCompletedEvent,
)
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, PIPELINE_END

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def setup():
    """Return (app, service, input_queue, output_queue, text_output_queue, should_listen, stop_event, response_playing, cancel_scope)."""
    text_prompt_queue = Queue()
    should_listen = ThreadingEvent()
    should_listen.set()
    service = RealtimeService(
        text_prompt_queue=text_prompt_queue,
        should_listen=should_listen,
    )
    input_queue = Queue()
    output_queue = Queue()
    text_output_queue = Queue()
    stop_event = ThreadingEvent()
    response_playing = ThreadingEvent()
    cancel_scope = CancelScope()
    # Connect-time build (Phase B): the pipeline is built by the factory on
    # connect, wrapping these fixed queues so the test bodies inject/inspect them
    # unchanged.
    factory = FixedQueueSessionFactory(
        recv_audio=input_queue,
        send_audio=output_queue,
        text_output=text_output_queue,
        should_listen=should_listen,
        response_playing=response_playing,
        cancel_scope=cancel_scope,
        text_prompt=text_prompt_queue,
    )
    app = create_app(service, factory, stop_event)
    return (
        app,
        service,
        input_queue,
        output_queue,
        text_output_queue,
        should_listen,
        stop_event,
        response_playing,
        cancel_scope,
    )


def _pcm_bytes(n_samples: int) -> bytes:
    return b"\x00" * (n_samples * 2)


# Realtime output lifecycle, in spec order, before the first audio delta.
BEGIN_EVENT_TYPES = [
    "response.created",
    "response.output_item.added",
    "response.content_part.added",
]
# Close-of-response lifecycle, after the last audio delta.
FINISH_EVENT_TYPES = [
    "response.output_audio.done",
    "response.content_part.done",
    "response.output_item.done",
    "response.done",
]


def _receive_types(ws, n: int) -> list[str]:
    return [ws.receive_json()["type"] for _ in range(n)]


# ===================================================================
# Connection
# ===================================================================


class TestConnection:
    def test_connect_receives_session_created(self, setup):
        app, *_ = setup
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                msg = ws.receive_json()
                assert msg["type"] == "session.created"
                assert msg["event_id"].startswith("event_")
                assert "session" in msg

    def test_second_connection_rejected(self, setup):
        app, *_ = setup
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws1:
                ws1.receive_json()  # session.created
                with client.websocket_connect("/v1/realtime") as ws2:
                    msg = ws2.receive_json()
                    assert msg["type"] == "error"


# ===================================================================
# Client event dispatch
# ===================================================================


class TestClientEventDispatch:
    def test_audio_append_forwarded_to_input_queue(self, setup):
        app, _, input_queue, *_ = setup
        audio_b64 = base64.b64encode(_pcm_bytes(512)).decode("ascii")
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()  # session.created
                ws.send_json(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": audio_b64,
                    }
                )
                time.sleep(0.1)
                item = input_queue.get(timeout=1)
                # (raw, aec-cleaned, far_active, runtime_config)
                assert isinstance(item, tuple) and len(item) == 4
                chunk, cleaned, far_active, rt_cfg = item
                assert isinstance(chunk, bytes)
                assert len(chunk) == CHUNK_SIZE_BYTES
                assert isinstance(cleaned, bytes)
                assert isinstance(far_active, bool)

    def test_session_update_applied(self, setup):
        app, service, *_ = setup
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()
                ws.send_json(
                    {
                        "type": "session.update",
                        "session": {
                            "type": "realtime",
                            "audio": {"output": {"voice": "coral"}},
                        },
                    }
                )
                time.sleep(0.1)
                cid = service.connection_ids[0]
                assert service._state(cid).runtime_config.session.audio.output.voice == "coral"

    def test_conversation_item_create_returns_events(self, setup):
        app, *_ = setup
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()  # session.created
                ws.send_json(
                    {
                        "type": "conversation.item.create",
                        "item": {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": "ping"}],
                        },
                    }
                )
                msg = ws.receive_json()
                assert msg["type"] == "conversation.item.created"
                assert msg["item"]["content"][0]["text"] == "ping"

    def test_response_create_error_when_active(self, setup):
        app, service, *_ = setup
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()
                conn_id = list(service._conns.keys())[0]
                service.response._ensure_response(conn_id)
                ws.send_json({"type": "response.create"})
                msg = ws.receive_json()
                assert msg["type"] == "error"
                assert "another response is in progress" in msg["error"]["message"].lower()

    def test_response_cancel_returns_events(self, setup):
        app, service, *_ = setup
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()
                conn_id = list(service._conns.keys())[0]
                service.response._ensure_response(conn_id)
                ws.send_json({"type": "response.cancel"})
                msg1 = ws.receive_json()
                msg2 = ws.receive_json()
                types = {msg1["type"], msg2["type"]}
                assert "response.output_audio.done" in types
                assert "response.done" in types

    def test_response_cancel_flushes_queues(self, setup):
        app, service, _, output_queue, text_output_queue, _, _, response_playing, cancel_scope = setup
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()
                conn_id = list(service._conns.keys())[0]
                service.response._ensure_response(conn_id)
                response_playing.set()
                output_queue.put(_pcm_bytes(256))
                output_queue.put(_pcm_bytes(256))
                text_output_queue.put(AssistantTextEvent(text="stale"))
                ws.send_json({"type": "response.cancel"})
                ws.receive_json()  # response.output_audio.done
                ws.receive_json()  # response.done
                time.sleep(0.1)
                assert output_queue.empty()
                assert text_output_queue.empty()
                assert not response_playing.is_set()
                assert cancel_scope.discarding

    def test_response_cancel_spurious_does_not_set_discarding(self, setup):
        """response.cancel when no response is active must NOT enable discarding,
        otherwise it would stick True forever (no __RESPONSE_DONE__ to clear it)."""
        app, service, _, _, _, _, _, _, cancel_scope = setup
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()  # session.created
                assert not service._state(list(service._conns.keys())[0]).in_response
                ws.send_json({"type": "response.cancel"})
                time.sleep(0.1)
                assert not cancel_scope.discarding

    def test_response_cancel_late_audio_is_discarded(self, setup):
        """Audio arriving after response.cancel is silently dropped (discard guard)."""
        app, service, _, output_queue, _, _, _, response_playing, cancel_scope = setup
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()  # session.created
                conn_id = list(service._conns.keys())[0]
                service.response._ensure_response(conn_id)
                response_playing.set()
                ws.send_json({"type": "response.cancel"})
                ws.receive_json()  # response.output_audio.done
                ws.receive_json()  # response.done
                time.sleep(0.1)
                assert cancel_scope.discarding
                output_queue.put(_pcm_bytes(256))
                time.sleep(0.15)
                # No response.created or audio delta should appear; only
                # __RESPONSE_DONE__ will eventually clear the guard.
                output_queue.put(AUDIO_RESPONSE_DONE)
                time.sleep(0.15)
                assert not cancel_scope.discarding

    def test_unknown_event_returns_error(self, setup):
        app, *_ = setup
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()
                ws.send_json({"type": "bogus.event"})
                msg = ws.receive_json()
                assert msg["type"] == "error"


# ===================================================================
# Send loop (pipeline -> client)
# ===================================================================


class TestSendLoop:
    def test_audio_output_ignores_session_end_control_message(self, setup):
        app, _, _, output_queue, *_ = setup
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()  # session.created
                output_queue.put(SESSION_END)
                output_queue.put(_pcm_bytes(256))

                types = _receive_types(ws, 4)
                assert types == BEGIN_EVENT_TYPES + ["response.output_audio.delta"]

    def test_audio_output_sends_lifecycle_and_delta(self, setup):
        app, _, _, output_queue, *_ = setup
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()  # session.created
                output_queue.put(_pcm_bytes(256))
                msg1 = ws.receive_json()
                assert msg1["type"] == "response.created"
                assert msg1["response"]["status"] == "in_progress"
                assert ws.receive_json()["type"] == "response.output_item.added"
                assert ws.receive_json()["type"] == "response.content_part.added"
                msg2 = ws.receive_json()
                assert msg2["type"] == "response.output_audio.delta"
                assert "delta" in msg2

    def test_audio_output_batches_immediately_available_chunks(self, setup):
        app, _, _, output_queue, *_ = setup
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()  # session.created
                output_queue.put(_pcm_bytes(256))
                output_queue.put(_pcm_bytes(256))
                output_queue.put(PIPELINE_END)

                types = _receive_types(ws, 3)
                assert types == BEGIN_EVENT_TYPES

                # MAX_AUDIO_BATCH_BYTES (20 ms = 640 bytes at the pipeline rate)
                # keeps the two 512-byte chunks in separate deltas; each is
                # resampled 16 kHz -> 24 kHz (default client rate), so the total
                # PCM grows by 1.5x.
                total_pcm = 0
                for _ in range(2):
                    msg = ws.receive_json()
                    assert msg["type"] == "response.output_audio.delta"
                    total_pcm += len(base64.b64decode(msg["delta"]))
                assert total_pcm == len(_pcm_bytes(512)) * 3 // 2

                assert _receive_types(ws, 4) == FINISH_EVENT_TYPES

    def test_end_marker_sends_finish_events(self, setup):
        app, _, _, output_queue, *_ = setup
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()
                output_queue.put(_pcm_bytes(256))
                _receive_types(ws, 4)  # begin lifecycle + audio delta
                output_queue.put(PIPELINE_END)
                assert _receive_types(ws, 4) == FINISH_EVENT_TYPES

    def test_text_output_sends_pipeline_events(self, setup):
        app, _, _, _, text_output_queue, *_ = setup
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()
                text_output_queue.put(SpeechStartedEvent())
                msg = ws.receive_json()
                assert msg["type"] == "input_audio_buffer.speech_started"
                assert msg["audio_start_ms"] == 0

    def test_barge_in_discard_clears_after_response_done(self, setup):
        """After barge-in sets discarding=True, __RESPONSE_DONE__ must clear it back to False."""
        app, service, _, output_queue, text_output_queue, _, _, response_playing, cancel_scope = setup
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()  # session.created
                conn_id = list(service._conns.keys())[0]
                service.response._ensure_response(conn_id)
                response_playing.set()
                # Trigger barge-in
                text_output_queue.put(SpeechStartedEvent())
                ws.receive_json()  # input_audio_buffer.speech_started
                ws.receive_json()  # response.output_audio.done
                ws.receive_json()  # response.done
                time.sleep(0.1)
                assert cancel_scope.discarding
                output_queue.put(AUDIO_RESPONSE_DONE)
                time.sleep(0.15)
                assert not cancel_scope.discarding

    def test_speech_started_does_not_cancel_when_interrupt_disabled(self, setup):
        """With interrupt_response=False, speech during playback should NOT cancel or flush."""
        from openai.types.realtime.realtime_audio_input_turn_detection import ServerVad

        app, service, _, output_queue, text_output_queue, _, _, response_playing, cancel_scope = setup
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()  # session.created
                conn_id = list(service._conns.keys())[0]
                service._state(conn_id).runtime_config.session.audio.input.turn_detection = ServerVad(
                    type="server_vad",
                    interrupt_response=False,
                )
                service.response._ensure_response(conn_id)
                response_playing.set()
                text_output_queue.put(SpeechStartedEvent())
                msg = ws.receive_json()
                assert msg["type"] == "input_audio_buffer.speech_started"
                time.sleep(0.15)
                assert response_playing.is_set(), "response_playing should remain set"
                assert not cancel_scope.discarding, "cancel_scope should not be discarding"
                assert service._state(conn_id).in_response, "response should still be active"


# ===================================================================
# Turn-start lifecycle and keepalive (the silent "thinking" gap)
# ===================================================================


class TestThinkingGap:
    def _start_turn(self, ws, text_output_queue):
        """Drive a VAD turn to completion and return the early response.created."""
        text_output_queue.put(SpeechStartedEvent())
        assert ws.receive_json()["type"] == "input_audio_buffer.speech_started"
        text_output_queue.put(SpeechStoppedEvent(duration_s=1.0))
        assert ws.receive_json()["type"] == "input_audio_buffer.speech_stopped"
        text_output_queue.put(TranscriptionCompletedEvent(transcript="hello"))
        assert ws.receive_json()["type"] == "conversation.item.input_audio_transcription.completed"
        created = ws.receive_json()
        assert created["type"] == "response.created"
        assert created["response"]["status"] == "in_progress"
        return created

    def test_response_created_emitted_at_turn_start(self, setup):
        """response.created arrives with the transcription, before any audio exists."""
        app, service, _, output_queue, text_output_queue, *_ = setup
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()  # session.created
                created = self._start_turn(ws, text_output_queue)
                conn_id = list(service._conns.keys())[0]
                assert service._state(conn_id).in_response

                # First audio attaches to the SAME response: no second created.
                output_queue.put(_pcm_bytes(256))
                assert ws.receive_json()["type"] == "response.output_item.added"
                assert ws.receive_json()["type"] == "response.content_part.added"
                delta = ws.receive_json()
                assert delta["type"] == "response.output_audio.delta"
                assert delta["response_id"] == created["response"]["id"]

    def test_keepalive_during_silent_gap(self, setup, monkeypatch):
        """With a response in_progress and nothing on the wire, s2s.keepalive is emitted."""
        import speech_to_speech.api.openai_realtime.websocket_router as router_mod

        monkeypatch.setattr(router_mod, "HEARTBEAT_S", 0.2)
        app, _, _, _, text_output_queue, *_ = setup
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()  # session.created
                created = self._start_turn(ws, text_output_queue)
                # Nothing else is sent: the next event must be the keepalive.
                ka = ws.receive_json()
                assert ka["type"] == "s2s.keepalive"
                assert ka["response_id"] == created["response"]["id"]
                assert ka["event_id"].startswith("event_")
                # And it repeats while the gap continues.
                assert ws.receive_json()["type"] == "s2s.keepalive"

    def test_no_keepalive_outside_response(self, setup, monkeypatch):
        """Idle session (no in-flight response) must stay silent."""
        import speech_to_speech.api.openai_realtime.websocket_router as router_mod

        monkeypatch.setattr(router_mod, "HEARTBEAT_S", 0.2)
        app, _, _, _, text_output_queue, *_ = setup
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()  # session.created
                # Idle for several heartbeat intervals with no response open.
                time.sleep(0.7)
                # The next frame on the wire must be this event — if any
                # keepalive had been sent while idle, it would arrive first.
                text_output_queue.put(SpeechStartedEvent())
                assert ws.receive_json()["type"] == "input_audio_buffer.speech_started"

    def test_barge_in_during_thinking_gap_cancels(self, setup):
        """Speech while the LLM is in flight (no audio yet) cancels the early response."""
        app, service, _, output_queue, text_output_queue, _, _, _, cancel_scope = setup
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()  # session.created
                created = self._start_turn(ws, text_output_queue)

                # User speaks again before any audio was produced.
                text_output_queue.put(SpeechStartedEvent())
                # finish (cancelled, no output item yet) + speech_started
                msgs = [ws.receive_json() for _ in range(3)]
                types = [m["type"] for m in msgs]
                assert "response.output_audio.done" in types
                assert "response.done" in types
                assert "input_audio_buffer.speech_started" in types
                done = next(m for m in msgs if m["type"] == "response.done")
                assert done["response"]["id"] == created["response"]["id"]
                assert done["response"]["status"] == "cancelled"
                time.sleep(0.1)
                assert cancel_scope.discarding

    def test_stale_response_done_does_not_close_next_response(self, setup):
        """A cancelled generation's __RESPONSE_DONE__ must not finish the next
        (already-created) response; it only clears the discard guard."""
        app, service, _, output_queue, text_output_queue, _, _, _, cancel_scope = setup
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()  # session.created
                self._start_turn(ws, text_output_queue)

                # Barge-in during the gap: turn 1's response is cancelled.
                text_output_queue.put(SpeechStartedEvent())
                for _ in range(3):
                    ws.receive_json()
                time.sleep(0.1)
                assert cancel_scope.discarding

                # Turn 2 opens its response while turn 1's LLM is still winding down.
                text_output_queue.put(SpeechStoppedEvent(duration_s=1.0))
                assert ws.receive_json()["type"] == "input_audio_buffer.speech_stopped"
                text_output_queue.put(TranscriptionCompletedEvent(transcript="again"))
                assert ws.receive_json()["type"] == "conversation.item.input_audio_transcription.completed"
                created2 = ws.receive_json()
                assert created2["type"] == "response.created"

                # Turn 1's terminator finally drains: guard clears, turn 2 stays open.
                output_queue.put(AUDIO_RESPONSE_DONE)
                time.sleep(0.2)
                assert not cancel_scope.discarding
                conn_id = list(service._conns.keys())[0]
                assert service._state(conn_id).in_response, "stale done must not close the new response"

                # Turn 2 then streams and completes normally.
                output_queue.put(_pcm_bytes(256))
                assert ws.receive_json()["type"] == "response.output_item.added"
                assert ws.receive_json()["type"] == "response.content_part.added"
                delta = ws.receive_json()
                assert delta["type"] == "response.output_audio.delta"
                assert delta["response_id"] == created2["response"]["id"]
                output_queue.put(AUDIO_RESPONSE_DONE)
                assert _receive_types(ws, 4) == FINISH_EVENT_TYPES


# ===================================================================
# Cleanup
# ===================================================================


class TestCleanup:
    def test_new_connection_resets_discard_after_invalidating_generation(self, setup):
        """connect-time clean_session cancels+resets: stale work is invalidated, discarding cleared."""
        app, _, *_rest, cancel_scope = setup
        cancel_scope.cancel()
        assert cancel_scope.discarding
        assert cancel_scope.generation == 1
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()
                assert not cancel_scope.discarding
                assert cancel_scope.generation == 2

    def test_disconnect_bumps_cancel_scope_generation(self, setup):
        """clean_session() calls cancel() so in-flight pipeline generations go stale."""
        app, _, _, _, _, _, _, _, cancel_scope = setup
        assert cancel_scope.generation == 0
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()
                assert cancel_scope.generation == 1
            time.sleep(0.2)
        assert cancel_scope.generation == 2

    def test_disconnect_unregisters(self, setup):
        app, service, input_queue, *_ = setup
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()
                assert len(service._conns) == 1
            time.sleep(0.2)
            assert len(service._conns) == 0
            end = input_queue.get(timeout=1)
            assert is_control_message(end, SESSION_END.kind)

    def test_last_disconnect_cancels_and_clears_response_state(self, setup):
        app, service, input_queue, output_queue, text_output_queue, _, _, response_playing, cancel_scope = setup
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()
                conn_id = list(service._conns.keys())[0]
                service.response._ensure_response(conn_id)
                response_playing.set()
                output_queue.put(_pcm_bytes(256))
                text_output_queue.put(AssistantTextEvent(text="stale"))
            time.sleep(0.2)

        assert not cancel_scope.discarding
        assert cancel_scope.generation == 2
        assert not response_playing.is_set()
        assert output_queue.empty()
        assert text_output_queue.empty()
        end = input_queue.get(timeout=1)
        assert is_control_message(end, SESSION_END.kind)


# ===================================================================
# Phase B: connect-time build / disconnect-time teardown leaks no threads
# ===================================================================


class _IdentityHandler(BaseHandler):
    """Trivial pass-through handler that runs the real BaseHandler loop, so it
    starts a thread on connect and must be joined (via PIPELINE_END cascade /
    stop_event) on disconnect."""

    def process(self, item):
        yield item


class _ThreadedStubFactory:
    """Session factory that builds a real (but trivial) 3-handler thread chain
    wired recv_audio → … → send_audio, so a connect/disconnect cycle exercises
    actual thread start + join."""

    def __init__(self):
        self.built: list = []

    def build_session_pipeline(self, session_id: str):
        from speech_to_speech.audio.echo_canceller import EchoCanceller
        from speech_to_speech.pipeline.session_pipeline import SessionPipeline
        from speech_to_speech.utils.thread_manager import ThreadManager

        stop_event = ThreadingEvent()
        recv, q1, q2, send = Queue(), Queue(), Queue(), Queue()
        handlers = [
            _IdentityHandler(stop_event, recv, q1),
            _IdentityHandler(stop_event, q1, q2),
            _IdentityHandler(stop_event, q2, send),
        ]
        pipeline = SessionPipeline(
            session_id=session_id,
            recv_audio=recv,
            spoken_prompt=q1,
            stt_output=q2,
            text_prompt=Queue(),
            lm_response=Queue(),
            lm_processed=Queue(),
            send_audio=send,
            text_output=Queue(),
            stop_event=stop_event,
            should_listen=ThreadingEvent(),
            response_playing=ThreadingEvent(),
            cancel_scope=CancelScope(),
            echo_canceller=EchoCanceller(sample_rate=16000, enabled=False),
            handlers=handlers,
            threads=ThreadManager(handlers, daemon=True),
        )
        self.built.append(pipeline)
        return pipeline


def _wait_until(predicate, timeout: float = 5.0, interval: float = 0.02) -> bool:
    """Poll *predicate* until true or *timeout*; returns its final value. Used
    instead of a fixed sleep so the assertion is robust to CPU contention from
    other tests (the disconnect teardown joins threads asynchronously)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


class TestThreadLifecycle:
    def test_reconnect_cycle_leaks_no_threads(self):
        """connect → disconnect → reconnect, three times: every session's handler
        threads must be started on connect and fully joined on disconnect."""
        factory = _ThreadedStubFactory()
        service = RealtimeService()
        stop_event = ThreadingEvent()
        app = create_app(service, factory, stop_event)

        with TestClient(app) as client:
            for _ in range(3):
                with client.websocket_connect("/v1/realtime") as ws:
                    ws.receive_json()  # session.created
                    # the freshly built pipeline's threads are alive while connected
                    pipeline = factory.built[-1]
                    assert all(t.is_alive() for t in pipeline.threads.threads)
                # after disconnect, that session's threads are all joined (the
                # teardown join is async, so poll rather than sleep a fixed time)
                joined = _wait_until(lambda: not any(t.is_alive() for t in pipeline.threads.threads))
                assert joined, "session handler threads not joined after disconnect"

        assert _wait_until(lambda: len(service._conns) == 0)
        # No session pipeline left a live thread behind across all three cycles.
        for pipeline in factory.built:
            assert not any(t.is_alive() for t in pipeline.threads.threads)
        assert len(factory.built) == 3


# ===================================================================
# Phase C: multiple concurrent sessions, fully isolated
# ===================================================================


TRANSCRIPTION_COMPLETED_TYPE = "conversation.item.input_audio_transcription.completed"


def _recv_until(ws, target_type: str, limit: int = 10) -> dict:
    """Read events from *ws* until one of *target_type* arrives (or *limit* tried)."""
    for _ in range(limit):
        msg = ws.receive_json()
        if msg["type"] == target_type:
            return msg
    raise AssertionError(f"{target_type} not received within {limit} events")


def _multi_app(factory, max_sessions: int):
    service = RealtimeService()
    stop_event = ThreadingEvent()
    app = create_app(service, factory, stop_event, max_sessions=max_sessions)
    return app, service


class TestMultiSession:
    def test_capacity_allows_concurrent_sessions(self):
        factory = IndependentSessionStubFactory()
        app, service = _multi_app(factory, max_sessions=2)
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws_a:
                assert ws_a.receive_json()["type"] == "session.created"
                with client.websocket_connect("/v1/realtime") as ws_b:
                    assert ws_b.receive_json()["type"] == "session.created"
                    assert len(service._conns) == 2
                    assert len(factory.built) == 2

    def test_session_limit_reached_at_capacity(self):
        factory = IndependentSessionStubFactory()
        app, service = _multi_app(factory, max_sessions=1)
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws_a:
                ws_a.receive_json()  # session.created
                with client.websocket_connect("/v1/realtime") as ws_b:
                    msg = ws_b.receive_json()
                    assert msg["type"] == "error"
                    assert msg["error"]["type"] == "session_limit_reached"
                    assert "1" in msg["error"]["message"]
                # the rejected connection never registered
                assert len(service._conns) == 1

    def test_no_cross_talk_between_sessions(self):
        """Each session's transcription routes only to its own socket."""
        factory = IndependentSessionStubFactory()
        app, _service = _multi_app(factory, max_sessions=2)
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws_a:
                ws_a.receive_json()
                with client.websocket_connect("/v1/realtime") as ws_b:
                    ws_b.receive_json()
                    pa, pb = factory.built[0], factory.built[1]

                    pa.text_output.put(TranscriptionCompletedEvent(transcript="alpha"))
                    pb.text_output.put(TranscriptionCompletedEvent(transcript="bravo"))

                    a_done = _recv_until(ws_a, TRANSCRIPTION_COMPLETED_TYPE)
                    b_done = _recv_until(ws_b, TRANSCRIPTION_COMPLETED_TYPE)
                    assert a_done["transcript"] == "alpha"
                    assert b_done["transcript"] == "bravo"

    def test_barge_in_isolated_between_sessions(self):
        """Speech (barge-in) in session A cancels A only; B keeps streaming."""
        factory = IndependentSessionStubFactory()
        app, service = _multi_app(factory, max_sessions=2)
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws_a:
                ws_a.receive_json()
                with client.websocket_connect("/v1/realtime") as ws_b:
                    ws_b.receive_json()
                    pa, pb = factory.built[0], factory.built[1]
                    conn_a, conn_b = service.connection_ids

                    # Both sessions are mid-response.
                    service.response._ensure_response(conn_a)
                    service.response._ensure_response(conn_b)
                    pa.response_playing.set()
                    pb.response_playing.set()

                    # User speaks in A → A cancels.
                    pa.text_output.put(SpeechStartedEvent())
                    assert _wait_until(lambda: pa.cancel_scope.discarding)

                    # B is untouched: not discarding, still "playing".
                    assert not pb.cancel_scope.discarding
                    assert pb.response_playing.is_set()
                    assert service._state(conn_b).in_response

    def test_disconnect_one_session_leaves_other_running(self):
        """B disconnects mid-response; its pipeline tears down, A keeps working."""
        factory = IndependentSessionStubFactory()
        app, service = _multi_app(factory, max_sessions=2)
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws_a:
                ws_a.receive_json()
                with client.websocket_connect("/v1/realtime") as ws_b:
                    ws_b.receive_json()
                    assert len(service._conns) == 2
                    conn_b = service.connection_ids[1]
                    service.response._ensure_response(conn_b)  # B mid-response
                    factory.built[1].response_playing.set()
                # B disconnected
                assert _wait_until(lambda: len(service._conns) == 1)
                assert conn_b not in service._pipelines

                # A still fully functional.
                pa = factory.built[0]
                pa.text_output.put(TranscriptionCompletedEvent(transcript="still-here"))
                a_done = _recv_until(ws_a, TRANSCRIPTION_COMPLETED_TYPE)
                assert a_done["transcript"] == "still-here"

    def test_keepalive_fires_independently_per_session(self, monkeypatch):
        """Each in-flight session gets its own keepalive on the silent gap."""
        import speech_to_speech.api.openai_realtime.websocket_router as router_mod

        monkeypatch.setattr(router_mod, "HEARTBEAT_S", 0.2)
        factory = IndependentSessionStubFactory()
        app, service = _multi_app(factory, max_sessions=2)
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws_a:
                ws_a.receive_json()
                with client.websocket_connect("/v1/realtime") as ws_b:
                    ws_b.receive_json()
                    conn_a, conn_b = service.connection_ids
                    service.response._ensure_response(conn_a)
                    service.response._ensure_response(conn_b)
                    # Each socket independently sees a keepalive during its gap.
                    assert _recv_until(ws_a, "s2s.keepalive")["response_id"] == service._state(conn_a).current_response_id
                    assert _recv_until(ws_b, "s2s.keepalive")["response_id"] == service._state(conn_b).current_response_id
