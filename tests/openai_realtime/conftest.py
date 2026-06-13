from queue import Queue
from threading import Event as ThreadingEvent

import pytest
from openai.types.realtime import RealtimeSessionCreateRequest
from openai.types.realtime.realtime_audio_config import RealtimeAudioConfig
from openai.types.realtime.realtime_audio_config_input import RealtimeAudioConfigInput
from openai.types.realtime.realtime_audio_config_output import RealtimeAudioConfigOutput
from openai.types.realtime.realtime_audio_formats import AudioPCM

from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.api.openai_realtime.service import RealtimeService
from speech_to_speech.audio.echo_canceller import EchoCanceller
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.session_pipeline import SessionPipeline
from speech_to_speech.utils.thread_manager import ThreadManager


class FixedQueueSessionFactory:
    """Test session factory for ``create_app``.

    ``build_session_pipeline`` returns a :class:`SessionPipeline` that wraps
    pre-created queues/events (so tests can inject into / inspect them exactly as
    before the connect-time-build refactor) with **no** real handlers and a no-op
    thread manager. AEC is disabled (pass-through)."""

    def __init__(
        self,
        *,
        recv_audio: Queue,
        send_audio: Queue,
        text_output: Queue,
        should_listen: ThreadingEvent,
        response_playing: ThreadingEvent,
        cancel_scope: CancelScope,
        text_prompt: Queue | None = None,
    ) -> None:
        self.recv_audio = recv_audio
        self.send_audio = send_audio
        self.text_output = text_output
        self.should_listen = should_listen
        self.response_playing = response_playing
        self.cancel_scope = cancel_scope
        self.text_prompt = text_prompt if text_prompt is not None else Queue()
        self.built: list[SessionPipeline] = []

    def build_session_pipeline(self, session_id: str) -> SessionPipeline:
        pipeline = SessionPipeline(
            session_id=session_id,
            recv_audio=self.recv_audio,
            spoken_prompt=Queue(),
            stt_output=Queue(),
            text_prompt=self.text_prompt,
            lm_response=Queue(),
            lm_processed=Queue(),
            send_audio=self.send_audio,
            text_output=self.text_output,
            stop_event=ThreadingEvent(),
            should_listen=self.should_listen,
            response_playing=self.response_playing,
            cancel_scope=self.cancel_scope,
            echo_canceller=EchoCanceller(sample_rate=16000, enabled=False),
            handlers=[],
            threads=ThreadManager([]),
        )
        self.built.append(pipeline)
        return pipeline


def _session_16k() -> RealtimeSessionCreateRequest:
    """Build a test session with 16 kHz audio rates (matches PIPELINE_SAMPLE_RATE)."""
    fmt = AudioPCM.model_construct(rate=16000, type="audio/pcm")
    return RealtimeSessionCreateRequest.model_construct(
        type="realtime",
        audio=RealtimeAudioConfig.model_construct(
            input=RealtimeAudioConfigInput.model_construct(format=fmt),
            output=RealtimeAudioConfigOutput.model_construct(format=fmt),
        ),
    )


@pytest.fixture
def runtime_config():
    cfg = RuntimeConfig()
    cfg.session = _session_16k()
    return cfg


@pytest.fixture
def text_prompt_queue():
    return Queue()


@pytest.fixture
def should_listen():
    ev = ThreadingEvent()
    ev.set()
    return ev


@pytest.fixture
def service(runtime_config, text_prompt_queue, should_listen):
    svc = RealtimeService(
        text_prompt_queue=text_prompt_queue,
        should_listen=should_listen,
    )
    return svc


@pytest.fixture
def conn_id(service, runtime_config):
    cid = service.register()
    service._state(cid).runtime_config = runtime_config
    yield cid
    service.unregister(cid)
