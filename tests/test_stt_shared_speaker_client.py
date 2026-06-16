"""Regression: a session's STT teardown must NOT close the shared speaker client.

The RemoteSpeakerClient is built once by HandlerFactory and shared across every
session. The per-session STT handler holds a reference but does not own it; if its
cleanup() closes it, the first session teardown closes the shared httpx client for
the whole process and every later identify fails with "Cannot send a request, as
the client has been closed" until restart. (Especially easy to hit with a client
that recycles its connection after each turn.)
"""

from queue import Queue
from threading import Event as ThreadingEvent

from speech_to_speech.STT.remote_openai_stt_handler import RemoteOpenAISTTHandler


class _FakeSpeakerClient:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _handler(speaker_client):
    return RemoteOpenAISTTHandler(
        ThreadingEvent(),
        Queue(),
        Queue(),
        setup_kwargs={
            "base_url": "http://stt.invalid",
            "model": "whisper-1",
            "speaker_client": speaker_client,
        },
    )


def test_cleanup_does_not_close_shared_speaker_client():
    spk = _FakeSpeakerClient()
    handler = _handler(spk)
    handler.cleanup()
    # The shared client must survive a single session's teardown.
    assert spk.closed is False
    # The handler's OWN transcription client is closed (its httpx client).
    assert handler._client.is_closed is True


def test_cleanup_without_speaker_client_is_safe():
    handler = _handler(None)
    handler.cleanup()  # must not raise when speaker-id is disabled
    assert handler._client.is_closed is True
