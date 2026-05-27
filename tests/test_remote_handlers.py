# Copyright 2026 winoiknow (Eric Alborn, Anteon Group)
# Licensed under the Apache License, Version 2.0 (the "License").
# See the LICENSE file in the repository root for the full license text.

"""
Smoke tests for RemoteOpenAISTTHandler and RemoteOpenAITTSHandler.

Mocks the three external HTTP endpoints and exercises one full turn through
the pipeline: VADAudio → STT → text, text → TTS → PCM chunks.
"""
from __future__ import annotations

from queue import Queue
from threading import Event
from unittest.mock import MagicMock, patch

import httpx
import numpy as np

from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.messages import (
    AUDIO_RESPONSE_DONE,
    EndOfResponse,
    Transcription,
    TTSInput,
    VADAudio,
)
from speech_to_speech.STT.remote_openai_stt_handler import RemoteOpenAISTTHandler
from speech_to_speech.TTS.remote_openai_tts_handler import RemoteOpenAITTSHandler

# ── helpers ──────────────────────────────────────────────────────────────────


def _make_vad_audio(seconds: float = 0.5, sample_rate: int = 16000) -> VADAudio:
    """Return a VADAudio with silent float32 audio."""
    samples = int(seconds * sample_rate)
    audio = np.zeros(samples, dtype=np.float32)
    return VADAudio(audio=audio)


def _make_stt_handler(**kwargs) -> RemoteOpenAISTTHandler:
    stop_event = Event()
    q_in: Queue = Queue()
    q_out: Queue = Queue()
    defaults = dict(
        base_url="http://stt-server",
        api_key="sk-test",
        model="whisper-1",
        language="en",
        timeout=5.0,
    )
    defaults.update(kwargs)
    return RemoteOpenAISTTHandler(stop_event, queue_in=q_in, queue_out=q_out, setup_kwargs=defaults)


def _make_tts_handler(cancel_scope: CancelScope | None = None, **kwargs) -> RemoteOpenAITTSHandler:
    stop_event = Event()
    should_listen = Event()
    q_in: Queue = Queue()
    q_out: Queue = Queue()
    defaults = dict(
        base_url="http://tts-server",
        api_key="sk-test",
        voice="alloy",
        timeout=5.0,
    )
    defaults.update(kwargs)
    if cancel_scope is not None:
        defaults["cancel_scope"] = cancel_scope
    return RemoteOpenAITTSHandler(
        stop_event, queue_in=q_in, queue_out=q_out, setup_args=(should_listen,), setup_kwargs=defaults
    )


# ── STT tests ────────────────────────────────────────────────────────────────


class TestRemoteOpenAISTTHandler:
    def test_transcribes_audio(self):
        """Handler should POST audio and yield a Transcription with the returned text."""
        handler = _make_stt_handler()
        vad_audio = _make_vad_audio()

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"text": "Hello world"}

        with patch.object(handler._client, "post", return_value=mock_response) as mock_post:
            results = list(handler.process(vad_audio))

        assert len(results) == 1
        assert isinstance(results[0], Transcription)
        assert results[0].text == "Hello world"

        # Verify correct multipart fields were sent
        call_kwargs = mock_post.call_args
        files = call_kwargs.kwargs.get("files") or call_kwargs.args[1] if len(call_kwargs.args) > 1 else {}
        files = call_kwargs.kwargs.get("files", {})
        assert "file" in files
        assert files.get("model", (None, None))[1] == "whisper-1"

    def test_empty_transcript_yields_nothing(self):
        """An empty response text should produce no output."""
        handler = _make_stt_handler()
        vad_audio = _make_vad_audio()

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"text": "  "}

        with patch.object(handler._client, "post", return_value=mock_response):
            results = list(handler.process(vad_audio))

        assert results == []

    def test_http_error_yields_nothing(self):
        """An HTTP error should be caught and yield nothing (no crash)."""
        handler = _make_stt_handler()
        vad_audio = _make_vad_audio()

        with patch.object(handler._client, "post", side_effect=httpx.ConnectError("refused")):
            results = list(handler.process(vad_audio))

        assert results == []

    def test_audio_converted_to_int16_pcm(self):
        """VADAudio float32 → int16 PCM bytes must be sent, not raw float32."""
        handler = _make_stt_handler()
        # Audio with a known value so we can verify conversion
        audio = np.array([0.5, -0.5, 0.0], dtype=np.float32)
        vad_audio = VADAudio(audio=audio)

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"text": "test"}

        captured_files = {}

        def capture_post(url, **kwargs):
            captured_files.update(kwargs.get("files", {}))
            return mock_response

        with patch.object(handler._client, "post", side_effect=capture_post):
            list(handler.process(vad_audio))

        # The file field is (filename, file_obj, content_type)
        _, file_obj, _ = captured_files["file"]
        sent_bytes = file_obj.read()
        # Should be 3 int16 samples = 6 bytes
        assert len(sent_bytes) == 3 * 2
        sent_int16 = np.frombuffer(sent_bytes, dtype=np.int16)
        assert sent_int16[0] > 0   # 0.5 → positive
        assert sent_int16[1] < 0   # -0.5 → negative
        assert sent_int16[2] == 0  # 0.0 → zero


# ── TTS tests ─────────────────────────────────────────────────────────────────


def _make_raw_pcm(n_samples: int) -> bytes:
    """Create n_samples of int16 PCM (sine wave)."""
    t = np.linspace(0, 1, n_samples, dtype=np.float32)
    audio = np.sin(2 * np.pi * 440 * t)
    return (audio * 32767).astype(np.int16).tobytes()


class TestRemoteOpenAITTSHandler:
    def test_streams_pcm_chunks(self):
        """Handler should stream int16 ndarray chunks from the TTS endpoint."""
        cancel_scope = CancelScope()
        handler = _make_tts_handler(cancel_scope=cancel_scope)

        # 2048 samples = 4 full 512-sample chunks
        raw_pcm = _make_raw_pcm(2048)
        tts_input = TTSInput(text="Hello there", language_code="en")

        mock_stream_response = MagicMock()
        mock_stream_response.raise_for_status = MagicMock()
        mock_stream_response.iter_bytes.return_value = iter([raw_pcm])
        mock_stream_response.__enter__ = MagicMock(return_value=mock_stream_response)
        mock_stream_response.__exit__ = MagicMock(return_value=False)

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
            mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
            mock_client.stream.return_value = mock_stream_response

            results = list(handler.process(tts_input))

        assert len(results) >= 4
        for chunk in results:
            assert isinstance(chunk, np.ndarray)
            assert chunk.dtype == np.int16

    def test_end_of_response_yields_sentinel(self):
        """EndOfResponse must yield AUDIO_RESPONSE_DONE bytes sentinel."""
        handler = _make_tts_handler()
        results = list(handler.process(EndOfResponse()))
        assert results == [AUDIO_RESPONSE_DONE]

    def test_cancellation_stops_stream(self):
        """When cancel_scope is cancelled mid-stream, yielding should stop."""
        cancel_scope = CancelScope()
        handler = _make_tts_handler(cancel_scope=cancel_scope)

        # Generate enough PCM to exercise the loop
        raw_pcm = _make_raw_pcm(512 * 10)

        mock_stream_response = MagicMock()
        mock_stream_response.raise_for_status = MagicMock()

        yielded_count = 0

        def iter_bytes_with_cancel():
            nonlocal yielded_count
            # yield first chunk, then cancel
            yield raw_pcm[:512 * 2 * 2]
            cancel_scope.cancel()   # signal barge-in
            yield raw_pcm[512 * 2 * 2:]

        mock_stream_response.iter_bytes.return_value = iter_bytes_with_cancel()
        mock_stream_response.__enter__ = MagicMock(return_value=mock_stream_response)
        mock_stream_response.__exit__ = MagicMock(return_value=False)

        tts_input = TTSInput(text="A fairly long sentence to synthesize.", language_code="en")

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
            mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
            mock_client.stream.return_value = mock_stream_response

            results = list(handler.process(tts_input))

        # After cancellation, no further chunks should be emitted for the remainder
        # (exact count is ≤ chunks from the first delivery)
        assert all(isinstance(r, np.ndarray) for r in results)
        assert len(results) < 10  # much fewer than the full 10 chunks

    def test_empty_text_yields_nothing(self):
        """Empty or whitespace text should produce no output."""
        handler = _make_tts_handler()
        tts_input = TTSInput(text="   ", language_code="en")
        with patch("httpx.Client"):
            results = list(handler.process(tts_input))
        assert results == []

    def test_http_error_yields_nothing(self):
        """An HTTP error during streaming should be caught and produce no output."""
        handler = _make_tts_handler()
        tts_input = TTSInput(text="Hello", language_code="en")

        mock_stream_response = MagicMock()
        mock_stream_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500", request=MagicMock(), response=MagicMock()
        )
        mock_stream_response.__enter__ = MagicMock(return_value=mock_stream_response)
        mock_stream_response.__exit__ = MagicMock(return_value=False)

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
            mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
            mock_client.stream.return_value = mock_stream_response

            results = list(handler.process(tts_input))

        assert results == []
