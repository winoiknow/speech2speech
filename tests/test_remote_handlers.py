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
from speech_to_speech.TTS.remote_openai_tts_handler import CHUNK_SAMPLES, RemoteOpenAITTSHandler

# ── helpers ──────────────────────────────────────────────────────────────────

CHUNK_BYTES = CHUNK_SAMPLES * 2  # int16


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


def _make_stream_mock(pcm_chunks: list[bytes]) -> MagicMock:
    """Build a context-manager mock for handler._client.stream(...)."""
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.iter_bytes.return_value = iter(pcm_chunks)
    mock_response.__enter__ = MagicMock(return_value=mock_response)
    mock_response.__exit__ = MagicMock(return_value=False)
    return mock_response


def _make_raw_pcm(n_samples: int) -> bytes:
    """Create n_samples of int16 PCM (sine wave)."""
    t = np.linspace(0, 1, n_samples, dtype=np.float32)
    audio = np.sin(2 * np.pi * 440 * t)
    return (audio * 32767).astype(np.int16).tobytes()


# ── STT tests ────────────────────────────────────────────────────────────────


class TestRemoteOpenAISTTHandler:
    def test_transcribes_audio(self):
        """Handler should POST audio and yield a Transcription with the returned text."""
        handler = _make_stt_handler()
        vad_audio = _make_vad_audio()

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"text": "Hello world", "language": "en"}

        with patch.object(handler._client, "post", return_value=mock_response) as mock_post:
            results = list(handler.process(vad_audio))

        assert len(results) == 1
        assert isinstance(results[0], Transcription)
        assert results[0].text == "Hello world"
        assert results[0].language_code == "en"

        # Verify WAV container and correct multipart fields
        call_kwargs = mock_post.call_args
        files = call_kwargs.kwargs.get("files", {})
        assert "file" in files
        filename, file_obj, content_type = files["file"]
        assert filename == "audio.wav"
        assert content_type == "audio/wav"
        wav_bytes = file_obj.read()
        assert wav_bytes[:4] == b"RIFF"
        assert wav_bytes[8:12] == b"WAVE"
        assert files.get("model", (None, None))[1] == "whisper-1"
        assert files.get("response_format", (None, None))[1] == "verbose_json"

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

    def test_upload_is_wav_container(self):
        """Uploaded bytes must be a valid RIFF/WAV container, not raw PCM."""
        handler = _make_stt_handler()
        audio = np.array([0.5, -0.5, 0.0], dtype=np.float32)
        vad_audio = VADAudio(audio=audio)

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"text": "test"}

        captured_files: dict = {}

        def capture_post(url, **kwargs):
            captured_files.update(kwargs.get("files", {}))
            return mock_response

        with patch.object(handler._client, "post", side_effect=capture_post):
            list(handler.process(vad_audio))

        filename, file_obj, content_type = captured_files["file"]
        wav_bytes = file_obj.read()

        # Must be a RIFF/WAV container
        assert wav_bytes[:4] == b"RIFF", "Upload must start with RIFF"
        assert wav_bytes[8:12] == b"WAVE", "Upload must have WAVE at offset 8"
        assert filename == "audio.wav"
        assert content_type == "audio/wav"

        # PCM payload starts at offset 44; verify sample sign
        pcm_data = wav_bytes[44:]
        assert len(pcm_data) == 3 * 2  # 3 int16 samples
        sent_int16 = np.frombuffer(pcm_data, dtype=np.int16)
        assert sent_int16[0] > 0   # 0.5 → positive
        assert sent_int16[1] < 0   # -0.5 → negative
        assert sent_int16[2] == 0  # 0.0 → zero

    def test_language_code_propagated(self):
        """language field from verbose_json response must be passed to Transcription."""
        handler = _make_stt_handler()
        vad_audio = _make_vad_audio()

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"text": "Bonjour", "language": "fr"}

        with patch.object(handler._client, "post", return_value=mock_response):
            results = list(handler.process(vad_audio))

        assert len(results) == 1
        assert results[0].language_code == "fr"

    def test_missing_language_field_is_none(self):
        """If the server omits 'language' (non-verbose response), language_code is None."""
        handler = _make_stt_handler()
        vad_audio = _make_vad_audio()

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"text": "Hello"}

        with patch.object(handler._client, "post", return_value=mock_response):
            results = list(handler.process(vad_audio))

        assert len(results) == 1
        assert results[0].language_code is None


# ── TTS tests ─────────────────────────────────────────────────────────────────


class TestRemoteOpenAITTSHandler:
    def test_streams_pcm_chunks(self):
        """Handler should stream int16 ndarray chunks from the TTS endpoint."""
        cancel_scope = CancelScope()
        handler = _make_tts_handler(cancel_scope=cancel_scope)

        # 2048 samples = 4 full 512-sample chunks
        raw_pcm = _make_raw_pcm(2048)
        tts_input = TTSInput(text="Hello there", language_code="en")

        mock_response = _make_stream_mock([raw_pcm])

        with patch.object(handler._client, "stream", return_value=mock_response):
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
        """cancel_scope.cancel() after chunk 2 must yield exactly 2 chunks."""
        cancel_scope = CancelScope()
        handler = _make_tts_handler(cancel_scope=cancel_scope)

        # Ten individual CHUNK_BYTES deliveries
        single_chunk = _make_raw_pcm(CHUNK_SAMPLES)
        assert len(single_chunk) == CHUNK_BYTES

        def iter_bytes_with_cancel():
            for i in range(10):
                yield single_chunk
                if i == 1:
                    cancel_scope.cancel()  # cancel after delivering chunk 2

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.iter_bytes.return_value = iter_bytes_with_cancel()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        tts_input = TTSInput(text="A fairly long sentence to synthesize.", language_code="en")

        with patch.object(handler._client, "stream", return_value=mock_response):
            results = list(handler.process(tts_input))

        assert all(isinstance(r, np.ndarray) for r in results)
        assert len(results) == 2

    def test_trailing_chunk_padded_to_chunk_samples(self):
        """A sub-512-sample tail must be zero-padded to exactly CHUNK_SAMPLES."""
        handler = _make_tts_handler()
        # 600 samples: one full chunk (512) + 88-sample tail
        raw_pcm = _make_raw_pcm(600)
        tts_input = TTSInput(text="Short.", language_code="en")

        mock_response = _make_stream_mock([raw_pcm])

        with patch.object(handler._client, "stream", return_value=mock_response):
            results = list(handler.process(tts_input))

        # Should have 2 chunks: one full + one padded
        assert len(results) == 2
        assert len(results[0]) == CHUNK_SAMPLES
        assert len(results[1]) == CHUNK_SAMPLES
        # Padding samples are zero
        assert np.all(results[1][88:] == 0)

    def test_empty_text_yields_nothing(self):
        """Empty or whitespace text should produce no output."""
        handler = _make_tts_handler()
        tts_input = TTSInput(text="   ", language_code="en")
        results = list(handler.process(tts_input))
        assert results == []

    def test_http_error_yields_nothing(self):
        """An HTTP error during streaming should be caught and produce no output."""
        handler = _make_tts_handler()
        tts_input = TTSInput(text="Hello", language_code="en")

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500", request=MagicMock(), response=MagicMock()
        )
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch.object(handler._client, "stream", return_value=mock_response):
            results = list(handler.process(tts_input))

        assert results == []
