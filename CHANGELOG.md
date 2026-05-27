# Changelog

All notable changes to this fork of [huggingface/speech-to-speech](https://github.com/huggingface/speech-to-speech) are documented here.

---

## [Unreleased] — 2026-05-27

Forked from upstream commit [`99907c8`](https://github.com/huggingface/speech-to-speech/commit/99907c8ce393409ddf1fbc0287c89c2a8a2364ec) (2026-05-26).

### Added

#### `RemoteOpenAISTTHandler` (`--stt openai-remote`)
**File:** `src/speech_to_speech/STT/remote_openai_stt_handler.py`  
**Purpose:** Replaces local Whisper inference with an HTTP POST to any OpenAI-compatible `/v1/audio/transcriptions` endpoint (e.g. faster-whisper-server). Receives VAD-segmented float32 audio, converts to int16 PCM, and POSTs it as multipart form-data. Returns a `Transcription` message to the pipeline. No ML model is loaded in the server process.

#### `RemoteOpenAISTTHandlerArguments`
**File:** `src/speech_to_speech/arguments_classes/remote_openai_stt_arguments.py`  
**Purpose:** Dataclass of CLI arguments (`--stt_openai_base_url`, `--stt_openai_api_key`, `--stt_openai_model`, `--stt_openai_language`) with defaults sourced from environment variables (`STT_OPENAI_*`).

#### `RemoteOpenAITTSHandler` (`--tts openai-remote`)
**File:** `src/speech_to_speech/TTS/remote_openai_tts_handler.py`  
**Purpose:** Replaces local TTS model inference with an HTTP streaming request to a `/v1/audio/speech/stream` endpoint that returns raw 16 kHz int16 mono PCM over chunked transfer (e.g. the winoiknow/openai-f5-tts wrapper). Streams PCM chunks as `np.ndarray[int16]` directly into the audio output queue. Cleanly cancels the upstream HTTP connection (sends TCP FIN to the TTS server) when the pipeline's `CancelScope` signals a barge-in, preventing wasted computation on stale audio.

#### `RemoteOpenAITTSHandlerArguments`
**File:** `src/speech_to_speech/arguments_classes/remote_openai_tts_arguments.py`  
**Purpose:** Dataclass of CLI arguments (`--tts_openai_base_url`, `--tts_openai_api_key`, `--tts_openai_voice`) with defaults sourced from environment variables (`TTS_OPENAI_*`).

#### `Dockerfile.remote`
**Purpose:** CPU-only Docker image for the remote-handler configuration. Installs only the framework core, silero-vad, httpx, numpy, soundfile, and websocket libraries. Explicitly avoids pulling in torch CUDA wheels, transformers, mlx-lm, parakeet, kokoro, qwen3, pocket-tts, or ChatTTS. The resulting image contains no ML model weights.

#### `docker-compose.remote.yml`
**Purpose:** Docker Compose configuration that wires up the remote image with environment-variable configuration for all three external service URLs and credentials. Includes `host.docker.internal` mapping so containerised deployments can reach services running on the host machine.

#### `REMOTE_SETUP.md`
**Purpose:** Operational runbook covering environment variables, Docker Compose usage, how to verify the realtime endpoint is healthy, what a successful first turn looks like in the logs, and a note on LLM API compatibility.

#### `tests/test_remote_handlers.py`
**Purpose:** Smoke test suite (9 tests) for `RemoteOpenAISTTHandler` and `RemoteOpenAITTSHandler`. Mocks all three external HTTP endpoints and exercises: transcription, empty/whitespace audio, HTTP errors, float32→int16 PCM conversion, PCM chunk streaming, barge-in cancellation mid-stream, and the `EndOfResponse` sentinel path.

#### `.env.sample`
**Purpose:** Commented template for all environment variables — server API key, STT/TTS/LLM base URLs and credentials, and log level. Copy to `.env` and fill in values before running via Docker Compose.

### Modified

#### `src/speech_to_speech/arguments_classes/module_arguments.py`
- Added `"openai-remote"` to the `stt` and `tts` `Literal` type sets so the new handlers are valid CLI choices.
- Added `server_api_key` field (`--server_api_key` / `SERVER_API_KEY` env var). When set, the realtime server requires clients to supply `Authorization: Bearer <key>`; connections with a missing or incorrect token are rejected with WebSocket close code `4001`.

#### `src/speech_to_speech/api/openai_realtime/websocket_router.py`
Added Bearer token authentication check in `realtime_endpoint()`: if `server_api_key` is configured, the `Authorization` header is validated immediately after the WebSocket handshake, before any session state is created.

#### `src/speech_to_speech/api/openai_realtime/server.py`
Added `server_api_key` parameter to `RealtimeServer.__init__()` and forwards it to `create_app()`. Logs a startup notice when authentication is enabled.

#### `src/speech_to_speech/s2s_pipeline.py`
- Imported `RemoteOpenAISTTHandlerArguments` and `RemoteOpenAITTSHandlerArguments`.
- Added both argument classes to `ParsedArguments`, `HfArgumentParser`, `parse_arguments()`, `prepare_all_args()`, and `build_pipeline()`.
- Added `remote_openai_tts_handler_kwargs` to the `cancel_scope` injection loop (realtime mode only).
- Extended `get_stt_handler()` with an `"openai-remote"` branch that instantiates `RemoteOpenAISTTHandler`.
- Extended `get_tts_handler()` with an `"openai-remote"` branch that instantiates `RemoteOpenAITTSHandler`.
- Updated the error messages in both dispatch functions to name all valid choices.

#### `tests/test_cli_defaults.py`
Added `RemoteOpenAISTTHandlerArguments` and `RemoteOpenAITTSHandlerArguments` to the `EXPECTED_FIELD_TYPES` registry so the existing `ParsedArguments` completeness tests continue to pass.
