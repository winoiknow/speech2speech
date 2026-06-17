# Changelog

All notable changes to this fork of [huggingface/speech-to-speech](https://github.com/huggingface/speech-to-speech) are documented here.

---

## [0.4.0] — 2026-06-17

Streamlined to a **remote-only realtime** build. Everything that loaded an
in-process model or served a non-realtime transport was removed; the package now
reads as a purpose-built remote realtime server instead of a fork carrying its
parent's full model zoo. **Breaking** (CLI surface narrows; local modes gone).

### Removed

- **Local STT** handlers + arg classes: whisper, whisper-mlx, mlx-audio-whisper,
  faster-whisper, paraformer, parakeet-tdt (and `smart_progressive_streaming`).
- **Local TTS** handlers + arg classes: chatTTS, facebookMMS, pocket, kokoro,
  qwen3 (and the `ref_audio.wav` asset).
- **Local LLM**: the transformers / mlx-lm `LanguageModelHandler` + arg class.
- **Non-realtime modes**: `local` / `socket` / `websocket` transports (the whole
  `connections/` package) + their arg classes, and `HandlerFactory.build()`.
- **Dependencies**: the entire Darwin/MLX block and every local-model dep
  (nano-parakeet, faster-qwen3-tts, kokoro/pocket/chattts/paraformer extras, mlx*,
  misaki, spacy, phonemizer, lingua, miniaudio, sounddevice, soundfile, pillow).
  All `[project.optional-dependencies]` extras removed. `uv` now resolves ~99
  packages and `pip install -e .` is lean.
- The local `Dockerfile`, `Dockerfile.arm64`, and local `docker-compose.yml`.

### Changed

- `ModuleArguments` is remote-only: `stt=openai-remote`, `llm_backend=responses-api`,
  `tts=openai-remote|elevenlabs|minimax`, `mode=realtime`. Added `host`/`port`
  (`S2S_HOST`/`S2S_PORT`) — the realtime server no longer borrows host/port from
  the removed websocket-streamer args.
- `Dockerfile.remote` → `Dockerfile` and `docker-compose.remote.yml` →
  `docker-compose.yml` (now the only build); image pip list aligned with the lean
  deps; added `onnxruntime` to runtime deps (Smart Turn).
- `LLM/utils.py` lazy-imports Pillow/requests inside the VLM-only `image_url_to_pil`
  helper, so the voice path needs neither.

### Kept

- Remote STT (`openai-remote`), TTS (`openai-remote`/`elevenlabs`/`minimax`), LLM
  (`responses-api`), the realtime server, VAD + Smart Turn, AEC, speaker-id, and
  multi-session — all unchanged. `torch`/`torchaudio` (silero VAD), `transformers`
  (Smart Turn + `HfArgumentParser`), and `onnxruntime` remain required.

---

## [0.3.0] — 2026-06-17

First tagged release. Adds **multi-session** support and a set of capabilities that
make s2s a production warm-connection backend for any OpenAI-Realtime client. All
new behavior is **off/at-parity by default** — the shipped defaults preserve the
prior single-session behavior. Full reference:
[`docs/INSTALL_AND_CONFIGURATION.md`](docs/INSTALL_AND_CONFIGURATION.md).

This release also folds in the previously-unreleased *thinking-gap UX* and *QC*
work below.

### Added

- **Multi-session** (`S2S_MAX_SESSIONS`, default `1`). Each WebSocket connection
  gets its own `SessionPipeline` (queues, VAD, echo canceller, cancel scope, six
  handler threads), built at connect and torn down at disconnect. The cap governs
  concurrent **warm connections**, not simultaneous turns. Forced to `1` when any
  of STT/TTS/LLM is an in-process local model. `HandlerFactory.build_session_pipeline()`
  / `build_realtime_server()`; `RealtimeServer`/`create_app` take a session factory.
  (Phases A–D; design of record in `MULTI_SESSION_PLAN.md`.)
- **Observability** — `GET /v1/sessions` (per-session state/age/idle/turns/usage),
  `GET /v1/usage` (`active_sessions` + `per_session`), session-tagged handler thread
  names.
- **Operational hardening** — `S2S_IDLE_TIMEOUT_S` idle reaper (default off),
  `S2S_THREAD_SUPERVISOR_S` dead-thread supervisor, parallel teardown, per-service
  `STT/TTS/LLM_MAX_CONCURRENCY` caps (default `0` = unlimited).
- **Speaker identification** (`SPEAKER_ID_ENABLED`, off by default) — `/v1/identify`
  fired concurrently with STT, confident matches prefix the dialogue with a tag;
  fail-safe to `unknown`. Plus **async diarization** (`SPEAKER_DIARIZE_ENABLED`,
  off the hot path) driving idempotent, revision-versioned corrective events.
- **Acoustic echo cancellation** (`AEC_ENABLED`, off by default) — WebRTC AEC3
  (default) or speex on the input path, with far-aware VAD residual gating
  (`VAD_INPUT_RMS_GATE`, `VAD_INPUT_RMS_GATE_FAR`, `VAD_FAR_SUSTAIN_*`).
- **Smart Turn v3 end-of-turn** (`TURN_DETECTION=smart_turn`) — semantic
  endpointing (int8 ONNX) so a mid-thought pause doesn't cut the user off; loaded
  once and shared across sessions.
- **MiniMax TTS** (`TTS_SOURCE=minimax`) — T2A v2 WebSocket, streaming pcm@16k,
  voice cloning (~0.27 s warm TTFB).
- **Configurable WebSocket keepalive** — `S2S_WS_PING_INTERVAL` (default `20`,
  `0` disables) / `S2S_WS_PING_TIMEOUT` (default `60`), so warm clients can refresh
  infrequently and a long tool turn won't trip a false `1011`.
- **Startup pre-warm** — Silero VAD (deepcopied per session, bleed-safe), the shared
  Smart Turn model, and a once-per-process LLM warmup (`S2S_LLM_WARMUP_PER_SESSION=0`),
  so a session build (and any reconnect) is sub-second after startup.
- **`docs/INSTALL_AND_CONFIGURATION.md`** — complete install + configuration guide,
  linked from the README.

### Changed

- Per-connection pipeline build runs **off the asyncio event loop**
  (`asyncio.to_thread`) so one client's connect can't freeze other sessions.
- `S2S_LLM_WARMUP_PER_SESSION` defaults to off — the LLM warmup round-trip runs once
  per process instead of on every (re)connect.

### Fixed

- **Shared speaker-id client closed on per-session teardown** — the per-session STT
  handler's `cleanup()` closed the *shared* `RemoteSpeakerClient`, so the first
  disconnect killed identify process-wide (`Cannot send a request, as the client has
  been closed`) until restart. Now it closes only its own transcription client; the
  shared client is closed once at server shutdown.
- **Unsendable `aec3_py::Aec3` dropped on the wrong thread at teardown** — the AEC
  object is now released on the event-loop thread that created it, not the teardown
  worker.
- **Over-eager speaker-id failure warning** — a transient slow/timed-out identify
  (fail-safe → `unknown`, then recovers) no longer logs an outage line; warns only
  after several consecutive failures.
- **Benign disconnect noise** — a client that hangs up during connect no longer logs
  a scary `RuntimeError` traceback.
- Repo-wide `ruff check .` is clean.

### Notes on usage

- **Warm connections:** hold one WebSocket open across turns to skip the connect
  cost and to get multi-turn memory (a new connection = a fresh, empty chat by
  design). Keep a client-side keepalive inside the ping window, or set
  `S2S_WS_PING_INTERVAL=0` and own liveness yourself.
- **Capacity:** `S2S_MAX_SESSIONS` caps warm connections; size it to expected
  clients + headroom. Concurrent *active turns* are bounded by your external
  STT/TTS/LLM endpoints — see `LATENCY.md` §7 and the `*_MAX_CONCURRENCY` caps.

---

## [Unreleased] — 2026-06-12 (thinking-gap UX)

s2s is a client-agnostic realtime backend; these changes are designed for any OpenAI-Realtime-compatible client (voice assistants, smart speakers, Matrix/Element Call bridges, …), not any single one.

### Added

- **Early `response.created` at turn start** (`service.py`, `handlers/response.py` — `begin_turn_response`): the response lifecycle now opens the moment a turn's transcription completes and the LLM is triggered, instead of on the first audio chunk. Clients get an immediate "working" signal at the start of the otherwise-silent STT→LLM→TTS gap (measured 25–30 s on tool-free agent turns). `output_item.added` / `content_part.added` still fire with the first audio or transcript (idempotent lifecycle flags unchanged).
- **`s2s.keepalive` event** (`websocket_router.py`): while a response is `in_progress` and nothing has been sent for `S2S_HEARTBEAT_S` seconds (default 5; `0` disables), the server emits `{"type": "s2s.keepalive", "event_id": ..., "response_id": ...}`. Lets clients distinguish a slow agent turn (LLM/tool loop in flight) from a dead connection and refresh turn watchdogs — downstream watchdogs can be tightened back down. Verified the official OpenAI Python SDK parses the unknown event type without error; strict clients can disable with `S2S_HEARTBEAT_S=0`.
- **`--responses_api_request_timeout_s` / `LLM_REQUEST_TIMEOUT_S`** (`responses_api_language_model_arguments.py`): the LLM read timeout is now configurable; default raised 20 → 60 s. The old hardcoded 20 s was a streaming *read* timeout that could fire mid-turn on agent backends running tool loops (which stream nothing for tens of seconds), cutting off legitimate turns with the "slow today" apology.

### Fixed

- **Barge-in during the thinking gap**: with `in_response` now set at turn start, speech detected while the LLM is in flight cancels the in-flight response (via `cancel_scope`) instead of silently stacking a second generation behind it (previously produced back-to-back responses).
- **Stale `__RESPONSE_DONE__` race** (`websocket_router.py`): a cancelled generation's done-sentinel arriving while `discarding` no longer closes the *next* (already-created) response; it only clears the discard guard. Its `response.done(cancelled)` was already emitted at barge-in time.
- **Silent LLM failures now speak** (`responses_api_language_model.py`): a generic generation error yields a spoken apology (like the existing timeout path) instead of leaving the user in dead silence. Suppressed when the generation was cancelled.
- **`X-Sample-Rate` parse hardening** (`remote_openai_tts_handler.py`): non-numeric or implausible (outside 4 kHz–192 kHz) header values are rejected instead of triggering absurd resample ratios.

### Changed

- **Send loop routes to the owning session** (`websocket_router.py`): pipeline output (text events, audio deltas, finish events) is sent to the active session's websocket instead of broadcast over all connection ids. Behavior-identical with the single-session guard in place; removes one blocker on the multi-session path (TODO #2).

### Tests

- Repaired 23 rotted tests across `tests/openai_realtime/` (lifecycle `output_item.added`/`content_part.added` events, 4-tuple AEC input queue payload, 24 kHz default client rate, 20 ms audio batching), `tests/test_cli_defaults.py` (missing elevenlabs/minimax kwargs), and `tests/test_remote_handlers.py` (MagicMock headers parsed as sample rate 1 Hz; whole-clip TTS cancellation semantics).
- New coverage: `TestThinkingGap` (early created, keepalive emission/suppression, barge-in during gap, stale-done race) and `TestSDKKeepalive` (official SDK tolerates `s2s.keepalive`).

---

## [Unreleased] — 2026-05-27 (QC pass 2)

### Fixed

- **B1 STT upload format** (`remote_openai_stt_handler.py`): raw headerless PCM replaced with a hand-rolled RIFF/WAV container (`_pcm_to_wav`); upload tuple is now `("audio.wav", buf, "audio/wav")`. Added `response_format=verbose_json` to the multipart form so the server returns the `language` field (see N4).
- **B2 Docker image on start** (`Dockerfile.remote`): added `transformers>=4.57.0`, `pillow>=10.0.0`, and `sounddevice>=0.5.0` to the explicit pip-install block. The image previously failed with `ModuleNotFoundError` because `s2s_pipeline.py` unconditionally imports `HfArgumentParser` from `transformers`.
- **B3 Compose version key** (`docker-compose.remote.yml`): removed stale `version: "3.9"` line that Compose v2 warns about on every `up`.

### Improved

- **N1 TTS persistent HTTP client** (`remote_openai_tts_handler.py`): `httpx.Client` is now created once in `setup()` and closed in `cleanup()`, matching the STT handler. Eliminates a TCP/TLS handshake per turn.
- **N2 TTS trailing chunk padding** (`remote_openai_tts_handler.py`): sub-512-sample tail is now zero-padded to `CHUNK_SAMPLES` for downstream alignment.
- **N3 TTS model configurable** (`remote_openai_tts_handler.py`, `remote_openai_tts_arguments.py`): added `tts_openai_model` arg (`--tts_openai_model` / `TTS_OPENAI_MODEL`, default `tts-1`). Hardcoded `"tts-1"` in the POST payload replaced with `self.model`.
- **N4 language_code propagation** (`remote_openai_stt_handler.py`): `verbose_json` response `language` field is now forwarded as `Transcription(language_code=...)`.
- **N5 Cancellation test tightened** (`tests/test_remote_handlers.py`): mock yields ten individual `CHUNK_BYTES` chunks, cancels after chunk 2, asserts exactly 2 results (was `< 10`).

### Tests

- `test_audio_converted_to_int16_pcm` → replaced by `test_upload_is_wav_container` (asserts `RIFF` magic, `WAVE` at offset 8, PCM values at offset 44).
- Added `test_language_code_propagated` and `test_missing_language_field_is_none`.
- Added `test_trailing_chunk_padded_to_chunk_samples`.
- TTS tests updated to patch `handler._client.stream` directly (persistent client pattern).

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
