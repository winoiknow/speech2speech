# winoiknow/speech2speech

A fork of [huggingface/speech-to-speech](https://github.com/huggingface/speech-to-speech) extended to run as a **zero-local-inference** OpenAI Realtime-compatible voice agent server.  All speech and language processing is delegated to three external HTTP services — no ML models are loaded in the server process itself.

> For a full description of the underlying pipeline architecture, modular handler system, VAD configuration, and the original model options, see the **[upstream README](https://github.com/huggingface/speech-to-speech/blob/main/README.md)** on the HuggingFace GitHub.

---

## What This Fork Adds

| Component | Original | This fork |
|---|---|---|
| STT | Loads Whisper / Parakeet / Paraformer locally | `--stt openai-remote` — HTTP POST to any OpenAI-compatible `/v1/audio/transcriptions` endpoint |
| TTS | Loads Qwen3 / Pocket / Kokoro / ChatTTS locally | `--tts openai-remote` — streams raw 16 kHz int16 PCM from any `/v1/audio/speech/stream` endpoint |
| LLM | Local transformers or OpenAI Responses API | Unchanged — uses `--llm_backend responses-api` pointed at your Hermes / vLLM / compatible server |
| Deployment | No Docker target for CPU-only remote mode | `Dockerfile.remote` + `docker-compose.remote.yml` — CPU-only, no CUDA, no model weights in image |

See [CHANGELOG.md](CHANGELOG.md) for a full list of changes.

---

## Quick Start

### Prerequisites

Three services must be reachable on your network:

| Service | Expected API | Default port used here |
|---|---|---|
| Whisper STT | OpenAI-compatible `POST /v1/audio/transcriptions` | 8000 |
| F5-TTS ([winoiknow/openai-f5-tts](https://github.com/winoiknow/openai-f5-tts)) | `POST /v1/audio/speech/stream` → chunked raw 16 kHz int16 PCM | 8880 |
| LLM (Hermes / vLLM / any) | OpenAI-compatible `POST /v1/responses` or `POST /v1/chat/completions` | 7860 |

### Docker Compose (recommended)

```bash
# 1. Clone this repo
git clone git@github.com:winoiknow/speech2speech.git
cd speech2speech

# 2. Copy the sample env file and edit for your environment
cp .env.sample .env
# Set STT_OPENAI_BASE_URL, TTS_OPENAI_BASE_URL, LLM_BASE_URL, SERVER_API_KEY, etc.

# 3. Build and run
docker compose -f docker-compose.remote.yml --env-file .env up --build
```

The server starts on `ws://0.0.0.0:8765/v1/realtime`.

### Without Docker

```bash
pip install -e .

speech-to-speech \
  --mode realtime \
  --stt openai-remote \
  --tts openai-remote \
  --llm_backend responses-api \
  --stt_openai_base_url http://localhost:8000 \
  --stt_openai_model Systran/faster-whisper-large-v3 \
  --tts_openai_base_url http://localhost:8880 \
  --tts_openai_voice default \
  --responses_api_base_url http://localhost:7860/v1 \
  --server_api_key my-secret-key
```

---

## Configuration Reference

All options can be set via CLI flags or environment variables.  Copy `.env.sample` to `.env` and fill in your values.

### Server Authentication

| CLI flag | Env var | Default | Description |
|---|---|---|---|
| `--server_api_key` | `SERVER_API_KEY` | *(unset — auth disabled)* | Bearer token clients must supply in `Authorization: Bearer <key>`. Omit or leave empty to run without authentication. |

### STT (`--stt openai-remote`)

| CLI flag | Env var | Default | Description |
|---|---|---|---|
| `--stt_openai_base_url` | `STT_OPENAI_BASE_URL` | `http://localhost:8000` | Whisper server base URL |
| `--stt_openai_api_key` | `STT_OPENAI_API_KEY` | `sk-unused` | Auth key |
| `--stt_openai_model` | `STT_OPENAI_MODEL` | `Systran/faster-whisper-large-v3` | Model name sent in requests |
| `--stt_openai_language` | `STT_OPENAI_LANGUAGE` | `en` | Language hint (ISO-639-1) |

### TTS (`--tts openai-remote`)

| CLI flag | Env var | Default | Description |
|---|---|---|---|
| `--tts_openai_base_url` | `TTS_OPENAI_BASE_URL` | `http://localhost:8880` | F5-TTS server base URL |
| `--tts_openai_api_key` | `TTS_OPENAI_API_KEY` | `sk-unused` | Auth key |
| `--tts_openai_voice` | `TTS_OPENAI_VOICE` | `default` | Voice name |
| `--tts_openai_model` | `TTS_OPENAI_MODEL` | `tts-1` | Model name sent in TTS requests |

### LLM (`--llm_backend responses-api`)

| CLI flag | Env var | Default | Description |
|---|---|---|---|
| `--responses_api_base_url` | `LLM_BASE_URL` | *(OpenAI)* | LLM server base URL |
| `--responses_api_api_key` | `LLM_API_KEY` | *(none)* | Auth key |
| `--model_name` | `LLM_MODEL` | `hermes-agent` | Model identifier sent in every request — must match what the server expects (e.g. `hermes-agent` for Hermes Agent API) |

> **Note on API key separation:** `STT_OPENAI_API_KEY`, `TTS_OPENAI_API_KEY`, and the LLM key are intentionally independent. None falls back to `OPENAI_API_KEY`.

---

## Barge-in / Cancellation

When the user speaks while the assistant is replying, the pipeline's `CancelScope` increments its generation counter. The TTS handler detects this on its next byte read, returns immediately from the generator, and the `httpx` streaming context manager exits — sending TCP FIN to the F5-TTS server and stopping upstream generation. No stale audio is sent to the client after cancellation.

---

## Running the Tests

```bash
# Unit / smoke tests for the remote handlers
python -m pytest tests/test_remote_handlers.py -v

# Full test suite
python -m pytest
```

---

## Project Structure (additions only)

```
src/speech_to_speech/
  STT/
    remote_openai_stt_handler.py     ← new: RemoteOpenAISTTHandler
  TTS/
    remote_openai_tts_handler.py     ← new: RemoteOpenAITTSHandler
  arguments_classes/
    remote_openai_stt_arguments.py   ← new: CLI args for STT handler
    remote_openai_tts_arguments.py   ← new: CLI args for TTS handler
    module_arguments.py              ← modified: added "openai-remote" to stt/tts Literals
  s2s_pipeline.py                    ← modified: handler dispatch, registration

Dockerfile.remote                    ← new: CPU-only Docker image
docker-compose.remote.yml            ← new: Docker Compose for remote mode
REMOTE_SETUP.md                      ← new: detailed runbook
CHANGELOG.md                         ← new: change history

tests/
  test_remote_handlers.py            ← new: smoke tests for remote handlers
```

---

## License

Apache 2.0 — same as the upstream project.  
Copyright 2024 The HuggingFace Inc. team (original work).  
Modifications Copyright 2026 winoiknow (Eric Alborn, Anteon Group).
