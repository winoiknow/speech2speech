# Remote Handler Setup

This guide covers running the speech-to-speech server in **remote mode**: all
inference is delegated to three external HTTP services.  No ML models are
loaded in the server process itself.

## Architecture

```
WebSocket client
       │
       ▼
  Realtime server (port 8765, /v1/realtime)
       │
       ├─ VAD (silero-vad, runs in-process, CPU only)
       │
       ├─ STT ──► Whisper server  (POST /v1/audio/transcriptions)
       ├─ LLM ──► Hermes          (POST /v1/responses or /v1/chat/completions)
       └─ TTS ──► F5-TTS          (POST /v1/audio/speech/stream → chunked PCM)
```

## Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `STT_OPENAI_BASE_URL` | `http://localhost:8000` | Whisper server base URL |
| `STT_OPENAI_API_KEY` | `sk-unused` | Auth key for STT endpoint |
| `STT_OPENAI_MODEL` | `Systran/faster-whisper-large-v3` | Model name sent in transcription requests |
| `STT_OPENAI_LANGUAGE` | `en` | Language hint (ISO-639-1) |
| `TTS_OPENAI_BASE_URL` | `http://localhost:8880` | F5-TTS server base URL |
| `TTS_OPENAI_API_KEY` | `sk-unused` | Auth key for TTS endpoint |
| `TTS_OPENAI_VOICE` | `default` | Voice name sent to TTS |
| `LLM_BASE_URL` | `http://localhost:11434/v1` | Hermes base URL |
| `LLM_API_KEY` | `sk-unused` | Auth key for LLM endpoint |

**Note on API key separation:** These env vars are intentionally separate from
each other and from `OPENAI_API_KEY`.  Do not mix them — the STT, TTS, and LLM
services may each use different credentials.

## Docker Compose

```bash
# Copy and edit the env file
cp .env.remote.example .env.remote
# edit STT_OPENAI_BASE_URL, TTS_OPENAI_BASE_URL, LLM_BASE_URL, etc.

# Build and start
docker compose -f docker-compose.remote.yml --env-file .env.remote up --build
```

The server starts on `ws://0.0.0.0:8765/v1/realtime`.

### `.env.remote.example`

```
STT_OPENAI_BASE_URL=http://192.168.1.10:8000
STT_OPENAI_API_KEY=sk-unused
STT_OPENAI_MODEL=Systran/faster-whisper-large-v3
STT_OPENAI_LANGUAGE=en

TTS_OPENAI_BASE_URL=http://192.168.1.10:8880
TTS_OPENAI_API_KEY=sk-unused
TTS_OPENAI_VOICE=default

LLM_BASE_URL=http://192.168.1.10:7860/v1
LLM_API_KEY=sk-unused

LOG_LEVEL=info
```

## Running Without Docker

```bash
pip install -e .

speech-to-speech \
  --mode realtime \
  --stt openai-remote \
  --tts openai-remote \
  --llm_backend responses-api \
  --stt_openai_base_url http://localhost:8000 \
  --stt_openai_api_key sk-unused \
  --stt_openai_model Systran/faster-whisper-large-v3 \
  --tts_openai_base_url http://localhost:8880 \
  --tts_openai_api_key sk-unused \
  --tts_openai_voice default \
  --responses_api_base_url http://localhost:7860/v1 \
  --responses_api_api_key sk-unused
```

## Verifying the Endpoint Is Up

```bash
# Should return HTTP 200 with {"message": "WebSocket server running"}
curl http://localhost:8765/

# Connect with the OpenAI Realtime client
npx -y @openai/realtime-api-beta \
  --server ws://localhost:8765/v1/realtime \
  --api-key sk-unused
```

## What a Successful First Turn Looks Like in the Logs

```
INFO  VADHandler: Speech started (confirmed, 520ms buffered)
INFO  VADHandler: Speech ended (840ms), stop listening
DEBUG RemoteOpenAISTTHandler: posting 26880 bytes to http://localhost:8000/v1/audio/transcriptions
INFO  USER: hello there
DEBUG RemoteOpenAISTTHandler: finished in 0.18s
INFO  ResponsesApiModelHandler: generating response
INFO  ASSISTANT: Hi! How can I help you today?
DEBUG RemoteOpenAITTSHandler: time-to-first-byte 0.72s
... (PCM chunks streaming) ...
INFO  response done
```

## LLM Compatibility Note

The `--llm_backend responses-api` flag uses the OpenAI Responses API
(`/v1/responses`).  If your Hermes server only exposes `/v1/chat/completions`,
check whether it also supports the Responses API endpoint — most
OpenAI-compatible servers (vLLM, LM Studio, Ollama) do.  If not, you can run
any proxy that translates `/v1/responses` → `/v1/chat/completions`.

## Cancellation (Barge-in)

When the user speaks while the assistant is replying:
1. VAD raises a speech-started event.
2. The realtime server calls `cancel_scope.cancel()`, incrementing the generation counter.
3. The TTS handler detects `cancel_scope.is_stale(gen)` on its next byte read.
4. It returns from the generator immediately — the `httpx` streaming context manager
   exits and sends TCP FIN to the F5-TTS server, stopping generation there too.

No audio from the cancelled response is sent to the client after this point.
