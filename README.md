# winoiknow/speech2speech

A fork of [huggingface/speech-to-speech](https://github.com/huggingface/speech-to-speech) extended to run as a **zero-local-inference** OpenAI Realtime-compatible voice agent server.  All speech and language processing is delegated to three external HTTP services — no ML models are loaded in the server process itself.

> 📖 **[Install & Configuration Guide →](docs/INSTALL_AND_CONFIGURATION.md)** — the
> complete, step-by-step setup and configuration reference (every knob, multi-session,
> speaker-id, AEC, Smart Turn, keepalive, troubleshooting). Start there for a real
> deployment.

> For a full description of the underlying pipeline architecture, modular handler system, VAD configuration, and the original model options, see the **[upstream README](https://github.com/huggingface/speech-to-speech/blob/main/README.md)** on the HuggingFace GitHub.

---

## New in 0.3.0

- **Multi-session** — one process serves up to `S2S_MAX_SESSIONS` concurrent warm connections, each with its own isolated `SessionPipeline`. Defaults to `1` (unchanged single-session behavior); only valid when STT/TTS/LLM are all remote.
- **Speaker identification & diarization** — optional, concurrent-with-STT `/v1/identify` tagging and off-hot-path conference diarization (both off by default).
- **Acoustic echo cancellation** — WebRTC AEC3 / speex on the input path with far-aware VAD gating, so a full-duplex client can barge in without phantom triggers.
- **Smart Turn v3 end-of-turn** — optional semantic endpointing (`TURN_DETECTION=smart_turn`) so a mid-thought pause doesn't cut the user off.
- **Warm-connection friendly** — startup pre-warm (VAD / Smart Turn / LLM), once-per-process LLM warmup, configurable WebSocket keepalive, observability (`/v1/sessions`, `/v1/usage`).

See the [Install & Configuration Guide](docs/INSTALL_AND_CONFIGURATION.md) and [CHANGELOG.md](CHANGELOG.md) for details.

---

## What This Fork Adds

| Component | Original | This fork |
|---|---|---|
| STT | Loads Whisper / Parakeet / Paraformer locally | `--stt openai-remote` — HTTP POST to any OpenAI-compatible `/v1/audio/transcriptions` endpoint |
| TTS | Loads Qwen3 / Pocket / Kokoro / ChatTTS locally | `TTS_SOURCE` toggle: `openai-remote` (F5-TTS `/v1/audio/speech/stream`), `elevenlabs` (cloud), or `minimax` (T2A v2 WebSocket, streaming pcm@16k, voice cloning) |
| LLM | Local transformers or OpenAI Responses API | Unchanged — uses `--llm_backend responses-api` pointed at your Hermes / vLLM / compatible server |
| Deployment | No Docker target for CPU-only remote mode | `Dockerfile` + `docker-compose.yml` — CPU-only, no CUDA, no model weights in image |

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
docker compose --env-file .env up --build
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

> The tables below cover the core STT/TTS/LLM/auth options. For the **complete**
> reference — multi-session, speaker-id & diarization, AEC tuning, Smart Turn,
> WebSocket keepalive, the turn-progress heartbeat, observability, and
> troubleshooting — see the **[Install & Configuration Guide](docs/INSTALL_AND_CONFIGURATION.md)**.

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

### TTS source toggle (`TTS_SOURCE`)

Pick where speech is synthesized. In the Compose file `TTS_SOURCE` drives the `--tts` flag, so a single env var flips the source:

| `TTS_SOURCE` | Handler | Use when |
|---|---|---|
| `openai-remote` *(default)* | `RemoteOpenAITTSHandler` → any `/v1/audio/speech/stream` endpoint (F5-TTS) | You run an on-prem / self-hosted TTS server |
| `elevenlabs` | `ElevenLabsTTSHandler` → ElevenLabs cloud TTS | No on-site TTS server; you have an ElevenLabs subscription |
| `minimax` | `MiniMaxTTSHandler` → MiniMax T2A v2 WebSocket | You need **voice cloning** with low latency (~0.5 s TTFB, streaming pcm@16k) |

#### TTS — `openai-remote` (F5-TTS `/v1/audio/speech/stream`)

| CLI flag | Env var | Default | Description |
|---|---|---|---|
| `--tts_openai_base_url` | `TTS_OPENAI_BASE_URL` | `http://localhost:8880` | F5-TTS server base URL |
| `--tts_openai_api_key` | `TTS_OPENAI_API_KEY` | `sk-unused` | Auth key |
| `--tts_openai_voice` | `TTS_OPENAI_VOICE` | `default` | Voice name |
| `--tts_openai_model` | `TTS_OPENAI_MODEL` | `tts-1` | Model name sent in TTS requests |

#### TTS — `elevenlabs` (ElevenLabs cloud)

Streams from `POST /v1/text-to-speech/{voice_id}/stream` over `httpx`, decodes to 16 kHz int16 PCM, and honors barge-in exactly like the F5 handler (early-aborts the upstream stream on cancellation). Args are read from the environment — no CLI flags needed; just set `TTS_SOURCE=elevenlabs` and the keys below.

| Env var | Default | Description |
|---|---|---|
| `ELEVENLABS_API_KEY` | *(none)* | Your ElevenLabs API key (sent as `xi-api-key`) |
| `ELEVENLABS_VOICE_ID` | *(none)* | Voice id to synthesize with |
| `ELEVENLABS_MODEL_ID` | `eleven_flash_v2_5` | Model id — `eleven_flash_v2_5` is the lowest-latency choice; use whatever your plan includes |
| `ELEVENLABS_OUTPUT_FORMAT` | `pcm_16000` | `pcm_<rate>` or `ulaw_8000` (see caveats) |
| `ELEVENLABS_STABILITY` | `0.5` | `voice_settings.stability` (0–1) |
| `ELEVENLABS_SIMILARITY_BOOST` | `0.75` | `voice_settings.similarity_boost` (0–1) |

**Caveats — streaming output format vs. subscription tier:**

- **`pcm_16000` is the cleanest path** — it matches the pipeline rate exactly, so no resampling is done. But **PCM output streaming requires a paid ElevenLabs tier** (Creator and above). On the free tier, requesting a `pcm_*` format will fail.
- **On the free tier, use `ELEVENLABS_OUTPUT_FORMAT=ulaw_8000`.** It's accepted on all tiers; the handler decodes µ-law and upsamples 8 kHz → 16 kHz. Quality is telephone-grade (fine for the Asterisk/AudioSocket path, which is already 8 kHz-clocked).
- Other `pcm_*` rates (`pcm_22050`, `pcm_24000`, `pcm_44100`) work too and are resampled to 16 kHz once over the whole clip.
- **`mp3` formats are intentionally not supported** — they'd need a heavy decoder (ffmpeg); PCM/µ-law cover the realtime path.
- **Latency / cost:** every assistant turn is a cloud round-trip billed against your ElevenLabs character quota. `eleven_flash_v2_5` keeps time-to-first-byte low; heavier models (e.g. `eleven_multilingual_v2`) sound better but add latency. The handler synthesizes a full sentence per call (it does not use ElevenLabs' websocket *input* streaming, since the pipeline hands it complete sentences).

#### TTS — `minimax` (MiniMax T2A v2 WebSocket, voice cloning)

Streams over MiniMax's T2A v2 **WebSocket** (stdlib client, no extra dependency), requesting `pcm` at 16 kHz so the audio drops straight onto the pipeline rate (no resample). It is **streaming-first** — each frame is yielded as it arrives (never buffered) — which is what keeps time-to-first-audio ~0.5 s (cold) / ~0.27 s (warm). Barge-in closes the socket immediately, like the other handlers. Args are env-backed; set `TTS_SOURCE=minimax` and the keys below.

| Env var | Default | Description |
|---|---|---|
| `MINIMAX_API_KEY` | *(none)* | MiniMax API key (sent as Bearer token) |
| `MINIMAX_VOICE_ID` | *(none)* | Cloned voice id, or a system voice (e.g. `English_expressive_narrator`) |
| `MINIMAX_MODEL` | `speech-02-turbo` | Model id (`speech-02-turbo` is low-latency) |
| `MINIMAX_WS_URL` | `wss://api.minimax.io/ws/v1/t2a_v2` | Use the `api.minimaxi.com` host for the mainland endpoint |
| `MINIMAX_GROUP_ID` | *(none)* | Optional; appended as `?GroupId=…` if your account requires it |
| `MINIMAX_SPEED` | `1.0` | `voice_setting.speed` |

**Notes:**

- **The WebSocket path requires a paid MiniMax account** (the free tier rejects the WS handshake). Voice cloning is a paid feature; a **system voice** works for testing the integration before committing to a clone.
- **Why MiniMax here:** it's the low-latency option that *also* supports voice cloning — `pcm_16000` over WS gives ElevenLabs-class TTFB (measured ~0.27 s warm) with a custom voice, where on-prem F5 buffers to ~3–9 s. See `scripts/probe_minimax.py` to measure TTFB from your network before integrating.
- **v1 connects per utterance** (~0.5 s cold TTFB, includes the WS handshake). A warm persistent connection (~0.27 s) is a planned optimization.

### LLM (`--llm_backend responses-api`)

| CLI flag | Env var | Default | Description |
|---|---|---|---|
| `--responses_api_base_url` | `LLM_BASE_URL` | *(OpenAI)* | LLM server base URL |
| `--responses_api_api_key` | `LLM_API_KEY` | *(none)* | Auth key |
| `--model_name` | `LLM_MODEL` | `hermes-agent` | Model identifier sent in every request — must match what the server expects (e.g. `hermes-agent` for Hermes Agent API) |
| `--responses_api_request_timeout_s` | `LLM_REQUEST_TIMEOUT_S` | `60` | Read timeout (s) between stream chunks. Agent backends running tool loops can stream nothing for tens of seconds — keep this above the slowest expected turn |

> **Note on API key separation:** `STT_OPENAI_API_KEY`, `TTS_OPENAI_API_KEY`, and the LLM key are intentionally independent. None falls back to `OPENAI_API_KEY`.

---

## Barge-in / Cancellation

When the user speaks while the assistant is replying, the pipeline's `CancelScope` increments its generation counter. The TTS handler detects this on its next byte read, returns immediately from the generator, and the `httpx` streaming context manager exits — sending TCP FIN to the F5-TTS server and stopping upstream generation. No stale audio is sent to the client after cancellation.

Barge-in also works during the "thinking" gap (after the user stops speaking, before the first audio): the response lifecycle opens at turn start, so new speech cancels the in-flight LLM generation instead of queuing a second response behind it.

---

## Turn progress / keepalive (the "thinking" gap)

Between end-of-user-speech and the first audio chunk, the pipeline is busy (STT → LLM/tool loop → TTS) but the wire would otherwise be silent — indistinguishable from a dead connection for any client. Two signals fix this, for **any** OpenAI-Realtime-compatible client:

1. **Early `response.created`** — emitted the moment the turn's transcription completes and LLM generation is triggered (not on the first audio chunk). Spec-standard; use it to show "working" feedback and arm a turn watchdog.
2. **`s2s.keepalive`** — a custom event `{"type": "s2s.keepalive", "event_id": "...", "response_id": "..."}` emitted every `S2S_HEARTBEAT_S` seconds (default `5`) while a response is in progress and nothing else has been sent. Refresh your watchdog on each one. Clients should ignore unknown event types per Realtime convention (the official OpenAI Python SDK parses it without error); set `S2S_HEARTBEAT_S=0` for strict clients.

---

## Running the Tests

```bash
# Unit / smoke tests for the remote handlers
python -m pytest tests/test_remote_handlers.py -v

# Full test suite
python -m pytest
```

---

## Project Structure

This is a remote-only realtime build — there are no in-process STT/TTS/LLM model
handlers (they were removed in 0.4.0; see the CHANGELOG).

```
src/speech_to_speech/
  api/openai_realtime/   the realtime server, service, websocket router
  pipeline/              SessionPipeline + HandlerFactory (per-connection isolation)
  STT/                   remote_openai_stt_handler.py (+ transcription_notifier)
  TTS/                   remote_openai / elevenlabs / minimax handlers
  LLM/                   responses_api_language_model.py (+ chat, tools, processor)
  VAD/                   silero VAD + Smart Turn v3
  audio/                 echo_canceller.py (AEC3 / speex)
  speaker_id/            remote_speaker_client.py (identify + diarize)
  arguments_classes/     remote handler + module/VAD/speaker-id arg dataclasses
  s2s_pipeline.py        CLI parse + remote handler dispatch + main()

Dockerfile               CPU-only image (no CUDA, no model weights)
docker-compose.yml       the deployment (s2s-remote + optional speaker-id profile)
docs/INSTALL_AND_CONFIGURATION.md   full setup + config guide
REMOTE_SETUP.md          original remote-mode runbook
LATENCY.md               latency + multi-session capacity
CHANGELOG.md             change history
```

---

## License

Apache 2.0 — same as the upstream project.  
Copyright 2024 The HuggingFace Inc. team (original work).  
Modifications Copyright 2026 winoiknow (Eric Alborn, Anteon Group).
