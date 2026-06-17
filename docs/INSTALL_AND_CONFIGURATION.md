# Install & Configuration Guide

This is the complete setup and configuration reference for **winoiknow/speech2speech** —
a zero-local-inference, OpenAI Realtime–compatible voice agent server. The server
process loads **no ML models** for STT/TTS/LLM; all of that work is delegated to
external HTTP/WebSocket services. The only thing that runs in-process is a tiny CPU
VAD (silero, ~2 MB) and, optionally, the Smart Turn end-of-turn model (int8 ONNX).

s2s is **client-agnostic**: it speaks the OpenAI Realtime protocol over WebSocket,
so any compatible client works (browser test clients, smart speakers, an
agent-voice-assistant, a Matrix/Element Call bridge, …). Nothing below assumes a
particular client.

- [1. Architecture](#1-architecture)
- [2. Prerequisites](#2-prerequisites)
- [3. Install](#3-install)
  - [3.1 Docker Compose (recommended)](#31-docker-compose-recommended)
  - [3.2 Bare metal / venv](#32-bare-metal--venv)
- [4. Configuration reference](#4-configuration-reference)
  - [4.1 Server & authentication](#41-server--authentication)
  - [4.2 STT](#42-stt)
  - [4.3 TTS (source toggle)](#43-tts-source-toggle)
  - [4.4 LLM](#44-llm)
  - [4.5 Turn detection (VAD / Smart Turn)](#45-turn-detection-vad--smart-turn)
  - [4.6 Acoustic echo cancellation (AEC)](#46-acoustic-echo-cancellation-aec)
  - [4.7 Speaker identification & diarization](#47-speaker-identification--diarization)
  - [4.8 Multi-session](#48-multi-session)
  - [4.9 WebSocket keepalive & warm connections](#49-websocket-keepalive--warm-connections)
  - [4.10 Turn progress / keepalive event](#410-turn-progress--keepalive-event)
  - [4.11 General / logging](#411-general--logging)
- [5. Verifying a deployment](#5-verifying-a-deployment)
- [6. Observability](#6-observability)
- [7. Troubleshooting](#7-troubleshooting)
- [8. Running the tests](#8-running-the-tests)

---

## 1. Architecture

```
                WebSocket client (OpenAI Realtime protocol)
                              │  ws://HOST:8765/v1/realtime
                              ▼
        ┌──────────────────────────────────────────────┐
        │  s2s realtime server (FastAPI + uvicorn)       │
        │  per-connection SessionPipeline:               │
        │    recv → VAD → STT → LLM → TTS → send          │
        │  + AEC (optional), Smart Turn (optional),      │
        │    speaker-id (optional)                       │
        └──────────────────────────────────────────────┘
            │              │                  │
            ▼              ▼                  ▼
     STT  /v1/audio/  LLM /v1/responses  TTS  F5 /v1/audio/speech/stream
     transcriptions   (Hermes/vLLM/…)    or ElevenLabs / MiniMax (WS)
            ▲
            └─ speaker-id /v1/identify (optional, concurrent with STT)
```

Each WebSocket connection gets its **own** `SessionPipeline` — fresh queues, VAD,
echo canceller, cancel scope, and six handler threads — built at connect and torn
down at disconnect. Everything dangerous to share is per-session; everything shared
(parsed config, the realtime service, a single pre-warmed VAD template / Smart Turn
model, the speaker-id HTTP client, global usage metrics) is read-only or locked.

---

## 2. Prerequisites

Three external services must be reachable from the s2s host. The ports below are
**examples only** — set the real URLs in your `.env`.

| Service | Expected API | Example |
|---|---|---|
| **STT** (Whisper-compatible) | `POST /v1/audio/transcriptions` (multipart, returns `verbose_json`) | `http://stt-host:8000` |
| **TTS** | F5-TTS `POST /v1/audio/speech/stream` (chunked 16 kHz int16 PCM) **or** ElevenLabs cloud **or** MiniMax T2A v2 WebSocket | `http://tts-host:8880` |
| **LLM** | OpenAI-compatible `POST /v1/responses` (or `/v1/chat/completions`) | `http://llm-host:11434/v1` |

Multi-session (`S2S_MAX_SESSIONS > 1`) is **only** valid when STT, TTS, and LLM are
all remote (as above). If any is an in-process local model, s2s forces
`S2S_MAX_SESSIONS=1` with a warning — multi-session needs the model behind a serving
endpoint, not in this process.

Host requirements: Docker + Docker Compose v2 (for the container path) **or** Python
3.10–3.12 (for the bare-metal path). No GPU is needed by s2s itself.

---

## 3. Install

### 3.1 Docker Compose (recommended)

```bash
# 1. Clone
git clone git@github.com:winoiknow/speech2speech.git
cd speech2speech

# 2. Configure
cp .env.sample .env
#   Edit .env — at minimum set STT_OPENAI_BASE_URL, TTS_OPENAI_BASE_URL,
#   LLM_BASE_URL, and (for any auth'd client) SERVER_API_KEY.

# 3. Build + run
docker compose --env-file .env up --build
```

The server listens on `ws://0.0.0.0:8765/v1/realtime`.

> **Compose gotcha:** a variable in `.env` only reaches the container if it is
> listed in the service's `environment:` block in `docker-compose.yml`.
> All documented knobs are already wired there as `${VAR:-default}`. If you add a
> brand-new env var, add it to that block too, or it will be silently ignored
> inside the container (`.env` alone only does `${VAR}` substitution in the compose
> file).

To run the bundled speaker-id service alongside s2s:

```bash
docker compose --profile speaker-id up --build
# then set SPEAKER_ID_ENABLED=1 and SPEAKER_ID_BASE_URL=http://speaker-id:9100
```

### 3.2 Bare metal / venv

```bash
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e .

speech-to-speech \
  --mode realtime \
  --stt openai-remote \
  --tts openai-remote \
  --llm_backend responses-api \
  --stt_openai_base_url http://stt-host:8000 \
  --stt_openai_model Systran/faster-whisper-large-v3 \
  --tts_openai_base_url http://tts-host:8880 \
  --tts_openai_voice default \
  --responses_api_base_url http://llm-host:11434/v1 \
  --model_name hermes-agent \
  --server_api_key my-secret-key
```

Every CLI flag has an environment-variable equivalent (see §4); env vars are the
intended path for Docker, CLI flags for ad-hoc runs. Where both are set, the CLI
flag wins.

---

## 4. Configuration reference

All knobs live in `.env.sample` (copy to `.env`). This section groups them by
concern and explains the *why*, not just the default. Defaults shown are the
shipped defaults — and they preserve today's single-session behavior.

### 4.1 Server & authentication

| Env var | Default | Description |
|---|---|---|
| `SERVER_API_KEY` | *(empty = auth off)* | Bearer token incoming clients must send as `Authorization: Bearer <key>`. Empty disables auth; a bad token is rejected with WS close `4001`. |

The server always binds `0.0.0.0:8765`. Put it behind your own TLS terminator /
reverse proxy if you expose it beyond a trusted LAN.

### 4.2 STT

`--stt openai-remote` posts VAD-segmented audio (as a WAV container) to an
OpenAI-compatible transcription endpoint.

| Env var | Default | Description |
|---|---|---|
| `STT_OPENAI_BASE_URL` | `http://localhost:8000` | Whisper server base URL |
| `STT_OPENAI_API_KEY` | `sk-unused` | Auth key for the STT service |
| `STT_OPENAI_MODEL` | `Systran/faster-whisper-large-v3` | Model name sent in requests |
| `STT_OPENAI_LANGUAGE` | `en` | Language hint (ISO-639-1); `auto` to let the server detect |

### 4.3 TTS (source toggle)

`TTS_SOURCE` selects where speech is synthesized; in Compose it also drives the
`--tts` flag, so one env var flips the backend.

| `TTS_SOURCE` | Backend | Use when |
|---|---|---|
| `openai-remote` *(default)* | F5-TTS `/v1/audio/speech/stream` | You self-host a TTS server |
| `elevenlabs` | ElevenLabs cloud | No on-site TTS; you have an ElevenLabs plan |
| `minimax` | MiniMax T2A v2 WebSocket | You need **voice cloning** at low latency (~0.27 s warm TTFB) |

**openai-remote (F5-TTS):**

| Env var | Default | Description |
|---|---|---|
| `TTS_OPENAI_BASE_URL` | `http://localhost:8880` | F5-TTS base URL |
| `TTS_OPENAI_API_KEY` | `sk-unused` | Auth key |
| `TTS_OPENAI_VOICE` | `default` | Voice name |
| `TTS_OPENAI_MODEL` | `tts-1` | Model name sent in TTS requests |

**elevenlabs** (env-only; no CLI flags — set `TTS_SOURCE=elevenlabs` + keys):

| Env var | Default | Description |
|---|---|---|
| `ELEVENLABS_API_KEY` | *(none)* | Sent as `xi-api-key` |
| `ELEVENLABS_VOICE_ID` | *(none)* | Voice to synthesize with |
| `ELEVENLABS_MODEL_ID` | `eleven_flash_v2_5` | Lowest-latency model |
| `ELEVENLABS_OUTPUT_FORMAT` | `pcm_16000` | `pcm_16000` (cleanest, **paid tier**) or `ulaw_8000` (free-tier; decoded + upsampled). `mp3` unsupported. |
| `ELEVENLABS_STABILITY` | `0.5` | `voice_settings.stability` (0–1) |
| `ELEVENLABS_SIMILARITY_BOOST` | `0.75` | `voice_settings.similarity_boost` (0–1) |

> `pcm_*` streaming requires a paid ElevenLabs tier. On the free tier use
> `ulaw_8000` (telephone-grade, fine for an 8 kHz Asterisk/AudioSocket path). Every
> turn is a billed cloud round-trip.

**minimax** (env-only; set `TTS_SOURCE=minimax` + keys):

| Env var | Default | Description |
|---|---|---|
| `MINIMAX_API_KEY` | *(none)* | Bearer token |
| `MINIMAX_VOICE_ID` | *(none)* | Cloned voice id, or a system voice (e.g. `English_expressive_narrator`) |
| `MINIMAX_MODEL` | `speech-02-turbo` | Low-latency model |
| `MINIMAX_WS_URL` | `wss://api.minimax.io/ws/v1/t2a_v2` | Use `api.minimaxi.com` for the mainland endpoint |
| `MINIMAX_GROUP_ID` | *(none)* | Optional `?GroupId=…` if your account requires it |
| `MINIMAX_SPEED` | `1.0` | `voice_setting.speed` |

> The MiniMax WS path requires a **paid** account (free tier rejects the
> handshake). `scripts/probe_minimax.py` measures TTFB from your network.

**Batching (all TTS backends):**

| Env var | Default | Description |
|---|---|---|
| `STREAM_BATCH_SENTENCES` | `3` | Sentences buffered before the first TTS call fires. Lower = lower time-to-first-audio, at the cost of more, shorter TTS calls. |

### 4.4 LLM

`--llm_backend responses-api` calls an OpenAI-compatible endpoint **statelessly** —
s2s sends the full per-session conversation each turn (no server-side conversation
id), so context is held entirely in the per-connection chat. (See §4.8 on what this
means for multi-turn memory.)

| Env var | Default | Description |
|---|---|---|
| `LLM_BASE_URL` | *(OpenAI)* | LLM server base URL |
| `LLM_API_KEY` | *(none)* | Auth key |
| `LLM_MODEL` | `hermes-agent` | Model id sent every request — must match what the server expects |
| `LLM_REQUEST_TIMEOUT_S` | `60` | Read timeout (s) between stream chunks. Agent tool loops can stream nothing for tens of seconds — keep this above your slowest turn. |
| `S2S_LLM_WARMUP_PER_SESSION` | `0` | The warmup round-trip runs **once per process** by default (warmed at startup; reconnects skip it). Set `1` to warm on every session build — only needed if the remote model goes cold between turns. |

> API keys are independent: `STT_OPENAI_API_KEY`, `TTS_OPENAI_API_KEY`,
> `LLM_API_KEY`, and `SERVER_API_KEY` never fall back to `OPENAI_API_KEY`.

### 4.5 Turn detection (VAD / Smart Turn)

| Env var | Default | Description |
|---|---|---|
| `TURN_DETECTION` | `vad` | `vad` = end turn after a fixed silence; `smart_turn` = Pipecat Smart Turn v3 (semantic endpointing — reads prosody/content to tell "done" from "just pausing"). |
| `TURN_MIN_SILENCE_MS` | `300` | *(smart_turn)* silence before asking the model |
| `TURN_MAX_S` | `30` | *(smart_turn)* hard per-turn ceiling so a turn can never hang |
| `SMART_TURN_THRESHOLD` | `0.5` | *(smart_turn)* completeness probability (0–1); higher = wait for stronger end-of-turn evidence |
| `TURN_HOLD_GRACE_MS` | `3000` | *(smart_turn)* don't-hang net: after an "incomplete" verdict, finalize anyway if the user stays silent this long. Must sit **above** a natural mid-sentence pause (~1.5–2.5 s) or it cuts you off. |
| `SMART_TURN_MODEL_PATH` | `/app/models/smart-turn-v3.2-cpu.onnx` | *(smart_turn)* ONNX model baked into the image |
| `SMART_TURN_NUM_THREADS` | `2` | *(smart_turn)* explicit ONNX intra-op threads — setting this disables CPU-affinity pinning that fails under a container cpuset |

The Smart Turn model is loaded **once at startup** and shared across all sessions
(its inference is stateless), so enabling it adds no per-connection load cost.

### 4.6 Acoustic echo cancellation (AEC)

Off by default. Subtracts the agent's own TTS from the caller's mic **before** the
VAD, so the VAD stops tripping on echo (phantom barge-ins / deaf-during-playback).
Requires a full-duplex client that actually delivers mic audio during playback.
Fail-safe — if the backend can't load, mic passes through unchanged.

| Env var | Default | Description |
|---|---|---|
| `AEC_ENABLED` | `0` | Master switch |
| `AEC_BACKEND` | `aec3` | `aec3` (WebRTC AEC3, delay-estimating, handles networked cross-clock echo) or `speex` (libspeexdsp adaptive filter) |
| `AEC_FILTER_LENGTH_MS` | `250` | *(speex only)* filter tail; must cover the echo round-trip delay |
| `VAD_INPUT_RMS_GATE` | `100` | Reject raw mic chunks below this RMS before the VAD. Tune with the `DEBUG_MODE` `maxpass` heartbeat: set above echo maxpass, below your speech level. `0` disables. |
| `VAD_INPUT_RMS_GATE_FAR` | `400` | While the agent speaks, gate on the AEC **residual** at this threshold instead of raw energy. Bias higher to avoid phantom barge-ins cancelling the agent. `0` disables far gating. |
| `VAD_FAR_SUSTAIN_MIN` | `6` | While the agent speaks, require ≥ MIN of the last WINDOW chunks above the residual gate — rejects sparse transient echo spikes, passes dense real speech. |
| `VAD_FAR_SUSTAIN_WINDOW` | `12` | Window size for the sustained-energy filter |

> Tuning AEC is iterative. Turn on `DEBUG_MODE=on` and watch the `far=N maxpass=M`
> VAD heartbeat during an agent turn: `M` is the echo-residual ceiling — set
> `VAD_INPUT_RMS_GATE_FAR` just above it. Lower the gates if a real barge-in is
> missed; raise them if phantoms persist.

### 4.7 Speaker identification & diarization

Off by default. When on, s2s fires the speaker-id service's `/v1/identify`
**concurrently** with transcribe (overlapping round-trips → ~0 added latency,
reusing the raw user audio) and, on a confident `known` match, prefixes the
dialogue with a tag. Any timeout/error → `unknown`; the turn never blocks. Run the
speaker-id service first (its own repo, or `docker compose --profile speaker-id`)
and enroll a voice.

| Env var | Default | Description |
|---|---|---|
| `SPEAKER_ID_ENABLED` | `0` | Master switch for inline identify |
| `SPEAKER_ID_BASE_URL` | `http://speaker-id:9100` | speaker-id service base URL |
| `SPEAKER_ID_API_KEY` | *(none)* | Auth key for the service |
| `SPEAKER_ID_TIMEOUT` | `0.8` | Hard timeout (s) on identify; on timeout → `unknown`, no retry. Because identify overlaps the transcribe round-trip, a modest bump (e.g. `1.5`) is mostly free and reduces `unknown` labels on a slow first call. |
| `SPEAKER_ID_LABEL_FORMAT` | `[{name}]` | Inline tag on a confident match; `{name}` / `{speaker_id}` available; empty = off |
| `SPEAKER_DIARIZE_ENABLED` | `0` | Async conference diarization, **off the hot path**: `/v1/diarize` runs after the turn and drives an idempotent, revision-versioned corrective event that replaces span labels by item_id. Dropped-safe. |
| `SPEAKER_DIARIZE_TIMEOUT` | `5.0` | Timeout (s) for the off-path diarize call |

> A transient slow identify is fail-safe (that turn is labeled `unknown`) and is
> **not** logged as an outage — s2s only warns after several consecutive failures,
> so an occasional timeout stays quiet while a real outage still surfaces.

### 4.8 Multi-session

s2s accepts up to `S2S_MAX_SESSIONS` concurrent **warm connections** in one
process. This caps *connections*, not simultaneous active turns — an idle warm
connection is a cheap set of threads blocked on empty queues. Size it to your
expected client count plus headroom.

| Env var | Default | Description |
|---|---|---|
| `S2S_MAX_SESSIONS` | `1` | Concurrent realtime sessions. `1` = exact single-session behavior. **Forced to 1** (with a warning) if any of STT/TTS/LLM is an in-process local model. |
| `S2S_IDLE_TIMEOUT_S` | `0` | Reap a session this many seconds after its last inbound traffic. `0` (default) never reaps — correct for warm smart-speaker connections held open indefinitely. |
| `S2S_THREAD_SUPERVISOR_S` | `2` | How often (s) the send loop checks its handler threads are alive; a crashed handler fails that session with a `server_error` instead of serving dead audio. `0` disables. |
| `STT_MAX_CONCURRENCY` | `0` | Cap concurrent in-flight STT requests across all sessions. `0` = no cap. |
| `TTS_MAX_CONCURRENCY` | `0` | Same, for TTS. |
| `LLM_MAX_CONCURRENCY` | `0` | Same, for the LLM. |

The `*_MAX_CONCURRENCY` caps let N sessions queue fairly behind a busy endpoint
instead of melting it; tune alongside the external service's own concurrency
(Whisper batch slots, F5 instances, Hermes workers). See `LATENCY.md` §7.

**Conversation memory note.** A fresh per-connection chat is created at connect.
If a client opens a new WebSocket per turn, each turn starts with an empty chat
(no memory of prior turns) — by design (isolation, no unbounded growth). For
multi-turn continuity, the client must hold **one** WebSocket open across turns:
one connection = one `session_id` = one accumulating chat.

### 4.9 WebSocket keepalive & warm connections

The server pings idle clients and closes them (`1011`) if they don't pong in time.
Defaults are generous so a warm-connection client can refresh infrequently and a
long tool turn won't trip a false `1011`.

| Env var | Default | Description |
|---|---|---|
| `S2S_WS_PING_INTERVAL` | `20` | Ping idle clients every N s. **`0` disables** server-side pings entirely (client/TCP own liveness). |
| `S2S_WS_PING_TIMEOUT` | `60` | Close `1011` if no pong arrives within N s |

> **Warm-connection clients:** keep one WebSocket open across turns to skip the
> connect cost. A client-side keepalive/refresh must stay inside the ping window
> (or set `S2S_WS_PING_INTERVAL=0` and own liveness yourself). With the idle reaper
> off (`S2S_IDLE_TIMEOUT_S=0`) and pings disabled, a client that vanishes without a
> TCP RST will linger until TCP timeout — leave pings on (the default) unless your
> client tends keepalive itself.

### 4.10 Turn progress / keepalive event

Between end-of-user-speech and the first audio chunk the pipeline is busy
(STT → LLM/tool loop → TTS) but the wire is otherwise silent. Two signals fix this
for **any** Realtime client:

1. **Early `response.created`** — emitted when transcription completes and the LLM
   is triggered (not on the first audio chunk). Spec-standard; arm a turn watchdog
   on it.
2. **`s2s.keepalive`** — `{"type":"s2s.keepalive", "event_id":…, "response_id":…}`
   emitted every `S2S_HEARTBEAT_S` seconds while a response is in progress and
   nothing else has been sent.

| Env var | Default | Description |
|---|---|---|
| `S2S_HEARTBEAT_S` | `5` | Keepalive event cadence while a response is in progress. `0` disables (for strict clients). The official OpenAI Python SDK tolerates the unknown event type. |

### 4.11 General / logging

| Env var | Default | Description |
|---|---|---|
| `LOG_LEVEL` | `info` | uvicorn/app log level |
| `DEBUG_MODE` | `off` | `on` re-enables verbose diagnostics (VAD heartbeat, audio-drop and TTS-timing logs). Leave off for clean production logs; turn on to tune AEC/VAD. |

---

## 5. Verifying a deployment

1. **Server is up** — startup logs show the bind line plus the pre-warm lines:
   ```
   OpenAI Realtime API server starting on ws://0.0.0.0:8765/v1/realtime
   Pre-warmed Silero VAD in … ms
   Pre-warmed shared Smart Turn detector in … ms        (if TURN_DETECTION=smart_turn)
   Pre-warmed remote LLM in … ms                        (responses-api)
   WebSocket keepalive: ping every 20s, 60s pong timeout
   ```
2. **Session roster** — `curl http://HOST:8765/v1/sessions` returns
   `{"count":0,"max_sessions":N,"sessions":[]}`. Confirm `max_sessions` matches your
   `S2S_MAX_SESSIONS` (if it shows `1` despite a higher setting, the var isn't
   reaching the container — see §3.1's Compose gotcha, or you have a local in-process
   model forcing 1).
3. **A first turn** — connect a client, speak, and watch the log: a `session.created`,
   the transcribe POST (`/v1/audio/transcriptions`), the LLM POST (`/v1/responses`),
   the TTS stream, and audio deltas back to the client.
4. **Speaker-id (if on)** — a `/v1/identify` POST appears alongside the transcribe,
   and a confident match prefixes the transcript with the configured tag.

---

## 6. Observability

| Endpoint | Returns |
|---|---|
| `GET /v1/sessions` | `{count, max_sessions, sessions:[…]}` — per-session state (speaking/thinking/listening/idle), age, idle time, turn count, per-session usage |
| `GET /v1/usage` | Cumulative usage plus `active_sessions` and a `per_session` breakdown |

Handler threads are tagged with a short session id (e.g. `VADHandler-2af96330`) so
log lines and stack dumps are attributable to the right session.

---

## 7. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `/v1/sessions` shows `max_sessions:1` despite a higher `S2S_MAX_SESSIONS` | The var isn't reaching the container (add it to the compose `environment:` block — §3.1), **or** a local in-process STT/TTS/LLM model is forcing 1 (multi-session needs all-remote). |
| Client connects then drops during a slow first connect | First connect warms models at startup now; if you still see multi-second builds, check `pipeline built + started in … ms` and which service is slow. |
| `1011` close on an idle/long turn | Raise `S2S_WS_PING_TIMEOUT`, or set `S2S_WS_PING_INTERVAL=0` and tend keepalive on the client. |
| Every turn labeled `unknown` + "speaker identify is failing … client has been closed" | Was a shared-client bug (fixed in 0.3.0). If seen on an older build, restart s2s; on 0.3.0 the shared speaker-id client is no longer closed per session. |
| Occasional `unknown` + a transient identify `ReadTimeout` that recovers | The service exceeded `SPEAKER_ID_TIMEOUT` on one call (fail-safe). Raise `SPEAKER_ID_TIMEOUT` (e.g. `1.5`) — it overlaps transcribe, so it's mostly free. |
| Phantom barge-ins / agent cuts itself off | Enable AEC and tune the VAD gates with `DEBUG_MODE=on` — see §4.6. |
| No memory across turns | The client is opening a new WebSocket per turn; hold one open across turns — see §4.8. |

---

## 8. Running the tests

```bash
# Full suite
python -m pytest

# Lint
ruff check .
```

See also: [`README.md`](../README.md) (overview), [`REMOTE_SETUP.md`](../REMOTE_SETUP.md)
(original remote-mode runbook), [`LATENCY.md`](LATENCY.md) (latency + multi-session
capacity testing), [`MULTI_SESSION_PLAN.md`](MULTI_SESSION_PLAN.md) (design of
record), and [`CHANGELOG.md`](../CHANGELOG.md).
