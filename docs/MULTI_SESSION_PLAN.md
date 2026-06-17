# Multi-Session Design & Implementation Plan

Status: **Phases A–D shipped** on the `multi-session` branch (2026-06-13); each
landed green on the full suite with `S2S_MAX_SESSIONS=1` preserving exact
single-session behavior. Phase E (replica scale-out) is deferred until a
measurement demands it. The §6.3/§6.4 soak + load runs are operator-run on the
real remote stack — see LATENCY.md §7 for the harness and the (to-be-filled)
recommended `S2S_MAX_SESSIONS`. Original design drawn up 2026-06-12.

## Goals

s2s is a client-agnostic realtime backend. The multi-session target is:

1. **Multiple smart speakers**, each holding a **warm WebSocket connection** to one
   s2s instance, any of which can become the active talker at any moment with zero
   connect latency.
2. **Truly multi-use**: heterogeneous clients sharing one server — a voice
   assistant, a Matrix/Element Call bridge, a browser test client — each with its
   own conversation, voice config, and barge-in state, fully isolated from the
   others.
3. **No regression** for the single-session deployment: same latency, same
   behavior, same tests passing.

Out of scope here: multi-*user* identity inside one session (that's the
speaker-id/diarization track), and cross-session shared memory.

---

## 1. Where we are today

One process runs **one pipeline** built at startup (`s2s_pipeline.py` →
`build_pipeline` → `ThreadManager`): six handler threads chained by global queues —

```
recv_audio ─→ VAD ─→ spoken_prompt ─→ STT ─→ stt_output ─→ TranscriptionNotifier
   ─→ text_prompt ─→ LLM ─→ lm_response ─→ LMOutputProcessor ─→ lm_processed
   ─→ TTS ─→ send_audio  (+ text_output for protocol events)
```

plus a `RealtimeServer` comms handler that owns uvicorn + `create_app`
(`websocket_router.py`), one `_send_loop` task, one `EchoCanceller`, one
`CancelScope`, one `should_listen`/`response_playing` pair.

**Already per-session (good):** `ConnState` in `service.py` keys everything
protocol-level by `session_id` — `runtime_config` (including the `Chat`
history), response lifecycle IDs, usage metrics. The send loop routes to the
owning session (done 2026-06-12). Auth is per-connection.

**Still process-global (the blockers):**

| # | Singleton | Where |
|---|---|---|
| 1 | The six pipeline threads + seven queues | `s2s_pipeline.build_pipeline` |
| 2 | `EchoCanceller` (one far-end buffer) | `create_app` closure |
| 3 | `CancelScope`, `should_listen`, `response_playing` | built in `main`, threaded everywhere |
| 4 | The `if app.state.websockets: reject` single-session guard | `realtime_endpoint` |
| 5 | One `_send_loop` draining the one output queue | `create_app` |

**The decisive observation:** in the production **remote profile**, STT, TTS, and
the LLM are *external HTTP services*. The s2s process itself runs only silero VAD
(~2 MB model), AEC, queue plumbing, and HTTP clients. A session's in-process
footprint is **6 mostly-idle threads + a tiny VAD model + buffers** — cheap. The
heavy GPU compute is already centralized behind the STT/TTS/LLM endpoints, which
are naturally shared by N sessions. This changes the calculus from the original
TODO note ("every session runs Whisper + Hermes + F5 on GPU"): the GPU load is
*not* per-process; it's per-*concurrent-turn*, and it lands on the external
services regardless of how s2s is structured.

---

## 2. Architecture decision

**In-process session pool** (per-session pipeline instances inside one s2s
process) for the remote profile, with **process-per-session / horizontal scale as
the later scale-out story**, not the first step.

Why this inverts the TODO's earlier process-per-session lean:

- **Warm connections are the dominant state.** Ten smart speakers = ten warm
  sessions, ~zero or one active at a time. Ten idle *processes* each holding a
  VAD model, an interpreter, and an httpx pool is pure waste; ten idle *thread
  sets* blocked on empty queues cost nearly nothing.
- **The isolation that matters is already cheap.** The dangerous shared state
  (AEC far-end, cancel scope, queues) is exactly what gets instantiated
  per-session in this design. There is no shared mutable model state in the
  remote profile to protect.
- **Connect latency.** A warm session's pipeline is built once at WS connect.
  Process-per-session would either pay process spawn + import + VAD load at
  connect (against the whole point of warm connections) or require a pre-forked
  warm pool — strictly more machinery for the same result.
- **Local-model profiles** (in-process Whisper/Kokoro/Qwen3) stay supported with
  `S2S_MAX_SESSIONS=1` (default for those profiles). Anyone needing multi-session
  with local models should move the model behind an HTTP serving endpoint (which
  this repo already has handlers for) — that *is* the process-isolation story,
  applied where it belongs: at the model, not the session.

**Scale-out later:** when one process saturates (CPU on resampling/AEC, or just
blast-radius concerns), run N replicas behind a dumb session-sticky router
(least-sessions assignment; any L7 LB with WS support works since sessions are
connection-scoped and share nothing). That is a deployment change, not a code
change — which is exactly why the in-process refactor should not bake in any
cross-session state.

---

## 3. Target design

### 3.1 `SessionPipeline` — the unit of isolation

New module `src/speech_to_speech/pipeline/session_pipeline.py`:

```python
class SessionPipeline:
    """Everything one connection owns. Built on WS connect, torn down on disconnect."""
    session_id: str
    # queues (all fresh instances)
    recv_audio, spoken_prompt, stt_output, text_prompt,
    lm_response, lm_processed, send_audio, text_output: Queue
    # control (all fresh instances)
    cancel_scope: CancelScope
    should_listen: ThreadingEvent
    response_playing: ThreadingEvent
    echo_canceller: EchoCanceller
    # the six handler threads
    handlers: list[BaseHandler]          # vad, stt, notifier, lm, lm_processor, tts
    threads: ThreadManager
    # per-session asyncio task
    send_task: asyncio.Task

    @classmethod
    def build(cls, session_id, factory: HandlerFactory) -> "SessionPipeline": ...
    def start(self) -> None: ...
    def shutdown(self, timeout=5.0) -> None:   # SESSION_END → PIPELINE_END → join
```

`HandlerFactory` is the parsed CLI/env args captured once at startup
(`ParsedArguments` + the `get_stt_handler`/`get_llm_handler`/`get_tts_handler`
dispatchers refactored to take queues as parameters — they already do, they just
need to be callable per-session instead of once).

**What is shared across sessions (deliberately, all read-only or thread-safe):**

- Parsed args / handler configuration.
- The silero VAD **weights**: load the torch module once, hand each session its
  own `VADIterator` (the stateful part). If sharing the module proves fiddly,
  per-session load is acceptable — it's ~2 MB and ~100 ms; measure first.
- One `httpx.Client` per service (STT/TTS/LLM) *or* per-session clients —
  per-session is simpler and gives per-session connection pools; start there.
- `RealtimeService` (already multi-conn by design) and the diarize thread pool.
- `GlobalUsageMetrics` — add a `threading.Lock` around `+=` rollups.

**What must never be shared:** queues, `CancelScope`, `EchoCanceller`,
`should_listen`, `response_playing`, `Chat`, audio remainder buffers, VAD
iterator state, pacing deadlines.

### 3.2 Router changes (`websocket_router.py`)

- `create_app(service, session_factory, stop_event, server_api_key, max_sessions)`
  — the queue/event parameters disappear from the signature; they live in the
  per-session pipeline.
- On connect (after auth):
  - `if len(app.state.sessions) >= max_sessions:` → existing
    `session_limit_reached` error (message updated to say how many).
  - `session_id = service.register()`; `pipeline = session_factory.build(session_id)`;
    `pipeline.start()`; store in `app.state.sessions[session_id]`.
  - Build cost budget: < 250 ms in the remote profile (VAD iterator + 6 threads +
    3 httpx clients). Log it. If it creeps, pre-warm one spare pipeline
    (`S2S_PREWARM=1`) and swap session_id at assign time — but don't build the
    pre-warm machinery until a measurement says so.
- The **receive loop** is already per-connection; it writes to
  `pipeline.recv_audio` and uses `pipeline.echo_canceller` / `pipeline.cancel_scope`
  instead of the closures.
- The **send loop** becomes `async def _session_send_loop(session_id, pipeline, ws)`
  — one task per session, started at connect, cancelled at disconnect. It is the
  current `_send_loop` body minus the `active_session` indirection: it drains
  *this* session's `send_audio`/`text_output` queues and sends to *this* ws.
  Keepalive, pacing, barge-in handling, and the stale-done guard all come along
  unchanged — they already operate on per-session state.
- On disconnect: cancel send task → `pipeline.shutdown()` (puts `SESSION_END`
  then `PIPELINE_END` on `recv_audio`, joins threads with timeout, closes httpx
  clients via `handler.cleanup()`) → `service.unregister(session_id)`.
  Shutdown must be fire-and-forget from the WS handler's perspective (run the
  joins in a worker thread) so a slow teardown can't block new connects.

### 3.3 Lifecycle & the warm-connection contract

A warm session is just a connected session with no audio flowing: VAD thread
blocked on `queue.get`, send task awaiting an empty queue at 10 ms cadence
(consider bumping the idle poll to 50 ms when the queue has been empty > 1 s —
ten sessions polling at 10 ms is harmless but pointless).

- **WS-level liveness:** uvicorn already answers protocol pings. Add an optional
  `S2S_IDLE_TIMEOUT_S` (default 0 = never) to reap sessions with no client
  traffic, for deployments that want it. Smart speakers holding warm connections
  set 0.
- **Crash isolation:** a handler thread dying must kill *its* session, not the
  process. `BaseHandler.run` already catches per-item exceptions; add a
  session-scoped supervisor check in the send task — if any pipeline thread is
  dead, close that WS with a `server_error` and tear down.
- **Server shutdown:** iterate sessions, close each WS (1001 going-away), shut
  down pipelines in parallel, then stop uvicorn.

### 3.4 Shared external services under concurrency

N sessions ⇒ up to N concurrent turns hitting the same Whisper/F5/Hermes
endpoints. s2s should degrade fairly rather than melt them:

- Per-service **concurrency caps**: `STT_MAX_CONCURRENCY`, `TTS_MAX_CONCURRENCY`,
  `LLM_MAX_CONCURRENCY` (semaphores acquired in the handlers around the HTTP
  call; default = unlimited to preserve today's behavior). With the early
  `response.created` + `s2s.keepalive` already shipped, a session queued behind
  a busy TTS still shows honest "working" feedback instead of dead air.
- Surface waiting in metrics (below) so capacity problems are visible before
  they're audible.
- The external services' own concurrency (Whisper batch slots, F5 instance
  count, Hermes worker pool) is the real capacity knob — document the pairing in
  REMOTE_SETUP.md and load-test §6.3 before raising `S2S_MAX_SESSIONS` defaults.

### 3.5 Observability

- `GET /v1/sessions`: list of `{session_id, connected_at, state (idle|listening|
  thinking|speaking), turns, last_activity_at, usage{...}}`.
- `/v1/usage` gains a `per_session` breakdown (live sessions) on top of the
  cumulative totals.
- Every log line in router/service/handlers gets a `[sess_xxxxxx]` prefix —
  thread names set to `f"{handler}-{session_id[:8]}"` so stack dumps are
  attributable.

### 3.6 Config surface (all new, all defaulted to today's behavior)

| Env / flag | Default | Meaning |
|---|---|---|
| `S2S_MAX_SESSIONS` | `1` | Concurrent session cap. `1` ⇒ exactly today's semantics. Raise deliberately after load testing. Forced to `1` with a warning if a local in-process STT/TTS model is selected. |
| `S2S_IDLE_TIMEOUT_S` | `0` | Reap sessions idle longer than this. `0` = never (warm connections). |
| `S2S_PREWARM` | `0` | Keep one pre-built spare pipeline (only if connect-time build proves slow). |
| `STT/TTS/LLM_MAX_CONCURRENCY` | `0` (∞) | Per-service in-flight request caps. |

---

## 4. Implementation phases

Each phase lands green on the full suite with `S2S_MAX_SESSIONS=1` and is
independently shippable.

### Phase A — extract `SessionPipeline` (no behavior change)
1. Create `session_pipeline.py`; move queue/event/scope construction out of
   `main`/`build_pipeline` into `SessionPipeline.build`.
2. Refactor `get_stt_handler`/`get_llm_handler`/`get_tts_handler`/VAD/notifier
   construction into a `HandlerFactory` holding parsed args (they already take
   queues as parameters — this is plumbing, not logic).
3. `main` (realtime mode) builds **one** `SessionPipeline` at startup and hands
   it to `create_app` exactly as the globals are handed today. Non-realtime
   modes (`local`, `websocket`, `socket`) keep the startup-built pipeline path.
4. Acceptance: byte-for-byte event streams on the existing suite; latency
   spot-check on hardware.

### Phase B — pipeline-per-connection (still max 1)
1. Move `SessionPipeline.build` to WS connect, shutdown to disconnect, behind
   the still-present single-session guard. The startup pipeline goes away in
   realtime mode.
2. Per-session send task replaces the lifespan-owned `_send_loop`;
   per-session `EchoCanceller` replaces the closure singleton.
3. Instrument connect-time build cost.
4. Acceptance: suite green; warm-connection reconnect cycle (connect → turn →
   disconnect → reconnect) leaks no threads (assert thread count returns to
   baseline in a test); barge-in/keepalive/stale-done tests unchanged.

### Phase C — lift the guard
1. Replace the second-connection rejection with the `S2S_MAX_SESSIONS` check;
   sessions dict replaces `active_session`.
2. Thread-safety pass on the shared bits: `GlobalUsageMetrics` lock; confirm
   `RealtimeService._conns` mutations happen only on the event loop thread;
   VAD weight sharing decision (measure per-session load first).
3. Per-service concurrency semaphores (default off).
4. Acceptance (new tests):
   - Two TestClient sessions stream interleaved audio → each receives only its
     own `speech_started`/transcription/response events (no cross-talk).
   - Barge-in in session A while session B is mid-response: B's audio
     uninterrupted, A cancels.
   - Session A disconnects mid-response: A's pipeline tears down; B unaffected.
   - `session_limit_reached` at N+1 with a correct message.
   - Keepalive fires independently per session.

### Phase D — operational hardening
1. `/v1/sessions`, per-session usage, session-prefixed logs/thread names.
2. Idle timeout, dead-thread supervisor, parallel shutdown.
3. Soak: 8 warm simulated speakers, randomized single active talker, 24 h —
   assert no thread/fd/memory growth (extend `scripts/` with a soak client).
4. Load test against real STT/F5/Hermes: 2, 4, 8 concurrent turns; record
   p50/p95 first-audio latency into LATENCY.md; pick the documented
   `S2S_MAX_SESSIONS` recommendation from the knee of that curve.

### Phase E (deferred until a measurement demands it) — scale-out
- N s2s replicas behind a WS-capable LB, least-sessions routing, no shared
  state. Compose example + REMOTE_SETUP.md section. No code changes expected —
  that property is the acceptance test for Phases A–D's "no cross-session
  state" discipline.

---

## 5. Risks & open questions

| Risk | Mitigation |
|---|---|
| Hidden global state in handlers (module-level caches, `nltk` data download race at first use) | Phase A audit: grep module-level mutables in `STT/ LLM/ TTS/ VAD/`; pre-download nltk punkt at startup (already done in Docker image — verify). |
| Thread growth: 6/session × N + send tasks | At the smart-speaker scale (≤ 16 sessions ≈ 100 threads, idle-blocked) this is fine; revisit only if someone wants 100s of sessions — that's Phase E territory. |
| Connect-time pipeline build latency hurts non-warm clients | Measured in Phase B; `S2S_PREWARM` exists as the documented answer; warm-connection clients are unaffected by design. |
| External service contention turns multi-session into shared-queue latency | Concurrency caps + keepalive honesty + the Phase D load test making real capacity visible before defaults are raised. |
| Local-model profiles silently broken by concurrency | Hard-force `S2S_MAX_SESSIONS=1` + startup warning when an in-process model handler is selected. |
| Chat compaction worker / diarize pool touching closed sessions | Already handled (`Chat.close()` on unregister; diarize is emit-and-drop keyed by item_id) — keep the tests. |

Open questions to settle during Phase A:
1. Share the silero VAD torch module across sessions or load per session?
   (Decide by measuring per-session load time + RSS; bias toward per-session
   for simplicity.)
2. Does `RealtimeServer` stay a pipeline "comms handler", or does realtime mode
   stop pretending to be a ThreadManager pipeline and own uvicorn directly in
   `main`? (Bias: keep the wrapper, it's harmless and non-realtime modes keep
   their symmetry.)

---

## 6. Why not the alternatives (recorded for posterity)

- **Process-per-session, router in front (original TODO lean):** right instinct
  for *hard* isolation, wrong cost model once the heavy models moved out of
  process. Revisits as Phase E in replica form, where it belongs.
- **One pipeline, session-tagged queue items:** smallest diff, worst outcome —
  every handler grows session-demux logic, one slow turn head-of-line blocks
  every session, barge-in/cancel scoping becomes a tagged-generation matrix.
  Rejected.
- **asyncio-rewrite of the pipeline:** would collapse threads into tasks and is
  arguably the "right" long-term shape, but it rewrites every handler's blocking
  HTTP/model call into async and re-litigates every timing behavior (pacing,
  VAD chunking) for no user-visible gain at this scale. Rejected for this
  effort.
