# speech2speech — TODO

Scoped work items captured 2026-06-11. s2s is a client-agnostic realtime backend
for any project that can use a realtime pipeline with the LLM as the local agent
(smart speaker, Matrix/Element Call voice client, …) — AVA is just one client.

---

## 1. ✅ DONE (2026-06-12) — Emit a progress / keepalive event during the silent "thinking" gap

**Shipped:** `response.created` (in_progress) is now emitted at turn start (when
the transcription completes and the LLM is triggered), and an `s2s.keepalive`
event fires every `S2S_HEARTBEAT_S` seconds (default 5, `0` disables) while a
response is in_progress with nothing on the wire. Side effects: barge-in now
works during the thinking gap (cancels the in-flight LLM instead of stacking a
second generation), and the LLM read timeout is configurable
(`LLM_REQUEST_TIMEOUT_S`, default raised 20 → 60 s — the old value could cut off
legitimate tool-loop turns). See CHANGELOG 2026-06-12.

**Client follow-up:** clients that raised their turn watchdogs to tolerate the
silent gap (e.g. a 75 s `turn_watchdog_s`) can now tighten them back down and
refresh on `s2s.keepalive` / early `response.created` instead.

Original problem statement follows for reference.

### (resolved) Original item

**Problem.** In a turn, everything from the moment the user stops speaking to the
first outbound audio chunk is **silent on the wire**: STT → the Hermes agent's
LLM + tool loop → TTS. Tools run *inside the Hermes agent*, not in s2s, so s2s
never forwards a `function_call` event — the entire tool loop is invisible to the
client. And s2s emits `response.created` **only on the first audio chunk**
(`begin_output_item_events` in
`src/speech_to_speech/api/openai_realtime/handlers/response.py`). So a long or
tool-calling agent turn is indistinguishable, from the client's side, from a dead
connection: pure silence.

**Impact.** The voice client runs a per-turn watchdog (aborts a turn that gets no
events, so it doesn't hang in "thinking" forever). With no events during the gap,
the watchdog false-fires on legitimately-slow turns. Measured on-device: even
**tool-free** Hermes turns take ~25–30 s to first audio; a tool round-trip blows
past that. (Client band-aid already shipped: raised its `turn_watchdog_s` default
30 → 75 s — but that's 75 s of "thinking" UI for a genuinely dead turn, so the real
fix belongs here.)

**Proposed fix.** As soon as s2s begins processing a turn (not on the first audio
chunk), emit an early `response.created` / `in_progress`, and/or emit a periodic
**keepalive/heartbeat** event during the silent gap — especially while a tool call
is in flight. That lets any client (a) refresh its watchdog on each heartbeat and
(b) show honest "working" feedback. Once heartbeats exist, downstream watchdogs can
be tightened back down.

**Touchpoints.**
- `src/speech_to_speech/api/openai_realtime/handlers/response.py` —
  `begin_output_item_events` currently fires the `response.created` /
  `output_item.added` / `content_part.added` sequence on first audio; consider
  firing `response.created` (in_progress) at turn start instead/also.
- `src/speech_to_speech/api/openai_realtime/websocket_router.py` — the `_send_loop`
  that drains the pipeline queues and translates to protocol events (natural place
  for a periodic heartbeat tick).
- Coordinate with the Hermes adapter (`hermes-realtime-platform-adapter`) if the
  "tool call started / still running" signal needs to originate there.

---

## 2. Support multiple concurrent, isolated sessions

**Requirement (Eric).** s2s must eventually handle **multiple unique concurrent
sessions, kept organized and isolated from each other** — a truly multi-use
backend: multiple smart speakers holding warm WebSocket connections, plus other
client types (Matrix/Element Call bridge, browser, …).

**➜ Detailed design of record: [MULTI_SESSION_PLAN.md](MULTI_SESSION_PLAN.md)**
(2026-06-12). Decision: in-process session pool (`SessionPipeline` per
connection) for the remote profile — the heavy models are already external HTTP
services, so a session costs ~6 idle threads — with replica scale-out as the
later story. Phases A–E with per-phase acceptance criteria are in the plan; the
notes below are the original capture, superseded where they conflict.

**Current state — hard single-session.** The WS endpoint `realtime_endpoint` in
`src/speech_to_speech/api/openai_realtime/websocket_router.py` rejects any second
connection:

```python
if app.state.websockets:
    ... await ws.close(code=1008, reason="Only one concurrent session is supported")
```

and it tracks a single `app.state.active_session`.

**What's already isolated vs. shared.**
- *Isolated (good):* each connection gets a `session_id` from `service.register()`,
  and the per-session state that matters — `runtime_config` and the LLM `chat`
  (conversation history) — is keyed by `session_id` (`service._state(session_id)`).
  Conversation/context isolation scaffolding largely exists.
- *Shared singleton (the blocker):* the heavy pipeline is process-global — one
  `_send_loop` task started in the FastAPI lifespan, shared `input` / `output` /
  `text_output` queues, one `echo_canceller`, one `cancel_scope`, shared
  `should_listen` / `response_playing` — and `_send_loop` **broadcasts** to
  `service.connection_ids` rather than routing to the owning connection.

**What multi-session isolation requires (rough).**
1. Per-session pipeline instances (or per-session queues + routing) instead of the
   shared singleton.
2. Per-session AEC / echo canceller — each session has its own far-end reference, so
   it cannot be shared.
3. ~~Route `_send_loop` output to the **owning** websocket, not a broadcast.~~
   ✅ Done 2026-06-12: `_send_loop` now routes to `app.state.active_session` only
   (behavior-identical for one session).
4. Per-session cancel / listen state; drop the single-session guard.
5. **Compute is the real constraint:** every session runs Whisper + Hermes LLM +
   F5 TTS, largely on GPU. N sessions ⇒ N× load, needing concurrency-safe / batched
   model serving, or horizontal scale.

**Design angle to weigh.** Rather than making one process internally multi-session
(touching AEC, queues, the send loop, cancellation), a **process-per-session** model
— one s2s instance per session behind a small router/load-balancer that assigns by
session — may be simpler and gives hard isolation for free, at the cost of running
multiple model copies (or a shared model-serving backend they all call).

**Decision (2026-06-12).** Process-per-session behind a thin session router is the
preferred architecture: the shared queues/AEC/cancel-scope make in-process
multi-session a high-risk rewrite, GPU compute caps concurrency per box anyway,
and hard isolation comes free. Build the router when the need materializes.

**Note.** A client's "warm connection" pattern (holding the single slot and
recycling it per wake) is fine for one device but assumes this single-session
server. Multi-session would let multiple devices/clients share one s2s.
