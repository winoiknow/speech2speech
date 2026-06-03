# Latency — measurement & tuning

Working notes for reducing end-to-end response latency in the remote s2s
pipeline. Goal: lower **time-to-first-audio** (end of the caller's speech → first
audio byte they hear) without regressing turn-taking or barge-in.

> Status: **measure first.** Fill in the results tables below from real runs
> before making architectural changes. Cheap config levers are wired and ready
> to A/B; bigger changes are catalogued but deferred until numbers exist.

---

## 1. Where the time goes (from call logs)

End-of-speech → first-audio, observed in the barge-in-test call (06-01, 10:34–10:37):

| Stage | Observed | Source in logs | s2s-controllable? |
|---|---|---|---|
| VAD end-of-turn silence wait | 0.7–0.9 s | `silence_duration_ms` (AVA `turn_detection`) | yes (config) |
| Remote STT (whisper) | ~0.3–0.5 s | whisper POST → transcription timestamps | partly |
| **LLM → first sentence batch** | **several s** | `ResponsesApiModelHandler: N s` (7–16 s full) | **batch size** |
| **TTS time-to-first-byte** | **2.2–4.0 s** (F5) | `RemoteOpenAITTS: time-to-first-byte …` | **yes** |
| pacing tail-hold (end only) | 0.4 s | `S2S_RESPONSE_DONE_TAIL_MS` | yes |

First-audio is dominated by **LLM-to-first-batch** and **TTS TTFB**. The rest is
sub-second. Two of the big levers are in s2s's hands.

---

## 2. How to measure

### 2a. TTS endpoints (offline, isolates the TTS box)
```bash
cd speech2speech && set -a && . .env && set +a
python3 scripts/bench_tts.py --engine both --iters 5
```
`bench_tts.py` reports **TTFB**, total, audio seconds, and RTF for F5 and
ElevenLabs across a greeting / one-sentence / three-sentence payload. TTFB is the
number that drives perceived latency; the three-sentence row is the current
`stream_batch_sentences=3` unit.

### 2b. Live call (end-to-end, from s2s logs)
With `DEBUG_MODE=on`, one turn shows the whole chain:
- `Speech ended (… ms), stop listening` → turn-end instant
- `ResponsesApiModelHandler: N s` → LLM batch time
- `RemoteOpenAITTS: time-to-first-byte …s` → TTS TTFB
- first `response.output_audio.delta` sent → first audio to client

Time from "Speech ended" to the first audio delta = the number we're minimizing.

---

## 3. Wired knobs (A/B without code edits)

| Knob | Env | Default | Effect |
|---|---|---|---|
| Sentence batch | `STREAM_BATCH_SENTENCES` | `3` | sentences buffered before the first TTS batch; **lower = lower TTFB**, more/shorter TTS calls |
| TTS engine | `TTS_SOURCE` | `openai-remote` | `elevenlabs` may have far lower TTFB — measure |
| TTS speed (F5) | `TTS_OPENAI_SPEED` | `1.0` | — |
| Response-done tail | `S2S_RESPONSE_DONE_TAIL_MS` | `400` | end-of-turn hold, not first-audio |
| Turn-end silence | AVA `turn_detection.silence_duration_ms` | 700–900 | per-turn fixed wait before STT |

---

## 4. Results — fill in from runs

### 4a. TTS TTFB (from `bench_tts.py`, median of 5)

| Engine / model | greeting | one sentence | three sentences |
|---|---|---|---|
| F5 (`openai-remote`) | ___ | ___ | ___ |
| ElevenLabs `eleven_flash_v2_5` | ___ | ___ | ___ |
| ElevenLabs `eleven_turbo_v2_5` | ___ | ___ | ___ |

### 4b. Live first-audio by batch size (one fixed test utterance)

| `STREAM_BATCH_SENTENCES` | TTS engine | Speech-ended → first audio | notes (choppiness?) |
|---|---|---|---|
| 3 | F5 | ___ | baseline |
| 2 | F5 | ___ | |
| 1 | F5 | ___ | |
| 1 | ElevenLabs | ___ | |

---

## 5. Lever catalog (ranked; revisit after §4 numbers)

| # | Lever | Effort | Status |
|---|---|---|---|
| 1 | `stream_batch_sentences` 3 → 1/2 | config | **wired** |
| 2 | TTS engine = ElevenLabs if TTFB wins | config | wired (`TTS_SOURCE`) |
| 3 | TTS **stream-through** (yield chunks as they arrive; today both handlers buffer the whole clip — `remote_openai_tts_handler.py:129-192`) | small code | deferred |
| 4 | VAD `silence_duration_ms` 900 → 500–700 | config (AVA) | deferred |
| 5 | Filler/backchannel ("let me check…") to mask LLM latency | small | deferred |
| 6 | LLM backend speed (the 7–16 s elephant) — smaller/faster model or better serving | infra | out of s2s |

## 6. Alternative approaches (architecture, deferred)

From the upstream HuggingFace pipeline this fork is based on:
- **Local fast TTS (`--tts kokoro`)** — co-located, near-real-time, can beat
  remote F5's TTFB by ~10×, at a voice-quality/cloning trade.
- **Streaming STT with partials (`parakeet-tdt`)** — incremental
  `PartialTranscription`; could begin LLM prefill on a near-final transcript
  (speculative start). (Note: *remote* progressive STT was disabled for the
  runaway-turn bug; a local streaming STT avoids that.)
- **Semantic endpointing** — a small turn-complete classifier on partials to cut
  the fixed VAD silence wait without cutting callers off.
- **Connection warmth** — confirm httpx keep-alive reuse to F5/whisper/LLM so
  each call doesn't pay TLS/connect setup.

Re-engage these once §4 has hard numbers showing which stage actually dominates
on this deployment.
