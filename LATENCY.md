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

TTFB = time to first audio byte (the number that drives perceived latency).
iters=5 (median shown)

=== openai-remote / F5 ===
text                             TTFB(s)  total(s)  audio(s)    RTF
greeting (6 words)                 2.346     2.346      2.10   1.18
one sentence (18 words)            3.058     3.058      6.52   0.47
three sentences (~45 words)        3.134     8.712     16.52   0.53

=== elevenlabs ===
text                             TTFB(s)  total(s)  audio(s)    RTF
greeting (6 words)                 0.261     0.342      1.81   0.19
one sentence (18 words)            0.266     0.436      5.43   0.08
three sentences (~45 words)        0.276     0.752     15.05   0.05

### 4b. Live first-audio by batch size (one fixed test utterance)

OpenAI-Remote F5 Endpoint

Batch 3
turn spch→1st audio spch→TTS-open  LLM(s)  TTFB(s)  synth(s)
   1         19.653        11.639   5.712    4.002     8.014
----------------------------------------------------------------
median speech→first-audio : 19.653 s   (n=1)
median speech→TTS-open    : 11.639 s   (= STT + LLM first batch + queueing)

Batch 2
turn spch→1st audio spch→TTS-open  LLM(s)  TTFB(s)  synth(s)
   1         15.603         9.310   5.078    2.200     6.293
   2          9.617         7.551   4.605    2.138     2.066
   3         10.572         8.162   4.668    2.063     2.410
----------------------------------------------------------------
median speech→first-audio : 10.572 s   (n=3)
median speech→TTS-open    : 8.162 s   (= STT + LLM first batch + queueing)

Batch 1
turn spch→1st audio spch→TTS-open  LLM(s)  TTFB(s)  synth(s)
   1          9.854         9.854   5.621    2.432     0.000
   2         10.469        10.469   4.931    2.508     0.000
----------------------------------------------------------------
median speech→first-audio : 10.162 s   (n=2)
median speech→TTS-open    : 10.162 s   (= STT + LLM first batch + queueing)


ElevenLabs

Batch 3
turn spch→1st audio spch→TTS-open  LLM(s)  TTFB(s)  synth(s)
   1            —             —     5.656    0.282       —  
----------------------------------------------------------------
Batch 2
turn spch→1st audio spch→TTS-open  LLM(s)  TTFB(s)  synth(s)
   1            —             —     5.670    0.409       —  
   2            —             —     8.563    0.247       —  
   3            —             —     5.695    0.274       —  
   4            —             —     6.142    0.262       —  
   5            —             —     5.060    0.274       —  
   6            —             —     5.813    0.291       —  
   7            —             —     3.809    0.264       —  
   8            —             —     3.182    0.261       —  
----------------------------------------------------------------

Minimax Websocket 

text                            connect  ttfb_cold  ttfb_warm    total   audio
greeting (6 words)                0.227      0.520      0.257    0.779    2.11
one sentence (18 words)           0.202      0.513      0.270    0.886    4.88
three sentences (~45 words)       0.198      0.512      0.277    1.308   13.45
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

---

## 7. Multi-session capacity (Phase D load test)

s2s now serves up to `S2S_MAX_SESSIONS` concurrent realtime sessions in one
process (remote profile only; local in-process models force `1`). N sessions ⇒
up to N concurrent turns hitting the *same* shared STT/F5/Hermes endpoints, so
first-audio latency degrades as concurrency rises. The job of this section is to
**measure that curve on real hardware and document the recommended cap from its
knee** — the point where p95 first-audio crosses your latency budget.

> Status: **measure first.** The harness below is wired; the tables are
> placeholders until filled from a real run against the production remote stack.

### 7.1 How to run — step by step

This is a complete runbook. Two roles are involved and they can be the **same
machine or two different machines**:

- **Server host** — runs the s2s container (needs Docker; typically Linux).
- **Load box** — runs the test scripts (needs Python 3.11+; can be the server
  host itself, another Linux machine, a Mac, or **Windows via WSL**).

The harness measures **speech_stopped → first audio delta** (the §1 first-audio
number) per turn.

#### Prerequisites (have these ready before you start)

1. Your three remote inference endpoints are already running and reachable from
   the server host: a Whisper-compatible STT, an F5/OpenAI-compatible TTS (or
   ElevenLabs/MiniMax), and an OpenAI-compatible LLM. You know each one's URL.
2. This repo is checked out on the server host, and Docker + the Docker Compose
   plugin are installed there.
3. A WAV file of a person speaking one or two sentences (any sample rate, mono or
   stereo, 16-bit PCM — the harness downmixes/resamples to 16 kHz). Call it
   `sample.wav`. **Silence will not work** — the server VAD needs real speech to
   trigger a turn.

#### Step 1 — Configure `.env` on the server host

```bash
cd /path/to/speech2speech          # the repo root on the server host
cp -n .env.sample .env             # create .env from the sample (keeps an existing .env)
nano .env                          # or any editor
```

In `.env`, set these exactly (replace the example IPs/ports with **your** real
endpoint addresses; keep `sk-unused` if your endpoints don't check the key):

```ini
# Disable auth for the test: put a '#' in front so no Bearer token is required.
#SERVER_API_KEY=

# STT — your Whisper-compatible endpoint
STT_OPENAI_BASE_URL=http://192.168.1.10:8000
STT_OPENAI_API_KEY=sk-unused
STT_OPENAI_MODEL=Systran/faster-whisper-large-v3

# TTS — your F5/OpenAI-compatible endpoint (or set TTS_SOURCE=elevenlabs/minimax
# and fill that block instead)
TTS_SOURCE=openai-remote
TTS_OPENAI_BASE_URL=http://192.168.1.10:8880
TTS_OPENAI_API_KEY=sk-unused
TTS_OPENAI_VOICE=default

# LLM — your OpenAI-compatible endpoint
LLM_BASE_URL=http://192.168.1.10:7860/v1
LLM_API_KEY=sk-unused
LLM_MODEL=hermes-agent

# Multi-session: accept up to 8 concurrent sessions for the test
S2S_MAX_SESSIONS=8
```

Two things people miss:

- **`#SERVER_API_KEY=` must be commented out (or empty).** If a token is set, the
  server rejects every connection that doesn't send it and the test fails at
  connect. (If you must keep auth on, instead pass `--api-key YOUR_TOKEN` to both
  scripts in Steps 5–6.)
- **`S2S_MAX_SESSIONS` must be ≥ the largest `--concurrencies` value** you drive
  (8 here). It only takes effect with all-remote backends; a local in-process
  model forces it back to 1.

#### Step 2 — Start the server (on the server host)

```bash
cd /path/to/speech2speech
docker compose -f docker-compose.remote.yml up
```

Leave this terminal running (it streams the server log). It listens on
**port 8765**.

#### Step 3 — Verify the server and the cap (on the server host)

In a second terminal on the server host:

```bash
curl -s localhost:8765/v1/sessions
```

Expected output (the important part is `"max_sessions":8`):

```json
{"count":0,"max_sessions":8,"sessions":[]}
```

If you see `"max_sessions":1`, your `.env` change didn't take effect — stop the
server (Ctrl-C in the Step 2 terminal) and bring it up again with
`docker compose -f docker-compose.remote.yml up` (this re-reads `.env`).

#### Step 4 — Prepare the load box

On whichever machine will run the scripts (the server host, another machine, or
Windows + WSL), create and activate a virtual environment, then install the
harness's only third-party dependency into it:

```bash
cd /path/to/speech2speech          # the repo checkout (or just copy the scripts/ dir)
python3 -m venv venv               # create the virtual environment (one time)
source venv/bin/activate           # activate it  (Windows: venv\Scripts\activate)
python3 -m pip install --upgrade pip
python3 -m pip install websockets  # the harness's only third-party dependency
```

If `python3 -m venv` errors that the `venv` module is missing, install it first
(Debian/Ubuntu: `sudo apt install python3-venv`), then re-run the command above.

Keep this terminal (with `venv` activated) for Steps 5–6 — the `python3` there now
has `websockets`. Put your `sample.wav` somewhere on this machine (e.g. the repo
root) and confirm it's there:

```bash
ls -l sample.wav
```

#### Step 5 — Run the load test (from the load box)

If the load box **is** the server host, use `127.0.0.1`:

```bash
cd /path/to/speech2speech
python3 scripts/load_test_sessions.py \
  --host 127.0.0.1 --port 8765 \
  --wav sample.wav \
  --concurrencies 2,4,8 --rounds 5 --markdown
```

If the load box is a **different machine or WSL**, replace `127.0.0.1` with the
server host's IP address (see "Connecting from another machine / WSL" below):

```bash
python3 scripts/load_test_sessions.py \
  --host 192.168.1.50 --port 8765 \
  --wav sample.wav \
  --concurrencies 2,4,8 --rounds 5 --markdown
```

It prints a per-concurrency latency table and, because of `--markdown`, a block
you paste straight into §7.2. It exits non-zero if the success rate drops below
95%, and warns first if the server's `max_sessions` is below your concurrency.

#### Step 6 — Run the soak test (optional leak check)

A 5-minute smoke (use `--duration-s 86400` for the full 24 h):

```bash
python3 scripts/soak_sessions.py \
  --host 127.0.0.1 --port 8765 \
  --wav sample.wav \
  --sessions 8 --duration-s 300 --turn-interval-s 8
```

To also track server RSS/threads/file-descriptors for the leak check, add
`--server-pid` **only when you run this on the server host** (it reads `/proc` of
that PID locally):

```bash
  --server-pid "$(pgrep -f s2s_pipeline | head -n1)"
```

Off-box (different machine / WSL) you can't read the server's `/proc`, so omit
`--server-pid`; the soak then tracks only the live session count.

#### Connecting from another machine / WSL

- **What IP to use:** on the server host run `hostname -I` and take the first
  address (e.g. `192.168.1.50`). Use that as `--host` from the load box.
- **Reachability:** from the load box, `curl -s http://SERVER_IP:8765/v1/sessions`
  must return the JSON from Step 3. If it hangs or refuses, port 8765 is blocked —
  open it in the server host's firewall (e.g. `sudo ufw allow 8765/tcp`).
- **WSL specifics:** if the server runs in Docker Desktop on the *same* Windows
  machine as your WSL distro, `--host 127.0.0.1` usually works (WSL2 forwards
  localhost to Windows). If the server is on a *different* machine, use that
  machine's LAN IP as above.

The networked load/soak runs are operator-run on the real remote stack; fill the
§7.2 tables and the §7.3 recommendation from their output.

### 7.2 Results — fill in from runs

First-audio latency (speech_stopped → first audio delta), seconds:

| Concurrent turns | Turns | OK% | p50 (s) | p95 (s) | p99 (s) | max (s) |
|---|---|---|---|---|---|---|
| 1 (baseline) | 5 | 100.0% | 7.245 | 8.819 | 9.009 | 9.056 |
| 2 | 10 | 100.0% | 7.562 | 12.834 | 14.697 | 15.163 |
| 4 | 20 | 70.0% | 8.491 | 12.604 | 14.022 | 14.377 |
| 8 | 40 | 50.0% | 11.900 | 15.352 | 18.061 | 18.738 |

_Run: 2026-06-14, remote stack (whisper.anteon.group STT, MiniMax WS TTS, Hermes
responses-api LLM @ 192.168.1.36), `--rounds 5`, direct-to-container._

**Reading it:** the binding constraint is the **success rate**, not latency.
Baseline first-audio (~7.2 s p50 / 8.8 s p95) is LLM/TTS-bound as in §1. It holds
**100% through 2 concurrent**, then collapses: **70% at 4, 50% at 8** — i.e. at 8
concurrent turns half never produced audio.

Note the latency columns at 4 and 8 are **survivorship-biased**: a failed turn
contributes no first-audio sample, so the p50/p95 there describe only the turns
that *survived* and understate the real degradation (the failures are effectively
"infinite latency"). So don't read the flat-looking p50 at 4 as healthy — 30% of
those turns were dropped.

The knee for a 100%-success bar is **2 concurrent**. Whether the cap can go higher
depends on *what* is failing at 4–8 (the shared endpoints saturating vs. an s2s
issue) — see §7.3.

### 7.3 Picking the cap

1. Run §7.1, fill §7.2.
2. The recommended `S2S_MAX_SESSIONS` is the **highest concurrency whose p95
   first-audio stays within budget** *and* whose success rate is 100% — beyond
   it, turns queue behind the shared endpoints and latency knees up.
3. Pair the cap with the external services' own concurrency (Whisper batch slots,
   F5 instance count, Hermes worker pool). If the endpoints saturate before s2s
   does, set the per-service caps (`STT_MAX_CONCURRENCY` / `TTS_MAX_CONCURRENCY` /
   `LLM_MAX_CONCURRENCY`, default unlimited) so s2s queues fairly — with the
   early `response.created` + `s2s.keepalive` already shipped, a queued turn still
   shows honest "working" feedback instead of dead air. Document the chosen
   pairing in REMOTE_SETUP.md.

> **Recommended `S2S_MAX_SESSIONS` for the production remote stack: 2**
> _(provisional, 2026-06-14)._ Highest concurrency with 100% success in the §7.2
> run; 4 dropped to 70%, 8 to 50%. This may be raisable once the failure mode at
> 4–8 is identified (see below) and the saturating service is given a fair queue
> via its `*_MAX_CONCURRENCY` cap.

**Next: root-cause the 4–8 failures.** The whole point of the per-service caps is
to convert "endpoint overwhelmed → turn fails" into "turn queues, shows keepalive,
completes slower." Before settling on 2:
1. Re-run with the error breakdown visible — `load_test_sessions.py` prints an
   `Errors observed:` block to stderr listing the distinct failure types
   (`client_timeout`, an `error` event type, a connect exception, …). That names
   the failing stage.
2. If it's a **shared endpoint** saturating (most likely the MiniMax WS account's
   concurrent-connection limit, or the single Hermes LLM serving one request at a
   time), set that service's cap to its real capacity — e.g. `TTS_MAX_CONCURRENCY`
   or `LLM_MAX_CONCURRENCY` = N — so s2s serializes the (N+1)th turn behind a slot
   instead of letting it fail, then re-run 4/8 and confirm success returns to
   ~100% at higher p95. The usable `S2S_MAX_SESSIONS` is then "as many warm
   sessions as you want, capacity-gated by the cap," not 2.
3. If failures are **s2s-side** (not an endpoint), that's a bug to fix, not a cap
   to set — capture the failing session's server log.
