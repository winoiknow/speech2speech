# Speaker-ID service

Adjacent microservice for **speaker recognition / diarization** in the
`speech2speech` (s2s) pipeline. It owns the speaker-embedding model, the speaker
store, and the enrollment UI — so the realtime `s2s-app` container stays lean and
torch-free (it just calls this over HTTP, like it already does for STT/TTS/LLM).

The service source is **vendored in this repo** under [`../speaker-id/`](../speaker-id)
(`app/` + `Dockerfile` + `requirements*.txt`), so it builds and runs from the same
`docker-compose.yml` — no separate checkout needed. The canonical upstream is
`github.com/winoiknow/speaker-id`; re-sync the vendored copy from there when the
service changes.

## Run it (with s2s, same compose)

The service is **opt-in** via a compose profile so the default `docker compose up`
stays fast (the torch + ECAPA build is heavy and only runs when you ask for it):

```bash
# from the repo root — builds and starts BOTH s2s-app and speaker-id
docker compose --profile speaker-id up --build

# torch-free smoke run (no embedding model):
EMBEDDING_MODEL=stub docker compose --profile speaker-id up --build
```

Then turn on the s2s → speaker-id wiring (see the `SPEAKER_ID_*` block in
`docker-compose.yml` / `.env.sample`): set `SPEAKER_ID_ENABLED=1`. Inside the
compose network s2s reaches the service at `http://speaker-id:9100` (the default
`SPEAKER_ID_BASE_URL`); if you run speaker-id elsewhere, point that URL at it and
skip the profile. When on, s2s fires `POST /v1/identify` **concurrently** with STT
and tags the dialogue `[name]` on a confident match (adds no latency to the turn).

Container names are fixed in the compose: **`s2s-app`** and **`speaker-id`**
(`docker compose logs speaker-id`, `docker exec -it speaker-id …`).

## Endpoints

Recognition core is **SpeechBrain ECAPA-TDNN** (192-d), a `SqliteNumpyStore`
(cosine, `known/unknown/ambiguous`), a browser **enrollment UI** at `/enroll`, and
**multi-speaker diarization** at `/v1/diarize` (**pyannote community-1**, opt-in).
All behind pluggable interfaces (`embedding.Embedder`, `store.SpeakerStore`,
`diarization.Diarizer`) so the model or backend swaps without touching the API.

| Endpoint | Behavior |
|---|---|
| `GET /healthz` | liveness + model/store/speaker count |
| `POST /v1/identify` | identify one turn segment → known/unknown/ambiguous |
| `POST /v1/speakers` | create a speaker (**consent required**) |
| `POST /v1/speakers/{id}/samples` | add sample (**quality-gated**, embeds trimmed speech) |
| `GET /v1/speakers`, `DELETE /v1/speakers/{id}` | list / remove (hard delete) |
| `GET /enroll` | browser enrollment UI (admin-gated) |
| `POST /v1/diarize` | multi-speaker spans (conference) |
| `GET /auth/login`, `/auth/oidc/*`, `POST /auth/local` | admin sign-in (OIDC SSO + local admin) |
| `POST/GET/DELETE /v1/invites` | admin: issue / list / revoke enrollment invites |
| `GET /enroll/invite?token=…` | scoped self-enrollment page (token-gated, no login) |
| `POST /v1/invite/{token}/consent\|samples\|test` | invitee actions on the bound speaker only |

**Enroll in the browser:** open `http://<host>:9100/enroll` → tick consent, create
a speaker → read each prompted sentence and record 3–5 clips (or upload) → hit
**Test my voice** to see your live `decision` + `score`. Bad clips (too short /
clipped / too quiet) are rejected server-side with a reason. Quality gate knob:
`MIN_SAMPLE_SECONDS` (default 2.0). Recording captures raw PCM via Web Audio and
uploads 16-bit WAV; the server resamples to 16 kHz.

## Smoke test (enroll → identify)

```bash
curl -s localhost:9100/healthz
# {"status":"ok","version":"...","embedding_model":"speechbrain/spkrec-ecapa-voxceleb",...,"speakers":0}

# 1) create a speaker
SID=$(curl -s -X POST localhost:9100/v1/speakers -H 'content-type: application/json' \
        -d '{"name":"Eric","language":"en"}' | python3 -c 'import sys,json;print(json.load(sys.stdin)["speaker_id"])')

# 2) enroll 3-5 clean ~5-10s samples of that voice (16k mono wav is ideal)
for f in eric_*.wav; do curl -s -F "file=@$f;type=audio/wav" localhost:9100/v1/speakers/$SID/samples; done

# 3) identify a new clip of the same voice  -> decision: known, name: Eric
curl -s -F "file=@eric_test.wav;type=audio/wav" localhost:9100/v1/identify
# a different/unenrolled voice            -> decision: unknown
```

## Admin access control

The `/enroll` console and the speaker **admin APIs** (`POST/GET/DELETE
/v1/speakers*`) can be gated two ways; the **machine data plane** (`/v1/identify`,
`/v1/diarize`) uses a separate shared key so s2s keeps working without an admin
login. `/healthz` is always open.

> **Default is open** (unauthenticated) so existing trusted-network deployments
> keep working — the service logs a loud warning until you configure a gate.

- **Local admin (fallback / first-run):** set `LOCAL_ADMIN_TOKEN` (a long random
  value) + `SESSION_SECRET`. Sign in at `/auth/login`, or pass it as a bearer
  (`Authorization: Bearer <token>`) for scripted admin.
- **OIDC SSO (in-app, authlib):** set the env below, then sign in via **Sign in
  with SSO** on `/auth/login`. Set both OIDC and the local token for SSO with a
  break-glass fallback.
- **Data plane key:** set `SERVICE_API_KEY` and match it to the s2s
  `SPEAKER_ID_API_KEY`; callers must then send it as a bearer. A signed-in admin
  is also allowed (so the enroll page's *test-my-voice* works).

### OIDC setup (in-app, authlib)

| Env | Required | Notes |
|---|---|---|
| `OIDC_ENABLED` | yes | `1` to turn on SSO |
| `OIDC_DISCOVERY_URL` | yes | the IdP's `…/.well-known/openid-configuration` |
| `OIDC_CLIENT_ID` / `OIDC_CLIENT_SECRET` | yes | the app credentials from your IdP |
| `OIDC_ALLOWED_EMAILS` | one of these | comma list of admin emails |
| `OIDC_ALLOWED_DOMAIN` | one of these | any email `@domain` (e.g. `anteon.group`) |
| `SESSION_SECRET` | recommended | stable random value so logins survive restarts |

> **Without an allowlist, *any* user your IdP authenticates becomes an admin** —
> always set `OIDC_ALLOWED_EMAILS` and/or `OIDC_ALLOWED_DOMAIN`.

**Redirect URI — register this in the IdP** (the OAuth callback, **not** `/enroll`):

```
https://<host>/auth/oidc/callback
```

It must match **byte-for-byte** (scheme + host + path, no trailing slash). The app
builds it from the incoming request honoring the proxy's `X-Forwarded-Proto`
(uvicorn runs with `--proxy-headers`), so **behind TLS it is `https://…`**. To see
the exact value, click *Sign in with SSO* and check the log:

```
docker compose logs speaker-id | grep redirect_uri
# OIDC redirect_uri (register this in your IdP): https://<host>/auth/oidc/callback
```

> **TLS** is terminated by your reverse proxy (no sidecar). OIDC redirect URIs are
> https there, and https is also what satisfies the browser **secure-context**
> requirement — so the same proxy that adds SSO lets the enroll page capture the
> mic from any origin (no SSH tunnel needed).

## Email-invite self-enrollment

So people enroll *themselves* without admin tooling or seeing anyone else. An admin
issues an invite; the invitee gets a personal link and records on a **scoped** page
bound to one speaker — consent + record/upload + self-test only, **no list of other
speakers, no admin controls**. The invitee endpoints are **token-gated, not
OIDC** — an outside user with no IdP account can enroll.

```bash
# admin (signed in, or local-admin bearer) issues an invite
curl -X POST https://<host>/v1/invites -H "Authorization: Bearer $LOCAL_ADMIN_TOKEN" \
     -H 'content-type: application/json' -d '{"name":"Dana","email":"dana@example.com"}'
# → { "invite_url": "https://<host>/enroll/invite?token=…", "email_sent": true, ... }
```

- The link is **emailed** via on-site SMTP (or **logged** when `SMTP_HOST` is unset
  — the stub — so you can copy `invite_url` from the response/logs to test).
- The token is a **time-window credential**: valid for `INVITE_TTL_HOURS` (default
  **72 h**) and **reusable** within that window — it is *not* single-use. It ends
  only by expiry or `DELETE /v1/invites/{id}`. Only a SHA-256 hash is stored.
- Consent is captured **on the page**; test-my-voice scores **only** against the
  invitee's own speaker — it never reveals or matches any other enrolled voice.
- Set `PUBLIC_BASE_URL` (e.g. `https://speaker-id.example.com`) so the emailed link
  has the right host behind a proxy. SMTP: `SMTP_HOST/PORT/USER/PASS/FROM/STARTTLS`.

Invalid/expired/revoked tokens return `404`/`410` and the page shows a friendly
"ask for a new link" message.

## Conference diarization (`/v1/diarize`)

Splits a (possibly multi-speaker) clip into per-speaker spans and identifies each.
It groups spans by the **diarizer's own speaker labels** (pyannote clusters the
audio internally), aggregates each speaker's segments, and runs recognition **once
per speaker** — so each voice gets one consistent label and pooling its audio
gives a steadier score than judging short segments alone. Enrolled voices get their
**name**; unenrolled voices get one **ephemeral per-call tag** (`S1`, `S2`, …),
numbered by first appearance — stable within the call only, never persisted.

The real diarizer (pyannote) is **opt-in** so the default image stays torch-light.
To enable it you need all three:

```bash
# in .env
INSTALL_PYANNOTE=1                                          # bakes pyannote.audio (BUILD step)
DIARIZATION_MODEL=pyannote/speaker-diarization-community-1  # the gated model
HF_TOKEN=hf_xxx                                             # read token, gated terms accepted on HF

docker compose --profile speaker-id up -d --build           # --build required: INSTALL_PYANNOTE is build-time
```

`HF_TOKEN` is a **runtime** var — the gated weights download on first use into the
`/data/hf` cache (`HF_HOME`), so a "not set" notice during build is expected. The
first diarize call is slow (weights download); it's cached after.

```bash
curl -s -X POST localhost:9100/v1/diarize \
  -F "file=@two_speaker.wav;type=audio/wav" | jq
# segments:[{start,end,decision,name|null,label,score}], one label per voice;
# check "diarization_model" in the response is the pyannote id (not "stub").
```

With `INSTALL_PYANNOTE=0` (default) the diarizer is a torch-free **stub** (one span
= the whole clip), so `/v1/diarize` behaves like `/v1/identify` in segment shape —
handy for wiring/CI without the heavy model. On the s2s side, async conference
diarization is gated separately by `SPEAKER_DIARIZE_ENABLED` (off the hot path).

## Threshold calibration (before production)

`SIMILARITY_THRESHOLD=0.65` is a **placeholder**. The right cutoff is model- and
channel-specific — calibrate on enrolled-vs-impostor clips recorded on the target
channel (telephony 8 kHz ≠ wideband) and pick from the FAR/FRR trade-off (admin
recognition favors low false-accept). Live conversational turns are short (1–3 s),
so genuine `/v1/identify` scores run lower than clean enrollment clips; measure on
your channel and set the cutoff just under your genuine scores:

```bash
docker compose logs speaker-id | grep identify   # per-turn score
docker compose logs speaker-id | grep diarize    # per-speaker aggregate score
```

There is large headroom in practice — an enrolled voice ~0.73, an unenrolled voice
~0.40, an impostor ~0.17 — so values near 0.5 still sit far above any non-match.
Diarization aggregates a speaker's segments before deciding, so its scores run
higher/steadier than the single-short-turn identify path. Re-calibrate if the
embedding model changes.

## Configuration

| Env | Default | Meaning |
|---|---|---|
| `EMBEDDING_MODEL` | `speechbrain/spkrec-ecapa-voxceleb` | embedding model id (`stub` for torch-free) |
| `EMBEDDING_DEVICE` | `cpu` | device for the embedder |
| `STORE_BACKEND` | `sqlite` | `sqlite` (Qdrant/Chroma later) |
| `STORE_PATH` | `/data/speakers.db` | store location (persist via the `speaker_data` volume) |
| `SIMILARITY_THRESHOLD` | `0.65` | cosine cutoff for "known" — **placeholder**, calibrate per model/channel |
| `AMBIGUOUS_MARGIN` | `0.08` | top-1 minus top-2 below this → `ambiguous` (only bites with multiple enrolled) |
| `MIN_SAMPLE_SECONDS` | `2.0` | reject enrollment clips with less speech than this (after trim) |
| `INSTALL_PYANNOTE` | `0` | **build arg** — `1` bakes pyannote.audio + ffmpeg into the image |
| `DIARIZATION_MODEL` | `stub` | `stub` (torch-free) or `pyannote/speaker-diarization-community-1` |
| `DIARIZATION_DEVICE` | `cpu` | device for the diarizer |
| `MIN_SEGMENT_SECONDS` | `0.4` | drop spans shorter than this before embedding |
| `MAX_DIARIZE_SEGMENTS` | `64` | cap spans embedded per clip (latency guard) |
| `HF_TOKEN` | *(none)* | HF token for gated weights (runtime; needed for pyannote) |
| `HF_HOME` | `/data/hf` | gated-weights cache (persisted via the volume) |
| `LOCAL_ADMIN_TOKEN` | *(none)* | local admin login + API bearer (admin gate) |
| `SESSION_SECRET` | *(none)* | signed-cookie session secret (set so logins survive restarts) |
| `OIDC_ENABLED` | `0` | `1` to turn on in-app OIDC SSO |
| `OIDC_DISCOVERY_URL` / `OIDC_CLIENT_ID` / `OIDC_CLIENT_SECRET` | *(none)* | IdP app credentials |
| `OIDC_ALLOWED_EMAILS` / `OIDC_ALLOWED_DOMAIN` | *(none)* | admin allowlist (set at least one) |
| `SERVICE_API_KEY` | *(none)* | data-plane bearer; match to s2s `SPEAKER_ID_API_KEY`. Empty = open |
| `INVITE_TTL_HOURS` | `72` | invite-link validity window (reusable until expiry/revoke) |
| `PUBLIC_BASE_URL` | *(none)* | external URL used to build the emailed invite link (behind a proxy) |
| `SMTP_HOST`/`PORT`/`USER`/`PASS`/`FROM`/`STARTTLS` | — | invite email relay (host unset → email is logged, not sent) |
| `LOG_LEVEL` | `info` | log verbosity |

The matching s2s-side client knobs (`SPEAKER_ID_ENABLED`, `SPEAKER_ID_BASE_URL`,
`SPEAKER_ID_API_KEY`, `SPEAKER_ID_TIMEOUT`, `SPEAKER_ID_LABEL_FORMAT`,
`SPEAKER_DIARIZE_ENABLED`, `SPEAKER_DIARIZE_TIMEOUT`) are documented in
[INSTALL_AND_CONFIGURATION.md](INSTALL_AND_CONFIGURATION.md) and `.env.sample`.

## Privacy note

Voice embeddings are **biometric data**. The enrollment flow captures consent, and
the store supports hard-delete; retention/TTL, encryption-at-rest, and an audit log
are tracked as remaining hardening. Encrypt the `speaker_data` volume at rest in
production.
