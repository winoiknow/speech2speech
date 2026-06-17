# Copyright 2026 winoiknow (Eric Alborn, Anteon Group)
# Licensed under the Apache License, Version 2.0 (the "License").

"""speaker-id service — Phase 1 (recognition core).

  GET  /healthz                    liveness + model/store/speaker count
  POST /v1/identify                identify one turn segment → known/unknown/ambiguous
  POST /v1/speakers                create a speaker            (basic enrollment)
  POST /v1/speakers/{id}/samples   add an enrollment sample (embed + store)
  GET  /v1/speakers                list enrolled speakers
  DELETE /v1/speakers/{id}         remove a speaker + its embeddings

The embedding model + store sit behind pluggable interfaces (embedding.Embedder,
store.SpeakerStore), so the model (ECAPA today, pyannote later) or the backend
(SQLite today, Qdrant/Chroma later) can change without touching the API.

NOTE: the enrollment endpoints here are basic CRUD for testing the recognition
core. Phase 2 layers on enrollment quality gates, consent capture, access control
and the browser UI before this is production-usable for biometric data.
"""

from __future__ import annotations

import hashlib
import logging
import os
import secrets
from collections import OrderedDict
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

import numpy as np
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
from starlette.middleware.sessions import SessionMiddleware

from . import auth, mailer

from .audio import decode_to_f32_16k
from .decision import Verdict, decide
from .diarization import make_diarizer
from .embedding import make_embedder
from .models import (
    AddSampleResponse,
    DiarizeResponse,
    DiarizeSegment,
    HealthResponse,
    IdentifyResponse,
    InviteCreate,
    InviteCreated,
    InviteInfo,
    InviteList,
    InvitePage,
    SpeakerCreate,
    SpeakerInfo,
    SpeakerList,
)
from .quality import check_sample
from .store import SqliteNumpyStore

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "info").upper())
logger = logging.getLogger("speaker_id")

VERSION = "0.5.0-phase5b"
_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
MIN_SAMPLE_SECONDS = float(os.environ.get("MIN_SAMPLE_SECONDS", "2.0"))
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "speechbrain/spkrec-ecapa-voxceleb")
EMBEDDING_SAVEDIR = os.environ.get("EMBEDDING_SAVEDIR", "/models/ecapa")
EMBEDDING_DEVICE = os.environ.get("EMBEDDING_DEVICE", "cpu")
STORE_BACKEND = os.environ.get("STORE_BACKEND", "sqlite")
STORE_PATH = os.environ.get("STORE_PATH", "/data/speakers.db")
SIMILARITY_THRESHOLD = float(os.environ.get("SIMILARITY_THRESHOLD", "0.65"))
AMBIGUOUS_MARGIN = float(os.environ.get("AMBIGUOUS_MARGIN", "0.08"))
# Phase 4 (Tier 2) diarization. Default "stub" = torch-free, one span per clip;
# set to pyannote/speaker-diarization-community-1 (gated, needs HF_TOKEN) to split.
DIARIZATION_MODEL = os.environ.get("DIARIZATION_MODEL", "stub")
DIARIZATION_DEVICE = os.environ.get("DIARIZATION_DEVICE", EMBEDDING_DEVICE)
# cosine to group two unenrolled spans under one ephemeral per-call label.
HF_TOKEN = os.environ.get("HF_TOKEN") or None
# cap spans embedded per clip so a pathological diarization can't blow the budget.
MAX_DIARIZE_SEGMENTS = int(os.environ.get("MAX_DIARIZE_SEGMENTS", "64"))
# spans shorter than this (seconds) are too brief for a reliable embedding → skipped.
MIN_SEGMENT_SECONDS = float(os.environ.get("MIN_SEGMENT_SECONDS", "0.4"))
# Phase 5B invites: how long a self-enrollment link stays valid (time-window token,
# reusable until it expires or is revoked — not single-use).
INVITE_TTL_HOURS = int(os.environ.get("INVITE_TTL_HOURS", "72"))
# External base URL used to build the invite link in the email. Falls back to the
# request's base URL if unset (set it when behind a proxy so the link is right).
PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "").rstrip("/")

_state: dict = {}


def _l2(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return (v / n).astype(np.float32) if n > 0 else v.astype(np.float32)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("speaker-id %s starting: model=%s store=%s@%s", VERSION, EMBEDDING_MODEL, STORE_BACKEND, STORE_PATH)
    embedder = make_embedder(EMBEDDING_MODEL, savedir=EMBEDDING_SAVEDIR, device=EMBEDDING_DEVICE)
    embedder.warmup()  # load weights now so the first identify isn't cold
    os.makedirs(os.path.dirname(STORE_PATH) or ".", exist_ok=True)
    store = SqliteNumpyStore(STORE_PATH, dim=embedder.dim, embedding_model=embedder.model_id)
    diarizer = make_diarizer(DIARIZATION_MODEL, device=DIARIZATION_DEVICE, hf_token=HF_TOKEN)
    diarizer.warmup()  # no-op for the stub; loads pyannote weights when selected
    _state["embedder"], _state["store"], _state["diarizer"] = embedder, store, diarizer
    logger.info("ready: dim=%d, %d enrolled speakers, diarizer=%s",
                embedder.dim, store.count(), diarizer.model_id)
    auth.log_startup_state()  # warn loudly if the admin gate is open
    yield
    _state.clear()


app = FastAPI(title="speaker-id", version=VERSION, lifespan=lifespan)
# Signed-cookie sessions back the admin login (OIDC / local-admin). Same-site lax
# so the OIDC redirect carries the session; https_only is enforced by your proxy.
app.add_middleware(SessionMiddleware, secret_key=auth.session_secret(),
                   same_site="lax", session_cookie="spk_admin")
app.include_router(auth.router)


@app.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    store = _state.get("store")
    embedder = _state.get("embedder")
    return HealthResponse(
        status="ok" if store and embedder else "degraded",
        version=VERSION,
        embedding_model=embedder.model_id if embedder else EMBEDDING_MODEL,
        store_backend=STORE_BACKEND,
        speakers=store.count() if store else 0,
        diarization_model=_state["diarizer"].model_id if _state.get("diarizer") else DIARIZATION_MODEL,
    )


@app.post("/v1/identify", response_model=IdentifyResponse)
async def identify(file: UploadFile = File(...), sample_rate: int | None = Form(default=None),
                   _: None = Depends(auth.require_service_key)) -> IdentifyResponse:
    embedder, store = _state["embedder"], _state["store"]
    data = await file.read()
    try:
        audio = decode_to_f32_16k(data)
        emb = embedder.embed(audio)
    except Exception as e:
        # Decode/embed failure must not look like a confident match.
        logger.warning("identify: decode/embed failed (%s) → unknown", e)
        return IdentifyResponse(decision="unknown", threshold=SIMILARITY_THRESHOLD,
                                ambiguous_margin=AMBIGUOUS_MARGIN, embedding_model=embedder.model_id)
    v = decide(store.query(emb, top_k=5), SIMILARITY_THRESHOLD, AMBIGUOUS_MARGIN)
    logger.info("identify: decision=%s id=%s score=%.3f rup=%.3f", v.decision, v.speaker_id, v.score, v.runner_up_score)
    return IdentifyResponse(
        decision=v.decision, speaker_id=v.speaker_id, name=v.name,
        score=round(v.score, 4), runner_up_score=round(v.runner_up_score, 4),
        threshold=SIMILARITY_THRESHOLD, ambiguous_margin=AMBIGUOUS_MARGIN,
        embedding_model=embedder.model_id,
    )


@app.post("/v1/diarize", response_model=DiarizeResponse)
async def diarize(file: UploadFile = File(...), sample_rate: int | None = Form(default=None),
                  _: None = Depends(auth.require_service_key)) -> DiarizeResponse:
    """Split a clip into speaker spans and identify each (Phase 4, Tier 2).

    Off the s2s hot path — drives the async corrective event. Decode → diarize →
    group spans by the diarizer's own speaker label → aggregate each speaker's
    audio → identify ONCE per speaker (same store/rule as /v1/identify) → ephemeral
    label per unenrolled speaker. Fail-safe: any failure yields an empty segment
    list rather than a wrong attribution.

    We trust the diarizer's clustering (pyannote already groups the audio by voice
    internally) instead of re-clustering short per-segment embeddings — short clips
    give noisy embeddings that fragment one voice into many labels. Aggregating a
    speaker's segments also means more audio per identify → better, more stable
    recall than judging each short segment alone.
    """
    embedder, store, diarizer = _state["embedder"], _state["store"], _state["diarizer"]
    data = await file.read()
    try:
        audio = decode_to_f32_16k(data)
        spans = diarizer.diarize(audio)
    except Exception as e:
        logger.warning("diarize: decode/diarize failed (%s) → no segments", e)
        return DiarizeResponse(threshold=SIMILARITY_THRESHOLD, ambiguous_margin=AMBIGUOUS_MARGIN,
                               embedding_model=embedder.model_id, diarization_model=diarizer.model_id)

    spans = [s for s in spans if (s.end - s.start) >= MIN_SEGMENT_SECONDS][:MAX_DIARIZE_SEGMENTS]

    # Group spans by the diarizer-local speaker label (SPEAKER_00, …).
    groups: "OrderedDict[str, list]" = OrderedDict()
    for sp in spans:
        groups.setdefault(sp.speaker, []).append(sp)

    # One verdict per speaker, from the mean of that speaker's segment embeddings.
    verdicts: dict[str, Verdict] = {}
    for spk, members in groups.items():
        embs = []
        for sp in members:
            a, b = max(0, int(sp.start * 16000)), int(sp.end * 16000)
            try:
                embs.append(embedder.embed(audio[a:b]))
            except Exception as e:
                logger.warning("diarize: %s span %.2f-%.2f embed failed (%s)", spk, sp.start, sp.end, e)
        if embs:
            agg = _l2(np.mean(np.vstack(embs), axis=0))
            verdicts[spk] = decide(store.query(agg, top_k=5), SIMILARITY_THRESHOLD, AMBIGUOUS_MARGIN)
        else:
            verdicts[spk] = decide([], SIMILARITY_THRESHOLD, AMBIGUOUS_MARGIN)

    # Ephemeral per-call labels (S1, S2, …) — one per UNENROLLED speaker, numbered
    # by first appearance. Stable within this call only; never persisted, never an
    # identity claim. A speaker recognized as enrolled uses their name instead.
    unknown_speakers = sorted(
        (spk for spk, v in verdicts.items() if v.decision != "known"),
        key=lambda spk: min(s.start for s in groups[spk]),
    )
    ephemeral_label = {spk: f"S{i + 1}" for i, spk in enumerate(unknown_speakers)}

    segments = [
        DiarizeSegment(
            start=round(sp.start, 3), end=round(sp.end, 3),
            decision=verdicts[sp.speaker].decision,
            speaker_id=verdicts[sp.speaker].speaker_id, name=verdicts[sp.speaker].name,
            label=(verdicts[sp.speaker].name if verdicts[sp.speaker].decision == "known"
                   else ephemeral_label.get(sp.speaker, "")) or "",
            score=round(verdicts[sp.speaker].score, 4),
            runner_up_score=round(verdicts[sp.speaker].runner_up_score, 4),
        )
        for sp in sorted(spans, key=lambda s: s.start)
    ]

    known = {v.speaker_id for v in verdicts.values() if v.decision == "known"}
    logger.info("diarize: %d spans, %d speakers → %d known + %d ephemeral",
                len(segments), len(groups), len(known), len(ephemeral_label))
    return DiarizeResponse(
        segments=segments, speakers=len(known) + len(ephemeral_label),
        threshold=SIMILARITY_THRESHOLD, ambiguous_margin=AMBIGUOUS_MARGIN,
        embedding_model=embedder.model_id, diarization_model=diarizer.model_id,
    )


@app.post("/v1/speakers", response_model=SpeakerInfo)
def create_speaker(body: SpeakerCreate, admin: dict = Depends(auth.require_admin)) -> SpeakerInfo:
    # Voice embeddings are biometric data — refuse to create a speaker without
    # recorded consent (Phase 5 adds the full audit log).
    if not body.consent:
        raise HTTPException(status_code=400, detail="consent required to store a voice profile")
    store = _state["store"]
    sid = store.add_speaker(name=body.name, language=body.language, speaker_id=body.speaker_id, consent=True)
    return SpeakerInfo(speaker_id=sid, name=body.name, language=body.language, samples=0)


@app.post("/v1/speakers/{speaker_id}/samples", response_model=AddSampleResponse)
async def add_sample(speaker_id: str, file: UploadFile = File(...),
                     admin: dict = Depends(auth.require_admin)) -> AddSampleResponse:
    embedder, store = _state["embedder"], _state["store"]
    if not any(s.speaker_id == speaker_id for s in store.list_speakers()):
        raise HTTPException(status_code=404, detail=f"unknown speaker_id {speaker_id}")
    data = await file.read()
    try:
        audio = decode_to_f32_16k(data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"bad audio: {e}")
    # Quality gate — embed the TRIMMED speech; reject unusable clips (422) so the
    # UI can show the reason and the bank isn't poisoned by bad samples.
    q = check_sample(audio, min_seconds=MIN_SAMPLE_SECONDS)
    if not q.ok:
        logger.info("add_sample: rejected (%s)", q.reason)
        raise HTTPException(status_code=422, detail=q.reason)
    try:
        emb = embedder.embed(q.audio)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"embed failed: {e}")
    sample_id = store.add_embedding(speaker_id, emb, sample_source=file.filename or "")
    total = next((s.samples for s in store.list_speakers() if s.speaker_id == speaker_id), 0)
    logger.info("add_sample: speaker=%s accepted %.1fs (rms=%.3f) → %d samples", speaker_id, q.duration_s, q.rms, total)
    return AddSampleResponse(speaker_id=speaker_id, sample_id=sample_id, samples=total, duration_s=round(q.duration_s, 2))


@app.get("/enroll", include_in_schema=False)
def enroll_page(request: Request):
    # Browser surface — redirect to the login page instead of a 401 when not signed in.
    if auth.admin_session(request) is None:
        return RedirectResponse(url="/auth/login", status_code=303)
    return FileResponse(os.path.join(_STATIC_DIR, "enroll.html"))


@app.get("/v1/speakers", response_model=SpeakerList)
def list_speakers(admin: dict = Depends(auth.require_admin)) -> SpeakerList:
    store = _state["store"]
    return SpeakerList(speakers=[
        SpeakerInfo(speaker_id=s.speaker_id, name=s.name, language=s.language, samples=s.samples, created=s.created)
        for s in store.list_speakers()
    ])


@app.delete("/v1/speakers/{speaker_id}")
def delete_speaker(speaker_id: str, admin: dict = Depends(auth.require_admin)) -> dict:
    store = _state["store"]
    if not store.delete_speaker(speaker_id):
        raise HTTPException(status_code=404, detail=f"unknown speaker_id {speaker_id}")
    return {"deleted": speaker_id}


# ── Email-invite self-enrollment (Phase 5B) ───────────────────────────────────
# Admin endpoints are gated by require_admin. The invitee endpoints below are
# gated ONLY by the invite token (the token IS the auth) — deliberately NO OIDC /
# admin session, so an outside user with no IdP account can enroll. The token is a
# time-window credential (reusable until it expires or is revoked), stored only as
# a SHA-256 hash; the scoped page sees just its own speaker.

def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _invite_status(inv) -> str:
    if inv.revoked:
        return "revoked"
    try:
        exp = datetime.strptime(inv.expires, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except Exception:
        return "active"
    return "expired" if datetime.now(timezone.utc) >= exp else "active"


def _validate_invite(token: str):
    """Return (Invite, speaker_name) for a usable token, else 404 (unknown) / 410 (gone)."""
    rec = _state["store"].get_invite_by_token_hash(_token_hash(token))
    if rec is None:
        raise HTTPException(status_code=404, detail="invalid invite token")
    inv, name = rec
    status = _invite_status(inv)
    if status != "active":
        raise HTTPException(status_code=410, detail=f"invite {status}")
    return inv, name


def _invite_page(inv, name: str | None) -> InvitePage:
    store = _state["store"]
    samples = next((s.samples for s in store.list_speakers() if s.speaker_id == inv.speaker_id), 0)
    return InvitePage(speaker_id=inv.speaker_id, name=name, expires=inv.expires,
                      consent=store.consent_status(inv.speaker_id), samples=samples)


@app.post("/v1/invites", response_model=InviteCreated)
def create_invite(body: InviteCreate, request: Request,
                  admin: dict = Depends(auth.require_admin)) -> InviteCreated:
    store = _state["store"]
    # Bind to an existing speaker, or create one now. Consent is captured later by
    # the invitee on the scoped page, so the speaker starts with consent=False.
    if body.speaker_id:
        match = next((s for s in store.list_speakers() if s.speaker_id == body.speaker_id), None)
        if match is None:
            raise HTTPException(status_code=404, detail=f"unknown speaker_id {body.speaker_id}")
        sid, name = body.speaker_id, match.name
    else:
        sid = store.add_speaker(name=body.name, language=None, consent=False)
        name = body.name
    token = secrets.token_urlsafe(32)
    expires = (datetime.now(timezone.utc) + timedelta(hours=INVITE_TTL_HOURS)).strftime("%Y-%m-%dT%H:%M:%SZ")
    iid = store.add_invite(sid, body.email, _token_hash(token), expires)
    base = PUBLIC_BASE_URL or str(request.base_url).rstrip("/")
    link = f"{base}/enroll/invite?token={token}"
    sent = mailer.send_invite_email(body.email, link, INVITE_TTL_HOURS, name)
    logger.info("invite %s created → speaker=%s email=%s expires=%s", iid, sid, body.email, expires)
    return InviteCreated(id=iid, speaker_id=sid, name=name, email=body.email, expires=expires,
                         status="active", samples_added=0, invite_url=link, email_sent=sent)


@app.get("/v1/invites", response_model=InviteList)
def list_invites(admin: dict = Depends(auth.require_admin)) -> InviteList:
    return InviteList(invites=[
        InviteInfo(id=inv.id, speaker_id=inv.speaker_id, name=name, email=inv.email,
                   created=inv.created, expires=inv.expires, status=_invite_status(inv),
                   samples_added=inv.samples_added, last_used=inv.last_used)
        for inv, name in _state["store"].list_invites()
    ])


@app.delete("/v1/invites/{invite_id}")
def delete_invite(invite_id: str, admin: dict = Depends(auth.require_admin)) -> dict:
    if not _state["store"].revoke_invite(invite_id):
        raise HTTPException(status_code=404, detail=f"unknown invite {invite_id}")
    return {"revoked": invite_id}


@app.get("/enroll/invite", include_in_schema=False)
def enroll_invite_page(token: str = "") -> FileResponse:
    # Token-gated, NO OIDC. Served regardless of token validity; the page calls
    # GET /v1/invite/{token} and shows a friendly message if it's invalid/expired.
    return FileResponse(os.path.join(_STATIC_DIR, "enroll_invite.html"))


@app.get("/v1/invite/{token}", response_model=InvitePage)
def invite_info(token: str) -> InvitePage:
    inv, name = _validate_invite(token)
    return _invite_page(inv, name)


@app.post("/v1/invite/{token}/consent", response_model=InvitePage)
def invite_consent(token: str) -> InvitePage:
    inv, name = _validate_invite(token)
    _state["store"].set_consent(inv.speaker_id, True)
    logger.info("invite %s: consent recorded for speaker=%s", inv.id, inv.speaker_id)
    return _invite_page(inv, name)


@app.post("/v1/invite/{token}/samples", response_model=AddSampleResponse)
async def invite_add_sample(token: str, file: UploadFile = File(...)) -> AddSampleResponse:
    inv, _name = _validate_invite(token)
    embedder, store = _state["embedder"], _state["store"]
    if not store.consent_status(inv.speaker_id):
        raise HTTPException(status_code=403, detail="consent required before recording")
    data = await file.read()
    try:
        audio = decode_to_f32_16k(data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"bad audio: {e}")
    q = check_sample(audio, min_seconds=MIN_SAMPLE_SECONDS)
    if not q.ok:
        raise HTTPException(status_code=422, detail=q.reason)
    try:
        emb = embedder.embed(q.audio)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"embed failed: {e}")
    sample_id = store.add_embedding(inv.speaker_id, emb, sample_source=f"invite:{inv.id}")
    store.touch_invite(inv.id)
    total = next((s.samples for s in store.list_speakers() if s.speaker_id == inv.speaker_id), 0)
    logger.info("invite %s: speaker=%s accepted %.1fs → %d samples", inv.id, inv.speaker_id, q.duration_s, total)
    return AddSampleResponse(speaker_id=inv.speaker_id, sample_id=sample_id, samples=total,
                             duration_s=round(q.duration_s, 2))


@app.post("/v1/invite/{token}/test")
async def invite_test(token: str, file: UploadFile = File(...)) -> dict:
    # Scoped self-test: score ONLY against the bound speaker — never reveals or
    # matches any other enrolled voice (no cross-user visibility).
    inv, _name = _validate_invite(token)
    embedder, store = _state["embedder"], _state["store"]
    data = await file.read()
    try:
        audio = decode_to_f32_16k(data)
        emb = embedder.embed(audio)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"bad audio: {e}")
    score = store.score_against_speaker(emb, inv.speaker_id)
    return {"score": round(score, 4), "threshold": SIMILARITY_THRESHOLD,
            "match": bool(score >= SIMILARITY_THRESHOLD)}
