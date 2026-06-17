# Copyright 2026 winoiknow (Eric Alborn, Anteon Group)
# Licensed under the Apache License, Version 2.0 (the "License").

"""API contract for the speaker-id service.

These models are the wire contract the s2s `RemoteSpeakerClient` codes against.
They are stable across phases — Phase 0 ships a stub that always returns
``decision="unknown"``; later phases fill in the real model + store without
changing the shape.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

Decision = Literal["known", "unknown", "ambiguous"]


class IdentifyResponse(BaseModel):
    """Verdict for a single turn segment.

    ``decision`` is a first-class enum so ``ambiguous`` (top-1 and top-2 too
    close) is distinct from ``unknown`` (nothing close enough) — the caller can
    choose not to label rather than guess. Scores + the rule inputs always echo
    back so callers/operators can see *why* a decision was made.
    """

    decision: Decision = "unknown"
    speaker_id: Optional[str] = None  # set when decision == "known"
    name: Optional[str] = None  # display name when known
    score: float = 0.0  # top cosine similarity
    runner_up_score: float = 0.0  # second-best match
    threshold: float = 0.0  # the cutoff applied (for observability)
    ambiguous_margin: float = 0.0  # top1-top2 below this → ambiguous
    embedding_model: str = "stub"  # which model produced the embedding


class DiarizeSegment(BaseModel):
    """One speaker span of a diarized clip, with a recognition verdict.

    Spans are time-ordered. ``decision`` reuses the recognition enum per span.
    ``label`` is what a consumer should display: the enrolled ``name`` when
    ``known``, otherwise an **ephemeral per-call** tag (``S1``, ``S2`` …) that is
    stable *within this call only* — it is not an identity and does not persist.
    """

    start: float  # seconds from clip start
    end: float  # seconds from clip start
    decision: Decision = "unknown"
    speaker_id: Optional[str] = None  # enrolled id when decision == "known"
    name: Optional[str] = None  # enrolled display name when known
    label: str = ""  # name if known, else ephemeral per-call tag (S1, S2, …)
    score: float = 0.0
    runner_up_score: float = 0.0


class DiarizeResponse(BaseModel):
    """Spans + identities for a (possibly multi-speaker) clip.

    Off the s2s hot path: the turn is already transcribed/labelled by Tier-1;
    this drives the async corrective event that replaces span labels by item_id.
    """

    segments: list[DiarizeSegment] = Field(default_factory=list)
    speakers: int = 0  # distinct labels in this clip (enrolled + ephemeral)
    threshold: float = 0.0
    ambiguous_margin: float = 0.0
    embedding_model: str = "stub"
    diarization_model: str = "stub"


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"] = "ok"
    version: str
    embedding_model: str
    store_backend: str
    speakers: int = 0  # enrolled speaker count (0 until the store lands)
    diarization_model: str = "stub"  # Phase 4: which diarizer is loaded


class ErrorResponse(BaseModel):
    detail: str


# ── enrollment (Phase 1: basic CRUD; Phase 2 adds quality gates, consent, UI) ──


class SpeakerCreate(BaseModel):
    name: Optional[str] = None
    language: Optional[str] = None
    speaker_id: Optional[str] = None  # let the server mint one if omitted
    consent: bool = False  # required true — voice embeddings are biometric data


class SpeakerInfo(BaseModel):
    speaker_id: str
    name: Optional[str] = None
    language: Optional[str] = None
    samples: int = 0
    created: Optional[str] = None


class SpeakerList(BaseModel):
    speakers: list[SpeakerInfo] = Field(default_factory=list)


class AddSampleResponse(BaseModel):
    speaker_id: str
    sample_id: str
    samples: int  # total samples for this speaker after the add
    duration_s: float = 0.0


# ── invites (Phase 5B: email-invite self-enrollment) ──────────────────────────


class InviteCreate(BaseModel):
    name: Optional[str] = None  # invitee display name (also names the new speaker)
    email: Optional[str] = None  # where the invite link is sent
    speaker_id: Optional[str] = None  # bind to an existing speaker instead of creating one


class InviteInfo(BaseModel):
    id: str
    speaker_id: str
    name: Optional[str] = None
    email: Optional[str] = None
    created: Optional[str] = None
    expires: Optional[str] = None
    status: Literal["active", "expired", "revoked"] = "active"
    samples_added: int = 0
    last_used: Optional[str] = None


class InviteCreated(InviteInfo):
    # The full link is returned to the admin on create (and emailed). The raw token
    # is never stored — only its hash — so this is the one chance to capture it.
    invite_url: str = ""
    email_sent: bool = False


class InviteList(BaseModel):
    invites: list[InviteInfo] = Field(default_factory=list)


class InvitePage(BaseModel):
    """What the scoped invite page needs to know about its bound speaker."""

    speaker_id: str
    name: Optional[str] = None
    expires: Optional[str] = None
    consent: bool = False  # whether consent has been recorded yet
    samples: int = 0
