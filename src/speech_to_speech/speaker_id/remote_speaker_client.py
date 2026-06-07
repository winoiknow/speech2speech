# Copyright 2026 winoiknow (Eric Alborn, Anteon Group)
# Licensed under the Apache License, Version 2.0 (the "License").

"""Thin client for the adjacent ``speaker-id`` service.

Fires ``POST /v1/identify`` for a turn segment and returns a ``SpeakerLabel``.
**Never raises** — any timeout/error/bad-response yields ``decision="unknown"``,
so identity can never block or break the realtime turn. The decision rule
(threshold / ambiguous margin) lives server-side; this just carries the verdict.
No retries: a retry would risk doubling the hot-path latency budget.
"""

from __future__ import annotations

import logging

import httpx

from speech_to_speech.pipeline.messages import SpeakerCorrection, SpeakerLabel, SpeakerSpan

logger = logging.getLogger(__name__)

_VALID = {"known", "unknown", "ambiguous"}


class RemoteSpeakerClient:
    def __init__(self, base_url: str, api_key: str = "", timeout: float = 0.8,
                 diarize_timeout: float = 5.0) -> None:
        base = base_url.rstrip("/")
        self.endpoint = base + "/v1/identify"
        self.diarize_endpoint = base + "/v1/diarize"
        self.api_key = api_key
        self._client = httpx.Client(timeout=timeout)
        # diarization runs off the hot path → its own, looser timeout.
        self._diar_client = httpx.Client(timeout=diarize_timeout)
        logger.info("RemoteSpeakerClient ready → %s (timeout=%.2fs, diarize_timeout=%.2fs)",
                    self.endpoint, timeout, diarize_timeout)

    def identify(self, wav_bytes: bytes) -> SpeakerLabel:
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            r = self._client.post(self.endpoint, headers=headers,
                                  files={"file": ("seg.wav", wav_bytes, "audio/wav")})
            r.raise_for_status()
            d = r.json()
            decision = d.get("decision", "unknown")
            if decision not in _VALID:
                decision = "unknown"
            return SpeakerLabel(
                decision=decision,
                speaker_id=d.get("speaker_id"),
                name=d.get("name"),
                score=float(d.get("score", 0.0) or 0.0),
                runner_up_score=float(d.get("runner_up_score", 0.0) or 0.0),
            )
        except Exception as e:  # timeout, connect error, bad json, anything → unknown
            logger.debug("speaker identify failed (%s) → unknown", e)
            return SpeakerLabel(decision="unknown")

    def diarize(self, wav_bytes: bytes, item_id: str, revision: int = 1) -> SpeakerCorrection:
        """POST /v1/diarize → a SpeakerCorrection keyed to ``item_id`` (Phase 4).

        Off the hot path. **Never raises** — any timeout/error/bad-response yields
        a correction with no segments (dropped-safe: the Tier-1 label stands). The
        caller decides whether to emit it.
        """
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            r = self._diar_client.post(self.diarize_endpoint, headers=headers,
                                       files={"file": ("turn.wav", wav_bytes, "audio/wav")})
            r.raise_for_status()
            d = r.json()
            segments = []
            for s in d.get("segments", []) or []:
                decision = s.get("decision", "unknown")
                if decision not in _VALID:
                    decision = "unknown"
                segments.append(SpeakerSpan(
                    start=float(s.get("start", 0.0) or 0.0),
                    end=float(s.get("end", 0.0) or 0.0),
                    decision=decision,
                    speaker_id=s.get("speaker_id"),
                    name=s.get("name"),
                    label=s.get("label", "") or "",
                    score=float(s.get("score", 0.0) or 0.0),
                    runner_up_score=float(s.get("runner_up_score", 0.0) or 0.0),
                ))
            return SpeakerCorrection(item_id=item_id, revision=revision, segments=segments)
        except Exception as e:  # timeout, connect error, bad json, anything → empty
            logger.debug("speaker diarize failed (%s) → no correction", e)
            return SpeakerCorrection(item_id=item_id, revision=revision, segments=[])

    def close(self) -> None:
        for c in (self._client, getattr(self, "_diar_client", None)):
            try:
                if c is not None:
                    c.close()
            except Exception:
                pass
