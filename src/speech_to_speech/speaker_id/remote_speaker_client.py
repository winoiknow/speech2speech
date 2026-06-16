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
import time

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
        # identify failures are fail-safe (→ unknown) and otherwise only log at
        # debug, which makes an unreachable/broken endpoint invisible at info
        # level. Surface it — but only once it's *sustained*: a single slow
        # identify (the service occasionally exceeds the hot-path timeout and the
        # turn is fail-safe-labeled 'unknown', then recovers) must not look like an
        # outage. Warn only after _FAIL_WARN_AFTER consecutive failures, then at
        # most every _FAIL_WARN_INTERVAL_S, and log a one-line recovery once it
        # succeeds again. State is single-writer (STT identify pool is max_workers=1).
        self._identify_failing = False
        self._consecutive_failures = 0
        self._last_fail_warn = 0.0
        logger.info("RemoteSpeakerClient ready → %s (timeout=%.2fs, diarize_timeout=%.2fs)",
                    self.endpoint, timeout, diarize_timeout)

    _FAIL_WARN_INTERVAL_S = 30.0
    # Consecutive identify failures before we warn — keeps a one-off slow/timed-out
    # identify (which recovers on the next turn) from logging a scary outage line.
    _FAIL_WARN_AFTER = 3

    def _note_identify_failure(self, exc: Exception) -> None:
        self._consecutive_failures += 1
        now = time.monotonic()
        sustained = self._consecutive_failures >= self._FAIL_WARN_AFTER
        due = not self._identify_failing or (now - self._last_fail_warn) >= self._FAIL_WARN_INTERVAL_S
        if sustained and due:
            logger.warning(
                "speaker identify is failing → all turns labeled 'unknown' "
                "(%d consecutive failures). Endpoint %s unreachable/erroring: %s: %s. "
                "(SPEAKER_ID_ENABLED is on; check the speaker-id service and SPEAKER_ID_BASE_URL reachability.)",
                self._consecutive_failures, self.endpoint, type(exc).__name__, exc,
            )
            self._last_fail_warn = now
            self._identify_failing = True

    def _note_identify_success(self) -> None:
        if self._identify_failing:
            logger.info("speaker identify recovered → %s reachable again", self.endpoint)
        self._identify_failing = False
        self._consecutive_failures = 0

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
            self._note_identify_success()
            return SpeakerLabel(
                decision=decision,
                speaker_id=d.get("speaker_id"),
                name=d.get("name"),
                score=float(d.get("score", 0.0) or 0.0),
                runner_up_score=float(d.get("runner_up_score", 0.0) or 0.0),
            )
        except Exception as e:  # timeout, connect error, bad json, anything → unknown
            # Fail-safe (→ unknown), but make a persistently-failing endpoint
            # visible at info level instead of only debug — see _note_identify_failure.
            self._note_identify_failure(e)
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
