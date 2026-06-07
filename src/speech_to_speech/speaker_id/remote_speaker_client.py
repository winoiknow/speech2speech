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

from speech_to_speech.pipeline.messages import SpeakerLabel

logger = logging.getLogger(__name__)

_VALID = {"known", "unknown", "ambiguous"}


class RemoteSpeakerClient:
    def __init__(self, base_url: str, api_key: str = "", timeout: float = 0.8) -> None:
        self.endpoint = base_url.rstrip("/") + "/v1/identify"
        self.api_key = api_key
        self._client = httpx.Client(timeout=timeout)
        logger.info("RemoteSpeakerClient ready → %s (timeout=%.2fs)", self.endpoint, timeout)

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

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass
