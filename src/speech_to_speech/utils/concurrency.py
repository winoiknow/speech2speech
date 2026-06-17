# Copyright 2026 winoiknow (Eric Alborn, Anteon Group)
# Licensed under the Apache License, Version 2.0 (the "License").
# See the LICENSE file in the repository root for the full license text.
#
# Multi-session Phase D3: per-service concurrency caps. N realtime sessions can
# drive up to N concurrent turns into the same shared STT/TTS/LLM endpoints. A
# process-wide semaphore per service lets s2s degrade fairly (queue) instead of
# melting an endpoint, while the default (0 = unlimited) preserves exactly
# today's single-session behavior. The real capacity knob is still the external
# service's own concurrency (Whisper batch slots, F5 instances, Hermes workers);
# see REMOTE_SETUP.md / docs/LATENCY.md before raising these.

from __future__ import annotations

import logging
import os
import threading
from contextlib import contextmanager
from typing import Iterator

logger = logging.getLogger(__name__)


def _cap(env_var: str) -> int:
    """Parse a concurrency cap from the environment. Missing/invalid/<=0 → 0
    (unlimited), so a typo can never accidentally throttle a deployment."""
    try:
        return max(0, int(os.getenv(env_var, "0")))
    except ValueError:
        logger.warning("%s is not an integer; treating as 0 (unlimited)", env_var)
        return 0


class ConcurrencyLimiter:
    """Process-wide cap on concurrent in-flight calls to one shared service.

    ``limit <= 0`` is unlimited: acquire/release/slot become no-ops, so there is
    zero overhead and byte-for-byte today's behavior when caps are unset. With a
    positive limit, callers over the cap block until a slot frees (a debug line
    is logged on the first wait so contention is visible)."""

    def __init__(self, name: str, limit: int) -> None:
        self.name = name
        self.limit = limit if limit and limit > 0 else 0
        self._sem = threading.BoundedSemaphore(self.limit) if self.limit else None

    def acquire(self) -> None:
        if self._sem is None:
            return
        if not self._sem.acquire(blocking=False):
            logger.debug("%s at concurrency cap %d — waiting for a slot", self.name, self.limit)
            self._sem.acquire()

    def release(self) -> None:
        if self._sem is None:
            return
        try:
            self._sem.release()
        except ValueError:
            # Defensive: a BoundedSemaphore raises on over-release. A cap must
            # never be the thing that crashes a handler, so swallow it.
            logger.debug("%s limiter over-release ignored", self.name)

    @contextmanager
    def slot(self) -> Iterator[None]:
        self.acquire()
        try:
            yield
        finally:
            self.release()


# Process-global singletons shared across every per-session handler instance.
# Sized once from the environment at import.
STT_LIMITER = ConcurrencyLimiter("STT", _cap("STT_MAX_CONCURRENCY"))
TTS_LIMITER = ConcurrencyLimiter("TTS", _cap("TTS_MAX_CONCURRENCY"))
LLM_LIMITER = ConcurrencyLimiter("LLM", _cap("LLM_MAX_CONCURRENCY"))
