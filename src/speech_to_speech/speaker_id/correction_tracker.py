# Copyright 2026 winoiknow (Eric Alborn, Anteon Group)
# Licensed under the Apache License, Version 2.0 (the "License").

"""Idempotent, revision-versioned bookkeeping for diarization corrections.

Phase 4 (Tier 2) corrections arrive asynchronously and possibly out of order.
The protocol (IMPLEMENTATION_PLAN.md) requires they be:

  * **idempotent** — re-applying the same correction is a no-op;
  * **versioned** — a correction is ignored once a newer ``revision`` for the
    same ``item_id`` has already applied;
  * **dropped-safe** — never re-opens a turn; a no-op just leaves the prior label.

This tracks the last-applied ``revision`` per ``item_id`` and answers "should I
apply this one?". It's tiny and pure so both the emit side (skip emitting a stale
revision) and any consumer can share the exact same rule. Bounded LRU so a long
session can't grow it without limit.
"""

from __future__ import annotations

from collections import OrderedDict


class CorrectionTracker:
    def __init__(self, max_items: int = 1024) -> None:
        self._applied: "OrderedDict[str, int]" = OrderedDict()
        self._max = max_items

    def should_apply(self, item_id: str, revision: int) -> bool:
        """True iff ``revision`` is newer than any already applied for ``item_id``.

        Equal or older revisions return False (idempotent + out-of-order safe).
        Pure query — does not record. Call :meth:`apply` once you've acted on it.
        """
        last = self._applied.get(item_id)
        return last is None or revision > last

    def apply(self, item_id: str, revision: int) -> bool:
        """Record ``revision`` as applied for ``item_id`` if it's newer.

        Returns whether it was applied (same truth as :meth:`should_apply` at call
        time), so a caller can do ``if tracker.apply(id, rev): <act>`` atomically.
        """
        if not self.should_apply(item_id, revision):
            return False
        self._applied[item_id] = revision
        self._applied.move_to_end(item_id)
        while len(self._applied) > self._max:
            self._applied.popitem(last=False)  # evict oldest
        return True

    def forget(self, item_id: str) -> None:
        self._applied.pop(item_id, None)
