# Copyright 2026 winoiknow (Eric Alborn, Anteon Group)
# Licensed under the Apache License, Version 2.0 (the "License").

"""The known / unknown / ambiguous decision rule.

Pure function of the ranked matches + two cutoffs, so it's trivially testable and
the *why* is always visible (every field echoes back to the caller).

  * top-1 score < threshold              → unknown   (nothing close enough)
  * top-1 − top-2 < ambiguous_margin     → ambiguous (two too close to call)
  * else                                 → known     (top-1's speaker)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .store import Match


@dataclass
class Verdict:
    decision: str  # "known" | "unknown" | "ambiguous"
    speaker_id: Optional[str]
    name: Optional[str]
    score: float
    runner_up_score: float


def decide(matches: list[Match], threshold: float, ambiguous_margin: float) -> Verdict:
    if not matches:
        return Verdict("unknown", None, None, 0.0, 0.0)
    top = matches[0]
    runner = matches[1].score if len(matches) > 1 else 0.0
    if top.score < threshold:
        return Verdict("unknown", None, None, top.score, runner)
    if (top.score - runner) < ambiguous_margin:
        return Verdict("ambiguous", None, None, top.score, runner)
    return Verdict("known", top.speaker_id, top.name, top.score, runner)
