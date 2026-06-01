# Copyright 2026 winoiknow (Eric Alborn, Anteon Group)
# Licensed under the Apache License, Version 2.0 (the "License").
# See the LICENSE file in the repository root for the full license text.

"""Single lever for the verbose diagnostic logging added while debugging the
AVA/Asterisk barge-in + audio-rate issues.

Set ``DEBUG_MODE=on`` in the environment (.env) to re-enable the per-second VAD
heartbeat, per-chunk drop logging, and the chatty TTS-stream timing lines.
Default is off, so production stays quiet without losing the instrumentation.
"""

from __future__ import annotations

import os

DEBUG_MODE: bool = os.environ.get("DEBUG_MODE", "off").strip().lower() in (
    "on",
    "1",
    "true",
    "yes",
)
