# Copyright 2026 winoiknow (Eric Alborn, Anteon Group)
# Licensed under the Apache License, Version 2.0 (the "License").

"""Env-backed config for the speaker-id integration (Phase 3).

All from env (no CLI flags) — constructed directly in HandlerFactory, like the
VAD knobs. Off by default: with SPEAKER_ID_ENABLED unset/false, no client is
built, no call is made, and the pipeline is byte-for-byte today's.
"""

import os
from dataclasses import dataclass, field


def _env_bool(name: str, default: bool = False) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


@dataclass
class SpeakerIdHandlerArguments:
    speaker_id_enabled: bool = field(default_factory=lambda: _env_bool("SPEAKER_ID_ENABLED", False))
    speaker_id_base_url: str = field(
        default_factory=lambda: os.environ.get("SPEAKER_ID_BASE_URL", "http://speaker-id:9100")
    )
    speaker_id_api_key: str = field(default_factory=lambda: os.environ.get("SPEAKER_ID_API_KEY", ""))
    speaker_id_timeout: float = field(default_factory=lambda: float(os.environ.get("SPEAKER_ID_TIMEOUT", "0.8")))
    # Inline dialogue tag applied ONLY on a confident `known` match. {name} and
    # {speaker_id} are available. Empty string disables inline labeling.
    speaker_id_label_format: str = field(default_factory=lambda: os.environ.get("SPEAKER_ID_LABEL_FORMAT", "[{name}] "))
    # ── Phase 4 (Tier 2): async conference diarization, OFF the hot path ──
    # Separate flag from recognition: identify can run without diarization. When
    # off, no /v1/diarize call is made and no correction event is ever emitted.
    speaker_diarize_enabled: bool = field(default_factory=lambda: _env_bool("SPEAKER_DIARIZE_ENABLED", False))
    # Looser than the identify timeout — diarization is off the turn's hot path.
    speaker_diarize_timeout: float = field(
        default_factory=lambda: float(os.environ.get("SPEAKER_DIARIZE_TIMEOUT", "5.0"))
    )
