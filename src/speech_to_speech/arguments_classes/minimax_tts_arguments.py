# Copyright 2026 winoiknow (Eric Alborn, Anteon Group)
# Licensed under the Apache License, Version 2.0 (the "License").
# See the LICENSE file in the repository root for the full license text.

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class MiniMaxTTSHandlerArguments:
    """Args for the MiniMax T2A v2 WebSocket streaming TTS handler.

    Env-backed so the source flips with a single ``TTS_SOURCE=minimax`` toggle
    plus credentials in ``.env``. Field names carry the ``tts_minimax_`` prefix
    so ``rename_args`` strips it to the handler's ``setup()`` kwarg names.
    """

    tts_minimax_api_key: str = field(
        default_factory=lambda: os.environ.get("MINIMAX_API_KEY", ""),
        metadata={"help": "MiniMax API key (Bearer token). Env: MINIMAX_API_KEY."},
    )
    tts_minimax_voice_id: str = field(
        default_factory=lambda: os.environ.get("MINIMAX_VOICE_ID", ""),
        metadata={
            "help": "MiniMax voice id — a cloned voice, or a system voice (e.g. "
            "'English_expressive_narrator'). Env: MINIMAX_VOICE_ID."
        },
    )
    tts_minimax_model: str = field(
        default_factory=lambda: os.environ.get("MINIMAX_MODEL", "speech-02-turbo"),
        metadata={"help": "MiniMax model id. Default 'speech-02-turbo' (low latency). Env: MINIMAX_MODEL."},
    )
    tts_minimax_ws_url: str = field(
        default_factory=lambda: os.environ.get("MINIMAX_WS_URL", "wss://api.minimax.io/ws/v1/t2a_v2"),
        metadata={
            "help": "MiniMax T2A v2 WebSocket URL. Use the api.minimaxi.com host for the mainland "
            "endpoint. Env: MINIMAX_WS_URL."
        },
    )
    tts_minimax_group_id: str = field(
        default_factory=lambda: os.environ.get("MINIMAX_GROUP_ID", ""),
        metadata={
            "help": "Optional GroupId (appended as ?GroupId=…) if your account requires it. Env: MINIMAX_GROUP_ID."
        },
    )
    tts_minimax_speed: float = field(
        default_factory=lambda: float(os.environ.get("MINIMAX_SPEED", "1.0")),
        metadata={"help": "voice_setting.speed (1.0 = normal). Env: MINIMAX_SPEED."},
    )
    tts_minimax_timeout: float = field(
        default_factory=lambda: float(os.environ.get("MINIMAX_TIMEOUT", "20.0")),
        metadata={"help": "WebSocket connect/read timeout in seconds. Env: MINIMAX_TIMEOUT."},
    )
