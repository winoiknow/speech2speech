# Copyright 2024 The HuggingFace Inc. team
# Licensed under the Apache License, Version 2.0 (the "License").
# See the LICENSE file in the repository root for the full license text.
#
# Modifications Copyright 2026 winoiknow (Eric Alborn, Anteon Group)
# Streamlined to a remote-only realtime build: the only valid STT/TTS/LLM backends
# are the remote handlers, and 'realtime' is the only mode.

import os
from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class ModuleArguments:
    mode: Literal["realtime"] = field(
        default="realtime",
        metadata={"help": "Run mode. This build only supports 'realtime'."},
    )
    host: str = field(
        default_factory=lambda: os.environ.get("S2S_HOST", "0.0.0.0"),
        metadata={"help": "Host the realtime WebSocket server binds to (env S2S_HOST, default 0.0.0.0)."},
    )
    port: int = field(
        default_factory=lambda: int(os.environ.get("S2S_PORT", "8765")),
        metadata={"help": "Port the realtime WebSocket server binds to (env S2S_PORT, default 8765)."},
    )
    stt: Optional[Literal["openai-remote"]] = field(
        default="openai-remote",
        metadata={"help": "STT backend. This build only supports 'openai-remote'."},
    )
    llm_backend: Optional[Literal["responses-api"]] = field(
        default="responses-api",
        metadata={"help": "LLM backend. This build only supports 'responses-api'."},
    )
    tts: Optional[Literal["openai-remote", "elevenlabs", "minimax"]] = field(
        default="openai-remote",
        metadata={"help": "TTS backend. Either 'openai-remote' (F5-TTS), 'elevenlabs', or 'minimax'."},
    )
    log_level: str = field(
        default="info",
        metadata={"help": "Provide logging level. Example --log_level debug, default=info."},
    )
    enable_live_transcription: bool = field(
        default=True,
        metadata={
            "help": "Enable live transcription display while user is speaking (works with parakeet-tdt). Default is true."
        },
    )
    live_transcription_update_interval: float = field(
        default=0.25,
        metadata={"help": "Update interval for live transcription in seconds (default: 0.25s = 250ms)"},
    )
    live_transcription_min_silence_ms: int = field(
        default=500,
        metadata={
            "help": "Minimum silence duration (ms) before ending speech when live transcription is enabled (default: 500ms)"
        },
    )
    server_api_key: Optional[str] = field(
        default_factory=lambda: os.environ.get("SERVER_API_KEY"),
        metadata={
            "help": (
                "Bearer token required for incoming WebSocket connections. "
                "If set, clients must supply 'Authorization: Bearer <key>'. "
                "Defaults to the SERVER_API_KEY environment variable; "
                "if neither is set, authentication is disabled."
            )
        },
    )
