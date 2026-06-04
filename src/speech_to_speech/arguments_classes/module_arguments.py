# Copyright 2024 The HuggingFace Inc. team
# Licensed under the Apache License, Version 2.0 (the "License").
# See the LICENSE file in the repository root for the full license text.
#
# Modifications Copyright 2026 winoiknow (Eric Alborn, Anteon Group)
# Modified: added "openai-remote" to the stt and tts Literal type sets.

import os
from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class ModuleArguments:
    device: Optional[str] = field(
        default=None,
        metadata={"help": "If specified, overrides the device for all handlers."},
    )
    mode: Optional[Literal["local", "socket", "websocket", "realtime"]] = field(
        default="realtime",
        metadata={
            "help": "The mode to run the pipeline in. Either 'local', 'socket', 'websocket', or 'realtime'. Default is 'realtime'."
        },
    )
    local_mac_optimal_settings: bool = field(
        default=False,
        metadata={
            "help": "If specified, sets the optimal settings for Mac OS. Sets Parakeet TDT for STT, MLX LM for language model, and Qwen3-TTS for TTS, with MPS device and local mode."
        },
    )
    stt: Optional[
        Literal["whisper", "whisper-mlx", "mlx-audio-whisper", "faster-whisper", "parakeet-tdt", "paraformer", "openai-remote"]
    ] = field(
        default="parakeet-tdt",
        metadata={
            "help": "The STT to use. Either 'whisper', 'whisper-mlx', 'mlx-audio-whisper', 'faster-whisper', 'parakeet-tdt', 'paraformer', or 'openai-remote'. Default is 'parakeet-tdt'."
        },
    )
    llm_backend: Optional[Literal["transformers", "mlx-lm", "responses-api"]] = field(
        default="responses-api",
        metadata={
            "help": "The LLM backend to use. Either 'transformers', 'mlx-lm', or 'responses-api'. Default is 'responses-api'."
        },
    )
    tts: Optional[Literal["melo", "chatTTS", "facebookMMS", "pocket", "kokoro", "qwen3", "openai-remote", "elevenlabs", "minimax"]] = field(
        default="qwen3",
        metadata={
            "help": "The TTS to use. Either 'chatTTS', 'facebookMMS', 'pocket', 'kokoro', 'qwen3', 'openai-remote', 'elevenlabs', or 'minimax'. Default is 'qwen3'."
        },
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
