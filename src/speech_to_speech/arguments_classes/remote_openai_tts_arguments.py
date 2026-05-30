# Copyright 2026 winoiknow (Eric Alborn, Anteon Group)
# Licensed under the Apache License, Version 2.0 (the "License").
# See the LICENSE file in the repository root for the full license text.

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class RemoteOpenAITTSHandlerArguments:
    tts_openai_base_url: str = field(
        default_factory=lambda: os.environ.get("TTS_OPENAI_BASE_URL", "http://localhost:8880"),
        metadata={"help": "Base URL for the OpenAI-compatible TTS endpoint. Env: TTS_OPENAI_BASE_URL."},
    )
    tts_openai_api_key: str = field(
        default_factory=lambda: os.environ.get("TTS_OPENAI_API_KEY", "sk-unused"),
        metadata={"help": "API key for the TTS endpoint. Env: TTS_OPENAI_API_KEY."},
    )
    tts_openai_voice: str = field(
        default_factory=lambda: os.environ.get("TTS_OPENAI_VOICE", "default"),
        metadata={"help": "Voice name sent to the TTS endpoint. Env: TTS_OPENAI_VOICE."},
    )
    tts_openai_model: str = field(
        default_factory=lambda: os.environ.get("TTS_OPENAI_MODEL", "tts-1"),
        metadata={"help": "Model name sent to the TTS endpoint. Env: TTS_OPENAI_MODEL. Default is 'tts-1'."},
    )
    tts_openai_timeout: float = field(
        default=60.0,
        metadata={"help": "HTTP timeout in seconds for TTS streaming requests. Default is 60."},
    )
    tts_openai_source_sample_rate: int = field(
        default_factory=lambda: int(os.environ.get("TTS_OPENAI_SOURCE_SAMPLE_RATE", "24000")),
        metadata={
            "help": "Sample rate (Hz) the TTS endpoint actually outputs, resampled to the 16 kHz "
            "pipeline rate. F5-TTS is natively 24000. Env: TTS_OPENAI_SOURCE_SAMPLE_RATE."
        },
    )
