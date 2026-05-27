# Copyright 2026 winoiknow (Eric Alborn, Anteon Group)
# Licensed under the Apache License, Version 2.0 (the "License").
# See the LICENSE file in the repository root for the full license text.

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RemoteOpenAISTTHandlerArguments:
    stt_openai_base_url: str = field(
        default_factory=lambda: os.environ.get("STT_OPENAI_BASE_URL", "http://localhost:8000"),
        metadata={"help": "Base URL for the OpenAI-compatible STT endpoint. Env: STT_OPENAI_BASE_URL."},
    )
    stt_openai_api_key: str = field(
        default_factory=lambda: os.environ.get("STT_OPENAI_API_KEY", "sk-unused"),
        metadata={"help": "API key for the STT endpoint. Env: STT_OPENAI_API_KEY."},
    )
    stt_openai_model: str = field(
        default_factory=lambda: os.environ.get("STT_OPENAI_MODEL", "Systran/faster-whisper-large-v3"),
        metadata={"help": "Model identifier sent to the STT endpoint. Env: STT_OPENAI_MODEL."},
    )
    stt_openai_language: Optional[str] = field(
        default_factory=lambda: os.environ.get("STT_OPENAI_LANGUAGE", "en"),
        metadata={"help": "Language hint for transcription (ISO-639-1). Env: STT_OPENAI_LANGUAGE."},
    )
    stt_openai_timeout: float = field(
        default=30.0,
        metadata={"help": "HTTP timeout in seconds for STT requests. Default is 30."},
    )
