# Copyright 2026 winoiknow (Eric Alborn, Anteon Group)
# Licensed under the Apache License, Version 2.0 (the "License").
# See the LICENSE file in the repository root for the full license text.

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class ElevenLabsTTSHandlerArguments:
    """Args for the ElevenLabs streaming TTS handler.

    All values default from the environment so the source can be flipped with a
    single ``TTS_SOURCE=elevenlabs`` toggle plus credentials in ``.env`` — no CLI
    args required. Field names carry the ``tts_elevenlabs_`` prefix so
    ``rename_args`` strips it to the handler's ``setup()`` kwarg names.
    """

    tts_elevenlabs_api_key: str = field(
        default_factory=lambda: os.environ.get("ELEVENLABS_API_KEY", ""),
        metadata={"help": "ElevenLabs API key (xi-api-key). Env: ELEVENLABS_API_KEY."},
    )
    tts_elevenlabs_voice_id: str = field(
        default_factory=lambda: os.environ.get("ELEVENLABS_VOICE_ID", ""),
        metadata={"help": "ElevenLabs voice id to synthesize with. Env: ELEVENLABS_VOICE_ID."},
    )
    tts_elevenlabs_model_id: str = field(
        default_factory=lambda: os.environ.get("ELEVENLABS_MODEL_ID", "eleven_flash_v2_5"),
        metadata={
            "help": "ElevenLabs model id. Default 'eleven_flash_v2_5' (lowest latency). "
            "Env: ELEVENLABS_MODEL_ID."
        },
    )
    tts_elevenlabs_output_format: str = field(
        default_factory=lambda: os.environ.get("ELEVENLABS_OUTPUT_FORMAT", "pcm_16000"),
        metadata={
            "help": "ElevenLabs output_format. 'pcm_16000' (default, no resample; PCM needs a "
            "paid tier) or other 'pcm_<rate>' (resampled to 16 kHz), or 'ulaw_8000' "
            "(free-tier friendly). mp3 is not supported. Env: ELEVENLABS_OUTPUT_FORMAT."
        },
    )
    tts_elevenlabs_base_url: str = field(
        default_factory=lambda: os.environ.get("ELEVENLABS_BASE_URL", "https://api.elevenlabs.io"),
        metadata={"help": "ElevenLabs API base URL. Env: ELEVENLABS_BASE_URL."},
    )
    tts_elevenlabs_timeout: float = field(
        default_factory=lambda: float(os.environ.get("ELEVENLABS_TIMEOUT", "60.0")),
        metadata={"help": "HTTP timeout (s) for streaming requests. Env: ELEVENLABS_TIMEOUT."},
    )
    tts_elevenlabs_stability: float = field(
        default_factory=lambda: float(os.environ.get("ELEVENLABS_STABILITY", "0.5")),
        metadata={"help": "voice_settings.stability (0..1). Env: ELEVENLABS_STABILITY."},
    )
    tts_elevenlabs_similarity_boost: float = field(
        default_factory=lambda: float(os.environ.get("ELEVENLABS_SIMILARITY_BOOST", "0.75")),
        metadata={"help": "voice_settings.similarity_boost (0..1). Env: ELEVENLABS_SIMILARITY_BOOST."},
    )
