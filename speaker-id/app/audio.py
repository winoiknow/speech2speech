# Copyright 2026 winoiknow (Eric Alborn, Anteon Group)
# Licensed under the Apache License, Version 2.0 (the "License").

"""Audio decode → float32 mono @ 16 kHz (what the embedder expects)."""

from __future__ import annotations

import io
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

TARGET_SR = 16000


def decode_to_f32_16k(data: bytes) -> np.ndarray:
    """Decode a WAV (or any libsndfile-readable) blob to mono float32 @ 16 kHz."""
    audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    if audio.ndim > 1:  # mixdown to mono
        audio = audio.mean(axis=1)
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    if sr != TARGET_SR and audio.size:
        g = gcd(int(sr), TARGET_SR)
        audio = resample_poly(audio, TARGET_SR // g, int(sr) // g).astype(np.float32)
    return audio
