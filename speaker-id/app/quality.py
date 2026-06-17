# Copyright 2026 winoiknow (Eric Alborn, Anteon Group)
# Licensed under the Apache License, Version 2.0 (the "License").

"""Enrollment quality gates (server-enforced).

Sample quality is the single biggest driver of recognition accuracy — bad clips
poison the bank — so the server (not just the UI) rejects unusable samples and
embeds the *trimmed* speech. Lightweight, torch-free: energy-based endpoint trim
+ duration / clipping / loudness checks (no silero needed here).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SR = 16000


@dataclass
class QualityResult:
    ok: bool
    reason: str
    audio: np.ndarray  # trimmed (speech-only) audio to embed when ok
    duration_s: float
    rms: float
    peak: float


def _energy_trim(audio: np.ndarray, frame: int = 320, ratio: float = 0.1) -> np.ndarray:
    """Trim leading/trailing low-energy regions (frame = 20 ms @ 16 kHz)."""
    n = (len(audio) // frame) * frame
    if n == 0:
        return audio
    frames = audio[:n].reshape(-1, frame)
    energy = np.sqrt((frames ** 2).mean(axis=1))
    if energy.max() <= 0:
        return audio[:0]
    voiced = np.where(energy > max(energy.max() * ratio, 1e-4))[0]
    if voiced.size == 0:
        return audio[:0]
    return audio[voiced[0] * frame : (voiced[-1] + 1) * frame]


def check_sample(
    audio: np.ndarray,
    min_seconds: float = 2.0,
    max_clip_ratio: float = 0.01,
    min_rms: float = 0.005,
) -> QualityResult:
    if audio.size == 0:
        return QualityResult(False, "empty audio", audio, 0.0, 0.0, 0.0)
    peak = float(np.max(np.abs(audio)))
    clip_ratio = float(np.mean(np.abs(audio) >= 0.99))
    if clip_ratio > max_clip_ratio:
        return QualityResult(False, f"clipping ({clip_ratio * 100:.1f}% at peak — record quieter)",
                             audio, len(audio) / SR, 0.0, peak)
    trimmed = _energy_trim(audio)
    dur = len(trimmed) / SR
    if dur < min_seconds:
        return QualityResult(False, f"too short — only {dur:.1f}s of speech (need ≥ {min_seconds:.0f}s)",
                             trimmed, dur, 0.0, peak)
    rms = float(np.sqrt(np.mean(trimmed ** 2)))
    if rms < min_rms:
        return QualityResult(False, f"too quiet / noisy (rms {rms:.4f}) — move closer or reduce background noise",
                             trimmed, dur, rms, peak)
    return QualityResult(True, "ok", trimmed, dur, rms, peak)
