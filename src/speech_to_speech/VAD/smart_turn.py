# Copyright 2026 winoiknow (Eric Alborn, Anteon Group)
# Licensed under the Apache License, Version 2.0 (the "License").
# See the LICENSE file in the repository root for the full license text.

"""Smart Turn v3 semantic end-of-turn detector.

Pipecat Smart Turn v3 (``pipecat-ai/smart-turn-v3``, BSD-2): a Whisper-tiny
encoder + linear head (~8M params) that scores whether a spoken turn is
*semantically complete* from the audio's prosody/content — not just a silence
timer. We run the int8 ONNX CPU model (~12 ms) alongside silero: silero detects
the pause, this decides whether the pause is end-of-turn or just a breath/think.

This replaces the "fixed silence_duration_ms" turn-end with a content-aware one,
so a trailing-off or mid-thought pause no longer cuts the user off.

Fail-safe: any load/inference error → report "complete" so turns never hang
(the pipeline falls back to plain VAD silence timing).

Inference contract mirrors pipecat-ai/smart-turn `inference.py`:
  audio (16 kHz mono float) → last 8 s → WhisperFeatureExtractor(80-mel,
  max_length=128000, do_normalize) → ONNX input ``input_features`` (float32)
  → sigmoid probability; complete iff prob > threshold (0.5).
"""

from __future__ import annotations

import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
MAX_AUDIO_S = 8
MAX_AUDIO_SAMPLES = SAMPLE_RATE * MAX_AUDIO_S  # 128000


class SmartTurnDetector:
    def __init__(self, model_path: str, threshold: float = 0.5, sample_rate: int = SAMPLE_RATE) -> None:
        self.model_path = model_path
        self.threshold = float(threshold)
        self.sample_rate = sample_rate
        self._sess = None
        self._fe = None
        self._ok = False
        self._load()

    def _load(self) -> None:
        try:
            import onnxruntime as ort
            from transformers import WhisperFeatureExtractor

            if not self.model_path or not os.path.exists(self.model_path):
                logger.error(
                    "Smart Turn model not found at %r — turn detection disabled (VAD silence timing only)",
                    self.model_path,
                )
                return
            so = ort.SessionOptions()
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            so.inter_op_num_threads = 1
            # Set intra-op threads EXPLICITLY: onnxruntime otherwise spawns one
            # thread per core and pins each with pthread_setaffinity_np, which
            # fails (EINVAL) under a container cpuset and spams the log. An
            # explicit count disables the affinity pinning. Tiny model run only
            # at pause boundaries, so a small count is ample. Env override:
            # SMART_TURN_NUM_THREADS.
            so.intra_op_num_threads = max(1, int(os.environ.get("SMART_TURN_NUM_THREADS", "2")))
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._sess = ort.InferenceSession(self.model_path, sess_options=so, providers=["CPUExecutionProvider"])
            # Direct constructor (no from_pretrained) → builds mel filters in-process, no network.
            self._fe = WhisperFeatureExtractor(
                feature_size=80, sampling_rate=self.sample_rate, chunk_length=MAX_AUDIO_S
            )
            self._ok = True
            logger.info(
                "SmartTurnDetector loaded (%s, threshold=%.2f)", os.path.basename(self.model_path), self.threshold
            )
        except Exception as e:
            logger.error("Smart Turn load failed (%s) — turn detection disabled (VAD silence timing only)", e)
            self._ok = False

    @property
    def available(self) -> bool:
        return self._ok

    def is_complete(self, audio_float32: np.ndarray) -> tuple[bool, float]:
        """Return (turn_complete, probability) for the utterance-so-far (16 kHz float)."""
        if not self._ok:
            return True, 1.0
        try:
            audio = np.asarray(audio_float32, dtype=np.float32).reshape(-1)
            if audio.shape[0] > MAX_AUDIO_SAMPLES:
                audio = audio[-MAX_AUDIO_SAMPLES:]  # model sees the last 8 s
            feats = self._fe(
                audio,
                sampling_rate=self.sample_rate,
                padding="max_length",
                max_length=MAX_AUDIO_SAMPLES,
                truncation=True,
                do_normalize=True,
                return_tensors="np",
            )
            input_features = np.asarray(feats.input_features, dtype=np.float32)
            out = self._sess.run(None, {"input_features": input_features})
            prob = float(np.asarray(out[0]).reshape(-1)[0])
            return prob > self.threshold, prob
        except Exception as e:
            logger.warning("Smart Turn inference failed (%s) — treating turn as complete", e)
            return True, 1.0
