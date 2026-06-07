"""Typed events flowing on ``text_output_queue``.

These are internal pipeline events produced by VAD, TranscriptionNotifier, and
LMOutputProcessor, consumed by the realtime ``WebSocketRouter`` send-loop and
``RealtimeService.dispatch_pipeline_event``.  They replace the raw ``dict``
literals that were previously put on the queue.
"""

from __future__ import annotations

from typing import Literal, Optional

from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall
from pydantic import BaseModel, Field


class PipelineEvent(BaseModel):
    """Base for all text_output_queue events.

    The ``type`` field mirrors the former dict ``"type"`` key and acts as a
    Pydantic discriminator.
    """

    type: str


# ── VAD events ────────────────────────────────────────────────────────


class SpeechStartedEvent(PipelineEvent):
    type: Literal["speech_started"] = "speech_started"
    audio_start_ms: int = 0


class SpeechStoppedEvent(PipelineEvent):
    type: Literal["speech_stopped"] = "speech_stopped"
    duration_s: float = 0.0
    audio_end_ms: int = 0


# ── Transcription events (TranscriptionNotifier) ─────────────────────


class PartialTranscriptionEvent(PipelineEvent):
    type: Literal["partial_transcription"] = "partial_transcription"
    delta: str


class TranscriptionCompletedEvent(PipelineEvent):
    type: Literal["transcription_completed"] = "transcription_completed"
    transcript: str
    language_code: Optional[str] = None
    # SpeakerLabel.model_dump() or None (speaker-id, Phase 0+). None when off,
    # so the client-facing transcript event is unchanged until Phase 3 wiring.
    speaker: Optional[dict] = None
    # Turn id a later TranscriptionCorrectedEvent can reference (Phase 4). None
    # unless diarization is on, so the event is unchanged when the feature is off.
    item_id: Optional[str] = None


class TranscriptionCorrectedEvent(PipelineEvent):
    """Async diarization correction for a prior transcript (Phase 4, Tier 2).

    Replaces the speaker label(s) for ``item_id``'s spans after the turn was
    already emitted. Idempotent and ``revision``-versioned (a consumer ignores a
    revision ≤ the last it applied for this ``item_id``). Dropped-safe: a consumer
    with no support simply ignores it and the Tier-1 transcript stands. Only
    produced when SPEAKER_DIARIZE_ENABLED is on — never emitted otherwise.
    """

    type: Literal["transcription_corrected"] = "transcription_corrected"
    item_id: str
    revision: int = 1
    # SpeakerSpan.model_dump() list — time-ordered spans with per-span labels.
    segments: list[dict] = Field(default_factory=list)
    # Optional re-rendered transcript with span labels applied; None = consumer
    # re-renders from segments itself.
    transcript: Optional[str] = None


# ── LLM output events (LMOutputProcessor) ────────────────────────────


class AssistantTextEvent(PipelineEvent):
    type: Literal["assistant_text"] = "assistant_text"
    text: str
    tools: list[ResponseFunctionToolCall] = Field(default_factory=list)


class TokenUsageEvent(PipelineEvent):
    type: Literal["token_usage"] = "token_usage"
    input_tokens: int = 0
    output_tokens: int = 0
