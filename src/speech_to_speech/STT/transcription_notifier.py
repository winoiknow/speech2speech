from __future__ import annotations

import logging
from queue import Queue
from threading import Event
from typing import Iterator, Union

from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.LLM.chat import make_user_message
from speech_to_speech.pipeline.events import PartialTranscriptionEvent, TranscriptionCompletedEvent
from speech_to_speech.pipeline.handler_types import LLMIn, STTOut
from speech_to_speech.pipeline.messages import GenerateResponseRequest, PartialTranscription, Transcription
from speech_to_speech.pipeline.queue_types import TextEventItem

logger = logging.getLogger(__name__)


class TranscriptionNotifier(BaseHandler[STTOut, Union[STTOut, LLMIn]]):
    """Sits between STT and LLM.

    For **realtime mode** (no ``runtime_config``): emits transcription events
    on ``text_output_queue`` for protocol translation but yields nothing -- the
    ``RealtimeService`` builds ``GenerateResponseRequest`` directly.

    For **legacy mode** (``runtime_config`` provided): appends the user
    message to ``runtime_config.chat`` and yields a
    ``GenerateResponseRequest`` so the LLM handler receives a uniform input
    type regardless of pipeline mode.
    """

    def setup(
        self,
        text_output_queue: Queue[TextEventItem] | None = None,
        runtime_config: RuntimeConfig | None = None,
        should_listen: Event | None = None,
        label_format: str = "",
    ) -> None:
        self.text_output_queue = text_output_queue
        self.runtime_config = runtime_config
        self.should_listen = should_listen
        # Inline speaker tag applied only on a confident `known` match. "" = off.
        self.label_format = label_format

    def process(self, transcription: STTOut) -> Iterator[Union[STTOut, LLMIn]]:
        if isinstance(transcription, PartialTranscription):
            if self.text_output_queue and transcription.text:
                self.text_output_queue.put(PartialTranscriptionEvent(delta=str(transcription.text)))
                logger.debug("Partial transcription: %s", str(transcription.text)[:80])
            return

        speaker = None
        if isinstance(transcription, Transcription):
            text = transcription.text
            language_code = transcription.language_code
            speaker = transcription.speaker
        else:
            text = transcription
            language_code = None

        raw = str(text)
        # Inline `[speaker]` tag — only on a confident `known` match, only on
        # non-empty text. unknown/ambiguous get no prefix (never guess). When
        # speaker-id is off, speaker is None and label_format "" → no-op.
        prefix = ""
        if raw and speaker is not None and speaker.decision == "known" and self.label_format:
            prefix = self.label_format.format(name=(speaker.name or speaker.speaker_id or ""),
                                              speaker_id=(speaker.speaker_id or ""))
        transcript = prefix + raw

        # Always close the client-visible transcription item. Empty final STT
        # results should not trigger the LLM, but clients may already have
        # received partial deltas and still need a completed event. The structured
        # label travels even when no inline prefix is applied.
        if self.text_output_queue is not None:
            self.text_output_queue.put(TranscriptionCompletedEvent(
                transcript=transcript,
                language_code=language_code,
                speaker=(speaker.model_dump() if speaker is not None else None),
            ))

        if not raw:
            logger.debug("Transcription completed with empty transcript")
            if self.should_listen is not None:
                self.should_listen.set()
                logger.debug("Empty transcription completed; listening re-enabled")
            return

        if language_code:
            logger.info("Transcription completed (language=%s): %s", language_code, transcript)
        else:
            logger.info("Transcription completed: %s", transcript)

        if self.runtime_config is not None:
            self.runtime_config.chat.add_item(make_user_message(transcript))
            yield GenerateResponseRequest(
                runtime_config=self.runtime_config,
                language_code=language_code,
            )
