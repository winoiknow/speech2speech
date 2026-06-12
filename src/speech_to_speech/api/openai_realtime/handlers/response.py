from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from openai.types.realtime import (
    RealtimeResponse,
    ResponseAudioDoneEvent,
    ResponseAudioTranscriptDoneEvent,
    ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent,
    ResponseCreatedEvent,
    ResponseCreateEvent,
    ResponseDoneEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
)
from openai.types.realtime.realtime_conversation_item_assistant_message import (
    Content as AssistantContent,
)
from openai.types.realtime.realtime_conversation_item_assistant_message import (
    RealtimeConversationItemAssistantMessage,
)
from openai.types.realtime.realtime_response import Audio, AudioOutput
from openai.types.realtime.realtime_response_status import RealtimeResponseStatus
from openai.types.realtime.realtime_response_usage import RealtimeResponseUsage
from openai.types.realtime.response_content_part_added_event import Part as AddedPart
from openai.types.realtime.response_content_part_done_event import Part as DonePart

from speech_to_speech.api.openai_realtime.handlers.base import RealtimeBaseHandler
from speech_to_speech.LLM.chat import ChatItemError
from speech_to_speech.pipeline.events import AssistantTextEvent
from speech_to_speech.pipeline.messages import GenerateResponseRequest
from speech_to_speech.utils.utils import _generate_id

if TYPE_CHECKING:
    from speech_to_speech.api.openai_realtime.service import ServerEvent, _ResponseStatus, _StatusReason

logger = logging.getLogger(__name__)


class ResponseHandler(RealtimeBaseHandler):
    """Owns the response lifecycle: create, cancel, finish, and ID management."""

    # ── ID / state helpers ────────────────────────

    def _ensure_response(self, conn_id: str) -> tuple[str, str]:
        """Ensure a response and output item exist, creating them if needed."""
        st = self._state(conn_id)
        if st.current_response_id is None:
            st.current_response_id = _generate_id("resp")
            self._start_item(conn_id)
            st.in_response = True
        return st.current_response_id, self._current_item_id(conn_id)

    def _end_response(self, conn_id: str, status: _ResponseStatus = "completed") -> None:
        st = self._state(conn_id)
        if status == "cancelled":
            st.response_usage.responses_cancelled += 1
        else:
            st.response_usage.responses_completed += 1
        self._service.total_usage += st.response_usage
        logger.info(
            "Response done (status=%s) — this response: input_tokens=%d, output_tokens=%d, audio=%.2fs"
            " | cumulative: input_tokens=%d, output_tokens=%d, audio=%.2fs",
            status,
            st.response_usage.input_tokens,
            st.response_usage.output_tokens,
            st.response_usage.audio_duration_s,
            self._service.total_usage.input_tokens,
            self._service.total_usage.output_tokens,
            self._service.total_usage.audio_duration_s,
        )
        st.response_usage.reset()
        st.current_response_id = None
        st.current_item_id = None
        st.content_index = 0
        st.in_response = False
        st.current_response_params = None
        st.response_created_sent = False
        st.output_item_added = False
        st.assistant_transcript = ""

    def _start_item(self, conn_id: str) -> str:
        """Generate a new item ID, reset content index, and store it."""
        st = self._state(conn_id)
        item_id = _generate_id("item")
        st.current_item_id = item_id
        st.content_index = 0
        st.input_audio_duration_s = 0.0
        st.output_item_added = False
        st.assistant_transcript = ""
        return item_id

    def _current_item_id(self, conn_id: str) -> str:
        return self._state(conn_id).current_item_id or self._start_item(conn_id)

    def _next_content_index(self, conn_id: str) -> int:
        """Return the current content index and advance it."""
        st = self._state(conn_id)
        idx = st.content_index
        st.content_index += 1
        return idx

    def _build_response(
        self,
        conn_id: str,
        status: _ResponseStatus,
        reason: _StatusReason | None = None,
    ) -> RealtimeResponse:
        """Build a fully-populated RealtimeResponse from the current connection state."""
        st = self._state(conn_id)
        status_details = None
        if reason or status in ("completed", "cancelled", "incomplete", "failed"):
            status_details = RealtimeResponseStatus(type=status, reason=reason)  # type: ignore[arg-type]

        rp = st.current_response_params
        metadata = rp.metadata if rp and rp.metadata else None

        voice: Optional[str] = None
        if rp and rp.audio and rp.audio.output and rp.audio.output.voice:
            voice = str(rp.audio.output.voice)
        if not voice:
            audio_cfg = st.runtime_config.session.audio
            audio_output = audio_cfg.output if audio_cfg is not None else None
            voice = str(audio_output.voice) if audio_output is not None and audio_output.voice else None

        return RealtimeResponse(
            id=st.current_response_id,
            object="realtime.response",
            status=status,
            status_details=status_details,
            audio=Audio(output=AudioOutput(voice=str(voice) if voice else None)),  # type: ignore[arg-type]
            conversation_id=st.conversation_id,
            metadata=metadata,
            usage=RealtimeResponseUsage(
                input_tokens=st.response_usage.input_tokens,
                output_tokens=st.response_usage.output_tokens,
                total_tokens=st.response_usage.input_tokens + st.response_usage.output_tokens,
            ),
        )

    def begin_output_item_events(self, conn_id: str) -> list[ServerEvent]:
        """Emit the response/item/content-part *begin* events, each exactly once
        per response, before any audio delta or transcript is sent.

        Spec order is ``response.created`` -> ``response.output_item.added`` ->
        ``response.content_part.added`` -> deltas.  Strict clients build their
        item/content-part state from these; without them they have nowhere to
        attach ``response.output_audio.delta`` and silently drop the audio.

        Idempotent: callers (the audio encoder and the transcript handler) may
        both invoke it; the per-connection flags ensure single emission.
        """
        st = self._state(conn_id)
        events: list[ServerEvent] = []
        resp_id, item_id = self._ensure_response(conn_id)

        if not st.response_created_sent:
            st.response_created_sent = True
            events.append(
                ResponseCreatedEvent(
                    type="response.created",
                    event_id=self._next_event_id(),
                    response=self._build_response(conn_id, "in_progress"),
                )
            )

        if not st.output_item_added:
            st.output_item_added = True
            events.append(
                ResponseOutputItemAddedEvent(
                    type="response.output_item.added",
                    event_id=self._next_event_id(),
                    output_index=0,
                    response_id=resp_id,
                    item=RealtimeConversationItemAssistantMessage(
                        id=item_id,
                        type="message",
                        role="assistant",
                        status="in_progress",
                        content=[],
                    ),
                )
            )
            events.append(
                ResponseContentPartAddedEvent(
                    type="response.content_part.added",
                    event_id=self._next_event_id(),
                    content_index=0,
                    item_id=item_id,
                    output_index=0,
                    response_id=resp_id,
                    part=AddedPart(type="audio"),
                )
            )
        if events:
            rp = st.current_response_params
            fmt = rp.audio.output.format if (rp and rp.audio and rp.audio.output) else None
            if fmt is None:
                ao = st.runtime_config.session.audio
                fmt = ao.output.format if (ao and ao.output) else None
            logger.info(
                "Realtime lifecycle BEGIN emitted: %s | negotiated output audio: format_type=%s "
                "declared_rate=%s — s2s sends int16 PCM resampled to the client rate (16000 Hz if "
                "unset). A non-'audio/pcm' format type (e.g. audio/pcmu, audio/pcma) means the "
                "client will decode these PCM bytes wrong → silence/noise.",
                [e.type for e in events],
                getattr(fmt, "type", None) if fmt else None,
                getattr(fmt, "rate", None) if fmt else None,
            )
        return events

    def begin_turn_response(self, conn_id: str) -> list[ServerEvent]:
        """Open the response lifecycle the moment the server starts working on a
        turn (server-VAD path), instead of waiting for the first audio chunk.

        Everything between end-of-speech and first audio — STT hand-off, the
        agent's LLM + tool loop, TTS — is otherwise silent on the wire, so a
        slow turn is indistinguishable from a dead connection. Emitting
        ``response.created`` (in_progress) here gives any client an immediate
        signal to show "working" feedback and arm/refresh a turn watchdog.

        ``begin_output_item_events`` stays idempotent via
        ``response_created_sent``; ``output_item.added`` / ``content_part.added``
        still fire with the first audio or transcript.

        Setting ``in_response`` here also arms barge-in for the thinking gap:
        speech detected while the LLM is in flight cancels this response
        instead of silently stacking a second generation behind it.
        """
        st = self._state(conn_id)
        if st.in_response:
            return []
        st.in_response = True
        st.current_response_params = None
        st.current_response_id = _generate_id("resp")
        self._start_item(conn_id)
        st.response_created_sent = True
        logger.debug("Turn response opened early (response.created at turn start)")
        return [
            ResponseCreatedEvent(
                type="response.created",
                event_id=self._next_event_id(),
                response=self._build_response(conn_id, "in_progress"),
            )
        ]

    # ── Public handlers ───────────────────────────

    def handle_response_create(self, conn_id: str, event: ResponseCreateEvent) -> ServerEvent | None:
        """Trigger a response.

        Returns a ``ResponseCreatedEvent`` on success, a ``RealtimeErrorEvent``
        on failure, or ``None`` if there is no text_prompt_queue.
        """
        st = self._state(conn_id)
        if event.response:
            if event.response.tool_choice and not isinstance(event.response.tool_choice, str):
                return self.make_error(
                    message="Only string tool_choice values are supported for now (auto, required, none).",
                    _type="tool_choice_not_supported",
                )
        if st.in_response:
            return self.make_error(
                message="Cannot create response while another response is in progress.",
                _type="conversation_already_has_active_response",
            )

        if event.response and event.response.input:
            for input_item in event.response.input:
                try:
                    self._service.conversation._append_item(conn_id, input_item)
                except ChatItemError as exc:
                    return self.make_error(message=str(exc), _type="invalid_input_item")

        st.in_response = True

        st.current_response_params = event.response
        st.current_response_id = _generate_id("resp")
        self._start_item(conn_id)

        cfg = st.runtime_config
        queue = self._queue(conn_id)
        if queue:
            queue.put(
                GenerateResponseRequest(
                    runtime_config=cfg,
                    response=event.response,
                )
            )
        st.response_created_sent = True
        logger.debug("response.create received, LLM generation triggered")
        return ResponseCreatedEvent(
            type="response.created",
            event_id=self._next_event_id(),
            response=self._build_response(conn_id, "in_progress"),
        )

    def handle_response_cancel(self, conn_id: str) -> list[ServerEvent]:
        """Cancel the in-progress response and re-enable listening."""
        events = self.finish_audio_response(conn_id, status="cancelled", reason="client_cancelled")
        should_listen = self._should_listen(conn_id)
        if should_listen:
            should_listen.set()
        logger.info("Response cancelled, listening re-enabled")
        return events

    def finish_audio_response(
        self,
        conn_id: str,
        status: _ResponseStatus = "completed",
        reason: _StatusReason | None = None,
    ) -> list[ServerEvent]:
        """Close the current response (audio done + response done)."""
        st = self._state(conn_id)
        events: list[ServerEvent] = []
        if st.in_response:
            resp_id, item_id = self._ensure_response(conn_id)
            events.append(
                ResponseAudioDoneEvent(
                    type="response.output_audio.done",
                    event_id=self._next_event_id(),
                    content_index=0,
                    item_id=item_id,
                    output_index=0,
                    response_id=resp_id,
                )
            )
            # Close the item/content-part lifecycle opened in
            # ``begin_output_item_events`` so strict clients finalize the item.
            if st.output_item_added:
                transcript = st.assistant_transcript or None
                item_status = "completed" if status == "completed" else "incomplete"
                events.append(
                    ResponseContentPartDoneEvent(
                        type="response.content_part.done",
                        event_id=self._next_event_id(),
                        content_index=0,
                        item_id=item_id,
                        output_index=0,
                        response_id=resp_id,
                        part=DonePart(type="audio", transcript=transcript),
                    )
                )
                events.append(
                    ResponseOutputItemDoneEvent(
                        type="response.output_item.done",
                        event_id=self._next_event_id(),
                        output_index=0,
                        response_id=resp_id,
                        item=RealtimeConversationItemAssistantMessage(
                            id=item_id,
                            type="message",
                            role="assistant",
                            status=item_status,
                            content=[AssistantContent(type="output_audio", transcript=transcript)],
                        ),
                    )
                )
                logger.info("Realtime lifecycle CLOSE emitted: content_part.done + output_item.done")
            events.append(
                ResponseDoneEvent(
                    type="response.done",
                    event_id=self._next_event_id(),
                    response=self._build_response(conn_id, status, reason),
                )
            )
            self._end_response(conn_id, status)
        return events

    # ── Pipeline event handlers ───────────────────

    def on_assistant_text(self, conn_id: str, event: AssistantTextEvent) -> list[ServerEvent]:
        """Handle assistant_text: emit transcript and/or tool-call events."""
        st = self._state(conn_id)
        events: list[ServerEvent] = []
        resp_id, item_id = self._ensure_response(conn_id)
        st.last_item_id = item_id
        output_idx = 0
        if event.text:
            # Open the item/content-part lifecycle before the transcript so the
            # transcript (and the audio deltas that follow) attach to a part the
            # client has already seen added. Idempotent with the audio encoder.
            events.extend(self.begin_output_item_events(conn_id))
            st.assistant_transcript += event.text
            events.append(
                ResponseAudioTranscriptDoneEvent(
                    type="response.output_audio_transcript.done",
                    event_id=self._next_event_id(),
                    content_index=0,
                    item_id=item_id,
                    output_index=output_idx,
                    response_id=resp_id,
                    transcript=event.text,
                )
            )
            output_idx += 1
        if event.tools:
            st.response_usage.tool_calls += len(event.tools)
            for tool in event.tools:
                events.append(
                    ResponseFunctionCallArgumentsDoneEvent(
                        type="response.function_call_arguments.done",
                        event_id=self._next_event_id(),
                        call_id=tool.call_id,
                        name=tool.name,
                        arguments=tool.arguments,
                        item_id=item_id,
                        output_index=output_idx,
                        response_id=resp_id,
                    )
                )
                output_idx += 1
        return events
