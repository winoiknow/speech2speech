from __future__ import annotations

import logging
from typing import Optional

from openai.types.realtime import (
    RealtimeErrorEvent,
    SessionCreatedEvent,
    SessionUpdatedEvent,
    SessionUpdateEvent,
)
from openai.types.realtime.realtime_transcription_session_create_request import (
    RealtimeTranscriptionSessionCreateRequest,
)

from speech_to_speech.api.openai_realtime.handlers.base import RealtimeBaseHandler

logger = logging.getLogger(__name__)


class SessionHandler(RealtimeBaseHandler):
    """Owns session lifecycle: config updates and session.created events."""

    def handle_session_update(
        self, conn_id: str, event: SessionUpdateEvent
    ) -> RealtimeErrorEvent | SessionUpdatedEvent:
        """Apply session config changes and acknowledge with ``session.updated``.

        Only ``RealtimeSessionCreateRequest`` sessions are accepted;
        ``RealtimeTranscriptionSessionCreateRequest`` sessions not yet supported.
        Incoming fields are deep-merged into the existing session so that
        partial updates preserve previously-set values.

        The ``session.updated`` ACK is required: clients (e.g. AVA) block on it
        to learn the negotiated audio formats. Without it they time out and guess
        the output codec, mis-decoding our PCM as G.711 → silence/noise.
        """
        s = event.session
        if s is None:
            return self.build_session_updated(conn_id)

        if isinstance(s, RealtimeTranscriptionSessionCreateRequest):
            return self.make_error(
                message="Only 'realtime' session type is supported; transcription sessions are not.",
                _type="invalid_session_type",
            )

        model = getattr(s, "model", None)
        if model is not None:
            logger.info(f"Session model set to: {model}")

        cfg = self._state(conn_id).runtime_config
        current = cfg.session
        if current is None:
            cfg.session = s
        else:
            cfg.apply_session_update(s)
        out = cfg.session.audio.output if (cfg.session.audio and cfg.session.audio.output) else None
        fmt = getattr(out, "format", None)
        logger.info(
            "Session configuration updated — ACKing session.updated (output format_type=%s rate=%s)",
            getattr(fmt, "type", None),
            getattr(fmt, "rate", None),
        )
        return self.build_session_updated(conn_id)

    def build_session_created(self, conn_id: str) -> SessionCreatedEvent:
        """Build a SessionCreatedEvent populated with the current config."""
        cfg = self._state(conn_id).runtime_config
        session = cfg.session
        return SessionCreatedEvent(
            type="session.created",
            event_id=self._next_event_id(),
            session=session,
        )

    def build_session_updated(self, conn_id: str) -> SessionUpdatedEvent:
        """Build a SessionUpdatedEvent echoing the current (merged) config."""
        cfg = self._state(conn_id).runtime_config
        return SessionUpdatedEvent(
            type="session.updated",
            event_id=self._next_event_id(),
            session=cfg.session,
        )
