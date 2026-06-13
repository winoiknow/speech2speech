from __future__ import annotations

import logging
from queue import Queue
from threading import Event as ThreadingEvent
from typing import TYPE_CHECKING

from openai.types.realtime import RealtimeErrorEvent

from speech_to_speech.utils.utils import _generate_id

if TYPE_CHECKING:
    from speech_to_speech.api.openai_realtime.service import ConnState, RealtimeService

logger = logging.getLogger(__name__)


class RealtimeBaseHandler:
    """Shared base for domain handlers.

    Provides conn_id-keyed accessors for per-connection state, config,
    and queues.  Each connection owns its own ``RuntimeConfig`` via
    ``ConnState``.
    """

    def __init__(self, service: RealtimeService) -> None:
        self._service = service

    # ── conn_id-keyed accessors ──

    def _queue(self, conn_id: str) -> Queue | None:
        return self._service._text_prompt_queue(conn_id)

    def _should_listen(self, conn_id: str) -> ThreadingEvent | None:
        return self._service._should_listen_for(conn_id)

    # ── shared helpers ────────────────────────────────

    def _state(self, conn_id: str) -> ConnState:
        return self._service._state(conn_id)

    @staticmethod
    def _next_event_id() -> str:
        return _generate_id("event")

    def make_error(self, message: str, _type: str) -> RealtimeErrorEvent:
        return self._service.make_error(message, _type)
