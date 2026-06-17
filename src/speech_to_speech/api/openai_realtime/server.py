import logging
import os
import threading
from threading import Event
from typing import TYPE_CHECKING

import uvicorn

from speech_to_speech.api.openai_realtime.service import RealtimeService
from speech_to_speech.api.openai_realtime.websocket_router import create_app

if TYPE_CHECKING:
    from speech_to_speech.pipeline.session_pipeline import HandlerFactory

logger = logging.getLogger(__name__)

# WebSocket keepalive. uvicorn's defaults (20s/20s) ping idle clients and close
# (1011) on no pong — fine for a browser, but for a warm-connection client that
# tends its own keepalive these defaults force an aggressive client refresh (each
# reconnect can re-prime the LLM and re-burn the system prompt). Defaults here are
# generous: still ping for dead-connection detection, but give a busy client loop
# (a long tool turn) or a relaxed refresh plenty of grace. Set
# S2S_WS_PING_INTERVAL=0 to disable server-side pings entirely (client/TCP own
# liveness — note: with the idle reaper also off, a vanished client lingers until
# TCP timeout).
_WS_PING_INTERVAL = float(os.getenv("S2S_WS_PING_INTERVAL", "20"))
_WS_PING_TIMEOUT = float(os.getenv("S2S_WS_PING_TIMEOUT", "60"))


def _ping_or_none(seconds: float) -> float | None:
    """Map a keepalive seconds value to uvicorn's arg: a positive number, or None
    to disable (0/negative)."""
    return seconds if seconds > 0 else None


class RealtimeServer:
    """Pipeline handler for the OpenAI Realtime API mode.

    Owns uvicorn + the FastAPI app and the shared :class:`RealtimeService`. It
    holds **no** pipeline queues: each WebSocket connection builds its own
    :class:`SessionPipeline` via ``session_factory`` at connect time (Phase B),
    so the only state here is server + service config.
    """

    def __init__(
        self,
        stop_event: Event,
        session_factory: "HandlerFactory",
        host: str = "0.0.0.0",
        port: int = 8765,
        chat_size: int = 10,
        server_api_key: str | None = None,
        speaker_client: object | None = None,
        speaker_diarize_enabled: bool = False,
        max_sessions: int = 1,
    ) -> None:
        self.stop_event = stop_event
        self.session_factory = session_factory
        self.host = host
        self.port = port
        self.chat_size = chat_size
        self.server_api_key = server_api_key
        self.max_sessions = max_sessions
        # Phase 4: the same speaker-id client + flag the service uses to run the
        # off-hot-path diarize and emit corrections. None/False → no-op.
        self.speaker_client = speaker_client
        self.speaker_diarize_enabled = speaker_diarize_enabled

    def run(self) -> None:
        """Start the FastAPI/uvicorn server (called from a ThreadManager thread)."""
        service = RealtimeService(
            chat_size=self.chat_size,
            speaker_client=self.speaker_client,
            diarize_enabled=self.speaker_diarize_enabled,
        )
        app = create_app(
            service=service,
            session_factory=self.session_factory,
            stop_event=self.stop_event,
            server_api_key=self.server_api_key,
            max_sessions=self.max_sessions,
        )

        if self.server_api_key:
            logger.info("Server API key authentication enabled")

        logger.info(f"OpenAI Realtime API server starting on ws://{self.host}:{self.port}/v1/realtime")

        # ws_ping_interval=None disables periodic server pings; a 0/negative env
        # value maps to that. ws_ping_timeout only matters when pinging.
        ping_interval = _ping_or_none(_WS_PING_INTERVAL)
        ping_timeout = _ping_or_none(_WS_PING_TIMEOUT)
        if ping_interval is None:
            logger.info("WebSocket server pings DISABLED (S2S_WS_PING_INTERVAL=0); client/TCP own liveness")
        else:
            logger.info(
                "WebSocket keepalive: ping every %.0fs, %ss pong timeout",
                ping_interval,
                int(ping_timeout) if ping_timeout else "no",
            )
        config = uvicorn.Config(
            app,
            host=self.host,
            port=self.port,
            log_level="info",
            ws_ping_interval=ping_interval,
            ws_ping_timeout=ping_timeout,
        )
        server = uvicorn.Server(config)

        server.install_signal_handlers = lambda: None  # type: ignore[attr-defined]

        def _watch_stop() -> None:
            self.stop_event.wait()
            server.should_exit = True

        watcher = threading.Thread(target=_watch_stop, daemon=True)
        watcher.start()

        server.run()
