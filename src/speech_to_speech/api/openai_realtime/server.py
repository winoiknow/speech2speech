import logging
import threading
from threading import Event
from typing import TYPE_CHECKING

import uvicorn

from speech_to_speech.api.openai_realtime.service import RealtimeService
from speech_to_speech.api.openai_realtime.websocket_router import create_app

if TYPE_CHECKING:
    from speech_to_speech.pipeline.session_pipeline import HandlerFactory

logger = logging.getLogger(__name__)


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
    ) -> None:
        self.stop_event = stop_event
        self.session_factory = session_factory
        self.host = host
        self.port = port
        self.chat_size = chat_size
        self.server_api_key = server_api_key
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
        )

        if self.server_api_key:
            logger.info("Server API key authentication enabled")

        logger.info(f"OpenAI Realtime API server starting on ws://{self.host}:{self.port}/v1/realtime")

        config = uvicorn.Config(
            app,
            host=self.host,
            port=self.port,
            log_level="info",
        )
        server = uvicorn.Server(config)

        server.install_signal_handlers = lambda: None  # type: ignore[attr-defined]

        def _watch_stop() -> None:
            self.stop_event.wait()
            server.should_exit = True

        watcher = threading.Thread(target=_watch_stop, daemon=True)
        watcher.start()

        server.run()
