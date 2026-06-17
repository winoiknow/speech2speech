import logging
import threading
from collections.abc import Sequence
from typing import Any

logger = logging.getLogger(__name__)


class ThreadManager:
    """
    Manages multiple threads used to execute given handler tasks.
    """

    def __init__(self, handlers: Sequence[Any], daemon: bool = False, name_prefix: str = "") -> None:
        self.handlers = handlers
        # Per-session pipelines pass daemon=True so a handler stuck in a blocking
        # call (e.g. an in-flight HTTP read at disconnect) can never wedge process
        # exit; the long-lived startup pipeline keeps daemon=False so its threads
        # are waited for on graceful shutdown.
        self.daemon = daemon
        # Suffix appended to each handler thread's name (e.g. a short session id),
        # so a faulthandler stack dump attributes every thread to its session.
        self.name_prefix = name_prefix
        self.threads: list[threading.Thread] = []

    def start(self) -> None:
        for handler in self.handlers:
            name = type(handler).__name__
            if self.name_prefix:
                name = f"{name}-{self.name_prefix}"
            thread = threading.Thread(target=handler.run, name=name)
            thread.daemon = self.daemon
            self.threads.append(thread)
            thread.start()

    def wait(self) -> None:
        for thread in self.threads:
            thread.join()

    def stop(self) -> None:
        # Signal all handlers to stop
        for handler in self.handlers:
            handler.stop_event.set()

        # Wait for all threads to finish with timeout
        for i, thread in enumerate(self.threads):
            if thread.is_alive():
                thread.join(timeout=5.0)
                if thread.is_alive():
                    logger.warning(f"Thread {i} ({thread.name}) did not terminate within timeout")
