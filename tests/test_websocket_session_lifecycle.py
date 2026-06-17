from queue import Queue
from threading import Event, Thread

from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.pipeline.control import SESSION_END, is_control_message
from speech_to_speech.pipeline.messages import PIPELINE_END


class EchoHandler(BaseHandler):
    def setup(self):
        self.processed = []
        self.session_end_calls = 0

    def process(self, item):
        self.processed.append(item)
        yield item.upper()

    def on_session_end(self):
        self.session_end_calls += 1


def test_base_handler_session_end_resets_without_stopping():
    stop_event = Event()
    queue_in = Queue()
    queue_out = Queue()
    handler = EchoHandler(stop_event, queue_in=queue_in, queue_out=queue_out)

    thread = Thread(target=handler.run)
    thread.start()

    queue_in.put(SESSION_END)
    queue_in.put("hello")
    queue_in.put(PIPELINE_END)

    thread.join(timeout=2)
    assert not thread.is_alive()

    outputs = [queue_out.get(timeout=1) for _ in range(3)]
    assert is_control_message(outputs[0], SESSION_END.kind)
    assert outputs[1] == "HELLO"
    assert outputs[2] == PIPELINE_END
    assert handler.processed == ["hello"]
    assert handler.session_end_calls == 1
