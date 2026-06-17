"""EchoCanceller.close(): drop the unsendable aec3 Aec3 on the calling thread.

The aec3 backend's Aec3 is PyO3-``unsendable`` — it must be dropped on the thread
that created it (the event loop). Session teardown runs in a worker thread, so the
connection handler calls close() on the loop first; otherwise the worker GCs the
Aec3 on the wrong thread → "unsendable, but is being dropped on another thread".
"""

from speech_to_speech.audio.echo_canceller import EchoCanceller


class _FakeAec3:
    """Stand-in for the unsendable native object; records when it's dropped."""

    dropped = 0

    def __del__(self):
        type(self).dropped += 1


def test_close_drops_aec3_reference():
    ec = EchoCanceller(sample_rate=16000, enabled=False, backend="aec3")
    ec._aec3 = _FakeAec3()
    _FakeAec3.dropped = 0
    ec.close()
    assert ec._aec3 is None
    assert _FakeAec3.dropped == 1  # freed synchronously on this (calling) thread


def test_close_is_idempotent_and_safe_without_aec3():
    ec = EchoCanceller(sample_rate=16000, enabled=False, backend="aec3")
    # No Aec3 was ever created (disabled). close() must not raise, repeatedly.
    ec.close()
    ec.close()
    assert ec._aec3 is None


def test_close_safe_on_speex_backend():
    ec = EchoCanceller(sample_rate=16000, enabled=False, backend="speex")
    ec.close()  # speex has no unsendable object; must be a no-op, not an error
    assert ec._aec3 is None
