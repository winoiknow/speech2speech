"""Unit tests for the per-service ConcurrencyLimiter (Phase D3 caps)."""

import threading
import time

from speech_to_speech.utils.concurrency import ConcurrencyLimiter


def test_unlimited_is_noop():
    lim = ConcurrencyLimiter("X", 0)
    assert lim._sem is None
    # acquire/release/slot are all no-ops and never block
    lim.acquire()
    lim.release()
    with lim.slot():
        pass


def test_negative_limit_treated_as_unlimited():
    assert ConcurrencyLimiter("X", -3)._sem is None


def test_limit_bounds_concurrency():
    """With limit=2, at most 2 holders are inside a slot at once."""
    lim = ConcurrencyLimiter("X", 2)
    inside = 0
    peak = 0
    lock = threading.Lock()
    release = threading.Event()

    def worker():
        nonlocal inside, peak
        with lim.slot():
            with lock:
                inside += 1
                peak = max(peak, inside)
            release.wait(timeout=2.0)
            with lock:
                inside -= 1

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    # Give the first batch time to occupy the two slots and stabilize.
    time.sleep(0.2)
    with lock:
        assert peak == 2  # never more than the cap, even with 5 contenders
    release.set()
    for t in threads:
        t.join(timeout=2.0)
    assert peak == 2


def test_blocked_acquirer_proceeds_after_release():
    """A 3rd acquirer blocks until a slot frees, then proceeds."""
    lim = ConcurrencyLimiter("X", 1)
    lim.acquire()  # take the only slot
    proceeded = threading.Event()

    def waiter():
        lim.acquire()
        proceeded.set()
        lim.release()

    t = threading.Thread(target=waiter)
    t.start()
    assert not proceeded.wait(timeout=0.3)  # still blocked
    lim.release()  # free the slot
    assert proceeded.wait(timeout=2.0)
    t.join(timeout=2.0)


def test_over_release_does_not_raise():
    lim = ConcurrencyLimiter("X", 1)
    lim.release()  # release without acquire — must be swallowed, not raise
