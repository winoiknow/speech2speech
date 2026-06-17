"""WebSocket keepalive knob: S2S_WS_PING_INTERVAL/TIMEOUT → uvicorn args.

A positive value passes through; 0 or negative maps to None, which disables the
corresponding uvicorn keepalive behavior (no periodic server pings).
"""

import pytest

from speech_to_speech.api.openai_realtime.server import _ping_or_none


@pytest.mark.parametrize(
    "seconds,expected",
    [
        (20.0, 20.0),
        (60.0, 60.0),
        (0.0, None),  # 0 disables → None
        (-1.0, None),  # negative also disables
    ],
)
def test_ping_or_none(seconds, expected):
    assert _ping_or_none(seconds) == expected
