"""Tests for RemoteSpeakerClient identify failure visibility (fail-safe + warn).

A failing/unreachable speaker-id endpoint must never break a turn (identify
returns decision='unknown'), but it must be visible at info level instead of only
debug — otherwise 'identify isn't firing' is undiagnosable from normal logs.
"""

import logging

import httpx
import pytest

from speech_to_speech.speaker_id.remote_speaker_client import RemoteSpeakerClient


class _FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def _client():
    return RemoteSpeakerClient("http://speaker-id:9100", api_key="", timeout=0.1)


def test_transient_failures_below_threshold_are_silent(caplog):
    """A one-off slow/timed-out identify that recovers must NOT warn — it's
    fail-safe (→ unknown) and not an outage."""
    c = _client()
    c._client.post = lambda *a, **k: (_ for _ in ()).throw(httpx.ReadTimeout("timed out"))

    with caplog.at_level(logging.WARNING, logger="speech_to_speech.speaker_id.remote_speaker_client"):
        first = c.identify(b"wav")
        second = c.identify(b"wav")  # still below _FAIL_WARN_AFTER (3)

    assert first.decision == "unknown"
    assert second.decision == "unknown"
    assert [r for r in caplog.records if r.levelno == logging.WARNING] == []
    assert c._identify_failing is False  # not flagged as a sustained outage yet


def test_sustained_failures_warn_once(caplog):
    c = _client()
    c._client.post = lambda *a, **k: (_ for _ in ()).throw(httpx.ConnectError("no route"))

    with caplog.at_level(logging.WARNING, logger="speech_to_speech.speaker_id.remote_speaker_client"):
        for _ in range(5):  # >= _FAIL_WARN_AFTER, all within the rate-limit window
            assert c.identify(b"wav").decision == "unknown"

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1  # warned once it became sustained, then rate-limited
    assert "speaker identify is failing" in warnings[0].message
    assert "consecutive failures" in warnings[0].message
    assert c._identify_failing is True


def test_one_success_resets_the_failure_streak(caplog):
    """A success between failures clears the streak, so it takes another full run
    of consecutive failures (not just one more) to warn."""
    c = _client()
    c._client.post = lambda *a, **k: (_ for _ in ()).throw(httpx.ReadTimeout("slow"))
    c.identify(b"wav")
    c.identify(b"wav")  # streak = 2, still silent

    c._client.post = lambda *a, **k: _FakeResp({"decision": "known", "name": "Eric"})
    c.identify(b"wav")  # success resets streak to 0
    assert c._consecutive_failures == 0

    c._client.post = lambda *a, **k: (_ for _ in ()).throw(httpx.ReadTimeout("slow"))
    with caplog.at_level(logging.WARNING, logger="speech_to_speech.speaker_id.remote_speaker_client"):
        c.identify(b"wav")
        c.identify(b"wav")  # streak only back to 2 → still no warning
    assert [r for r in caplog.records if r.levelno == logging.WARNING] == []


def test_identify_recovers_logs_info(caplog):
    c = _client()
    c._client.post = lambda *a, **k: (_ for _ in ()).throw(httpx.ConnectError("down"))
    for _ in range(3):  # reach the warn threshold so it's flagged failing
        c.identify(b"wav")
    assert c._identify_failing is True

    c._client.post = lambda *a, **k: _FakeResp({"decision": "known", "name": "Eric", "score": 0.9})
    with caplog.at_level(logging.INFO, logger="speech_to_speech.speaker_id.remote_speaker_client"):
        label = c.identify(b"wav")

    assert label.decision == "known"
    assert label.name == "Eric"
    assert c._identify_failing is False
    assert c._consecutive_failures == 0
    assert any("speaker identify recovered" in r.message for r in caplog.records)


@pytest.mark.parametrize("decision_in,expected", [("weird", "unknown"), ("ambiguous", "ambiguous")])
def test_decision_validation(decision_in, expected):
    c = _client()
    c._client.post = lambda *a, **k: _FakeResp({"decision": decision_in})
    assert c.identify(b"wav").decision == expected
