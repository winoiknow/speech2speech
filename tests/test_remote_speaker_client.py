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


def test_identify_failure_is_failsafe_and_warns_once(caplog):
    c = _client()
    c._client.post = lambda *a, **k: (_ for _ in ()).throw(httpx.ConnectError("no route"))

    with caplog.at_level(logging.WARNING, logger="speech_to_speech.speaker_id.remote_speaker_client"):
        first = c.identify(b"wav")
        second = c.identify(b"wav")  # within the rate-limit window

    # Fail-safe: never raises, always 'unknown'.
    assert first.decision == "unknown"
    assert second.decision == "unknown"
    # Exactly one warning despite two failures (rate-limited).
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "speaker identify is failing" in warnings[0].message
    assert c._identify_failing is True


def test_identify_recovers_logs_info(caplog):
    c = _client()
    c._client.post = lambda *a, **k: (_ for _ in ()).throw(httpx.ConnectError("down"))
    c.identify(b"wav")  # mark failing
    assert c._identify_failing is True

    c._client.post = lambda *a, **k: _FakeResp({"decision": "known", "name": "Eric", "score": 0.9})
    with caplog.at_level(logging.INFO, logger="speech_to_speech.speaker_id.remote_speaker_client"):
        label = c.identify(b"wav")

    assert label.decision == "known"
    assert label.name == "Eric"
    assert c._identify_failing is False
    assert any("speaker identify recovered" in r.message for r in caplog.records)


def test_repeated_failures_rate_limited(caplog):
    c = _client()
    c._client.post = lambda *a, **k: (_ for _ in ()).throw(httpx.ConnectError("down"))
    with caplog.at_level(logging.WARNING, logger="speech_to_speech.speaker_id.remote_speaker_client"):
        for _ in range(5):
            c.identify(b"wav")
    assert sum(1 for r in caplog.records if r.levelno == logging.WARNING) == 1


@pytest.mark.parametrize("decision_in,expected", [("weird", "unknown"), ("ambiguous", "ambiguous")])
def test_decision_validation(decision_in, expected):
    c = _client()
    c._client.post = lambda *a, **k: _FakeResp({"decision": decision_in})
    assert c.identify(b"wav").decision == expected
