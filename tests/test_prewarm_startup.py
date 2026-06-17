"""Startup pre-warm: warm the remote LLM once, gate Smart Turn / LLM pre-warm.

A client that opens a new connection per turn rebuilds the per-session pipeline
each time. The dominant build cost was ResponsesApiModelHandler.warmup() — a full
round-trip to the serving endpoint — running on every build. It now warms once per
process; the factory triggers it at startup and gates the pre-warms by config.
"""

from types import SimpleNamespace

from speech_to_speech.LLM import responses_api_language_model as ram
from speech_to_speech.pipeline.session_pipeline import HandlerFactory


class _WarmStub:
    """Reuses the real warmup() method; counts how often the actual round-trip runs."""

    n = 0

    def _do_warmup(self) -> None:
        type(self).n += 1


def test_llm_warmup_runs_once_per_process(monkeypatch):
    monkeypatch.setattr(ram, "_warmed", False)
    monkeypatch.setattr(ram, "_WARMUP_PER_SESSION", False)
    _WarmStub.n = 0

    warmup = ram.ResponsesApiModelHandler.warmup
    warmup(_WarmStub())
    warmup(_WarmStub())
    warmup(_WarmStub())

    assert _WarmStub.n == 1  # only the first build pays the round-trip
    assert ram._warmed is True


def test_llm_warmup_per_session_override_warms_each(monkeypatch):
    monkeypatch.setattr(ram, "_warmed", False)
    monkeypatch.setattr(ram, "_WARMUP_PER_SESSION", True)
    _WarmStub.n = 0

    warmup = ram.ResponsesApiModelHandler.warmup
    warmup(_WarmStub())
    warmup(_WarmStub())

    assert _WarmStub.n == 2  # forced per-session


def test_prewarm_smart_turn_skipped_when_not_smart_turn():
    f = HandlerFactory.__new__(HandlerFactory)  # bypass heavy __init__
    f._smart_turn = None
    f.args = SimpleNamespace(vad_handler_kwargs=SimpleNamespace(turn_detection="vad"))
    f.prewarm_smart_turn()
    assert f._smart_turn is None  # not loaded for plain VAD


def test_prewarm_llm_skipped_for_non_responses_backend():
    f = HandlerFactory.__new__(HandlerFactory)
    f.args = SimpleNamespace(module_kwargs=SimpleNamespace(llm_backend="transformers"))
    # Must return early without constructing anything / raising.
    f.prewarm_llm()
