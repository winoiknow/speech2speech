"""VAD pre-warm: load Silero once at startup, deepcopy per session.

The per-connect pipeline build used to run ``torch.hub.load`` on the hot connect
path (a multi-second spike, worst on the first connect). The factory now loads
Silero once and hands each session an independent ``deepcopy``. These tests pin:
  * the handler uses an injected model and skips ``torch.hub.load`` entirely;
  * the factory's deepcopy/None plumbing;
  * the no-bleed property — concurrent copies keep independent VAD state — so
    multi-session can share the pre-warm without cross-talk.
"""

from queue import Queue
from threading import Event as ThreadingEvent

import pytest

from speech_to_speech.pipeline.session_pipeline import HandlerFactory


class _StubModel:
    """Minimal stand-in for the Silero JIT model: VADIterator only calls
    reset_states() at construction; the handler never calls it directly."""

    def __init__(self) -> None:
        self.reset_calls = 0

    def reset_states(self) -> None:
        self.reset_calls += 1


def test_handler_uses_injected_model_without_loading(monkeypatch):
    """An injected vad_model is used as-is; torch.hub.load is never called."""
    from speech_to_speech.VAD import vad_handler as vh

    def _boom(*a, **k):  # would fire only if the inject path fell through to load
        raise AssertionError("torch.hub.load must not run when vad_model is injected")

    monkeypatch.setattr(vh.torch.hub, "load", _boom)

    stub = _StubModel()
    handler = vh.VADHandler(
        ThreadingEvent(),
        queue_in=Queue(),
        queue_out=Queue(),
        setup_args=(ThreadingEvent(),),  # should_listen
        setup_kwargs={"vad_model": stub},
    )
    assert handler.model is stub
    # VADIterator.__init__ resets the model exactly once on construction.
    assert stub.reset_calls == 1


def test_vad_uses_injected_turn_detector(monkeypatch):
    """When TURN_DETECTION=smart_turn and a shared detector is injected, the
    handler uses it as-is and never constructs/loads its own ONNX session."""
    from speech_to_speech.VAD import vad_handler as vh

    def _no_load(*a, **k):
        raise AssertionError("Silero must not load when a vad_model is injected")

    monkeypatch.setattr(vh.torch.hub, "load", _no_load)

    class _Detector:
        available = True

    class _Model:
        def reset_states(self):
            pass

    shared = _Detector()
    handler = vh.VADHandler(
        ThreadingEvent(),
        queue_in=Queue(),
        queue_out=Queue(),
        setup_args=(ThreadingEvent(),),
        setup_kwargs={
            "vad_model": _Model(),
            "turn_detection": "smart_turn",
            "turn_detector": shared,
        },
    )
    assert handler.turn_detector is shared


def test_new_vad_model_none_without_template():
    f = HandlerFactory.__new__(HandlerFactory)  # bypass heavy __init__
    f._vad_template = None
    assert f._new_vad_model() is None


def test_new_vad_model_returns_independent_copy():
    f = HandlerFactory.__new__(HandlerFactory)
    f._vad_template = _StubModel()
    a = f._new_vad_model()
    b = f._new_vad_model()
    assert a is not f._vad_template
    assert b is not f._vad_template
    assert a is not b


def test_new_vad_model_failsafe_on_deepcopy_error():
    class _Uncopyable:
        def __deepcopy__(self, memo):
            raise RuntimeError("nope")

    f = HandlerFactory.__new__(HandlerFactory)
    f._vad_template = _Uncopyable()
    # Falls back to None (handler will load Silero itself) rather than raising.
    assert f._new_vad_model() is None


# ── No-bleed property against the real model (skipped if Silero unavailable) ──

def _load_silero():
    try:
        import torch

        torch.set_grad_enabled(False)
        model, _ = torch.hub.load(
            "snakers4/silero-vad", "silero_vad", trust_repo=True, skip_validation=True
        )
        return torch, model
    except Exception:
        return None, None


def test_deepcopies_do_not_bleed_state():
    """Two deepcopied Silero instances keep independent RNN state: driving one
    must not perturb the other. This is the safety guarantee that lets all
    sessions deepcopy a single pre-warmed template."""
    torch, template = _load_silero()
    if template is None:
        pytest.skip("Silero VAD not available in this environment")

    f = HandlerFactory.__new__(HandlerFactory)
    f._vad_template = template
    a = f._new_vad_model()
    b = f._new_vad_model()

    # Prime B once, snapshot its state, then hammer A and confirm B is untouched.
    b(torch.randn(512) * 0.4, 16000)
    b_state_before = b._state.clone()
    for _ in range(15):
        a(torch.randn(512) * 0.6, 16000)
    assert torch.equal(b_state_before, b._state)

    # Resetting A leaves B alone.
    a.reset_states()
    assert torch.equal(b_state_before, b._state)
