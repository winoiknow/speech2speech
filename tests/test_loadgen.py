"""Unit tests for the multi-session load/soak harness (scripts/realtime_loadgen.py).

Covers the pure logic — percentiles, summary, WAV loading/resampling, and the
soak growth check — so the harness can't silently break. The networked paths
(connect/run_turn) need a live server and are exercised by the operators.
"""

import math
import struct
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

# The scripts dir is not a package; put it on the path so the harness imports.
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import realtime_loadgen as lg  # noqa: E402


def test_percentile_basic():
    vals = [1.0, 2.0, 3.0, 4.0]
    assert lg.percentile(vals, 0) == 1.0
    assert lg.percentile(vals, 100) == 4.0
    assert lg.percentile(vals, 50) == pytest.approx(2.5)
    assert math.isnan(lg.percentile([], 50))
    assert lg.percentile([7.0], 95) == 7.0


def test_summarize_counts_and_errors():
    results = [
        lg.TurnResult(first_audio_latency=1.0, status="completed", ok=True),
        lg.TurnResult(first_audio_latency=3.0, status="completed", ok=True),
        lg.TurnResult(error={"type": "session_limit_reached", "message": "full"}),
    ]
    s = lg.summarize(results)
    assert s.n == 3
    assert s.ok == 2
    assert s.success_rate == pytest.approx(2 / 3)
    assert s.p50 == pytest.approx(2.0)
    assert s.max == pytest.approx(3.0)
    assert any("session_limit_reached" in e for e in s.errors)


def _write_wav(path, audio_i16, rate, channels=1):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(audio_i16.astype("<i2").tobytes())


def test_load_wav_passthrough_16k_mono(tmp_path):
    audio = (np.sin(np.linspace(0, 20, 1600)) * 1000).astype(np.int16)
    p = tmp_path / "a.wav"
    _write_wav(p, audio, 16000)
    out = lg.load_wav_16k_mono(str(p))
    assert np.frombuffer(out, dtype="<i2").shape[0] == 1600


def test_load_wav_resamples_and_downmixes(tmp_path):
    # 48 kHz stereo → expect ~1/3 the frames, mono.
    n = 4800
    stereo = np.stack([np.arange(n) % 100, np.arange(n) % 100], axis=1).astype(np.int16).reshape(-1)
    p = tmp_path / "b.wav"
    _write_wav(p, stereo, 48000, channels=2)
    out = lg.load_wav_16k_mono(str(p))
    frames = np.frombuffer(out, dtype="<i2").shape[0]
    assert abs(frames - 1600) <= 2  # 4800 @ 48k → 1600 @ 16k


def test_load_wav_rejects_non_16bit(tmp_path):
    p = tmp_path / "c.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(1)  # 8-bit
        w.setframerate(16000)
        w.writeframes(struct.pack("8B", *range(8)))
    with pytest.raises(ValueError):
        lg.load_wav_16k_mono(str(p))


def test_growth_report_detects_thread_leak():
    from soak_sessions import _growth_report  # noqa: E402

    # Threads climb from 50 to 80 across the run → leak.
    samples = [lg.ResourceSample(t=float(i), sessions=8, threads=50 + i * 3, fds=100, rss_mb=200.0) for i in range(10)]
    healthy, _ = _growth_report(samples)
    assert healthy is False


def test_growth_report_passes_stable():
    from soak_sessions import _growth_report  # noqa: E402

    samples = [lg.ResourceSample(t=float(i), sessions=8, threads=60, fds=100, rss_mb=200.0) for i in range(10)]
    healthy, _ = _growth_report(samples)
    assert healthy is True


def test_load_modules_import_and_parse_args():
    import load_test_sessions as lt  # noqa: E402
    import soak_sessions as sk  # noqa: E402

    a = lt.parse_args(["--wav", "x.wav", "--concurrencies", "2,4,8", "--rounds", "3"])
    assert a.concurrencies == [2, 4, 8] and a.rounds == 3
    b = sk.parse_args(["--wav", "x.wav", "--sessions", "4"])
    assert b.sessions == 4
