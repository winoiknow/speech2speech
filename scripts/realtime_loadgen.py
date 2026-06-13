# Copyright 2026 winoiknow (Eric Alborn, Anteon Group)
# Licensed under the Apache License, Version 2.0 (the "License").
# See the LICENSE file in the repository root for the full license text.
#
# Multi-session Phase D load/soak harness — shared library.
#
# Drives synthetic turns over the OpenAI Realtime WS protocol (no audio devices),
# so the load test and the soak test share one client. A "turn" streams a real
# speech WAV (server-VAD endpoints it via trailing silence) and measures
# first-audio latency = speech_stopped → first response.output_audio.delta, the
# same number LATENCY.md minimizes. Requires a running s2s server in realtime
# mode; for meaningful numbers use the remote profile (remote STT/TTS/LLM), since
# multi-session is only enabled there.

from __future__ import annotations

import asyncio
import base64
import json
import os
import statistics
import time
import urllib.request
import wave
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import websockets

PIPELINE_RATE = 16000
BYTES_PER_SAMPLE = 2


# ── Audio ────────────────────────────────────────────────────────────────────


def _resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate:
        return audio
    n_dst = int(round(len(audio) * dst_rate / src_rate))
    x_src = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
    x_dst = np.linspace(0.0, 1.0, num=n_dst, endpoint=False)
    return np.interp(x_dst, x_src, audio.astype(np.float64)).astype(np.int16)


def load_wav_16k_mono(path: str) -> bytes:
    """Load a 16-bit PCM WAV as 16 kHz mono int16 little-endian bytes."""
    with wave.open(path, "rb") as w:
        n_channels = w.getnchannels()
        sampwidth = w.getsampwidth()
        rate = w.getframerate()
        frames = w.readframes(w.getnframes())
    if sampwidth != 2:
        raise ValueError(f"{path}: only 16-bit PCM WAV is supported (got sampwidth={sampwidth})")
    audio = np.frombuffer(frames, dtype="<i2")
    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1).astype(np.int16)
    audio = _resample(audio, rate, PIPELINE_RATE)
    return audio.astype("<i2").tobytes()


# ── WS protocol ──────────────────────────────────────────────────────────────


def realtime_ws_url(host: str, port: int) -> str:
    return f"ws://{host}:{port}/v1/realtime"


def _session_update_msg() -> dict[str, Any]:
    # Server-VAD turn detection so trailing silence endpoints the utterance and
    # triggers STT→LLM→TTS, exactly like a real client.
    return {
        "type": "session.update",
        "session": {
            "type": "realtime",
            "audio": {"input": {"turn_detection": {"type": "server_vad", "interrupt_response": True}}, "output": {}},
        },
    }


async def connect(host: str, port: int, api_key: Optional[str] = None):
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
    ws = await websockets.connect(realtime_ws_url(host, port), additional_headers=headers, max_size=None)
    first = json.loads(await ws.recv())  # session.created (or an error)
    if first.get("type") == "error":
        await ws.close()
        raise RuntimeError(f"connect rejected: {first.get('error')}")
    await ws.send(json.dumps(_session_update_msg()))
    return ws


@dataclass
class TurnResult:
    first_audio_latency: Optional[float] = None  # speech_stopped → first audio delta (s)
    speech_to_done: Optional[float] = None  # speech_stopped → response.done (s)
    status: Optional[str] = None  # response.done status
    error: Optional[dict[str, Any]] = None
    audio_bytes: int = 0
    ok: bool = False


async def run_turn(
    ws,
    pcm: bytes,
    *,
    chunk_ms: int = 20,
    pace: bool = True,
    tail_silence_ms: int = 800,
    timeout: float = 60.0,
) -> TurnResult:
    """Stream one utterance and measure first-audio latency.

    Sends the WAV in chunk_ms slices (paced to wall-clock when pace=True, as a
    real client would), then tail_silence_ms of silence so server-VAD endpoints
    the turn. Reads events until response.done or timeout.
    """
    bytes_per_chunk = int(PIPELINE_RATE * BYTES_PER_SAMPLE * chunk_ms / 1000)
    silence = b"\x00" * bytes_per_chunk

    async def _append(buf: bytes) -> None:
        for i in range(0, len(buf), bytes_per_chunk):
            chunk = buf[i : i + bytes_per_chunk]
            await ws.send(
                json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(chunk).decode("ascii")})
            )
            if pace:
                await asyncio.sleep(chunk_ms / 1000)

    async def _sender() -> None:
        await _append(pcm)
        for _ in range(int(tail_silence_ms / chunk_ms)):
            await ws.send(
                json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(silence).decode("ascii")})
            )
            if pace:
                await asyncio.sleep(chunk_ms / 1000)

    send_task = asyncio.create_task(_sender())
    res = TurnResult()
    t_stop: Optional[float] = None
    t_first: Optional[float] = None
    start = time.monotonic()
    try:
        while True:
            remaining = timeout - (time.monotonic() - start)
            if remaining <= 0:
                break
            raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
            ev = json.loads(raw)
            et = ev.get("type")
            if et == "input_audio_buffer.speech_stopped":
                t_stop = time.monotonic()
            elif et == "response.output_audio.delta":
                if t_first is None:
                    t_first = time.monotonic()
                res.audio_bytes += len(base64.b64decode(ev.get("delta", "")))
            elif et == "response.done":
                res.status = (ev.get("response") or {}).get("status")
                if t_stop is not None:
                    res.speech_to_done = time.monotonic() - t_stop
                break
            elif et == "error":
                res.error = ev.get("error")
                break
    except asyncio.TimeoutError:
        res.error = {"type": "client_timeout", "message": f"no response.done within {timeout}s"}
    finally:
        send_task.cancel()
        try:
            await send_task
        except (asyncio.CancelledError, Exception):
            pass
    if t_first is not None and t_stop is not None:
        res.first_audio_latency = t_first - t_stop
    res.ok = res.error is None and res.first_audio_latency is not None and res.status == "completed"
    return res


# ── Stats ────────────────────────────────────────────────────────────────────


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile (pct in 0..100). Empty → nan."""
    if not values:
        return float("nan")
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


@dataclass
class LatencySummary:
    n: int = 0
    ok: int = 0
    p50: float = float("nan")
    p95: float = float("nan")
    p99: float = float("nan")
    mean: float = float("nan")
    max: float = float("nan")
    errors: list[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return (self.ok / self.n) if self.n else 0.0


def summarize(results: list[TurnResult]) -> LatencySummary:
    lat = [r.first_audio_latency for r in results if r.first_audio_latency is not None]
    summ = LatencySummary(n=len(results), ok=sum(1 for r in results if r.ok))
    summ.errors = [
        f"{(r.error or {}).get('type', 'unknown')}: {(r.error or {}).get('message', '')}"
        for r in results
        if r.error is not None
    ]
    if lat:
        summ.p50 = percentile(lat, 50)
        summ.p95 = percentile(lat, 95)
        summ.p99 = percentile(lat, 99)
        summ.mean = statistics.fmean(lat)
        summ.max = max(lat)
    return summ


# ── Server-side resource sampling (Linux /proc) ──────────────────────────────


def http_get_json(host: str, port: int, path: str, timeout: float = 5.0) -> Any:
    url = f"http://{host}:{port}{path}"
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 (trusted local URL)
        return json.loads(resp.read().decode("utf-8"))


@dataclass
class ResourceSample:
    t: float
    sessions: int
    rss_mb: Optional[float] = None
    threads: Optional[int] = None
    fds: Optional[int] = None


def sample_proc(pid: int) -> tuple[Optional[float], Optional[int], Optional[int]]:
    """(rss_mb, threads, open_fds) for a pid via /proc, or Nones if unavailable."""
    rss_mb = threads = fds = None
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss_mb = int(line.split()[1]) / 1024.0
                elif line.startswith("Threads:"):
                    threads = int(line.split()[1])
    except OSError:
        pass
    try:
        fds = len(os.listdir(f"/proc/{pid}/fd"))
    except OSError:
        pass
    return rss_mb, threads, fds
