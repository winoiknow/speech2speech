#!/usr/bin/env python3
# Copyright 2026 winoiknow (Eric Alborn, Anteon Group)
# Licensed under the Apache License, Version 2.0 (the "License").
#
# Benchmark TTS latency for the openai-remote (F5) and ElevenLabs endpoints the
# s2s pipeline uses. The metric that matters for perceived latency is
# time-to-first-audio-byte (TTFB); total time and real-time-factor (RTF) are
# reported too. Run this ON THE SERVER that can reach both endpoints.
#
# Standard library only (urllib) — no pip install needed; runs with any python3.
#
# Reads the same env vars as the handlers:
#   F5:         TTS_OPENAI_BASE_URL, TTS_OPENAI_API_KEY, TTS_OPENAI_VOICE, TTS_OPENAI_MODEL, TTS_OPENAI_SPEED
#   ElevenLabs: ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID, ELEVENLABS_MODEL_ID,
#               ELEVENLABS_OUTPUT_FORMAT, ELEVENLABS_BASE_URL
#
# Usage:
#   python3 scripts/bench_tts.py --engine both --iters 3
#   python3 scripts/bench_tts.py --engine elevenlabs --iters 5
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

# Representative payloads: a short greeting, one sentence, and a 3-sentence batch
# (the current stream_batch_sentences=3 unit) so you can see how TTFB scales with
# the amount of text the first TTS call has to synthesize.
TEXTS = {
    "greeting (6 words)": "Hello, how can I help you today?",
    "one sentence (18 words)": (
        "I have gone ahead and checked the calendar for you and it looks "
        "completely clear for the afternoon."
    ),
    "three sentences (~45 words)": (
        "I have gone ahead and checked the calendar for you and it looks "
        "completely clear for the afternoon. There is one tentative hold at "
        "four o'clock that has not been confirmed yet. Would you like me to "
        "send a reminder, or should I leave it as it is for now?"
    ),
}


@dataclass
class Result:
    ttfb: float          # seconds to first audio byte
    total: float         # seconds to last byte
    audio_s: float       # decoded audio duration
    n_bytes: int

    @property
    def rtf(self) -> float:
        return self.total / self.audio_s if self.audio_s else float("nan")


def _read_some(resp, n: int) -> bytes:
    """Return whatever bytes are available in one underlying read (for true TTFB).
    read1() avoids blocking until a full n-byte buffer fills."""
    try:
        return resp.read1(n)  # http.client.HTTPResponse supports read1
    except AttributeError:
        return resp.read(n)


def _bench_stream(url: str, *, headers: dict, json_body: dict,
                  rate: int, bytes_per_sample: float) -> Result:
    """POST a streaming TTS request; measure TTFB and total. rate/bytes_per_sample
    convert received bytes → audio seconds (pcm16=2 B/sample; ulaw=1 B/sample)."""
    data = json.dumps(json_body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={**headers, "Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttfb = None
    n = 0
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            hdr_rate = resp.headers.get("X-Sample-Rate")
            if hdr_rate:
                rate = int(hdr_rate)
            while True:
                chunk = _read_some(resp, 65536)
                if not chunk:
                    break
                if ttfb is None:
                    ttfb = time.perf_counter() - t0
                n += len(chunk)
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", "replace")[:300]
        except Exception:
            pass
        raise RuntimeError(f"HTTP {e.code}: {body}") from None
    total = time.perf_counter() - t0
    audio_s = (n / bytes_per_sample) / rate if rate else 0.0
    return Result(ttfb=ttfb or total, total=total, audio_s=audio_s, n_bytes=n)


def bench_f5(text: str) -> Result:
    base = os.environ.get("TTS_OPENAI_BASE_URL", "http://localhost:8880").rstrip("/")
    return _bench_stream(
        base + "/v1/audio/speech/stream",
        headers={"Authorization": f"Bearer {os.environ.get('TTS_OPENAI_API_KEY','sk-unused')}"},
        json_body={"model": os.environ.get("TTS_OPENAI_MODEL", "tts-1"),
                   "input": text,
                   "voice": os.environ.get("TTS_OPENAI_VOICE", "default"),
                   "speed": float(os.environ.get("TTS_OPENAI_SPEED", "1.0"))},
        rate=int(os.environ.get("TTS_OPENAI_SOURCE_SAMPLE_RATE", "16000")),
        bytes_per_sample=2.0,
    )


def bench_elevenlabs(text: str) -> Result:
    base = os.environ.get("ELEVENLABS_BASE_URL", "https://api.elevenlabs.io").rstrip("/")
    voice = os.environ["ELEVENLABS_VOICE_ID"]
    fmt = os.environ.get("ELEVENLABS_OUTPUT_FORMAT", "pcm_16000")
    if fmt.startswith("pcm_"):
        rate, bps = int(fmt.split("_", 1)[1]), 2.0
    elif fmt == "ulaw_8000":
        rate, bps = 8000, 1.0
    else:
        raise SystemExit(f"unsupported ELEVENLABS_OUTPUT_FORMAT={fmt}")
    return _bench_stream(
        f"{base}/v1/text-to-speech/{voice}/stream?output_format={fmt}",
        headers={"xi-api-key": os.environ["ELEVENLABS_API_KEY"]},
        json_body={"text": text,
                   "model_id": os.environ.get("ELEVENLABS_MODEL_ID", "eleven_flash_v2_5"),
                   "voice_settings": {"stability": float(os.environ.get("ELEVENLABS_STABILITY", "0.5")),
                                      "similarity_boost": float(os.environ.get("ELEVENLABS_SIMILARITY_BOOST", "0.75"))}},
        rate=rate, bytes_per_sample=bps,
    )


def run(engine_name: str, fn, iters: int) -> None:
    print(f"\n=== {engine_name} ===")
    print(f"{'text':<30} {'TTFB(s)':>9} {'total(s)':>9} {'audio(s)':>9} {'RTF':>6}")
    for label, text in TEXTS.items():
        ttfbs, totals = [], []
        last = None
        for _ in range(iters):
            try:
                r = fn(text)
            except Exception as e:
                print(f"{label:<30}  ERROR: {e}")
                break
            ttfbs.append(r.ttfb); totals.append(r.total); last = r
        else:
            print(f"{label:<30} {statistics.median(ttfbs):>9.3f} "
                  f"{statistics.median(totals):>9.3f} {last.audio_s:>9.2f} {last.rtf:>6.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark s2s TTS endpoints (TTFB-focused).")
    ap.add_argument("--engine", choices=["f5", "elevenlabs", "both"], default="both")
    ap.add_argument("--iters", type=int, default=3, help="iterations per text (median reported)")
    args = ap.parse_args()

    print("TTFB = time to first audio byte (the number that drives perceived latency).")
    print(f"iters={args.iters} (median shown)")
    if args.engine in ("f5", "both"):
        run("openai-remote / F5", bench_f5, args.iters)
    if args.engine in ("elevenlabs", "both"):
        if not os.environ.get("ELEVENLABS_API_KEY") or not os.environ.get("ELEVENLABS_VOICE_ID"):
            print("\n=== ElevenLabs === skipped (set ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID)")
        else:
            run("elevenlabs", bench_elevenlabs, args.iters)


if __name__ == "__main__":
    main()
