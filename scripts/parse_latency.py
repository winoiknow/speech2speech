#!/usr/bin/env python3
# Copyright 2026 winoiknow (Eric Alborn, Anteon Group)
# Licensed under the Apache License, Version 2.0 (the "License").
#
# Parse an s2s container log (captured with DEBUG_MODE=on) and report, per turn,
# the time from end-of-caller-speech to first audio leaving s2s, with a breakdown.
# Use it for the STREAM_BATCH_SENTENCES A/B (LATENCY.md §4b): run a call, save the
# log, parse it, record the median.
#
# Stdlib only. Usage:
#   docker compose -f docker-compose.remote.yml logs --no-color s2s-remote > call.log
#   python3 scripts/parse_latency.py call.log
#
# Markers used (all present under DEBUG_MODE=on):
#   "Speech ended (… ms), stop listening"             → T0, end of caller's turn
#   "ResponsesApiModelHandler: N s"                    → LLM batch time (first = first batch)
#   "RemoteOpenAITTS|ElevenLabsTTS: stream opened"     → first TTS request started
#   "time-to-first-byte X.XXXs"                        → TTS TTFB
#   "iter_bytes loop done" / "stream done"             → first batch fully synthesized
#                                                         (≈ first audio starts leaving s2s,
#                                                          since the handlers buffer the clip)
from __future__ import annotations

import re
import statistics
import sys
from datetime import datetime

TS = re.compile(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d\d\d)")


def ts(line: str):
    m = TS.search(line)
    return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S,%f") if m else None


def parse(path: str):
    turns = []          # one dict per caller turn
    cur = None
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            t = ts(line)
            if "Speech ended" in line and "stop listening" in line:
                if cur:
                    turns.append(cur)
                cur = {"t0": t, "llm_s": None, "ttfb_s": None,
                       "t_stream_open": None, "t_first_audio": None}
                continue
            if cur is None:
                continue
            if cur["llm_s"] is None and "ResponsesApiModelHandler:" in line:
                m = re.search(r"ResponsesApiModelHandler:\s*([\d.]+)\s*s", line)
                if m:
                    cur["llm_s"] = float(m.group(1))
            if cur["t_stream_open"] is None and "stream opened" in line:
                cur["t_stream_open"] = t
            if cur["ttfb_s"] is None:
                m = re.search(r"time-to-first-byte\s*([\d.]+)s", line)
                if m:
                    cur["ttfb_s"] = float(m.group(1))
            # First-audio marker. Streaming handlers (MiniMax) emit "first audio
            # chunk" at the true first frame; buffering handlers (F5/ElevenLabs)
            # have no progressive marker, so "iter_bytes loop done"/"stream done"
            # (end of the buffered clip) is the closest proxy. Whichever appears
            # first wins. Gated on t_stream_open so a prior response still draining
            # can't mis-pair (which would show up as negative synth time).
            if (cur["t_first_audio"] is None and cur["t_stream_open"] is not None
                    and ("first audio chunk" in line
                         or "iter_bytes loop done" in line or "stream done" in line)):
                cur["t_first_audio"] = t
    if cur:
        turns.append(cur)
    return turns


def secs(a, b):
    return (b - a).total_seconds() if (a and b) else None


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("usage: parse_latency.py <s2s-log-file>")
    turns = parse(sys.argv[1])
    if not turns:
        sys.exit("no caller turns found (need DEBUG_MODE=on; look for 'Speech ended … stop listening')")

    print(f"{'turn':>4} {'spch→1st audio':>14} {'spch→TTS-open':>13} {'LLM(s)':>7} {'TTFB(s)':>8} {'synth(s)':>9}")
    first_audio, to_open = [], []
    for i, c in enumerate(turns, 1):
        fa = secs(c["t0"], c["t_first_audio"])
        op = secs(c["t0"], c["t_stream_open"])
        synth = secs(c["t_stream_open"], c["t_first_audio"])
        if fa is not None:
            first_audio.append(fa)
        if op is not None:
            to_open.append(op)
        def fmt(x):
            return f"{x:.3f}" if x is not None else "  —  "
        print(f"{i:>4} {fmt(fa):>14} {fmt(op):>13} "
              f"{fmt(c['llm_s']):>7} {fmt(c['ttfb_s']):>8} {fmt(synth):>9}")

    print("-" * 64)
    if first_audio:
        print(f"median speech→first-audio : {statistics.median(first_audio):.3f} s   (n={len(first_audio)})")
    if to_open:
        print(f"median speech→TTS-open    : {statistics.median(to_open):.3f} s   "
              f"(= STT + LLM first batch + queueing)")
    print("\nNote: 'first audio' ≈ first batch fully synthesized (handlers buffer the clip),")
    print("which is when audio starts leaving s2s. Lower STREAM_BATCH_SENTENCES should pull")
    print("both 'speech→TTS-open' (fewer sentences to generate) and 'synth' (shorter clip) down.")


if __name__ == "__main__":
    main()
