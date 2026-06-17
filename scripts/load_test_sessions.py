# Copyright 2026 winoiknow (Eric Alborn, Anteon Group)
# Licensed under the Apache License, Version 2.0 (the "License").
# See the LICENSE file in the repository root for the full license text.
#
# Multi-session Phase D §6.4 load test: drive 2/4/8 (configurable) concurrent
# turns against a running s2s realtime server and report p50/p95 first-audio
# latency per concurrency level. Pick the documented S2S_MAX_SESSIONS from the
# knee of the curve (where p95 first-audio crosses your latency budget), then
# record the table in LATENCY.md §7.
#
# Prereqs: an s2s server in realtime mode with the REMOTE profile (remote
# STT/TTS/LLM) and S2S_MAX_SESSIONS >= the largest concurrency you test, plus a
# short speech WAV to use as the test utterance.
#
# Example:
#   S2S_MAX_SESSIONS=8 python -m speech_to_speech.s2s_pipeline --mode realtime ... &
#   python scripts/load_test_sessions.py --wav sample.wav --concurrencies 2,4,8 --rounds 5

from __future__ import annotations

import argparse
import asyncio
import sys

from realtime_loadgen import (
    LatencySummary,
    TurnResult,
    connect,
    http_get_json,
    load_wav_16k_mono,
    run_turn,
    summarize,
)


async def _run_level(host, port, api_key, pcm, concurrency, rounds, timeout):
    """Open `concurrency` sessions; fire `rounds` waves of concurrent turns."""
    conns = []
    try:
        for _ in range(concurrency):
            conns.append(await connect(host, port, api_key))
    except Exception as e:
        for c in conns:
            await c.close()
        raise RuntimeError(f"could not open {concurrency} sessions: {e}") from e

    results: list[TurnResult] = []
    try:
        for r in range(rounds):
            wave_results = await asyncio.gather(
                *(run_turn(ws, pcm, timeout=timeout) for ws in conns), return_exceptions=True
            )
            for res in wave_results:
                if isinstance(res, Exception):
                    results.append(TurnResult(error={"type": "exception", "message": str(res)}))
                else:
                    results.append(res)
            print(f"  level {concurrency}: round {r + 1}/{rounds} done", file=sys.stderr)
    finally:
        for c in conns:
            await c.close()
    return summarize(results)


def _fmt(summ: LatencySummary) -> str:
    return (
        f"n={summ.n:<3} ok={summ.ok:<3} ({summ.success_rate * 100:5.1f}%)  "
        f"p50={summ.p50:6.3f}  p95={summ.p95:6.3f}  p99={summ.p99:6.3f}  "
        f"mean={summ.mean:6.3f}  max={summ.max:6.3f}"
    )


async def main_async(args) -> int:
    pcm = load_wav_16k_mono(args.wav)
    dur = len(pcm) / (16000 * 2)
    print(f"Test utterance: {args.wav} ({dur:.2f}s @ 16 kHz mono)", file=sys.stderr)

    # Sanity-check the server's advertised capacity before hammering it.
    try:
        info = http_get_json(args.host, args.port, "/v1/sessions")
        cap = info.get("max_sessions")
        print(f"Server reports max_sessions={cap}", file=sys.stderr)
        biggest = max(args.concurrencies)
        if cap is not None and cap < biggest:
            print(
                f"WARNING: max_sessions={cap} < requested concurrency {biggest}; "
                f"higher levels will see session_limit_reached. Set S2S_MAX_SESSIONS>={biggest}.",
                file=sys.stderr,
            )
    except Exception as e:
        print(f"(could not read /v1/sessions: {e})", file=sys.stderr)

    summaries: dict[int, LatencySummary] = {}
    for c in args.concurrencies:
        print(f"\n=== concurrency {c} ===", file=sys.stderr)
        summaries[c] = await _run_level(args.host, args.port, args.api_key, pcm, c, args.rounds, args.timeout)
        print(f"concurrency {c:>2}: {_fmt(summaries[c])}")

    print("\n--- first-audio latency (speech_stopped → first audio delta), seconds ---")
    print(f"{'concurrency':>11}  {'turns':>5}  {'ok%':>6}  {'p50':>6}  {'p95':>6}  {'p99':>6}  {'mean':>6}  {'max':>6}")
    for c in args.concurrencies:
        s = summaries[c]
        print(
            f"{c:>11}  {s.n:>5}  {s.success_rate * 100:>5.1f}  "
            f"{s.p50:>6.3f}  {s.p95:>6.3f}  {s.p99:>6.3f}  {s.mean:>6.3f}  {s.max:>6.3f}"
        )

    if args.markdown:
        print("\n<!-- paste into LATENCY.md §7 -->")
        print("| Concurrent turns | Turns | OK% | p50 (s) | p95 (s) | p99 (s) | max (s) |")
        print("|---|---|---|---|---|---|---|")
        for c in args.concurrencies:
            s = summaries[c]
            print(
                f"| {c} | {s.n} | {s.success_rate * 100:.1f} | "
                f"{s.p50:.3f} | {s.p95:.3f} | {s.p99:.3f} | {s.max:.3f} |"
            )

    # Surface any errors so a "fast but failing" run isn't mistaken for success.
    all_errors = {e for s in summaries.values() for e in s.errors}
    if all_errors:
        print("\nErrors observed:", file=sys.stderr)
        for e in sorted(all_errors):
            print(f"  - {e}", file=sys.stderr)

    worst_ok = min((s.success_rate for s in summaries.values()), default=1.0)
    return 0 if worst_ok >= args.min_success else 1


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--api-key", default=None, help="Bearer token if the server requires one")
    p.add_argument("--wav", required=True, help="Speech WAV used as the test utterance (16-bit PCM)")
    p.add_argument(
        "--concurrencies",
        type=lambda s: [int(x) for x in s.split(",") if x],
        default=[2, 4, 8],
        help="Comma-separated concurrency levels (default: 2,4,8)",
    )
    p.add_argument("--rounds", type=int, default=5, help="Concurrent-turn waves per level (default: 5)")
    p.add_argument("--timeout", type=float, default=60.0, help="Per-turn timeout seconds (default: 60)")
    p.add_argument("--min-success", type=float, default=0.95, help="Min success rate for exit 0 (default: 0.95)")
    p.add_argument("--markdown", action="store_true", help="Also emit a LATENCY.md table block")
    return p.parse_args(argv)


def main() -> None:
    args = parse_args()
    try:
        rc = asyncio.run(main_async(args))
    except KeyboardInterrupt:
        rc = 130
    sys.exit(rc)


if __name__ == "__main__":
    main()
