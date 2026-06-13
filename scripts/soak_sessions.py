# Copyright 2026 winoiknow (Eric Alborn, Anteon Group)
# Licensed under the Apache License, Version 2.0 (the "License").
# See the LICENSE file in the repository root for the full license text.
#
# Multi-session Phase D §6.3 soak test: hold N warm sessions open, with a single
# randomized active talker running turns, for a long duration. Asserts the server
# does not grow threads / file descriptors / RSS over time (the warm-connection
# leak check). Run it for minutes in CI-ish smoke mode or 24 h on real hardware.
#
# Resource sampling is best-effort via /proc when --server-pid is given (the
# server must be on this host); otherwise it tracks only the session count from
# /v1/sessions. Pair with the dead-thread supervisor (D2): a crashed handler
# shows up as a session that drops out and a server_error on its socket.
#
# Example (5-minute smoke):
#   python scripts/soak_sessions.py --wav sample.wav --sessions 8 \
#       --duration-s 300 --turn-interval-s 8 --server-pid $(pgrep -f s2s_pipeline)

from __future__ import annotations

import argparse
import asyncio
import random
import sys
import time

from realtime_loadgen import (
    ResourceSample,
    connect,
    http_get_json,
    load_wav_16k_mono,
    run_turn,
    sample_proc,
)


async def _sampler(args, stop: asyncio.Event, samples: list[ResourceSample]) -> None:
    while not stop.is_set():
        sessions = -1
        try:
            sessions = http_get_json(args.host, args.port, "/v1/sessions").get("count", -1)
        except Exception:
            pass
        rss = threads = fds = None
        if args.server_pid:
            rss, threads, fds = sample_proc(args.server_pid)
        s = ResourceSample(t=time.monotonic(), sessions=sessions, rss_mb=rss, threads=threads, fds=fds)
        samples.append(s)
        extra = ""
        if rss is not None:
            extra = f"  rss={rss:7.1f}MB  threads={threads}  fds={fds}"
        print(f"[{s.t - samples[0].t:7.1f}s] sessions={sessions}{extra}", file=sys.stderr)
        try:
            await asyncio.wait_for(stop.wait(), timeout=args.sample_interval_s)
        except asyncio.TimeoutError:
            pass


async def _talker(args, pcm, conns, stop: asyncio.Event, stats: dict) -> None:
    """Randomly pick one warm session to be the active talker each interval."""
    while not stop.is_set():
        ws = random.choice(conns)
        res = await run_turn(ws, pcm, timeout=args.timeout)
        stats["turns"] += 1
        if res.ok:
            stats["ok"] += 1
        else:
            stats["fail"] += 1
            err = (res.error or {}).get("type", f"status={res.status}")
            print(f"  turn FAILED: {err}", file=sys.stderr)
        try:
            await asyncio.wait_for(stop.wait(), timeout=args.turn_interval_s)
        except asyncio.TimeoutError:
            pass


def _growth_report(samples: list[ResourceSample]) -> tuple[bool, str]:
    """Compare the second half's peak against the first half's median; flag growth."""
    if len(samples) < 4:
        return True, "too few samples to assess growth"
    half = len(samples) // 2
    first, second = samples[:half], samples[half:]
    lines = []
    leaked = False

    def med(xs):
        xs = sorted(x for x in xs if x is not None)
        return xs[len(xs) // 2] if xs else None

    for name, attr, tol in (("threads", "threads", 0), ("fds", "fds", 4), ("rss_mb", "rss_mb", 0.10)):
        base = med(getattr(s, attr) for s in first)
        peak = max((getattr(s, attr) for s in second if getattr(s, attr) is not None), default=None)
        if base is None or peak is None:
            lines.append(f"  {name:8}: n/a")
            continue
        if name == "rss_mb":
            grew = peak > base * (1 + tol)
            lines.append(f"  {name:8}: baseline={base:.1f}  peak={peak:.1f}  (+{(peak / base - 1) * 100:.1f}%)")
        else:
            grew = peak > base + tol
            lines.append(f"  {name:8}: baseline={base}  peak={peak}  (Δ{peak - base:+})")
        leaked = leaked or grew
    return (not leaked), "\n".join(lines)


async def main_async(args) -> int:
    pcm = load_wav_16k_mono(args.wav)
    print(f"Opening {args.sessions} warm sessions to {args.host}:{args.port}", file=sys.stderr)
    conns = []
    try:
        for _ in range(args.sessions):
            conns.append(await connect(args.host, args.port, args.api_key))
    except Exception as e:
        for c in conns:
            await c.close()
        print(f"FAILED to open warm sessions: {e}", file=sys.stderr)
        return 1

    stop = asyncio.Event()
    samples: list[ResourceSample] = []
    stats = {"turns": 0, "ok": 0, "fail": 0}
    sampler = asyncio.create_task(_sampler(args, stop, samples))
    talker = asyncio.create_task(_talker(args, pcm, conns, stop, stats))

    try:
        await asyncio.sleep(args.duration_s)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        await asyncio.gather(sampler, talker, return_exceptions=True)
        for c in conns:
            await c.close()

    print(f"\nTurns: {stats['turns']}  ok={stats['ok']}  fail={stats['fail']}")
    healthy, report = _growth_report(samples)
    print("Resource trend (baseline = first-half median, peak = second-half max):")
    print(report)
    print("RESULT:", "no growth detected ✓" if healthy else "GROWTH DETECTED ✗")
    # Fail if turns regressed or resources grew.
    ok = healthy and stats["fail"] == 0 and stats["turns"] > 0
    return 0 if ok else 1


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--api-key", default=None)
    p.add_argument("--wav", required=True, help="Speech WAV used by the active talker")
    p.add_argument("--sessions", type=int, default=8, help="Warm sessions to hold open (default: 8)")
    p.add_argument("--duration-s", type=float, default=300.0, help="Soak duration seconds (default: 300)")
    p.add_argument("--turn-interval-s", type=float, default=8.0, help="Seconds between turns (default: 8)")
    p.add_argument("--sample-interval-s", type=float, default=10.0, help="Resource sample period (default: 10)")
    p.add_argument("--timeout", type=float, default=60.0, help="Per-turn timeout seconds (default: 60)")
    p.add_argument("--server-pid", type=int, default=None, help="s2s server PID for /proc RSS/thread/fd sampling")
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
