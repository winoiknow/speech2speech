#!/usr/bin/env python3
# Copyright 2026 winoiknow (Eric Alborn, Anteon Group)
# Licensed under the Apache License, Version 2.0 (the "License").
#
# Probe MiniMax T2A v2 WebSocket streaming TTS for time-to-first-audio, before
# committing to a full handler. PCM 16000 maps straight onto the s2s pipeline
# rate (no resample), and the WS path is true progressive streaming.
#
# Standard library only (minimal WebSocket client) — no pip install needed.
#
# Env:
#   MINIMAX_API_KEY    (required)  Bearer token from the MiniMax console
#   MINIMAX_VOICE_ID   (required)  cloned voice id, or a system voice for a
#                                  free-tier probe (e.g. "male-qn-qingse",
#                                  "English_expressive_narrator")
#   MINIMAX_MODEL      default "speech-02-turbo" (low-latency); speech-02-hd etc.
#   MINIMAX_GROUP_ID   optional; appended as ?GroupId=… if your account needs it
#   MINIMAX_WS_URL     default "wss://api.minimax.io/ws/v1/t2a_v2"
#                      (use api.minimaxi.com host for the mainland endpoint)
#
# Usage:
#   python3 scripts/probe_minimax.py --iters 3
#   python3 scripts/probe_minimax.py --verbose      # dump events to debug protocol
from __future__ import annotations

import argparse
import base64
import json
import os
import secrets
import socket
import ssl
import statistics
import struct
import time
import urllib.parse

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

SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2  # pcm16


# ── minimal WebSocket client (stdlib) ──────────────────────────────────────
class WS:
    def __init__(self, url: str, headers: dict, timeout: float = 20.0):
        u = urllib.parse.urlsplit(url)
        host = u.hostname
        port = u.port or (443 if u.scheme == "wss" else 80)
        path = (u.path or "/") + (("?" + u.query) if u.query else "")
        s = socket.create_connection((host, port), timeout=timeout)
        if u.scheme == "wss":
            s = ssl.create_default_context().wrap_socket(s, server_hostname=host)
        key = base64.b64encode(secrets.token_bytes(16)).decode()
        req = (f"GET {path} HTTP/1.1\r\nHost: {host}\r\nUpgrade: websocket\r\n"
               f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\n"
               f"Sec-WebSocket-Version: 13\r\n")
        for k, v in headers.items():
            req += f"{k}: {v}\r\n"
        s.sendall((req + "\r\n").encode())
        resp = b""
        while b"\r\n\r\n" not in resp:
            chunk = s.recv(4096)
            if not chunk:
                raise ConnectionError("connection closed during handshake")
            resp += chunk
        status = resp.split(b"\r\n", 1)[0].decode("latin1")
        if " 101 " not in status:
            raise RuntimeError(f"WS handshake failed: {status} :: {resp[:300]!r}")
        self.s = s

    def send_text(self, data: str) -> None:
        payload = data.encode("utf-8")
        n = len(payload)
        frame = bytearray([0x81])  # FIN + text
        if n < 126:
            frame.append(0x80 | n)
        elif n < 65536:
            frame.append(0x80 | 126); frame += struct.pack(">H", n)
        else:
            frame.append(0x80 | 127); frame += struct.pack(">Q", n)
        mask = secrets.token_bytes(4)
        frame += mask
        frame += bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        self.s.sendall(frame)

    def _readn(self, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            chunk = self.s.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("closed")
            buf += chunk
        return buf

    def recv(self) -> tuple[str, bytes]:
        """Return (kind, payload). kind in {'text','binary','close'}."""
        data = b""
        while True:
            b0, b1 = self._readn(2)
            fin = b0 & 0x80
            opcode = b0 & 0x0F
            masked = b1 & 0x80
            length = b1 & 0x7F
            if length == 126:
                length = struct.unpack(">H", self._readn(2))[0]
            elif length == 127:
                length = struct.unpack(">Q", self._readn(8))[0]
            mask = self._readn(4) if masked else b""
            payload = self._readn(length)
            if masked:
                payload = bytes(p ^ mask[i % 4] for i, p in enumerate(payload))
            if opcode == 0x8:
                return ("close", payload)
            if opcode in (0x9, 0xA):   # ping/pong — ignore for a probe
                continue
            data += payload
            if fin:
                return ("binary" if opcode == 0x2 else "text", data)

    def close(self) -> None:
        try:
            self.s.sendall(b"\x88\x80" + secrets.token_bytes(4))  # masked close
        except Exception:
            pass
        try:
            self.s.close()
        except Exception:
            pass


def _extract_audio(ev: dict) -> bytes:
    """MiniMax T2A v2 returns hex-encoded pcm in data.audio."""
    a = (ev.get("data") or {}).get("audio")
    if not a:
        return b""
    try:
        return bytes.fromhex(a)
    except ValueError:
        try:
            return base64.b64decode(a)
        except Exception:
            return b""


def probe_once(text: str, *, verbose: bool = False) -> dict:
    api_key = os.environ["MINIMAX_API_KEY"]
    voice = os.environ["MINIMAX_VOICE_ID"]
    model = os.environ.get("MINIMAX_MODEL", "speech-02-turbo")
    url = os.environ.get("MINIMAX_WS_URL", "wss://api.minimax.io/ws/v1/t2a_v2")
    if os.environ.get("MINIMAX_GROUP_ID"):
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}GroupId={os.environ['MINIMAX_GROUP_ID']}"

    t_start = time.perf_counter()
    ws = WS(url, {"Authorization": f"Bearer {api_key}"})
    # connected_success
    kind, msg = ws.recv()
    if verbose:
        print("  <", msg[:200])
    t_connected = time.perf_counter()

    ws.send_text(json.dumps({
        "event": "task_start",
        "model": model,
        "voice_setting": {"voice_id": voice, "speed": 1.0, "vol": 1.0, "pitch": 0},
        "audio_setting": {"sample_rate": SAMPLE_RATE, "format": "pcm", "channel": 1},
    }))
    kind, msg = ws.recv()                 # task_started (or error)
    if verbose:
        print("  <", msg[:200])
    ev = json.loads(msg)
    if ev.get("base_resp", {}).get("status_code", 0) not in (0, None) or ev.get("event") == "task_failed":
        ws.close()
        raise RuntimeError(f"task_start rejected: {msg[:300]}")

    t_text = time.perf_counter()
    ws.send_text(json.dumps({"event": "task_continue", "text": text}))

    ttfb = None
    nbytes = 0
    while True:
        kind, msg = ws.recv()
        if kind == "close":
            break
        ev = json.loads(msg)
        if verbose and not _extract_audio(ev):
            print("  <", msg[:200])
        chunk = _extract_audio(ev)
        if chunk:
            if ttfb is None:
                ttfb = time.perf_counter()
            nbytes += len(chunk)
        if ev.get("is_final") or ev.get("event") in ("task_finished", "task_failed"):
            break
    t_done = time.perf_counter()
    ws.close()

    if ttfb is None:
        raise RuntimeError("no audio received (check voice_id/model/credentials; try --verbose)")
    audio_s = (nbytes / BYTES_PER_SAMPLE) / SAMPLE_RATE
    return {
        "connect_s": t_connected - t_start,
        "ttfb_cold": ttfb - t_start,     # connect-per-utterance: request start → first audio
        "ttfb_warm": ttfb - t_text,      # persistent connection: text sent → first audio
        "total_s": t_done - t_start,
        "audio_s": audio_s,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Probe MiniMax T2A v2 WebSocket TTS TTFB.")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--verbose", action="store_true", help="dump non-audio events")
    args = ap.parse_args()

    for v in ("MINIMAX_API_KEY", "MINIMAX_VOICE_ID"):
        if not os.environ.get(v):
            raise SystemExit(f"set {v} (and MINIMAX_GROUP_ID if your account needs it)")

    print("ttfb_cold = request-start → first audio (connect-per-utterance, what a v1 handler pays)")
    print("ttfb_warm = text-sent → first audio (what a persistent-connection handler would pay)")
    print(f"model={os.environ.get('MINIMAX_MODEL','speech-02-turbo')}  iters={args.iters} (median)\n")
    print(f"{'text':<30} {'connect':>8} {'ttfb_cold':>10} {'ttfb_warm':>10} {'total':>8} {'audio':>7}")
    for label, text in TEXTS.items():
        rows = []
        for _ in range(args.iters):
            try:
                rows.append(probe_once(text, verbose=args.verbose))
            except Exception as e:
                print(f"{label:<30}  ERROR: {e}")
                break
        else:
            def med(k):
                return statistics.median(r[k] for r in rows)
            print(f"{label:<30} {med('connect_s'):>8.3f} {med('ttfb_cold'):>10.3f} "
                  f"{med('ttfb_warm'):>10.3f} {med('total_s'):>8.3f} {rows[-1]['audio_s']:>7.2f}")


if __name__ == "__main__":
    main()
