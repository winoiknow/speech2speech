# Copyright 2026 winoiknow (Eric Alborn, Anteon Group)
# Licensed under the Apache License, Version 2.0 (the "License").
# See the LICENSE file in the repository root for the full license text.

"""Minimal synchronous WebSocket client (stdlib only).

Just enough of RFC 6455 to talk to a streaming TTS endpoint (MiniMax T2A v2)
from a sync pipeline handler: TLS handshake, masked client text frames, and
frame reassembly for server text/binary frames. No external dependency, and it
fits the sync-generator handler model (no asyncio in the thread).
"""

from __future__ import annotations

import base64
import secrets
import socket
import ssl
import struct
import urllib.parse


class WSClient:
    def __init__(self, url: str, headers: dict | None = None, timeout: float = 20.0):
        u = urllib.parse.urlsplit(url)
        host = u.hostname
        if host is None:
            raise ValueError(f"invalid ws url: {url!r}")
        port = u.port or (443 if u.scheme == "wss" else 80)
        path = (u.path or "/") + (("?" + u.query) if u.query else "")
        s = socket.create_connection((host, port), timeout=timeout)
        if u.scheme == "wss":
            s = ssl.create_default_context().wrap_socket(s, server_hostname=host)
        key = base64.b64encode(secrets.token_bytes(16)).decode()
        req = (f"GET {path} HTTP/1.1\r\nHost: {host}\r\nUpgrade: websocket\r\n"
               f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\n"
               f"Sec-WebSocket-Version: 13\r\n")
        for k, v in (headers or {}).items():
            req += f"{k}: {v}\r\n"
        s.sendall((req + "\r\n").encode())
        resp = b""
        while b"\r\n\r\n" not in resp:
            chunk = s.recv(4096)
            if not chunk:
                raise ConnectionError("connection closed during WS handshake")
            resp += chunk
        status = resp.split(b"\r\n", 1)[0].decode("latin1")
        if " 101 " not in status:
            raise RuntimeError(f"WS handshake failed: {status} :: {resp[:300]!r}")
        self._s = s

    def send_text(self, data: str) -> None:
        payload = data.encode("utf-8")
        n = len(payload)
        frame = bytearray([0x81])  # FIN + text opcode
        if n < 126:
            frame.append(0x80 | n)
        elif n < 65536:
            frame.append(0x80 | 126); frame += struct.pack(">H", n)
        else:
            frame.append(0x80 | 127); frame += struct.pack(">Q", n)
        mask = secrets.token_bytes(4)
        frame += mask
        frame += bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        self._s.sendall(frame)

    def _readn(self, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            chunk = self._s.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("connection closed")
            buf += chunk
        return buf

    def recv(self) -> tuple[str, bytes]:
        """Read one message. Returns (kind, payload); kind in text|binary|close.
        Reassembles fragmented frames; ping/pong are skipped."""
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
            if opcode in (0x9, 0xA):  # ping/pong — ignore
                continue
            data += payload
            if fin:
                return ("binary" if opcode == 0x2 else "text", data)

    def close(self) -> None:
        try:
            self._s.sendall(b"\x88\x80" + secrets.token_bytes(4))  # masked close frame
        except Exception:
            pass
        try:
            self._s.close()
        except Exception:
            pass
