# Copyright 2026 winoiknow (Eric Alborn, Anteon Group)
# Licensed under the Apache License, Version 2.0 (the "License").

"""Speaker store behind a pluggable interface.

``SpeakerStore`` is the contract; ``SqliteNumpyStore`` is the v1 implementation —
embeddings as BLOBs in SQLite, mirrored into an in-memory (N, D) matrix so a
query is one normalized matrix-vector product (sub-ms for N < ~thousands).

Limits (documented): SQLite is single-writer (one enroll/delete at a time; reads
concurrent). Fine for a single-instance, read-heavy, occasional-enrollment
service. Horizontal scale / high write rate → swap in QdrantStore/ChromaStore
behind this same interface (collection per embedding_model).
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np

logger = logging.getLogger("speaker_id.store")


@dataclass
class Match:
    speaker_id: str
    name: Optional[str]
    score: float


@dataclass
class SpeakerMeta:
    speaker_id: str
    name: Optional[str]
    language: Optional[str]
    samples: int
    created: str


@dataclass
class Invite:
    id: str
    speaker_id: str
    email: Optional[str]
    created: str
    expires: str
    revoked: bool
    samples_added: int
    last_used: Optional[str]


class SpeakerStore(Protocol):
    def add_speaker(self, name: Optional[str], language: Optional[str], speaker_id: Optional[str] = None) -> str: ...
    def add_embedding(self, speaker_id: str, embedding: np.ndarray, sample_source: str = "") -> str: ...
    def query(self, embedding: np.ndarray, top_k: int = 5) -> list[Match]: ...
    def list_speakers(self) -> list[SpeakerMeta]: ...
    def delete_speaker(self, speaker_id: str) -> bool: ...
    def count(self) -> int: ...


class SqliteNumpyStore:
    def __init__(self, path: str, dim: int, embedding_model: str) -> None:
        self.dim = dim
        self.embedding_model = embedding_model
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS speakers (
                speaker_id TEXT PRIMARY KEY,
                name       TEXT,
                language   TEXT,
                created    TEXT,
                consent    INTEGER DEFAULT 0,
                consent_ts TEXT
            );
            CREATE TABLE IF NOT EXISTS embeddings (
                id              TEXT PRIMARY KEY,
                speaker_id      TEXT NOT NULL REFERENCES speakers(speaker_id) ON DELETE CASCADE,
                vec             BLOB NOT NULL,
                dim             INTEGER NOT NULL,
                embedding_model TEXT NOT NULL,
                sample_source   TEXT,
                date            TEXT
            );
            CREATE TABLE IF NOT EXISTS invites (
                id            TEXT PRIMARY KEY,
                speaker_id    TEXT NOT NULL REFERENCES speakers(speaker_id) ON DELETE CASCADE,
                email         TEXT,
                token_hash    TEXT NOT NULL UNIQUE,
                created       TEXT,
                expires       TEXT,
                revoked       INTEGER DEFAULT 0,
                samples_added INTEGER DEFAULT 0,
                last_used     TEXT
            );
            """
        )
        self._conn.commit()
        self._migrate()
        # in-memory mirror: (N, dim) matrix + parallel speaker_id list
        self._mat = np.zeros((0, dim), dtype=np.float32)
        self._ids: list[str] = []
        self._reload()

    def _migrate(self) -> None:
        """Add columns introduced after a DB was first created (CREATE TABLE IF
        NOT EXISTS won't alter an existing table) so a Phase-1 store keeps its data."""
        cols = {r[1] for r in self._conn.execute("PRAGMA table_info(speakers)").fetchall()}
        for col, ddl in (("consent", "INTEGER DEFAULT 0"), ("consent_ts", "TEXT")):
            if col not in cols:
                self._conn.execute(f"ALTER TABLE speakers ADD COLUMN {col} {ddl}")
        self._conn.commit()

    # ── matrix mirror ────────────────────────────────────────────────────
    def _reload(self) -> None:
        rows = self._conn.execute(
            "SELECT speaker_id, vec FROM embeddings WHERE embedding_model=? AND dim=?",
            (self.embedding_model, self.dim),
        ).fetchall()
        if rows:
            self._ids = [r[0] for r in rows]
            self._mat = np.vstack([np.frombuffer(r[1], dtype=np.float32) for r in rows]).astype(np.float32)
        else:
            self._ids = []
            self._mat = np.zeros((0, self.dim), dtype=np.float32)
        logger.info("store: loaded %d embeddings for model %s", len(self._ids), self.embedding_model)

    # ── writes ───────────────────────────────────────────────────────────
    def add_speaker(
        self,
        name: Optional[str],
        language: Optional[str],
        speaker_id: Optional[str] = None,
        consent: bool = False,
    ) -> str:
        sid = speaker_id or f"u_{uuid.uuid4().hex[:12]}"
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO speakers(speaker_id, name, language, created, consent, consent_ts)"
                " VALUES (?,?,?,?,?,?)",
                (sid, name, language, now, 1 if consent else 0, now if consent else None),
            )
            self._conn.commit()
        return sid

    def add_embedding(self, speaker_id: str, embedding: np.ndarray, sample_source: str = "") -> str:
        v = np.ascontiguousarray(embedding, dtype=np.float32).reshape(-1)
        if v.shape[0] != self.dim:
            raise ValueError(f"embedding dim {v.shape[0]} != store dim {self.dim}")
        eid = uuid.uuid4().hex
        with self._lock:
            self._conn.execute(
                "INSERT INTO embeddings(id, speaker_id, vec, dim, embedding_model, sample_source, date)"
                " VALUES (?,?,?,?,?,?,?)",
                (eid, speaker_id, v.tobytes(), self.dim, self.embedding_model, sample_source,
                 time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
            )
            self._conn.commit()
            self._reload()
        return eid

    def delete_speaker(self, speaker_id: str) -> bool:
        with self._lock:
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.execute("DELETE FROM embeddings WHERE speaker_id=?", (speaker_id,))
            self._conn.execute("DELETE FROM invites WHERE speaker_id=?", (speaker_id,))
            cur = self._conn.execute("DELETE FROM speakers WHERE speaker_id=?", (speaker_id,))
            self._conn.commit()
            self._reload()
            return cur.rowcount > 0

    def set_consent(self, speaker_id: str, consent: bool = True) -> bool:
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with self._lock:
            cur = self._conn.execute(
                "UPDATE speakers SET consent=?, consent_ts=? WHERE speaker_id=?",
                (1 if consent else 0, now if consent else None, speaker_id),
            )
            self._conn.commit()
            return cur.rowcount > 0

    # ── invites (Phase 5B: scoped self-enrollment) ───────────────────────
    def add_invite(self, speaker_id: str, email: Optional[str], token_hash: str, expires: str) -> str:
        iid = f"inv_{uuid.uuid4().hex[:12]}"
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with self._lock:
            self._conn.execute(
                "INSERT INTO invites(id, speaker_id, email, token_hash, created, expires, revoked, samples_added)"
                " VALUES (?,?,?,?,?,?,0,0)",
                (iid, speaker_id, email, token_hash, now, expires),
            )
            self._conn.commit()
        return iid

    def get_invite_by_token_hash(self, token_hash: str) -> Optional[tuple[Invite, str]]:
        """Return (Invite, speaker_name) for a token hash, or None. Read-only — the
        caller checks expiry/revocation so it can distinguish 'gone' from 'unknown'."""
        r = self._conn.execute(
            "SELECT i.id, i.speaker_id, i.email, i.created, i.expires, i.revoked, i.samples_added,"
            " i.last_used, s.name FROM invites i JOIN speakers s ON s.speaker_id=i.speaker_id"
            " WHERE i.token_hash=?", (token_hash,),
        ).fetchone()
        if not r:
            return None
        return Invite(id=r[0], speaker_id=r[1], email=r[2], created=r[3], expires=r[4],
                      revoked=bool(r[5]), samples_added=r[6], last_used=r[7]), r[8]

    def touch_invite(self, invite_id: str) -> None:
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with self._lock:
            self._conn.execute(
                "UPDATE invites SET samples_added=samples_added+1, last_used=? WHERE id=?", (now, invite_id))
            self._conn.commit()

    def revoke_invite(self, invite_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute("UPDATE invites SET revoked=1 WHERE id=?", (invite_id,))
            self._conn.commit()
            return cur.rowcount > 0

    def list_invites(self) -> list[tuple[Invite, Optional[str]]]:
        rows = self._conn.execute(
            "SELECT i.id, i.speaker_id, i.email, i.created, i.expires, i.revoked, i.samples_added,"
            " i.last_used, s.name FROM invites i JOIN speakers s ON s.speaker_id=i.speaker_id"
            " ORDER BY i.created DESC"
        ).fetchall()
        return [(Invite(id=r[0], speaker_id=r[1], email=r[2], created=r[3], expires=r[4],
                        revoked=bool(r[5]), samples_added=r[6], last_used=r[7]), r[8]) for r in rows]

    # ── reads ────────────────────────────────────────────────────────────
    def query(self, embedding: np.ndarray, top_k: int = 5) -> list[Match]:
        if not self._ids:
            return []
        q = np.ascontiguousarray(embedding, dtype=np.float32).reshape(-1)
        sims = self._mat @ q  # both L2-normalized → cosine
        # aggregate per speaker by max similarity across that speaker's samples
        best: dict[str, float] = {}
        for sid, s in zip(self._ids, sims):
            if s > best.get(sid, -1.0):
                best[sid] = float(s)
        names = self._names()
        ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [Match(speaker_id=sid, name=names.get(sid), score=score) for sid, score in ranked]

    def score_against_speaker(self, embedding: np.ndarray, speaker_id: str) -> float:
        """Max cosine of an embedding against ONE speaker's samples (0 if none).

        Lets the scoped invite test-my-voice report a self-match score without
        querying — or leaking — any other enrolled speaker.
        """
        if not self._ids:
            return 0.0
        q = np.ascontiguousarray(embedding, dtype=np.float32).reshape(-1)
        sims = self._mat @ q
        vals = [float(s) for sid, s in zip(self._ids, sims) if sid == speaker_id]
        return max(vals) if vals else 0.0

    def list_speakers(self) -> list[SpeakerMeta]:
        rows = self._conn.execute(
            "SELECT s.speaker_id, s.name, s.language, s.created,"
            " (SELECT COUNT(*) FROM embeddings e WHERE e.speaker_id=s.speaker_id) "
            "FROM speakers s ORDER BY s.created"
        ).fetchall()
        return [SpeakerMeta(speaker_id=r[0], name=r[1], language=r[2], created=r[3], samples=r[4]) for r in rows]

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM speakers").fetchone()[0]

    def consent_status(self, speaker_id: str) -> bool:
        r = self._conn.execute("SELECT consent FROM speakers WHERE speaker_id=?", (speaker_id,)).fetchone()
        return bool(r[0]) if r else False

    def _names(self) -> dict[str, Optional[str]]:
        return {r[0]: r[1] for r in self._conn.execute("SELECT speaker_id, name FROM speakers").fetchall()}
