"""SQLite storage substrate for the two memory layers.

Both banks may share one SQLite file (two logical tables whose interfaces and
semantics stay strictly separate). Write discipline inherited from the legacy
project: cross-process ``fcntl.flock`` around write transactions and atomic
file replacement. There are no global registries, no singletons — callers
always pass an explicit path (``--home`` / ``OR_HARNESS_HOME``).
"""

from __future__ import annotations

import contextlib
import json
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

try:  # POSIX flock; gracefully degrade elsewhere.
    import fcntl  # type: ignore
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None  # type: ignore


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS executions (
    execution_id TEXT PRIMARY KEY,
    task_id      TEXT NOT NULL,
    strategy_id  TEXT NOT NULL,
    family       TEXT NOT NULL,
    group_l1     TEXT NOT NULL,
    source       TEXT NOT NULL DEFAULT 'executed',
    created_at   REAL NOT NULL,
    payload      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_executions_group ON executions(group_l1, strategy_id);
CREATE INDEX IF NOT EXISTS idx_executions_task ON executions(task_id);

CREATE TABLE IF NOT EXISTS strategic_entries (
    entry_id     TEXT PRIMARY KEY,
    strategy_id  TEXT NOT NULL,
    scope_level  TEXT NOT NULL,
    status       TEXT NOT NULL,
    payload      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_entries_status ON strategic_entries(status);

CREATE TABLE IF NOT EXISTS cold_archive (
    pattern_hash TEXT PRIMARY KEY,
    payload      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class StorageError(Exception):
    """Raised on malformed payloads or storage-level integrity problems."""


class Store:
    """A single SQLite file holding both memory layers plus the cold archive.

    The two layers share the file but never each other's tables; all access
    goes through the bank classes in :mod:`or_harness.strategy`.
    """

    def __init__(self, home: os.PathLike[str] | str):
        self.home = Path(home).expanduser().resolve()
        self.home.mkdir(parents=True, exist_ok=True)
        self.db_path = self.home / "or_harness.db"
        self._lock_path = self.home / ".or_harness.lock"
        self._thread_lock = threading.RLock()
        self._local = threading.local()
        self._init_schema()

    # -- connection management -------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    @property
    def conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._connect()
            self._local.conn = conn
        return conn

    def _init_schema(self) -> None:
        with self.locked():
            self.conn.executescript(SCHEMA_SQL)
            self.conn.commit()

    # -- locking ----------------------------------------------------------------

    @contextlib.contextmanager
    def locked(self) -> Iterator[None]:
        """Thread lock + cross-process flock around a write transaction."""
        with self._thread_lock:
            lock_fd: Optional[int] = None
            try:
                lock_fd = os.open(str(self._lock_path), os.O_CREAT | os.O_RDWR, 0o644)
                if fcntl is not None:
                    fcntl.flock(lock_fd, fcntl.LOCK_EX)
                yield
            finally:
                if lock_fd is not None:
                    try:
                        if fcntl is not None:
                            fcntl.flock(lock_fd, fcntl.LOCK_UN)
                    finally:
                        os.close(lock_fd)

    @contextlib.contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Locked write transaction; rolls back on error."""
        with self.locked():
            conn = self.conn
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    # -- helpers -----------------------------------------------------------------

    @staticmethod
    def dumps(obj: Any) -> str:
        return json.dumps(obj, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def loads(raw: str) -> Dict[str, Any]:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise StorageError(f"malformed JSON payload: {exc}") from exc
        if not isinstance(data, dict):
            raise StorageError("payload must be a JSON object")
        return data

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None


def resolve_home(home: Optional[str] = None) -> Path:
    """Explicit memory location: --home flag, else OR_HARNESS_HOME, else ./or_harness_home."""
    if home:
        return Path(home).expanduser().resolve()
    env = os.environ.get("OR_HARNESS_HOME")
    if env:
        return Path(env).expanduser().resolve()
    return (Path.cwd() / "or_harness_home").resolve()
