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
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

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

CREATE TABLE IF NOT EXISTS pending_executions (
    execution_id TEXT PRIMARY KEY,
    task_id      TEXT NOT NULL,
    created_at   REAL NOT NULL,
    payload      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- World-model M1 substrate: belief snapshots and unified action records.
-- Index/log tables only — they reference the two knowledge banks but never
-- constitute a third one (no generalization claims live here).
CREATE TABLE IF NOT EXISTS belief_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    task_id     TEXT NOT NULL,
    episode_id  TEXT,
    created_at  REAL NOT NULL,
    payload     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_snapshots_task ON belief_snapshots(task_id);

CREATE TABLE IF NOT EXISTS action_records (
    action_id   TEXT PRIMARY KEY,
    action_type TEXT NOT NULL,
    task_id     TEXT NOT NULL,
    episode_id  TEXT,
    parent_action_id TEXT,
    source      TEXT NOT NULL DEFAULT 'executed',
    status      TEXT NOT NULL DEFAULT 'running',
    started_at  REAL NOT NULL,
    ended_at    REAL,
    payload     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_actions_task ON action_records(task_id);
CREATE INDEX IF NOT EXISTS idx_actions_episode ON action_records(episode_id);

-- World-model M2: frozen outcome predictions (shadow evaluation). A LOG
-- table — predictions are hypotheses, never knowledge; nothing here enters
-- the Strategy Bank or execution statistics.
CREATE TABLE IF NOT EXISTS world_model_predictions (
    prediction_id TEXT PRIMARY KEY,
    task_id      TEXT NOT NULL,
    episode_id   TEXT,
    created_at   REAL NOT NULL,
    payload      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_predictions_task ON world_model_predictions(task_id);

-- Task text: the SOURCE DOCUMENT retrieval is built from. NOT a knowledge
-- bank: it makes no generalization claim, participates in no statistic, and
-- has no lifecycle. It is the factual attachment of the retrieval input.
-- The primary key is (task_id, text_digest): the same task_id may be solved
-- with different content, each version its own row, and an execution
-- references the exact version it was produced under via
-- ``ExecutionRecord.task_text_digest``.
--
-- NOTE: this table is created here for the same reason every other table is
-- (idempotent IF NOT EXISTS inside Store.__init__, the pre-existing open
-- path). Read-only commands must never ADD a write of their own — see the
-- read-only discipline documented on Store.
CREATE TABLE IF NOT EXISTS task_texts (
    text_digest TEXT NOT NULL,
    task_id     TEXT NOT NULL,
    text        TEXT NOT NULL,
    created_at  REAL NOT NULL,
    PRIMARY KEY (task_id, text_digest)
);
CREATE INDEX IF NOT EXISTS idx_task_texts_task ON task_texts(task_id);

-- World-model phase 2: FROZEN prediction input contexts. A LOG table, like
-- world_model_predictions: a context is a frozen INPUT, not knowledge, and
-- nothing here enters either bank. It is stored so a prediction can be
-- reproduced from the content that was actually used (a mutable id alone
-- would not be enough) and so a historical context can be replayed without
-- reading today's banks.
CREATE TABLE IF NOT EXISTS prediction_contexts (
    context_id  TEXT PRIMARY KEY,
    task_id     TEXT NOT NULL,
    episode_id  TEXT,
    created_at  REAL NOT NULL,
    payload     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_contexts_task ON prediction_contexts(task_id);

-- World-model M3: strategy-outcome predictions under the wm-so/1 protocol.
-- A LOG table like world_model_predictions: a prediction is a frozen
-- hypothesis, never knowledge, and nothing here enters either bank. Stored
-- so a prediction can be bound to the real execution that followed it and
-- so the frozen input (context id) stays resolvable.
CREATE TABLE IF NOT EXISTS contract_predictions (
    prediction_id TEXT PRIMARY KEY,
    task_id       TEXT NOT NULL,
    episode_id    TEXT,
    created_at    REAL NOT NULL,
    payload       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_contract_preds_task
    ON contract_predictions(task_id);

-- World-model M4 (calibration retention): the CLOSE-OUT REGISTRY. One row
-- per closed (task, episode). It exists so the calibration WINDOW can be
-- located by an indexed ``closed_at`` ordering instead of scanning and
-- deserialising every stored evaluation, and so the idempotence of a
-- repeated close survives the archiving of the episode's detail: this
-- tombstone is tiny and stays ONLINE permanently, which is what makes
-- "archive the detail" safe (a re-close is still detected, and the window
-- is still locatable, without the payloads).
CREATE TABLE IF NOT EXISTS episode_closeouts (
    task_id        TEXT NOT NULL,
    episode_id     TEXT NOT NULL DEFAULT '',
    closed_at      REAL NOT NULL,
    terminal_state TEXT NOT NULL DEFAULT 'completed',
    evaluation_ids TEXT NOT NULL DEFAULT '[]',
    published      INTEGER NOT NULL DEFAULT 0,
    archived       INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (task_id, episode_id)
);
CREATE INDEX IF NOT EXISTS idx_closeouts_closed_at
    ON episode_closeouts(closed_at);
CREATE INDEX IF NOT EXISTS idx_closeouts_published
    ON episode_closeouts(published);

-- that table is read as strategy-outcome payloads by the budget ledger and
-- the query entry, so a capability record stored there would either
-- mis-deserialize as an OR prediction or silently disappear from the
-- ledger. Both readers stay honest by keeping the generations apart.
CREATE TABLE IF NOT EXISTS capability_predictions (
    prediction_id TEXT PRIMARY KEY,
    task_id       TEXT NOT NULL DEFAULT '',
    episode_id    TEXT,
    created_at    REAL NOT NULL,
    payload       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_capability_preds_task
    ON capability_predictions(task_id);
"""

#: Schema version marker (idempotent). Written once per store; M1 = "wm1",
#: M2 = "wm2" (adds world_model_predictions). The task_texts table reuses the
#: same IF NOT EXISTS mechanism, so no new version token is introduced —
#: the marker records the release that last touched the schema contract.
SCHEMA_VERSION_KEY = "schema_version"
SCHEMA_VERSION = "wm2"


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
            # Idempotent schema-version marker: old databases gain the new
            # tables via IF NOT EXISTS above; nothing is migrated or
            # rewritten. Re-running is always safe.
            self.conn.execute(
                "INSERT OR IGNORE INTO meta (key, value) VALUES (?,?)",
                (SCHEMA_VERSION_KEY, SCHEMA_VERSION))
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

    # -- task texts (retrieval source documents) -------------------------------

    def put_task_text(self, task_id: str, text: str,
                      text_digest: str) -> bool:
        """Record one task-text VERSION. Idempotent.

        ``INSERT OR IGNORE`` on the (task_id, text_digest) primary key: a
        version that is already stored is left EXACTLY as it is. Two
        executions that share a text digest therefore share one row, and
        re-running a command can never duplicate or rewrite a source
        document. Returns True when a new row was written.

        The digest is computed by the CALLER (``_stable_digest(task)``) so
        storage stays a plain substrate with no opinion about task shape.
        """
        with self.transaction() as conn:
            cur = conn.execute(
                "INSERT OR IGNORE INTO task_texts "
                "(text_digest, task_id, text, created_at) VALUES (?,?,?,?)",
                (str(text_digest), str(task_id), str(text), time.time()))
            return cur.rowcount > 0

    def get_task_text(self, task_id: str,
                      text_digest: str) -> Optional[str]:
        """The stored text of one version, or None when it is not retained
        (a legacy execution whose text was never captured). Never fabricates."""
        row = self.conn.execute(
            "SELECT text FROM task_texts WHERE task_id=? AND text_digest=?",
            (str(task_id), str(text_digest))).fetchone()
        return str(row["text"]) if row else None

    def task_texts_for(self, task_id: str) -> List[Dict[str, Any]]:
        """Every retained text version of one task (newest first).

        This is the documented look-up entry point for records that carry no
        vector: an unindexed execution is never silently dropped, it stays
        visible here (and via ``inspect``) by its ``task_text_digest``.
        """
        rows = self.conn.execute(
            "SELECT text_digest, text, created_at FROM task_texts "
            "WHERE task_id=? ORDER BY created_at DESC, text_digest ASC",
            (str(task_id),)).fetchall()
        return [{"text_digest": str(r["text_digest"]),
                 "text": str(r["text"]),
                 "created_at": float(r["created_at"])}
                for r in rows]

    def count_task_texts(self) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) AS n FROM task_texts").fetchone()
        return int(row["n"])

    def count_prediction_contexts(self) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) AS n FROM prediction_contexts").fetchone()
        return int(row["n"])

    # -- outcome predictions (M2, legacy protocol) -----------------------------

    def put_world_model_prediction(self, prediction_id: str, task_id: str,
                                   episode_id: Optional[str], payload: str,
                                   created_at: Optional[float] = None) -> None:
        """Persist one outcome prediction (a frozen hypothesis)."""
        with self.transaction() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO world_model_predictions "
                "(prediction_id, task_id, episode_id, created_at, payload) "
                "VALUES (?,?,?,?,?)",
                (str(prediction_id), str(task_id), episode_id,
                 float(created_at if created_at is not None else time.time()),
                 str(payload)))

    def get_world_model_prediction(self, prediction_id: str
                                   ) -> Optional[str]:
        """The stored payload of one outcome prediction, or None."""
        row = self.conn.execute(
            "SELECT payload FROM world_model_predictions "
            "WHERE prediction_id=?", (str(prediction_id),)).fetchone()
        return str(row["payload"]) if row else None

    def world_model_predictions_for(self, task_id: Optional[str] = None,
                                    episode_id: Optional[str] = None
                                    ) -> List[str]:
        """Stored outcome-prediction payloads (oldest first), filtered."""
        sql = "SELECT payload FROM world_model_predictions"
        params: List[Any] = []
        clauses: List[str] = []
        if task_id is not None:
            clauses.append("task_id=?")
            params.append(str(task_id))
        if episode_id is not None:
            clauses.append("episode_id=?")
            params.append(episode_id)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at ASC, prediction_id ASC"
        rows = self.conn.execute(sql, params).fetchall()
        return [str(r["payload"]) for r in rows]

    # -- prediction input contexts (frozen phase-2 inputs) ---------------------

    def put_prediction_context(self, context_id: str, task_id: str,
                               episode_id: Optional[str],
                               payload: str,
                               created_at: Optional[float] = None) -> None:
        """Persist one FROZEN prediction input context.

        ``INSERT OR REPLACE`` on the primary key is safe here (unlike the
        banks) because a context id is generated fresh per build: the replace
        only ever rewrites the SAME frozen object, and a caller that wants a
        different input must build a new context — which is exactly the
        versioning rule this phase requires.
        """
        with self.transaction() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO prediction_contexts "
                "(context_id, task_id, episode_id, created_at, payload) "
                "VALUES (?,?,?,?,?)",
                (str(context_id), str(task_id), episode_id,
                 float(created_at if created_at is not None else time.time()),
                 str(payload)))

    def get_prediction_context(self, context_id: str) -> Optional[str]:
        """The stored payload of one context, or None when it is unknown.

        Returns the RAW payload (a JSON string): storage stays a plain
        substrate and the schema version check belongs to the caller, which
        is the only layer that knows which versions it understands.
        """
        row = self.conn.execute(
            "SELECT payload FROM prediction_contexts WHERE context_id=?",
            (str(context_id),)).fetchone()
        return str(row["payload"]) if row else None

    def prediction_contexts_for(self, task_id: Optional[str] = None,
                                episode_id: Optional[str] = None
                                ) -> List[str]:
        """Stored context payloads (newest first), optionally filtered.

        An episode filter of ``None`` means "no episode filter" — an
        episode-less context is a real state, and asking for one task's
        contexts must not silently hide it.
        """
        sql = "SELECT payload FROM prediction_contexts"
        params: List[Any] = []
        clauses: List[str] = []
        if task_id is not None:
            clauses.append("task_id=?")
            params.append(str(task_id))
        if episode_id is not None:
            clauses.append("episode_id=?")
            params.append(episode_id)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at DESC, context_id DESC"
        rows = self.conn.execute(sql, params).fetchall()
        return [str(r["payload"]) for r in rows]

    # -- strategy-outcome contract predictions (M3, wm-so/1) ------------------

    def put_contract_prediction(self, prediction_id: str, task_id: str,
                                episode_id: Optional[str], payload: str,
                                created_at: Optional[float] = None) -> None:
        """Persist one strategy-outcome prediction (a frozen hypothesis)."""
        with self.transaction() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO contract_predictions "
                "(prediction_id, task_id, episode_id, created_at, payload) "
                "VALUES (?,?,?,?,?)",
                (str(prediction_id), str(task_id), episode_id,
                 float(created_at if created_at is not None else time.time()),
                 str(payload)))

    def get_contract_prediction(self, prediction_id: str) -> Optional[str]:
        """The stored payload of one contract prediction, or None."""
        row = self.conn.execute(
            "SELECT payload FROM contract_predictions "
            "WHERE prediction_id=?", (str(prediction_id),)).fetchone()
        return str(row["payload"]) if row else None

    def contract_predictions_for(self, task_id: Optional[str] = None,
                                 episode_id: Optional[str] = None
                                 ) -> List[str]:
        """Stored contract-prediction payloads (oldest first), filtered."""
        sql = "SELECT payload FROM contract_predictions"
        params: List[Any] = []
        clauses: List[str] = []
        if task_id is not None:
            clauses.append("task_id=?")
            params.append(str(task_id))
        if episode_id is not None:
            clauses.append("episode_id=?")
            params.append(episode_id)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at ASC, prediction_id ASC"
        rows = self.conn.execute(sql, params).fetchall()
        return [str(r["payload"]) for r in rows]

    def count_contract_predictions(self) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) AS n FROM contract_predictions").fetchone()
        return int(row["n"])

    # -- capability-evolution predictions (M5, wm-ce/1) -----------------------

    def put_capability_prediction(self, prediction_id: str, task_id: str,
                                  episode_id: Optional[str], payload: str,
                                  created_at: Optional[float] = None) -> None:
        """Persist one capability-evolution prediction (a frozen
        hypothesis about FUTURE performance, never a fact)."""
        with self.transaction() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO capability_predictions "
                "(prediction_id, task_id, episode_id, created_at, payload) "
                "VALUES (?,?,?,?,?)",
                (str(prediction_id), str(task_id or ""), episode_id,
                 float(created_at if created_at is not None else time.time()),
                 str(payload)))

    def get_capability_prediction(self, prediction_id: str
                                  ) -> Optional[str]:
        """The stored payload of one capability prediction, or None."""
        row = self.conn.execute(
            "SELECT payload FROM capability_predictions "
            "WHERE prediction_id=?", (str(prediction_id),)).fetchone()
        return str(row["payload"]) if row else None

    def capability_predictions_for(self, task_id: Optional[str] = None,
                                   episode_id: Optional[str] = None
                                   ) -> List[str]:
        """Stored capability-prediction payloads (oldest first)."""
        sql = "SELECT payload FROM capability_predictions"
        params: List[Any] = []
        clauses: List[str] = []
        if task_id is not None:
            clauses.append("task_id=?")
            params.append(str(task_id))
        if episode_id is not None:
            clauses.append("episode_id=?")
            params.append(episode_id)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at ASC, prediction_id ASC"
        rows = self.conn.execute(sql, params).fetchall()
        return [str(r["payload"]) for r in rows]

    def count_capability_predictions(self) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) AS n FROM capability_predictions").fetchone()
        return int(row["n"])

    # -- close-out registry (calibration window / retention) -------------------

    def put_closeout_registry(self, task_id: str, episode_id: Optional[str],
                              *, closed_at: float, terminal_state: str,
                              evaluation_ids: Sequence[str],
                              published: bool,
                              archived: bool = False) -> None:
        """Record (or refresh) one closed episode's registry row.

        ``INSERT OR REPLACE`` on the (task_id, episode_id) primary key: the
        row is DERIVED state about one episode's close, so re-writing it
        with the same identity is a refresh, never a duplicate. The row is
        deliberately tiny and is never archived — it is the tombstone that
        keeps a repeated close idempotent after the episode's detail has
        been moved to the archive.
        """
        with self.transaction() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO episode_closeouts "
                "(task_id, episode_id, closed_at, terminal_state, "
                " evaluation_ids, published, archived) "
                "VALUES (?,?,?,?,?,?,?)",
                (str(task_id), str(episode_id or ""), float(closed_at),
                 str(terminal_state), json.dumps(list(evaluation_ids)),
                 1 if published else 0, 1 if archived else 0))

    def get_closeout_registry(self, task_id: str,
                              episode_id: Optional[str]
                              ) -> Optional[Dict[str, Any]]:
        """One episode's registry row, or None when it was never closed."""
        row = self.conn.execute(
            "SELECT task_id, episode_id, closed_at, terminal_state, "
            "evaluation_ids, published, archived FROM episode_closeouts "
            "WHERE task_id=? AND episode_id=?",
            (str(task_id), str(episode_id or ""))).fetchone()
        if row is None:
            return None
        return {
            "task_id": str(row["task_id"]),
            "episode_id": row["episode_id"] or None,
            "closed_at": float(row["closed_at"]),
            "terminal_state": str(row["terminal_state"]),
            "evaluation_ids": list(json.loads(row["evaluation_ids"] or "[]")),
            "published": bool(row["published"]),
            "archived": bool(row["archived"]),
        }

    def closeout_registry(self, *, limit: Optional[int] = None,
                          offset: int = 0,
                          published: Optional[bool] = None,
                          archived: Optional[bool] = None,
                          before: Optional[float] = None
                          ) -> List[Dict[str, Any]]:
        """Registry rows NEWEST FIRST, by ``closed_at``.

        The ORDER BY ``closed_at`` uses the index, so the calibration window
        is located without touching any evaluation payload: the window is
        "the first N rows", not "every evaluation, filtered after reading".
        ``before`` bounds the rows to those closed strictly before a
        timestamp (used by the archive pass to find episodes past the grace
        period).
        """
        sql = ("SELECT task_id, episode_id, closed_at, terminal_state, "
               "evaluation_ids, published, archived FROM episode_closeouts")
        clauses: List[str] = []
        params: List[Any] = []
        if published is not None:
            clauses.append("published=?")
            params.append(1 if published else 0)
        if archived is not None:
            clauses.append("archived=?")
            params.append(1 if archived else 0)
        if before is not None:
            clauses.append("closed_at < ?")
            params.append(float(before))
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY closed_at DESC, task_id DESC, episode_id DESC"
        if limit is not None:
            sql += " LIMIT ? OFFSET ?"
            params.extend([int(limit), int(offset)])
        rows = self.conn.execute(sql, params).fetchall()
        return [{
            "task_id": str(r["task_id"]),
            "episode_id": r["episode_id"] or None,
            "closed_at": float(r["closed_at"]),
            "terminal_state": str(r["terminal_state"]),
            "evaluation_ids": list(json.loads(r["evaluation_ids"] or "[]")),
            "published": bool(r["published"]),
            "archived": bool(r["archived"]),
        } for r in rows]

    def count_closeouts(self, **kwargs: Any) -> int:
        """How many registry rows match (used for the auto-archive check)."""
        return len(self.closeout_registry(**kwargs))

    def mark_closeout_archived(self, task_id: str,
                               episode_id: Optional[str]) -> None:
        with self.transaction() as conn:
            conn.execute(
                "UPDATE episode_closeouts SET archived=1 "
                "WHERE task_id=? AND episode_id=?",
                (str(task_id), str(episode_id or "")))

    def set_closeout_published(self, task_id: str, episode_id: Optional[str],
                               published: bool) -> None:
        with self.transaction() as conn:
            conn.execute(
                "UPDATE episode_closeouts SET published=? "
                "WHERE task_id=? AND episode_id=?",
                (1 if published else 0, str(task_id),
                 str(episode_id or "")))

    def delete_evaluation(self, evaluation_id: str) -> None:
        """Remove one evaluation payload from the ONLINE store.

        Called only by the archive pass, AFTER the payload has been written
        to the archive file: the row is detail, not identity, and the
        registry tombstone keeps the episode locatable without it.
        """
        self.delete_evaluation_many([evaluation_id])

    def delete_evaluation_many(self, evaluation_ids: Sequence[str]) -> int:
        """Remove several evaluation payloads (archive pass only)."""
        ids = [str(e) for e in evaluation_ids]
        if not ids:
            return 0
        removed = 0
        with self.transaction() as conn:
            for chunk_start in range(0, len(ids), 400):
                chunk = ids[chunk_start:chunk_start + 400]
                placeholders = ",".join("?" for _ in chunk)
                cur = conn.execute(
                    f"DELETE FROM meta WHERE key IN ({placeholders})",
                    [f"strategy_evaluation|{e}" for e in chunk])
                removed += cur.rowcount
        return removed

    def delete_contract_predictions(self, prediction_ids: Sequence[str]
                                    ) -> int:
        """Remove strategy-outcome prediction payloads from the online store.

        Same rule as :meth:`delete_evaluation`: only the archive pass calls
        this, and only after the payloads are safely on disk.
        """
        ids = [str(p) for p in prediction_ids]
        if not ids:
            return 0
        removed = 0
        with self.transaction() as conn:
            for chunk_start in range(0, len(ids), 500):
                chunk = ids[chunk_start:chunk_start + 500]
                placeholders = ",".join("?" for _ in chunk)
                cur = conn.execute(
                    f"DELETE FROM contract_predictions WHERE prediction_id "
                    f"IN ({placeholders})", chunk)
                removed += cur.rowcount
        return removed

    def delete_prediction_contexts(self, context_ids: Sequence[str]) -> int:
        """Remove frozen-context payloads from the online store (archive
        pass only, after the payloads are on disk)."""
        ids = [str(c) for c in context_ids]
        if not ids:
            return 0
        removed = 0
        with self.transaction() as conn:
            for chunk_start in range(0, len(ids), 500):
                chunk = ids[chunk_start:chunk_start + 500]
                placeholders = ",".join("?" for _ in chunk)
                cur = conn.execute(
                    f"DELETE FROM prediction_contexts WHERE context_id "
                    f"IN ({placeholders})", chunk)
                removed += cur.rowcount
        return removed

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
