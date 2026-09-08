"""Strategic Bank: the derived layer of calibrated commitments.

Entries are claims about the future — prediction intervals, calibration
tracking, cross-group feature predicates. They are rebuildable from the
Experience Bank at any time (``induce --rebuild``), and they alone carry the
disposal ladder:

  candidate --(n>=5 & hit_rate>=0.7)--> validated
  any hot   --(3 consecutive misses)--> suspect    (score x0.5, reversible)
  suspect   --(harness confirms)-----> retired -> cold archive (leaves hot store)
  any hot   --(10 tasks unconsulted)-> dormant   (selector-excluded, reversible)

The cold archive is the anti-resurrection mechanism: before inducting a new
entry, matching tombstones veto re-creating the same failed generalization
from the same evidence. ``--force`` revives only when the environment has
genuinely drifted (harness's explicit call).
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from or_harness.core.schema import (
    ENTRY_STATUSES,
    ColdArchiveCard,
    StrategicEntry,
    pattern_hash,
)
from or_harness.core.storage import Store, StorageError

PROMOTE_MIN_PREDICTIONS = 5
PROMOTE_MIN_HIT_RATE = 0.7
DEMOTE_CONSECUTIVE_MISSES = 3
DORMANT_AFTER_TASKS = 10
SUSPECT_SCORE_FACTOR = 0.5


class StrategicBank:
    """CRUD + lifecycle for :class:`StrategicEntry`, plus the cold archive."""

    def __init__(self, store: Store):
        self.store = store

    # -- CRUD --------------------------------------------------------------------

    def add(self, entry: StrategicEntry) -> str:
        self._validate(entry)
        with self.store.transaction() as conn:
            conn.execute(
                "INSERT INTO strategic_entries "
                "(entry_id, strategy_id, scope_level, status, payload) "
                "VALUES (?,?,?,?,?)",
                (entry.entry_id, entry.strategy_id, entry.scope_level,
                 entry.status, self.store.dumps(entry.to_dict())))
        return entry.entry_id

    def update(self, entry: StrategicEntry) -> None:
        self._validate(entry)
        with self.store.transaction() as conn:
            cur = conn.execute(
                "UPDATE strategic_entries SET strategy_id=?, scope_level=?, "
                "status=?, payload=? WHERE entry_id=?",
                (entry.strategy_id, entry.scope_level, entry.status,
                 self.store.dumps(entry.to_dict()), entry.entry_id))
            if cur.rowcount == 0:
                raise StorageError(f"unknown entry_id {entry.entry_id!r}")

    def get(self, entry_id: str) -> Optional[StrategicEntry]:
        row = self.store.conn.execute(
            "SELECT payload FROM strategic_entries WHERE entry_id=?",
            (entry_id,)).fetchone()
        return self._decode(row) if row else None

    def remove(self, entry_id: str) -> None:
        with self.store.transaction() as conn:
            cur = conn.execute("DELETE FROM strategic_entries WHERE entry_id=?",
                               (entry_id,))
            if cur.rowcount == 0:
                raise StorageError(f"unknown entry_id {entry.entry_id!r}")

    def list(self, *, status: Optional[str] = None,
             strategy_id: Optional[str] = None,
             include_dormant: bool = True) -> List[StrategicEntry]:
        sql = "SELECT payload FROM strategic_entries"
        clauses, params = [], []
        if status is not None:
            clauses.append("status=?"); params.append(status)
        if strategy_id is not None:
            clauses.append("strategy_id=?"); params.append(strategy_id)
        if not include_dormant and status is None:
            clauses.append("status!='dormant'")
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY rowid ASC"
        return [self._decode(r) for r in
                self.store.conn.execute(sql, params).fetchall()]

    def count(self) -> int:
        return int(self.store.conn.execute(
            "SELECT COUNT(*) AS n FROM strategic_entries").fetchone()["n"])

    def matching(self, profile) -> List[StrategicEntry]:
        """Hot (non-dormant) entries whose predicates match the profile."""
        return [e for e in self.list(include_dormant=False) if e.matches(profile)]

    def mark_consulted(self, entry_ids: List[str], at: Optional[float] = None) -> None:
        at = time.time() if at is None else at
        with self.store.transaction() as conn:
            for entry_id in entry_ids:
                row = conn.execute(
                    "SELECT payload FROM strategic_entries WHERE entry_id=?",
                    (entry_id,)).fetchone()
                if not row:
                    continue
                entry = StrategicEntry.from_dict(Store.loads(row["payload"]))
                entry.last_consulted_at = at
                conn.execute("UPDATE strategic_entries SET payload=? WHERE entry_id=?",
                             (self.store.dumps(entry.to_dict()), entry_id))

    # -- lifecycle -----------------------------------------------------------------

    def record_prediction(self, entry_id: str, hit: bool,
                          calibration_err: float = 0.0,
                          latest_task_at: Optional[float] = None
                          ) -> Tuple[StrategicEntry, List[str]]:
        """Forward validation: an execution checked this entry's prediction.

        Applies the automatic transitions (promotion, demotion, dormancy
        wakeup) and returns the updated entry plus the transitions taken.
        Retirement is *not* automatic — the harness confirms irreversible
        disposal explicitly via :meth:`retire`.
        """
        entry = self.get(entry_id)
        if entry is None:
            raise StorageError(f"unknown entry_id {entry_id!r}")
        transitions: List[str] = []
        entry.prediction_track.record(hit, calibration_err)
        if entry.status == "dormant":
            entry.status = "candidate"
            transitions.append("awakened:dormant->candidate")
        if (entry.status == "candidate"
                and entry.prediction_track.n_predictions >= PROMOTE_MIN_PREDICTIONS
                and entry.prediction_track.hit_rate >= PROMOTE_MIN_HIT_RATE):
            entry.status = "validated"
            transitions.append("promoted:candidate->validated")
        if (entry.status in ("candidate", "validated")
                and entry.prediction_track.consecutive_misses >= DEMOTE_CONSECUTIVE_MISSES):
            entry.status = "suspect"
            transitions.append(f"demoted:->{entry.status}")
        self.update(entry)
        return entry, transitions

    def retire(self, entry_id: str, reason: str,
               outcome: str = "retired") -> ColdArchiveCard:
        """Harness-confirmed irreversible disposal into the cold archive."""
        entry = self.get(entry_id)
        if entry is None:
            raise StorageError(f"unknown entry_id {entry_id!r}")
        card = ColdArchiveCard(
            pattern_hash=pattern_hash(entry.predicates, entry.strategy_id),
            strategy_id=entry.strategy_id,
            predicates=entry.predicates,
            outcome=outcome,
            reason=reason,
            evidence_summary={
                "support_n": entry.support_n,
                "hit_rate": round(entry.prediction_track.hit_rate, 4),
                "n_predictions": entry.prediction_track.n_predictions,
                "scope_level": entry.scope_level,
                "provenance": list(entry.provenance)[:10],
            },
        )
        with self.store.transaction() as conn:
            conn.execute("INSERT OR REPLACE INTO cold_archive (pattern_hash, payload) "
                         "VALUES (?,?)",
                         (card.pattern_hash, self.store.dumps(card.to_dict())))
            conn.execute("DELETE FROM strategic_entries WHERE entry_id=?",
                         (entry_id,))
        return card

    def age(self, dormant_after: int = DORMANT_AFTER_TASKS) -> List[str]:
        """Mark entries dormant after ``dormant_after`` tasks unconsulted.

        Recency window: the N most recent task timestamps from the Experience
        Bank. Entries whose last consultation (or creation) predates the
        window go dormant. A matching future execution wakes them again via
        :meth:`record_prediction`. Returns affected entry ids."""
        rows = self.store.conn.execute(
            "SELECT DISTINCT created_at FROM executions ORDER BY created_at DESC"
        ).fetchall()
        if len(rows) < dormant_after:
            return []
        threshold = float(rows[dormant_after - 1]["created_at"])
        affected: List[str] = []
        for entry in self.list():
            if entry.status == "dormant":
                continue
            seen = entry.last_consulted_at or entry.created_at
            if seen < threshold:
                entry.status = "dormant"
                self.update(entry)
                affected.append(entry.entry_id)
        return affected

    # -- cold archive ----------------------------------------------------------------

    def cold_archive(self) -> List[ColdArchiveCard]:
        rows = self.store.conn.execute(
            "SELECT payload FROM cold_archive ORDER BY rowid ASC").fetchall()
        return [ColdArchiveCard.from_dict(Store.loads(r["payload"])) for r in rows]

    def archive_vetoes(self, strategy_id: str,
                       predicates: Dict[str, Any]) -> Optional[ColdArchiveCard]:
        """Anti-resurrection check: an archived tombstone with the same
        (strategy, predicates) hash blocks re-induction of the same failed
        generalization."""
        digest = pattern_hash(predicates, strategy_id)
        row = self.store.conn.execute(
            "SELECT payload FROM cold_archive WHERE pattern_hash=?",
            (digest,)).fetchone()
        return ColdArchiveCard.from_dict(Store.loads(row["payload"])) if row else None

    def revive(self, pattern_digest: str, force: bool = False) -> None:
        """Remove a tombstone so induction may retry. Requires ``force`` —
        the escape hatch for genuine environment drift, exercised explicitly
        by the harness."""
        if not force:
            raise StorageError(
                "cold-archive revival requires force=True: reviving a vetoed "
                "generalization is the harness's explicit call")
        with self.store.transaction() as conn:
            cur = conn.execute("DELETE FROM cold_archive WHERE pattern_hash=?",
                               (pattern_digest,))
            if cur.rowcount == 0:
                raise StorageError(f"no cold-archive card {pattern_digest!r}")

    # -- internals -----------------------------------------------------------------

    @staticmethod
    def _validate(entry: StrategicEntry) -> None:
        if entry.status not in ENTRY_STATUSES:
            raise StorageError(f"illegal entry status {entry.status!r}")
        lo, hi = entry.quality_interval
        if not (0.0 <= lo <= hi <= 1.0):
            raise StorageError("quality_interval must satisfy 0 <= lo <= hi <= 1")

    @staticmethod
    def _decode(row: Any) -> StrategicEntry:
        try:
            return StrategicEntry.from_dict(Store.loads(row["payload"]))
        except (StorageError, ValueError, KeyError) as exc:
            raise StorageError(f"corrupt strategic entry: {exc}") from exc
