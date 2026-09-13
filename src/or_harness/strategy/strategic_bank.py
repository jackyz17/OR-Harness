"""Strategic Knowledge Bank: the derived layer of calibrated commitments.

Derived knowledge, not primary facts: "what to do next time" — expected
quality, expected cost, expected failure risk. Mutation happens at
INDUCTION time only: online execution records evidence (the frozen quality
checks live on the facts, see
``ExecutionRecord.execution_features.quality_feedback``) and never touches
an entry; ``orx induce`` re-derives lifecycle state from that evidence
(``InductionEngine.revise``). Entries keep lightweight origin metadata
(``provenance`` / ``support_n``) and their continued validity does NOT
depend on the survival of the original evidence rows. Entries alone carry
the disposal ladder:

  candidate --(n>=5 & hit_rate>=0.7)--> validated
  any hot   --(3 consecutive misses)--> suspect    (score x0.5, reversible)
  suspect   --(harness confirms)-----> retired -> cold archive (leaves hot store)
  any hot   --(10 tasks unconsulted)-> dormant   (selector-excluded, reversible)

Mutability: derived beliefs may be re-estimated, validated, revised,
deprecated, and replaced — unlike facts, which are never rewritten.

The cold archive is the anti-resurrection mechanism: before inducting a new
entry, matching cards veto re-creating the same failed generalization from
the same evidence. ``orx induce --force`` REMOVES the card for that pattern
(the harness's explicit "the environment drifted" call), so the judgment is
made once instead of being repeated on every induction.
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


def apply_transitions(entry: StrategicEntry) -> List[str]:
    """Automatic lifecycle transitions for the entry's CURRENT track.

    One definition, two callers: the per-event API
    (:meth:`StrategicBank.record_prediction`) and the offline revalidation
    pass (``InductionEngine.revise``) — so promotion, demotion, and dormancy
    wakeup can never drift apart between the two.

    Retirement is never automatic: it stays the harness's explicit call via
    :meth:`StrategicBank.retire`.
    """
    transitions: List[str] = []
    if entry.status == "dormant":
        entry.status = "candidate"
        transitions.append("awakened:dormant->candidate")
    if (entry.status == "candidate"
            and entry.prediction_track.n_predictions >= PROMOTE_MIN_PREDICTIONS
            and entry.prediction_track.hit_rate >= PROMOTE_MIN_HIT_RATE
            and entry.verification_state == "verified"):
        # Forward calibration alone never promotes: the claim itself must
        # have passed admission verification. (Demotion below is unaffected —
        # observed misses are evidence regardless of admission state.)
        entry.status = "validated"
        transitions.append("promoted:candidate->validated")
    if (entry.status in ("candidate", "validated")
            and entry.prediction_track.consecutive_misses >= DEMOTE_CONSECUTIVE_MISSES):
        entry.status = "suspect"
        transitions.append(f"demoted:->{entry.status}")
    return transitions


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
                (entry.entry_id, entry.strategy_id, self._scope_token(entry),
                 entry.status, self.store.dumps(entry.to_dict())))
        return entry.entry_id

    def update(self, entry: StrategicEntry) -> None:
        self._validate(entry)
        with self.store.transaction() as conn:
            cur = conn.execute(
                "UPDATE strategic_entries SET strategy_id=?, scope_level=?, "
                "status=?, payload=? WHERE entry_id=?",
                (entry.strategy_id, self._scope_token(entry), entry.status,
                 self.store.dumps(entry.to_dict()), entry.entry_id))
            if cur.rowcount == 0:
                raise StorageError(f"unknown entry_id {entry.entry_id!r}")

    def get(self, entry_id: str) -> Optional[StrategicEntry]:
        row = self.store.conn.execute(
            "SELECT payload FROM strategic_entries WHERE entry_id=?",
            (entry_id,)).fetchone()
        return self._decode(row) if row else None


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

    def matching(self, profile, *,
                 include_dormant: bool = False) -> List[StrategicEntry]:
        """Entries whose predicates match the profile.

        ``include_dormant=False`` (default) is the RETRIEVAL view: dormant
        entries are not consulted. ``include_dormant=True`` is the EVIDENCE
        view used when recording: a matching execution is a fact about the
        pattern regardless of whether the entry is currently consulted, and
        waking it back up is an offline induction decision."""
        return [e for e in self.list(include_dormant=include_dormant)
                if e.matches(profile)]

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
        """Record ONE forward check against this entry's prediction.

        This is the per-event API of the lifecycle. The automatic record
        chain no longer calls it: recording writes the frozen check into the
        EXECUTION evidence instead (``quality_feedback`` on the fact) and
        ``InductionEngine.revise`` replays a whole batch of checks at the next
        offline induction. Both callers apply exactly the same transition
        rules (:func:`apply_transitions`).

        Retirement is *not* automatic — the harness confirms irreversible
        disposal explicitly via :meth:`retire`.
        """
        entry = self.get(entry_id)
        if entry is None:
            raise StorageError(f"unknown entry_id {entry_id!r}")
        entry.prediction_track.record(hit, calibration_err)
        transitions = apply_transitions(entry)
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
                "predicates": entry.predicates,
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
        """Anti-resurrection check: a cold-archive card with the same
        (strategy, predicates) hash blocks re-induction of the same failed
        generalization."""
        digest = pattern_hash(predicates, strategy_id)
        row = self.store.conn.execute(
            "SELECT payload FROM cold_archive WHERE pattern_hash=?",
            (digest,)).fetchone()
        return ColdArchiveCard.from_dict(Store.loads(row["payload"])) if row else None

    def revive(self, pattern_digest: str, force: bool = False) -> None:
        """Remove a cold-archive card so induction may re-create the pattern.
        Requires ``force`` — lifting a veto is the harness's explicit call
        (the ``--force`` flag on ``orx induce`` does exactly this)."""
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
    def _scope_token(entry: StrategicEntry) -> str:
        """Index token for the entry's applicability: the family it names,
        or ``*`` for a family-free pattern. Stored in the legacy
        ``scope_level`` column (kept for on-disk compatibility)."""
        return str(entry.predicates.get("family", "*"))

    @staticmethod
    def _validate(entry: StrategicEntry) -> None:
        if entry.status not in ENTRY_STATUSES:
            raise StorageError(f"illegal entry status {entry.status!r}")
        lo, hi = entry.quality_interval
        if not (0.0 <= lo <= hi <= 1.0):
            raise StorageError("quality_interval must satisfy 0 <= lo <= hi <= 1")
        # Admission invariant: a validated entry must have PASSED admission
        # verification, and a refuted claim may never be validated. Forward
        # calibration (n>=5 checks, hit rate) tracks how the entry's
        # predictions fared — it can never substitute for verifying the
        # claim itself.
        state = entry.verification_state
        if entry.status == "validated" and state != "verified":
            raise StorageError(
                "status='validated' requires verification.state='verified' "
                f"(got {state!r})")
        if state == "refuted" and entry.status == "validated":
            raise StorageError("a refuted claim may never be 'validated'")

    @staticmethod
    def _decode(row: Any) -> StrategicEntry:
        try:
            return StrategicEntry.from_dict(Store.loads(row["payload"]))
        except (StorageError, ValueError, KeyError) as exc:
            raise StorageError(f"corrupt strategic entry: {exc}") from exc
