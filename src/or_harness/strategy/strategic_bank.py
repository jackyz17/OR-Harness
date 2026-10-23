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

#: The lifecycle actions an AGENT may request through an effect attribution.
#: They are ADMINISTRATIVE requests: the agent states, from the accumulated
#: USE evidence and its own reasoning, how the entry's maturity should move.
#: ``promote`` -> validated, ``demote`` -> suspect, ``keep`` -> leave the
#: status as it is (so a prior management decision is not silently undone).
#: Promotion is NEVER automatic and NEVER driven by adoption counts or a
#: prediction hit rate — the agent must compare premises, operation, effect,
#: cross-task support and boundary, and the framework only checks that the
#: request names real adopted evidence.
LIFECYCLE_ACTIONS = ("promote", "demote", "keep")

#: The verdicts an AGENT may attach to how an entry's USE turned out. These
#: are the agent's own attributions (never the framework's): ``helped`` /
#: ``neutral`` / ``unrelated`` are REPORTED, and ``refuting`` is a NEGATIVE
#: judgement that the agent may attach (it no longer moves lifecycle state by
#: itself — a management request does). No new score or threshold is
#: introduced.
EFFECT_ATTRIBUTION_VERDICTS = ("helped", "neutral", "unrelated", "refuting")


def apply_transitions(entry: StrategicEntry) -> List[str]:
    """The ONLY automatic lifecycle transition: dormancy wakeup.

    r21: the automatic rules that changed an entry's MATURITY from prediction
    calibration (``candidate --(n>=5 & hit_rate>=0.7)--> validated`` and
    ``any hot --(3 consecutive misses)--> suspect``) are REMOVED. Prediction
    calibration measures how a DECLARED forecast fared; it is NOT a measure of
    whether USING the knowledge helped, and the two meanings are kept apart.
    Calibration is now a DIAGNOSTIC (replayed and reported by ``revise``), and
    maturity changes only by the AGENT's explicit management request
    (``lifecycle_action`` on an effect attribution —
    :meth:`StrategicBank.apply_lifecycle_action`).

    ``dormant -> candidate`` wakeup stays automatic: a matching execution is
    evidence the pattern is live again, and waking is not a promotion.

    Retirement is never automatic: it stays the harness's explicit call via
    :meth:`StrategicBank.retire`.
    """
    transitions: List[str] = []
    if entry.status == "dormant":
        entry.status = "candidate"
        transitions.append("awakened:dormant->candidate")
    return transitions



class StrategicBank:
    """CRUD + lifecycle for :class:`StrategicEntry`, plus the cold archive."""

    def __init__(self, store: Store):
        self.store = store

    # -- CRUD --------------------------------------------------------------------

    def add(self, entry: StrategicEntry) -> str:
        """Add a new entry, ASSIGNING it the next number.

        The framework owns the entry's number: it is ``MAX(existing)+1`` over
        the whole table (including rows written earlier and later archived),
        so numbers are strictly increasing and are NEVER reused or
        resequenced. A caller-supplied ``entry_id`` is IGNORED — the agent
        cannot invent an identity, and no ``se_``-style name is accepted. The
        returned id is the assigned decimal string.
        """
        with self.store.transaction() as conn:
            row = conn.execute(
                "SELECT MAX(CAST(entry_id AS INTEGER)) AS m "
                "FROM strategic_entries").fetchone()
            next_number = int(row["m"] or 0) + 1
            # The cold archive also holds retired numbers: an archived number
            # is never handed out again, so the sequence spans both tables.
            try:
                crow = conn.execute(
                    "SELECT MAX(entry_number) AS m FROM cold_archive"
                ).fetchone()
                if crow is not None and crow["m"] is not None:
                    next_number = max(next_number, int(crow["m"]) + 1)
            except Exception:  # noqa: BLE001 - older table without the column
                pass
            entry.entry_id = str(next_number)
            self._validate(entry)
            conn.execute(
                "INSERT INTO strategic_entries "
                "(entry_id, strategy_id, scope_level, status, payload) "
                "VALUES (?,?,?,?,?)",
                (entry.entry_id, entry.strategy_id, self._scope_token(entry),
                 entry.status, self.store.dumps(entry.to_dict())))
            from or_harness.world_model.conditions import record_availability
            record_availability(conn, self.store, entry.entry_id, entry.predicates)
        return entry.entry_id

    def next_number(self) -> int:
        """The number the next :meth:`add` will assign (read-only preview)."""
        row = self.store.conn.execute(
            "SELECT MAX(CAST(entry_id AS INTEGER)) AS m "
            "FROM strategic_entries").fetchone()
        highest = int(row["m"] or 0)
        try:
            crow = self.store.conn.execute(
                "SELECT MAX(entry_number) AS m FROM cold_archive").fetchone()
            if crow is not None and crow["m"] is not None:
                highest = max(highest, int(crow["m"]))
        except Exception:  # noqa: BLE001 - older table without the column
            pass
        return highest + 1

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
            from or_harness.world_model.conditions import record_availability
            record_availability(conn, self.store, entry.entry_id, entry.predicates)

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

        r21: this records the CALIBRATION sample only. It does NOT change the
        entry's maturity — a hit rate is a diagnostic of a declared forecast,
        not a measure of whether USING the knowledge helped, and the two are
        no longer bound. The only automatic transition left is the dormancy
        wakeup (:func:`apply_transitions`).

        The automatic record chain no longer calls it: recording writes the
        frozen check into the EXECUTION evidence instead (``quality_feedback``
        on the fact) and ``InductionEngine.revise`` replays a whole batch of
        checks at the next offline induction.

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

    def apply_lifecycle_action(self, entry_id: str, action: str, *,
                               reason: Optional[str] = None
                               ) -> Tuple[StrategicEntry, List[str]]:
        """Apply ONE explicit AGENT management request to an entry's status.

        The maturity counterpart of ``record_prediction`` (which only records
        calibration). ``action`` is one of :data:`LIFECYCLE_ACTIONS`:

        - ``promote`` -> ``validated`` (``candidate``/``suspect`` only;
          already-``validated`` is a no-op; ``dormant`` is awakened through
          the EXISTING wakeup, never resurrected from the cold archive);
        - ``demote`` -> ``suspect`` (unless already ``dormant``/retired);
        - ``keep`` -> status unchanged.

        The framework REQUIRES a positive reason (``reason``) and performs no
        logical certification: whether the accumulated USE evidence supports
        the move is the agent's judgement. It does not rewrite
        ``verification_state`` and does not touch the claim, number, evidence
        or prediction track.
        """
        if action not in LIFECYCLE_ACTIONS:
            raise StorageError(
                f"illegal lifecycle action {action!r} "
                f"(expected one of {list(LIFECYCLE_ACTIONS)})")
        if not (reason and str(reason).strip()):
            raise StorageError(
                f"lifecycle action {action!r} requires a non-empty reason")
        entry = self.get(entry_id)
        if entry is None:
            raise StorageError(f"unknown entry_id {entry_id!r}")
        transitions: List[str] = []
        if entry.status == "dormant":
            # Reuse the existing wakeup; a dormant entry is not promoted from
            # a stale state in one step (the caller re-issues the request).
            transitions.extend(apply_transitions(entry))
        if action == "promote":
            if entry.status != "validated":
                transitions.append(f"promoted:->validated "
                                   f"(agent: {str(reason).strip()})")
                entry.status = "validated"
        elif action == "demote":
            if entry.status != "suspect":
                transitions.append(f"demoted:->suspect "
                                   f"(agent: {str(reason).strip()})")
                entry.status = "suspect"
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
            entry_id=entry.entry_id,
            evidence_summary={
                "support_n": entry.support_n,
                "hit_rate": round(entry.prediction_track.hit_rate, 4),
                "n_predictions": entry.prediction_track.n_predictions,
                "predicates": entry.predicates,
                "provenance": list(entry.provenance)[:10],
            },
        )
        with self.store.transaction() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO cold_archive "
                "(pattern_hash, entry_number, payload) VALUES (?,?,?)",
                (card.pattern_hash,
                 entry.entry_number if entry.entry_number else None,
                 self.store.dumps(card.to_dict())))
            conn.execute("DELETE FROM strategic_entries WHERE entry_id=?",
                         (entry_id,))
            from or_harness.world_model.conditions import record_availability
            record_availability(conn, self.store, entry_id, entry.predicates,
                                present=False)
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
        # Invariant: a REFUTED claim may never be `validated`. This is about
        # the CLAIM's verification, NOT about knowledge utility: an
        # assertion failing is not the agent judging the knowledge useless,
        # and the two meanings are kept apart. Management (promote/demote)
        # never rewrites ``verification_state`` to force a promotion through;
        # only an actual refuted state blocks ``validated``.
        state = entry.verification_state
        if state == "refuted" and entry.status == "validated":
            raise StorageError("a refuted claim may never be 'validated'")
        # An agent use-effect attribution must be a KNOWN verdict, and the
        # agent must be its author (the framework never fabricates one). An
        # optional management request must be a KNOWN action.
        for attribution in (entry.effect_attribution or []):
            verdict = str(attribution.get("verdict") or "")
            if verdict not in EFFECT_ATTRIBUTION_VERDICTS:
                raise StorageError(
                    f"illegal effect_attribution verdict {verdict!r} "
                    f"(expected one of {list(EFFECT_ATTRIBUTION_VERDICTS)})")
            if str(attribution.get("by") or "") != "agent":
                raise StorageError(
                    "effect_attribution must be authored by the agent "
                    f"(by='agent'), got {attribution.get('by')!r}")
            action = attribution.get("lifecycle_action")
            if action is not None and str(action) not in LIFECYCLE_ACTIONS:
                raise StorageError(
                    f"illegal effect_attribution lifecycle_action {action!r} "
                    f"(expected one of {list(LIFECYCLE_ACTIONS)})")

    @staticmethod
    def _decode(row: Any) -> StrategicEntry:
        try:
            return StrategicEntry.from_dict(Store.loads(row["payload"]))
        except (StorageError, ValueError, KeyError) as exc:
            raise StorageError(f"corrupt strategic entry: {exc}") from exc
