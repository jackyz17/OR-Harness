"""Execution Evidence Bank: the append-only episodic fact layer.

Records what actually happened — never what will happen. Facts are permanently
neutral: the disposal ladder (suspect/dormant/retired/cold archive) applies
only to the derived Strategic Knowledge Bank. The Evidence Bank is the single
source of truth. NOTE (target semantics, next Induction migration round):
"Strategic Knowledge is induced AND validated against it at induction time"
is the migration TARGET — today entries are born candidate and promoted
online by forward quality checks; admission never depends on the survival of
any particular evidence row (``induce --rebuild`` re-induces from whatever
evidence is currently retained).

Mutability contract (fact-preserving, append-first):
  - ``append``: the only way a new fact enters. Duplicate ids are rejected.
  - ``update_cost``: the sole backfill channel (e.g. llm_tokens becomes known
    later). Only cost dimensions may change; nothing else is ever rewritten.
    Backfill is EXPLICITLY REPLACEMENT by default (idempotent: re-applying
    the same measurement never double-counts); ``increment`` mode exists for
    harness-owned counters delivered in parts within the attempt's declared
    scope. Backfilled dimensions become measured. Any persisted cost
    feedback is re-computed against the amended value so the stored summary
    can never disagree with the fact (no stale actual=110 vs 1000).
  - ``set_cost_feedback``: the only channel that writes the record's cost
    feedback annotation (a computed summary over the frozen prediction
    snapshot and the actual cost). Narrow by design — it cannot rewrite any
    other execution feature.
  - ``stage_pending`` / ``clear_pending``: a no-lost-facts safety net between
    execution and the harness's explicit recording decision.

Lossy compaction is deferred until the summary consumption contract exists
(statistics and induction ignore ``source="compacted"`` rows); when it
arrives it will need its own replacement channel. No such channel is kept
around unused today.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from or_harness.core.schema import (
    COST_DIMENSIONS,
    ExecutionRecord,
    compute_cost_feedback,
    group_key,
)
from or_harness.core.storage import Store, StorageError


class ExperienceBank:
    """Append-only store of :class:`ExecutionRecord` facts (the Execution
    Evidence layer). One fact = one episode of what actually happened:
    actual strategy, actual quality, actual cost, observed failures,
    implementation artifacts."""

    def __init__(self, store: Store):
        self.store = store

    # -- writes -----------------------------------------------------------------

    def append(self, record: ExecutionRecord) -> str:
        """Append a fact. Raises on duplicate id or malformed payload."""
        # Validate by round-trip before persisting (malformed rejection).
        payload = ExecutionRecord.from_dict(record.to_dict())
        with self.store.transaction() as conn:
            try:
                conn.execute(
                    "INSERT INTO executions "
                    "(execution_id, task_id, strategy_id, family, group_l1, "
                    " source, created_at, payload) VALUES (?,?,?,?,?,?,?,?)",
                    (
                        payload.execution_id,
                        payload.task_id,
                        payload.strategy_id,
                        payload.profile_snapshot.family,
                        payload.group_l1,
                        payload.source,
                        payload.created_at,
                        self.store.dumps(payload.to_dict()),
                    ),
                )
            except Exception as exc:
                if "UNIQUE" in str(exc).upper():
                    raise StorageError(
                        f"duplicate execution_id {payload.execution_id!r}: "
                        "the Experience Bank is append-only") from exc
                raise
        return payload.execution_id

    def update_cost(self, execution_id: str, *,
                    mode: str = "replace", **dimensions: float) -> ExecutionRecord:
        """Backfill cost dimensions (harness-owned llm_tokens via --override).

        Appends nothing: this amends the fact's measured fields in place, which
        is the documented supplement channel. Only cost dimensions may change.

        ``mode`` makes the accounting explicit so nothing is double-counted:
        - ``replace`` (default): the value IS the measurement. Re-applying
          the same override is idempotent — the same tokens can never be
          added twice.
        - ``increment``: the value is an additional measured amount within
          the record's declared scope.

        Backfilled dimensions are marked measured. Cost feedback is
        re-computed against the frozen prediction snapshot whenever a
        snapshot exists: the stored summary never disagrees with the stored
        fact, and a dimension that was unknown at record time produces
        feedback as soon as it becomes comparable.
        """
        if mode not in ("replace", "increment"):
            raise StorageError(f"unknown backfill mode {mode!r} "
                               "(expected 'replace' or 'increment')")
        unknown = set(dimensions) - set(COST_DIMENSIONS)
        if unknown:
            raise StorageError(f"unknown cost dimensions: {sorted(unknown)}")
        rec = self.get(execution_id)
        if rec is None:
            raise StorageError(f"unknown execution_id {execution_id!r}")
        # A declared tool_calls below the sandbox's provable floor is not a
        # measurement, it is a contradiction: the executor demonstrably made
        # at least that many calls. Refuse it rather than store a number the
        # record itself refutes.
        floor = rec.execution_features.get("tool_calls_lower_bound")
        if "tool_calls" in dimensions and floor is not None:
            declared = float(dimensions["tool_calls"])
            if declared < float(floor):
                raise StorageError(
                    f"tool_calls={declared} is below the provable lower bound "
                    f"({floor}): the executor itself ran the solve script at "
                    "least that many times. tool_calls counts ALL tool "
                    "invocations in the attempt's scope (shell commands, file "
                    "reads/writes, sandbox runs, solver calls)")
        for d, v in dimensions.items():
            if mode == "increment":
                setattr(rec.cost, d, float(getattr(rec.cost, d)) + float(v))
            else:
                setattr(rec.cost, d, float(v))
        rec.cost.mark_measured(*dimensions)
        # Recompute feedback whenever a prediction snapshot exists — not
        # only when feedback was already stored. A dimension that was
        # UNKNOWN at record time (so no feedback could be computed then)
        # becomes comparable once it is backfilled, and must produce
        # feedback now. No snapshot -> nothing to compare -> untouched.
        if rec.prediction_snapshot is not None:
            feedback = compute_cost_feedback(rec.prediction_snapshot,
                                             rec.strategy_id,
                                             rec.measurement_scope,
                                             rec.cost)
            if feedback is None:
                rec.execution_features.pop("cost_feedback", None)
            else:
                rec.execution_features["cost_feedback"] = feedback
        with self.store.transaction() as conn:
            conn.execute("UPDATE executions SET payload=? WHERE execution_id=?",
                         (self.store.dumps(rec.to_dict()), execution_id))
        return rec

    def set_cost_feedback(self, execution_id: str,
                          feedback: Optional[Dict[str, Any]]) -> ExecutionRecord:
        """Write (or remove) the record's cost feedback annotation.

        A narrow, single-purpose channel: it only ever touches the
        ``cost_feedback`` key inside ``execution_features`` (a computed
        summary over the frozen prediction snapshot and the actual cost).
        It cannot rewrite any other execution feature, historical fact, or
        derived-layer state.
        """
        rec = self.get(execution_id)
        if rec is None:
            raise StorageError(f"unknown execution_id {execution_id!r}")
        if feedback is None:
            rec.execution_features.pop("cost_feedback", None)
        else:
            rec.execution_features["cost_feedback"] = dict(feedback)
        with self.store.transaction() as conn:
            conn.execute("UPDATE executions SET payload=? WHERE execution_id=?",
                         (self.store.dumps(rec.to_dict()), execution_id))
        return rec

    def exclude(self, execution_id: str, reason: str, *,
                superseded_by: Optional[str] = None) -> ExecutionRecord:
        """Withdraw a fact from the evidence set WITHOUT deleting it.

        The Evidence Bank is append-only, so a wrong observation cannot be
        erased — but it must stop counting. This is the explicit correction
        channel: the row is preserved for audit, its ``source`` becomes
        ``"excluded"`` (every statistics / induction / trigger / retrieval
        path requires ``source == "executed"``, so the fact drops out of
        all of them at once), and the reason is recorded on the fact under
        ``execution_features.correction``.

        ``superseded_by`` names the execution that replaces it (e.g. a
        re-run with a corrected answer) — a link, never an auto-inference:
        the correction is always the harness's explicit statement. Nothing
        is recomputed here; derived layers pick the change up at the next
        ``induce`` / ``rebuild-index``."""
        rec = self.get(execution_id)
        if rec is None:
            raise StorageError(f"unknown execution_id {execution_id!r}")
        if rec.source == "excluded":
            raise StorageError(
                f"execution {execution_id!r} is already excluded")
        if superseded_by is not None and self.get(superseded_by) is None:
            raise StorageError(
                f"superseded_by execution {superseded_by!r} does not exist")
        rec.execution_features["correction"] = {
            "excluded": True,
            "reason": str(reason),
            "superseded_by": (str(superseded_by)
                              if superseded_by is not None else None),
            "excluded_at": time.time(),
        }
        rec.source = "excluded"
        with self.store.transaction() as conn:
            conn.execute(
                "UPDATE executions SET payload=?, source=? "
                "WHERE execution_id=?",
                (self.store.dumps(rec.to_dict()), rec.source, execution_id))
        return rec

    def restore(self, execution_id: str, reason: str) -> ExecutionRecord:
        """Reverse :meth:`exclude`: the fact counts as evidence again.

        A second explicit statement, because an exclusion can itself be
        wrong. The correction record is kept (``restored`` + reason) rather
        than erased, so the fact's history shows both decisions."""
        rec = self.get(execution_id)
        if rec is None:
            raise StorageError(f"unknown execution_id {execution_id!r}")
        if rec.source != "excluded":
            raise StorageError(
                f"execution {execution_id!r} is not excluded (source="
                f"{rec.source!r})")
        correction = dict(rec.execution_features.get("correction") or {})
        correction["restored"] = True
        correction["restore_reason"] = str(reason)
        correction["restored_at"] = time.time()
        rec.execution_features["correction"] = correction
        rec.source = "executed"
        with self.store.transaction() as conn:
            conn.execute(
                "UPDATE executions SET payload=?, source=? "
                "WHERE execution_id=?",
                (self.store.dumps(rec.to_dict()), rec.source, execution_id))
        return rec

    # -- reads --------------------------------------------------------------------

    def get(self, execution_id: str) -> Optional[ExecutionRecord]:
        row = self.store.conn.execute(
            "SELECT payload FROM executions WHERE execution_id=?",
            (execution_id,)).fetchone()
        return self._decode(row) if row else None

    def query(self, *, task_id: Optional[str] = None,
              strategy_id: Optional[str] = None,
              family: Optional[str] = None,
              group_l1: Optional[str] = None,
              source: Optional[str] = None,
              scope: Optional[str] = None,
              limit: Optional[int] = None) -> List[ExecutionRecord]:
        """Query facts.

        ``family`` is the authoritative selector. ``group_l1`` is a DERIVED
        INDEX column whose FORMAT has changed over time — it has held at least
        ``family=routing|sc[..]|rc[..]|..`` (ladder era) and plain
        ``family=routing`` (the version that retired the ladder) — so it is
        never used to decide membership:

        - with ``group_l1`` given, the query first narrows to that family and
          then re-derives the structural key from each record's own profile
          snapshot, so facts written under any historical index format stay
          visible. Relying on the column directly is exactly how a whole
          generation of records dropped out of the statistics.
        """
        sql = "SELECT payload FROM executions"
        clauses, params = [], []
        if task_id is not None:
            clauses.append("task_id=?"); params.append(task_id)
        if strategy_id is not None:
            clauses.append("strategy_id=?"); params.append(strategy_id)
        if group_l1 is not None and family is None:
            family = group_l1.split("|", 1)[0].replace("family=", "", 1) or None
        if family is not None:
            clauses.append("family=?"); params.append(family)
        if source is not None:
            clauses.append("source=?"); params.append(source)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at ASC, execution_id ASC"
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        records = [self._decode(r) for r in
                   self.store.conn.execute(sql, params).fetchall()]
        if group_l1 is not None:
            records = [r for r in records
                       if group_key(r.profile_snapshot) == group_l1]
        if scope is not None:
            records = [r for r in records if r.measurement_scope == scope]
        return records

    def index_health(self) -> Dict[str, Any]:
        """Read-only diagnostic: how many rows carry a stale ``group_l1``.

        ``group_l1`` is a DERIVED index (it must equal
        :func:`or_harness.core.schema.group_key` of the row's own profile),
        and it has held at least two historical formats. Reads never depend
        on it now, but a database upgraded from an older version is worth
        reporting honestly — including the previous release's plain
        ``family=routing`` form, which an earlier check mistook for healthy.
        Nothing is written here."""
        stale = 0
        for row in self.store.conn.execute("SELECT group_l1, payload FROM executions"):
            try:
                record = self._decode(row)
            except StorageError:
                stale += 1
                continue
            if str(row["group_l1"]) != record.group_l1:
                stale += 1
        total = self.count()
        return {"rows": total, "stale_group_index": stale,
                "note": ("group_l1 is a derived index; reads re-derive the key "
                         "from each record's profile snapshot, so stale rows "
                         "are still counted correctly")}

    def all(self) -> List[ExecutionRecord]:
        return self.query()

    def count(self) -> int:
        row = self.store.conn.execute("SELECT COUNT(*) AS n FROM executions").fetchone()
        return int(row["n"])

    # -- pending staging (the no-lost-facts safety net) --------------------------

    def stage_pending(self, record: ExecutionRecord) -> str:
        """Stage an execution produced by `orx execute` before the harness
        decides to record it. Every execution — successes AND failures — is
        staged automatically, so a failed attempt can never be silently lost
        when the harness immediately retries with a different approach.
        Staging is not recording: the fact enters the Experience Bank only
        via :meth:`append` (the harness's explicit decision)."""
        with self.store.transaction() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO pending_executions "
                "(execution_id, task_id, created_at, payload) VALUES (?,?,?,?)",
                (record.execution_id, record.task_id, record.created_at,
                 self.store.dumps(record.to_dict())))
        return record.execution_id

    def pending(self, *, task_id: Optional[str] = None) -> List[ExecutionRecord]:
        """Staged-but-unrecorded executions, oldest first."""
        sql = "SELECT payload FROM pending_executions"
        params: List[Any] = []
        if task_id is not None:
            sql += " WHERE task_id=?"
            params.append(task_id)
        sql += " ORDER BY created_at ASC, execution_id ASC"
        return [self._decode(r) for r in
                self.store.conn.execute(sql, params).fetchall()]

    def get_pending(self, execution_id: str) -> Optional[ExecutionRecord]:
        row = self.store.conn.execute(
            "SELECT payload FROM pending_executions WHERE execution_id=?",
            (execution_id,)).fetchone()
        return self._decode(row) if row else None

    def clear_pending(self, execution_id: str) -> None:
        """Remove a staged execution after it was recorded (or explicitly
        discarded by the harness)."""
        with self.store.transaction() as conn:
            conn.execute("DELETE FROM pending_executions WHERE execution_id=?",
                         (execution_id,))

    # -- internals -----------------------------------------------------------------

    @staticmethod
    def _decode(row: Any) -> ExecutionRecord:
        try:
            return ExecutionRecord.from_dict(Store.loads(row["payload"]))
        except (StorageError, ValueError, KeyError) as exc:
            raise StorageError(f"corrupt execution row: {exc}") from exc
