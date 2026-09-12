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
  - ``replace_all``: reserved for future evidence compaction — currently
    unused, because lossy compaction is deferred until the summary
    consumption contract exists (statistics and induction ignore
    ``source="compacted"`` rows). Historical facts are never rewritten
    because later beliefs changed.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional

from or_harness.core.schema import COST_DIMENSIONS, ExecutionRecord, compute_cost_feedback
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

        Backfilled dimensions are marked measured. If a cost feedback summary
        was persisted, it is re-computed against the amended value (or
        removed when no longer computable), so the stored summary never
        disagrees with the stored fact.
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
        for d, v in dimensions.items():
            if mode == "increment":
                setattr(rec.cost, d, float(getattr(rec.cost, d)) + float(v))
            else:
                setattr(rec.cost, d, float(v))
        rec.cost.mark_measured(*dimensions)
        # Keep any persisted feedback consistent with the amended fact.
        if rec.execution_features.get("cost_feedback") is not None:
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
              limit: Optional[int] = None) -> List[ExecutionRecord]:
        sql = "SELECT payload FROM executions"
        clauses, params = [], []
        if task_id is not None:
            clauses.append("task_id=?"); params.append(task_id)
        if strategy_id is not None:
            clauses.append("strategy_id=?"); params.append(strategy_id)
        if family is not None:
            clauses.append("family=?"); params.append(family)
        if group_l1 is not None:
            clauses.append("group_l1=?"); params.append(group_l1)
        if source is not None:
            clauses.append("source=?"); params.append(source)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at ASC, execution_id ASC"
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        return [self._decode(r) for r in
                self.store.conn.execute(sql, params).fetchall()]

    def all(self) -> List[ExecutionRecord]:
        return self.query()

    def count(self) -> int:
        row = self.store.conn.execute("SELECT COUNT(*) AS n FROM executions").fetchone()
        return int(row["n"])

    def replace_all(self, records: List[ExecutionRecord]) -> None:
        """Rewrite the whole bank. Reserved for future evidence compaction —
        currently unused: lossy compaction is deferred until the summary
        consumption contract exists (statistics and induction ignore
        ``source="compacted"`` rows, so summarizing raw facts would bias
        conditional statistics)."""
        with self.store.transaction() as conn:
            conn.execute("DELETE FROM executions")
            for rec in records:
                conn.execute(
                    "INSERT INTO executions "
                    "(execution_id, task_id, strategy_id, family, group_l1, "
                    " source, created_at, payload) VALUES (?,?,?,?,?,?,?,?)",
                    (rec.execution_id, rec.task_id, rec.strategy_id,
                     rec.profile_snapshot.family, rec.group_l1, rec.source,
                     rec.created_at, self.store.dumps(rec.to_dict())))

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

    def iter_group(self, group_l1: str) -> Iterator[ExecutionRecord]:
        return iter(self.query(group_l1=group_l1))

    # -- internals -----------------------------------------------------------------

    @staticmethod
    def _decode(row: Any) -> ExecutionRecord:
        try:
            return ExecutionRecord.from_dict(Store.loads(row["payload"]))
        except (StorageError, ValueError, KeyError) as exc:
            raise StorageError(f"corrupt execution row: {exc}") from exc
