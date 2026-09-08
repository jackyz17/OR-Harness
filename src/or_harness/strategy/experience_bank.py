"""Experience Bank: the append-only episodic fact layer.

Records what happened — never what will happen. Facts are permanently neutral:
the disposal ladder (suspect/dormant/retired/cold archive) applies only to the
derived Strategic Bank. The Experience Bank is the single source of truth; the
Strategic Bank can always be rebuilt from it (``induce --rebuild``).
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional

from or_harness.core.schema import ExecutionRecord
from or_harness.core.storage import Store, StorageError


class ExperienceBank:
    """Append-only store of :class:`ExecutionRecord` facts."""

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

    def update_cost(self, execution_id: str, **dimensions: float) -> ExecutionRecord:
        """Backfill cost dimensions (harness-owned llm_tokens via --override).

        Appends nothing: this amends the fact's measured fields in place, which
        is the documented supplement channel. Only cost dimensions may change.
        """
        from or_harness.core.schema import COST_DIMENSIONS

        unknown = set(dimensions) - set(COST_DIMENSIONS)
        if unknown:
            raise StorageError(f"unknown cost dimensions: {sorted(unknown)}")
        rec = self.get(execution_id)
        if rec is None:
            raise StorageError(f"unknown execution_id {execution_id!r}")
        for d, v in dimensions.items():
            setattr(rec.cost, d, float(v))
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
        """Rewrite the whole bank (used by gc compaction). Fact-neutral: only
        rows eligible for compaction are replaced by compacted ledger lines."""
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

    def iter_group(self, group_l1: str) -> Iterator[ExecutionRecord]:
        return iter(self.query(group_l1=group_l1))

    # -- internals -----------------------------------------------------------------

    @staticmethod
    def _decode(row: Any) -> ExecutionRecord:
        try:
            return ExecutionRecord.from_dict(Store.loads(row["payload"]))
        except (StorageError, ValueError, KeyError) as exc:
            raise StorageError(f"corrupt execution row: {exc}") from exc
