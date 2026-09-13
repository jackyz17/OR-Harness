"""Unified action log: the seven action classes and real transitions (M1).

An ActionRecord is one action of the unified contract — understand / model /
select_strategy / execute_strategy / verify / finish_task / induce — with a
two-phase lifecycle:

    begin_action(...)  ->  binds and persists the PRE snapshot, mints an
                           action id, writes status="running"
    end_action(...)     ->  persists the actual result, updates task
                           progress, binds the POST snapshot

A "running" record that never ends is itself a legal, inspectable fact (the
process was interrupted). A retro-reported action with no pre snapshot says
so explicitly (``pre_snapshot_missing``) — absence is recorded as absence.

Lifecycle status and business outcome are SEPARATE: an induce that produced
no new entry is not "failed" — it may have revised an entry, refused a
candidate at the gate, or legitimately changed nothing. The business result
(``created`` / ``updated`` / ``revised`` / ``refused`` / ``unchanged``) is
recorded alongside, so future benefit learning (M4) never trains on wrong
labels.

Idempotency and retry semantics:
- ``action_id`` is the primary key; a duplicate insert is rejected.
- ``end_action`` replayed with the SAME content (status/outcome/cost) is
  idempotent and returns the stored record; DIFFERENT content is a conflict.
- Cost amendments go through ``amend_action_cost`` (an explicit channel,
  replace semantics — re-applying the same measurement never double-counts).

This is a LOG, not a knowledge bank: nothing here promotes, verifies, or
retires strategic knowledge.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from or_harness.core.schema import COST_DIMENSIONS, CostVector
from or_harness.core.storage import Store, StorageError
from or_harness.world_model.state import MAINTENANCE_TASK_ID, BeliefSnapshot

#: The seven action classes of the unified contract.
ACTION_TYPES = ("understand", "model", "select_strategy", "execute_strategy",
                "verify", "finish_task", "induce")

#: Lifecycle statuses. ``running`` = begun, not ended (interrupted actions
#: stay here). ``no_valid_entry`` is induce-specific: the induction ran and
#: produced no valid entry — a business outcome, not a crash.
ACTION_STATUSES = ("running", "completed", "failed", "cancelled", "timeout",
                   "no_valid_entry")

#: Provenance of the record itself.
ACTION_SOURCES = ("executed", "agent_reported", "hypothetical")

#: Business outcomes of an induce action (lifecycle-independent).
INDUCE_OUTCOMES = ("created", "updated", "revised", "refused", "unchanged")

#: How a macro action's cost relates to its children's.
#: ``reference`` = the macro's cost field REFERENCES child costs (already
#: counted on the child execution records / child actions) and must not be
#: summed again; ``own`` = the macro's cost is its own additional spend.
ROLLUP_MODES = ("own", "reference")


def _outcome_fingerprint(status: str, outcome: Dict[str, Any],
                          cost: Optional[CostVector]) -> str:
    blob = json.dumps({
        "status": status,
        "outcome": outcome or {},
        "cost": (cost.to_dict() if cost is not None else None),
    }, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


@dataclass
class ActionRecord:
    action_id: str
    action_type: str
    task_id: str
    episode_id: Optional[str]
    started_at: float
    parent_action_id: Optional[str] = None
    pre_snapshot_id: Optional[str] = None
    post_snapshot_id: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    status: str = "running"
    ended_at: Optional[float] = None
    outcome: Dict[str, Any] = field(default_factory=dict)
    cost: Optional[CostVector] = None
    source: str = "executed"
    linked_execution_id: Optional[str] = None
    rollup: str = "own"

    @staticmethod
    def new_id() -> str:
        return f"ac_{uuid.uuid4().hex[:12]}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_id": self.action_id,
            "action_type": self.action_type,
            "task_id": self.task_id,
            "episode_id": self.episode_id,
            "parent_action_id": self.parent_action_id,
            "pre_snapshot_id": self.pre_snapshot_id,
            "post_snapshot_id": self.post_snapshot_id,
            "params": copy.deepcopy(self.params),
            "status": self.status,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "outcome": copy.deepcopy(self.outcome),
            "cost": (self.cost.to_dict() if self.cost is not None else None),
            "cost_measured": (sorted(self.cost.measured)
                              if self.cost is not None
                              and self.cost.measured is not None else None),
            "source": self.source,
            "linked_execution_id": self.linked_execution_id,
            "rollup": self.rollup,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ActionRecord":
        if not isinstance(data, dict) or not data.get("action_id"):
            raise ValueError("ActionRecord.action_id is required")
        action_type = str(data.get("action_type", ""))
        if action_type not in ACTION_TYPES:
            raise ValueError(
                f"action_type must be one of {ACTION_TYPES}")
        status = str(data.get("status", "running"))
        if status not in ACTION_STATUSES:
            raise ValueError(f"status must be one of {ACTION_STATUSES}")
        source = str(data.get("source", "executed"))
        if source not in ACTION_SOURCES:
            raise ValueError(f"source must be one of {ACTION_SOURCES}")
        rollup = str(data.get("rollup", "own"))
        if rollup not in ROLLUP_MODES:
            raise ValueError(f"rollup must be one of {ROLLUP_MODES}")
        raw_cost = data.get("cost")
        cost = (CostVector.from_dict(raw_cost)
                if isinstance(raw_cost, dict) else None)
        raw_measured = data.get("cost_measured")
        if cost is not None and isinstance(raw_measured, list):
            cost.measured = {str(d) for d in raw_measured
                             if d in COST_DIMENSIONS}
        return cls(
            action_id=str(data["action_id"]),
            action_type=action_type,
            task_id=str(data.get("task_id", "")),
            episode_id=data.get("episode_id"),
            parent_action_id=data.get("parent_action_id"),
            pre_snapshot_id=data.get("pre_snapshot_id"),
            post_snapshot_id=data.get("post_snapshot_id"),
            params=copy.deepcopy(dict(data.get("params") or {})),
            status=status,
            started_at=float(data.get("started_at", time.time())),
            ended_at=(float(data["ended_at"])
                      if data.get("ended_at") is not None else None),
            outcome=copy.deepcopy(dict(data.get("outcome") or {})),
            cost=cost,
            source=source,
            linked_execution_id=data.get("linked_execution_id"),
            rollup=rollup,
        )


class ActionLog:
    """Persistence + lifecycle for ActionRecords (a log, not a bank)."""

    def __init__(self, store: Store):
        self.store = store

    # -- lifecycle ------------------------------------------------------------

    def begin_action(self, action_type: str, task_id: str,
                     episode_id: Optional[str], *,
                     pre_snapshot: Optional[BeliefSnapshot] = None,
                     params: Optional[Dict[str, Any]] = None,
                     parent_action_id: Optional[str] = None,
                     source: str = "executed",
                     started_at: Optional[float] = None) -> ActionRecord:
        """Begin: bind the PRE snapshot (taken BEFORE the action runs),
        mint an id, persist status=running."""
        if action_type not in ACTION_TYPES:
            raise ValueError(f"action_type must be one of {ACTION_TYPES}")
        if source not in ACTION_SOURCES:
            raise ValueError(f"source must be one of {ACTION_SOURCES}")
        record = ActionRecord(
            action_id=ActionRecord.new_id(),
            action_type=action_type,
            task_id=task_id,
            episode_id=episode_id,
            started_at=started_at if started_at is not None else time.time(),
            parent_action_id=parent_action_id,
            pre_snapshot_id=(pre_snapshot.snapshot_id
                             if pre_snapshot is not None else None),
            params=copy.deepcopy(params or {}),
            status="running",
            source=source,
        )
        self._insert(record)
        return record

    def end_action(self, action_id: str, *,
                    status: str = "completed",
                    outcome: Optional[Dict[str, Any]] = None,
                    cost: Optional[CostVector] = None,
                    post_snapshot: Optional[BeliefSnapshot] = None,
                    linked_execution_id: Optional[str] = None,
                    rollup: str = "own") -> ActionRecord:
        """End: persist the actual result and bind the POST snapshot.

        Idempotent replay: same content (status/outcome/cost fingerprint)
        returns the stored record; different content raises a conflict.
        Ending a non-running action or an unknown id is an error."""
        record = self.get(action_id)
        if record is None:
            raise StorageError(f"unknown action_id {action_id!r}")
        if record.status != "running":
            fingerprint = _outcome_fingerprint(status, outcome or {}, cost)
            stored = _outcome_fingerprint(record.status, record.outcome,
                                         record.cost)
            if fingerprint == stored:
                return record  # idempotent replay of the same ending
            raise StorageError(
                f"action {action_id!r} already ended with different content "
                f"(stored status={record.status!r}): conflicting re-end is "
                "rejected; amend via amend_action_cost for cost changes")
        if status not in ACTION_STATUSES or status == "running":
            raise ValueError(f"end status must be one of {ACTION_STATUSES} "
                             "(excluding 'running')")
        record.status = status
        record.ended_at = time.time()
        record.outcome = copy.deepcopy(outcome or {})
        record.cost = cost
        record.post_snapshot_id = (post_snapshot.snapshot_id
                                   if post_snapshot is not None else None)
        if linked_execution_id is not None:
            record.linked_execution_id = linked_execution_id
        record.rollup = rollup
        self._update(record)
        return record

    def report_action(self, action_type: str, task_id: str,
                      episode_id: Optional[str], *,
                      params: Optional[Dict[str, Any]] = None,
                      outcome: Optional[Dict[str, Any]] = None,
                      status: str = "completed",
                      cost: Optional[CostVector] = None,
                      pre_snapshot: Optional[BeliefSnapshot] = None,
                      post_snapshot: Optional[BeliefSnapshot] = None,
                      started_at: Optional[float] = None,
                      ended_at: Optional[float] = None) -> ActionRecord:
        """Retro-report an action the OUTER agent performed (source=
        agent_reported). The library did not execute it; the report is the
        caller's statement, labelled as such. A missing pre snapshot is
        recorded explicitly (``pre_snapshot_missing``), never fabricated."""
        if action_type not in ACTION_TYPES:
            raise ValueError(f"action_type must be one of {ACTION_TYPES}")
        if action_type == "execute_strategy":
            raise ValueError(
                "execute_strategy is library-executed: use api.execute, not "
                "report_action")
        if status not in ACTION_STATUSES or status == "running":
            raise ValueError("a retro-reported action must already be ended")
        outcome = copy.deepcopy(outcome or {})
        if pre_snapshot is None:
            outcome.setdefault("pre_snapshot_missing", True)
        record = ActionRecord(
            action_id=ActionRecord.new_id(),
            action_type=action_type,
            task_id=task_id,
            episode_id=episode_id,
            started_at=started_at if started_at is not None else time.time(),
            pre_snapshot_id=(pre_snapshot.snapshot_id
                             if pre_snapshot is not None else None),
            post_snapshot_id=(post_snapshot.snapshot_id
                              if post_snapshot is not None else None),
            params=copy.deepcopy(params or {}),
            status=status,
            ended_at=ended_at if ended_at is not None else time.time(),
            outcome=outcome,
            cost=cost,
            source="agent_reported",
        )
        self._insert(record)
        return record

    def amend_action_cost(self, action_id: str, **dimensions: float
                          ) -> ActionRecord:
        """Explicit cost-amendment channel (replace semantics, idempotent).

        Only cost dimensions may change — mirroring the Experience Bank's
        ``update_cost`` discipline. Re-applying the same measurement never
        double-counts."""
        record = self.get(action_id)
        if record is None:
            raise StorageError(f"unknown action_id {action_id!r}")
        unknown = set(dimensions) - set(COST_DIMENSIONS)
        if unknown:
            raise StorageError(f"unknown cost dimensions: {sorted(unknown)}")
        if record.cost is None:
            record.cost = CostVector(measured=set())
        for d, v in dimensions.items():
            setattr(record.cost, d, float(v))
        record.cost.mark_measured(*dimensions)
        self._update(record)
        return record

    # -- queries ----------------------------------------------------------------

    def get(self, action_id: str) -> Optional[ActionRecord]:
        row = self.store.conn.execute(
            "SELECT payload FROM action_records WHERE action_id=?",
            (action_id,)).fetchone()
        return self._decode(row) if row else None

    def query(self, *, task_id: Optional[str] = None,
              episode_id: Optional[str] = None,
              action_type: Optional[str] = None,
              source: Optional[str] = None,
              status: Optional[str] = None) -> List[ActionRecord]:
        sql = "SELECT payload FROM action_records"
        clauses, params = [], []
        if task_id is not None:
            clauses.append("task_id=?"); params.append(task_id)
        if episode_id is not None:
            clauses.append("episode_id=?"); params.append(episode_id)
        if action_type is not None:
            clauses.append("action_type=?"); params.append(action_type)
        if source is not None:
            clauses.append("source=?"); params.append(source)
        if status is not None:
            clauses.append("status=?"); params.append(status)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY started_at ASC, action_id ASC"
        return [self._decode(r) for r in
                self.store.conn.execute(sql, params).fetchall()]

    def running(self, *, task_id: Optional[str] = None) -> List[ActionRecord]:
        """Actions begun but never ended (interrupted processes)."""
        return self.query(task_id=task_id, status="running")

    # -- internals -----------------------------------------------------------------

    def _insert(self, record: ActionRecord) -> None:
        payload = ActionRecord.from_dict(record.to_dict())  # round-trip check
        with self.store.transaction() as conn:
            try:
                conn.execute(
                    "INSERT INTO action_records "
                    "(action_id, action_type, task_id, episode_id, "
                    " parent_action_id, source, started_at, ended_at, payload) "
                    "VALUES (?,?,?,?,?,?,?,?,?)",
                    (payload.action_id, payload.action_type, payload.task_id,
                     payload.episode_id, payload.parent_action_id,
                     payload.source, payload.started_at, payload.ended_at,
                     self.store.dumps(payload.to_dict())))
            except Exception as exc:
                if "UNIQUE" in str(exc).upper():
                    raise StorageError(
                        f"duplicate action_id {payload.action_id!r}: the "
                        "action log is append-only") from exc
                raise

    def _update(self, record: ActionRecord) -> None:
        with self.store.transaction() as conn:
            cur = conn.execute(
                "UPDATE action_records SET ended_at=?, payload=? "
                "WHERE action_id=?",
                (record.ended_at, self.store.dumps(record.to_dict()),
                 record.action_id))
            if cur.rowcount == 0:
                raise StorageError(f"unknown action_id {record.action_id!r}")

    @staticmethod
    def _decode(row: Any) -> ActionRecord:
        try:
            return ActionRecord.from_dict(Store.loads(row["payload"]))
        except (StorageError, ValueError, KeyError) as exc:
            raise StorageError(f"corrupt action record: {exc}") from exc


#: The task-progress keys each action type updates when it ends (the
#: post-state update rules). ``induce`` updates no single-task progress —
#: it belongs to the maintenance scope.
PROGRESS_UPDATES: Dict[str, str] = {
    "understand": "understanding",
    "model": "model_artifact",
    "select_strategy": "selected_plan",
    "execute_strategy": "current_solution",
    "verify": "verification_evidence",
    "finish_task": "finished",
}
