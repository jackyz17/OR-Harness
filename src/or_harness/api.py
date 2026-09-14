"""ORHarness facade: the Python entry point for the outer harness agent.

The harness is the orchestrator. This layer advises and executes; the harness
may refuse recommendations, request alternatives, execute without recording,
override recorded costs, and decides when to induce and when to collect
garbage. No conversation loop, no runtime LLM calls, no hidden global state —
the memory location is always explicit (``home`` / ``OR_HARNESS_HOME``).
"""

from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from or_harness.adapters.solver import available_families, probe_all
from or_harness.core.coupling import understand as cir_understand
from or_harness.core.schema import (
    COST_DIMENSIONS,
    CostVector,
    ExecutionRecord,
    PredictionSnapshot,
    ProblemProfile,
    compute_cost_feedback,
    group_key,
    profile_matches,
)
from or_harness.core.storage import StorageError, Store, resolve_home
from or_harness.execution.executor import SafePythonExecutor
from or_harness.profiling.profiler import derivation_report, profile_task
from or_harness.strategy.catalog import load_catalog
from or_harness.strategy.experience_bank import ExperienceBank
from or_harness.strategy.gc import GarbageCollector
from or_harness.strategy.induction import InductionEngine
from or_harness.strategy.selector import Selector, is_publishable
from or_harness.strategy.stats import ConditionalStats, quality_score
from or_harness.strategy.strategic_bank import StrategicBank
from or_harness.strategy.triggers import check_triggers, solver_advisories
from or_harness.world_model.actions import (
    ActionLog,
)
from or_harness.world_model.budget import BudgetLedger
from or_harness.world_model.state import (
    MAINTENANCE_TASK_ID,
    BeliefSnapshot,
    KnowledgeRef,
    verified_knowledge_view,
)

#: Prediction hit tolerance: an observation counts as a miss when it falls
#: outside the entry's interval by more than this fraction of the interval
#: width (relative slack keeps wide honest intervals meaningful).
PREDICTION_HIT_SLACK = 0.15


class ORHarness:
    def __init__(self, home: Optional[str] = None, *,
                 alpha: float = 1.0, beta: float = 1.0, gamma: float = 1.0,
                 cost_weights: Optional[Dict[str, float]] = None,
                 catalog_path: Optional[str] = None,
                 executor: Optional[SafePythonExecutor] = None):
        self.home = resolve_home(home)
        self.store = Store(self.home)
        self.bank = ExperienceBank(self.store)
        self.sbank = StrategicBank(self.store)
        self.stats = ConditionalStats(self.bank)
        self.catalog = load_catalog(catalog_path)
        self.selector = Selector(self.catalog, self.sbank, self.stats,
                                 alpha=alpha, beta=beta, gamma=gamma,
                                 cost_weights=cost_weights)
        self.executor = executor or SafePythonExecutor()
        self.induction = InductionEngine(self.stats, self.sbank)
        self.gc = GarbageCollector(self.bank, self.sbank, self.stats)
        # World-model M1 substrate: action log + budget ledger (log/index
        # facilities — NOT a third knowledge bank).
        self.actions = ActionLog(self.store)
        self.budget = BudgetLedger(self.bank, self.actions)
        # Episode budget declarations (task_id/episode_id -> {dim: limit}),
        # in-memory only: the harness re-declares per session; snapshots
        # freeze the declaration they were taken under.
        self._episode_budgets: Dict[str, Dict[str, float]] = {}

    # -- capabilities ----------------------------------------------------------

    def _episode_progress(self, task_id: str,
                          episode_id: Optional[str]) -> Dict[str, Any]:
        """The episode's CURRENT information state (X): the latest real
        task_progress among this episode's snapshots.

        Continuity rule: each action's post snapshot merges its own progress
        update into the accumulated state, so the next action's pre state
        inherits what earlier actions established (selected_plan, model
        artifact, current solution, verification evidence...). Hypothetical
        snapshots NEVER contribute — an imagined outcome is not progress."""
        if episode_id is None:
            return {}
        best: Optional[BeliefSnapshot] = None
        for snap in self.snapshots(task_id=task_id):
            if snap.episode_id != episode_id or snap.hypothetical:
                continue
            if best is None or (snap.created_at, snap.snapshot_id) > \
                    (best.created_at, best.snapshot_id):
                best = snap
        if best is None:
            return {}
        return copy.deepcopy(best.task_progress)

    def snapshot(self, task: Dict[str, Any], episode_id: Optional[str] = None,
                 *, task_progress: Optional[Dict[str, Any]] = None,
                 hypothetical: bool = False) -> BeliefSnapshot:
        """Freeze and persist the current information state for this task.

        The snapshot is the ``b_t`` the world model will condition on: H
        (value-copied knowledge refs + experience counts + tool config), P
        (profile + task digest + CIR/model digests), X (the episode's
        accumulated progress MERGED with this call's updates, labelled), B
        (declared budget + consumption view), and a coverage view layered
        by admission state. Everything is copied by value at freeze time —
        later bank writes never change what this snapshot meant.

        ``task_progress`` (when given) is merged OVER the episode's
        accumulated state: the caller's update for one key never discards
        the other keys earlier actions established. ``hypothetical=True``
        marks a rollout successor state — excluded from real-state queries
        and from episode progress inheritance."""
        profile = self.profile(task)
        knowledge = verified_knowledge_view(profile, self.sbank)
        recent = self.bank.query(task_id=str(task.get("task_id", "")))
        harness_state = {
            "knowledge": knowledge,
            "experience": {
                "task_execution_count": len(recent),
                "total_executions": self.bank.count(),
                "recent_execution_ids": [r.execution_id for r in recent[:20]],
            },
            "tool_config": {
                "available_solver_families": available_families(),
                "executor_timeout_seconds": getattr(
                    self.executor, "timeout_seconds", None),
            },
        }
        budget = self._load_budget(str(task.get("task_id", "")), episode_id)
        budget_state = self.budget.view(
            str(task.get("task_id", "")), episode_id, budget=budget)
        progress = self._episode_progress(
            str(task.get("task_id", "")), episode_id)
        if task_progress:
            progress.update(copy.deepcopy(task_progress))
        snap = BeliefSnapshot.build(
            task, episode_id,
            harness_state=harness_state,
            problem_state={"profile": profile.to_dict()},
            task_progress=progress,
            budget_state=budget_state,
            coverage=self._coverage_view(profile, knowledge),
            hypothetical=hypothetical,
        )
        with self.store.transaction() as conn:
            conn.execute(
                "INSERT INTO belief_snapshots "
                "(snapshot_id, task_id, episode_id, created_at, payload) "
                "VALUES (?,?,?,?,?)",
                (snap.snapshot_id, snap.task_id, snap.episode_id,
                 snap.created_at, self.store.dumps(snap.to_dict())))
        return snap

    def get_snapshot(self, snapshot_id: str) -> Optional[BeliefSnapshot]:
        row = self.store.conn.execute(
            "SELECT payload FROM belief_snapshots WHERE snapshot_id=?",
            (snapshot_id,)).fetchone()
        return (BeliefSnapshot.from_dict(self.store.loads(row["payload"]))
                if row else None)

    def snapshots(self, *, task_id: Optional[str] = None) -> List[BeliefSnapshot]:
        sql = "SELECT payload FROM belief_snapshots"
        params: List[Any] = []
        if task_id is not None:
            sql += " WHERE task_id=?"
            params.append(task_id)
        sql += " ORDER BY created_at ASC, snapshot_id ASC"
        return [BeliefSnapshot.from_dict(self.store.loads(r["payload"]))
                for r in self.store.conn.execute(sql, params).fetchall()]

    def _coverage_view(self, profile, knowledge: Dict[str, Any]) -> Dict[str, Any]:
        """Conditional capability evidence: cell statistics + layered
        knowledge, with coverage gaps. NO composite capability score."""
        cells = self.stats.for_profile(profile)
        return {
            "cell_statistics": {sid: cell.to_dict() for sid, cell
                                in cells.items()},
            "knowledge_layers": {
                "verified": knowledge["verified"],
                "legacy_unknown": knowledge["legacy_unknown"],
                "unverified": knowledge["unverified"],
            },
            "coverage_gaps": {
                "strategies_without_evidence": [
                    sid for sid in self.catalog
                    if sid not in cells or cells[sid].n == 0],
                "note": "strategies with no attempt-scope evidence in this "
                        "structural cell; no quality claim is made for them",
            },
        }

    def declare_budget(self, task_id: str, budget: Dict[str, float],
                       episode_id: Optional[str] = None) -> Dict[str, Any]:
        """Declare (or replace) an episode budget.

        PERSISTED in the store's meta table (key ``budget|<task>|<episode>``)
        so separate CLI invocations — each with a fresh ORHarness instance —
        see the same declaration. Explicit and replaceable; snapshots freeze
        the declaration they were taken under."""
        key = f"budget|{task_id}|{episode_id or ''}"
        clean = {d: float(v) for d, v in budget.items()}
        with self.store.transaction() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?,?)",
                (key, self.store.dumps(clean)))
        self._episode_budgets[f"{task_id}|{episode_id or ''}"] = dict(clean)
        return {"task_id": task_id, "episode_id": episode_id,
                "budget": dict(clean)}

    def _load_budget(self, task_id: str,
                     episode_id: Optional[str] = None) -> Optional[Dict[str, float]]:
        """The declared budget for (task, episode): in-memory cache first,
        then the persisted meta row. None when nothing was declared."""
        key = f"{task_id}|{episode_id or ''}"
        if key in self._episode_budgets:
            return self._episode_budgets[key] or None
        row = self.store.conn.execute(
            "SELECT value FROM meta WHERE key=?",
            (f"budget|{task_id}|{episode_id or ''}",)).fetchone()
        if row is None:
            return None
        loaded = self.store.loads(row["value"])
        self._episode_budgets[key] = dict(loaded)
        return dict(loaded)

    def budget_view(self, task_id: str,
                    episode_id: Optional[str] = None,
                    budget: Optional[Dict[str, float]] = None
                    ) -> Dict[str, Any]:
        """The honest budget view (see BudgetLedger).

        ``budget`` (optional) overrides the declared budget for THIS call
        only — nothing is persisted."""
        if budget is None:
            budget = self._load_budget(task_id, episode_id)
        return self.budget.view(task_id, episode_id, budget=budget)

    # -- unified action contract ------------------------------------------------

    def begin_action(self, action_type: str, task: Dict[str, Any],
                     episode_id: Optional[str] = None, *,
                     params: Optional[Dict[str, Any]] = None,
                     parent_action_id: Optional[str] = None,
                     source: str = "executed") -> Dict[str, Any]:
        """Begin an action: freeze the PRE snapshot, mint an id, persist
        status=running. The pre snapshot is taken BEFORE the action runs —
        that is the point."""
        pre = self.snapshot(task, episode_id)
        record = self.actions.begin_action(
            action_type, str(task.get("task_id", "")), episode_id,
            pre_snapshot=pre, params=params,
            parent_action_id=parent_action_id, source=source)
        return {"action_id": record.action_id,
                "pre_snapshot_id": pre.snapshot_id, "status": record.status}

    def end_action(self, action_id: str, *,
                   status: str = "completed",
                   outcome: Optional[Dict[str, Any]] = None,
                   cost: Optional[Dict[str, float]] = None,
                   task: Optional[Dict[str, Any]] = None,
                   episode_id: Optional[str] = None,
                   linked_execution_id: Optional[str] = None,
                   rollup: str = "own") -> Dict[str, Any]:
        """End an action: persist the result, bind the POST snapshot (when
        the task is supplied), update nothing else implicitly.

        Save order (deliberate): (1) the replay/conflict check runs FIRST —
        an idempotent replay returns the stored record without writing a
        new snapshot; (2) the result and cost are persisted; (3) the POST
        snapshot is generated AFTER the cost is on the books, so the saved
        transition reflects the action's actual budget change."""
        cost_vector = self._cost_from_dict(cost)
        # (1) Replay/conflict check first — an idempotent replay returns
        # the stored record without writing anything (no new snapshot); a
        # conflict raises before anything persists.
        record = self.actions.get(action_id)
        if record is None:
            raise StorageError(f"unknown action_id {action_id!r}")
        replayed = record.status != "running"
        if replayed:
            from or_harness.world_model.actions import \
                _outcome_fingerprint
            fingerprint = _outcome_fingerprint(status, outcome or {},
                                               cost_vector)
            stored = _outcome_fingerprint(record.status, record.outcome,
                                           record.cost)
            if fingerprint != stored:
                raise StorageError(
                    f"action {action_id!r} already ended with different "
                    "content: conflicting re-end is rejected")
            return {"action_id": record.action_id,
                    "status": record.status,
                    "post_snapshot_id": record.post_snapshot_id}
        # (2) Persist result + cost.
        record = self.actions.end_action(
            action_id, status=status, outcome=outcome, cost=cost_vector,
            linked_execution_id=linked_execution_id, rollup=rollup)
        # (3) Post snapshot AFTER the cost is recorded, so its budget view
        # includes this action's spend. Hypothetical actions get a
        # hypothetical successor snapshot — never merged into the real
        # state chain.
        post = None
        if task is not None:
            progress = self._progress_from_outcome(action_id, outcome or {})
            post = self.snapshot(task, episode_id, task_progress=progress,
                                 hypothetical=record.source == "hypothetical")
            self.actions._bind_post_snapshot(action_id, post.snapshot_id)
        return {"action_id": record.action_id, "status": record.status,
                "post_snapshot_id": (post.snapshot_id if post is not None
                                    else record.post_snapshot_id)}

    @staticmethod
    def _cost_from_dict(cost: Optional[Dict[str, float]]) -> Optional[CostVector]:
        """Build a CostVector from an explicit dict. Every dimension the
        caller SUPPLIED is marked measured — including an explicit zero
        (a reported 0 is an observed zero, not an unknown)."""
        if cost is None:
            return None
        vector = CostVector(**{d: float(v) for d, v in cost.items()})
        vector.measured = {d for d in cost if d in COST_DIMENSIONS}
        return vector

    def _progress_from_outcome(self, action_id: str,
                               outcome: Dict[str, Any]) -> Dict[str, Any]:
        """Task-progress fields implied by an ended action (the post-state
        update rules): each action type owns one progress key.

        A hypothetical action's outcome is labelled ``epistemic="inferred"``
        and never enters the real state chain (its post snapshot is marked
        hypothetical and excluded from episode inheritance)."""
        from or_harness.world_model.actions import PROGRESS_UPDATES
        record = self.actions.get(action_id)
        if record is None:
            return {}
        key = PROGRESS_UPDATES.get(record.action_type)
        if key is None:
            return {}
        hypothetical = record.source == "hypothetical"
        return {key: {
            "value": outcome,
            "provenance": ("observed" if record.source == "executed"
                           else "agent_reported"),
            "epistemic": ("inferred" if hypothetical else "fact"),
            "evidence_ref": action_id,
            "hypothetical": hypothetical,
        }}

    def report_action(self, action_type: str, task: Dict[str, Any],
                      episode_id: Optional[str] = None, *,
                      params: Optional[Dict[str, Any]] = None,
                      outcome: Optional[Dict[str, Any]] = None,
                      status: str = "completed",
                      cost: Optional[Dict[str, float]] = None,
                      started_at: Optional[float] = None,
                      ended_at: Optional[float] = None) -> Dict[str, Any]:
        """Report an action the OUTER agent performed (understand / model /
        select_strategy / verify / finish_task). The library did not execute
        it — the report is the caller's statement, labelled agent_reported.
        A missing pre snapshot is recorded as missing, never fabricated.
        The report also generates a POST snapshot (merged into the
        episode's progress chain) so the next action inherits what this
        action established."""
        cost_vector = self._cost_from_dict(cost)
        record = self.actions.report_action(
            action_type, str(task.get("task_id", "")), episode_id,
            params=params, outcome=outcome, status=status,
            cost=cost_vector, started_at=started_at, ended_at=ended_at)
        # Post snapshot AFTER the record (and its cost) are persisted, so
        # the budget view reflects the reported spend.
        progress = self._progress_from_outcome(record.action_id,
                                               outcome or {})
        post = self.snapshot(task, episode_id, task_progress=progress)
        self.actions._bind_post_snapshot(record.action_id, post.snapshot_id)
        return {"action_id": record.action_id, "status": record.status,
                "source": record.source,
                "post_snapshot_id": post.snapshot_id}

    def amend_action_cost(self, action_id: str,
                          **dimensions: float) -> Dict[str, Any]:
        """Explicit cost-amendment channel for an action (replace
        semantics, idempotent — re-applying never double-counts)."""
        record = self.actions.amend_action_cost(action_id, **dimensions)
        return {"action_id": record.action_id,
                "cost": record.cost.to_dict() if record.cost else None}

    def understand(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Pre-model coupling-aware understanding.

        Validates the task's optional ``coupling`` field (a CIR), infers
        structural relations deterministically, derives coupling groups, and
        renders modeling guidance — all *before* the canonical model is
        written.  When no CIR is supplied, returns a prompt to submit one.
        """
        return cir_understand(task)

    def profile(self, task: Dict[str, Any], code: Optional[str] = None,
                cir: Optional[Any] = None) -> ProblemProfile:
        return profile_task(task, code, cir=cir)

    def derivation_report(self, task: Dict[str, Any],
                          code: Optional[str] = None,
                          cir: Optional[Any] = None) -> Dict[str, Any]:
        """Per-dimension coupling derivation report: value, origin
        (model/code/spec/supplied/null), notes, model verification issues,
        and cross-check warnings (including CIR ↔ model when a CIR is
        provided)."""
        return derivation_report(self.profile(task, code, cir=cir))

    def recall(self, task: Dict[str, Any], *, top: int = 3,
               exclude: Optional[Sequence[str]] = None,
               memory_mode: str = "cost-aware",
               code: Optional[str] = None) -> Dict[str, Any]:
        profile = self.profile(task, code)
        recs = self.selector.recall(profile, top=top, exclude=exclude,
                                    memory_mode=memory_mode)
        solvers = available_families()
        result = {
            "profile": profile.to_dict(),
            "recommendations": [r.to_dict() for r in recs],
            "available_solver_families": solvers,
            "solver_advisories": solver_advisories(self.bank),
        }
        profiling = profile.annotations.get("profiling") or {}
        if profiling.get("coupling_warnings"):
            result["coupling_warnings"] = profiling["coupling_warnings"]
        return result

    def predict_cost(self, task: Dict[str, Any], strategy_id: str,
                     code: Optional[str] = None) -> PredictionSnapshot:
        """Pre-execution cost expectation for (problem conditions, strategy).

        Estimation scope: one execution ATTEMPT under the conditions visible
        BEFORE execution (task JSON / frozen profile — never post-modeling
        artifacts). Provenance ladder:
        1. a matching StrategicEntry's expected cost (entry);
        2. comparable conditional statistics over attempt-scope Evidence
           Bank records (stats — a recount, with per-dimension support and
           a scale-coverage check);
        3. unknown (no usable evidence — never a default zero presented as
           cheap, and never a task-scope total re-labelled as an attempt
           prediction).
        """
        if strategy_id not in self.catalog:
            raise ValueError(f"unknown strategy_id {strategy_id!r}")
        profile = self.profile(task, code)
        # Published knowledge only: a candidate whose admission verification
        # is missing a verdict must not act as a verified entry prediction
        # either. Gating recall alone would leave this second door open —
        # the snapshot then falls back to the statistics ladder below, which
        # is exactly what "we have no verified knowledge yet" should look
        # like.
        entry = next((e for e in self.sbank.matching(profile)
                      if e.strategy_id == strategy_id
                      and is_publishable(e)), None)
        if entry is not None:
            return PredictionSnapshot(
                strategy_id=strategy_id,
                expected_cost=entry.expected_cost_hat,
                source="entry", measurement_scope="attempt",
                support_n=entry.support_n,
                support_per_dim=dict(entry.cost_support_n),
                evidence_refs=[entry.entry_id],
                note=f"strategic entry {entry.entry_id} ({entry.status})")
        cell = self.stats.for_profile(profile).get(strategy_id)
        if cell is not None and cell.n > 0:
            mismatch = self._scale_mismatch(profile, cell)
            if mismatch:
                return PredictionSnapshot(
                    strategy_id=strategy_id, expected_cost=None,
                    source="unknown", measurement_scope="attempt",
                    support_n=cell.n,
                    evidence_refs=list(cell.execution_ids),
                    note=("insufficient evidence: historical samples are not "
                          "comparable at this scale — " + mismatch))
            return PredictionSnapshot(
                strategy_id=strategy_id,
                expected_cost=cell.mean_cost,
                source="stats", measurement_scope="attempt",
                support_n=cell.n,
                support_per_dim=dict(cell.n_measured),
                evidence_refs=list(cell.execution_ids),
                note=f"conditional statistics over n={cell.n} attempt-scope "
                     "executions in this structural group")
        return PredictionSnapshot(strategy_id=strategy_id,
                                  expected_cost=None, source="unknown",
                                  measurement_scope="attempt",
                                  note="no attempt-scope cost evidence for "
                                       "this strategy×profile pair")

    @staticmethod
    def _scale_mismatch(profile, cell) -> Optional[str]:
        """Lightweight scale-comparability check: a target scale feature
        outside the historical sample coverage is flagged (no preset
        thresholds — sample coverage is the only yardstick)."""
        notes = []
        for feat, (lo, hi) in sorted(cell.scale_ranges.items()):
            value = profile.scale_features.get(feat)
            if value is None:
                continue
            if value < lo or value > hi:
                notes.append(f"{feat}={value} outside sample coverage "
                             f"[{lo}, {hi}]")
        return "; ".join(notes) if notes else None

    def execute(self, task: Dict[str, Any], strategy_id: str, code_path: str,
                workspace: str, *, solver: str,
                verification_level: str = "basic",
                episode_id: Optional[str] = None) -> ExecutionRecord:
        """Run one episode and assemble its Execution Evidence record.

        The returned record is an evidence unit: the strategy ACTUALLY used,
        the quality/cost ACTUALLY observed, the failures actually seen, and
        the implementation artifacts (solver output, diagnostics). When the
        task carries a CIR (``coupling`` field), a snapshot is preserved on
        the record so offline induction can re-bin this episode by structural
        context. The record makes no generalization claim.
        """
        if strategy_id not in self.catalog:
            raise ValueError(f"unknown strategy_id {strategy_id!r}")
        # Frozen pre-strategy signature: profile is derived from the task's
        # coupling (CIR) / spec / annotations / model fields ONLY — never
        # from solve.py.  The generated solve script is a post-strategy
        # artifact; letting it redefine the problem's identity would create
        # a self-reinforcing loop (strategy → code → profile → grouping →
        # future strategy choice).
        profile = self.profile(task)
        # Unified action record (macro): the PRE snapshot is frozen BEFORE
        # execution; the action ends when the execution ends (independent
        # of the harness's later record decision — execute/record
        # separation is unchanged).
        pre = self.snapshot(task, episode_id)
        action = self.actions.begin_action(
            "execute_strategy", str(task["task_id"]), episode_id,
            pre_snapshot=pre,
            params={"strategy_id": strategy_id, "solver": solver,
                    "verification_level": verification_level})
        record = self.executor.execute(
            Path(code_path), Path(workspace), solver=solver,
            task_id=str(task["task_id"]), strategy_id=strategy_id,
            profile=profile, verification_level=verification_level)
        # Evidence completeness: preserve the coupling-aware representation
        # snapshot (CIR) that was actually solved. Snapshot only — CIR
        # extraction and coupling understanding are untouched.
        if task.get("coupling"):
            record.cir_snapshot = dict(task["coupling"])
        # Safety net: stage every execution — successes AND failures — so a
        # failed attempt is never silently lost when the harness immediately
        # retries. Staging is not recording; recording stays the harness's
        # explicit decision (`orx record`).
        self.bank.stage_pending(record)
        # End the macro action: status follows the EXECUTION outcome (not
        # the record decision); cost is a REFERENCE to the execution's own
        # cost (counted there — never summed again); the post snapshot
        # carries the solution/error state.
        exec_status = record.quality.get("status")
        action_status = ("completed" if exec_status in ("optimal", "feasible")
                         else "failed" if exec_status in ("error",)
                         else "timeout" if exec_status == "timeout"
                         else "completed")
        progress = {
            "current_solution": {
                "value": {"status": exec_status,
                          "feasible": record.quality.get("feasible"),
                          "objective": record.quality.get("objective"),
                          "gap": record.quality.get("gap")},
                "provenance": "observed", "epistemic": "fact",
                "evidence_ref": record.execution_id},
        }
        if record.failures:
            progress["errors"] = {
                "value": [f.to_dict() for f in record.failures],
                "provenance": "observed", "epistemic": "fact",
                "evidence_ref": record.execution_id}
        post = self.snapshot(task, episode_id, task_progress=progress)
        self.actions.end_action(
            action.action_id, status=action_status,
            outcome={"execution_status": exec_status,
                     "feasible": record.quality.get("feasible"),
                     "objective": record.quality.get("objective")},
            cost=record.cost, post_snapshot=post,
            linked_execution_id=record.execution_id, rollup="reference")
        record.action_id = action.action_id
        return record

    def record(self, record: ExecutionRecord,
               override: Optional[Dict[str, float]] = None,
               retain_reason: Optional[str] = None, *,
               override_mode: str = "replace",
               prediction: Optional[PredictionSnapshot] = None) -> Dict[str, Any]:
        """Append a fact, then run the automatic chain:
        frozen quality checks -> cost backfill -> cost feedback -> C1-C6
        hints.

        The chain is EVIDENCE-ONLY: it never promotes, demotes, or awakens a
        Strategic Knowledge entry. Quality checks are written onto the fact
        (``execution_features.quality_feedback``); the next explicit
        ``induce`` replays them offline (``InductionEngine.revise``).

        ``retain_reason`` is an EXPLICIT, optional representative-evidence
        mark (reserved for future compaction policies): a non-empty value
        wins; otherwise the mark the record already carries is preserved.
        No automatic retention marking is performed.

        ``prediction`` is the pre-execution cost prediction ACTUALLY used
        (from :meth:`predict_cost`); it is frozen onto the record as the
        snapshot that feedback is computed against — feedback never re-reads
        a post-hoc current estimate. ``override`` backfills harness-owned
        dimensions (llm_tokens, retries the harness declares, extra tool
        calls) with explicit accounting: replace (default, idempotent —
        re-applying the same measurement never double-counts) or increment.

        ``override_mode`` / ``prediction`` are keyword-only so the historical
        positional call ``record(record, override, retain_reason)`` keeps its
        original meaning.

        Cost feedback is computed AFTER the backfill, from the frozen
        snapshot and the amended actual value, and persisted with the fact.
        A later ``update_cost`` re-computes it, so the stored summary can
        never disagree with the stored cost.

        Also reports staged-but-unrecorded executions for the same task, so
        the harness notices a dropped failure (e.g. an abandoned first
        attempt) before it is forgotten."""
        # Persist failure classification once — a first-class fact, not a
        # re-derived view (environment vs model errors feed solver advisories
        # and future failure-pattern induction).
        from or_harness.strategy.triggers import classify_failure
        for failure in record.failures:
            if failure.error_class is None:
                failure.error_class = classify_failure(record)
        # Explicit retention mark wins; otherwise keep the record's value.
        if retain_reason and retain_reason.strip():
            record.retention_reason = retain_reason.strip()
        # Freeze the pre-execution prediction snapshot when the harness
        # supplies one and the record does not already carry it.
        if prediction is not None and record.prediction_snapshot is None:
            record.prediction_snapshot = prediction
        # Frozen quality checks are computed against the interval in force
        # RIGHT NOW and persisted with the fact — nothing downstream
        # re-scores a later execution against a post-hoc interval.
        prediction_checks = self._check_predictions(record)
        if prediction_checks:
            record.execution_features["quality_feedback"] = prediction_checks
        self.bank.append(record)
        if override:
            self.bank.update_cost(record.execution_id,
                                  mode=override_mode, **override)
            record = self.bank.get(record.execution_id)
        self.bank.clear_pending(record.execution_id)

        cost_feedback = compute_cost_feedback(
            record.prediction_snapshot, record.strategy_id,
            record.measurement_scope, record.cost)
        if cost_feedback is not None:
            self.bank.set_cost_feedback(record.execution_id, cost_feedback)
        expected_map = {e.strategy_id: {"quality": e.expected_quality_hat}
                        for e in self.sbank.matching(record.profile_snapshot)}
        # expected_map already uses {sid: {"quality": q}} format, matching
        # the new check_triggers signature.
        prior_failures = self._prior_failures(record)
        hints = check_triggers(record, self.stats, self.catalog, expected_map,
                               prior_failures=prior_failures)
        unrecorded = [p.execution_id for p in
                      self.bank.pending(task_id=record.task_id)]
        result = {
            "execution_id": record.execution_id,
            "recorded": True,
            "prediction_checks": prediction_checks,
            "induction_hints": [h.to_dict() for h in hints],
        }
        if cost_feedback is not None:
            result["cost_feedback"] = cost_feedback
        if unrecorded:
            result["unrecorded_staged_executions"] = unrecorded
        return result

    def induce(self, *, strategy_id: Optional[str] = None, all_: bool = False,
               rebuild: bool = False,
               dry_run: bool = False, force: bool = False,
               notes: Optional[List[str]] = None,
               verify: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Consolidate Execution Evidence into Strategic Knowledge.

        Input = facts (ExecutionRecord rows, source="executed"); output =
        derived StrategicEntry commitments (expected quality/cost/failure
        risk). This is where knowledge changes: recording only accumulates
        evidence, and `induce` (i) forms candidates from the statistics of
        each structural cell, (ii) creates/refreshes entries, and (iii)
        REVISES existing entries from the frozen forward checks recorded on
        the facts — promotion (n>=5, hit rate>=0.7), demotion (3 consecutive
        misses), dormancy wakeup — reported under ``revisions``. Cost feedback
        never alters entry state.
        Once an entry exists its validity does not depend on the survival of
        the supporting evidence rows. New entries inherit the catalog
        vocabulary's strategy_type/actions — extension points for future
        induction — without ever overwriting harness-supplied values.

        ``verify`` carries the harness's admission check for the candidate
        this call forms (see ``InductionEngine.induce``); the verdict is
        computed by the framework from real executions. Without it the entry
        is ``unverified`` and is not published as strategic knowledge —
        recall falls back to the raw conditional statistics.
        """
        # The maintenance action begins BEFORE induction runs: the PRE
        # snapshot freezes the knowledge state as it was, so the recorded
        # transition shows what the induction actually changed. Dry-run
        # persists nothing (no action, no snapshots).
        maintenance = None
        if not dry_run:
            maintenance = self._begin_induce_action(strategy_id, all_,
                                                    rebuild)
        try:
            if rebuild:
                result = self.induction.rebuild(dry_run=dry_run)
                if not dry_run:
                    for entry_id in result.get("entry_ids", []):
                        self._enrich_entry(entry_id)
                    result["revisions"] = self.induction.revise()
            else:
                targets = self._induction_targets(strategy_id, all_)
                results = []
                for profile, sid in targets:
                    results.append(self.induction.induce(
                        profile, sid, dry_run=dry_run, force=force,
                        notes=notes, verify=verify))
                if not dry_run:
                    for r in results:
                        entry_id = r.get("created") or r.get("updated")
                        if entry_id:
                            self._enrich_entry(entry_id)
                result: Dict[str, Any] = {"results": results}
                if targets:
                    # Offline revision of the entries this call covers:
                    # lifecycle state is re-derived from the frozen checks
                    # on the facts.
                    result["revisions"] = self.induction.revise(
                        strategy_id=strategy_id, dry_run=dry_run)
        except Exception:
            # An induction that crashed still happened: end the action as
            # failed with the error, so the maintenance history keeps the
            # attempt (and its pre state) instead of silently losing it.
            if maintenance is not None:
                import traceback
                self.actions.end_action(
                    maintenance["action_id"], status="failed",
                    outcome={"error": traceback.format_exc(limit=3)})
            raise
        if maintenance is not None:
            result["action"] = self._end_induce_action(maintenance, result)
        return result

    def _begin_induce_action(self, strategy_id: Optional[str],
                             all_: bool, rebuild: bool) -> Dict[str, Any]:
        """Begin the maintenance-scope induce action with a real PRE
        snapshot: the knowledge state (entries + verification layers) as it
        stood BEFORE induction."""
        episode_id = f"maint_{int(time.time())}"
        knowledge_before = {
            "entries": [KnowledgeRef.from_entry(e).to_dict()
                        for e in self.sbank.list()],
            "entry_count": self.sbank.count(),
        }
        action = self.actions.begin_action(
            "induce", MAINTENANCE_TASK_ID, episode_id,
            params={"scope": "maintenance",
                    "strategy_id": strategy_id,
                    "all": bool(all_),
                    "rebuild": bool(rebuild),
                    "knowledge_before": knowledge_before})
        return {"action_id": action.action_id, "episode_id": episode_id,
                "knowledge_before": knowledge_before}

    def _end_induce_action(self, maintenance: Dict[str, Any],
                           result: Dict[str, Any]) -> Dict[str, Any]:
        """End the maintenance induce action: business result, verification
        outcomes, and the knowledge delta (before vs after), plus the POST
        snapshot of the knowledge state.

        Business-result labels distinguish CANDIDATES from VERIFIED
        knowledge: ``created`` counts only entries whose admission verdict
        is verified; unverified candidates are reported separately
        (``created_unverified``) so knowledge-growth accounting never counts
        them."""
        results = result.get("results") or []
        created = [r for r in results if r.get("created")]
        created_verified = [
            r for r in created
            if (r.get("verification") or {}).get("state") == "verified"]
        created_unverified = [r for r in created
                              if r not in created_verified]
        updated = [r for r in results if r.get("updated")]
        refused = [r for r in results if r.get("skipped")]
        revised = result.get("revisions") or []
        if created_verified:
            business = "created"
        elif created_unverified:
            business = "created_unverified"
        elif updated:
            business = "updated"
        elif revised:
            business = "revised"
        elif refused:
            business = "refused"
        else:
            business = "unchanged"
        knowledge_after = {
            "entries": [KnowledgeRef.from_entry(e).to_dict()
                        for e in self.sbank.list()],
            "entry_count": self.sbank.count(),
        }
        before_ids = {k["entry_id"] for k in
                     maintenance["knowledge_before"]["entries"]}
        after_ids = {k["entry_id"] for k in knowledge_after["entries"]}
        outcome = {
            "business_result": business,
            "created_verified": [r["created"] for r in created_verified],
            "created_unverified": [r["created"] for r in created_unverified],
            "updated": [r["updated"] for r in updated],
            "refused_reasons": [r.get("skipped") for r in refused],
            "revisions": len(revised),
            "verification_results": [
                {"entry": r.get("created") or r.get("updated"),
                 "verification": r.get("verification")}
                for r in results if r.get("verification")],
            "knowledge_delta": {
                "entries_created": sorted(after_ids - before_ids),
                "entries_removed": sorted(before_ids - after_ids),
                "entry_count_before":
                    maintenance["knowledge_before"]["entry_count"],
                "entry_count_after": knowledge_after["entry_count"],
            },
        }
        # Induction cost is UNKNOWN unless the harness amends it explicitly
        # (amend_action_cost) — never fabricated for report completeness.
        self.actions.end_action(
            maintenance["action_id"],
            status=("completed" if business not in ("refused",)
                    else "no_valid_entry"),
            outcome=outcome)
        return {"action_id": maintenance["action_id"],
                "episode_id": maintenance["episode_id"],
                "business_result": business}

    def _enrich_entry(self, entry_id: str) -> None:
        """Inherit catalog vocabulary (strategy_type, actions) into an entry.

        Only fills EMPTY fields — never overwrites values the harness already
        supplied. Keeps InductionEngine decoupled from the catalog while
        letting new entries carry the vocabulary's structural knowledge.
        """
        entry = self.sbank.get(entry_id)
        if entry is None:
            return
        strat = self.catalog.get(entry.strategy_id)
        if strat is None:
            return
        changed = False
        if not entry.strategy_type and strat.strategy_type:
            entry.strategy_type = strat.strategy_type
            changed = True
        if not entry.actions and strat.actions:
            entry.actions = list(strat.actions)
            changed = True
        if not entry.fallback_strategy_id and strat.fallback:
            entry.fallback_strategy_id = strat.fallback
            changed = True
        if changed:
            self.sbank.update(entry)

    def inspect(self, *, bank: str = "experience",
                task_id: Optional[str] = None,
                strategy_id: Optional[str] = None,
                status: Optional[str] = None,
                episode_id: Optional[str] = None) -> Dict[str, Any]:
        if bank == "experience":
            records = self.bank.query(task_id=task_id, strategy_id=strategy_id)
            return {"bank": "experience", "count": len(records),
                    "records": [r.to_dict() for r in records]}
        if bank == "strategic":
            entries = self.sbank.list(status=status, strategy_id=strategy_id)
            return {"bank": "strategic", "count": len(entries),
                    "entries": [e.to_dict() for e in entries]}
        if bank == "archive":
            cards = self.sbank.cold_archive()
            # Echo the REQUESTED bank name ("archive"): the other branches do
            # the same, and callers switch on this field.
            return {"bank": "archive", "count": len(cards),
                    "cards": [c.to_dict() for c in cards]}
        if bank == "actions":
            records = self.actions.query(task_id=task_id,
                                         episode_id=episode_id,
                                         action_type=strategy_id
                                         if strategy_id and strategy_id
                                         in ("understand", "model",
                                             "select_strategy",
                                             "execute_strategy", "verify",
                                             "finish_task", "induce")
                                         else None,
                                         status=status)
            return {"bank": "actions", "count": len(records),
                    "actions": [a.to_dict() for a in records]}
        if bank == "snapshots":
            snaps = self.snapshots(task_id=task_id)
            return {"bank": "snapshots", "count": len(snaps),
                    "snapshots": [s.to_dict() for s in snaps]}
        raise ValueError("bank must be experience|strategic|archive|"
                         "actions|snapshots")

    def collect_garbage(self, mode: str = "compact",
                        dry_run: bool = False) -> Dict[str, Any]:
        return self.gc.run(mode=mode, dry_run=dry_run)

    def retire(self, entry_id: str, reason: str) -> Dict[str, Any]:
        card = self.sbank.retire(entry_id, reason=reason)
        return {"retired": entry_id, "cold_archive_card": card.to_dict()}

    def doctor(self) -> Dict[str, Any]:
        reports = probe_all()
        pending = self.bank.pending()
        return {
            "home": str(self.home),
            "solvers": [r.to_dict() for r in reports],
            "available_families": available_families(),
            "memory": {"executions": self.bank.count(),
                       "entries": self.sbank.count(),
                       "cold_archive": len(self.sbank.cold_archive()),
                       "pending_staged": len(pending)},
            # group_l1 is a derived index: report staleness (read-only — the
            # open path never rewrites a bank) so an upgraded database is
            # visibly diagnosed instead of silently mysterious.
            "index_health": self.bank.index_health(),
            "pending_staged_executions": [
                {"execution_id": p.execution_id, "task_id": p.task_id,
                 "strategy_id": p.strategy_id,
                 "status": p.quality.get("status")} for p in pending],
        }

    def close(self) -> None:
        self.store.close()

    # -- automatic chain internals ------------------------------------------------

    def _check_predictions(self, record: ExecutionRecord) -> List[Dict[str, Any]]:
        """Frozen forward checks: this execution against matching entries'
        intervals, as EVIDENCE.

        Online the harness only accumulates: each check is computed against
        the interval in force at this moment and written onto the fact
        (``execution_features.quality_feedback``). No entry is promoted,
        demoted, tightened, or awakened here — ``InductionEngine.revise``
        replays these checks at the next offline induction.

        Isolation rules (mirroring the cost-feedback contract):
        - the strategy that ACTUALLY ran owns the check (A never audits B);
        - only attempt-scope executions produce checks: a task-scope total
          never audits attempt-scope knowledge;
        - dormant entries are included — a matching execution is evidence
          about the pattern, and waking the entry is an offline decision.
        """
        if record.measurement_scope != "attempt":
            return []
        observed = quality_score(record)
        events: List[Dict[str, Any]] = []
        for entry in self.sbank.matching(record.profile_snapshot,
                                         include_dormant=True):
            if entry.strategy_id != record.strategy_id:
                continue
            lo, hi = entry.quality_interval
            width = max(hi - lo, 1e-6)
            slack = PREDICTION_HIT_SLACK * width
            events.append({
                "entry_id": entry.entry_id,
                "predicted": round(entry.expected_quality_hat, 4),
                "interval": [lo, hi],
                "observed": round(observed, 4),
                "hit": bool(lo - slack <= observed <= hi + slack),
            })
        return events

    def task_cost_summary(self, task_id: str, *,
                          end_to_end_latency_s: Optional[float] = None
                          ) -> Dict[str, Any]:
        """Aggregate a task's full cost across its explicitly linked records.

        Only attempt-scope facts (the sole explicit association: shared
        ``task_id``) participate; a task-scope record can never be merged in.
        Every attempt is charged once, to the strategy that actually ran it —
        "A fails -> A retries -> B succeeds" charges all three, never all to
        B and never only the last success.

        Aggregation rules:
        - Cumulative dimensions (llm_tokens, tool_calls, solver_runtime_s)
          SUM over measured attempts.
        - retries sums too, because a record's retries expresses the NEW
          retries this attempt adds (a harness declaration, not a running
          total) — three attempts declared 0/1/1 total two retries.
        - Per-attempt latencies are reported as facts; end-to-end latency is
          NEVER inferred by taking max or summing them. It is only reported
          when the harness provides explicit task timing
          (``end_to_end_latency_s``); otherwise it is unknown.
        - A dimension any attempt did not measure is reported as incomplete
          ("known partial total"), never as a complete total.
        """
        records = [r for r in self.bank.query(task_id=task_id)
                   if r.source == "executed"
                   and r.measurement_scope == "attempt"]
        total: Dict[str, float] = {d: 0.0 for d in COST_DIMENSIONS}
        n_measured: Dict[str, int] = {d: 0 for d in COST_DIMENSIONS}
        per_attempt = []
        for rec in records:
            measured = rec.cost.measured_dims()
            for dim in COST_DIMENSIONS:
                if dim in measured:
                    n_measured[dim] += 1
                    if dim != "latency_s":
                        total[dim] += getattr(rec.cost, dim)
            per_attempt.append({
                "execution_id": rec.execution_id,
                "strategy_id": rec.strategy_id,
                "status": rec.quality.get("status"),
                "cost": rec.cost.to_dict(),
                "cost_measured": sorted(measured),
                "latency_s": rec.cost.latency_s,
            })
        complete = {d: (n_measured[d] == len(records) and len(records) > 0)
                    for d in COST_DIMENSIONS}
        # Latency never appears in total_cost: attempts may overlap, and
        # end-to-end wait is a separate harness-supplied fact (see below).
        total_cost = {d: (round(total[d], 4) if n_measured[d] > 0 else None)
                      for d in COST_DIMENSIONS if d != "latency_s"}
        return {
            "task_id": task_id,
            "n_attempts": len(records),
            "attempts": per_attempt,
            "total_cost": total_cost,
            "complete": complete,
            "n_measured": n_measured,
            "end_to_end_latency_s": (
                round(end_to_end_latency_s, 4)
                if end_to_end_latency_s is not None else None),
            "end_to_end_latency_source": (
                "harness-supplied" if end_to_end_latency_s is not None
                else "unknown"),
            "aggregation_note": (
                "cumulative dimensions summed over measured attempts; "
                "retries = sum of per-attempt NEW retries; per-attempt "
                "latencies reported separately and end-to-end latency is "
                "unknown unless the harness supplies explicit task timing "
                "(never inferred by max or sum); incomplete dimensions are "
                "marked complete=false — known partial totals are not "
                "complete totals"),
        }

    def _prior_failures(self, record: ExecutionRecord) -> List[ExecutionRecord]:
        """Failed executions (bank + staged) for the same task, excluding
        this record itself."""
        prior = [r for r in self.bank.query(task_id=record.task_id)
                 if r.execution_id != record.execution_id
                 and not r.quality.get("feasible", False)]
        prior += [p for p in self.bank.pending(task_id=record.task_id)
                  if p.execution_id != record.execution_id
                  and not p.quality.get("feasible", False)]
        return prior

    def _induction_targets(self, strategy_id: Optional[str], all_: bool):
        """One induction target per (structural group, strategy).

        The group is DERIVED from each record's own profile snapshot rather
        than read from the stored index column: legacy rows carry the old
        index format, and letting that decide targets would make the facts
        invisible to induction."""
        targets = []
        seen = set()
        for rec in self.bank.all():
            if rec.source != "executed" or rec.measurement_scope != "attempt":
                continue
            if strategy_id and rec.strategy_id != strategy_id:
                continue
            key = (group_key(rec.profile_snapshot), rec.strategy_id)
            if key in seen:
                continue
            if not all_ and strategy_id is None:
                continue
            seen.add(key)
            targets.append((rec.profile_snapshot, rec.strategy_id))
        return targets
