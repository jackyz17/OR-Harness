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
from typing import Any, Dict, List, Optional, Sequence, Union

from or_harness.adapters.solver import available_families, probe_all
from or_harness.core.schema import (
    COST_DIMENSIONS,
    CostVector,
    ExecutionRecord,
    PredictionSnapshot,
    ProblemProfile,
    accumulate_measured_costs,
    compute_cost_feedback,
    group_key,
    profile_matches,
)
from or_harness.core.storage import StorageError, Store, resolve_home
from or_harness.execution.executor import SafePythonExecutor
from or_harness.profiling.profiler import derivation_report, profile_task
from or_harness.strategy.catalog import load_catalog
from or_harness.strategy.embedding_index import (
    EmbeddingBackend,
    EmbeddingIndex,
    create_embedding_backend,
    index_dir_for,
)
from or_harness.strategy.experience_bank import ExperienceBank
from or_harness.strategy.gc import GarbageCollector
from or_harness.strategy.index_sync import IndexSynchronizer
from or_harness.strategy.induction import InductionEngine
from or_harness.strategy.selector import Selector, is_publishable
from or_harness.strategy.stats import ConditionalStats, quality_score
from or_harness.strategy.strategic_bank import StrategicBank
from or_harness.strategy.triggers import check_triggers, solver_advisories
from or_harness.strategy.vector_recall import (
    VectorRecallUnavailable,
    recall_vectors,
)
from or_harness.world_model.actions import (
    ActionLog,
)
from or_harness.world_model.budget import BudgetLedger
from or_harness.world_model.contracts import (
    LEGACY_CONTRACT_VERSION,
    LEGACY_UNMAPPABLE,
    PAYLOAD_VERSION_UNKNOWN_PREFIX,
    PREDICTION_KINDS,
    BaselineStatement,
    BenefitEstimate,
    CandidateRef,
    CapabilityEvolutionPrediction,
    EvidenceRef,
    ExpectedChange,
    ExpectedCost,
    ExperienceScope,
    HarnessCapabilityEvidence,
    LearningOperation,
    PredictionTrace,
    RiskStatement,
    StrategyOutcomePrediction,
    TaskTargeting,
    UncertaintyStatement,
    VerificationCondition,
    capability_evidence_from_legacy_harness_state,
    detect_payload_version,
    legacy_prediction_view,
    load_contract_payload,
    validate_capability_evolution,
    validate_strategy_outcome,
)
from or_harness.world_model.execution_window import (
    StrategyExecutionWindow,
    window_from_records,
)
from or_harness.world_model.prediction import (
    ActionSpec,
    OutcomePrediction,
)
from or_harness.world_model.provider import (
    NotConfiguredProvider,
    WorldModelProvider,
)
from or_harness.world_model.service import PredictionService
from or_harness.world_model.state import (
    MAINTENANCE_TASK_ID,
    BeliefSnapshot,
    KnowledgeRef,
    task_text,
    task_text_digest,
    task_text_from_payload,
    verified_knowledge_view,
)

#: Prediction hit tolerance: an observation counts as a miss when it falls
#: outside the entry's interval by more than this fraction of the interval
#: width (relative slack keeps wide honest intervals meaningful).
PREDICTION_HIT_SLACK = 0.15

#: M6 experiment modes (see ORHarness.__init__).
PREDICTION_MODES = ("x-b-only", "h-x-b", "h-x-b-value")


class ORHarness:
    def __init__(self, home: Optional[str] = None, *,
                 alpha: float = 1.0, beta: float = 1.0, gamma: float = 1.0,
                 delta: float = 0.0,
                 cost_weights: Optional[Dict[str, float]] = None,
                 catalog_path: Optional[str] = None,
                 executor: Optional[SafePythonExecutor] = None,
                 world_model: Optional[WorldModelProvider] = None,
                 embedding: Optional[EmbeddingBackend] = None,
                 planning: bool = True,
                 plan_mode: str = "advise",
                 induction_assessment: str = "advise",
                 prediction_mode: str = "h-x-b-value"):
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
        # World-model M2: structured outcome predictions (shadow mode).
        # The provider is EXPLICITLY injected by the caller — never a
        # hidden default model, never an implicit network call. Without one,
        # predict_outcome returns not_configured and every existing path
        # is untouched.
        self.world_model = world_model or NotConfiguredProvider()
        self.predictions = PredictionService(self.store, self.world_model)
        # World-model M3: bounded planning. ``planning=False`` restores the
        # exact M2 behaviour (plan_next refuses with status=disabled);
        # ``plan_mode`` is "advise" (a suggestion is produced for the agent
        # to accept or reject) or "shadow" (paths are evaluated and
        # recorded but no suggestion field is emitted — observation only).
        # There is no unattended auto-execution mode, ever.
        if plan_mode not in ("advise", "shadow"):
            raise ValueError("plan_mode must be 'advise' or 'shadow'")
        self.planning = bool(planning)
        self.plan_mode = plan_mode
        # World-model M4: offline maintenance assessment.
        # "advise" = recommendations emitted for the agent to explicitly
        # accept/reject; "shadow" = evaluations recorded without advice;
        # "disabled" = returns disabled status without model calls.
        if induction_assessment not in ("advise", "shadow", "disabled"):
            raise ValueError(
                "induction_assessment must be 'advise', 'shadow', or 'disabled'")
        self.induction_assessment_mode = induction_assessment
        # World-model M6: what the world model is asked to predict, and
        # whether a knowledge term may influence the choice.
        #
        #   x-b-only     predict X/B only: no knowledge targets are
        #                proposed and knowledge_changes are not requested
        #   h-x-b        predict H too (targets proposed, changes kept and
        #                evaluated) but the knowledge term is forced off
        #   h-x-b-value  the knowledge term is live at the configured delta
        #
        # ``delta`` DEFAULTS TO 0.0 on purpose: an unmodified call must
        # score exactly as it did before the term existed. A positive delta
        # is a deliberate experimental choice pulled from this instance's
        # own configuration, never inherited implicitly.
        if prediction_mode not in PREDICTION_MODES:
            raise ValueError(
                f"prediction_mode must be one of {PREDICTION_MODES}")
        self.prediction_mode = prediction_mode
        self.delta = float(delta)
        self.selector.delta = self.delta
        # Retrieval embedding backend: EXPLICITLY injected by the caller
        # (same discipline as ``world_model``), else read from the
        # OR_EMBEDDING_* environment, else None. None is a first-class
        # state, not a failure: retrieval falls back to the profile path and
        # says so (``degraded``). The backend is a DISCOVERY capability —
        # it never scores, filters, or widens applicability.
        self.embedding_backend = (embedding if embedding is not None
                                 else create_embedding_backend())
        self.embedding_index = (EmbeddingIndex(
            index_dir_for(self.home), self.embedding_backend)
            if self.embedding_backend is not None else None)
        # Index synchronization (writes) — deliberately separate from recall
        # (read-only). A failed sync is reported, never fatal.
        self.index_sync = IndexSynchronizer(self.bank, self.sbank,
                                           self.catalog, self.store,
                                           self.embedding_index)
        # Episode budget declarations (task_id/episode_id -> {dim: limit}).
        # Declarations PERSIST in the store's meta table (declare_budget);
        # this dict is only an in-memory cache of what this instance has
        # loaded or declared. Snapshots freeze the declaration they were
        # taken under.
        self._episode_budgets: Dict[str, Dict[str, float]] = {}

    # -- capabilities ----------------------------------------------------------

    def _episode_progress(self, task_id: str,
                          episode_id: Optional[str],
                          *, task_digest: Optional[str] = None
                          ) -> Dict[str, Any]:
        """The episode's CURRENT information state (X): the latest real
        task_progress among this episode's snapshots.

        Continuity rule: each action's post snapshot merges its own progress
        update into the accumulated state, so the next action's pre state
        inherits what earlier actions established (selected_plan, model
        artifact, current solution, verification evidence...). Hypothetical
        snapshots NEVER contribute — an imagined outcome is not progress.

        Task boundary (M6): progress is inherited only from snapshots taken
        under the SAME task version. The caller passes the current
        ``task_digest``; a snapshot whose recorded ``problem_state``
        digest differs describes a DIFFERENT task context and must not be
        inherited. Checking here — at state-construction time — is what
        keeps a stale X from being used for a whole planning round before
        anything downstream notices; the bind-time check alone would be too
        late. Unknown digests (legacy snapshots without one) are treated as
        matching, since their version cannot be established either way.
        """
        if episode_id is None:
            return {}
        best: Optional[BeliefSnapshot] = None
        for snap in self.snapshots(task_id=task_id):
            if snap.episode_id != episode_id or snap.hypothetical:
                continue
            recorded = (snap.problem_state or {}).get("task_digest")
            if (task_digest is not None and recorded is not None
                    and recorded != task_digest):
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
        harness_state = self._harness_state_view(task, knowledge)
        budget = self._load_budget(str(task.get("task_id", "")), episode_id)
        budget_state = self.budget.view(
            str(task.get("task_id", "")), episode_id, budget=budget)
        progress = self._episode_progress(
            str(task.get("task_id", "")), episode_id,
            task_digest=task_text_digest(task))
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

    def _harness_state_view(self, task: Dict[str, Any],
                            knowledge: Dict[str, Any]) -> Dict[str, Any]:
        """The frozen ``harness_state`` block of a belief snapshot.

        This is EVIDENCE ABOUT H, not a measured H: value-copied knowledge
        references, experience volume, and tool configuration. The name and
        the shape are kept for backward compatibility with stored snapshots;
        the unified capability view is
        :meth:`capability_evidence`, which labels each item's evidential
        status and defines no composite score.
        """
        recent = self.bank.query(task_id=str(task.get("task_id", "")))
        return {
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

    # -- unified world-model contracts ------------------------------------------

    def capability_evidence(self, task: Optional[Dict[str, Any]] = None, *,
                            snapshot: Optional[BeliefSnapshot] = None
                            ) -> HarnessCapabilityEvidence:
        """Observable evidence about the harness capability state H.

        ``H = F(M, W_OR, Pi, R, T)``: five interacting sources, carried with
        an explicit evidence STATUS each and NO composite score. Knowledge
        references, experience counts and tool configuration are evidence
        ABOUT H — they are never renamed into a measured capability level.

        Built from a belief snapshot when one is supplied (its frozen
        ``harness_state`` + ``coverage``), otherwise from the live banks.
        Sources nothing has observed stay ``no_evidence`` rather than being
        filled from an unrelated count.

        A ``task`` is optional: with neither a task nor a snapshot the
        knowledge component cannot be scoped to a structural cell, so it is
        omitted (and said to be omitted) rather than reported as an empty
        knowledge base — "not consulted" and "nothing there" are different
        facts.
        """
        if snapshot is not None:
            return capability_evidence_from_legacy_harness_state(
                getattr(snapshot, "harness_state", None) or {},
                coverage=getattr(snapshot, "coverage", None) or {})
        if task is not None:
            return self.capability_evidence(
                snapshot=self.snapshot(task))
        evidence = capability_evidence_from_legacy_harness_state({
            "experience": {"total_executions": self.bank.count()},
            "tool_config": {
                "available_solver_families": available_families(),
                "executor_timeout_seconds": getattr(
                    self.executor, "timeout_seconds", None),
            },
        })
        evidence.notes.append(
            "no task or snapshot supplied: the knowledge component is "
            "cell-scoped and was NOT consulted, so M reports evidence "
            "volume only — an empty cell list is not evidence of no "
            "knowledge")
        return evidence

    def strategy_execution_window(self, task_id: str,
                                  episode_id: Optional[str] = None, *,
                                  strategy_id: Optional[str] = None,
                                  in_scope_action_types: Optional[
                                      Sequence[str]] = None,
                                  auxiliary_action_types: Optional[
                                      Sequence[str]] = None
                                  ) -> StrategyExecutionWindow:
        """The REAL strategy execution window derived from the action log.

        One ``execute_strategy`` call is one attempt; a window is the larger
        scope a strategy actually occupies (modeling, solving, repairing,
        verifying). Only a window that really corresponds to recorded
        actions is ``comparable`` — the flag a window-scope prediction needs
        before it may be scored.
        """
        return window_from_records(
            self, task_id, episode_id, strategy_id,
            in_scope_action_types=in_scope_action_types,
            auxiliary_action_types=auxiliary_action_types)

    def build_strategy_outcome_contract(
            self, task: Dict[str, Any],
            candidate: Union[CandidateRef, ActionSpec],
            episode_id: Optional[str] = None,
            *, benefit: Optional[BenefitEstimate] = None,
            cost: Optional[ExpectedCost] = None,
            risk: Optional[RiskStatement] = None,
            uncertainty: Optional[UncertaintyStatement] = None,
            evidence_basis: Optional[Sequence[EvidenceRef]] = None,
            unsupported_fields: Optional[Dict[str, str]] = None,
            window: Optional[StrategyExecutionWindow] = None,
            ) -> StrategyOutcomePrediction:
        """Build (and validate) a :class:`StrategyOutcomePrediction`.

        The trace is filled from what the framework already knows: the
        frozen input snapshot, its task digest, the prediction-service
        availability of THIS instance, and the generation time. A caller
        never re-types the contract version or the prediction type.

        With no prediction service configured the returned contract carries
        ``status="contract_only"`` — the contract is implemented, the
        service is not attached — and the notes say so. Nothing is invented
        to make it look like a prediction happened.
        """
        if isinstance(candidate, ActionSpec):
            candidate = CandidateRef.from_action_spec(candidate)
        elif isinstance(candidate, dict):
            candidate = CandidateRef.from_dict(candidate)
        snapshot = self.snapshot(task, episode_id)
        if candidate.task_id and candidate.task_id != snapshot.task_id:
            raise ValueError(
                f"candidate.task_id {candidate.task_id!r} does not match the "
                f"task being predicted for ({snapshot.task_id!r})")
        candidate = copy.deepcopy(candidate)
        if not candidate.task_id:
            candidate.task_id = snapshot.task_id
        if candidate.episode_id is None:
            candidate.episode_id = episode_id
        trace = PredictionTrace(
            prediction_kind="strategy_outcome",
            input_snapshot_id=snapshot.snapshot_id,
            input_version=(snapshot.problem_state or {}).get("task_digest"),
            prediction_version=self.prediction_service_version(),
            evidence_basis=list(evidence_basis or []),
            unsupported_fields=dict(unsupported_fields or {}),
            comparable=False,
        )
        notes: List[str] = []
        if candidate.scope == "strategy_window":
            window = window or self.strategy_execution_window(
                snapshot.task_id, episode_id,
                strategy_id=candidate.strategy_id)
            candidate.window_id = candidate.window_id or window.window_id
            trace.comparable = bool(window.comparable)
            trace.not_comparable_reasons = list(window.not_comparable_reasons)
            notes.append(
                "window scope: " + window.scope_rationale)
        else:
            trace.comparable = True
            notes.append(
                "attempt scope: this prediction covers ONE solve attempt, "
                "not the whole strategy execution (modeling / repair / "
                "verify are auxiliary and reported separately)")
        service = self.prediction_service_available("strategy_outcome")
        if not service:
            notes.append(
                "contract_only: no prediction service is attached to this "
                "instance, so no benefit/cost/risk values were produced")
        prediction = StrategyOutcomePrediction(
            candidate=candidate,
            status="valid" if service else "contract_only",
            benefit=benefit,
            cost=cost,
            risk=risk,
            uncertainty=uncertainty,
            trace=trace,
            service_available=service,
            notes=notes,
        )
        problems = validate_strategy_outcome(prediction)
        if problems:
            prediction.status = "invalid"
            prediction.notes.extend(f"validation: {p}" for p in problems)
        return prediction

    def build_capability_evolution_contract(
            self, task: Optional[Dict[str, Any]] = None,
            operation: Optional[LearningOperation] = None, *,
            snapshot: Optional[BeliefSnapshot] = None,
            experience_scope: Optional[ExperienceScope] = None,
            task_targeting: Optional[TaskTargeting] = None,
            baseline: Optional[BaselineStatement] = None,
            horizon: str = "",
            horizon_tasks: Optional[int] = None,
            expected_changes: Optional[Sequence[ExpectedChange]] = None,
            learning_cost: Optional[ExpectedCost] = None,
            degradation_risk: Optional[RiskStatement] = None,
            uncertainty: Optional[UncertaintyStatement] = None,
            verification_conditions: Optional[
                Sequence[VerificationCondition]] = None,
            evidence_basis: Optional[Sequence[EvidenceRef]] = None,
            unsupported_fields: Optional[Dict[str, str]] = None,
            ) -> CapabilityEvolutionPrediction:
        """Build (and validate) a :class:`CapabilityEvolutionPrediction`.

        The current capability evidence is read from the frozen snapshot
        when one is available, otherwise from the live banks. Without a
        prediction service the contract is returned as ``contract_only``:
        the schema is implemented, the capability prediction service is NOT
        — and the object says exactly that instead of implying a capability
        forecast was made.
        """
        if snapshot is None and task is not None:
            snapshot = self.snapshot(task)
        evidence = self.capability_evidence(task, snapshot=snapshot)
        operation = operation or LearningOperation(
            operation_type="induce",
            description="(no candidate operation supplied)")
        trace = PredictionTrace(
            prediction_kind="capability_evolution",
            input_snapshot_id=(snapshot.snapshot_id
                               if snapshot is not None else ""),
            input_version=((snapshot.problem_state or {}).get("task_digest")
                           if snapshot is not None else None),
            prediction_version=self.prediction_service_version(),
            evidence_basis=list(evidence_basis or []),
            unsupported_fields=dict(unsupported_fields or {}),
            comparable=False,
        )
        notes: List[str] = []
        service = self.prediction_service_available("capability_evolution")
        if not service:
            notes.append(
                "contract_only: this build implements the capability "
                "evolution CONTRACT; no capability prediction service is "
                "attached, so no performance change is forecast. H is judged "
                "through observable consequences, never a latent vector")
        if not verification_conditions:
            notes.append(
                "no verification condition declared: predicting, binding "
                "the fact, and verifying the effect are three different "
                "things, and only the third supports a claim of improvement")
        prediction = CapabilityEvolutionPrediction(
            current_evidence=evidence,
            candidate_operation=operation,
            status="valid" if service else "contract_only",
            experience_scope=experience_scope,
            task_targeting=task_targeting,
            baseline=baseline,
            horizon=horizon,
            horizon_tasks=horizon_tasks,
            expected_changes=list(expected_changes or []),
            learning_cost=learning_cost,
            degradation_risk=degradation_risk,
            uncertainty=uncertainty,
            verification_conditions=list(verification_conditions or []),
            trace=trace,
            service_available=service,
            notes=notes,
        )
        problems = validate_capability_evolution(prediction)
        if problems:
            prediction.status = "invalid"
            prediction.notes.extend(f"validation: {p}" for p in problems)
        return prediction

    def prediction_service_available(self, kind: str) -> bool:
        """Whether a real prediction SERVICE is attached for one kind.

        Both kinds share this build's single explicit provider; the split is
        named per kind so a future deployment can wire them independently
        without changing the contract.
        """
        if kind not in PREDICTION_KINDS:
            raise ValueError(f"unknown prediction kind {kind!r}")
        return not isinstance(self.world_model, NotConfiguredProvider)

    def prediction_service_version(self) -> str:
        """Version label of the attached prediction service (or its absence)."""
        if isinstance(self.world_model, NotConfiguredProvider):
            return "not-attached"
        return str(getattr(self.world_model, "name", "unknown"))

    @staticmethod
    def read_prediction_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
        """Read any stored prediction payload, whatever generation it is.

        Returns a dict with ``contract_version`` (as identified, never
        guessed), ``legacy`` (True for an unversioned payload), and either
        ``contract`` (a loaded current-contract object) or ``legacy_view``
        (the mapped view of an old payload) plus ``unmappable`` explaining
        what could NOT be converted.
        """
        version = detect_payload_version(payload)
        if version == LEGACY_CONTRACT_VERSION:
            view = legacy_prediction_view(payload)
            return {"contract_version": version, "legacy": True,
                    "supported": True, "legacy_view": view,
                    "unmappable": dict(LEGACY_UNMAPPABLE),
                    "note": ("read through the legacy view: the old "
                             "semantics are preserved and no new capability "
                             "increment, risk severity or measurement is "
                             "derived")}
        if version.startswith(PAYLOAD_VERSION_UNKNOWN_PREFIX):
            return {"contract_version": version, "legacy": False,
                    "supported": False,
                    "error": (f"unsupported contract version "
                              f"{version[len(PAYLOAD_VERSION_UNKNOWN_PREFIX):]!r}: "
                              "this build will not guess at its semantics")}
        try:
            contract = load_contract_payload(payload)
        except ValueError as exc:
            return {"contract_version": version, "legacy": False,
                    "supported": False, "error": str(exc)}
        return {"contract_version": version, "legacy": False,
                "supported": True, "contract": contract.to_dict()}

    # -- budget / actions ------------------------------------------------------

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
        # conflict raises before anything persists. The check lives ONLY
        # in ActionLog.end_action (one completion rule, one replay rule):
        # the API layer must not maintain a second fingerprint comparison
        # — it once treated "no cost argument" as an empty fingerprint and
        # rejected a legitimate replay of an action whose cost had been
        # amended beforehand.
        prior = self.actions.get(action_id)
        if prior is None:
            raise StorageError(f"unknown action_id {action_id!r}")
        replayed = prior.status != "running"
        # (2) Persist result + cost (idempotent replay returns the stored
        # record unchanged; conflicts raise StorageError).
        record = self.actions.end_action(
            action_id, status=status, outcome=outcome, cost=cost_vector,
            linked_execution_id=linked_execution_id, rollup=rollup)
        if replayed:
            return {"action_id": record.action_id,
                    "status": record.status,
                    "post_snapshot_id": record.post_snapshot_id}
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
        """Report an action the OUTER agent performed (model /
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

    # -- world-model M2: structured outcome prediction (shadow) ------------------

    def predict_outcome(self, task: Dict[str, Any],
                        action_spec: ActionSpec,
                        episode_id: Optional[str] = None,
                        *, parent_action_id: Optional[str] = None
                        ) -> OutcomePrediction:
        """Predict the consequences of a CANDIDATE action from the current
        frozen state — explicitly, in shadow mode.

        The input snapshot is frozen BEFORE anything else happens; the
        provider is invoked exactly once; the prediction (including its
        own call cost) is persisted frozen. The existing strategy flow is
        untouched: this prediction never changes recall, predict_cost,
        execute, or record behaviour.

        ``parent_action_id`` (optional): the action this prediction call
        belongs to (e.g. the selection process that is evaluating
        candidates). The model call's own cost is charged to it as an
        own-cost amendment — never to the PREDICTED action, which has not
        happened.

        Identity discipline: the prediction belongs to the SAME task and
        episode as the snapshot it is conditioned on. Before calling the
        provider, the spec's task_id/episode_id are reconciled with the
        call arguments (explicit spec values that disagree are overridden
        and the adjustment recorded in ``model_info.identity_adjusted``) —
        an episode-less spec is never persisted as a wildcard that later
        leaks this call's cost into every episode's budget."""
        task_id = str(task.get("task_id", ""))
        adjusted: Dict[str, Any] = {}
        if action_spec.task_id != task_id:
            adjusted["task_id"] = {"spec": action_spec.task_id,
                                   "call": task_id}
            action_spec.task_id = task_id
        if action_spec.episode_id != episode_id:
            if action_spec.episode_id is not None:
                adjusted["episode_id"] = {"spec": action_spec.episode_id,
                                          "call": episode_id}
            action_spec.episode_id = episode_id
        snap = self.snapshot(task, episode_id)
        targets = self.knowledge_targets(snap, action_spec)
        prediction = self.predictions.predict_outcome(
            task, action_spec, snap,
            knowledge_targets=targets,
            reliability=self.prediction_reliability_table())
        # The full target set is frozen WITH the prediction: a later verdict
        # needs the claim's interval / entry id / strategy as they were at
        # prediction time, not a re-read of banks that may have moved since.
        prediction.model_info["knowledge_targets_proposed"] = [
            t.strategy_id for t in targets]
        prediction.model_info["knowledge_targets_proposed_full"] = [
            t.to_dict() for t in targets]
        self.predictions._save(prediction)
        if adjusted:
            prediction.model_info["identity_adjusted"] = adjusted
            self.predictions._save(prediction)
        self._charge_call_cost(prediction, parent_action_id)
        return prediction

    @staticmethod
    def _call_cost_dims(prediction) -> Dict[str, float]:
        """The model call's OWN measured spend, per dimension (empty when
        the provider reported no usage — unknown, never zero)."""
        if prediction.call_cost is None:
            return {}
        return {d: getattr(prediction.call_cost, d) for d in
                prediction.call_cost.measured_dims()}

    def _charge_call_cost(self, prediction,
                          parent_action_id: Optional[str]
                          ) -> Dict[str, float]:
        """Record the model call's own spend, charged to the parent action
        when one is named.

        INCREMENT semantics throughout: each prediction call is a separate
        real spend, so two 50-token calls under one selection action total
        100 rather than overwriting each other. With no parent action the
        cost stays on the prediction itself and is aggregated into the
        budget view by ``BudgetLedger.consumption`` — never silently
        dropped, and never counted twice (the
        ``charged_to_parent_action`` mark keeps it on one side only)."""
        dims = self._call_cost_dims(prediction)
        if not dims or parent_action_id is None:
            return dims
        if self.actions.get(parent_action_id) is None:
            raise StorageError(
                f"unknown parent_action_id {parent_action_id!r}")
        self.actions.amend_action_cost_increment(parent_action_id, **dims)
        prediction.model_info["charged_to_parent_action"] = parent_action_id
        self.predictions._save(prediction)
        return dims

    def knowledge_targets(self, snapshot, action_spec) -> List[Any]:
        """The framework's structural knowledge-target proposals for ONE
        candidate action (M6).

        A transparent heuristic, not a value estimate: it says what COULD
        be learned and which quantities a value would need. Under
        ``x-b-only`` no targets are proposed at all, which is exactly what
        makes that mode reproduce the pre-M6 request byte for byte.

        The cell's real evidence records are fetched via
        ``ConditionalStats.evidence`` because ``GroupStats`` carries no
        distinct-task count, and reuse basis must count INDEPENDENT TASKS
        (repeat runs of one task are one replication).
        """
        if self.prediction_mode == "x-b-only":
            return []
        from or_harness.world_model.knowledge import learning_needs
        strategy_id = getattr(action_spec, "strategy_id", None)
        records = None
        if strategy_id:
            try:
                profile = ProblemProfile.from_dict(
                    (snapshot.problem_state or {}).get("profile") or {})
                records = self.stats.evidence(profile, strategy_id)
            except Exception:
                records = None
        return learning_needs(snapshot, action_spec, records=records)

    def evaluate_knowledge_execution(self, record: ExecutionRecord
                                     ) -> Dict[str, Any]:
        """Judge the ``after_execution`` knowledge predictions a REAL
        execution just made possible (M6, fast channel).

        Finds the predictions bound to this record's action, re-reads the
        targets those predictions were made against, and records one verdict
        per unresolved ``after_execution`` item. Stage-partitioned: a
        prediction whose immediate X/B comparison already ran is still
        eligible here, and re-running this is a no-op for stages that
        already have a verdict.

        Prediction-conditioned targets live on the prediction itself
        (``predicted.knowledge_changes``), so this needs no new linkage: no
        side table, no event log.
        """
        from or_harness.world_model.knowledge_track import (
            STAGE_EXECUTION,
            evaluate_execution,
            unresolved_stages,
        )
        action_id = getattr(record, "action_id", None)
        action = self.actions.get(action_id) if action_id else None
        if action is None:
            # The record may have been appended without the action handle
            # (a bare ``record`` call). Fall back to the action whose
            # linked execution is this one, matched over the same task.
            for candidate in self.actions.query(task_id=record.task_id):
                if candidate.linked_execution_id == record.execution_id:
                    action = candidate
                    break
        if action is None:
            return {}
        out: Dict[str, Any] = {}
        for prediction in self.predictions_query(task_id=record.task_id):
            if prediction.bound_action_id != action.action_id:
                continue
            if prediction.status != "valid":
                continue
            # A binding already reported as mismatched means the executed
            # strategy/solver differs from the predicted candidate: this
            # execution is evidence about a DIFFERENT claim, so no H verdict
            # may be drawn from it. (The X/B comparison already records the
            # mismatch and skips scoring; the knowledge channel must agree.)
            if prediction.binding_mismatch:
                continue
            # The judgement is about THIS execution, so the prediction's
            # action must be the one that produced it.
            if action.linked_execution_id != record.execution_id:
                continue
            targets = self._prediction_targets(prediction)
            for index, stage in unresolved_stages(prediction):
                if stage != STAGE_EXECUTION:
                    continue
                item = prediction.predicted["knowledge_changes"][index]
                target = self._match_target(targets, item)
                # Let the verdict know which task the prediction was made
                # for, so an `independent_replication` precondition can be
                # judged (it needs evidence from a DIFFERENT task).
                item = dict(item)
                item["_predicted_for_task"] = prediction.action_spec.task_id
                status, detail = evaluate_execution(item, target, record)
                detail["targets_available"] = bool(targets)
                entry = self._record_knowledge_verdict(
                    prediction, index, stage, status, detail)
                if entry is not None:
                    out[prediction.prediction_id] = entry
        return out

    @staticmethod
    def _prediction_targets(prediction) -> List[Any]:
        """The structural targets a prediction was made against.

        Read from the prediction's own frozen copy, never re-derived from
        the current banks: a verdict must judge the claim as it stood when
        the prediction was made, and an entry's interval may have moved
        since."""
        from or_harness.world_model.knowledge import KnowledgeTarget
        return [KnowledgeTarget.from_dict(t) for t in
                (prediction.model_info.get("knowledge_targets_proposed_full")
                 or [])]

    def _record_knowledge_verdict(self, prediction, index: int, stage: str,
                                  status: str, detail: Dict[str, Any]
                                  ) -> Optional[Dict[str, Any]]:
        """Persist one stage verdict; return its result entry, or None when
        that stage already had one (the write is idempotent per stage)."""
        from or_harness.world_model.knowledge_track import record_stage_verdict
        if not record_stage_verdict(prediction, index, stage, status, detail):
            return None
        self.predictions._save(prediction)
        return {"change_index": index, "stage": stage, "status": status}

    @staticmethod
    def _match_target(targets: List[Any], item: Dict[str, Any]):
        """The proposed target a knowledge-change item refers to, or None."""
        target = item.get("target") or {}
        if target.get("kind") == "existing_entry":
            for candidate in targets:
                if (candidate.kind == "existing_entry"
                        and candidate.entry_id == target.get("entry_id")):
                    return candidate
        for candidate in targets:
            if candidate.strategy_id == target.get("strategy_id"):
                return candidate
        return None

    @staticmethod
    def _induction_transition(result: Dict[str, Any]) -> Dict[str, Any]:
        """The knowledge transition an ``induce`` call recorded.

        Accepts all three shapes the transition travels in: the ``induce``
        result's own flat keys, the action summary nested under ``action``,
        and the maintenance action outcome that summary was built from. The
        delta is a before/after transition, computed where both states are
        known; this only locates it, and returns empty pieces when the
        induction never ran (dry run, or a refusal)."""
        source = result or {}
        action = source.get("action") or {}
        outcome = action.get("outcome") or {}
        delta = (source.get("knowledge_delta")
                 or action.get("knowledge_delta")
                 or outcome.get("knowledge_delta") or {})
        after = (source.get("knowledge_after")
                 or action.get("knowledge_after")
                 or outcome.get("knowledge_after") or {})
        return {
            "entry_changes": delta.get("entry_changes") or [],
            "created": delta.get("entries_created") or [],
            "knowledge_after": after,
        }

    def evaluate_knowledge_consolidation(self, result: Dict[str, Any],
                                         execution_ids: Sequence[str] = (),
                                         strategy_id: Optional[str] = None
                                         ) -> Dict[str, Any]:
        """Judge the ``after_consolidation`` knowledge predictions an
        induction just made possible (M6, slow channel).

        Eligibility has THREE requirements, and dropping any one of them
        lets an untouched prediction be scored as a hit:

        1. **the prediction was bound to a real action** — an unexecuted
           prediction describes something that never happened, so no
           induction can be its outcome;
        2. **that action's execution is inside the induction's own evidence
           scope** — the induction must have actually consolidated the
           evidence the prediction was about, not merely run nearby;
        3. **the induction touched the entry or strategy the prediction
           named** — the opportunity window must really be open.

        A prediction failing any of these has not had its chance and stays
        ``pending``; it is never counted as a miss, and never as a hit.
        """
        from or_harness.world_model.knowledge_track import (
            STAGE_CONSOLIDATION,
            evaluate_consolidation,
            unresolved_stages,
        )
        delta = self._induction_transition(result)
        entry_changes = delta["entry_changes"]
        created = delta["created"]
        knowledge_after = delta["knowledge_after"]
        # NOTE: an induction that changed nothing is NOT a no-op for
        # evaluation purposes. If it genuinely ran over a prediction's own
        # evidence and its own strategy, the opportunity HAS arrived and the
        # absence of a change is a real (negative) outcome. Returning early
        # here would leave such predictions unevaluated forever and bias the
        # reliability statistics toward the cases that succeeded.
        created_strategies: List[str] = []
        # ``knowledge_after`` travels in two shapes: the layered view
        # (verified/unverified/legacy_unknown, also used by the induction
        # ACTION) and the flat ``{"entries": [...], "entry_count": N}``
        # form the action summary carries. Reading both keeps one code path.
        entries_after: List[Dict[str, Any]] = []
        for layer in ("verified", "unverified", "legacy_unknown"):
            entries_after.extend(knowledge_after.get(layer) or [])
        entries_after.extend(knowledge_after.get("entries") or [])
        for entry in entries_after:
            if entry.get("entry_id") in set(created):
                created_strategies.append(
                    str(entry.get("strategy_id") or ""))
        touched = {c.get("entry_id") for c in entry_changes}
        # The induction's evidence scope. Two shapes, because the caller may
        # or may not name one:
        #
        # - an explicit ``execution_ids`` (the bundle path) is a STRICT
        #   scope: only those executions took part.
        # - without one, ``induce`` consolidates whatever evidence the
        #   covered cells hold, so the scope is every real execution of the
        #   strategies this induction acted on. Demanding an explicit scope
        #   there would empty the whole channel on the normal CLI path.
        #
        # Either way the scope is about REAL, RECORDED evidence: an
        # execution that was never banked cannot be inside it.
        explicit_scope = {str(e) for e in (execution_ids or ())}
        banked = {r.execution_id for r in self.bank.all()}
        out: Dict[str, Any] = {}
        for prediction in self.predictions_query():
            if prediction.status != "valid":
                continue
            if prediction.action_spec.action_type != "execute_strategy":
                continue
            # (1) A prediction that was never bound to a real action
            #     describes an action that never ran: no induction can be
            #     its outcome, so it is not evaluable at all. Left
            #     untouched (and therefore still pending), never scored as
            #     a hit.
            action = (self.actions.get(prediction.bound_action_id)
                      if prediction.bound_action_id else None)
            if action is None:
                continue
            # A binding already reported as mismatched means the executed
            # action differs from the predicted candidate: whatever
            # evidence exists is evidence about a DIFFERENT claim.
            if prediction.binding_mismatch:
                continue
            # (2) The execution must be real, recorded evidence, and inside
            #     the induction's scope when one was named.
            execution_id = action.linked_execution_id
            if not execution_id or execution_id not in banked:
                continue
            if explicit_scope and execution_id not in explicit_scope:
                continue
            targets = self._prediction_targets(prediction)
            for index, stage in unresolved_stages(prediction):
                if stage != STAGE_CONSOLIDATION:
                    continue
                item = prediction.predicted["knowledge_changes"][index]
                target = self._match_target(targets, item)
                # (3) Opportunity window: the induction must have touched
                #     what the prediction named, or covered the strategy it
                #     named. Otherwise it had no chance to resolve and
                #     stays pending — never "missed".
                if target is None:
                    continue
                touched_now = (
                    (target.entry_id is not None
                     and target.entry_id in touched)
                    or (target.entry_id is not None
                        and target.entry_id in set(created))
                    or (target.strategy_id in created_strategies))
                # A scoped induction that ran for THIS prediction's own
                # strategy over its own recorded evidence is an opportunity
                # even when it changed nothing: the empty transition is
                # precisely what makes the prediction `missed` rather than
                # permanently pending. Returning early on "no change" would
                # leave such predictions unevaluated forever and bias the
                # reliability statistics toward the successes.
                scoped_to_this = (strategy_id is not None
                                  and target.strategy_id == strategy_id)
                if not (touched_now or scoped_to_this):
                    continue
                status, detail = evaluate_consolidation(
                    item, target, entry_changes, created, created_strategies)
                detail["consolidated_execution"] = execution_id
                detail["scope_basis"] = ("explicit execution_ids"
                                         if explicit_scope
                                         else "cell evidence of the "
                                              "covered strategy")
                if not entry_changes and not created:
                    detail["induction_changed_nothing"] = True
                entry = self._record_knowledge_verdict(
                    prediction, index, stage, status, detail)
                if entry is not None:
                    out[prediction.prediction_id] = entry
        return out

    def prediction_reliability_table(self) -> Dict[str, Any]:
        """Measured reliability of PAST knowledge predictions, by class.

        This is the self-correction channel: what the model's predictions
        have actually been worth, computed by the framework from evaluated
        verdicts. It is deliberately NOT the model's own confidence — that
        would let a model raise its own standing by asserting it. Classes
        with too few resolved samples report ``None``, and an unknown
        reliability grants no knowledge value.
        """
        from or_harness.world_model.knowledge_track import (
            class_reliability,
            collect_verdicts,
        )
        return class_reliability(collect_verdicts(self.predictions_query()))

    def bind_outcome(self, prediction_id: str,
                     action_id: str) -> OutcomePrediction:
        """Bind a prediction to the real action that ran. Request identity
        is checked (type/task/episode/strategy/solver/timing/scope);
        mismatches recorded, not silently compared."""
        action = self.actions.get(action_id)
        if action is None:
            raise StorageError(f"unknown action_id {action_id!r}")
        # The linked execution's measurement scope (when the execution is
        # already staged/recorded) — an attempt-scope prediction must not
        # be scored against a task-scope total.
        record_scope = None
        if action.linked_execution_id is not None:
            record_scope = (self.bank.get_pending(action.linked_execution_id)
                            or self.bank.get(action.linked_execution_id))
        return self.predictions.bind_outcome(prediction_id, action,
                                             record_scope=record_scope)

    def compare_prediction(self, prediction_id: str) -> OutcomePrediction:
        """Compare the frozen prediction against the bound action's real
        execution (appended feedback; idempotent; never re-invokes the
        model)."""
        prediction = self.predictions.get(prediction_id)
        if prediction is None:
            raise StorageError(
                f"unknown prediction_id {prediction_id!r}")
        if prediction.bound_action_id is None:
            raise StorageError(
                f"prediction {prediction_id!r} is not bound to an action")
        action = self.actions.get(prediction.bound_action_id)
        if action is None or action.linked_execution_id is None:
            raise StorageError(
                f"bound action has no linked execution to compare against")
        record = self.bank.get_pending(action.linked_execution_id) \
            or self.bank.get(action.linked_execution_id)
        if record is None:
            raise StorageError(
                f"linked execution {action.linked_execution_id!r} not found")
        return self.predictions.compare_prediction(prediction_id, record)

    def get_prediction(self, prediction_id: str) -> Optional[OutcomePrediction]:
        return self.predictions.get(prediction_id)

    def predictions_query(self, *, task_id: Optional[str] = None,
                          episode_id: Optional[str] = None
                          ) -> List[OutcomePrediction]:
        return self.predictions.query(task_id=task_id, episode_id=episode_id)

    # -- world-model M3: bounded planning ----------------------------------------

    def _candidate_specs(self, task: Dict[str, Any],
                         episode_id: Optional[str],
                         candidates: Optional[Sequence[ActionSpec]],
                         limit: int) -> List[ActionSpec]:
        """Assemble the bounded root-candidate list.

        Sources (no new LLM agent, no retrieval system):
        1. caller-supplied ActionSpecs (taken as-is, identity reconciled by
           predict-time discipline);
        2. otherwise the catalog vocabulary, filtered by applicability and
           available solver families, turned into execute_strategy specs
           whose params FREEZE the strategy description (meaning, actions,
           fallback) the candidate stands for. With no memory this menu
           carries NO fabricated performance claims — the prediction step
           is where consequences come from."""
        task_id = str(task.get("task_id", ""))
        if candidates:
            specs = []
            for spec in list(candidates)[:limit]:
                spec = ActionSpec.from_dict(spec.to_dict())  # value copy
                if not spec.task_id:
                    spec.task_id = task_id
                if spec.episode_id is None:
                    spec.episode_id = episode_id
                specs.append(spec)
            return specs
        from or_harness.adapters.solver import available_families
        profile = self.profile(task)
        families = set(available_families())
        specs = []
        for strategy in self.catalog.values():
            if len(specs) >= limit:
                break
            if strategy.solver_family and families \
                    and strategy.solver_family not in families:
                continue  # tool capability filter
            from or_harness.core.schema import profile_matches
            if strategy.applicability and not profile_matches(
                    profile, strategy.applicability):
                continue
            specs.append(ActionSpec(
                action_type="execute_strategy",
                task_id=task_id,
                episode_id=episode_id,
                strategy_id=strategy.strategy_id,
                params={
                    "strategy_name": strategy.name,
                    "strategy_type": strategy.strategy_type,
                    "actions": list(strategy.actions),
                    "fallback_strategy_id": strategy.fallback,
                    "solver_family": strategy.solver_family,
                },
            ))
        return specs

    def plan_next(self, task: Dict[str, Any],
                  episode_id: Optional[str] = None, *,
                  candidates: Optional[Sequence[ActionSpec]] = None,
                  limits: Optional[Any] = None,
                  second_step: Optional[Sequence[ActionSpec]] = None
                  ) -> Dict[str, Any]:
        """Bounded next-step planning over predicted action consequences.

        Freezes ONE root snapshot, evaluates <= limits.max_root_candidates
        root candidates (horizon 1 or 2; the second step is predicted FROM
        the hypothetical successor state of the first), and recommends the
        first step of the best path — the agent then explicitly accepts,
        rejects, or overrides it (``choose_next``).

        The decision is recorded as a real ``select_strategy`` action whose
        own cost carries the planning calls' spend (also aggregated in the
        returned plan). Nothing is executed by this call; the suggestion
        never writes X.selected_plan — only ``choose_next`` does."""
        from or_harness.world_model.planner import (
            PlanLimits,
            PlanResult,
            PLANNABLE_ACTION_TYPES,
            build_hypothetical_successor,
            comparison_norms,
            evaluate_path,
            second_step_dependency_ok,
        )
        limits = (limits if isinstance(limits, PlanLimits)
                  else PlanLimits.from_dict(limits))
        limits.horizon = max(1, min(2, limits.horizon))
        # Inherit the harness's configured evaluation yardstick unless the
        # caller explicitly overrode it: planning must not silently ignore
        # the alpha/beta/gamma and cost weights the rest of the system
        # scores with (an empty default would have made every cost
        # dimension weight zero).
        if limits.alpha is None:
            limits.alpha = self.selector.alpha
        if limits.beta is None:
            limits.beta = self.selector.beta
        if limits.gamma is None:
            limits.gamma = self.selector.gamma
        if limits.cost_weights is None:
            limits.cost_weights = dict(self.selector.cost_weights)
        if limits.delta is None:
            # Defaults to this harness's configured delta, which is 0.0
            # unless the caller deliberately chose otherwise — so an
            # unmodified call is scored exactly as before the knowledge
            # term existed, while ``prediction_mode="x-b-only"`` forces it
            # off regardless of configuration.
            limits.delta = (0.0 if self.prediction_mode == "x-b-only"
                            else self.delta)
        if self.prediction_mode != "h-x-b-value":
            limits.delta = 0.0
        task_id = str(task.get("task_id", ""))
        plan = PlanResult(plan_id=PlanResult.new_id(),
                          root_snapshot_id="",
                          decision_action_id=None,
                          task_id=task_id,
                          episode_id=episode_id,
                          limits=limits)
        if not self.planning:
            plan.status = "disabled"
            plan.truncation_reason = ("planning is disabled "
                                      "(ORHarness(planning=False))")
            return plan.to_dict()
        # (1) Freeze the root state ONCE for the whole decision.
        root = self.snapshot(task, episode_id)
        plan.root_snapshot_id = root.snapshot_id
        # Budget honesty up front: an exceeded REAL budget stops planning
        # (planning itself would spend more); unknown consumption is
        # reported, never claimed as "within budget".
        declared = self._load_budget(task_id, episode_id)
        budget_view = self.budget.view(task_id, episode_id,
                                       budget=declared)
        plan.budget_confirmation = (
            budget_view["status"] if declared else "unknown")
        if budget_view["status"] == "exceeded":
            plan.status = "fallback"
            plan.truncation_reason = (
                "declared budget already exceeded by real consumption; "
                "planning would spend more — returning without model "
                "calls. Report the current best solution instead.")
            return plan.to_dict()
        # (2) Bounded candidate list, with identity validation: every
        # candidate must belong to THIS task/episode (a spec naming
        # another task or episode is rejected, never silently re-labelled
        # — the root state and the prediction must describe the same
        # decision).
        specs = self._candidate_specs(task, episode_id, candidates,
                                      limits.max_root_candidates)
        valid_specs = []
        identity_conflicts = []
        for spec in specs:
            if spec.task_id and spec.task_id != task_id:
                identity_conflicts.append(
                    f"{spec.action_type}/{spec.strategy_id}: task_id "
                    f"{spec.task_id!r} != {task_id!r}")
                continue
            if (spec.episode_id is not None and episode_id is not None
                    and spec.episode_id != episode_id):
                identity_conflicts.append(
                    f"{spec.action_type}/{spec.strategy_id}: episode_id "
                    f"{spec.episode_id!r} != {episode_id!r}")
                continue
            spec.task_id = task_id
            spec.episode_id = episode_id
            valid_specs.append(spec)
        if identity_conflicts:
            plan.truncation_reason = (
                "candidate identity conflict (rejected): "
                + "; ".join(identity_conflicts))
        specs = [s for s in valid_specs
                 if s.action_type in PLANNABLE_ACTION_TYPES]
        if not specs:
            plan.status = "no_candidates"
            reason = ("no plannable candidates: planning currently "
                      f"supports {PLANNABLE_ACTION_TYPES} only, after "
                      "applicability and tool-capability filtering. Fall "
                      "back to `recall` for a memory-based ordering.")
            if plan.truncation_reason:
                reason = plan.truncation_reason + ". " + reason
            plan.truncation_reason = reason
            return plan.to_dict()
        # (3) Decision (parent) action: the planning spend lands here.
        decision = self.actions.begin_action(
            "select_strategy", task_id, episode_id, pre_snapshot=root,
            params={"kind": "plan_next", "horizon": limits.horizon,
                    "limits": limits.to_dict()})
        plan.decision_action_id = decision.action_id
        started = time.monotonic()
        predictions: List[OutcomePrediction] = []
        calls_made = 0
        stop_reason: Optional[str] = None
        hypothetical_snaps: Dict[str, str] = {}  # pred_id -> snapshot_id
        # M6: the structural targets proposed per root candidate, plus the
        # measured reliability of past predictions by class. Both are read
        # ONCE per decision (the banks are not re-read mid-decision).
        knowledge_targets: Dict[str, List[Any]] = {}
        reliability_table = self.prediction_reliability_table()
        def _real_budget_exceeded() -> Optional[str]:
            """Re-check the REAL ledger before the next model call: the
            budget view is refreshed (planning spend itself lands in it),
            so an overrun mid-decision stops further calls instead of
            reporting ok against an already-exceeded budget."""
            if not declared:
                return None
            view = self.budget.view(task_id, episode_id, budget=declared)
            if view["status"] == "exceeded":
                return ("declared budget exceeded by real consumption "
                        "(including this planning's own spend); no "
                        "further model calls")
            return None

        def _bounds_exhausted() -> Optional[str]:
            """The shared budget gate, re-evaluated BEFORE every model call.

            All three bounds are checked in one place so the root loop and
            the horizon-2 loop cannot drift apart in what they enforce:
            the model-call count, the wall clock, and the REAL ledger (whose
            view is refreshed each time so planning's own spend counts
            against the decision that is spending it)."""
            if calls_made >= limits.max_model_calls:
                return (f"model-call budget exhausted "
                        f"({limits.max_model_calls})")
            if time.monotonic() - started > limits.time_budget_s:
                return ("planning time budget exhausted "
                        f"({limits.time_budget_s}s)")
            return _real_budget_exceeded()

        for spec in specs:
            stop_reason = _bounds_exhausted()
            if stop_reason:
                break
            # Remaining time budget caps THIS call's own timeout: the
            # provider never waits longer than the decision's budget
            # allows. A sync provider that cannot honour it is declared
            # limited, not silently over-budget (see WorldModelProvider).
            remaining = limits.time_budget_s - (time.monotonic() - started)
            if remaining <= 0:
                stop_reason = ("planning time budget exhausted "
                               f"({limits.time_budget_s}s)")
                break
            target_set = self.knowledge_targets(root, spec)
            for target in target_set:
                knowledge_targets.setdefault(target.strategy_id, []).append(
                    target)
            prediction = self.predictions.predict_outcome(
                task, spec, root, timeout_s=remaining,
                knowledge_targets=target_set,
                reliability=reliability_table)
            prediction.model_info["knowledge_targets_proposed"] = [
                t.strategy_id for t in target_set]
            # The FULL targets are kept too: judging a later verdict needs
            # the claim's own interval / entry id / strategy, and re-deriving
            # them after the fact would read banks that may have changed
            # since the prediction was frozen.
            prediction.model_info["knowledge_targets_proposed_full"] = [
                t.to_dict() for t in target_set]
            self.predictions._save(prediction)
            calls_made += 1
            predictions.append(prediction)
        # (4) Horizon=2: condition the second step on the first step's
        # hypothetical successor — never a second independent root
        # prediction.
        continuation: Dict[str, List[OutcomePrediction]] = {}
        planning_error: Optional[str] = None
        try:
            if limits.horizon == 2 and stop_reason is None:
                followups = list(second_step or [])
                for first in predictions:
                    stop_reason = _bounds_exhausted()
                    if stop_reason:
                        break
                    if first.status != "valid":
                        continue
                    if not second_step_dependency_ok(first):
                        continuation[first.prediction_id] = []
                        continue
                    hypo = build_hypothetical_successor(root, first, task)
                    # Persist for auditability: the hypothetical flag keeps it
                    # out of every real-state query and out of episode
                    # progress inheritance; the rollout stays inspectable.
                    with self.store.transaction() as conn:
                        conn.execute(
                            "INSERT OR REPLACE INTO belief_snapshots "
                            "(snapshot_id, task_id, episode_id, created_at, "
                            "payload) VALUES (?,?,?,?,?)",
                            (hypo.snapshot_id, hypo.task_id, hypo.episode_id,
                             hypo.created_at, self.store.dumps(hypo.to_dict())))
                    hypothetical_snaps[first.prediction_id] = hypo.snapshot_id
                    # Second-step identity: the follow-up candidate gets
                    # the SAME task/episode validation as a root
                    # candidate — a spec explicitly naming ANOTHER task
                    # is rejected (skipped, never silently re-labelled),
                    # and the caller's object is never mutated (the spec
                    # is normalized on a value copy).
                    followup_source = (followups[0] if followups else
                                       first.action_spec)
                    if (followup_source.task_id
                            and followup_source.task_id != task_id):
                        continuation[first.prediction_id] = []
                        path_note = (
                            f"second-step identity conflict (rejected): "
                            f"task_id {followup_source.task_id!r} != "
                            f"{task_id!r}; no model call made")
                        stop_reason = (stop_reason + "; " + path_note
                                       if stop_reason else path_note)
                        continue
                    second_spec = ActionSpec.from_dict(
                        followup_source.to_dict())
                    second_spec.episode_id = root.episode_id
                    if not second_spec.task_id:
                        second_spec.task_id = task_id
                    remaining = limits.time_budget_s -                         (time.monotonic() - started)
                    if remaining <= 0:
                        stop_reason = ("planning time budget exhausted")
                        break
                    second = self.predictions.predict_outcome(
                        task, second_spec, hypo, timeout_s=remaining)
                    calls_made += 1
                    continuation[first.prediction_id] = [second]
        except Exception as exc:
            # A planning failure mid-way is NOT a silent success: persist
            # the completed part (predictions already made keep their known
            # cost), charge the real spend to the decision action, and end
            # it as failed. The error is recorded, never re-invoked.
            planning_error = f"{type(exc).__name__}: {exc}"
            stop_reason = f"planning aborted: {planning_error}"
        # Post-call deadline check: the LAST call's return may already be
        # over budget (a sync provider cannot be interrupted mid-call).
        # That is reported as truncation — never a quiet "ok" against an
        # exhausted budget. Costs already incurred stay recorded.
        if stop_reason is None and planning_error is None \
                and time.monotonic() - started > limits.time_budget_s:
            stop_reason = ("planning time budget exhausted after the "
                           f"final model call ({limits.time_budget_s}s); "
                           "the completed calls are kept and their cost "
                           "is recorded")
        # (5) Common yardstick, then per-path evaluation.
        all_predictions = predictions + [p for ps in continuation.values()
                                         for p in ps]
        cost_basis, norms = comparison_norms(all_predictions,
                                             limits.cost_weights)
        for first in predictions:
            steps = [first] + continuation.get(first.prediction_id, [])
            # Knowledge value of THIS path, assembled from the structural
            # needs of the root candidate that produced it. Skipped entirely
            # when the term is off, so a disabled term costs nothing and
            # changes nothing.
            knowledge: Optional[tuple] = None
            if limits.delta:
                from or_harness.world_model.knowledge import knowledge_value
                from or_harness.world_model.knowledge_track import (
                    reliability_lookup,
                )
                targets = knowledge_targets.get(
                    first.action_spec.strategy_id or "", [])
                knowledge = knowledge_value(
                    targets, steps,
                    reliability=reliability_lookup(reliability_table))
            if (limits.horizon == 2 and first.status == "valid"
                    and not continuation.get(first.prediction_id)):
                # Second step was required but could not be produced.
                path = evaluate_path([first], limits, norms, cost_basis,
                                     knowledge=knowledge)
                if stop_reason is None:
                    path.notes.append(
                        "conditional_unsupported: the first prediction did "
                        "not establish what a second execution step "
                        "depends on (e.g. a usable incumbent); evaluated "
                        "as a one-step path")
                else:
                    path.notes.append(f"second step truncated: {stop_reason}")
            else:
                path = evaluate_path(steps, limits, norms, cost_basis,
                                     knowledge=knowledge)
            if first.prediction_id in hypothetical_snaps:
                path.hypothetical_snapshot_id = \
                    hypothetical_snaps[first.prediction_id]
            plan.paths.append(path)
        # (6) Real planning spend -> decision action own cost.
        plan.model_calls_made = calls_made
        planning_total: Dict[str, float] = {}
        planning_measured: set = set()
        for prediction in all_predictions:
            if prediction.call_cost is None:
                continue
            for dim in prediction.call_cost.measured_dims():
                planning_total[dim] = planning_total.get(dim, 0.0) + \
                    getattr(prediction.call_cost, dim)
                planning_measured.add(dim)
        if planning_measured:
            self.actions.amend_action_cost_increment(
                decision.action_id,
                **{d: planning_total[d] for d in planning_measured})
            for prediction in all_predictions:
                if prediction.call_cost is not None:
                    prediction.model_info["charged_to_parent_action"] = \
                        decision.action_id
                    self.predictions._save(prediction)
            plan.planning_cost = {
                "cost": {d: round(planning_total[d], 4)
                         for d in sorted(planning_measured)},
                "measured": sorted(planning_measured),
                "note": "REAL spend of the planning model calls, charged "
                        "once to the decision action as own cost — sunk, "
                        "never part of any path's utility",
            }
        # (7) Budget re-check AFTER the planning spend is on the books,
        # BEFORE the suggestion: an exceeded real budget withholds the
        # suggestion (the plan's own cost is real and stays recorded).
        if declared:
            final_view = self.budget.view(task_id, episode_id,
                                          budget=declared)
            plan.budget_confirmation = final_view["status"]
            if final_view["status"] == "exceeded" and not planning_error:
                plan.status = "fallback"
                plan.truncation_reason = (
                    "declared budget exceeded by real consumption "
                    "(including this planning's own spend); the "
                    "suggestion is withheld — report the current best "
                    "solution instead.")
                self.actions.end_action(
                    decision.action_id, status="completed",
                    outcome={"kind": "plan_next_evaluation",
                             "plan_id": plan.plan_id,
                             "n_paths": len(plan.paths),
                             "suggested": None,
                             "suggestion_withheld": True,
                             "status": plan.status,
                             "truncation_reason": plan.truncation_reason})
                return plan.to_dict()
        # (8) Suggestion (first step only).
        comparable = [p for p in plan.paths if p.utility is not None]
        suggested_path = (max(comparable, key=lambda p: p.utility)
                          if comparable else None)
        if suggested_path is not None and self.plan_mode == "advise":
            plan.suggested = suggested_path.steps[0].action_spec
            parts = [
                f"U={suggested_path.utility} = "
                f"{limits.alpha}*Q({suggested_path.q_terminal}) - "
                f"{limits.beta}*C({suggested_path.c_path}) - "
                f"{limits.gamma}*R({suggested_path.r_terminal})",
                f"evaluated {len(plan.paths)} path(s) from root snapshot "
                f"{plan.root_snapshot_id}",
            ]
            if suggested_path.incomparable:
                parts.append("incomparable: "
                             + "; ".join(suggested_path.incomparable.values()))
            plan.suggestion_basis = "; ".join(parts)
        elif suggested_path is not None:
            plan.suggestion_basis = ("shadow mode: paths evaluated and "
                                     "recorded; suggestion withheld")
        plan.status = "truncated" if stop_reason else "ok"
        if stop_reason:
            plan.truncation_reason = stop_reason
        # A mid-planning exception is a FAILED decision, not a truncated
        # one: the evaluation itself could not complete.
        if planning_error:
            plan.status = "failed"
            plan.truncation_reason = stop_reason
        # Honesty: when EVERY candidate's prediction was unusable (all
        # paths incomparable), "ok" would falsely suggest the evaluation
        # succeeded. Report the real reason — no valid prediction — so
        # the caller never mistakes "model output unusable" for a quiet
        # "no recommendation".
        if not comparable and plan.status == "ok":
            plan.status = "no_valid_predictions"
            statuses = sorted({p.status for p in all_predictions})
            plan.truncation_reason = (
                "no candidate carried a usable prediction "
                f"(statuses: {statuses}); no suggestion is possible. "
                "This is the model output's problem, not a framework "
                "failure — the calls happened and their cost is recorded.")
        # (8) End the decision action. The outcome records the FULL
        # evaluation (frozen input refs, candidates, paths with utilities,
        # prediction refs, decomposition, comparison basis/norms, budget,
        # stopping reasons, and suggestion) — NOT a selection:
        # X.selected_plan is written only by choose_next. A planning
        # error ends the action as failed with the error recorded.
        self.actions.end_action(
            decision.action_id,
            status="failed" if planning_error else "completed",
            outcome={"kind": "plan_next_evaluation",
                     "plan_id": plan.plan_id,
                     "root_snapshot_id": plan.root_snapshot_id,
                     "n_paths": len(plan.paths),
                     "paths": [p.to_dict() for p in plan.paths],
                     "suggested": (plan.suggested.to_dict()
                                   if plan.suggested else None),
                     "suggestion_basis": plan.suggestion_basis,
                     "suggestion_withheld": self.plan_mode == "shadow",
                     "status": plan.status,
                     "error": planning_error,
                     "truncation_reason": plan.truncation_reason,
                     "model_calls_made": int(plan.model_calls_made),
                     "planning_cost": copy.deepcopy(plan.planning_cost),
                     "budget_confirmation": plan.budget_confirmation,
                     "cost_basis": list(cost_basis),
                     "cost_norms": dict(norms),
                     "hypothetical_snapshots": dict(hypothetical_snaps)})
        return plan.to_dict()

    def choose_next(self, decision_action_id: str, *,
                    chosen: Optional[ActionSpec] = None,
                    rejected: bool = False,
                    deviation_note: Optional[str] = None) -> Dict[str, Any]:
        """Record the agent's EXPLICIT choice after a plan.

        ``chosen`` is the ActionSpec the agent decided to take (the
        suggested one or another — a deviation); ``rejected=True`` records
        that no suggestion was taken. Only this call writes
        X.selected_plan (via a completed select_strategy report linked to
        the decision action): a generated suggestion alone never counts as
        a selection, and a selection record produces no execution quality.
        """
        from or_harness.world_model.actions import PROGRESS_UPDATES
        decision = self.actions.get(decision_action_id)
        if decision is None:
            raise StorageError(
                f"unknown decision_action_id {decision_action_id!r}")
        if decision.action_type != "select_strategy":
            raise StorageError(
                f"action {decision_action_id!r} is a "
                f"{decision.action_type}, not a select_strategy decision")
        outcome: Dict[str, Any] = {
            "kind": "plan_next_choice",
            "decision_action_id": decision_action_id,
            "decision_plan_id": (decision.outcome or {}).get("plan_id"),
        }
        if rejected:
            outcome["rejected"] = True
            if deviation_note:
                outcome["note"] = deviation_note
        elif chosen is not None:
            outcome["selected"] = chosen.to_dict()
            suggested = (decision.outcome or {}).get("suggested")
            if suggested and suggested != chosen.to_dict():
                outcome["deviation"] = {
                    "suggested": suggested,
                    "chosen": chosen.to_dict(),
                    "note": deviation_note,
                }
            elif deviation_note:
                outcome["note"] = deviation_note
        else:
            raise ValueError("choose_next requires chosen=... or "
                             "rejected=True")
        record = self.actions.report_action(
            "select_strategy", decision.task_id, decision.episode_id,
            params={"decision_action_id": decision_action_id},
            outcome=outcome, status="completed")
        # Link the choice to the decision it answers (parent-child).
        record.parent_action_id = decision_action_id
        self.actions._update(record)
        # Explicit choice -> X.selected_plan update (the ONLY writer of
        # that field from the planning flow). A REJECTION is not a
        # selection: it must NOT overwrite an earlier selected_plan —
        # only a new explicit choice (or an explicit cancellation) does.
        # The task payload is recovered from the decision's frozen root
        # snapshot (its P carries the problem content) — the caller does
        # not have to re-supply it.
        if rejected:
            # The rejection is recorded as its own labelled progress
            # event (queryable), never as a selected_plan overwrite.
            progress = {"last_rejected_suggestion": {
                "value": outcome,
                "provenance": "agent_reported",
                "epistemic": "fact",
                "evidence_ref": record.action_id,
            }}
        else:
            progress = self._progress_from_outcome(record.action_id,
                                                   outcome)
        root_snap = self.get_snapshot(decision.pre_snapshot_id) \
            if decision.pre_snapshot_id else None
        task = dict((root_snap.problem_state.get("task_payload") or {})
                    if root_snap is not None else {})
        task.setdefault("task_id", decision.task_id)
        if "family" not in task and root_snap is not None:
            family = ((root_snap.problem_state.get("profile") or {})
                      .get("family"))
            if family:
                task["family"] = family
        post = self.snapshot(task, decision.episode_id,
                             task_progress=progress)
        self.actions._bind_post_snapshot(record.action_id,
                                         post.snapshot_id)
        return {"action_id": record.action_id,
                "status": record.status,
                "selected": outcome.get("selected"),
                "rejected": bool(outcome.get("rejected")),
                "deviation": outcome.get("deviation"),
                "post_snapshot_id": post.snapshot_id}

    def plan_decision(self, decision_action_id: str) -> Optional[Dict[str, Any]]:
        """Read back a planning decision record (the select_strategy action
        plus its outcome payload)."""
        record = self.actions.get(decision_action_id)
        return record.to_dict() if record is not None else None

    # -- world-model M4: offline maintenance assessment ----------------------

    def induction_candidates(self) -> List[Dict[str, Any]]:
        """Form traceable induction candidate bundles from current evidence.

        Scans the Experience Bank and Strategic Bank for cells with
        sufficient evidence (>=2 executions from >=2 tasks) where a new
        claim or a substantive revision is indicated. Returns frozen
        candidate bundles — no dynamic re-querying."""
        from or_harness.world_model.maintenance import (
            build_induction_candidates,
        )
        bundles = build_induction_candidates(self)
        return [b.to_dict() for b in bundles]

    def assess_induction(self, bundle: Dict[str, Any], *,
                         budget: Optional[Dict[str, float]] = None,
                         workload_forecast: Optional[Dict[str, Any]] = None,
                         time_budget_s: float = 30.0,
                         ) -> Dict[str, Any]:
        """Explicitly evaluate the consequences and value of an induction
        action using the world model (M4).

        Input = an InductionCandidateBundle (frozen evidence), an optional
        maintenance budget, and an optional workload forecast.

        Asks the world model to predict the induction action's consequences:
        candidate formation probability, expected reuse benefit,
        generalization risk, and execution cost under the induced claim.
        The assessment value = reuse_benefit - induction_cost - risk.

        Returns an auditable :class:`InductionAssessment` dict. The
        assessment cost (model calls) is charged ONCE to a maintenance
        decision action. Does NOT perform induction and does NOT modify
        the Strategic Bank."""
        from or_harness.world_model.maintenance import (
            InductionAssessment,
            InductionCandidateBundle,
        )
        b = InductionCandidateBundle.from_dict(bundle)
        assessment = InductionAssessment(
            assessment_id=InductionAssessment.new_id(),
            bundle_id=b.bundle_id,
            decision_action_id=None,
            recommendation="defer",
            target_strategy_id=b.strategy_id,
            target_family=b.family,
            workload_forecast=copy.deepcopy(workload_forecast),
            # The evidence scope travels with the assessment so a later
            # ``accept_induction`` can restrict the induction to exactly the
            # bundle's own executions (no silent widening) and so the
            # delayed binding can verify the scope it judges against.
            execution_ids=list(b.execution_ids),
        )
        if self.induction_assessment_mode == "disabled":
            assessment.status = "disabled"
            assessment.recommendation_basis = (
                "induction assessment is disabled "
                "(ORHarness(induction_assessment='disabled'))")
            return assessment.to_dict()

        # (1) Maintenance decision action.
        episode_id = f"maint_eval_{int(time.time())}"
        decision = self.actions.begin_action(
            "induce", MAINTENANCE_TASK_ID, episode_id,
            params={"kind": "induction_assessment",
                    "assessment_id": assessment.assessment_id,
                    "bundle_id": b.bundle_id,
                    "target_strategy_id": b.strategy_id,
                    "target_family": b.family,
                    "bundle_kind": b.kind,
                    "workload_forecast": workload_forecast})
        assessment.decision_action_id = decision.action_id

        # (2) Freeze maintenance belief state.
        # Task representation for maintenance snapshot.
        maint_task = {
            "task_id": MAINTENANCE_TASK_ID,
            "family": b.family,
            "spec": {"target_strategy_id": b.strategy_id,
                     "n_supporting": b.n_supporting},
            "annotations": {"bundle": b.to_dict()},
        }
        root = self.snapshot(maint_task, episode_id)

        # (3) ActionSpec for the induction candidate.
        spec = ActionSpec(
            action_type="induce",
            task_id=MAINTENANCE_TASK_ID,
            episode_id=episode_id,
            strategy_id=b.strategy_id,
            params={"bundle": b.to_dict(),
                    "kind": b.kind,
                    "target_entry_id": b.target_entry_id,
                    "workload_forecast": workload_forecast},
            measurement_scope="task",
        )

        # (4) Provider call with timeout budget.
        started = time.monotonic()
        prediction = self.predictions.predict_outcome(
            maint_task, spec, root, timeout_s=time_budget_s)
        assessment.prediction_id = prediction.prediction_id

        # (5) Real call cost -> decision action own cost.
        dims = self._charge_call_cost(prediction, decision.action_id)
        if dims:
            assessment.assessment_cost = {
                "cost": {d: round(dims[d], 4) for d in sorted(dims)},
                "measured": sorted(dims),
                "note": "REAL spend of the maintenance assessment call",
            }

        # (6) Form recommendation from prediction + evidence.
        if prediction.status != "valid":
            assessment.status = prediction.status
            assessment.recommendation = "insufficient_evidence"
            assessment.recommendation_basis = (
                f"prediction returned status {prediction.status!r}: "
                f"{prediction.error or 'no usable prediction'}; "
                "deferring induction until valid assessment is possible")
            if prediction.status == "not_configured":
                assessment.status = "not_configured"
        else:
            p = prediction.predicted
            # Direct consequence fields.
            form_prob = float(p.get("candidate_formation_prob", 1.0)
                              if p.get("candidate_formation_prob") is not None
                              else 1.0)
            assessment.candidate_formation_prob = form_prob
            assessment.predicted_quality_claim = p.get("quality")
            if isinstance(p.get("cost"), dict):
                assessment.predicted_cost_claim = dict(p["cost"])
            assessment.confidence = prediction.confidence
            assessment.unsupported_fields = dict(
                prediction.unsupported_fields)

            # Value decomposition.
            # reuse_benefit: from model or workload forecast.
            reuse_benefit = p.get("expected_reuse_benefit")
            gen_risk = p.get("generalization_risk", p.get("failure_prob", 0.0))
            if gen_risk is not None:
                gen_risk = float(gen_risk)
            assessment.predicted_generalization_risk = gen_risk

            # Workload multiplier: if forecast supplied, scale benefit.
            workload_mult = 1.0
            if workload_forecast and isinstance(
                    workload_forecast.get("expected_matching_tasks"),
                    (int, float)):
                workload_mult = max(
                    0.0, float(workload_forecast["expected_matching_tasks"]))

            if reuse_benefit is not None:
                reuse_benefit = float(reuse_benefit) * workload_mult
            assessment.expected_reuse_benefit = reuse_benefit

            # Net value = reuse_benefit - (alpha*0 + beta*cost + gamma*risk)
            # Normalized scalar using harness selector weights.
            if reuse_benefit is not None and gen_risk is not None:
                net = (self.selector.alpha * reuse_benefit
                       - self.selector.gamma * gen_risk)
                assessment.net_value = round(net, 4)

            # Recommendation logic:
            # - Insufficient evidence: bundle has fewer than 2 tasks
            #   (should not happen via build_induction_candidates, but checked)
            # - Defer: net_value <= 0 or candidate_formation_prob < 0.5
            # - Induce_new / Revise: net_value > 0 and formation_prob >= 0.5
            if len(b.tasks) < 2 and b.kind == "new_claim":
                assessment.recommendation = "insufficient_evidence"
                assessment.recommendation_basis = (
                    f"bundle has evidence from only {len(b.tasks)} task "
                    f"({b.tasks}); claim requires >=2 distinct tasks")
            elif form_prob < 0.5:
                assessment.recommendation = "defer"
                assessment.recommendation_basis = (
                    f"low predicted candidate formation probability "
                    f"({form_prob:.2f}); deferring until stronger evidence")
            elif assessment.net_value is not None and assessment.net_value <= 0:
                assessment.recommendation = "defer"
                assessment.recommendation_basis = (
                    f"expected net value is non-positive ({assessment.net_value:.4f} "
                    f"= benefit {reuse_benefit} - risk {gen_risk}); "
                    "cost/risk outweighs expected reuse benefit")
            elif b.kind == "revision":
                assessment.recommendation = "revise"
                basis_parts = [
                    f"revision of entry {b.target_entry_id} recommended",
                    f"evidence: {b.n_supporting} executions across {len(b.tasks)} tasks",
                ]
                if assessment.net_value is not None:
                    basis_parts.append(f"net_value={assessment.net_value:.4f}")
                assessment.recommendation_basis = "; ".join(basis_parts)
            else:
                assessment.recommendation = "induce_new"
                basis_parts = [
                    f"new claim for ({b.family}, {b.strategy_id}) recommended",
                    f"evidence: {b.n_supporting} executions across {len(b.tasks)} tasks",
                ]
                if assessment.net_value is not None:
                    basis_parts.append(f"net_value={assessment.net_value:.4f}")
                assessment.recommendation_basis = "; ".join(basis_parts)

        # (7) Shadow mode withholds advice.
        if self.induction_assessment_mode == "shadow":
            assessment.recommendation_basis = (
                "shadow mode: induction consequence evaluated and "
                f"recorded (predicted recommendation: {assessment.recommendation}); "
                "advice withheld — no action recommended to agent")

        # (8) End maintenance decision action.
        self.actions.end_action(
            decision.action_id,
            status="completed" if assessment.status == "ok" else "failed",
            outcome={"kind": "induction_assessment",
                     "assessment_id": assessment.assessment_id,
                     "bundle_id": b.bundle_id,
                     "recommendation": assessment.recommendation,
                     "recommendation_basis": assessment.recommendation_basis,
                     "status": assessment.status,
                     "prediction_id": assessment.prediction_id,
                     "net_value": assessment.net_value,
                     "target_strategy_id": b.strategy_id,
                     "target_family": b.family,
                     "bundle_kind": b.kind})
        return assessment.to_dict()

    def accept_induction(self, assessment_dict: Dict[str, Any], *,
                         verify: Optional[Dict[str, Any]] = None,
                         notes: Optional[List[str]] = None,
                         force: bool = False) -> Dict[str, Any]:
        """Accept an induction assessment recommendation and execute the
        induction strictly on the chosen bundle's scope.

        The bundle's exact execution IDs are passed to ``induce`` —
        preventing silent scope widening. Records an explicit adoption
        action linked to the assessment decision."""
        bundle_id = assessment_dict.get("bundle_id")
        strategy_id = assessment_dict.get("target_strategy_id")
        recommendation = assessment_dict.get("recommendation")
        if recommendation not in ("induce_new", "revise"):
            raise ValueError(
                f"cannot accept assessment with recommendation {recommendation!r} "
                "(only 'induce_new' and 'revise' can be accepted)")

        # Record explicit adoption action.
        #
        # The action is reported BEFORE the induction (the adoption decision
        # is what triggers it), then AMENDED with the induction's own result.
        # Without the amendment, ``bind_induction_outcome`` would have to
        # guess which induction followed this decision; carrying the result
        # and its evidence scope on the record makes the binding verifiable
        # rather than inferred.
        decision_id = assessment_dict.get("decision_action_id")
        adoption = self.actions.report_action(
            "induce", MAINTENANCE_TASK_ID,
            f"maint_adopt_{int(time.time())}",
            params={"assessment_id": assessment_dict.get("assessment_id"),
                    "bundle_id": bundle_id,
                    "action": "accepted",
                    "recommendation": recommendation,
                    "execution_ids": list(assessment_dict.get(
                        "execution_ids") or [])},
            outcome={"accepted": True,
                     "assessment_id": assessment_dict.get("assessment_id")},
            status="completed")
        if decision_id:
            adoption.parent_action_id = decision_id
            self.actions._update(adoption)

        # Execute induction with scope restricted to the bundle.
        # Find the profile from the bundle's evidence.
        rec_ids = assessment_dict.get("execution_ids")
        induce_res = self.induce(
            strategy_id=strategy_id,
            verify=verify,
            notes=notes,
            force=force,
            execution_ids=rec_ids)
        # Amend the adoption record with the induction that actually ran:
        # its transition, the evidence scope it consolidated, and the
        # assessment it answers. This is the reference the delayed binding
        # reads.
        adoption.outcome = {
            "accepted": True,
            "assessment_id": assessment_dict.get("assessment_id"),
            "induction_result": {
                "assessment_id": assessment_dict.get("assessment_id"),
                "business_result": induce_res.get("business_result"),
                "knowledge_delta": induce_res.get("knowledge_delta"),
                "knowledge_after": induce_res.get("knowledge_after"),
                "execution_ids": list(rec_ids or []),
            },
        }
        self.actions._update(adoption)
        return {
            "adoption_action_id": adoption.action_id,
            "assessment_id": assessment_dict.get("assessment_id"),
            "accepted": True,
            "induction_result": induce_res,
        }

    def bind_induction_outcome(self, assessment_id: str) -> Dict[str, Any]:
        """Compare an induction assessment's predictions against the REAL
        induction outcome and record the verdict (M6).

        This is the delayed feedback of the MAINTENANCE decision: it
        answers "was it right to expect this induction to form a claim /
        yield a usable entry?" — the slow counterpart of
        :meth:`compare_prediction`, which judges the fast X/B predictions.

        Deliberately NOT the same as :meth:`evaluate_knowledge_consolidation`:
        that one judges what an ORDINARY solving action predicted about the
        knowledge it fed; this one judges the induction DECISION's own
        expectation. Both are needed, and neither substitutes for the other.

        The verdict is computed from the induction that actually followed
        the assessment (found through the actions that reference it), never
        from the assessment's own opinion of itself. Idempotent: a second
        call returns the stored verdict rather than recomputing it.
        """
        assessment_action = None
        adoption_action = None
        for action in self.actions.query():
            # The assessment ID lives in the decision action's OUTCOME (it is
            # a result of running the assessment), while an adoption action
            # repeats it in its params. Reading both is what makes the
            # documented assess -> accept -> bind chain work.
            params = action.params or {}
            outcome = action.outcome or {}
            if (params.get("assessment_id") != assessment_id
                    and outcome.get("assessment_id") != assessment_id):
                continue
            if params.get("action") in ("accepted", "rejected"):
                adoption_action = action
            elif outcome.get("kind") == "induction_assessment":
                assessment_action = action
        if assessment_action is None:
            raise StorageError(
                f"unknown assessment_id {assessment_id!r}: no assessment "
                "action references it")
        outcome = assessment_action.outcome or {}
        # The adoption action must BELONG to this assessment, not merely
        # mention the id: an adoption carrying another assessment's result
        # would bind this one to the wrong induction.
        if adoption_action is not None:
            linked = ((adoption_action.outcome or {})
                      .get("induction_result") or {})
            if linked and linked.get("assessment_id") not in (
                    None, assessment_id):
                adoption_action = None
        prediction_id = outcome.get("prediction_id")
        if not prediction_id:
            return {"assessment_id": assessment_id, "compared": False,
                    "reason": "the assessment recorded no prediction id"}
        prediction = self.predictions.get(prediction_id)
        if prediction is None:
            return {"assessment_id": assessment_id, "compared": False,
                    "reason": "the referenced prediction no longer exists"}
        if prediction.feedback is not None \
                and prediction.feedback.get("induction_binding"):
            return {"assessment_id": assessment_id, "compared": True,
                    "verdict": prediction.feedback["induction_binding"],
                    "note": "already bound: the stored verdict stands"}
        if adoption_action is None:
            return {"assessment_id": assessment_id, "compared": False,
                    "reason": "no accept/reject decision was recorded for "
                              "this assessment; nothing to bind against"}
        adopted = bool((adoption_action.params or {}).get("action")
                       == "accepted")
        if not adopted:
            # A rejection is not a wrong prediction: the agent declined to
            # act, so the assessment's expectations were never given a
            # chance to come true. Recorded, never scored as a miss.
            verdict = {
                "compared": False,
                "status": "inconclusive",
                "reason": "the recommendation was rejected or deferred, so "
                          "the predicted consequence was never given a "
                          "chance to occur",
                "adoption_action_id": adoption_action.action_id,
            }
            prediction.feedback = dict(prediction.feedback or {})
            prediction.feedback["induction_binding"] = verdict
            self.predictions._save(prediction)
            return {"assessment_id": assessment_id, "compared": False,
                    "verdict": verdict}
        induction_result = ((adoption_action.outcome or {}).get(
            "induction_result") or {})
        delta = self._induction_transition(induction_result)
        created = delta["created"]
        predicted = prediction.predicted or {}
        formed = bool(created)
        formation_pred = predicted.get("candidate_formation_prob")
        compared: Dict[str, Any] = {}
        not_compared: Dict[str, str] = {}
        if formation_pred is None:
            not_compared["candidate_formation_prob"] = "not predicted"
        else:
            compared["candidate_formation_prob"] = {
                "predicted": formation_pred,
                "actual": 1.0 if formed else 0.0,
                "match": (float(formation_pred) >= 0.5) == formed,
            }
        status = "fulfilled" if formed else "missed"
        verdict = {
            "compared": True,
            "status": status,
            "entries_created": list(created),
            "compared_fields": compared,
            "not_compared": not_compared,
            "adoption_action_id": adoption_action.action_id,
            "note": ("an entry FORMING is not the same as publishable "
                     "strategic knowledge: admission verification still "
                     "decides that, and this verdict does not speak to it"),
        }
        prediction.feedback = dict(prediction.feedback or {})
        prediction.feedback["induction_binding"] = verdict
        self.predictions._save(prediction)
        return {"assessment_id": assessment_id, "compared": True,
                "verdict": verdict}

    def reject_induction(self, assessment_dict: Dict[str, Any], *,
                         reason: Optional[str] = None) -> Dict[str, Any]:
        """Explicitly reject or defer an induction recommendation.

        Records the rejection as a maintenance action — no induction
        is executed, Strategic Bank is untouched."""
        decision_id = assessment_dict.get("decision_action_id")
        record = self.actions.report_action(
            "induce", MAINTENANCE_TASK_ID,
            f"maint_reject_{int(time.time())}",
            params={"assessment_id": assessment_dict.get("assessment_id"),
                    "bundle_id": assessment_dict.get("bundle_id"),
                    "action": "rejected",
                    "reason": reason},
            outcome={"accepted": False,
                     "rejected": True,
                     "reason": reason,
                     "assessment_id": assessment_dict.get("assessment_id")},
            status="completed")
        if decision_id:
            record.parent_action_id = decision_id
            self.actions._update(record)
        return {
            "rejection_action_id": record.action_id,
            "assessment_id": assessment_dict.get("assessment_id"),
            "accepted": False,
            "rejected": True,
            "reason": reason,
        }

    # -- world-model M3: bounded planning ------------------------------------
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
               code: Optional[str] = None,
               include_unverified: bool = False,
               vector_top_k: Optional[int] = None) -> Dict[str, Any]:
        """Recall accumulated experience for this task.

        TWO INDEPENDENT CHANNELS, never blended into one number:

        - ``recommendations`` / ``available_solver_families`` /
          ``solver_advisories``: the structural channel, unchanged —
          applicable candidates with their evidence-based scores.
        - ``vector_recall``: the text-similarity channel (embedding), which
          surfaces memories whose TEXT is close regardless of structural
          cell. Its ``similarity`` is a discovery signal only; it is never a
          quality, cost, or risk estimate, and a cross-cell hit never enters
          the target cell's statistics.

        When the text channel cannot run (no backend, no task text, missing
        or model-incompatible index, backend error) the structural result is
        returned BY ITSELF with ``degraded`` explaining why, rather than
        silently looking like a text search that found nothing.

        READ-ONLY: the query text is embedded in memory and never written;
        no index item is created and no migration is triggered.
        """
        profile = self.profile(task, code)
        recs = self.selector.recall(profile, top=top, exclude=exclude,
                                    memory_mode=memory_mode,
                                    include_unverified=include_unverified)
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
        # Text-similarity channel (discovery only). A failure here NEVER
        # removes or modifies a structural result — it only explains itself.
        try:
            result["vector_recall"] = recall_vectors(
                self, task_text(task),
                top_k=vector_top_k or max(1, top),
                include_unverified=include_unverified,
                task_profile=profile)
        except VectorRecallUnavailable as exc:
            result["degraded"] = {"path": "profile_only",
                                  "reason": str(exc)}
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

    # -- task texts (the retrieval document's source) ----------------------------

    def capture_task_text(self, task: Dict[str, Any]) -> Optional[str]:
        """Persist the task's text as ONE VERSION and return its digest.

        Called on the write paths (``execute`` / ``record``) only — never on
        a read path. Idempotent per version: the same task solved twice
        re-uses the same row. A task carrying no textual field stores
        nothing (an empty retrieval document would be matched against every
        memory as a zero vector) and returns None, so the fact is honestly
        reported as unindexed instead of being indexed as noise.
        """
        text = task_text(task)
        if not text.strip():
            return None
        digest = task_text_digest(task)
        self.store.put_task_text(str(task.get("task_id", "")), text, digest)
        return digest

    def _recover_task_text(self, record: ExecutionRecord) -> Optional[str]:
        """Recover the text of an execution recorded without ``execute``.

        Priority: (1) the digest the caller already supplied, looked up in
        the task-text store; (2) the most recent REAL belief snapshot of the
        task (``hypothetical=False``), whose frozen ``task_payload`` is the
        task as it was — rebuilt through the SAME reader used at capture
        time, and re-keyed by the FULL task digest (a snapshot's payload is
        a subset of the task, so only the whole-task digest the snapshot
        stores is authoritative); (3) nothing.

        Nothing is ever INVENTED: no snapshot and no digest means the record
        stays unindexed, visible only through profile retrieval,
        ``inspect`` and ``task_texts_for``.
        """
        if record.task_text_digest:
            stored = self.store.get_task_text(record.task_id,
                                             record.task_text_digest)
            if stored is not None:
                return record.task_text_digest
            # The caller named a version this memory does not hold: fall
            # through to snapshot recovery rather than failing the record.
        latest: Optional[BeliefSnapshot] = None
        for snap in self.snapshots(task_id=record.task_id):
            if snap.hypothetical:
                continue
            problem = snap.problem_state or {}
            if not problem.get("task_digest"):
                continue
            if latest is None or (snap.created_at, snap.snapshot_id) > \
                    (latest.created_at, latest.snapshot_id):
                latest = snap
        if latest is None:
            return None
        problem = latest.problem_state or {}
        text = task_text_from_payload(problem.get("task_payload") or {})
        digest = str(problem.get("task_digest"))
        if not text.strip():
            # No textual payload was frozen (the task carried a task_ref or
            # had no text): the digest alone cannot reconstruct a document.
            return None
        self.store.put_task_text(record.task_id, text, digest)
        return digest

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
        # The text channel's source document: persisted here (a write path)
        # so a later recall can embed this exact version. The digest goes on
        # the record, making "the text this execution was produced under"
        # checkable; a task with no text stores nothing.
        task_text_ver = self.capture_task_text(task)
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
        if record.task_text_digest is None:
            record.task_text_digest = task_text_ver
        # Safety net: stage every execution — successes AND failures — so a
        # failed attempt is never silently lost when the harness immediately
        # retries. Staging is not recording; recording stays the harness's
        # explicit decision (`orx record`).
        self.bank.stage_pending(record)
        # End the macro action: status follows the EXECUTION outcome (not
        # the record decision); cost is a REFERENCE to the execution's own
        # cost (counted there — never summed again). Save order mirrors
        # end_action: the action (with its linked_execution_id) is persisted
        # FIRST, then the post snapshot is generated — so the snapshot's
        # budget view already sees this execution (the episode filter
        # resolves via the linkage) and its cost.
        exec_status = record.quality.get("status")
        action_status = ("completed" if exec_status in ("optimal", "feasible")
                         else "failed" if exec_status in ("error",)
                         else "timeout" if exec_status == "timeout"
                         else "completed")
        self.actions.end_action(
            action.action_id, status=action_status,
            outcome={"execution_status": exec_status,
                     "feasible": record.quality.get("feasible"),
                     "objective": record.quality.get("objective")},
            cost=record.cost,
            linked_execution_id=record.execution_id, rollup="reference")
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
        self.actions._bind_post_snapshot(action.action_id, post.snapshot_id)
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
        attempt) before it is forgotten.

        Task text: ``record`` can be reached WITHOUT ``execute`` (the agent
        may append a record directly), so the text link is resolved here too
        — from the digest the caller supplied, else from the task's most
        recent real belief snapshot (see :meth:`_recover_task_text`). A
        legacy execution whose text cannot be honestly recovered keeps
        ``task_text_digest=None`` and is simply not vector-indexed; it is
        never fabricated.

        Index synchronization is BEST EFFORT: the fact is appended first, and
        an embedding failure is reported as ``index_sync`` (deferred) rather
        than rolling anything back."""
        # Persist failure classification once — a first-class fact, not a
        # re-derived view (environment vs model errors feed solver advisories
        # and future failure-pattern induction).
        from or_harness.strategy.triggers import classify_failure
        for failure in record.failures:
            if failure.error_class is None:
                failure.error_class = classify_failure(record)
        if record.task_text_digest is None:
            record.task_text_digest = self._recover_task_text(record)
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
        # M6: a NORMAL solving action's knowledge prediction is evaluated
        # here, against the evidence this very execution produced. This is
        # the fast channel of the H loop — without it the knowledge
        # predictions of ordinary executions would never be judged, which
        # was the largest gap in the M6 plan's first version.
        knowledge_feedback = self.evaluate_knowledge_execution(record)
        if knowledge_feedback:
            result["knowledge_feedback"] = knowledge_feedback
        # Index sync happens AFTER the fact is durable: the memory exists
        # whether or not the embedding call works, and the outcome is always
        # reported (never a silent divergence between bank and index).
        result["index_sync"] = self.index_sync.sync_execution(record)
        if unrecorded:
            result["unrecorded_staged_executions"] = unrecorded
        return result

    def induce(self, *, strategy_id: Optional[str] = None, all_: bool = False,
               rebuild: bool = False,
               dry_run: bool = False, force: bool = False,
               notes: Optional[List[str]] = None,
               verify: Optional[Dict[str, Any]] = None,
               execution_ids: Optional[Sequence[str]] = None) -> Dict[str, Any]:
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
                        notes=notes, verify=verify,
                        execution_ids=execution_ids))
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
        # Knowledge changed, so the knowledge documents changed: refresh
        # their index items (best effort, never blocking). A dry run writes
        # nothing at all — a rehearsal must not touch the index either.
        if not dry_run:
            result["index_sync"] = self.index_sync.sync_entries()
            # M6 slow channel: the knowledge predictions of the ORDINARY
            # actions whose evidence this induction consolidated are judged
            # now, against what the induction actually changed. Running it
            # after the knowledge delta exists is what makes the verdict
            # speak about a real transition rather than a rehearsal.
            try:
                scope_ids = execution_ids or result.get("execution_ids") \
                    or []
                consolidation = self.evaluate_knowledge_consolidation(
                    result, scope_ids, strategy_id=strategy_id)
                if consolidation:
                    result["knowledge_feedback"] = consolidation
            except Exception as exc:  # never fail a real induction for this
                result["knowledge_feedback_error"] = (
                    f"{type(exc).__name__}: {exc}")
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
        before_by_id = {k["entry_id"]: k for k in
                        maintenance["knowledge_before"]["entries"]}
        after_by_id = {k["entry_id"]: k for k in knowledge_after["entries"]}
        before_ids = set(before_by_id)
        after_ids = set(after_by_id)
        # Per-entry diff: what actually changed in the entries that stayed.
        # A bare id list cannot reconstruct the transition — the modified
        # values (expected quality/cost, predicates, verification state)
        # are what future analysis needs to replay this knowledge change.
        entry_changes = []
        for entry_id in sorted(before_ids & after_ids):
            before, after = before_by_id[entry_id], after_by_id[entry_id]
            changed_fields = {}
            for field in ("predicates", "expected_quality_hat",
                          "quality_interval", "expected_cost_hat",
                          "cost_interval", "failure_prob", "support_n",
                          "status", "verification_state"):
                if before.get(field) != after.get(field):
                    changed_fields[field] = {
                        "before": before.get(field),
                        "after": after.get(field),
                    }
            if changed_fields:
                entry_changes.append({"entry_id": entry_id,
                                      "changed": changed_fields})
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
                "entry_changes": entry_changes,
            },
            # The full AFTER state (value-copied entry refs): the frozen
            # record can reconstruct what the knowledge looked like after
            # this induction, not just which ids moved.
            "knowledge_after": knowledge_after,
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
                "business_result": business,
                # The transition itself travels back with the summary so a
                # caller (and the delayed knowledge evaluation) can judge
                # what the induction changed without re-reading the action
                # record. The full outcome stays persisted on the action —
                # this is a convenience view of the same data, not a second
                # source of truth.
                "knowledge_delta": outcome["knowledge_delta"],
                "knowledge_after": outcome["knowledge_after"]}

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
                                         in ("model", "select_strategy",
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
        if bank == "predictions":
            preds = self.predictions_query(task_id=task_id)
            return {"bank": "predictions", "count": len(preds),
                    "predictions": [p.to_dict() for p in preds]}
        if bank == "texts":
            # The retrieval SOURCE documents (not a knowledge bank): the
            # documented look-up entry point for records that carry no
            # vector. Without --task, every retained version is listed.
            if task_id is None:
                rows = self.store.conn.execute(
                    "SELECT task_id, text_digest, text, created_at FROM "
                    "task_texts ORDER BY task_id ASC, created_at DESC"
                ).fetchall()
                entries = [{"task_id": str(r["task_id"]),
                            "text_digest": str(r["text_digest"]),
                            "text": str(r["text"]),
                            "created_at": float(r["created_at"])}
                           for r in rows]
            else:
                entries = [{"task_id": task_id, **row}
                           for row in self.store.task_texts_for(task_id)]
            return {"bank": "texts", "count": len(entries),
                    "task_texts": entries}
        raise ValueError("bank must be experience|strategic|archive|"
                         "actions|snapshots|predictions|texts")

    def collect_garbage(self, mode: str = "compact",
                        dry_run: bool = False) -> Dict[str, Any]:
        return self.gc.run(mode=mode, dry_run=dry_run)

    def retire(self, entry_id: str, reason: str) -> Dict[str, Any]:
        card = self.sbank.retire(entry_id, reason=reason)
        # The entry left the hot store, so its vector must leave the index
        # too: a vector that outlives its record would surface a claim that
        # no longer exists.
        self.index_sync.forget_entry(entry_id)
        return {"retired": entry_id, "cold_archive_card": card.to_dict()}

    # -- retrieval index maintenance ----------------------------------------------

    def rebuild_index(self, layer: str = "both",
                      dry_run: bool = False) -> Dict[str, Any]:
        """Rebuild the retrieval index from the current facts and entries.

        EXPLICIT maintenance (first build, repair after a model change,
        settling a deferred sync) — not a routine path: ``record`` /
        ``induce`` keep the index fresh incrementally. ``layer`` is
        ``"both"`` / ``"execution"`` / ``"strategic"``; ``dry_run`` counts
        what would be indexed and touches nothing.

        This is a WRITE operation, so it is the right place for it: unlike
        ``recall`` / ``inspect`` / ``doctor`` it may embed, write files and
        re-key vectors. It never rewrites a fact or an entry — only derived
        index data.
        """
        return self.index_sync.rebuild(layer=layer, dry_run=dry_run)

    def index_health(self) -> Dict[str, Any]:
        """Read-only index health (counts, model id, stale/missing items)."""
        return self.index_sync.health()

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
            "retrieval_index": self.index_health(),
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
            # Shared measured-only accumulation kernel (same arithmetic as
            # BudgetLedger.consumption; the AGGREGATION SCOPE differs and
            # stays here: recorded attempts only).
            accumulate_measured_costs([rec.cost], total, n_measured)
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
