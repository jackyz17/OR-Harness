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
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

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
    SERVICE_IMPLEMENTED_KINDS,
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
    PredictionServiceStatus,
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
    parse_window_id,
    window_from_records,
    window_identity_problems,
)
from or_harness.world_model.context import (
    PREDICTION_CONTEXT_VERSION,
    JointProblemRepresentation,
    PredictionContext,
    RetrievalView,
    UnsupportedContextVersion,
    build_context,
    build_retrieval_view,
    capability_evidence_with_sources,
    capability_version,
    cell_evidence_from_stats,
    context_identity_problems,
    effective_input_version,
    evidence_identity,
    frozen_knowledge_available,
    frozen_knowledge_view,
    knowledge_targets_from_context,
    memory_content_digest,
    resolve_effective_cir,
    retrieval_reuse_problems,
    snapshot_from_context,
    structure_problems,
    task_with_effective_cir,
)
from or_harness.world_model.prediction import (
    ActionSpec,
    OutcomePrediction,
)
from or_harness.world_model.contracts import (
    BaselineStatement,
    CapabilityEvolutionPrediction,
    EvidenceRef,
    ExperienceScope,
    LearningOperation,
    TaskTargeting,
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

#: The learning-operation types this build can really CARRY OUT through an
#: existing path. A candidate of another type may be predicted, compared and
#: deferred, but accepting it must NOT quietly run a different operation:
#: the binding would then attest to an operation that never happened.
CAPABILITY_EXECUTABLE_OPERATIONS = ("induce", "revise", "retire")
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
        # World-model M3 (this round): the strategy-outcome prediction
        # SERVICE under the wm-so/1 protocol. Same provider, same explicit
        # injection discipline — the service is what turns a frozen context
        # + a candidate into a StrategyOutcomePrediction.
        from or_harness.world_model.strategy_prediction import (
            StrategyOutcomeService,
        )
        self.strategy_predictions = StrategyOutcomeService(
            self.store, self.world_model)
        # World-model M5: the capability-evolution prediction SERVICE under
        # the wm-ce/1 protocol. Same provider, same explicit injection
        # discipline; a SEPARATE prediction table, so a capability record
        # can never be read as an OR strategy prediction.
        from or_harness.world_model.capability_evolution import (
            CapabilityEvolutionService,
        )
        self.capability_predictions = CapabilityEvolutionService(
            self.store, self.world_model)
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
                                  round_index: Optional[int] = None,
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

        ``round_index`` (M4) selects ONE selection round of this (task,
        episode, strategy): the same strategy chosen again later (possibly
        under a different config) is a DIFFERENT round and a different
        evaluation sample. ``None`` keeps the legacy whole-episode
        aggregation.
        """
        return window_from_records(
            self, task_id, episode_id, strategy_id,
            round_index=round_index,
            in_scope_action_types=in_scope_action_types,
            auxiliary_action_types=auxiliary_action_types)

    # -- world-model phase 2: the prediction input context -----------------------

    def _memory_created_at(self, layer: str,
                           evidence_id: str) -> Optional[float]:
        """When the memory behind one evidence id came into existence.

        Used to bound a retrieval to a frozen moment: a memory created AFTER
        a historical snapshot is not part of that state, so it must not enter
        a context built from that snapshot.
        """
        try:
            if layer == "execution_evidence":
                record = self.bank.get(str(evidence_id))
                return (float(record.created_at)
                        if record is not None else None)
            entry = self.sbank.get(str(evidence_id))
            return float(entry.created_at) if entry is not None else None
        except Exception:
            return None

    def _bound_recall_result(self, recall_result: Dict[str, Any],
                             as_of: Optional[float]) -> Tuple[
                                 Dict[str, Any], Dict[str, Any]]:
        """Drop memories that POSTDATE ``as_of``, and report what was dropped.

        A context built from a historical snapshot describes the state as it
        was; evidence that arrived afterwards belongs to a later state and
        must not leak in. A memory whose creation time cannot be established
        is KEPT but counted as unbounded — dropping it would silently discard
        real evidence, and keeping it silently would claim a bound that was
        never verified.
        """
        if as_of is None:
            return recall_result, {}
        result = copy.deepcopy(recall_result)
        dropped: List[Dict[str, Any]] = []
        unbounded = 0

        def _keep(layer: str, evidence_id: str) -> bool:
            nonlocal unbounded
            created = self._memory_created_at(layer, evidence_id)
            if created is None:
                unbounded += 1
                return True
            if created > as_of:
                dropped.append({"layer": layer, "evidence_id": evidence_id,
                                "created_at": created})
                return False
            return True

        kept_recs = []
        for rec in result.get("recommendations") or []:
            refs = [str(r) for r in (rec.get("evidence_refs") or [])
                    if str(r).startswith("se_")]
            if refs:
                if all(_keep("strategic_knowledge", r) for r in refs):
                    kept_recs.append(rec)
                continue
            kept_recs.append(rec)
        result["recommendations"] = kept_recs

        vector = result.get("vector_recall")
        if isinstance(vector, dict):
            vector["execution_evidence"] = [
                row for row in (vector.get("execution_evidence") or [])
                if _keep("execution_evidence", str(row.get("execution_id")))]
            vector["strategic_knowledge"] = [
                row for row in (vector.get("strategic_knowledge") or [])
                if _keep("strategic_knowledge", str(row.get("entry_id")))]

        bounding = {
            "as_of": float(as_of),
            "dropped": len(dropped),
            "dropped_items": dropped[:20],
            "unbounded_kept": unbounded,
            "note": ("this retrieval was bounded to the frozen moment: "
                     "memories created AFTER it were dropped rather than "
                     "entering a context that describes an earlier state"
                     if dropped else
                     "this retrieval was bounded to the frozen moment; no "
                     "memory postdating it was encountered"),
        }
        return result, bounding

    def build_prediction_context(
            self, task: Dict[str, Any],
            episode_id: Optional[str] = None, *,
            top: int = 3,
            vector_top_k: Optional[int] = None,
            include_unverified: bool = False,
            cir: Optional[Any] = None,
            math: Optional[Dict[str, Any]] = None,
            recall_result: Optional[Dict[str, Any]] = None,
            snapshot: Optional[BeliefSnapshot] = None,
            context_spec: Optional[ActionSpec] = None,
            candidates: Optional[Sequence[ActionSpec]] = None,
            historical: Optional[bool] = None,
            persist: bool = True) -> PredictionContext:
        """Build the FROZEN prediction input context for one decision.

        This is the phase-2 entry point: ONE call gathers everything a
        prediction may condition on, freezes it, and (by default) persists
        it so the input can be reproduced later.

        What it does:

        - freezes ONE belief snapshot (unless one is supplied) — the X/B and
          capability evidence are read from that same snapshot, so a bank
          that moves afterwards cannot change the context;
        - runs the EXISTING recall path once (both channels), or reuses a
          supplied result after verifying it belongs to the CURRENT task
          version;
        - assembles the joint problem representation (task text + payload,
          CIR relations, math attributes with origins, profile + derivation
          report) and the capability evidence (phase-1 contract, plus what
          this phase can really observe).

        What it does NOT do: no prediction-model call, no solver execution,
        no induction, no capability re-measurement. It may call the embedding
        backend through the existing retrieval path — that is the one
        external call, and it is a read.

        **Current gathering vs historical reconstruction.** The two are
        different operations and the ``historical`` flag is what separates
        them — NOT the mere presence of a snapshot:

        - ``historical=False`` (the default when no snapshot is supplied, or
          when the caller explicitly passes it): this decision gathers the
          CURRENT information and freezes it. A snapshot the CALLER supplies
          is the frozen X/B to gather around (``plan_next`` passes its root
          snapshot this way so the whole comparison shares one state); the
          retrieval, the knowledge view, the reliability table and the cell
          evidence are all read NOW and frozen.
        - ``historical=True`` (the default when an EXTERNAL snapshot is
          supplied without an explicit flag): the snapshot is an HISTORICAL
          artifact fixing the moment this context describes, and everything
          the prediction conditions on must come from what that snapshot
          SAVED — never from today's banks (a later revision of an entry
          would otherwise leak into a context describing the earlier state).

        ``recall_result`` reuse is refused when its recorded ``task_digest``
        disagrees with the current task, a supplied ``snapshot`` is refused
        when it belongs to another task / episode / version, and a supplied
        recall result is REFUSED on the historical path (a result gathered
        now cannot be proven to belong to the frozen moment — creation-time
        filtering cannot see a later revision of an existing entry). Build a
        new context instead of silently conditioning on stale inputs.

        ``persist=False`` skips the store write (useful when the caller only
        wants the object, e.g. inside a planning call that already records
        its own root snapshot).

        ``context_spec`` / ``candidates`` name the candidate(s) this context
        is for. The structural knowledge targets are proposed for them and
        FROZEN into the context, so a reused context sends the proposal set
        as it stood at build time. With neither, the context still carries
        the representation, X/B, retrieval evidence and capability evidence —
        it simply proposes no targets (and says so).
        """
        if recall_result is not None:
            supplied = recall_result.get("task_digest") \
                if isinstance(recall_result, dict) else None
            problems = retrieval_reuse_problems(
                RetrievalView(task_digest=(str(supplied) if supplied
                                           else None)),
                task_digest=task_text_digest(task))
            if problems:
                raise ValueError("the supplied recall result cannot be "
                                 "reused: " + "; ".join(problems))

        # ONE effective CIR for this request, resolved BEFORE any identity
        # check. Routing the SAME structure into the profile, the snapshot
        # and the recall is what stops one request from carrying two
        # structural judgments; checking the supplied artifacts FIRST is what
        # stops a snapshot built from a different structure from passing the
        # identity check and then being kept alongside the new CIR.
        effective_task = task_with_effective_cir(task, cir)
        resolved_cir = effective_task.get("coupling")
        profile = self.profile(effective_task)
        derivation = derivation_report(profile)
        # The version identity of the EFFECTIVE input (task + resolved CIR).
        # Identity checks below compare against THIS, not the plain task
        # digest: a context built with an explicit CIR must be reusable
        # against the same task, and the plain digests differ only because
        # the caller-supplied task JSON differs.
        effective_digest = effective_input_version(task, cir)

        # -- historical reconstruction vs current gathering -------------------
        # An EXTERNAL snapshot (one the caller did not just take for this
        # decision) fixes a historical moment: everything must then come
        # from what it SAVED. A snapshot the caller passes together with
        # historical=False is the frozen X/B of a CURRENT gathering (the
        # plan_next root-snapshot pattern) — the banks are read NOW and
        # frozen, which is exactly what a live decision needs.
        if historical is None:
            historical = snapshot is not None
        history_notes: List[str] = []
        if snapshot is not None:
            problems = context_identity_problems(
                PredictionContext(context_id="(supplied snapshot)",
                                  task_id=str(getattr(snapshot, "task_id",
                                                      "")),
                                  task_digest=str(
                                      (getattr(snapshot, "problem_state",
                                               None) or {}).get(
                                          "task_digest") or ""),
                                  episode_id=getattr(snapshot, "episode_id",
                                                     None)),
                task_id=str(task.get("task_id", "")),
                task_digest=task_text_digest(task),
                episode_id=episode_id,
                allow_unknown_digest=True)
            # A snapshot of another episode is a mismatch; a snapshot with no
            # recorded version cannot be established either way, so it is
            # used but reported as unconfirmed.
            hard = [p for p in problems if "records no task version"
                    not in p]
            if hard:
                raise ValueError("the supplied snapshot does not describe "
                                 "this task: " + "; ".join(hard))
            # Structure: the snapshot's own frozen profile must describe the
            # SAME structure as the effective input. A conflict is refused —
            # the joint representation would otherwise speak about one
            # problem while the state spoke about another.
            frozen_profile = (getattr(snapshot, "problem_state", None)
                              or {}).get("profile")
            structure = structure_problems(
                expected=profile, actual=frozen_profile,
                label="the supplied snapshot")
            if structure:
                raise ValueError(
                    "the supplied snapshot cannot be combined with this "
                    "problem input: " + "; ".join(structure) +
                    ". Build a new context from the current input instead of "
                    "reusing a snapshot taken under a different structure.")
        if snapshot is None:
            snapshot = self.snapshot(effective_task, episode_id)

        bounding: Dict[str, Any] = {}
        if recall_result is not None:
            if historical:
                # A result gathered NOW cannot be PROVEN to belong to the
                # frozen moment: creation-time filtering sees when a memory
                # was created, not when an EXISTING entry was later revised,
                # so a revised entry would slip through at its new value.
                # The historical path accepts only retrieval that was SAVED
                # with the snapshot; anything else is refused.
                raise ValueError(
                    "a recall result gathered now cannot be used for a "
                    "historical reconstruction: it cannot be proven to "
                    "belong to the frozen moment (an entry revised after "
                    "the snapshot would enter at its NEW value). The "
                    "historical path accepts only retrieval that was saved "
                    "with the snapshot; build a current context instead")
            if snapshot is not None:
                # A SUPPLIED result is current-collected, so it may carry
                # memory that postdates the snapshot this context freezes
                # around. It is bounded the same way a freshly gathered one
                # is — a supplied result does not get to smuggle later
                # evidence past the moment the decision froze.
                recall_result, bounding = self._bound_recall_result(
                    recall_result,
                    float(getattr(snapshot, "created_at", 0.0) or 0.0))
        elif historical:
            # No historical retrieval was SAVED with this snapshot, so
            # there is none to rebuild. Reading today's index would
            # fabricate a retrieval for an earlier state; the gap is
            # reported instead.
            recall_result = {"recommendations": []}
            bounding = {"as_of": float(getattr(snapshot, "created_at",
                                               0.0) or 0.0),
                        "rebuilt_from": "none_saved",
                        "dropped": 0, "dropped_items": [],
                        "unbounded_kept": 0,
                        "note": ("no retrieval was saved with this "
                                 "historical snapshot, so no retrieval "
                                 "was rebuilt: reading today's index "
                                 "would fabricate evidence for an "
                                 "earlier state. The evidence set is "
                                 "EMPTY BECAUSE IT WAS NOT RECORDED, "
                                 "not because nothing existed")}
        else:
            recall_result = self.recall(
                effective_task, top=top,
                include_unverified=include_unverified,
                vector_top_k=vector_top_k)
            if snapshot is not None:
                # A CURRENT gathering around a caller-supplied snapshot is
                # still bounded to that snapshot's moment: the snapshot is
                # the X/B this decision froze, and evidence that arrived
                # AFTER it belongs to a later state.
                recall_result, bounding = self._bound_recall_result(
                    recall_result,
                    float(getattr(snapshot, "created_at", 0.0) or 0.0))

        if historical:
            # Knowledge and reliability come from the SNAPSHOT's saved
            # content, and a gap is reported as missing.
            if frozen_knowledge_available(snapshot):
                knowledge = frozen_knowledge_view(snapshot)
                history_notes.append(
                    "knowledge was read from the snapshot's FROZEN "
                    "knowledge view (the entries as they stood when the "
                    "snapshot was taken), never from today's bank")
            else:
                knowledge = {"verified": [], "legacy_unknown": [],
                             "unverified": []}
                history_notes.append(
                    "MISSING: this snapshot carries no frozen knowledge "
                    "view, so the knowledge component is EMPTY BY ABSENCE — "
                    "today's bank was NOT consulted to fill the gap")
            reliability = {}
            history_notes.append(
                "MISSING: the measured prediction reliability was not saved "
                "with this snapshot, so none is sent (today's prediction log "
                "was NOT consulted)")
        else:
            knowledge = verified_knowledge_view(profile, self.sbank)
            reliability = self.prediction_reliability_table()
        # M4: the published EXPERIENCE calibration of the strategy-outcome
        # service (closed episodes only). Frozen into the context as its
        # own block — separate from the legacy knowledge reliability, and
        # carrying the summary's CONTENT (groups, counts, sources) so a
        # later reader of the context sees what the model was told, not
        # just an id. An active episode never sees its own not-yet-closed
        # feedback: only closed episodes contribute.
        if historical:
            strategy_calibration = {}
            history_notes.append(
                "MISSING: the strategy-outcome experience calibration was "
                "not saved with this snapshot, so none is sent (today's "
                "summary was NOT consulted for a historical reconstruction)")
        else:
            from or_harness.world_model.episode_closeout import (
                calibration_summary_for_context,
            )
            strategy_calibration = calibration_summary_for_context(self)
        retrieval_view = build_retrieval_view(
            recall_result, task_digest=task_text_digest(task),
            top_k=max(1, vector_top_k or top),
            include_unverified=include_unverified)
        recorded_choices = self._recorded_choice_summary(
            str(task.get("task_id", "")), episode_id) if not historical else {}
        evidence = capability_evidence_with_sources(
            knowledge=knowledge,
            coverage=getattr(snapshot, "coverage", None),
            experience=(getattr(snapshot, "harness_state", None) or {}).get(
                "experience"),
            tools=(getattr(snapshot, "harness_state", None) or {}).get(
                "tool_config"),
            retrieval_view=retrieval_view,
            reliability=reliability,
            recorded_choices=recorded_choices)
        if historical:
            constraints = self._historical_constraints_view(snapshot)
        else:
            constraints = self._execution_constraints_view(effective_task,
                                                           episode_id)
        version_block = capability_version(
            harness_config={
                "alpha": self.selector.alpha, "beta": self.selector.beta,
                "gamma": self.selector.gamma, "delta": self.delta,
                "prediction_mode": self.prediction_mode,
                "plan_mode": self.plan_mode,
                "induction_assessment": self.induction_assessment_mode,
            },
            provider=self.world_model.describe(),
            prompt_template_version=self._prompt_template_version(),
            tools=constraints.get("available_solver_families") or [],
            knowledge_content=memory_content_digest(
                knowledge=knowledge, retrieval=retrieval_view,
                entries=([] if historical
                         else self.sbank.list(include_dormant=True))))
        # Freeze the structural proposal set and the cell evidence the
        # proposal was derived from. Both come from live banks, so a reused
        # context must carry them rather than re-derive them.
        spec_list = ([context_spec] if context_spec is not None
                     else list(candidates or []))
        frozen_targets: List[Any] = []
        if historical:
            # The knowledge-target heuristic reads the cell's REAL evidence
            # records (``stats.evidence``) — a live bank read. A historical
            # reconstruction may not consult today's records: an execution
            # recorded after the snapshot would change the proposal. No
            # proposal was SAVED with the snapshot, so none is made and the
            # gap is reported.
            history_notes.append(
                "MISSING: the structural knowledge-target proposal was not "
                "saved with this snapshot, and re-deriving it would read "
                "today's evidence records, so no target is proposed for a "
                "historical reconstruction")
        else:
            for spec in spec_list:
                frozen_targets.extend(self.knowledge_targets(snapshot, spec))
        strategies = [s.strategy_id for s in spec_list if s.strategy_id]
        if historical:
            # Cell statistics describe the bank as it is NOW, so they are not
            # part of a reconstruction. The gap is recorded instead.
            cell_evidence: Dict[str, Any] = {}
            history_notes.append(
                "MISSING: per-cell evidence counts were not saved with this "
                "snapshot, so the reconstruction carries none (today's "
                "statistics were NOT consulted)")
        else:
            cell_evidence = cell_evidence_from_stats(self.stats, profile,
                                                     strategies)
        context = build_context(
            task=task, profile=profile, snapshot=snapshot,
            recall_result=recall_result, derivation=derivation,
            cir=resolved_cir, math_declared=math, capability=evidence,
            capability_version_block=version_block,
            execution_constraints=constraints,
            knowledge_targets=frozen_targets,
            reliability=reliability,
            cell_evidence=cell_evidence,
            strategy_calibration=strategy_calibration,
            cir_source=("caller_supplied" if cir is not None
                        else "task_coupling"),
            retrieval_bounding=bounding,
            top_k=max(1, vector_top_k or top),
            include_unverified=include_unverified)
        if bounding:
            context.execution_constraints = {
                **(context.execution_constraints or {}),
                "retrieval_bounding": copy.deepcopy(bounding)}
        if historical:
            context.missing.extend(history_notes)
            context.notes.append(
                "HISTORICAL reconstruction: every condition was taken from "
                "the supplied snapshot's saved content; nothing was read "
                "from today's banks, and each gap is reported as missing "
                "rather than filled")
        context.notes.append(
            "this context is the FROZEN input of a prediction; building it "
            "performed no model call, no solver execution and no induction")
        if persist:
            self.store.put_prediction_context(
                context.context_id, context.task_id, context.episode_id,
                self.store.dumps(context.to_dict()),
                created_at=context.created_at)
        return context

    def get_prediction_context(self,
                               context_id: str) -> Optional[PredictionContext]:
        """Read a stored context back (never re-runs retrieval or a model).

        The version is CHECKED, not guessed at: a payload written by a build
        that used a different context schema is refused rather than
        half-read."""
        raw = self.store.get_prediction_context(context_id)
        if raw is None:
            return None
        return PredictionContext.from_dict(self.store.loads(raw))

    def prediction_contexts(self, *, task_id: Optional[str] = None,
                            episode_id: Optional[str] = None
                            ) -> List[PredictionContext]:
        """Every stored context (newest first), optionally filtered."""
        out: List[PredictionContext] = []
        for raw in self.store.prediction_contexts_for(task_id=task_id,
                                                      episode_id=episode_id):
            try:
                out.append(PredictionContext.from_dict(self.store.loads(raw)))
            except (UnsupportedContextVersion, ValueError):
                continue
        return out

    def _prompt_template_version(self) -> str:
        from or_harness.world_model.prediction import PROMPT_TEMPLATE_VERSION
        return PROMPT_TEMPLATE_VERSION

    def _execution_constraints_view(self, task: Dict[str, Any],
                                    episode_id: Optional[str]
                                    ) -> Dict[str, Any]:
        """External constraints on what may be executed for this task.

        Real declarations and real availability only: the declared budget and
        its consumption view, the solver families actually importable in
        this environment, and the executor's limits. Nothing here is a
        capability claim — availability is a precondition, not an ability.
        """
        task_id = str(task.get("task_id", ""))
        # Reuse the ONE budget loader (its cache key format lives with it) so
        # a declared budget is read the same way here as everywhere else.
        declared = self._load_budget(task_id, episode_id)
        budget = self.budget.view(task_id, episode_id, budget=declared)
        return {
            "declared_budget": copy.deepcopy(declared) if declared else None,
            "budget_status": budget.get("status"),
            "available_solver_families": available_families(),
            "executor": {
                "timeout_seconds": getattr(self.executor, "timeout_seconds",
                                           None),
                "workspace_policy": ("solve scripts must live inside their "
                                     "execution workspace"),
            },
            "note": ("declared budget and real tool availability: these "
                     "constrain what can be executed, they do not predict "
                     "what it would achieve"),
        }

    def _historical_constraints_view(self,
                                     snapshot: BeliefSnapshot) -> Dict[str, Any]:
        """The execution constraints of a HISTORICAL reconstruction.

        Only what the snapshot itself froze: its ``budget_state`` (the
        declaration and consumption view as they stood). The CURRENT
        declared budget, today's consumption and today's tool availability
        describe a later state — presenting them as the conditions of an
        earlier moment would fabricate history. Each gap is reported.
        """
        frozen_budget = copy.deepcopy(
            getattr(snapshot, "budget_state", None) or {})
        return {
            "declared_budget": copy.deepcopy(frozen_budget.get("budget")
                                             or None),
            "budget_status": frozen_budget.get("status"),
            "budget_state_frozen": frozen_budget,
            "available_solver_families": None,
            "executor": None,
            "note": ("HISTORICAL constraints: the budget view is the one the "
                     "snapshot FROZE; today's declared budget, consumption "
                     "and tool availability were NOT consulted, so they are "
                     "reported missing rather than substituted"),
            "missing": [
                "available_solver_families: today's tool availability is "
                "not a condition of the frozen moment",
                "executor limits: not saved with the snapshot",
            ],
        }

    def _recorded_choice_summary(self, task_id: str,
                                 episode_id: Optional[str]
                                 ) -> Dict[str, Any]:
        """What choices were RECORDED for this task (evidence about Pi).

        Observations only: a recorded selection or rejection says what was
        decided, never that the decision was optimal. An empty result means
        nothing was recorded, which is not evidence of poor selection.
        """
        actions = [a for a in self.actions.query(task_id=task_id)
                   if a.action_type in ("select_strategy", "execute_strategy")]
        if episode_id is not None:
            actions = [a for a in actions if a.episode_id == episode_id]
        chosen = [str((a.params or {}).get("strategy_id"))
                  for a in actions
                  if (a.params or {}).get("strategy_id")]
        return {
            "n_recorded": len(actions),
            "strategies_seen": sorted(set(chosen)),
            "note": ("recorded selections and executions are OBSERVATIONS of "
                     "what was decided; they are not evidence that the "
                     "decision was optimal"),
        }

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
            prediction_completed: Optional[bool] = None,
            ) -> StrategyOutcomePrediction:
        """Build (and validate) a :class:`StrategyOutcomePrediction`.

        The trace is filled from what the framework already knows: the
        frozen input snapshot, its task digest, the prediction-service
        state of THIS instance, and the generation time. A caller never
        re-types the contract version or the prediction type.

        **Three different facts are kept apart**, because conflating them is
        how an empty object came to be labelled a forecast:

        1. ``provider_configured`` — a provider object is attached;
        2. ``service_available`` — this build implements the service AND a
           provider is configured;
        3. ``prediction_completed`` — a prediction really was produced and
           passed validation.

        Only (3) yields ``status="valid"``. Merely constructing a contract
        returns ``contract_only``: the contract is implemented, no forecast
        was made — and a configured provider with zero model calls is NOT a
        prediction. ``prediction_completed`` is inferred from the supplied
        content (a benefit/cost/risk/uncertainty object is what a prediction
        IS) and may be set explicitly by a caller that knows better.
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
        # A caller-supplied window_id must name THIS candidate's window. A
        # string pointing at another task/episode/strategy is an outright
        # mismatch, never a comparable scope.
        if candidate.window_id:
            parsed = parse_window_id(candidate.window_id)
            if parsed is not None:
                mismatches = window_identity_problems(
                    parsed,
                    task_id=candidate.task_id,
                    episode_id=candidate.episode_id,
                    strategy_id=candidate.strategy_id)
                if mismatches:
                    raise ValueError(
                        "candidate.window_id does not describe this "
                        "candidate: " + "; ".join(mismatches))
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
            # A window handed in by the caller is checked against the
            # candidate it is being used for: a window of another task,
            # episode or strategy may NOT become this prediction's scope.
            mismatches = window_identity_problems(
                window, task_id=candidate.task_id,
                episode_id=candidate.episode_id,
                strategy_id=candidate.strategy_id)
            if mismatches:
                raise ValueError(
                    "the supplied window does not describe this candidate: "
                    + "; ".join(mismatches))
            candidate.window_id = candidate.window_id or window.window_id
            trace.comparable = bool(window.comparable)
            trace.not_comparable_reasons = list(window.not_comparable_reasons)
            notes.append(
                "window scope: " + window.scope_rationale)
            if not window.comparable:
                notes.append(
                    "NOT comparable: this window is not a completed real "
                    "scope, so no window-scope result may be scored against "
                    "it (" + "; ".join(window.not_comparable_reasons) + ")")
        else:
            trace.comparable = True
            notes.append(
                "attempt scope: this prediction covers ONE solve attempt, "
                "not the whole strategy execution (modeling / repair / "
                "verify are auxiliary and reported separately)")
        has_content = any(value is not None for value in
                          (benefit, cost, risk, uncertainty))
        completed = bool(has_content if prediction_completed is None
                         else prediction_completed)
        service = self.prediction_service_status(
            "strategy_outcome", prediction_completed=completed)
        if not service.provider_configured:
            notes.append(
                "contract_only: no prediction provider is attached to this "
                "instance, so no benefit/cost/risk values were produced")
        elif not service.service_available:
            notes.append(
                "contract_only: a provider is configured but this build "
                "implements no service for this kind")
        elif not completed:
            notes.append(
                "contract_only: a provider is configured but NO prediction "
                "was produced for this candidate (no benefit/cost/risk/"
                "uncertainty content), so no forecast was made — a "
                "configured provider is not a prediction")
        prediction = StrategyOutcomePrediction(
            candidate=candidate,
            status=service.status,
            benefit=benefit,
            cost=cost,
            risk=risk,
            uncertainty=uncertainty,
            trace=trace,
            service_available=service.service_available,
            provider_configured=service.provider_configured,
            service_implemented=service.service_implemented,
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
            prediction_completed: Optional[bool] = None,
            ) -> CapabilityEvolutionPrediction:
        """Build (and validate) a :class:`CapabilityEvolutionPrediction`.

        The current capability evidence is read from the frozen snapshot
        when one is available, otherwise from the live banks.

        This is the SCHEMA-level builder. It performs no model call: to
        actually PREDICT, use
        :meth:`predict_capability_evolution`, which drives the ``wm-ce/1``
        service. Building a contract here yields ``contract_only`` unless
        the caller really supplies the observable consequences (see
        ``prediction_completed``) — a configured provider with zero model
        calls is not a forecast.
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
        changes = list(expected_changes or [])
        completed = bool(changes if prediction_completed is None
                         else prediction_completed)
        service = self.prediction_service_status(
            "capability_evolution", prediction_completed=completed)
        if not service.service_implemented:
            notes.append(
                "contract_only: this build implements the capability "
                "evolution CONTRACT but NO service for it, so no "
                "performance change is forecast — configuring a provider "
                "does not implement the service. H is judged through "
                "observable consequences, never a latent vector")
        elif not service.provider_configured:
            notes.append(
                "contract_only: no prediction provider is attached to this "
                "instance")
        elif not completed:
            notes.append(
                "contract_only: a provider is configured but NO capability "
                "forecast was produced (no expected_changes), so nothing "
                "was predicted")
        if not verification_conditions:
            notes.append(
                "no verification condition declared: predicting, binding "
                "the fact, and verifying the effect are three different "
                "things, and only the third supports a claim of improvement")
        prediction = CapabilityEvolutionPrediction(
            current_evidence=evidence,
            candidate_operation=operation,
            status=service.status,
            experience_scope=experience_scope,
            task_targeting=task_targeting,
            baseline=baseline,
            horizon=horizon,
            horizon_tasks=horizon_tasks,
            expected_changes=changes,
            learning_cost=learning_cost,
            degradation_risk=degradation_risk,
            uncertainty=uncertainty,
            verification_conditions=list(verification_conditions or []),
            trace=trace,
            service_available=service.service_available,
            provider_configured=service.provider_configured,
            service_implemented=service.service_implemented,
            notes=notes,
        )
        problems = validate_capability_evolution(prediction)
        if problems:
            prediction.status = "invalid"
            prediction.notes.extend(f"validation: {p}" for p in problems)
        return prediction

    def prediction_service_status(
            self, kind: str, *,
            prediction_completed: bool = False) -> PredictionServiceStatus:
        """The THREE separate facts about a prediction service for a kind.

        Kept apart on purpose, because reporting one as another is exactly
        how an empty contract gets read as a forecast:

        1. ``provider_configured`` — a provider object is attached to this
           instance;
        2. ``service_available`` — this build IMPLEMENTS a service for this
           kind *and* a provider is configured. Configuring a provider does
           not implement a service: a kind in ``PREDICTION_KINDS`` but not in
           ``SERVICE_IMPLEMENTED_KINDS`` stays unavailable no matter what is
           configured.
        3. ``prediction_completed`` — a prediction really was produced and
           passed validation. Only this makes a contract ``valid``.
        """
        if kind not in PREDICTION_KINDS:
            raise ValueError(f"unknown prediction kind {kind!r}")
        configured = not isinstance(self.world_model, NotConfiguredProvider)
        return PredictionServiceStatus(
            kind=kind,
            provider_configured=configured,
            service_implemented=kind in SERVICE_IMPLEMENTED_KINDS,
            prediction_completed=bool(prediction_completed),
            provider_name=self.prediction_service_version(),
        )

    def prediction_service_available(self, kind: str) -> bool:
        """Whether a usable prediction SERVICE exists for one kind.

        Deliberately stronger than "a provider is configured": a provider
        serves nothing for a kind this build does not implement. It is still
        weaker than "a prediction was made" — see
        :meth:`prediction_service_status`, and note that constructing a
        contract never makes a prediction happen.
        """
        return self.prediction_service_status(kind).service_available

    def prediction_service_version(self) -> str:
        """Version label of the attached prediction service (or its absence)."""
        if isinstance(self.world_model, NotConfiguredProvider):
            return "not-attached"
        return str(getattr(self.world_model, "name", "unknown"))

    # -- world-model M3: strategy-outcome prediction service (wm-so/1) --------

    def predict_strategy_outcome(
            self, task: Dict[str, Any],
            candidate: Union[CandidateRef, ActionSpec, Dict[str, Any]],
            episode_id: Optional[str] = None, *,
            context: Optional[PredictionContext] = None,
            cir: Optional[Any] = None,
            timeout_s: Optional[float] = None,
            benefit_baseline_hint: Optional[Dict[str, Any]] = None,
            ) -> "StrategyOutcomePrediction":
        """Predict ONE candidate's consequences under the wm-so/1 protocol.

        The input is a FROZEN context and a candidate; the output is a
        :class:`~or_harness.world_model.contracts.StrategyOutcomePrediction`
        (benefit G with metric/unit/baseline, resource cost c as a
        CostVector with its predicted-dimension mask, risk L as named
        events separate from cost, uncertainty with execution randomness
        separated from the knowledge gap), persisted so it can be bound to
        the real execution later.

        ``context`` decides the frozen input: a supplied
        :class:`PredictionContext` is reused after its identity (including
        the EFFECTIVE input version — pass the same ``cir`` you built it
        with) is verified; ``None`` builds one current context for this
        call. ``cir`` resolves the effective problem input exactly as
        ``build_prediction_context(..., cir=...)`` does.

        The candidate may be a ``CandidateRef``, a legacy ``ActionSpec``
        (mapped through ``from_action_spec``, which preserves its execution
        config verbatim and refuses an unmappable scope), or a candidate
        dict. Same strategy_id with different solver / time_limit / mip_gap
        / seed / step scope is a DIFFERENT candidate — the config travels
        with the candidate, so predictions, choices and executions bind to
        the configuration actually proposed.

        One provider call, no retries, no defaults: a failed call is a
        persisted failure with whatever usage it consumed.
        """
        from or_harness.world_model.strategy_prediction import (
            StrategyOutcomeService,
        )
        if isinstance(candidate, ActionSpec):
            candidate_ref = CandidateRef.from_action_spec(candidate)
        elif isinstance(candidate, dict):
            candidate_ref = CandidateRef.from_dict(candidate)
        else:
            candidate_ref = copy.deepcopy(candidate)
        task_id = str(task.get("task_id", ""))
        if candidate_ref.task_id and candidate_ref.task_id != task_id:
            raise ValueError(
                f"candidate.task_id {candidate_ref.task_id!r} does not "
                f"match the task being predicted for ({task_id!r})")
        if not candidate_ref.task_id:
            candidate_ref.task_id = task_id
        if candidate_ref.episode_id is None:
            candidate_ref.episode_id = episode_id

        if context is None:
            context = self.build_prediction_context(
                task, episode_id, cir=cir, context_spec=None,
                candidates=None)
        else:
            problems = context_identity_problems(
                context, task_id=task_id,
                task_digest=task_text_digest(task),
                episode_id=episode_id,
                effective_input_digest=effective_input_version(task, cir))
            problems = problems + structure_problems(
                expected=self.profile(task_with_effective_cir(task, cir)),
                actual=context.joint.profile,
                label="the supplied prediction context")
            if problems:
                raise ValueError(
                    "the supplied prediction context does not describe this "
                    "prediction: " + "; ".join(problems))
        service: StrategyOutcomeService = self.strategy_predictions
        prediction = service.predict(
            context, candidate_ref, timeout_s=timeout_s,
            benefit_baseline_hint=benefit_baseline_hint)
        # The frozen input reference, on every outcome (failures included):
        # a failed call was still made against this input.
        prediction.trace.model_info["prediction_context_id"] = \
            context.context_id
        prediction.trace.model_info["prediction_context_version"] = \
            context.version
        prediction.trace.model_info["prediction_context_task_digest"] = \
            context.task_digest
        prediction.trace.model_info["effective_input_digest"] = \
            context.effective_input_digest
        service._save(prediction)
        return prediction

    def get_strategy_outcome_prediction(
            self, prediction_id: str
    ) -> Optional["StrategyOutcomePrediction"]:
        """Read a stored strategy-outcome prediction (never calls a model)."""
        return self.strategy_predictions.get(prediction_id)

    def strategy_outcome_predictions(
            self, *, task_id: Optional[str] = None,
            episode_id: Optional[str] = None
    ) -> List["StrategyOutcomePrediction"]:
        """Stored strategy-outcome predictions, optionally filtered."""
        return self.strategy_predictions.query(task_id=task_id,
                                               episode_id=episode_id)

    # -- world-model M4: episode close-out and experience calibration ------

    def close_episode(self, task_id: str, episode_id: Optional[str] = None,
                      *, terminal_state: str = "completed",
                      finish_action_id: Optional[str] = None,
                      min_calibration_samples: Optional[int] = None
                      ) -> Dict[str, Any]:
        """Close ONE episode: evaluate its bound predictions against their
        real outcomes and publish the experience calibration.

        The single M4 close-out entry. It reads what was recorded — no
        solver run, no model call, no induction. Unfinished actions are
        reported (their predictions stay pending, never fabricated into
        endings); a failed/aborted/budget-exhausted episode closes honestly
        under its own terminal state. Idempotent: closing an already-closed
        episode returns the stored record and counts nothing twice.

        ``finish_action_id`` optionally names the ``finish_task`` action
        that ended the episode (the close-out links to it; it does not
        create a second task lifecycle).
        """
        from or_harness.world_model.episode_closeout import (
            DEFAULT_MIN_CALIBRATION_SAMPLES,
            close_episode,
        )
        return close_episode(
            self, task_id, episode_id, terminal_state=terminal_state,
            finish_action_id=finish_action_id,
            min_calibration_samples=(min_calibration_samples
                                     if min_calibration_samples is not None
                                     else DEFAULT_MIN_CALIBRATION_SAMPLES))

    def episode_closeout_record(self, task_id: str,
                                episode_id: Optional[str] = None
                                ) -> Optional[Dict[str, Any]]:
        """The stored close-out of one episode, or None while it is open.

        Read-only: no model call, no re-evaluation, no re-billing."""
        from or_harness.world_model.episode_closeout import (
            episode_closeout_record,
        )
        record = episode_closeout_record(self, task_id, episode_id)
        return record.to_dict() if record is not None else None

    def get_strategy_evaluation(self, evaluation_id: str
                                ) -> Optional[Dict[str, Any]]:
        """Read one stored post-hoc prediction evaluation (read-only)."""
        from or_harness.world_model.episode_closeout import get_evaluation
        evaluation = get_evaluation(self.store, evaluation_id)
        return evaluation.to_dict() if evaluation is not None else None

    def strategy_prediction_evaluations(
            self, *, task_id: Optional[str] = None,
            episode_id: Optional[str] = None
            ) -> List[Dict[str, Any]]:
        """Stored post-hoc evaluations of strategy-outcome predictions.

        Only CLOSED episodes' evaluations are admissible calibration
        samples, but this query lists what exists (an open episode's
        pending evaluations are visible as pending, never hidden)."""
        from or_harness.world_model.episode_closeout import (
            StrategyPredictionEvaluation,
        )
        rows = self.store.conn.execute(
            "SELECT key, value FROM meta WHERE key LIKE "
            "'strategy_evaluation|%' ORDER BY key").fetchall()
        out: List[Dict[str, Any]] = []
        for row in rows:
            try:
                evaluation = StrategyPredictionEvaluation.from_dict(
                    self.store.loads(row["value"]))
            except Exception:
                continue
            if task_id is not None and evaluation.task_id != task_id:
                continue
            if episode_id is not None \
                    and evaluation.episode_id != episode_id:
                continue
            out.append(evaluation.to_dict())
        return out

    def calibration_summary(self, *, min_samples: Optional[int] = None
                            ) -> Dict[str, Any]:
        """The published experience-calibration summary (wm-so/1).

        Aggregated from CLOSED episodes' eligible evaluations only — an
        active episode never reads its own not-yet-closed feedback. This
        is the OR strategy-outcome reliability, kept separate from the
        legacy knowledge-prediction reliability; it is a measured record,
        not a fitted calibrator and not a promise of future accuracy."""
        from or_harness.world_model.episode_closeout import (
            DEFAULT_MIN_CALIBRATION_SAMPLES,
            build_calibration_summary,
        )
        return build_calibration_summary(
            self, min_samples=(min_samples
                               if min_samples is not None
                               else DEFAULT_MIN_CALIBRATION_SAMPLES))

    def bind_strategy_outcome(self, prediction_id: str,
                              action_id: str) -> "StrategyOutcomePrediction":
        """Bind a strategy-outcome prediction to the REAL action that ran.

        The binding checks request identity — action type, task, episode,
        strategy, solver and the candidate's execution config (a different
        time_limit/seed is a different candidate, and its result is not
        this prediction's truth) — and records a mismatch rather than
        silently comparing.

        **Unknown is not a match.** A field the executed action could not
        observe (no episode recorded, a config key the action log never
        carries) is recorded under ``binding_unknown`` — separately from a
        KNOWN disagreement (``binding_mismatch``). Neither may enter an
        evaluation that needs that identity evidence: a mismatch makes the
        prediction not comparable, and an unknown makes the fields that
        depend on it unevaluable (the close-out reports which).

        The prediction's ``trace.comparable`` is set only when the bound
        action's window is a completed real scope AND no mismatch exists.
        Re-binding the same action is idempotent AND re-evaluates the
        comparability (an action that was still running at the first bind
        may have completed since — a pending binding must not stay pending
        forever); binding another action raises. No model call is made and
        nothing is re-billed.
        """
        prediction = self.strategy_predictions.get(prediction_id)
        if prediction is None:
            raise StorageError(
                f"unknown prediction_id {prediction_id!r}")
        action = self.actions.get(action_id)
        if action is None:
            raise StorageError(f"unknown action_id {action_id!r}")
        model_info = prediction.trace.model_info
        bound = model_info.get("bound_action_id")
        if bound is not None and bound != action_id:
            raise StorageError(
                f"prediction {prediction_id!r} is already bound to action "
                f"{bound!r}")
        candidate = prediction.candidate
        mismatch: Dict[str, Any] = {}
        unknown: Dict[str, Any] = {}
        # Action type: the prediction is about executing a strategy; a
        # bound action of another type is a different thing entirely.
        if action.action_type != candidate.action_type:
            mismatch["action_type"] = {"predicted": candidate.action_type,
                                       "actual": action.action_type}
        if candidate.task_id and action.task_id != candidate.task_id:
            mismatch["task_id"] = {"predicted": candidate.task_id,
                                   "actual": action.task_id}
        # Episode: three states. Both known and equal -> match; both known
        # and different -> mismatch; the action's episode unknown -> an
        # UNKNOWN (the prediction declared one, the record cannot confirm
        # it — episode ownership is never guessed).
        if candidate.episode_id is not None:
            if action.episode_id is None:
                unknown["episode_id"] = {
                    "predicted": candidate.episode_id,
                    "actual": None,
                    "reason": ("the executed action records no episode id, "
                               "so the prediction's episode claim cannot be "
                               "confirmed or refuted"),
                }
            elif action.episode_id != candidate.episode_id:
                mismatch["episode_id"] = {
                    "predicted": candidate.episode_id,
                    "actual": action.episode_id}
        params = dict(action.params or {})
        if (candidate.strategy_id is not None
                and params.get("strategy_id") not in
                (None, candidate.strategy_id)):
            mismatch["strategy_id"] = {"predicted": candidate.strategy_id,
                                       "actual": params.get("strategy_id")}
        if (candidate.solver is not None
                and params.get("solver") not in
                (None, candidate.solver)):
            mismatch["solver"] = {"predicted": candidate.solver,
                                  "actual": params.get("solver")}
        # Execution-config identity: the candidate's own config (time_limit,
        # mip_gap, seed, ...) is what was predicted. A key the action
        # actually recorded with a DIFFERENT value is a mismatch; a key the
        # action log never carries is an UNKNOWN — the predicted config is
        # never copied onto the action to manufacture a match.
        for key, value in (candidate.config or {}).items():
            actual = params.get(key)
            if actual is None:
                unknown.setdefault("config", {})[key] = {
                    "predicted": value,
                    "actual": None,
                    "reason": ("the executed action records no value for "
                               "this config key, so the predicted "
                               "configuration cannot be confirmed for it"),
                }
            elif actual != value:
                mismatch.setdefault(
                    "config", {})[key] = {"predicted": value,
                                          "actual": actual}
        # Timing: the prediction must predate the action.
        if prediction.trace.created_at > action.started_at:
            mismatch["timing"] = {
                "predicted_at": prediction.trace.created_at,
                "action_started_at": action.started_at,
                "reason": "prediction was generated after the action began"}
        # Effective problem version: the prediction's frozen input digest
        # against the action's own pre-snapshot digest. Both sides come
        # from frozen records; an unestablishable side is an unknown.
        predicted_input = model_info.get("effective_input_digest")
        actual_digest = None
        if action.pre_snapshot_id:
            pre_snap = self.get_snapshot(action.pre_snapshot_id)
            if pre_snap is not None:
                actual_digest = (pre_snap.problem_state or {}).get(
                    "task_digest")
        if predicted_input and actual_digest:
            if predicted_input != actual_digest:
                mismatch["effective_input"] = {
                    "predicted": predicted_input,
                    "actual": actual_digest,
                    "reason": ("the task's content changed between the "
                               "prediction and the execution: this is a "
                               "different problem input"),
                }
        elif predicted_input and not actual_digest:
            unknown["effective_input"] = {
                "predicted": predicted_input,
                "actual": None,
                "reason": ("the executed action's pre snapshot records no "
                           "task version, so the problem input the "
                           "execution ran under cannot be established"),
            }
        model_info["bound_action_id"] = action_id
        # Comparability: only a completed, MATCHING real scope may be
        # scored. An unknown identity field does not by itself block the
        # linkage (the close-out decides field-by-field what may be
        # evaluated on it), but a mismatch always does.
        if candidate.scope == "strategy_window":
            # The DECLARED window identity (including its selection round,
            # when the window_id carries one) decides which window this
            # prediction is about: a round-1 prediction is scored against
            # round 1's window, never the whole-episode aggregation.
            declared_round = None
            if candidate.window_id:
                parsed = parse_window_id(candidate.window_id)
                if parsed is not None:
                    declared_round = parsed.round_index
            window = self.strategy_execution_window(
                action.task_id, action.episode_id,
                strategy_id=candidate.strategy_id,
                round_index=declared_round)
            # The bound action must be INSIDE the declared round's window:
            # an action of another round is a different scope, never this
            # prediction's truth.
            if declared_round is not None \
                    and action.action_id not in \
                    {a.action_id for a in window.attempts}:
                mismatch["window_round"] = {
                    "predicted": candidate.window_id,
                    "actual": (f"the bound action is not among round "
                               f"{declared_round}'s window attempts"),
                }
            model_info["binding_mismatch"] = mismatch or None
            model_info["binding_unknown"] = unknown or None
            model_info["bound_window_id"] = window.window_id
            model_info["bound_window_round_index"] = window.round_index
            model_info["window_comparable"] = bool(window.comparable)
            model_info["window_not_comparable_reasons"] = list(
                window.not_comparable_reasons)
            prediction.trace.comparable = bool(
                window.comparable and not mismatch)
            prediction.trace.not_comparable_reasons = list(
                window.not_comparable_reasons) if not window.comparable else (
                [] if not mismatch else
                ["binding mismatch: the executed action differs from the "
                 "predicted candidate"])
        else:
            model_info["binding_mismatch"] = mismatch or None
            model_info["binding_unknown"] = unknown or None
            comparable = bool(action.linked_execution_id
                              and action.status != "running"
                              and not mismatch)
            prediction.trace.comparable = comparable
            prediction.trace.not_comparable_reasons = (
                [] if comparable else
                (["binding mismatch: the executed action differs from the "
                  "predicted candidate"] if mismatch else
                 ["the bound action has no completed linked execution to "
                  "score against"]))
        model_info["binding_recorded_at"] = time.time()
        self.strategy_predictions._save(prediction)
        return prediction

    # -- world-model M5: capability evolution prediction service (wm-ce/1) --

    def predict_capability_evolution(
            self,
            operation: Union[LearningOperation, Dict[str, Any]],
            *,
            task: Optional[Dict[str, Any]] = None,
            bundle: Optional[Dict[str, Any]] = None,
            experience_scope: Optional[ExperienceScope] = None,
            task_targeting: Optional[TaskTargeting] = None,
            baseline: Optional[BaselineStatement] = None,
            baselines_by_metric: Optional[Dict[str, BaselineStatement]] =
            None,
            horizon: str = "",
            horizon_tasks: Optional[int] = None,
            maintenance_budget: Optional[Dict[str, float]] = None,
            timeout_s: Optional[float] = None,
            task_id: str = "",
            episode_id: Optional[str] = None,
            snapshot: Optional[BeliefSnapshot] = None,
            ) -> "CapabilityEvolutionPrediction":
        """Predict what ONE offline learning operation would change (M5).

        The input is FROZEN capability evidence, the candidate operation,
        the exact experience scope it would consume, the task types the
        claim is about, a baseline and a horizon. The output is a
        :class:`~or_harness.world_model.contracts
        .CapabilityEvolutionPrediction` under the ``wm-ce/1`` protocol:
        expected future-performance changes, the predicted LEARNING cost,
        degradation risk, uncertainty and what would verify it.

        Three things stay separate and this method only produces the first:
        a prediction is MADE here; the real maintenance FACT is bound later
        (``bind_capability_maintenance``); the capability EFFECT is judged
        after qualified later tasks (``evaluate_capability_effect``). No
        knowledge entry, no effect verdict and no capability increment is
        created here.

        The operation, the scope, the target, the baseline and the horizon
        are fixed by the FRAMEWORK (the model may not restate them). A
        failed call is a persisted failure with whatever usage it consumed.
        """
        from or_harness.world_model.capability_evolution import (
            learning_material_for_bundle,
        )
        if isinstance(operation, dict):
            operation = LearningOperation.from_dict(operation)
        if experience_scope is None and bundle is not None:
            experience_scope = self._scope_from_bundle(bundle)
        if task_targeting is None and bundle is not None:
            task_targeting = self._targeting_from_bundle(bundle)
        if baseline is None and bundle is not None:
            baseline = self._baseline_from_bundle(bundle)
        # The per-metric references the framework FREEZES: a change is only
        # measurable against a yardstick taken on ITS OWN metric, so the
        # bundle's frozen statistics are published keyed by metric. The
        # model may cite one of these; it may not set its own.
        baselines_by_metric = dict(baselines_by_metric or {})
        if not baselines_by_metric:
            if bundle is not None:
                baselines_by_metric = self._baselines_from_bundle(bundle)
            elif baseline is not None:
                metric = baseline.metric or "normalized_solution_quality"
                baselines_by_metric = {metric: baseline}
        if baseline is None and baselines_by_metric:
            baseline = next(iter(baselines_by_metric.values()))
        if not horizon:
            horizon = ("the next matching tasks after the operation, over "
                       "the declared evaluation horizon")
        # The evidence a capability prediction conditions on INCLUDES the
        # VERIFIED effects of earlier offline operations. Using the bare
        # evidence here would leave the loop open: an operation whose effect
        # was really confirmed on later tasks would never reach the next
        # prediction, so the framework would keep re-deciding from the same
        # starting point. Unverified, pending and refuted evaluations change
        # nothing — only a real later-task observation does.
        evidence = self.capability_evidence_with_effects(
            task=task, snapshot=snapshot)
        # The evidence VERSION is what makes two predictions' shared input
        # checkable: it is a digest of the evidence CONTENT (not a
        # timestamp), so two predictions made on the same evidence really
        # do share an input.
        if not evidence.version:
            evidence.version = self._evidence_version(evidence)
        material = (learning_material_for_bundle(self, bundle)
                    if bundle is not None else None)
        service = self.capability_predictions
        prediction = service.predict(
            evidence, operation, experience_scope=experience_scope,
            task_targeting=task_targeting, baseline=baseline,
            baselines_by_metric=baselines_by_metric,
            horizon=horizon, horizon_tasks=horizon_tasks,
            learning_material=material,
            maintenance_budget=maintenance_budget, timeout_s=timeout_s,
            task_id=task_id or (str(task.get("task_id", "")) if task
                                else ""),
            episode_id=episode_id)
        prediction.trace.model_info["capability_evidence_version"] = \
            evidence.version
        service._save(prediction, task_id=task_id or (
            str(task.get("task_id", "")) if task else ""),
            episode_id=episode_id)
        return prediction

    @staticmethod
    def _evidence_version(evidence: HarnessCapabilityEvidence) -> str:
        """A content digest of one capability evidence object.

        Two predictions made against the SAME evidence content share this
        version, which is what lets a comparison tell a shared frozen input
        from two unrelated ones. Read timestamps are excluded, so merely
        reading the banks again does not fabricate a new version.
        """
        import hashlib
        content = evidence.to_dict()
        content.pop("as_of", None)
        for item in (content.get("sources") or {}).values():
            item.pop("notes", None)
        blob = __import__("json").dumps(
            content, sort_keys=True, separators=(",", ":"), default=str)
        return "cev_" + hashlib.sha256(
            blob.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _scope_from_bundle(bundle: Dict[str, Any]) -> ExperienceScope:
        """The exact experience scope a candidate bundle rests on."""
        return ExperienceScope(
            execution_ids=[str(e) for e in
                           (bundle.get("execution_ids") or [])],
            task_ids=[str(t) for t in
                      (bundle.get("tasks")
                       or bundle.get("task_ids") or [])],
            family=(str(bundle["family"]) if bundle.get("family") else None),
            cell_token=(str(bundle["cell_token"])
                        if bundle.get("cell_token") else None),
            note="the candidate bundle's own evidence scope")

    @staticmethod
    def _targeting_from_bundle(bundle: Dict[str, Any]
                               ) -> TaskTargeting:
        """The task types a candidate bundle's claim speaks about."""
        return TaskTargeting(
            description=(f"tasks of family {bundle.get('family')!r} "
                         f"matching the cell of strategy "
                         f"{bundle.get('strategy_id')!r}"),
            family=(str(bundle["family"]) if bundle.get("family") else None),
            cell_token=(str(bundle["cell_token"])
                        if bundle.get("cell_token") else None),
        )

    @staticmethod
    def _baseline_from_bundle(bundle: Dict[str, Any]
                              ) -> BaselineStatement:
        """The frozen baseline a candidate bundle is measured against.

        The bundle's OWN frozen statistics are the reference: a prediction
        about improving on them is falsifiable; a prediction with no
        reference is not.
        """
        value = bundle.get("mean_quality")
        return BaselineStatement(
            kind="conditional_stats",
            value=(float(value) if value is not None else None),
            ref=EvidenceRef(ref_type="candidate_bundle",
                            ref_id=str(bundle.get("bundle_id", ""))),
            metric="normalized_solution_quality",
            note=("the bundle's frozen mean quality over its supporting "
                  "executions: the reference the predicted change is "
                  "measured against, frozen before the operation runs"),
        )

    @classmethod
    def _baselines_from_bundle(cls, bundle: Dict[str, Any]
                               ) -> Dict[str, BaselineStatement]:
        """The per-metric frozen references a candidate bundle supplies.

        A reference is only meaningful for the metric it was taken on: the
        bundle's mean quality says nothing about solver seconds, and a
        per-dimension cost means nothing for a quality change. Keying them
        by metric is what stops one metric's yardstick from being silently
        applied to another.

        Cost references are keyed by the SPECIFIC DIMENSION
        (``cost:solver_runtime_s``, ``cost:llm_tokens``), not by the generic
        ``resource_cost`` metric: several cost dimensions share that metric
        name, and collapsing them onto one key would let whichever
        dimension happened to be written last become the reference for all
        of them — a 20% runtime saving would then be computed against a
        token count.
        """
        out: Dict[str, BaselineStatement] = {}
        quality = cls._baseline_from_bundle(bundle)
        if quality.value is not None:
            out["normalized_solution_quality"] = quality
        mean_cost = bundle.get("mean_cost") or {}
        for dim, value in mean_cost.items():
            if value is None:
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            unit = cls._unit_for_cost_dim(dim)
            out[cls.baseline_key_for_cost_dim(dim)] = BaselineStatement(
                kind="conditional_stats",
                value=numeric,
                ref=EvidenceRef(ref_type="candidate_bundle",
                                ref_id=str(bundle.get("bundle_id", ""))),
                metric="resource_cost",
                unit=unit,
                note=(f"the bundle's frozen mean {dim} over its supporting "
                      "executions: frozen before the operation runs"))
        return out

    @staticmethod
    def baseline_key_for_cost_dim(dim: str) -> str:
        """The key a cost dimension's frozen reference is stored under.

        A dimension-qualified key, because the generic ``resource_cost``
        metric covers several dimensions whose units do not convert.
        """
        return f"cost:{dim}"

    @staticmethod
    def _unit_for_cost_dim(dim: str) -> str:
        """The unit a cost dimension's reference is expressed in."""
        from or_harness.world_model.maintenance_decision import (
            COST_DIMENSION_UNITS,
        )
        return COST_DIMENSION_UNITS.get(str(dim), str(dim))

    def get_capability_evolution_prediction(
            self, prediction_id: str
    ) -> Optional["CapabilityEvolutionPrediction"]:
        """Read a stored capability-evolution prediction (no model call)."""
        return self.capability_predictions.get(prediction_id)

    def capability_evolution_predictions(
            self, *, task_id: Optional[str] = None,
            episode_id: Optional[str] = None
    ) -> List["CapabilityEvolutionPrediction"]:
        """Stored capability-evolution predictions, optionally filtered."""
        return self.capability_predictions.query(task_id=task_id,
                                                episode_id=episode_id)

    def compare_capability_evolution(
            self, prediction_ids: Sequence[str], *,
            horizon_tasks: Optional[int] = None,
            require_quality_nondegradation: bool = True,
            persist: bool = True,
            ) -> Dict[str, Any]:
        """Compare frozen capability predictions and recommend one, or
        defer (M5).

        Read-only with respect to knowledge: the Strategic Bank is never
        touched, no operation runs, no model is called. The bounded rule is
        stated in the result — among candidates predicting a QUANTIFIED
        per-task resource saving with no predicted quality degradation, the
        largest saving wins; everything else is reported as incomparable
        for the agent to choose between. ``defer`` is a legitimate result.

        The real spend of the predictions being compared is reported
        separately from their predicted learning cost — a comparison is not
        an operation and its cost is not the operation's cost.
        """
        from or_harness.world_model.maintenance_decision import (
            compare_capability_predictions,
        )
        predictions = []
        missing = []
        for prediction_id in prediction_ids:
            prediction = self.capability_predictions.get(prediction_id)
            if prediction is None:
                missing.append(str(prediction_id))
            else:
                predictions.append(prediction)
        if missing:
            raise StorageError(
                f"unknown capability prediction(s) {missing!r}")
        recommendation = compare_capability_predictions(
            predictions, horizon_tasks=horizon_tasks,
            require_quality_nondegradation=require_quality_nondegradation)
        # The FROZEN input is the same for every candidate: the evidence
        # version and the horizon. Sharing it is what makes the comparison
        # a comparison rather than a ranking of unrelated numbers.
        versions = {p.current_evidence.version for p in predictions}
        horizons = {p.horizon for p in predictions}
        recommendation.shared_input = {
            "n_predictions": len(predictions),
            "capability_evidence_versions": sorted(
                v for v in versions if v is not None),
            "shared_evidence": len(versions) <= 1,
            "horizons": sorted(h for h in horizons if h),
            "shared_horizon": len(horizons) <= 1,
            "note": ("candidates compared under one shared frozen input: a "
                     "different evidence version or horizon means the "
                     "numbers are not strictly comparable and is reported "
                     "rather than hidden"),
        }
        if len(versions) > 1:
            recommendation.notes.append(
                "the compared predictions were made against DIFFERENT "
                "capability evidence versions: the comparison is reported "
                "with that caveat")
        if len(horizons) > 1:
            recommendation.notes.append(
                "the compared predictions declare DIFFERENT horizons: their "
                "per-task figures are comparable, their totals are not")
        call_cost = self._capability_call_costs(predictions)
        if call_cost:
            recommendation.comparison_cost = {
                "per_dim": call_cost,
                "measured": sorted(call_cost),
                "note": ("the REAL spend of the prediction calls being "
                         "compared, kept separate from the predicted "
                         "learning cost and from any execution cost"),
            }
        if persist:
            self._persist_recommendation(recommendation)
        return recommendation.to_dict()

    def _capability_call_costs(
            self, predictions: Sequence[Any]) -> Dict[str, float]:
        """The real per-dimension spend of the given prediction calls."""
        totals: Dict[str, float] = {}
        for prediction in predictions:
            cost = prediction.trace.call_cost
            if cost is None:
                continue
            for dim in cost.measured_dims():
                totals[dim] = totals.get(dim, 0.0) + float(
                    getattr(cost, dim))
        return {d: round(v, 6) for d, v in sorted(totals.items())}

    def _persist_recommendation(self, recommendation) -> None:
        """Store one maintenance recommendation (append-only by id)."""
        with self.store.transaction() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?,?)",
                (f"maintenance_recommendation|"
                 f"{recommendation.recommendation_id}",
                 self.store.dumps(recommendation.to_dict())))

    def accept_capability_operation(
            self, recommendation: Dict[str, Any], *,
            prediction_id: Optional[str] = None,
            verify: Optional[Dict[str, Any]] = None,
            notes: Optional[List[str]] = None,
            force: bool = False,
            ) -> Dict[str, Any]:
        """EXPLICITLY accept a recommendation and run the real operation.

        This is the only M5 entry that changes knowledge: it records the
        adoption decision (so the binding can find which operation followed
        which prediction) and invokes the EXISTING ``induce`` machinery on
        the prediction's OWN experience scope — the scope is never widened
        by re-reading the current bank.

        A recommendation that is not ``accept`` cannot be accepted: a
        ``defer`` has no operation behind it.
        """
        if recommendation.get("recommendation") != "accept":
            raise ValueError(
                f"cannot accept a recommendation with verdict "
                f"{recommendation.get('recommendation')!r}: only 'accept' "
                "names an operation to run (a defer is a legitimate "
                "outcome, not an action)")
        target_id = prediction_id or recommendation.get(
            "selected_prediction_id")
        if not target_id:
            raise ValueError(
                "no prediction was named: pass prediction_id= or use a "
                "recommendation that selected one")
        prediction = self.capability_predictions.get(target_id)
        if prediction is None:
            raise StorageError(
                f"unknown capability prediction {target_id!r}")
        operation_type = prediction.candidate_operation.operation_type
        # The operation TYPE decides what really runs. Only the types this
        # build can carry out through an existing path are dispatched;
        # anything else is REFUSED outright, because silently running
        # ``induce`` for a ``retire`` candidate would execute an operation
        # the prediction was never about — the worst kind of substitution,
        # since it looks like the recommendation was honoured.
        if operation_type not in CAPABILITY_EXECUTABLE_OPERATIONS:
            raise ValueError(
                f"operation_type {operation_type!r} has no execution path in "
                f"this build (supported: "
                f"{sorted(CAPABILITY_EXECUTABLE_OPERATIONS)}). The "
                "recommendation is NOT executed as a different operation: "
                "predicting an operation the framework cannot carry out and "
                "then doing something else would make the binding "
                "meaningless")
        scope = prediction.experience_scope
        execution_ids = list(scope.execution_ids) if scope is not None else []
        if not execution_ids:
            raise ValueError(
                "the prediction records no experience scope: the operation "
                "cannot be restricted to the evidence it was about, and "
                "widening it silently would make the prediction answer a "
                "different question")
        strategy_id = prediction.candidate_operation.strategy_id
        if operation_type == "retire":
            entry_id = self._retire_target_entry(prediction)
            if entry_id is None:
                raise ValueError(
                    "a 'retire' operation needs a target entry: the "
                    "prediction names none, so there is nothing to retire "
                    "and the operation is refused rather than reinterpreted")
        else:
            entry_id = None
        adoption = self.actions.report_action(
            "induce", MAINTENANCE_TASK_ID,
            f"maint_capability_{int(time.time())}",
            params={
                "capability_prediction_id": target_id,
                "recommendation_id": recommendation.get("recommendation_id"),
                "operation_type": operation_type,
                "strategy_id": strategy_id,
                "execution_ids": execution_ids,
                "target_entry_id": entry_id,
                "action": "accepted",
            },
            outcome={"accepted": True, "capability_prediction_id": target_id},
            status="completed")
        try:
            if operation_type == "retire":
                result = self.retire(entry_id, reason=(
                    prediction.candidate_operation.description
                    or "an accepted offline maintenance recommendation"))
                result = {"business_result": "retired",
                          "entry_id": entry_id,
                          "knowledge_delta": {
                              "entry_changes": [],
                              "entries_created": [],
                              "entries_retired": [entry_id],
                          },
                          "execution_ids": execution_ids}
            else:
                result = self.induce(strategy_id=strategy_id, verify=verify,
                                     notes=notes, force=force,
                                     execution_ids=execution_ids)
        except Exception:
            self.actions.end_action(
                adoption.action_id, status="failed",
                outcome={"accepted": True,
                         "capability_prediction_id": target_id,
                         "operation_type": operation_type,
                         "error": "the operation raised"})
            raise
        # The knowledge transition travels in several shapes (the induce
        # result's flat keys, the action summary nested under ``action``, or
        # that action's outcome). ``_induction_transition`` is the ONE
        # locator for it, so the binding reads the same delta the rest of
        # the harness does rather than a key that may not be there. A
        # retirement has its own shape: it removes an entry rather than
        # creating or revising one, and it is recorded as such.
        if operation_type == "retire":
            knowledge_delta = {
                "entry_changes": [],
                "entries_created": [],
                "entries_retired": [entry_id],
            }
            knowledge_after: List[Any] = []
        else:
            transition = self._induction_transition(result)
            knowledge_delta = {
                "entry_changes": transition["entry_changes"],
                "entries_created": transition["created"],
            }
            knowledge_after = transition["knowledge_after"]
        adoption.outcome = {
            "accepted": True,
            "capability_prediction_id": target_id,
            "operation_type": operation_type,
            "operation_action_id": (result.get("action") or {}).get(
                "action_id"),
            "operation_result": {
                "business_result": result.get("business_result"),
                "knowledge_delta": knowledge_delta,
                "knowledge_after": knowledge_after,
                "execution_ids": execution_ids,
                "verification": result.get("verification"),
            },
        }
        self.actions._update(adoption)
        return {
            "adoption_action_id": adoption.action_id,
            "capability_prediction_id": target_id,
            "accepted": True,
            "operation_type": operation_type,
            "execution_ids": execution_ids,
            "operation_result": result,
            "note": ("the operation ran on the prediction's OWN scope; bind "
                     "the real fact with bind_capability_maintenance, and "
                     "remember the capability effect is still unverified "
                     "until qualified later tasks produce real results"),
        }

    def _retire_target_entry(self, prediction: Any) -> Optional[str]:
        """The entry a ``retire`` prediction is about, when it names one.

        The target comes from the operation's own declaration (its
        ``config``/``scope``) — never from re-reading the current bank for
        a plausible victim. An unnamed target means the operation cannot be
        carried out, which is reported rather than guessed.
        """
        operation = prediction.candidate_operation
        config = operation.config or {}
        for key in ("target_entry_id", "entry_id"):
            value = config.get(key)
            if value:
                return str(value)
        scope = operation.scope or prediction.experience_scope
        if scope is not None and scope.note:
            marker = "entry:"
            if marker in scope.note:
                return scope.note.split(marker, 1)[1].strip() or None
        return None

    def reject_capability_operation(
            self, recommendation: Dict[str, Any], *,
            reason: Optional[str] = None,
            prediction_id: Optional[str] = None,
            ) -> Dict[str, Any]:
        """Explicitly decline or defer: NO knowledge is modified.

        A defer/reject is recorded as a maintenance decision with its
        reason, so the decision history is auditable — and the Strategic
        Bank is untouched.
        """
        target_id = prediction_id or recommendation.get(
            "selected_prediction_id")
        record = self.actions.report_action(
            "induce", MAINTENANCE_TASK_ID,
            f"maint_capability_reject_{int(time.time())}",
            params={
                "capability_prediction_id": target_id,
                "recommendation_id": recommendation.get("recommendation_id"),
                "verdict": recommendation.get("recommendation"),
                "action": "rejected",
                "reason": reason,
            },
            outcome={"accepted": False, "rejected": True,
                     "capability_prediction_id": target_id,
                     "reason": reason},
            status="completed")
        return {
            "rejection_action_id": record.action_id,
            "capability_prediction_id": target_id,
            "accepted": False,
            "strategic_bank_touched": False,
            "note": ("the recommendation was declined: no operation ran and "
                     "no knowledge changed. A declined recommendation is "
                     "not a wrong prediction — it was never given a chance "
                     "to come true"),
        }

    def bind_capability_maintenance(self, prediction_id: str, *,
                                    adoption_action_id: Optional[str] = None
                                    ) -> Dict[str, Any]:
        """Stage 1: bind the REAL maintenance fact to a prediction (M5).

        Records that the operation happened (or did not), what knowledge
        actually changed, the REAL cost, the verification outcome, and
        whether the scope used matches the scope predicted. Idempotent by
        prediction: a second bind returns the stored fact and counts
        nothing twice.

        It CANNOT set ``effect_verified``: entries being created and even
        verified is a knowledge-change fact, not evidence that future
        performance improved.
        """
        from or_harness.world_model.maintenance_decision import (
            bind_maintenance_fact,
            get_maintenance_binding,
            record_maintenance_binding,
        )
        stored = get_maintenance_binding(self, prediction_id)
        if stored is not None:
            return {
                "binding": stored.to_dict(),
                "already_bound": True,
                "note": ("already bound: the stored fact stands, nothing was "
                         "re-counted or re-billed"),
            }
        binding = bind_maintenance_fact(
            self, prediction_id, adoption_action_id=adoption_action_id)
        if binding.adoption_action_id is None:
            return {
                "binding": binding.to_dict(),
                "already_bound": False,
                "state": "not_adopted",
                "note": ("no operation was accepted for this prediction: "
                         "there is no maintenance fact to bind. A deferred "
                         "or declined recommendation leaves the knowledge "
                         "alone, and nothing is recorded as if it ran"),
            }
        record_maintenance_binding(self, binding)
        return {
            "binding": binding.to_dict(),
            "already_bound": False,
            "state": "bound",
        }

    def capability_maintenance_binding(
            self, prediction_id: str) -> Optional[Dict[str, Any]]:
        """The stored stage-1 maintenance fact, or None (read-only)."""
        from or_harness.world_model.maintenance_decision import (
            get_maintenance_binding,
        )
        binding = get_maintenance_binding(self, prediction_id)
        return binding.to_dict() if binding is not None else None

    def record_capability_paired_evaluation(
            self, prediction_id: str, *, metric: str,
            reference_value: float, treated_value: float,
            unit: str = "", source: str = "external_paired_evaluation",
            reference_task_ids: Optional[Sequence[str]] = None,
            note: str = "") -> Dict[str, Any]:
        """Record a pre-arranged PAIRED comparison (the causal reference).

        The framework does not clone the harness or replay counterfactual
        operations. It accepts a comparison the caller really arranged; the
        record names its own source and covered tasks so a later reader can
        tell a real comparison from an assertion.
        """
        from or_harness.world_model.maintenance_decision import (
            record_paired_evaluation,
        )
        if self.capability_predictions.get(prediction_id) is None:
            raise StorageError(
                f"unknown capability prediction {prediction_id!r}")
        return record_paired_evaluation(
            self, prediction_id, metric=metric,
            reference_value=reference_value, treated_value=treated_value,
            unit=unit, source=source,
            reference_task_ids=reference_task_ids, note=note)

    def evaluate_capability_effect(
            self, prediction_id: str, *,
            task_ids: Optional[Sequence[str]] = None,
            require_paired_reference: bool = True,
            persist: bool = True) -> Dict[str, Any]:
        """Stage 2: judge one prediction against REAL later-task results.

        Reads the M4 evaluations of CLOSED episodes of tasks that are NOT
        part of the prediction's own experience scope (reusing the
        induction tasks checks consistency, never transfer), or a
        pre-arranged paired comparison. An unreached horizon stays
        ``pending`` and re-evaluable; a change with no comparable reference
        is ``inconclusive`` rather than attributed to the operation.

        Only an ``observed_improvement`` under a comparable setup sets
        ``effect_verified``. Repeating the call REPLACES the stored
        evaluation instead of adding a second sample, so a repeated look
        never inflates the evidence. Only a FINAL verdict (an observed
        improvement/degradation, or a provably no-change operation) short-
        circuits the call; a pending horizon, a descriptive movement with
        no reference, or insufficient evidence stays re-evaluable, so a
        later paired reference or a newly closed episode can still turn it
        into a verdict.
        """
        from or_harness.world_model.maintenance_decision import (
            EFFECT_FINAL_STATES,
            evaluate_capability_effect,
            get_effect_evaluation,
            record_effect_evaluation,
        )
        stored = get_effect_evaluation(self, prediction_id)
        if stored is not None and stored.state in EFFECT_FINAL_STATES:
            return {
                "evaluation": stored.to_dict(),
                "already_evaluated": True,
                "note": ("already evaluated: the stored verdict stands and "
                         "the sample was not counted again"),
            }
        evaluation = evaluate_capability_effect(
            self, prediction_id, task_ids=task_ids,
            require_paired_reference=require_paired_reference)
        if persist:
            record_effect_evaluation(self, evaluation)
        return {
            "evaluation": evaluation.to_dict(),
            "already_evaluated": False,
        }

    def capability_effect_evaluation(
            self, prediction_id: str) -> Optional[Dict[str, Any]]:
        """The stored stage-2 effect evaluation, or None (read-only)."""
        from or_harness.world_model.maintenance_decision import (
            get_effect_evaluation,
        )
        evaluation = get_effect_evaluation(self, prediction_id)
        return evaluation.to_dict() if evaluation is not None else None

    def capability_feedback_summary(self) -> Dict[str, Any]:
        """Every capability prediction's two-stage feedback state (M5).

        Read-only: no model call, no re-evaluation, no re-billing. Shows
        the difference between a BOUND FACT (the operation happened) and a
        VERIFIED EFFECT (real later performance moved).
        """
        from or_harness.world_model.maintenance_decision import (
            capability_effect_summary,
        )
        return capability_effect_summary(self)

    def capability_evidence_with_effects(self, **kwargs
                                         ) -> HarnessCapabilityEvidence:
        """Capability evidence that includes VERIFIED effect results (M5).

        Only ``observed_improvement`` evaluations contribute
        ``direct_evidence``, and only for the sources they really observed.
        W_OR is never advanced by a capability prediction's own report —
        its evidence must come from independent OR strategy-outcome
        prediction-error records.
        """
        from or_harness.world_model.maintenance_decision import (
            capability_evidence_from_effects,
        )
        return capability_evidence_from_effects(self, **kwargs)

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
                        *, parent_action_id: Optional[str] = None,
                        context: Union[PredictionContext, bool, None] = None,
                        cir: Optional[Any] = None,
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
        leaks this call's cost into every episode's budget.

        ``context`` (phase 2) decides what the provider is GIVEN:

        - ``None`` (default): one prediction input context is built for this
          call (task representation + X/B + retrieval evidence + capability
          evidence + execution constraints) and passed to the provider;
        - a :class:`PredictionContext`: that FROZEN context is reused after
          its identity is verified against this task/version/episode — the
          way several candidates of one decision share one input, and the
          way a stored context can be replayed without reading today's banks;
        - ``False``: no context is assembled and the request is exactly what
          this call sent before this phase existed (the compatibility
          escape hatch, and what ``x-b-only`` byte-compatibility rests on).

        ``cir`` (optional) resolves the effective problem input the same way
        ``build_prediction_context(..., cir=...)`` does. It is what lets a
        context built with an EXPLICIT CIR be reused: pass the same CIR
        again here, and the identity check compares the effective inputs
        (which match) instead of the plain task digests (which differ only
        because the caller-supplied task JSON differs).
        """
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
        frozen_from_context = False
        if context is False:
            resolved_context: Optional[PredictionContext] = None
            snap = self.snapshot(task, episode_id)
        elif isinstance(context, PredictionContext):
            problems = context_identity_problems(
                context, task_id=task_id,
                task_digest=task_text_digest(task),
                episode_id=episode_id,
                effective_input_digest=effective_input_version(task, cir))
            # Structure too: a context built under another structural input
            # may not be reused for this one — its joint representation and
            # its retrieval would describe a different problem.
            problems = problems + structure_problems(
                expected=self.profile(task_with_effective_cir(task, cir)),
                actual=context.joint.profile,
                label="the supplied prediction context")
            if problems:
                raise ValueError(
                    "the supplied prediction context does not describe this "
                    "prediction: " + "; ".join(problems))
            resolved_context = context
            # ALL prediction conditions come from the FROZEN context: the
            # snapshot is rebuilt from the context's own condition blocks
            # (never a fresh one, which would carry progress that happened
            # AFTER the context was built), and the recorded
            # input_snapshot_id is the context's snapshot.
            snap = snapshot_from_context(context)
            frozen_from_context = True
        else:
            snap = self.snapshot(task, episode_id)
            # A CURRENT gathering around the snapshot this call just froze:
            # the retrieval, knowledge view, reliability and cell evidence
            # are read NOW and frozen into the context. Passing the snapshot
            # explicitly is what keeps the context's X/B and the prediction's
            # input_snapshot_id identical.
            resolved_context = self.build_prediction_context(
                task, episode_id, snapshot=snap, historical=False,
                context_spec=action_spec, cir=cir)
        if frozen_from_context:
            targets = knowledge_targets_from_context(resolved_context,
                                                     action_spec)
            reliability = copy.deepcopy(resolved_context.reliability or {})
        else:
            targets = self.knowledge_targets(snap, action_spec)
            reliability = self.prediction_reliability_table()
        # The pre-call budget check reads the CURRENT ledger on purpose: the
        # budget is a real external constraint on whether a call may be made,
        # not a prediction condition. It is recorded as an override so the
        # frozen constraint the context carried is never silently replaced.
        budget_override = self._budget_override_view(
            task_id, episode_id, resolved_context)
        prediction = self.predictions.predict_outcome(
            task, action_spec, snap,
            knowledge_targets=targets,
            reliability=reliability,
            prediction_context=resolved_context)
        # The full target set is frozen WITH the prediction: a later verdict
        # needs the claim's interval / entry id / strategy as they were at
        # prediction time, not a re-read of banks that may have moved since.
        prediction.model_info["knowledge_targets_proposed"] = [
            t.strategy_id for t in targets]
        prediction.model_info["knowledge_targets_proposed_full"] = [
            t.to_dict() for t in targets]
        if budget_override:
            prediction.model_info["budget_checked_at_call_time"] = \
                budget_override
        if resolved_context is not None:
            # Which frozen input this prediction was conditioned on. The
            # reference is recorded on the prediction, so a later reader can
            # resolve the CONTENT that was actually used.
            prediction.model_info["prediction_context_id"] = \
                resolved_context.context_id
            prediction.model_info["prediction_context_version"] = \
                resolved_context.version
            prediction.model_info["prediction_context_task_digest"] = \
                resolved_context.task_digest
            prediction.model_info["conditions_source"] = (
                "frozen_context" if frozen_from_context
                else "context_built_for_this_call")
        self.predictions._save(prediction)
        if adjusted:
            prediction.model_info["identity_adjusted"] = adjusted
            self.predictions._save(prediction)
        self._charge_call_cost(prediction, parent_action_id)
        return prediction

    def _budget_override_view(self, task_id: str, episode_id: Optional[str],
                              context: Optional[PredictionContext]
                              ) -> Dict[str, Any]:
        """The CURRENT budget view, when it disagrees with the frozen one.

        A reused context's ``execution_constraints`` recorded the budget
        condition as it stood at build time. The real ledger may have moved
        since, and a pre-call check must use the CURRENT state — but it must
        not silently rewrite what the context froze. So a difference is
        REPORTED as an override, and a matching view reports nothing.
        """
        if context is None:
            return {}
        frozen = (context.execution_constraints or {}).get("declared_budget")
        current = self._load_budget(task_id, episode_id)
        if (frozen or None) == (current or None):
            return {}
        return {
            "frozen_declared_budget": copy.deepcopy(frozen),
            "current_declared_budget": copy.deepcopy(current),
            "note": ("the declared budget moved since this context was "
                     "built; the pre-call check used the CURRENT value while "
                     "the frozen constraint is left untouched — a budget is "
                     "an external limit, not a prediction condition"),
        }

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
        mismatches recorded, not silently compared.

        ``action_id`` accepts EITHER the action id (``ac_...``) OR the
        ``execution_id`` (``ex_...``) that the action produced — the latter
        is what ``orx execute`` prints, and requiring the caller to first
        look up the action id made the normal predict -> execute -> bind
        loop fail whenever the two sides' episode ids did not line up. An
        execution id is resolved through the action's ``linked_execution_id``
        link; an ambiguous or unknown id raises rather than guessing."""
        action = self.actions.get(action_id)
        if action is None:
            action = self.actions.by_execution(action_id)
        if action is None:
            raise StorageError(
                f"unknown action_id/execution_id {action_id!r}: no action "
                "carries that id and no action links to that execution")
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
                  second_step: Optional[Sequence[ActionSpec]] = None,
                  protocol: str = "legacy",
                  ) -> Dict[str, Any]:
        """Bounded next-step planning over predicted action consequences.

        Freezes ONE root snapshot, evaluates <= limits.max_root_candidates
        root candidates (horizon 1 or 2; the second step is predicted FROM
        the hypothetical successor state of the first), and recommends the
        first step of the best path — the agent then explicitly accepts,
        rejects, or overrides it (``choose_next``).

        ``protocol`` selects the prediction protocol:

        - ``"legacy"`` (the default): the existing ``OutcomePrediction``
          path, unchanged — horizon 1 or 2, the old request shape and the
          old scoring. This is the compatibility boundary; nothing about
          it moved.
        - ``"strategy-outcome"``: the M3 protocol. ONE frozen context is
          built for the whole decision, every candidate is predicted under
          the wm-so/1 strategy-outcome protocol (benefit/cost/risk/
          uncertainty), the candidates are compared on a conservative
          yardstick, and the suggestion is the first step of the best
          candidate. Horizon is FIXED at 1 for this protocol (no
          multi-step imagination tree is built for it); passing
          ``horizon=2`` with it is refused rather than silently ignored.

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
        if protocol == "strategy-outcome":
            if limits.horizon != 1:
                raise ValueError(
                    "the strategy-outcome protocol plans at horizon=1 only: "
                    "one macro strategy comparison, then re-planning from "
                    "the real observation. Pass horizon=1 (the default) or "
                    "use the legacy protocol for two-step rollouts")
            return self._plan_next_strategy_outcome(
                task, episode_id, candidates=candidates, limits=limits)
        if protocol not in ("legacy",):
            raise ValueError(
                f"unknown protocol {protocol!r}; expected 'legacy' or "
                "'strategy-outcome'")
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
        # Phase 2: ONE shared prediction input context for the whole
        # decision. Every root candidate is evaluated against the SAME
        # problem representation, X/B, retrieval evidence, capability
        # evidence and execution constraints — only the candidate's own
        # config/scope changes. Building it per candidate would let a bank
        # that moved mid-decision contaminate the comparison, and would
        # re-embed the same query once per candidate.
        plan_context = self.build_prediction_context(
            task, episode_id, snapshot=root, historical=False,
            candidates=specs, top=limits.max_root_candidates)

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
                reliability=reliability_table,
                prediction_context=plan_context)
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
                        task, second_spec, hypo, timeout_s=remaining,
                        prediction_context=plan_context)
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
                     "prediction_context_id": plan_context.context_id,
                     "prediction_context_version": plan_context.version,
                     "hypothetical_snapshots": dict(hypothetical_snaps)})
        result = plan.to_dict()
        # The ONE shared input context of this decision, referenced (not
        # copied) on the plan record: every candidate of the comparison was
        # conditioned on it, and the stored context resolves its content.
        result["prediction_context_id"] = plan_context.context_id
        result["prediction_context_version"] = plan_context.version
        return result

    def _plan_next_strategy_outcome(
            self, task: Dict[str, Any],
            episode_id: Optional[str] = None, *,
            candidates: Optional[Sequence[ActionSpec]] = None,
            limits: Any = None) -> Dict[str, Any]:
        """One macro strategy comparison under the wm-so/1 protocol.

        The M3 decision loop: ONE frozen context for the whole decision,
        one strategy-outcome prediction per candidate (a failed prediction
        does not drag the others down), a conservative comparison, and a
        suggestion the agent accepts, rejects or overrides through
        ``choose_next`` — the SAME decision action and choice recording the
        legacy protocol uses, so the downstream flow is one flow.

        Fallback: when NO candidate carries a usable prediction (provider
        down, every payload invalid), the plan reports
        ``status="no_valid_predictions"`` with the reasons and NO
        suggestion — the caller falls back to ``recall``/Selector or
        chooses itself. The fallback is reported as what it is; it is never
        dressed up as a completed world-model comparison.
        """
        from or_harness.world_model.planner import (
            PlanResult,
            score_strategy_outcome_predictions,
        )
        from or_harness.world_model.strategy_prediction import (
            STRATEGY_OUTCOME_PROTOCOL_VERSION,
        )
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
        root = self.snapshot(task, episode_id)
        plan.root_snapshot_id = root.snapshot_id
        declared = self._load_budget(task_id, episode_id)
        budget_view = self.budget.view(task_id, episode_id, budget=declared)
        plan.budget_confirmation = (
            budget_view["status"] if declared else "unknown")
        if budget_view["status"] == "exceeded":
            plan.status = "fallback"
            plan.truncation_reason = (
                "declared budget already exceeded by real consumption; "
                "planning would spend more — returning without model calls. "
                "Report the current best solution instead.")
            return plan.to_dict()
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
                 if s.action_type == "execute_strategy"]
        if not specs:
            plan.status = "no_candidates"
            reason = ("no execute_strategy candidates to compare under the "
                      "strategy-outcome protocol")
            if plan.truncation_reason:
                reason = plan.truncation_reason + ". " + reason
            plan.truncation_reason = reason
            return plan.to_dict()
        decision = self.actions.begin_action(
            "select_strategy", task_id, episode_id, pre_snapshot=root,
            params={"kind": "plan_next", "protocol": "strategy-outcome",
                    "protocol_version": STRATEGY_OUTCOME_PROTOCOL_VERSION,
                    "limits": limits.to_dict()})
        plan.decision_action_id = decision.action_id
        started = time.monotonic()
        calls_made = 0
        stop_reason: Optional[str] = None
        # ONE frozen context for the whole decision: every candidate is
        # conditioned on the SAME problem representation, X/B, retrieval
        # evidence, capability evidence and constraints.
        plan_context = self.build_prediction_context(
            task, episode_id, snapshot=root, historical=False,
            candidates=specs, top=limits.max_root_candidates)
        predictions: List[Any] = []
        # Per-call cost accounting, IN the loop: each prediction's known
        # spend is charged to the decision action the MOMENT it returns, so
        # the NEXT iteration's budget check sees the real consumption. The
        # old behaviour (one bulk amendment after the loop) let a decision
        # keep calling past an already-exceeded budget and only notice at
        # the end — the ledger must see each call's cost before the next
        # one is allowed.
        planning_total: Dict[str, float] = {}
        planning_measured: set = set()

        def _charge_call(prediction) -> None:
            if prediction.trace.call_cost is None:
                return
            for dim in prediction.trace.call_cost.measured_dims():
                planning_total[dim] = planning_total.get(dim, 0.0) + \
                    getattr(prediction.trace.call_cost, dim)
                planning_measured.add(dim)
            # Increment semantics: each call is a separate real spend.
            self.actions.amend_action_cost_increment(
                decision.action_id,
                **{d: getattr(prediction.trace.call_cost, d)
                   for d in prediction.trace.call_cost.measured_dims()})
            prediction.trace.model_info[
                "charged_to_parent_action"] = decision.action_id
            self.strategy_predictions._save(prediction)

        for spec in specs:
            if calls_made >= limits.max_model_calls:
                stop_reason = (f"model-call budget exhausted "
                               f"({limits.max_model_calls})")
                break
            if time.monotonic() - started > limits.time_budget_s:
                stop_reason = ("planning time budget exhausted "
                               f"({limits.time_budget_s}s)")
                break
            if declared:
                view = self.budget.view(task_id, episode_id, budget=declared)
                if view["status"] == "exceeded":
                    stop_reason = ("declared budget exceeded by real "
                                   "consumption (including this planning's "
                                   "own spend); no further model calls")
                    break
            remaining = limits.time_budget_s - (time.monotonic() - started)
            if remaining <= 0:
                stop_reason = ("planning time budget exhausted "
                               f"({limits.time_budget_s}s)")
                break
            prediction = self.predict_strategy_outcome(
                task, spec, episode_id, context=plan_context,
                timeout_s=max(0.001, remaining))
            calls_made += 1
            # Charge THIS call's known spend immediately (failed calls
            # included: their tokens are real) so the loop's own budget
            # gate sees them before the next call is considered.
            _charge_call(prediction)
            predictions.append((spec, prediction))
        plan.model_calls_made = calls_made
        # Conservative comparison on ONE yardstick.
        only_predictions = [p for _spec, p in predictions]
        scores = score_strategy_outcome_predictions(only_predictions, limits)
        comparable = [s for s in scores if s.utility is not None]
        suggested_spec: Optional[ActionSpec] = None
        suggestion_basis = ""
        if comparable and self.plan_mode == "advise":
            best = max(comparable, key=lambda s: s.utility)
            for spec, prediction in predictions:
                if prediction.prediction_id == best.prediction_id:
                    suggested_spec = spec
                    break
            suggestion_basis = (
                f"U={best.utility} = {limits.alpha}*G({best.benefit_value})"
                f" - {limits.beta}*C({best.cost_normalized}) - "
                f"{limits.gamma}*R({best.risk_effective}); "
                f"{len(scores)} candidate(s) compared under protocol "
                f"{STRATEGY_OUTCOME_PROTOCOL_VERSION} from context "
                f"{plan_context.context_id}; conservative yardstick "
                "(unknown cost => peak share, unknown risk => full weight, "
                "unknown benefit => nothing)")
        elif comparable:
            suggestion_basis = ("shadow mode: candidates evaluated and "
                                "recorded; suggestion withheld")
        # Real planning spend was charged to the decision action PER CALL
        # inside the loop (see _charge_call): the totals below are the SAME
        # numbers, reported once — never a second amendment.
        if planning_measured:
            plan.planning_cost = {
                "cost": {d: round(planning_total[d], 4)
                         for d in sorted(planning_measured)},
                "measured": sorted(planning_measured),
                "note": ("REAL spend of the planning model calls (failed "
                         "calls included), charged to the decision action "
                         "as own cost AS EACH CALL RETURNED — sunk, never "
                         "part of any candidate's utility"),
            }
        if declared:
            final_view = self.budget.view(task_id, episode_id,
                                          budget=declared)
            plan.budget_confirmation = final_view["status"]
            if final_view["status"] == "exceeded":
                plan.status = "fallback"
                plan.truncation_reason = (
                    "declared budget exceeded by real consumption "
                    "(including this planning's own spend); the suggestion "
                    "is withheld — report the current best solution "
                    "instead.")
                self.actions.end_action(
                    decision.action_id, status="completed",
                    outcome={"kind": "plan_next_evaluation",
                             "plan_id": plan.plan_id,
                             "protocol": "strategy-outcome",
                             "n_candidates": len(predictions),
                             "suggested": None,
                             "suggestion_withheld": True,
                             "status": plan.status,
                             "truncation_reason": plan.truncation_reason})
                result = plan.to_dict()
                result["prediction_context_id"] = plan_context.context_id
                result["prediction_context_version"] = \
                    plan_context.version
                result["protocol"] = "strategy-outcome"
                result["candidates"] = [
                    {"action_spec": spec.to_dict(),
                     "prediction_id": prediction.prediction_id,
                     "prediction_status": prediction.status,
                     "score": score.to_dict()}
                    for (spec, prediction), score
                    in zip(predictions, scores)]
                return result
        if suggested_spec is not None:
            plan.suggested = suggested_spec
            plan.suggestion_basis = suggestion_basis
        elif comparable:
            plan.suggestion_basis = suggestion_basis
        plan.status = "truncated" if stop_reason else "ok"
        if stop_reason:
            plan.truncation_reason = stop_reason
        if not comparable:
            plan.status = "no_valid_predictions"
            statuses = sorted({p.status for p in only_predictions})
            plan.truncation_reason = (
                "no candidate carried a usable strategy-outcome prediction "
                f"(statuses: {statuses}); no suggestion is possible. Fall "
                "back to `recall`/Selector ordering or choose yourself — "
                "the calls that happened and their cost are recorded.")
        self.actions.end_action(
            decision.action_id,
            status="completed",
            outcome={"kind": "plan_next_evaluation",
                     "plan_id": plan.plan_id,
                     "protocol": "strategy-outcome",
                     "protocol_version": STRATEGY_OUTCOME_PROTOCOL_VERSION,
                     "root_snapshot_id": plan.root_snapshot_id,
                     "n_candidates": len(predictions),
                     "candidates": [
                         {"action_spec": spec.to_dict(),
                          "prediction_id": prediction.prediction_id,
                          "prediction_status": prediction.status,
                          "score": score.to_dict()}
                         for (spec, prediction), score
                         in zip(predictions, scores)],
                     "suggested": (plan.suggested.to_dict()
                                   if plan.suggested else None),
                     "suggestion_basis": plan.suggestion_basis,
                     "suggestion_withheld": self.plan_mode == "shadow",
                     "status": plan.status,
                     "truncation_reason": plan.truncation_reason,
                     "model_calls_made": int(plan.model_calls_made),
                     "planning_cost": copy.deepcopy(plan.planning_cost),
                     "budget_confirmation": plan.budget_confirmation,
                     "prediction_context_id": plan_context.context_id,
                     "prediction_context_version": plan_context.version})
        result = plan.to_dict()
        result["prediction_context_id"] = plan_context.context_id
        result["prediction_context_version"] = plan_context.version
        result["protocol"] = "strategy-outcome"
        result["candidates"] = [
            {"action_spec": spec.to_dict(),
             "prediction_id": prediction.prediction_id,
             "prediction_status": prediction.status,
             "score": score.to_dict()}
            for (spec, prediction), score in zip(predictions, scores)]
        return result

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
    def profile(self, task: Dict[str, Any],
                cir: Optional[Any] = None) -> ProblemProfile:
        return profile_task(task, cir=cir)

    def derivation_report(self, task: Dict[str, Any],
                          cir: Optional[Any] = None) -> Dict[str, Any]:
        """Per-dimension coupling derivation report: value, origin
        (cir/spec/supplied/null), notes, model verification issues, and
        cross-check warnings.

        No solve-script parameter: the profile is IDENTITY, frozen before
        the strategy, and a solve script is a post-strategy artifact."""
        return derivation_report(self.profile(task, cir=cir))

    def recall(self, task: Dict[str, Any], *, top: int = 3,
               exclude: Optional[Sequence[str]] = None,
               memory_mode: str = "cost-aware",
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
        profile = self.profile(task)
        recs = self.selector.recall(profile, top=top, exclude=exclude,
                                    memory_mode=memory_mode,
                                    include_unverified=include_unverified)
        solvers = available_families()
        result = {
            "profile": profile.to_dict(),
            # The task VERSION this result was produced for. Recorded so a
            # caller that wants to reuse the result as frozen prediction
            # input can PROVE it belongs to the current task version —
            # an unversioned result must not be dressed up as aligned
            # evidence (see world_model.context.retrieval_reuse_problems).
            "task_digest": task_text_digest(task),
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

    def predict_cost(self, task: Dict[str, Any], strategy_id: str
                     ) -> PredictionSnapshot:
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
        profile = self.profile(task)
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
        # The execution config ACTUALLY used, recorded on the action so a
        # later binding can compare the predicted candidate's config
        # against what really ran — never the other way round (copying the
        # PREDICTED config onto the action would fabricate the proof it is
        # supposed to provide). Only what this call really knows: the
        # strategy, the solver, the verification level. Anything else
        # (a time limit the outer agent applied inside solve.py, a seed)
        # is NOT observable here and stays unknown.
        exec_params: Dict[str, Any] = {
            "strategy_id": strategy_id, "solver": solver,
            "verification_level": verification_level}
        action = self.actions.begin_action(
            "execute_strategy", str(task["task_id"]), episode_id,
            pre_snapshot=pre, params=exec_params)
        record = self.executor.execute(
            Path(code_path), Path(workspace), solver=solver,
            task_id=str(task["task_id"]), strategy_id=strategy_id,
            profile=profile, verification_level=verification_level)
        # Evidence completeness: preserve the coupling-aware representation
        # snapshot (CIR) that was actually solved. Snapshot only — CIR
        # extraction and coupling understanding are untouched. The PARSED
        # form is stored: freezing the raw payload made the record look like
        # it carried a structure while every consumer parsed nothing out of
        # it, and a malformed CIR is refused rather than persisted.
        if task.get("coupling") is not None:
            from or_harness.core.coupling import cir_from_task
            cir = cir_from_task(task)
            if cir is not None:
                record.cir_snapshot = cir.to_dict()
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
               execution_ids: Optional[Sequence[str]] = None,
               family: Optional[str] = None,
               cell: Optional[str] = None) -> Dict[str, Any]:
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

        ``family`` / ``cell`` narrow the target set so a verification check
        applies ONLY to the structural unit it was written for. ``--verify``
        is per-claim, and one payload applied to every cell of a strategy
        would cross-contaminate families (an allocation check overwriting a
        scheduling verdict). ``family`` selects one family's targets;
        ``cell`` selects one structural cell (family + coupling tokens) and
        implies its family. Both are applied to ``_induction_targets`` and
        compose with ``strategy_id``."""
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
                targets = self._induction_targets(strategy_id, all_,
                                                  family=family, cell=cell)
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

    def exclude_execution(self, execution_id: str, reason: str, *,
                          superseded_by: Optional[str] = None
                          ) -> Dict[str, Any]:
        """Withdraw a wrong execution fact from the evidence set.

        The Evidence Bank is append-only, so a bad observation is never
        deleted — it is EXCLUDED: the row stays for audit, its ``source``
        becomes ``"excluded"``, and every statistics / induction / trigger /
        retrieval path (all of which require ``source == "executed"``) stops
        counting it. ``superseded_by`` links the corrected re-run.

        The change lands in the derived layers at the next ``induce`` /
        ``rebuild-index``: exclusion rewrites no entry. Its vector is removed
        from the execution index immediately (a stale vector would keep
        surfacing the withdrawn fact in recall)."""
        record = self.bank.exclude(execution_id, reason,
                                   superseded_by=superseded_by)
        unindexed = None
        if self.embedding_index is not None:
            from or_harness.strategy.embedding_index import LAYER_EXECUTION
            try:
                unindexed = self.embedding_index.remove(LAYER_EXECUTION,
                                                        [execution_id])
            except Exception as exc:  # noqa: BLE001 - never block the exclusion
                unindexed = {"removed": 0,
                             "deferred": f"{type(exc).__name__}: {exc}"}
        return {"excluded": execution_id, "reason": str(reason),
                "superseded_by": superseded_by,
                "correction": record.execution_features.get("correction"),
                "index": unindexed}

    def restore_execution(self, execution_id: str, reason: str
                          ) -> Dict[str, Any]:
        """Reverse :meth:`exclude_execution`: the fact counts again.

        A second explicit statement (an exclusion can itself be wrong). The
        correction history is kept on the fact. Re-indexing is left to the
        caller via ``rebuild_index`` — a restore is rare and explicit, so it
        does not silently rewrite derived index state."""
        record = self.bank.restore(execution_id, reason)
        return {"restored": execution_id, "reason": str(reason),
                "correction": record.execution_features.get("correction")}

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

    def _induction_targets(self, strategy_id: Optional[str], all_: bool,
                           family: Optional[str] = None,
                           cell: Optional[str] = None):
        """One induction target per (structural group, strategy).

        The group is DERIVED from each record's own profile snapshot rather
        than read from the stored index column: legacy rows carry the old
        index format, and letting that decide targets would make the facts
        invisible to induction.

        ``family`` narrows to one family; ``cell`` narrows to one structural
        cell (a full ``group_key`` token). ``cell`` is compared against the
        record's DERIVED key, so it works regardless of the stored index
        format. Passing either widens the implicit ``--all`` scope: naming a
        unit is itself an explicit selection, so it does not also require
        ``--all``."""
        explicit_unit = family is not None or cell is not None
        targets = []
        seen = set()
        for rec in self.bank.all():
            if rec.source != "executed" or rec.measurement_scope != "attempt":
                continue
            if strategy_id and rec.strategy_id != strategy_id:
                continue
            derived = group_key(rec.profile_snapshot)
            if family is not None and rec.profile_snapshot.family != family:
                continue
            if cell is not None and derived != cell:
                continue
            key = (derived, rec.strategy_id)
            if key in seen:
                continue
            if not all_ and strategy_id is None and not explicit_unit:
                continue
            seen.add(key)
            targets.append((rec.profile_snapshot, rec.strategy_id))
        return targets
