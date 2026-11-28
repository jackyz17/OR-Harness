"""ORHarness facade: the Python entry point for the outer harness agent.

The harness is the orchestrator. This layer advises and executes; the harness
may refuse recommendations, request alternatives, execute without recording,
override recorded costs, and decides when to induce and when to collect
garbage. No conversation loop, no runtime LLM calls, no hidden global state —
the memory location is always explicit (``home`` / ``OR_HARNESS_HOME``).
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from or_harness.adapters.solver import available_families, probe_all
from or_harness.core.schema import (
    StrategicEntry,
    COST_DIMENSIONS,
    DEFAULT_PREDICTED_COST_WEIGHTS,
    NON_CUMULATIVE_DIMENSIONS,
    CostVector,
    ExecutionRecord,
    FailureRecord,
    PredictionSnapshot,
    ProblemProfile,
    accumulate_measured_costs,
    compare_methods,
    compute_cost_feedback,
    group_key,
    normalize_method,
    profile_matches,
    task_check_block,
    task_check_state,
)
from or_harness.core.storage import BudgetExhausted, StorageError, Store, resolve_home
from or_harness.execution.executor import SafePythonExecutor
from or_harness.execution.inputs import PreparedTask, prepare_task_inputs, execution_task_inputs
from or_harness.profiling.profiler import derivation_report, profile_task
from or_harness.strategy.embedding_index import (
    EmbeddingBackend,
    EmbeddingIndex,
    create_embedding_backend,
    index_dir_for,
)
from or_harness.strategy.experience_bank import ExperienceBank
from or_harness.strategy.index_sync import IndexSynchronizer
from or_harness.strategy.induction import InductionEngine
from or_harness.strategy.selector import Selector, is_publishable
from or_harness.strategy.stats import ConditionalStats, quality_score
from or_harness.strategy.strategic_bank import (
    EFFECT_ATTRIBUTION_VERDICTS,
    LIFECYCLE_ACTIONS,
    StrategicBank,
)
from or_harness.strategy.triggers import solver_advisories
from or_harness.strategy.vector_recall import (
    VectorRecallUnavailable,
    recall_vectors,
)
from or_harness.world_model.actions import (
    ActionLog,
)
from or_harness.world_model.attribution import check_candidate_config
from or_harness.world_model.budget import BudgetLedger
from or_harness.world_model.contracts import (
    LEGACY_CONTRACT_VERSION,
    LEGACY_UNMAPPABLE,
    PAYLOAD_VERSION_UNKNOWN_PREFIX,
    PREDICTION_KINDS,
    SERVICE_IMPLEMENTED_KINDS,
    BenefitEstimate,
    CandidateRef,
    EvidenceRef,
    ExpectedCost,
    HarnessCapabilityEvidence,
    PredictionServiceStatus,
    PredictionTrace,
    RiskStatement,
    StrategyOutcomePrediction,
    UncertaintyStatement,
    capability_evidence_from_legacy_harness_state,
    detect_payload_version,
    legacy_prediction_view,
    load_contract_payload,
    normalize_candidate_payload,
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
    context_identity_problems,
    effective_input_version,
    evidence_identity,
    frozen_knowledge_available,
    frozen_knowledge_view,
    memory_content_digest,
    problem_identity_version,
    resolve_effective_cir,
    retrieval_reuse_problems,
    structure_problems,
    task_with_effective_cir,
)
from or_harness.world_model.contracts import (
    ActionSpec,
    OutcomePrediction,
)
from or_harness.world_model.contracts import (
    EvidenceRef,
)
from or_harness.world_model.provider import (
    NotConfiguredProvider,
    WorldModelProvider,
)
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

def _capability_gain_view(prediction: Any) -> Optional[Dict[str, Any]]:
    """The prediction's capability-gain block as READ-ONLY explanatory info.

    H+ is surfaced next to the candidate so a reader sees what the candidate
    might TEACH, but it is deliberately NOT the utility: the planner's score
    is benefit/cost/risk only, and a gain block never enters it. An absent
    or empty block returns ``None`` — "no gain claimed", never a default
    positive.

    The view carries the claim, the applicability, the expected changes (in
    their OWN metrics, never a compressed 0-1 score), what real evidence
    would confirm them, and the degradation risk — plus an explicit note
    that it is a prediction, verified only later by the offline capability
    path.
    """
    signal = (getattr(prediction.trace, 'model_info', {}) or {}).get('hplus')
    if signal:
        return {'kind':'frozen_induction_signal','decision':signal['decision'],
                'decision_status':signal['decision_status'],
                'explanation':signal.get('claim') or '; '.join(signal.get('basis') or []),
                'verification_conditions':copy.deepcopy(signal.get('verification_conditions') or []),
                'gaps':list(signal.get('gaps') or [])}
    gain = getattr(prediction, "capability_gain", None)
    if gain is None or not getattr(gain, "claimed", False):
        return None
    view = gain.to_dict()
    view["kind"] = "predicted_capability_gain"
    view["note"] = ("a PREDICTION about what this candidate might teach. It "
                    "is explanatory only: it is not the prediction's utility, "
                    "it is never written into the harness capability "
                    "evidence, and it becomes 'verified' only if a LATER task "
                    "observes the improvement (the offline capability path)")
    return view


def _candidate_reference(spec: Dict[str, Any]) -> tuple:
    """The identity tuple that makes one candidate distinguishable.

    Used to decide whether a choice DEVIATES from a suggestion and whether
    two names refer to the same candidate. It covers action type, strategy,
    solver, scope and the EXECUTION CONFIG — the fields that make two
    candidates genuinely different. It deliberately excludes cosmetic keys
    (a note, a description, insertion order), so an unchanged candidate
    never reads as a deviation, and it deliberately includes ``config``, so
    the same strategy under a different time limit / seed is NOT treated as
    "the same candidate" (which would hide a real deviation).
    """
    spec = spec or {}
    config = spec.get("config", spec.get("params")) or {}
    return (
        spec.get("action_type"),
        spec.get("strategy_id"),
        spec.get("solver"),
        spec.get("scope"),
        json.dumps(config, sort_keys=True, default=str),
    )


class ORHarness:
    def __init__(self, home: Optional[str] = None, *,
                 alpha: float = 1.0, beta: float = 1.0, gamma: float = 1.0,
                 cost_weights: Optional[Dict[str, float]] = None,
                 executor: Optional[SafePythonExecutor] = None,
                 world_model: Optional[WorldModelProvider] = None,
                 embedding: Optional[EmbeddingBackend] = None,
                 planning: bool = True,
                 plan_mode: str = "advise",
                 execution_context: Optional[Dict[str, Any]] = None,
                 induction_mode: Optional[str] = None):
        # The caller can declare the solving phase and interface before its
        # first action. Freeze these conditions in snapshots.
        if execution_context is None:
            import os
            raw_context = os.environ.get("OR_EXECUTION_CONTEXT")
            if raw_context:
                raw_context = raw_context.strip()
                execution_context = json.loads(raw_context if raw_context.startswith("{")
                    else Path(raw_context.removeprefix("@")).read_text(encoding="utf-8"))
        from or_harness.world_model.cost_signal import normalize_execution_context
        self.execution_context = normalize_execution_context(execution_context)
        self.home = resolve_home(home)
        self.store = Store(self.home)
        import os
        induction_mode = induction_mode or os.environ.get("OR_INDUCTION_MODE")
        if induction_mode is not None:
            from or_harness.world_model.hplus import configure
            configure(self, induction_mode)
        # The utility weights belong to the WORLD MODEL's consequence
        # comparison (score_strategy_outcome_predictions), NOT to recall: recall only
        # discovers material. They are kept on the harness so the planner
        # defaults can read them.
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        # Formal C has its own four-dimensional defaults. The selector keeps
        # its legacy reporting weights, so changing C does not change raw
        # cost reports. An explicit map, including {}, replaces the defaults.
        self.cost_weights = dict(DEFAULT_PREDICTED_COST_WEIGHTS
                                 if cost_weights is None else cost_weights)
        self.bank = ExperienceBank(self.store)
        self.sbank = StrategicBank(self.store)
        self.stats = ConditionalStats(self.bank)
        # NO built-in strategy directory. The candidate set is derived from
        # memory at recall time (real executions + real entries), and the
        # outer agent proposes the candidates it wants compared. A built-in
        # directory would be a second, fabricated source of candidates and of
        # method content (names, descriptions, actions, fallbacks) that no
        # execution ever observed.
        self.selector = Selector(self.sbank, self.stats,
                                 cost_weights=cost_weights)
        self.executor = executor or SafePythonExecutor()
        self.induction = InductionEngine(self.stats, self.sbank, harness=self)
        # World-model M1 substrate: action log + budget ledger (log/index
        # facilities — NOT a third knowledge bank).
        self.actions = ActionLog(self.store)
        self.budget = BudgetLedger(self.bank, self.actions)
        # One explicitly configured provider for current G/C/R/H+ predictions.
        self.world_model = world_model or NotConfiguredProvider()
        from or_harness.world_model.strategy_prediction import (
            StrategyOutcomeService,
        )
        self.strategy_predictions = StrategyOutcomeService(
            self.store, self.world_model)
        # The shared prediction freezes the binary H+ signal through this harness.
        self.strategy_predictions.harness = self
        # Bounded planning. ``planning=False`` explicitly disables plan_next;
        # ``plan_mode`` is "advise" (a suggestion is produced for the agent
        # to accept or reject) or "shadow" (paths are evaluated and
        # recorded but no suggestion field is emitted — observation only).
        # There is no unattended auto-execution mode, ever.
        if plan_mode not in ("advise", "shadow"):
            raise ValueError("plan_mode must be 'advise' or 'shadow'")
        self.planning = bool(planning)
        self.plan_mode = plan_mode
        # Candidates the ENTRY-POINT normalizer refused for the CURRENT
        # planning call: a correctable input error reported back with the
        # plan, never a doomed model call. Reset at the start of each
        # ``plan_next``.
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
                                           self.store,
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
                          *, task_digest: Optional[str] = None,
                          problem_identity: Optional[str] = None
                          ) -> Dict[str, Any]:
        """The episode's CURRENT information state (X): the latest real
        task_progress among this episode's snapshots.

        Continuity rule: each action's post snapshot merges its own progress
        update into the accumulated state, so the next action's pre state
        inherits what earlier actions established (selected_plan, model
        artifact, current solution, verification evidence...). Hypothetical
        snapshots NEVER contribute — an imagined outcome is not progress.

        Task boundary: progress is inherited only from snapshots taken
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
        from or_harness.world_model.conditions import latest_snapshot
        best = latest_snapshot(self.store, task_id, episode_id,
                               task_digest=task_digest, problem_identity=problem_identity)
        return copy.deepcopy(best.task_progress) if best else {}

    def snapshot(self, task: Dict[str, Any], episode_id: Optional[str] = None,
                 *, task_progress: Optional[Dict[str, Any]] = None,
                 hypothetical: bool = False, persist: bool = True,
                 include_memory: bool = True) -> BeliefSnapshot:
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
        task = prepare_task_inputs(task, self.home)
        # The receipt and the availability high-water mark describe the
        # same bank state, even if another process is publishing knowledge.
        with self.store.locked():
            profile = self.profile(task)
            knowledge = (verified_knowledge_view(profile, self.sbank) if include_memory else {
                "available": [{"entry_id": entry.entry_id, "predicates": entry.predicates}
                              for entry in self.sbank.matching(profile, include_dormant=True)]})
            harness_state = self._harness_state_view(task, knowledge)
            if self.execution_context:
                harness_state["execution_context"] = copy.deepcopy(self.execution_context)
            budget = self._load_budget(str(task.get("task_id", "")), episode_id)
            budget_state = self.budget.view(
                str(task.get("task_id", "")), episode_id, budget=budget)
            progress = self._episode_progress(
                str(task.get("task_id", "")), episode_id,
                task_digest=task_text_digest(task),
                problem_identity=problem_identity_version(task))
            if task_progress:
                progress.update(copy.deepcopy(task_progress))
            snap = BeliefSnapshot.build(
                task, episode_id,
                harness_state=harness_state,
                problem_state={
                    "profile": profile.to_dict(),
                    "task_input": copy.deepcopy(task),
                    # The PROBLEM's identity at freeze time, excluding the
                    # post-strategy model artifact. The binding check compares
                    # the prediction's value against this one, so writing the
                    # model after choosing the strategy is not reported as a
                    # change of problem while a real change still is.
                    "problem_identity_digest": problem_identity_version(task),
                },
                task_progress=progress,
                budget_state=budget_state,
                coverage=(self._coverage_view(profile, knowledge)
                          if include_memory else {}),
                hypothetical=hypothetical,
            )
            if persist:
                from or_harness.world_model.conditions import (
                    freeze_availability, save_snapshot,
                )
                revision = freeze_availability(self.store, knowledge)
                save_snapshot(self.store, snap, availability_revision=revision)
            return snap

    def get_snapshot(self, snapshot_id: str) -> Optional[BeliefSnapshot]:
        """Read a lightweight condition receipt or a complete legacy snapshot.

        New receipts explicitly mark coverage as unavailable; this never
        fabricates a historical prediction input from the current bank.
        """
        from or_harness.world_model.conditions import read_snapshot
        return read_snapshot(self.store, snapshot_id)

    def snapshots(self, *, task_id: Optional[str] = None) -> List[BeliefSnapshot]:
        ids = []
        for table in ("frozen_conditions", "belief_snapshots"):
            sql = f"SELECT snapshot_id FROM {table}"
            params = []
            if task_id is not None:
                sql += " WHERE task_id=?"
                params.append(task_id)
            ids.extend(r["snapshot_id"] for r in self.store.conn.execute(sql, params))
        return sorted((self.get_snapshot(key) for key in set(ids)),
                      key=lambda snap: (snap.created_at, snap.snapshot_id))

    def _coverage_view(self, profile, knowledge: Dict[str, Any], *,
                       knowledge_version: Optional[str] = None
                       ) -> Dict[str, Any]:
        """Conditional capability evidence: cell statistics + layered
        knowledge, with coverage gaps. NO composite capability score.

        ``strategies_without_evidence`` lists strategies that this family's
        memory really contains but that have NO attempt-scope evidence in
        THIS structural cell. It is derived from the evidence, not from a
        directory: a strategy nobody ever ran in this family is not "missing
        evidence", it is simply not part of the memory at all.

        ``knowledge_version`` is the CONTENT HASH of the layered knowledge
        (excluding per-freeze timestamps). The layers travel by value (the
        view must stand on its own and survive archiving), but the hash is
        recorded so a reader can tell two snapshots that froze the SAME
        knowledge apart from a real revision, and so the shared copy can be
        reused rather than re-stored.
        """
        cells = self.stats.for_profile(profile)
        seen = self.stats.strategy_ids_in_family(profile.family)
        out = {
            "cell_statistics": {sid: cell.to_dict() for sid, cell
                                in cells.items()},
            "knowledge_layers": {
                "verified": knowledge["verified"],
                "legacy_unknown": knowledge["legacy_unknown"],
                "unverified": knowledge["unverified"],
            },
            "coverage_gaps": {
                "strategies_without_evidence": sorted(
                    sid for sid in seen
                    if sid not in cells or cells[sid].n == 0),
                "note": "strategies with recorded evidence elsewhere in this "
                        "family but none in this structural cell; no quality "
                        "claim is made for them. A strategy absent from this "
                        "list was never tried in this family at all",
            },
        }
        if knowledge_version:
            out["knowledge_version"] = knowledge_version
        return out

    def _harness_state_view(self, task: Dict[str, Any],
                            knowledge: Dict[str, Any]) -> Dict[str, Any]:
        """The frozen ``harness_state`` block of a belief snapshot.

        This is EVIDENCE ABOUT H, not a measured H: value-copied knowledge
        references, experience volume, and tool configuration. The name and
        the shape are kept for backward compatibility with stored snapshots;
        the unified capability view is
        :meth:`capability_evidence`, which labels each item's evidential
        status and defines no composite score.

        The knowledge references travel in ``coverage.knowledge_layers``
        ONLY. Storing the same ``KnowledgeRef`` batch here as well doubled
        every snapshot's size for no information gain, so this block keeps
        the experience and tool-config components and the ONE knowledge
        copy lives in the coverage view (readers that historically looked
        here fall back to it — see ``frozen_knowledge_view``).
        """
        recent = self.bank.query(task_id=str(task.get("task_id", "")))
        return {
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
            # Only a strategic_entry row's refs are entry numbers; a
            # conditional_stats row's refs are execution ids. The row's own
            # ``evidence`` field distinguishes them — no id-prefix guess.
            refs = ([str(r) for r in (rec.get("evidence_refs") or [])]
                    if rec.get("evidence") == "strategic_entry" else [])
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
          retrieval, the knowledge view and the cell
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
        if not (historical or (snapshot is not None and historical is None)):
            task = prepare_task_inputs(task, self.home)
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
            snapshot = self.snapshot(effective_task, episode_id, persist=persist)

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
            # Knowledge comes from the SNAPSHOT's saved
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
        else:
            knowledge = verified_knowledge_view(profile, self.sbank)
        # M4: the published EXPERIENCE calibration of the strategy-outcome
        # service (closed episodes only). Frozen into the context as its
        # own block,
        # carrying the summary's CONTENT (groups, counts, sources) so a
        # later reader of the context sees what the model was told, not
        # just an id. An active episode never sees its own not-yet-closed
        # feedback: only closed episodes contribute.
        if historical:
            strategy_calibration = {}
            paired_feedback = {}
            hplus_feedback = {}
            history_notes.append(
                "MISSING: the strategy-outcome experience calibration was "
                "not saved with this snapshot, so none is sent (today's "
                "summary was NOT consulted for a historical reconstruction)")
            history_notes.append(
                "MISSING: the paired prediction-execution feedback was not "
                "saved with this snapshot, so none is sent (today's store "
                "was NOT consulted for a historical reconstruction)")
            history_notes.append(
                "MISSING: the capability-gain follow-up state was not saved "
                "with this snapshot, so none is sent (today's traces were "
                "NOT consulted for a historical reconstruction)")
        else:
            from or_harness.world_model.episode_closeout import (
                calibration_summary_for_context,
            )
            # The published summary, filtered to the ATTACHED provider's
            # model identity: another model's error statistics are not
            # evidence about this one, so they are withheld (and the
            # withholding is reported on the block).
            #
            # The identity passed here MUST be the same STRING the grouping
            # used, so it comes from the one shared label function rather
            # than from the raw `provider_model` field: groups are keyed
            # `...|<model>@<version>|...`, and passing the bare model name
            # matched NOTHING — the attached model's own calibration was
            # withheld, while an UNKNOWN identity (None) skipped the filter
            # entirely and let every other model's statistics through.
            # "Unknown" is a real identity value, never a reason to stop
            # filtering.
            strategy_calibration = calibration_summary_for_context(
                self,
                model_identity=self.strategy_predictions.model_identity_label)
            # The compact per-episode prediction-execution pairs of closed
            # episodes (M5). Built from the SAME closed-episode window as
            # the summary above, so it can never carry an active episode's
            # unclosed feedback. It is a read: no model call, no storage.
            # r15: the pair set is not dumped whole — the RELEVANT, bounded
            # subset is SELECTED (using the current problem, the recall
            # already gathered and the candidate(s) this context is for) and
            # FROZEN, so every candidate of this decision shares one
            # historical set and a later reuse never re-selects.
            from or_harness.world_model.episode_closeout import (
                paired_feedback_for_context,
                paired_feedback_selection_from_block,
            )
            paired_feedback_block = paired_feedback_for_context(
                self,
                model_identity=self.strategy_predictions.model_identity_label)
            selection_candidates: List[Any] = []
            if context_spec is not None:
                selection_candidates.append(context_spec)
            else:
                selection_candidates.extend(list(candidates or []))
            paired_feedback = paired_feedback_selection_from_block(
                paired_feedback_block, task=effective_task, profile=profile,
                recall_result=recall_result,
                candidates=selection_candidates, scope=None)
            # Relevant completed H+ reviews, selected before the request budget.
            from or_harness.world_model.hplus import feedback
            hplus_feedback = feedback(self, task=task, budget_chars=3000)
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
            recorded_choices=recorded_choices)
        if historical:
            constraints = self._historical_constraints_view(snapshot)
        else:
            constraints = self._execution_constraints_view(effective_task,
                                                           episode_id)
        version_block = capability_version(
            harness_config={
                "alpha": self.alpha, "beta": self.beta,
                "gamma": self.gamma,
                "plan_mode": self.plan_mode,
            },
            provider=self.world_model.describe(),
            prompt_template_version=self._prompt_template_version(),
            tools=constraints.get("available_solver_families") or [],
            knowledge_content=memory_content_digest(
                knowledge=knowledge, retrieval=retrieval_view,
                entries=([] if historical
                         else self.sbank.list(include_dormant=True))))
        spec_list = ([context_spec] if context_spec is not None
                     else list(candidates or []))
        strategies = [s.strategy_id for s in spec_list if s.strategy_id]
        # ONE frozen historical-execution retrieval for this decision. It is
        # gathered NOW for a current decision and bounded to the decision
        # moment (the root snapshot's time): an execution recorded after the
        # decision can never enter its history. A historical reconstruction
        # does not gather it at all — reading today's index would fabricate
        # a retrieval for an earlier state — and reports the gap instead.
        if historical:
            history_retrieval: Dict[str, Any] = {
                "version": "wm-history/1",
                "status": "not_reconstructed",
                "pool": [],
                "global": {},
                "accounting": {},
                "degraded": [{
                    "part": "local_history",
                    "reason": ("historical reconstruction: today's "
                               "execution index and closed-episode window "
                               "were NOT consulted (a later execution would "
                               "leak into an earlier decision)")}],
                "missing": ["history_retrieval: no retrieval was saved with "
                            "the supplied snapshot, so none is rebuilt"],
            }
            history_notes.append(
                "MISSING: the unified historical retrieval was not saved "
                "with this snapshot; it is NOT gathered from today's index")
        else:
            from or_harness.world_model.history_retrieval import (
                build_history_retrieval,
            )
            history_retrieval = build_history_retrieval(
                self, task=effective_task, profile=profile,
                candidates=spec_list, episode_id=episode_id,
                cutoff=float(getattr(snapshot, "created_at", 0.0) or 0.0),
                # The published internal observation projection covers the
                # full calibration window; detailed pairs remain display-bounded.
                paired_feedback=paired_feedback_block,
                strategy_calibration=strategy_calibration)
        context = build_context(
            task=task, profile=profile, snapshot=snapshot,
            recall_result=recall_result, derivation=derivation,
            cir=resolved_cir, math_declared=math, capability=evidence,
            capability_version_block=version_block,
            execution_constraints=constraints,
            strategy_calibration=strategy_calibration,
            paired_feedback=paired_feedback,
            history_retrieval=history_retrieval,
            prediction_reminders=(strategy_calibration.get("reminders")
                                  if not historical else []),
            hplus_feedback=hplus_feedback,
            cir_source=("caller_supplied" if cir is not None
                        else "task_coupling"),
            retrieval_bounding=bounding,
            top_k=max(1, vector_top_k or top),
            include_unverified=include_unverified)
        # Keep candidate-related frozen knowledge facts independently of
        # the retired knowledge-change proposal heuristic.
        selected_entries = {str(ref.get("entry_id")) for refs in knowledge.values()
                            for ref in refs if ref.get("strategy_id") in strategies}
        selected_entries.update(str(hit.get("content", {}).get("entry_id"))
                                for hit in context.retrieval.hits)
        selected_strategies = set(strategies)
        selected_strategies.update(str(hit.get("content", {}).get("strategy_id"))
                                   for hit in context.retrieval.hits)
        cells = (snapshot.coverage or {}).get("cell_statistics") or {}
        context.snapshot["coverage"] = {
            "cell_statistics": {sid: {k: copy.deepcopy(v) for k, v in cell.items()
                                      if k != "execution_ids"}
                                for sid, cell in cells.items() if sid in selected_strategies},
            "knowledge_layers": {layer: [copy.deepcopy(ref) for ref in refs
                if str(ref.get("entry_id")) in selected_entries]
                for layer, refs in knowledge.items()},
            "projection": "candidate-related statistics; support lists omitted",
        }
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
            from or_harness.world_model.conditions import persist_context
            persist_context(self.store, context)
        return context

    def get_prediction_context(self,
                               context_id: str) -> Optional[PredictionContext]:
        """Read a stored context back (never re-runs retrieval or a model).

        The version is CHECKED, not guessed at: a payload written by a build
        that used a different context schema is refused rather than
        half-read."""
        from or_harness.world_model.conditions import read_context
        return read_context(self.store, context_id)

    def prediction_contexts(self, *, task_id: Optional[str] = None,
                            episode_id: Optional[str] = None
                            ) -> List[PredictionContext]:
        """Active inputs and readable legacy contexts; released manifests
        remain available through Store.get_prediction_context/inspect."""
        from or_harness.world_model.conditions import ContextExpired
        out = []
        for raw in self.store.prediction_contexts_for(task_id=task_id,
                                                     episode_id=episode_id):
            data = self.store.loads(raw)
            try:
                out.append(self.get_prediction_context(data["context_id"]))
            except ContextExpired:
                continue
        return out

    def _prompt_template_version(self) -> str:
        from or_harness.world_model.strategy_prediction import STRATEGY_OUTCOME_PROTOCOL_VERSION
        return STRATEGY_OUTCOME_PROTOCOL_VERSION

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
        consumption = budget.get("consumption") or {}
        episode_dims = [d for d in (declared or {}) if d != "latency_s"]
        unknown_dims = list(consumption.get("unknown_dims") or [])
        episode_unknown = [d for d in unknown_dims if d in episode_dims]
        return {
            "declared_budget": copy.deepcopy(declared) if declared else None,
            "budget_status": budget.get("status"),
            # The FROZEN budget CONDITION a ``budget_exhausted`` prediction is
            # about: the declared limits (and the episode-scoped subset), the
            # KNOWN consumption (a LOWER BOUND whenever a dimension is
            # unmeasured), whether the remaining budget is CONFIRMED, and the
            # unmeasured dimensions WITH their located reason. The model may
            # decline to predict when no valid condition exists; a probability
            # is never defaulted to 0.
            "budget_condition": {
                "declared_limits": (copy.deepcopy(declared)
                                    if declared else None),
                "episode_dimensions": episode_dims,
                "known_consumption_lower_bound": {
                    d: (consumption.get("total_cost") or {}).get(d)
                    for d in episode_dims},
                "remaining_budget_confirmed": bool(declared)
                and not episode_unknown,
                "unknown_dimensions": episode_unknown,
                "unknown_dimension_reasons": {
                    d: (consumption.get("unknown_dim_reasons") or {}).get(d)
                    for d in episode_unknown},
                "note": ("the condition a budget_exhausted prediction is "
                         "about: from this decision to the end of the solving "
                         "episode. A known consumption with an unmeasured "
                         "declared dimension is a LOWER BOUND and the "
                         "remaining budget is NOT confirmed; with no valid "
                         "condition the model may decline to predict (never "
                         "a default probability of 0)"),
            },
            "available_solver_families": available_families(),
            # API hints for discovered modules only; doctor has the full
            # installation report. Uninstalled solver tutorials should not
            # inflate every protected prediction core.
            "solver_capabilities": [r.to_dict() for r in probe_all() if r.available],
            "executor": self.executor.capabilities() if hasattr(self.executor, "capabilities") else {
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
            benefit_convention: Optional[Dict[str, Any]] = None,
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
        task = prepare_task_inputs(task, self.home)
        # The budget gate: a stopped task must not spend a NEW model call.
        # Placed before the candidate is normalized so a refusal touches
        # nothing — no context build, no provider call, no persistence.
        gate = self._budget_gate(str(task.get("task_id", "")), episode_id)
        if not gate["allowed"]:
            raise BudgetExhausted(gate["reason"], gate.get("view"))
        # ENTRY-POINT NORMALIZATION, before ANY model call: the minimal
        # candidate format (top-level name/steps, a method under its own key)
        # is turned into the ONE canonical shape here, shared with the
        # plan-next entry so the same hand-written input is never read two
        # ways. A payload that cannot be normalized (two disagreeing method
        # spellings, no name/step to identify) raises NOW, so a doomed model
        # call is never spent on it.
        entry_notes: List[str] = []
        if isinstance(candidate, dict):
            candidate = normalize_candidate_payload(candidate)
            entry_notes = list(candidate.pop("normalization_notes", []) or [])
        elif isinstance(candidate, ActionSpec):
            # A legacy ActionSpec is mirrored to a dict and re-normalized so
            # the minimal format reaches it too, then mapped back.
            raw = candidate.to_dict()
            entry_notes = list(candidate.normalization_notes)
            normalized = normalize_candidate_payload(raw)
            entry_notes.extend(normalized.pop("normalization_notes", []) or [])
            entry_notes = list(dict.fromkeys(entry_notes))
            candidate = ActionSpec.from_dict(normalized)
        if isinstance(candidate, ActionSpec):
            candidate_ref = CandidateRef.from_action_spec(candidate)
        elif isinstance(candidate, dict):
            candidate_ref = CandidateRef.from_dict(candidate)
        else:
            candidate_ref = copy.deepcopy(candidate)
        # ENTRY-POINT FIELD DISCIPLINE. A METHOD placed in ``config`` is
        # folded into the candidate's ``method`` field (the framework has
        # ONE field for the approach); a key that names no execution
        # parameter at all is refused NOW — never after an attempt has been
        # spent on a sample that could only ever read as an unknown.
        entry = check_candidate_config(candidate_ref.config,
                                       candidate_ref.method)
        if entry["errors"]:
            raise ValueError(
                "candidate.config is misused: " + "; ".join(entry["errors"]))
        candidate_ref.config = entry["config"]
        # The method is normalized to ONE shape here (a bare string becomes
        # ``{"name": ..., "steps": [...]}``), so a stored candidate can never
        # carry a shape that crashes on read (``dict("assignment MILP")``).
        normalized_method = normalize_method(entry["method"])
        candidate_ref.method = normalized_method or {}
        config_notes = entry_notes + list(entry["notes"])
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
            # The candidate is the retrieval ANCHOR for this single
            # prediction too: the unified history request carries its
            # declared method as one query view, exactly as a multi-candidate
            # plan does. A context built without the candidate would silently
            # degrade to problem-text-only retrieval.
            context = self.build_prediction_context(
                task, episode_id, cir=cir,
                candidates=[candidate_ref])
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
            benefit_baseline_hint=benefit_baseline_hint,
            benefit_convention=benefit_convention)
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
        # The PROBLEM's identity, which excludes post-strategy artifacts
        # (the ``model`` written after the strategy is chosen). The binding
        # check compares THIS, so the ordinary sequence "predict, write the
        # model, execute" is not reported as a content change, while a real
        # change to the task, its data, the CIR or the candidate still is.
        prediction.trace.model_info["problem_identity_digest"] = (
            problem_identity_version(task, cir))
        prediction.trace.model_info["execution_decision_required"] = True
        if task.get("input_resources"):
            prediction.trace.model_info["input_resources"] = copy.deepcopy(task["input_resources"])
        if config_notes:
            # A normalization that was performed rather than refused is
            # RECORDED on the prediction, so a reader can see the field was
            # re-homed instead of silently changing the candidate.
            prediction.trace.model_info["config_normalized"] = config_notes
            prediction.notes.extend(config_notes)
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
                      min_calibration_samples: Optional[int] = None,
                      induction_decision=None, execution_id=None,
                      effect_attributions: Optional[
                          Sequence[Dict[str, Any]]] = None
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

        ``effect_attributions`` records the agent's utility review of the
        entries THIS episode adopted, in the SAME call: each item is the
        existing ``{"entry_id","verdict","execution_id"?, "note"?,
        "lifecycle_action"?}`` shape. It uses the same validation and
        recording channel as ``induce --attribute-effect`` and needs no
        induction job — so it works when H+ = 0. A ``refuting`` verdict keeps
        its existing default (pause); an explicit ``keep``/``demote``/
        ``promote`` is the agent's management decision. Retiring remains the
        explicit ``retire`` command.

        The result also carries ``entry_usage``: the entries this episode
        actually adopted (version resolved from their frozen ``used_entries``
        declaration), their read-time utility summary and the adoptions that
        still have no effect judgement. Reading it writes nothing.
        """
        validated_attributions: List[Dict[str, Any]] = []
        if effect_attributions:
            from or_harness.world_model.hplus import mode
            if mode(self) == "frozen":
                raise ValueError(
                    "the strategy library is frozen for testing; utility "
                    "management is not applied in this home")
            # Validate BEFORE closing so a malformed attribution cannot leave
            # the caller with a closed episode and a rejected review.
            validated_attributions = [
                self._validate_effect_attribution(raw)
                for raw in effect_attributions]
        from or_harness.world_model.episode_closeout import (
            DEFAULT_MIN_CALIBRATION_SAMPLES,
            close_episode,
        )
        from or_harness.world_model.hplus import resolve_episode
        episode_id=resolve_episode(self,task_id,episode_id,execution_id=execution_id,finish_action_id=finish_action_id)
        result = close_episode(
            self, task_id, episode_id, terminal_state=terminal_state,
            finish_action_id=finish_action_id,
            min_calibration_samples=(min_calibration_samples
                                     if min_calibration_samples is not None
                                     else DEFAULT_MIN_CALIBRATION_SAMPLES))

        if result.get("unfinished_actions"):
            return result

        from or_harness.world_model.induction_gate import record_decision
        if induction_decision is None:
            finishes = self.actions.query(task_id=task_id,episode_id=episode_id,action_type="finish_task")
            finish = self.actions.get(finish_action_id) if finish_action_id else (finishes[-1] if finishes else None)
            induction_decision = copy.deepcopy((finish.outcome or {}).get("induction_decision")) if finish else None
            if induction_decision is not None:
                induction_decision = {"state":induction_decision} if isinstance(induction_decision,str) else induction_decision
                induction_decision.setdefault("source","finish_task_receipt")
                induction_decision.setdefault("source_ref",finish.action_id)
        if 'rule_validation' not in result:
            from or_harness.strategy.rule_versions import settle_episode_uses
            result['rule_validation'] = settle_episode_uses(self,task_id,episode_id)
        from or_harness.world_model.hplus import schedule_episode, formal
        if 'induction_schedule' not in result:
            result['induction_schedule'] = schedule_episode(self,task_id,episode_id)
        if formal(self):
            pending = result['induction_schedule']['pending_ids']
            result['induction_gate'] = {'state':'allow' if pending else 'skip',
                'source':'frozen_hplus','pending_ids':pending,
                'reason':'settle positive executed jobs' if pending else 'no positive executed job remains'}
            if induction_decision is not None:
                result['induction_gate']['ignored_closeout_judgement'] = induction_decision
        else:
            result["induction_gate"] = record_decision(self, task_id, episode_id, induction_decision)

        # The agent's utility review is recorded AT THE SAME close-out (also
        # when H+ = 0): this is management of EXISTING knowledge and needs no
        # induction job or pending state.
        if validated_attributions:
            recorded = self._record_attributions_at_close(
                validated_attributions)
            result["effect_attributions"] = recorded
            # Pause/promote also changes which version ordinary recall may
            # offer: synchronize the rule index for exactly the entries the
            # agent managed (same interface the induce path uses).
            from or_harness.strategy.rule_versions import sync_rule_entries
            affected = sorted({plan["entry_id"]
                               for plan in validated_attributions})
            result["index_sync"] = sync_rule_entries(self, affected, affected)
        # Read-time usage material for the entries this episode adopted.
        # Generated AFTER the review above, so the summary reflects it.
        from or_harness.strategy.rule_versions import episode_entry_usage
        result["entry_usage"] = episode_entry_usage(self, task_id,
                                                    episode_id)
        from or_harness.world_model.conditions import release_context
        for row in self.store.conn.execute(
                "SELECT context_id FROM runtime_inputs WHERE task_id=? AND episode_id IS ?",
                (task_id, episode_id)).fetchall():
            release_context(self.store, row["context_id"], reason="closed")
        return result

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
                    self.store.loads(row["value"]),self.store)
            except Exception:
                continue
            if task_id is not None and evaluation.task_id != task_id:
                continue
            if episode_id is not None \
                    and evaluation.episode_id != episode_id:
                continue
            out.append(evaluation.to_dict())
        return out

    def calibration_summary(self, *, min_samples: Optional[int] = None,
                            rebuild: bool = False
                            ) -> Dict[str, Any]:
        """The published experience-calibration summary (wm-calib/3).

        By default this READS the published summary — one row lookup, no
        history scan. The published object is rebuilt whenever the window
        changes (a close-out, a late task check, an exclusion, a restore),
        so the read path never has to.

        ``rebuild=True`` (``orx calibration --rebuild``) rebuilds and
        republishes it from the current window — the explicit migration
        path for a store that predates this version, and the repair path
        for a summary that was never published.

        Aggregated from the WINDOW's eligible evaluations only — an active
        episode never reads its own not-yet-closed feedback. This is the OR
        strategy-outcome reliability; it is a measured record, not a
        fitted calibrator and not a promise of future accuracy."""
        from or_harness.world_model.episode_closeout import (
            DEFAULT_MIN_CALIBRATION_SAMPLES,
            CalibrationPolicy,
            build_calibration_summary,
            published_calibration_summary,
            republish_calibration,
        )
        policy = CalibrationPolicy.from_env()
        if rebuild:
            return republish_calibration(
                self, policy=policy,
                min_calibration_samples=(min_samples
                                         if min_samples is not None
                                         else DEFAULT_MIN_CALIBRATION_SAMPLES))
        published = published_calibration_summary(self)
        if published is None:
            # Nothing published yet: build one now so an explicit read is
            # never empty, and publish it so later prediction reads are
            # O(1). (The prediction path itself does NOT do this — it
            # reports the gap instead of turning a read into a write.)
            return republish_calibration(
                self, policy=policy,
                min_calibration_samples=(min_samples
                                         if min_samples is not None
                                         else DEFAULT_MIN_CALIBRATION_SAMPLES))
        if min_samples is not None:
            return build_calibration_summary(
                self, min_samples=min_samples, policy=policy)
        return published

    def archive_calibration(self, *, dry_run: bool = False,
                            policy: Optional[Any] = None
                            ) -> Dict[str, Any]:
        """Move OUT-OF-WINDOW episode detail to the archive (retention).

        Only DETAIL moves (evaluation, prediction and frozen-context
        payloads); the close-out registry tombstone stays online so a
        repeated close remains idempotent and the window stays locatable.
        The archive is bounded by a per-file size cap, a total size cap and
        an age cap, so "the archive is bounded" is a true statement.
        """
        from or_harness.world_model.episode_closeout import (
            CalibrationPolicy,
            archive_calibration_detail,
        )
        return archive_calibration_detail(
            self, policy=policy or CalibrationPolicy.from_env(),
            dry_run=dry_run)

    def calibration_retention(self) -> Dict[str, Any]:
        """What the online store currently holds vs the retention policy.

        A read-only view of the three scopes (window / grace / archive) so
        an operator can see the online capacity, the archive capacity and
        the retention period as three separate numbers instead of one."""
        from or_harness.world_model.episode_closeout import (
            CalibrationPolicy,
            archive_dir,
            calibration_window,
        )
        policy = CalibrationPolicy.from_env()
        # READ-ONLY: reporting the retention state must not migrate a
        # legacy store. A never-migrated store reports 0 window episodes
        # here and the note says the window is derived; the migration
        # happens at `calibration --rebuild` / `close_episode`.
        window = calibration_window(self, policy, readonly=True)
        total_closeouts = len(self.store.closeout_registry())
        archived = len(self.store.closeout_registry(archived=True))
        directory = archive_dir(self)
        archive_files = []
        if directory.exists():
            for path in sorted(directory.iterdir()):
                if path.is_file():
                    archive_files.append({
                        "name": path.name,
                        "bytes": path.stat().st_size,
                    })
        return {
            "policy": policy.to_dict(),
            "online": {
                "n_window_episodes": len(window),
                "n_closeouts_total": total_closeouts,
                "n_closeouts_archived": archived,
                "n_contract_predictions":
                    self.store.count_contract_predictions(),
                "n_prediction_contexts":
                    self.store.count_prediction_contexts(),
            },
            "archive": {
                "directory": str(directory),
                "files": archive_files,
                "total_bytes": sum(f["bytes"] for f in archive_files),
            },
            "note": ("three separate scopes: the WINDOW decides which closed "
                     "episodes calibrate; the GRACE period decides how long "
                     "online detail is kept for a possible late check; the "
                     "ARCHIVE caps bound what history is kept on disk. "
                     "Registry tombstones stay online permanently so a "
                     "repeated close is still idempotent"),
        }

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
        if prediction.trace.model_info.get('execution_decision_required'):
            self._scored_decision(prediction,action_id=action_id)
            if (action.params or {}).get('prediction_id')!=prediction_id or not (action.params or {}).get('scored_adoption'):
                raise ValueError('bind-strategy cannot authorize an unscored or post-hoc execution; use plan-next then execute --prediction')
        model_info = prediction.trace.model_info
        bound = model_info.get("bound_action_id")
        if bound is not None and bound != action_id:
            raise StorageError(
                f"prediction {prediction_id!r} is already bound to action "
                f"{bound!r}")
        candidate = prediction.candidate
        # EPISODE BACKFILL: a prediction made without an episode (the
        # caller omitted `--episode`) whose REAL execution later names one
        # learns it here. Without this the prediction sits under episode
        # None while its action sits under ep1, and `close-episode task ep1`
        # — which matches (task, episode) exactly — never evaluates it: the
        # bound execution leaks out of the close-out and is silently absent
        # from calibration. A KNOWN action episode is the newer fact, so it
        # fills a MISSING prediction episode; a prediction that already
        # declared one is never overwritten (that is the mismatch path
        # below).
        if candidate.episode_id is None and action.episode_id is not None:
            candidate.episode_id = action.episode_id
            model_info["episode_backfilled_from_action"] = True
        mismatch: Dict[str, Any] = {}
        unknown: Dict[str, Any] = {}
        from or_harness.world_model.conditions import prediction_conditions
        frozen_facts = prediction_conditions(self.store, prediction)
        hypothetical = frozen_facts.get("hypothetical")
        if hypothetical is None:
            unknown["input_state"] = {"predicted": None, "actual": action.source,
                "reason": "the historical input records no recoverable real/hypothetical source"}
        if hypothetical is True or action.source == "hypothetical":
            mismatch["input_state"] = {"hypothetical": True,
                                       "reason": "hypothetical state is not real feedback"}
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
        # mip_gap, seed, ...) is what was predicted. The right-hand side is
        # the config that REALLY TOOK EFFECT — read from the action's
        # ``execution_config`` (the executor's own report plus the solve
        # script's config receipt) — never copied from the candidate. Three
        # outcomes per key:
        #   * the real value is present and EQUAL: observed match;
        #   * present and DIFFERENT: a mismatch (a different configuration
        #     is not this prediction's truth);
        #   * absent: an UNKNOWN — the run did not report that key, so the
        #     predicted value cannot be confirmed. It is never manufactured
        #     into a match by copying the prediction over.
        observed_config = params.get("execution_config") or {}
        observed_values = dict(observed_config.get("values") or {})
        # A key the SCRIPT reported that contradicts an executor-controlled
        # field is not an observation: the executor's own value stands and
        # the run carries a ``conflicts`` record. Using the script's number
        # here would let a solve script make itself match a prediction (e.g.
        # report ``script_timeout_s=1`` while the sandbox really allowed
        # 120), which is exactly how a wrong sample reaches the calibration.
        # An executor-INSTRUCTED value (``executor_configured``:
        # ``solver_timeout_s``) is not an observation either — the framework
        # cannot see whether the solver honoured it, so it is not used as
        # evidence of what the solver did.
        config_conflicts = dict(observed_config.get("conflicts") or {})
        executor_configured = dict(
            observed_config.get("executor_configured") or {})
        for key, value in (candidate.config or {}).items():
            # DUPLICATE FIELD: the solver is carried by the candidate's
            # TOP-LEVEL ``solver`` (compared against the action's own
            # ``params.solver`` above). When a caller ALSO writes it into
            # ``config`` it is the same fact typed twice, and the two
            # spellings need not agree (``pulp_cbc`` in config vs the
            # normalized ``pulp`` at the top level) — that disagreement is
            # NOT a different configuration, so it must not enter the
            # identity comparison. The top-level check stands; config.solver
            # is neither matched nor mistaken for a mismatch. It is never
            # used to INFER or overwrite the top-level solver.
            if key == "solver":
                continue
            # Legacy actions recorded a few keys flat in params; keep
            # reading those so an old record still binds the way it did.
            if key in observed_values:
                actual = observed_values[key]
            else:
                actual = params.get(key)
            if key in config_conflicts:
                # The run's OWN record for this key is the executor's value;
                # the script's contradicting number is reported, never used.
                entry = {
                    "predicted": value,
                    "actual": config_conflicts[key].get("executor"),
                    "reason": ("the solve script reported a value for a key "
                               "the executor controls and the two disagree; "
                               "the executor's own value is kept and the "
                               "script's report is not used as an "
                               "observation"),
                    "script_reported": config_conflicts[key].get(
                        "script_reported"),
                }
                if value == config_conflicts[key].get("executor"):
                    # The prediction matches the EXECUTOR's truth; the script
                    # merely wrote a wrong number of its own.
                    observed_values[key] = config_conflicts[key].get("executor")
                    continue
                mismatch.setdefault("config", {})[key] = entry
                continue
            if key in executor_configured and key not in observed_values:
                # An instruction, not a measurement: unknown, with the
                # reason. It must not count as a confirmed match.
                unknown.setdefault("config", {})[key] = {
                    "predicted": value,
                    "actual": None,
                    "reason": ("the executor INSTRUCTED the script with this "
                               "value, but the framework cannot observe "
                               "whether the solver honoured it: it is not "
                               "evidence of what the solver used. Have the "
                               "script read the effective value back to "
                               "confirm it"),
                }
                continue
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
        # The observations travel with the binding so the caller (and the
        # close-out) can see the difference between "predicted and
        # confirmed", "predicted and contradicted" and "never observed".
        model_info["config_observed"] = {
            key: {"predicted": (candidate.config or {}).get(key),
                  "actual": observed_values.get(key)}
            for key in observed_values if key in (candidate.config or {})}
        model_info["config_unknown"] = sorted(
            (unknown.get("config") or {}).keys()) or None
        # Method identity: the candidate's PLANNED method against the method
        # the run reports it ACTUALLY performed. One-sided evidence (no plan,
        # or an unobserved performance) yields None — not comparable, never a
        # manufactured match. This is recorded, never scored: a plan and a
        # performance are different kinds of statement, and the framework
        # does not pretend a plan was carried out.
        planned_method = params.get("method_planned") or candidate.method
        actual_method = None
        if action.linked_execution_id:
            linked = self.bank.get(action.linked_execution_id)
            if linked is None:
                # The auto-bind inside ``execute`` runs BEFORE the harness
                # decides to record, so the execution is only STAGED at that
                # point. Reading the staged payload keeps the observation
                # available at the moment it is first made; a later explicit
                # ``bind-strategy`` re-reads the recorded row.
                linked = self.bank.get_pending(action.linked_execution_id)
            if linked is not None:
                actual_method = linked.method_actual
        model_info["method_observed"] = compare_methods(planned_method,
                                                        actual_method)
        # Timing: the prediction must predate the action.
        if prediction.trace.created_at > action.started_at:
            mismatch["timing"] = {
                "predicted_at": prediction.trace.created_at,
                "action_started_at": action.started_at,
                "reason": "prediction was generated after the action began"}
        # Effective problem version: the prediction's frozen input digest
        # against the action's own pre-snapshot digest. Both sides come
        # from frozen records; an unestablishable side is an unknown.
        #
        # The comparison is on the PROBLEM IDENTITY, which excludes the
        # post-strategy artifacts (``model``). Comparing the raw task digest
        # would flag the ordinary "predict, then write the model, then
        # execute" sequence as a content change, because the model is
        # written AFTER the strategy is selected — the false mismatch the
        # field run hit. A real change (task text, data, CIR, candidate
        # config) still differs.
        predicted_input = frozen_facts.get("problem_identity_digest")
        actual_digest = None
        if action.pre_snapshot_id:
            pre_snap = self.get_snapshot(action.pre_snapshot_id)
            if pre_snap is not None:
                actual_digest = (pre_snap.problem_state or {}).get(
                    "problem_identity_digest")
        if predicted_input and actual_digest:
            if predicted_input != actual_digest:
                mismatch["effective_input"] = {
                    "predicted": predicted_input,
                    "actual": actual_digest,
                    "reason": ("the problem changed between the prediction "
                               "and the execution: this is a different "
                               "problem input (the post-strategy model "
                               "artifact is excluded from this identity, so "
                               "writing the model is not a change)"),
                }
        elif predicted_input and not actual_digest:
            unknown["effective_input"] = {
                "predicted": predicted_input,
                "actual": None,
                "reason": ("the executed action's pre snapshot records no "
                           "problem identity, so the problem input the "
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
                window.comparable and not mismatch and "input_state" not in unknown)
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
                              and not mismatch and "input_state" not in unknown)
            prediction.trace.comparable = comparable
            prediction.trace.not_comparable_reasons = (
                [] if comparable else
                (["binding mismatch: the executed action differs from the "
                  "predicted candidate"] if mismatch else
                 ["the bound action has no completed linked execution to "
                  "score against"]))
        if "input_state" in unknown:
            prediction.trace.not_comparable_reasons = [unknown["input_state"]["reason"]] + [
                reason for reason in prediction.trace.not_comparable_reasons
                if reason != "the bound action has no completed linked execution to score against"]
        model_info["binding_recorded_at"] = time.time()
        self.strategy_predictions._save(prediction)
        return prediction

    # -- generic prediction payload reading, including retired history -------

    @staticmethod
    def read_prediction_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
        """Read any stored prediction payload, whatever generation it is.

        Returns a dict with ``contract_version`` (as identified, never
        guessed), ``legacy`` (True for an unversioned payload), and either
        ``contract`` (current strategy data or an unchanged retired payload) or ``legacy_view``
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
                "supported": True, "retired": isinstance(contract, dict),
                "contract": copy.deepcopy(contract) if isinstance(contract, dict) else contract.to_dict()}

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

    def _budget_gate(self, task_id: str,
                     episode_id: Optional[str] = None) -> Dict[str, Any]:
        """Whether NEW solving work may start under the declared budget.

        The gate is SCOPE-AWARE, not a bare status check. Only an
        EPISODE-scoped overrun (a cumulative dimension over its limit, or an
        explicitly recorded episode-termination mark) stops the task; an
        ATTEMPT-scoped overrun (``latency_s`` on one attempt) is a failure to
        KEEP and retry around, never a reason to forbid every future attempt.
        ``unconfirmed`` is allowed: "not known to exceed" is not "exceeded".

        Returns ``{"allowed": bool, "reason": str, "view": ...}``. An
        ``allowed=False`` result is a live, structured refusal — callers must
        not create an action, claim a prediction, or touch the sandbox.
        """
        declared = self._load_budget(task_id, episode_id)
        view = self.budget.view(task_id, episode_id, budget=declared)
        # An explicit episode-termination mark (written by the host when it
        # cancelled the same task, e.g. the 100-tool-call stop) is the
        # authoritative signal even when the ledger's numbers do not by
        # themselves read "over the limit" — reaching the cap and refusing
        # the NEXT call leaves the count exactly AT the limit, not above it.
        # A task-level mark (no episode) also stops any episode of that task:
        # the stop is about the TASK's session, not one episode id.
        term = self._episode_termination(task_id, episode_id)
        if term is None and episode_id is not None:
            term = self._episode_termination(task_id, None)
        if term is not None:
            return {"allowed": False, "view": view,
                    "reason": ("episode terminated: "
                               + (term.get("reason") or "budget/host stop")),
                    "termination": term}
        if view.get("episode_exhausted"):
            dims = ", ".join(d["dimension"] for d in view["exceeded_dims"]
                             if d["scope"] == "episode")
            return {"allowed": False, "view": view,
                    "reason": (f"declared budget exhausted on episode-scoped "
                               f"dimension(s): {dims}")}
        return {"allowed": True, "view": view, "termination": None}

    def _episode_termination(self, task_id: str,
                             episode_id: Optional[str]) -> Optional[Dict[str, Any]]:
        """A recorded episode-termination mark, or None.

        Written by :meth:`record_session` when the host confirms a cancel.
        It survives the process that wrote it (stored in meta) so a fresh CLI
        invocation's gate sees the same stop."""
        row = self.store.conn.execute(
            "SELECT value FROM meta WHERE key=?",
            (f"episode_termination|{task_id}|{episode_id or ''}",)).fetchone()
        if row is None:
            return None
        try:
            return self.store.loads(row["value"])
        except Exception:
            return None

    # -- unified action contract ------------------------------------------------

    def begin_action(self, action_type: str, task: Dict[str, Any],
                     episode_id: Optional[str] = None, *,
                     params: Optional[Dict[str, Any]] = None,
                     parent_action_id: Optional[str] = None,
                     source: str = "executed") -> Dict[str, Any]:
        """Begin an action: freeze the PRE snapshot, mint an id, persist
        status=running. The pre snapshot is taken BEFORE the action runs —
        that is the point."""
        pre = self.snapshot(task, episode_id, include_memory=False)
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
        if isinstance(outcome, dict) and any(key in outcome for key in (
                "implementation_facts", "implementation_coverage", "correction_coverage")):
            from or_harness.world_model.implementation_facts import normalize_report
            outcome = normalize_report(self, outcome, task_id=prior.task_id,
                                       episode_id=prior.episode_id,
                                       ended_at=prior.ended_at if replayed else time.time())
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
            post = self.snapshot(task, episode_id, task_progress=progress, include_memory=False,
                                 hypothetical=record.source == "hypothetical")
            self.actions._bind_post_snapshot(action_id, post.snapshot_id)
        if any(key in (outcome or {}) for key in ("implementation_facts", "implementation_coverage", "correction_coverage")):
            self._republish_calibration_if_in_window(prior.task_id, prior.episode_id)
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
                      ended_at: Optional[float] = None,
                      host_usage: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Report an action the OUTER agent performed (model /
        select_strategy / verify / finish_task). The library did not execute
        it — the report is the caller's statement, labelled agent_reported.
        A missing pre snapshot is recorded as missing, never fabricated.
        The report also generates a POST snapshot (merged into the
        episode's progress chain) so the next action inherits what this
        action established."""
        for key, value in (('task', task), ('params', params), ('outcome', outcome)):
            if (key == 'task' or value is not None) and not isinstance(value, dict):
                raise ValueError(f'{key} must be a JSON object')
        if not task.get('task_id'):
            raise ValueError('task must contain task_id')
        from or_harness.world_model.g_signal import require_g_attribution_location
        require_g_attribution_location(
            {"params": params, "outcome": outcome}, path="action", allow_root=False)
        outcome = copy.deepcopy(outcome or {})
        if any(key in outcome for key in ("implementation_facts", "implementation_coverage", "correction_coverage")):
            from or_harness.world_model.implementation_facts import normalize_report
            ended_at = ended_at if ended_at is not None else time.time()
            outcome = normalize_report(self, outcome, task_id=str(task["task_id"]),
                                       episode_id=episode_id, ended_at=ended_at)
        reported_usage = host_usage if host_usage is not None else outcome.get("host_usage")
        if reported_usage is not None:
            from or_harness.world_model.usage import host_report_dimensions
            _, info = host_report_dimensions(reported_usage)
            outcome["host_usage"] = info
        cost_vector = self._cost_from_dict(cost)
        record = self.actions.report_action(
            action_type, str(task.get("task_id", "")), episode_id,
            params=params, outcome=outcome, status=status,
            cost=cost_vector, started_at=started_at, ended_at=ended_at)
        # Post snapshot AFTER the record (and its cost) are persisted, so
        # the budget view reflects the reported spend.
        progress = self._progress_from_outcome(record.action_id,
                                               outcome or {})
        post = self.snapshot(task, episode_id, task_progress=progress, include_memory=False)
        self.actions._bind_post_snapshot(record.action_id, post.snapshot_id)
        if reported_usage is not None or any(key in outcome for key in (
                "implementation_facts", "implementation_coverage", "correction_coverage")):
            self._republish_calibration_if_in_window(str(task.get("task_id", "")), episode_id)
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

    # -- read-only archived outcome predictions --------------------------------


    def get_prediction(self, prediction_id: str) -> Optional[OutcomePrediction]:
        raw = self.store.get_world_model_prediction(prediction_id)
        return OutcomePrediction.from_dict(self.store.loads(raw)) if raw is not None else None

    def predictions_query(self, *, task_id: Optional[str] = None,
                          episode_id: Optional[str] = None
                          ) -> List[OutcomePrediction]:
        return [OutcomePrediction.from_dict(self.store.loads(raw))
                for raw in self.store.world_model_predictions_for(
                    task_id=task_id, episode_id=episode_id)]

    # -- world-model M3: bounded planning ----------------------------------------

    def _candidate_specs(self, task: Dict[str, Any],
                         episode_id: Optional[str],
                         candidates: Optional[Sequence[ActionSpec]],
                         limit: int) -> Tuple[List[ActionSpec], List[str]]:
        """Normalize caller proposals once; bound valid candidates, not raw rows.

        Return correctable refusals alongside the usable specs, without
        mutating caller objects or retaining input state on the harness.
        """
        task_id = str(task.get("task_id", ""))
        specs, rejections = [], []
        for index, candidate in enumerate(candidates or []):
            try:
                raw = candidate.to_dict() if hasattr(candidate, "to_dict") else candidate
                spec = ActionSpec.from_dict(normalize_candidate_payload(raw))
                if spec.task_id and spec.task_id != task_id:
                    raise ValueError(
                        f"candidate identity conflict: task_id {spec.task_id!r} != {task_id!r}")
                if (spec.episode_id is not None and episode_id is not None
                        and spec.episode_id != episode_id):
                    raise ValueError(
                        f"candidate identity conflict: episode_id {spec.episode_id!r} != {episode_id!r}")
                if spec.action_type != "execute_strategy":
                    raise ValueError("only execute_strategy candidates can enter this decision")
                spec.task_id = task_id
                spec.episode_id = episode_id
            except (TypeError, ValueError) as exc:
                rejections.append(f"candidate #{index + 1}: {exc}")
                continue
            if len(specs) < max(0, limit):
                specs.append(spec)
        return specs, rejections

    def plan_next(self, task: Dict[str, Any],
                  episode_id: Optional[str] = None, *,
                  candidates: Optional[Sequence[ActionSpec]] = None,
                  limits: Optional[Any] = None,
                  ) -> Dict[str, Any]:
        """Score one or more proposed candidates in one shared frozen context.

        Each candidate is predicted once. Return and persist the highest-U
        executable prediction, preserving stable ties and existing budget
        limits. Execute that ID directly; actual adoption is recorded by execute.
        Shadow/disabled/unscorable decisions authorize no execution."""
        from or_harness.world_model.planner import PlanLimits, PlanResult
        task = prepare_task_inputs(task, self.home)
        limits = (limits if isinstance(limits, PlanLimits)
                  else PlanLimits.from_dict(limits))
        # ONE protocol: the wm-so/1 macro strategy comparison. The legacy
        # OutcomePrediction rollout (and its horizon-2 imagination tree)
        # was REMOVED rather than left reachable: two protocols behind one
        # entry point meant two prediction logs, two scoring rules and an
        # agent that had to know which one it was in. Re-planning after the
        # real observation replaces the second imagined step.
        if limits.horizon != 1:
            raise ValueError(
                f"planning supports horizon=1 only (got "
                f"{limits.horizon}): it compares macro strategy candidates "
                "once, then you re-plan from the REAL observation of the "
                "chosen step. There is no imagined multi-step rollout — "
                "execute the step and plan again with what actually "
                "happened")
        # Inherit the harness's configured evaluation yardstick unless the
        # caller explicitly overrode it: planning must not silently ignore
        # the alpha/beta/gamma and cost weights the rest of the system
        # scores with (an empty default would have made every cost
        # dimension weight zero).
        if limits.alpha is None:
            limits.alpha = self.alpha
        if limits.beta is None:
            limits.beta = self.beta
        if limits.gamma is None:
            limits.gamma = self.gamma
        if limits.cost_weights is None:
            limits.cost_weights = dict(self.cost_weights)
        specs, rejections = self._candidate_specs(
            task, episode_id, candidates, limits.max_root_candidates)
        result = self._plan_next_strategy_outcome(
            task, episode_id, candidates=specs, limits=limits,
            candidate_rejections=rejections)
        if rejections:
            result["candidate_rejections"] = rejections
        return result
    def _plan_next_strategy_outcome(
            self, task: Dict[str, Any],
            episode_id: Optional[str] = None, *,
            candidates: Sequence[ActionSpec],
            limits: Any,
            candidate_rejections: Sequence[str] = ()) -> Dict[str, Any]:
        """Predict, score and persist one bounded strategy decision.

        U uses comparable G and shared available C/R; frozen H+ independently
        schedules review after an actual attempt. No usable G means no decision."""
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
        identity_refusals = [reason for reason in candidate_rejections
                             if "identity conflict" in reason]
        if identity_refusals:
            plan.truncation_reason = "; ".join(identity_refusals)
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
        specs = candidates
        if not specs:
            plan.status = "no_candidates"
            reason = ("no execute_strategy candidates were supplied: the "
                      "framework does not generate a candidate menu — "
                      "propose the methods you want compared under the "
                      "strategy-outcome protocol")
            if candidate_rejections:
                reason = ("every candidate was REFUSED at input "
                          "normalization (correctable): "
                          + "; ".join(candidate_rejections) + ". " + reason)
            if plan.truncation_reason:
                reason = plan.truncation_reason + ". " + reason
            plan.truncation_reason = reason
            result = plan.to_dict()
            if candidate_rejections:
                result["candidate_rejections"] = candidate_rejections
            return result
        decision = self.actions.begin_action(
            "select_strategy", task_id, episode_id, pre_snapshot=root,
            params={"kind": "plan_next", "protocol": "strategy-outcome",
                    "protocol_version": STRATEGY_OUTCOME_PROTOCOL_VERSION,
                    "benefit_convention": {
                        "kind": getattr(limits, "benefit_kind", None),
                        "metric": getattr(limits, "benefit_metric", None),
                        "unit": getattr(limits, "benefit_unit", None),
                    },
                    "limits": limits.to_dict()})
        plan.decision_action_id = decision.action_id
        started = time.monotonic()
        calls_made = 0
        stop_reason: Optional[str] = None
        # The convention DECLARED for this decision (if any) is pushed into
        # every candidate's request, so the candidates are PREDICTED under
        # the currency the comparison will use. A convention the caller did
        # not declare is left to the candidates' agreement and is resolved
        # after the calls (see below).
        declared_kind = getattr(limits, "benefit_kind", None)
        declared_metric = getattr(limits, "benefit_metric", None)
        declared_convention: Optional[Dict[str, Any]] = None
        if declared_kind and declared_metric:
            from or_harness.world_model.planner import (
                normalize_benefit_convention,
                KNOWN_BENEFIT_CONVENTIONS,
            )
            resolved = normalize_benefit_convention(declared_kind,
                                                    declared_metric)
            if resolved is not None:
                known = KNOWN_BENEFIT_CONVENTIONS[resolved]
                declared_convention = {
                    "kind": resolved[0], "metric": resolved[1],
                    "unit": getattr(limits, "benefit_unit", None)
                    or known["unit"],
                    "observed_from": known["observed_from"],
                    "source": "declared",
                }
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
            # Association is independent of whether this call measured any spend.
            prediction.trace.model_info["decision_action_id"] = decision.action_id
            if prediction.trace.call_cost is None:
                self.strategy_predictions._save(prediction)
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
            from or_harness.world_model.conditions import ContextExpired
            try:
                prediction = self.predict_strategy_outcome(
                    task, CandidateRef.from_action_spec(spec), episode_id,
                    context=plan_context,
                    timeout_s=max(0.001, remaining),
                    benefit_convention=declared_convention)
            except ContextExpired as exc:
                stop_reason = str(exc)
                break
            calls_made += 1
            # The decision already normalized these proposals. Reuse the
            # canonical CandidateRef entry and retain the original diagnostics.
            if spec.normalization_notes:
                notes = prediction.trace.model_info.get("config_normalized", [])
                prediction.trace.model_info["config_normalized"] = list(
                    dict.fromkeys(notes + spec.normalization_notes))
                prediction.notes.extend(note for note in spec.normalization_notes
                                        if note not in prediction.notes)
            # Charge THIS call's known spend immediately (failed calls
            # included: their tokens are real) so the loop's own budget
            # gate sees them before the next call is considered.
            _charge_call(prediction)
            predictions.append((spec, prediction))
        # Post-call deadline check: the LAST call's return may already be
        # over budget (a sync provider cannot be interrupted mid-call), and
        # no further iteration would catch it. That is reported as
        # truncation — never a quiet "ok" against an exhausted budget. The
        # costs already incurred stay recorded.
        if stop_reason is None \
                and time.monotonic() - started > limits.time_budget_s:
            stop_reason = ("planning time budget exhausted after the "
                           f"final model call ({limits.time_budget_s}s); "
                           "the completed calls are kept and their cost "
                           "is recorded")
        plan.model_calls_made = calls_made
        # The ONE benefit convention this decision compares under: declared
        # by the caller, agreed by the candidates, or the build's default.
        # Resolved BEFORE scoring so every candidate is judged against the
        # SAME currency, and recorded so a reader (and the prompt sent to
        # the model) can see which one was in force. A declared convention
        # is also pushed into the prediction requests, so the candidates are
        # PREDICTED under it rather than merely compared under it.
        from or_harness.world_model.planner import resolve_benefit_convention
        convention = resolve_benefit_convention(
            [p for _spec, p in predictions],
            kind=getattr(limits, "benefit_kind", None),
            metric=getattr(limits, "benefit_metric", None))
        plan.benefit_convention = convention
        # Conservative comparison on ONE yardstick.
        only_predictions = [p for _spec, p in predictions]
        pre_scores=score_strategy_outcome_predictions(only_predictions,limits,cost_scales={})
        from or_harness.world_model.cost_signal import decision_cost_scales
        scale_candidates=[p.candidate for p,s in zip(only_predictions,pre_scores) if s.utility is not None]
        scales=decision_cost_scales(plan_context,scale_candidates,limits.cost_scales)
        scores = score_strategy_outcome_predictions(only_predictions, limits,cost_scales=scales)
        comparable = [s for s in scores if s.utility is not None]
        suggested_spec: Optional[ActionSpec] = None
        suggestion_basis = ""
        if comparable and self.plan_mode == "advise":
            best = max(comparable, key=lambda s: s.utility)
            plan.selected_prediction_id=best.prediction_id
            for spec, prediction in predictions:
                if prediction.prediction_id == best.prediction_id:
                    suggested_spec = spec
                    break
            suggestion_basis = (
                f"U={best.utility} over common terms {best.scoring_basis.get('terms')}; "
                f"G={best.benefit_value}; C={best.cost_normalized if best.cost_normalized is not None else 'omitted'}; "
                f"R={best.risk_effective if best.risk_effective is not None else 'omitted'}; "
                f"cost dimensions={best.scoring_basis.get('cost_dimensions')}; "
                f"weights alpha/beta/gamma={limits.alpha}/{limits.beta}/{limits.gamma}; "
                f"{len(comparable)} candidate(s) ranked under {STRATEGY_OUTCOME_PROTOCOL_VERSION} "
                f"from context {plan_context.context_id}; benefit convention "
                f"{convention['kind']}/{convention['metric']} (source={convention['source']}); "
                "omitted terms incur no candidate-specific penalty")
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
        budget_exceeded = False
        if declared:
            final_view = self.budget.view(task_id, episode_id, budget=declared)
            plan.budget_confirmation = final_view["status"]
            budget_exceeded = final_view["status"] == "exceeded"
            if budget_exceeded:
                plan.status = "fallback"
                plan.selected_prediction_id = None
                plan.truncation_reason = (
                    "declared budget exceeded by real consumption "
                    "(including this planning's own spend); the suggestion "
                    "is withheld — report the current best solution instead.")
        if not budget_exceeded:
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
                    "revise candidates or repair the forecast and call plan-next again. "
                    "No execution is authorized; actual prediction spend is recorded.")
        # One set of score rows feeds persistence, CLI/Python output and
        # execution authorization; no parallel approval state is maintained.
        candidate_rows = []
        for (spec, prediction), score in zip(predictions, scores):
            allowed = prediction.prediction_id == plan.selected_prediction_id
            candidate_rows.append({
                "action_spec": spec.to_dict(),
                "candidate_ref": prediction.candidate.to_dict(),
                "prediction_id": prediction.prediction_id,
                "prediction_status": prediction.status,
                "execution_allowed": allowed,
                "execution_refusal_reason": None if allowed else (
                    "not selected by this U decision" if plan.selected_prediction_id
                    else plan.truncation_reason or plan.suggestion_basis),
                "score": score.to_dict(),
                "capability_gain": _capability_gain_view(prediction)})
        plan.selected_score = next((row["score"] for row in candidate_rows
                                    if row["execution_allowed"]), None)
        decision_result = {
            "kind": "plan_next_evaluation", "plan_id": plan.plan_id,
            "protocol": "strategy-outcome",
            "protocol_version": STRATEGY_OUTCOME_PROTOCOL_VERSION,
            "root_snapshot_id": plan.root_snapshot_id,
            "n_candidates": len(candidate_rows),
            "execution_contract": "wm-u-decision/1",
            "implementation_anchor": "decision_completion" if plan.selected_prediction_id else None,
            "selected_prediction_id": plan.selected_prediction_id,
            "selected_score": plan.selected_score,
            "candidates": candidate_rows,
            "suggested": plan.suggested.to_dict() if plan.suggested else None,
            "suggestion_basis": plan.suggestion_basis,
            "suggestion_withheld": self.plan_mode == "shadow" or budget_exceeded,
            "status": plan.status, "truncation_reason": plan.truncation_reason,
            "model_calls_made": int(plan.model_calls_made),
            "planning_cost": copy.deepcopy(plan.planning_cost),
            "budget_confirmation": plan.budget_confirmation,
            "benefit_convention": copy.deepcopy(plan.benefit_convention),
            "prediction_context_id": plan_context.context_id,
            "prediction_context_version": plan_context.version}
        if candidate_rejections:
            decision_result["candidate_rejections"] = list(candidate_rejections)
        self.actions.end_action(decision.action_id, status="completed",
                                outcome=decision_result)
        if plan.selected_prediction_id:
            # Establish the implementation anchor before returning to the
            # solving Agent. Actual implementation is confirmed by evidence.
            selection = self.choose_next(decision.action_id, prediction_id=plan.selected_prediction_id)
            settled = self.actions.get(decision.action_id)
            settled.outcome["implementation_selection_action_id"] = selection["action_id"]
            self.actions._update(settled)
        result = plan.to_dict()
        result.update({key: decision_result[key] for key in (
            "prediction_context_id", "prediction_context_version", "protocol",
            "candidates")})
        return result

    def choose_next(self, decision_action_id: str, *,
                    chosen: Optional[ActionSpec] = None,
                    prediction_id: Optional[str] = None,
                    rejected: bool = False,
                    deviation_note: Optional[str] = None) -> Dict[str, Any]:
        """Read/confirm the scored selection, or reject an existing decision.

        plan_next establishes new implementation anchors internally. Optional
        repeated choices reuse that action; execute alone consumes permission.
        Rejection requires a new decision to solve."""
        from or_harness.world_model.actions import PROGRESS_UPDATES
        decision = self.actions.get(decision_action_id)
        if decision is None:
            raise StorageError(
                f"unknown decision_action_id {decision_action_id!r}")
        if decision.action_type != "select_strategy":
            raise StorageError(
                f"action {decision_action_id!r} is a "
                f"{decision.action_type}, not a select_strategy decision")
        if prediction_id is not None:
            if rejected:
                raise ValueError(
                    "choose-next takes --prediction to name the candidate "
                    "you CHOSE; it cannot be combined with --rejected")
            resolved = self._candidate_from_decision_prediction(
                decision, prediction_id)
            if chosen is not None:
                # Both given: they must name the same candidate, or the
                # recorded choice would be ambiguous.
                if _candidate_reference(chosen.to_dict()) \
                        != _candidate_reference(resolved):
                    raise ValueError(
                        "the --chosen candidate and --prediction "
                        f"{prediction_id!r} name different candidates: "
                        "pass one of them, or make them agree")
            chosen = ActionSpec.from_dict(resolved)
        if not rejected:
            if prediction_id is None:
                wanted=chosen.to_dict() if chosen is not None else {}
                matches=[r for r in (decision.outcome or {}).get('candidates') or []
                         if _candidate_reference(r.get('action_spec'))==_candidate_reference(wanted)]
                if len(matches)!=1:raise ValueError('manual chosen candidate must identify one scored plan-next candidate')
                prediction_id=matches[0]['prediction_id']
            # Re-reading/reaffirming the same choice must not start a second
            # implementation window. It still does not consume an execution.
            existing = [a for a in self.actions.query(task_id=decision.task_id, episode_id=decision.episode_id)
                        if a.parent_action_id == decision_action_id
                        and (a.outcome or {}).get("kind") == "plan_next_choice"
                        and (a.outcome or {}).get("prediction_id") == prediction_id
                        and not (a.outcome or {}).get("rejected")]
            # Reaffirming an adopted choice is a read of its original
            # selection, not permission for another execution.
            self._scored_decision(self.strategy_predictions.get(prediction_id),
                action_id=(existing[0].outcome.get("adopted_by_action_id") if existing else None))
            if existing:
                item = existing[0]
                return {"action_id": item.action_id, "status": item.status,
                        "selected": item.outcome.get("selected"), "rejected": False,
                        "deviation": item.outcome.get("deviation"),
                        "hplus": item.outcome.get("hplus"), "post_snapshot_id": item.post_snapshot_id}
        outcome: Dict[str, Any] = {
            "kind": "plan_next_choice",
            "decision_action_id": decision_action_id,
            "decision_plan_id": (decision.outcome or {}).get("plan_id"),
        }
        if prediction_id is not None:
            outcome["prediction_id"] = str(prediction_id)
            outcome['hplus'] = _capability_gain_view(self.strategy_predictions.get(prediction_id))
        if rejected:
            outcome["rejected"] = True
            with self.store.transaction():
                decision.outcome['selection_rejected']=True
                self.actions._update(decision)
            if deviation_note:
                outcome["note"] = deviation_note
        elif chosen is not None:
            outcome["selected"] = chosen.to_dict()
            suggested = (decision.outcome or {}).get("suggested")
            if suggested and _candidate_reference(suggested) \
                    != _candidate_reference(chosen.to_dict()):
                # A deviation is judged on the candidate REFERENCE (action
                # type, strategy, solver, config) — not on the whole plan
                # JSON (a cosmetic key must not read as a deviation) and not
                # on `strategy_id + solver` alone (the same pair under a
                # different config is a DIFFERENT candidate).
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
        automatic = (decision.outcome or {}).get("implementation_anchor") == "decision_completion" and not rejected
        record = self.actions.report_action(
            "select_strategy", decision.task_id, decision.episode_id,
            params={"decision_action_id": decision_action_id},
            outcome=outcome, status="completed",
            started_at=decision.ended_at if automatic else None,
            ended_at=decision.ended_at if automatic else None)
        if automatic:
            record.source = "executed"
            record.outcome["selection_source"] = "plan_next"
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
        task = PreparedTask((root_snap.problem_state.get("task_input")
                     or root_snap.problem_state.get("task_payload") or {})
                    if root_snap is not None else {})
        task.setdefault("task_id", decision.task_id)
        if "family" not in task and root_snap is not None:
            family = ((root_snap.problem_state.get("profile") or {})
                      .get("family"))
            if family:
                task["family"] = family
        post = self.snapshot(task, decision.episode_id,
                             task_progress=progress, include_memory=False)
        self.actions._bind_post_snapshot(record.action_id,
                                         post.snapshot_id)
        from or_harness.world_model.conditions import release_context
        context_id = (decision.outcome or {}).get("prediction_context_id")
        if context_id:
            release_context(self.store, context_id)
        return {"action_id": record.action_id,
                "status": record.status,
                "selected": outcome.get("selected"),
                "rejected": bool(outcome.get("rejected")),
                "deviation": outcome.get("deviation"),
                'hplus':outcome.get('hplus'),
                "post_snapshot_id": post.snapshot_id}

    def plan_decision(self, decision_action_id: str) -> Optional[Dict[str, Any]]:
        """Read back a planning decision record (the select_strategy action
        plus its outcome payload)."""
        record = self.actions.get(decision_action_id)
        return record.to_dict() if record is not None else None

    def _candidate_from_decision_prediction(
            self, decision: Any, prediction_id: str) -> Dict[str, Any]:
        """The candidate a decision's own candidate list holds for a
        prediction id.

        This is what makes ``choose-next --prediction`` trustworthy: the
        candidate is read from the DECISION's recorded comparison, so a
        prediction id that belongs to a DIFFERENT decision is refused
        instead of silently choosing a candidate this decision never
        compared.
        """
        compared = ((decision.outcome or {}).get("candidates") or [])
        for entry in compared:
            if str(entry.get("prediction_id") or "") == str(prediction_id):
                spec = entry.get("action_spec")
                if isinstance(spec, dict):
                    return spec
        known = [str(e.get("prediction_id")) for e in compared
                 if e.get("prediction_id")]
        raise ValueError(
            f"prediction {prediction_id!r} is not among the candidates "
            f"compared by decision {decision.action_id!r} "
            f"(known: {known or 'none'}): the choice would name a candidate "
            "this decision never evaluated. Use the prediction id "
            "`plan-next` returned for THIS decision, or pass --chosen")

    # -- world-model M4: the offline induction material entry point ----------

    def induction_material(self, *, strategy_id: Optional[str] = None,
                           task_id: Optional[str] = None,
                           limit: Optional[int] = None,
                           cursor: Optional[str] = None,
                           related_top_k: int = 5,
                           view: str = "compact", purpose: str = "modelling",
                           induction_id: Optional[str] = None,episode_id=None,query=None,execution_ids=None
                           ) -> Dict[str, Any]:
        """Organize a BATCH of completed tasks as induction material.

        The ONE material entry point: it reads the Experience Bank directly
        and needs NO candidate and NO sample-count gate, so new, failed,
        cross-cell and cross-method-name material all reach the agent. The
        framework gathers the evidence (the methods actually performed, the
        outcomes, the failures, the costs, the before/after changes, the
        same-task attempt chains, the sources needed to re-read raw material,
        and the existing strategies a reader might extend, merge or revise)
        so the OUTER AGENT can read it, compare and form a strategy in its
        own words. The framework does not summarize the material into a
        strategy, never invokes a model, and emits NO candidate verdict.

        Scope filters (all optional) narrow the batch: ``strategy_id`` /
        ``task_id`` to one strategy or task, ``limit`` to the newest N
        attempts, ``cursor`` to continue with material OLDER than a previous
        batch's ``budget.next_cursor``. The character budget
        (``OR_HARNESS_INDUCTION_MATERIAL_CHARS``) bounds how much travels and
        REPORTS eviction — a page is never taken for the whole bank.

        Read-only: nothing is written and no strategy is formed here.
        """
        from or_harness.world_model.maintenance import (
            build_induction_material,
        )
        return build_induction_material(self, strategy_id=strategy_id,
                                        task_id=task_id, limit=limit,
                                        cursor=cursor,
                                        related_top_k=related_top_k,
                                        view=view, purpose=purpose, induction_id=induction_id,
                                        episode_id=episode_id,query=query,execution_ids=execution_ids)

    def induction_status(self, *, task_id=None, episode_id=None):
        from or_harness.world_model.hplus import jobs, schedule_view, receipt_observation
        items=jobs(self, task_id=task_id, episode_id=episode_id)
        predictions=self.strategy_predictions.query(task_id=task_id,episode_id=episode_id)
        signals=[p.trace.model_info.get('hplus') for p in predictions]
        unobserved=[p.prediction_id for p in predictions if p.trace.model_info.get('hplus') and
                    not any(j['prediction_id']==p.prediction_id for j in items)]
        result=schedule_view(items,unobserved)
        result['signal_counts']={'explicit_1':sum(s is not None and s['decision_status']=='explicit' and s['decision']==1 for s in signals),
            'explicit_0':sum(s is not None and s['decision_status']=='explicit' and s['decision']==0 for s in signals),
            'missing':sum(s is not None and s['decision_status']=='missing' for s in signals),
            'invalid':sum(s is not None and s['decision_status']=='invalid' for s in signals),
            'legacy':sum(s is None for s in signals)}
        observations=[receipt_observation(self,j) for j in items]
        result['observation_counts']={'bound_attempts':len(items),
            'positive':sum(o['label']==1 for o in observations),'negative':sum(o['label']==0 for o in observations),
            'unknown':sum(o['label'] is None for o in observations)}
        return result

    def process_induction_jobs(self, reviewer, *, task_id=None, episode_id=None):
        """Outer runner callback: one material and one submission per pending job.

        Reviewer receives (material, frozen job) and returns induce keyword
        arguments, which may be the compact ``{"submission": {...}}`` payload.
        A crash leaves the same durable job available for retry.
        """
        from or_harness.world_model.hplus import jobs
        results = []
        for job in jobs(self, task_id=task_id, episode_id=episode_id):
            if job['state'] != 'pending':
                continue
            material = self.induction_material(induction_id=job['induction_id'])
            submission = reviewer(copy.deepcopy(material), copy.deepcopy(job))
            if not isinstance(submission, dict):
                raise ValueError('reviewer must return a short submission or an empty review')
            results.append(self.induce(induction_id=job['induction_id'], **submission))
        return {'results': results, **self.induction_status(task_id=task_id, episode_id=episode_id)}

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
               candidates: Optional[Sequence[str]] = None,
               memory_mode: str = "cost-aware",
               include_unverified: bool = False,
               vector_top_k: Optional[int] = None,
               profile: Optional[ProblemProfile] = None) -> Dict[str, Any]:
        """Recall accumulated experience for this task.

        TWO INDEPENDENT CHANNELS, never blended into one number, and never
        one erased by the other:

        - ``recommendations`` / ``available_solver_families`` /
          ``solver_advisories``: the structural channel — the strategies this
          cell has REAL memory about, with their evidence-based scores. It
          requires confirmed ``compatible`` or ``partial`` machine support;
          empty-predicate and unmeasured-condition rules are not structural
          matches and remain reachable through the vector channel.
        - ``vector_recall``: the text-similarity channel (embedding), which
          surfaces memories whose TEXT is close regardless of structural
          cell. Its ``similarity`` is a discovery signal only; it is never a
          quality, cost, or risk estimate, and a cross-cell hit never enters
          the target cell's statistics.

        **No candidate menu.** There is no built-in strategy directory to
        fall back on: when memory holds nothing for this cell,
        ``recommendations`` is EMPTY and ``recommendations_basis`` says so in
        words. An empty structural channel does NOT suppress the text
        channel — a semantic hit is still returned, and the two facts are
        reported side by side.

        ``candidates`` is the CALLER's proposal set (the methods the outer
        agent is considering). Supplying it restricts ``recommendations`` to
        those ids and reports the rest under
        ``candidates_without_evidence`` — the framework never invents
        evidence for them, never ranks them, and never blocks their
        execution (``execute``/``predict_cost`` accept any id).

        ``profile`` lets a caller that has ALREADY profiled this task
        (``self.profile(task)``, e.g. a host that ran ``profile`` and
        ``recall`` in one request) pass the object in, so the task's CIR and
        structure are parsed ONCE instead of twice. The caller OWNS the
        consistency and the profile must come from ``self.profile(task)`` for
        THIS task; a profile whose ``problem_id`` is a different task is
        refused (a cheap check that does not re-parse the CIR), never
        silently used. A same-id profile built from a DIFFERENT effective CIR
        is the caller's error to avoid — like a reused prediction context,
        the profile must describe the same frozen input.

        When the text channel cannot run (no backend, no task text, missing
        or model-incompatible index, backend error) the structural result is
        returned BY ITSELF with ``degraded`` explaining why, rather than
        silently looking like a text search that found nothing. A retrieval
        FAILURE and a retrieval that ran with NO HITS are different facts and
        are labelled differently.

        READ-ONLY: the query text is embedded in memory and never written;
        no index item is created and no migration is triggered.
        """
        if profile is None:
            profile = self.profile(task)
        elif str(getattr(profile, "problem_id", "")) \
                != str(task.get("task_id", "")):
            # The caller supplies a profile it computed for this task; a
            # profile of a DIFFERENT task would key recall on the wrong
            # structural cell. Refused as a correctable error, never a quiet
            # fallback — and the check is a cheap id comparison, so the whole
            # point (one CIR parse) is preserved.
            raise ValueError(
                f"the supplied profile is for problem "
                f"{getattr(profile, 'problem_id', None)!r}, not this task "
                f"({task.get('task_id')!r}): recall would search the wrong "
                "structural cell. Pass the profile computed from THIS task "
                "(self.profile(task)), or omit profile= to build one here")
        proposed = (None if candidates is None
                    else sorted({str(c) for c in candidates}))
        recalled = self.selector.recall(profile, top=top, exclude=exclude,
                                        candidates=proposed,
                                        memory_mode=memory_mode,
                                        include_unverified=include_unverified,
                                        with_meta=True)
        recs = recalled["recall"]
        solvers = available_families()
        known = self.selector.candidate_ids(
            profile, memory_mode=memory_mode,
            include_unverified=include_unverified)
        result = {
            "profile": profile.to_dict(),
            # The task VERSION this result was produced for. Recorded so a
            # caller that wants to reuse the result as frozen prediction
            # input can PROVE it belongs to the current task version —
            # an unversioned result must not be dressed up as aligned
            # evidence (see world_model.context.retrieval_reuse_problems).
            "task_digest": task_text_digest(task),
            "recommendations": [r.to_dict() for r in recs],
            # RECALL DISCOVERS, IT DOES NOT DECIDE: the rows are in
            # relevance order and any `top` cap is a MATERIAL BUDGET whose
            # omission is reported here, never a utility cut.
            "recall_budget": {
                "n_matched": recalled["n_matched"],
                "returned": recalled["returned"],
                "material_budget": recalled["material_budget"],
                "omitted": recalled["omitted"],
                "omitted_without_machine_support":
                    recalled.get("omitted_without_machine_support", 0),
                "note": recalled["note"],
            },
            "recommendations_basis": self._recall_basis(
                recs, known, proposed, memory_mode),
            "available_solver_families": solvers,
            "solver_advisories": solver_advisories(self.bank),
            # Unverified claims travel in their own section. The structural
            # channel above is keyed on the strategy ids MEMORY holds and its
            # entry filter (`is_publishable`) speaks about the admitted
            # claim; a claim-only entry names a free-form subject rather than
            # a strategy and would not survive the recommendation filter.
            # This section carries the NOT-YET-PUBLISHED claims (with their
            # verification state intact) so a caller can see what is held
            # rather than admitted.
            "held_claims": self._held_claims(profile, include_unverified),
        }
        omitted_structural = recalled.get(
            "omitted_without_machine_support", 0)
        if omitted_structural:
            basis = result["recommendations_basis"]
            basis["omitted_without_machine_support"] = omitted_structural
            if not recs:
                basis["reason"] = (
                    f"{omitted_structural} knowledge entr"
                    f"{'y has' if omitted_structural == 1 else 'ies have'} "
                    "no confirmed machine structural support (empty or "
                    "unmeasured predicates). They are not ranked in the "
                    "structural channel; read vector_recall for semantic "
                    "discovery and assess `when` before adoption.")
        if proposed is not None:
            result["candidates_proposed"] = list(proposed)
            result["candidates_without_evidence"] = [
                sid for sid in proposed if sid not in known]
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
        # The strategic channel is the FIRST place to look. A recommendation
        # backed by an ADMITTED entry (`evidence == "strategic_entry"`) is
        # knowledge that applies here; a `conditional_stats` recommendation
        # is only a recount over the evidence bank. When NO admitted entry
        # applies and the text channel found comparable EXECUTIONS, their
        # recorded methods are surfaced here as candidates to consider — the
        # evidence channel standing in for the (empty) knowledge channel,
        # exactly as the two-layer split intends. They are suggestions only:
        # the caller still predicts, executes and checks whichever it picks.
        suggestions = (result.get("vector_recall") or {}).get(
            "suggested_candidates") or []
        applied_knowledge = [r for r in result["recommendations"]
                             if r.get("evidence") == "strategic_entry"]
        if not applied_knowledge and suggestions:
            result["evidence_candidates"] = suggestions
            result["evidence_candidates_note"] = (
                "no admitted strategic knowledge applies here; these are "
                "methods actually recorded on comparable past executions, "
                "offered as candidates — not ranked, not vouched for, and not "
                "a replacement for predicting and checking the choice")
        return result

    @staticmethod
    def _recall_basis(recs: List[Any], known: List[str],
                      proposed: Optional[List[str]],
                      memory_mode: str) -> Dict[str, Any]:
        """WHY ``recommendations`` holds what it holds (always present).

        An empty list is a real answer that must be distinguishable from a
        failure and from a filter the caller applied. The three cases named
        here — no memory at all, memory outside the caller's proposal set,
        and a memory-free mode — are the only reasons this build can produce
        an empty structural channel.
        """
        evidence_by_kind = sorted({r.evidence for r in recs})
        if memory_mode == "none":
            reason = ("memory_mode='none': memory was deliberately not "
                      "consulted, so nothing is recalled. This is a choice, "
                      "not an absence of experience")
        elif recs:
            reason = ("recalled from real memory: "
                      f"{len(recs)} strategy/strategies with "
                      f"{'/'.join(evidence_by_kind)} evidence in this "
                      "structural cell")
        elif known:
            reason = ("this cell HAS memory for "
                      f"{len(known)} strategy/strategies "
                      f"({', '.join(known)}), but none of them survived the "
                      "filters you applied (exclude / candidates / top)")
        else:
            reason = ("NO MEMORY for this structural cell: nothing has been "
                      "executed or induced here. The framework does not "
                      "supply a candidate menu — propose the methods you "
                      "want to try (see `orx execute` / `orx plan-next "
                      "--candidates`), and they will be predicted, executed "
                      "and recorded")
        return {
            "n_recommendations": len(recs),
            "evidence_kinds": evidence_by_kind,
            "memory_mode": memory_mode,
            "candidates_with_memory": list(known),
            "candidates_proposed": (None if proposed is None
                                    else list(proposed)),
            "reason": reason,
        }

    def _held_claims(self, profile, include_unverified: bool
                     ) -> List[Dict[str, Any]]:
        """The applicable knowledge CLAIMS NOT yet published as knowledge.

        Published claims reach a caller through the ordinary recommendation
        path (their entry is an admitted entry). What does NOT survive that
        path is a HELD claim: either not yet verified, or a claim-only entry
        naming a free-form subject instead of a strategy. This section
        carries those, read through ``verified_knowledge_view`` so the SAME
        layering rules apply everywhere, with each claim's own verification
        state and the ``newer_evidence_since_verification`` annotation.

        Empty when nothing is held (the common case) — a caller that wants to
        see candidates-in-waiting sets ``include_unverified``.
        """
        from or_harness.world_model.state import verified_knowledge_view
        layers = verified_knowledge_view(profile, self.sbank)
        out: List[Dict[str, Any]] = []
        for layer in ("legacy_unknown", "unverified"):
            if layer == "unverified" and not include_unverified:
                continue
            for ref in layers.get(layer) or []:
                claim = ref.get("claim")
                if not claim:
                    continue
                item = {
                    "layer": layer,
                    "entry_id": ref.get("entry_id"),
                    "strategy_id": ref.get("strategy_id"),
                    # Whether this entry is knowledge about a CONDITION
                    # (no statistical claim) or a strategy's claim.
                    "claim_only": bool(ref.get("support_n", 0) == 0),
                    "text": claim.get("text"),
                    "kind": claim.get("kind"),
                    "conditions": copy.deepcopy(claim.get("conditions") or {}),
                    "evidence": copy.deepcopy(claim.get("evidence") or []),
                    "tasks": list(claim.get("tasks") or []),
                    "verification_state": ref.get("verification_state"),
                    "published": False,
                    "newer_evidence_since_verification":
                        self._newer_evidence_count(ref),
                }
                out.append(item)
        return out

    def _newer_evidence_count(self, ref: Dict[str, Any]) -> Optional[int]:
        """Matching executions recorded AFTER this claim's verdict.

        A visibility annotation, never a lifecycle state: a frozen batch of
        evidence remains a true historical fact, and this count simply tells
        a reader that the world has moved on since the check ran. The verdict
        time is read from the ENTRY's verification block (the claim shares the
        entry's one verdict)."""
        entry = self.sbank.get(str(ref.get("entry_id")))
        if entry is None:
            return None
        block = (entry.verification or {})
        verified_at = block.get("verified_at")
        if not verified_at:
            return None
        conditions = ((entry.claim or {}).get("conditions") or {}).get(
            "predicates") or {}
        count = 0
        for record in self.bank.all():
            if record.source != "executed":
                continue
            if record.created_at <= float(verified_at):
                continue
            if conditions and not profile_matches(record.profile_snapshot,
                                                  conditions):
                continue
            count += 1
        return count

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

        ``strategy_id`` is NOT checked against any directory: the outer
        agent may name a method the framework has never heard of, and the
        honest answer is rung 3 (``source="unknown"``), not a refusal. A
        missing cost basis never blocks a legitimate execution.
        """
        profile = self.profile(task)
        # Published knowledge only: a candidate whose admission verification
        # is missing a verdict must not act as a verified entry prediction
        # either. Gating recall alone would leave this second door open —
        # the snapshot then falls back to the statistics ladder below, which
        # is exactly what "we have no verified knowledge yet" should look
        # like.
        entry = next((e for e in self.sbank.matching(
                          profile, compatible_only=True)
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
                expected_cost=self._complete_cost(cell),
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
    def _complete_cost(cell) -> CostVector:
        """A stats-derived expected cost carrying only COMPLETE dimensions.

        The same completeness bar induction uses: a dimension measured on
        only some of the cell's records must not be published as an
        expectation. An incomplete dimension keeps a placeholder zero and
        stays OUT of the mask, so consumers see unknown rather than a
        partial-subset mean presented as a measurement."""
        complete = cell.complete_dims()
        return CostVector(
            **{d: (float(getattr(cell.mean_cost, d)) if d in complete else 0.0)
               for d in COST_DIMENSIONS},
            measured=set(complete))

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
            # A known historical identity cannot be replaced with another
            # version merely because its text is unavailable.
            return None
        from or_harness.world_model.conditions import latest_snapshot
        latest = latest_snapshot(self.store, record.task_id, episode_scoped=False,
                                 at=record.created_at)
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

    def _scored_decision(self, prediction, *, action_id=None):
        """Resolve saved U permission; manual choices cannot grant permission."""
        import math
        from or_harness.world_model.conditions import digest

        decision_id = (prediction.trace.model_info or {}).get("decision_action_id")
        decision = self.actions.get(decision_id) if decision_id else None
        outcome = (decision.outcome or {}) if decision else {}
        if (not decision or decision.action_type != "select_strategy"
                or decision.status != "completed"
                or (decision.params or {}).get("kind") != "plan_next"
                or outcome.get("execution_contract") != "wm-u-decision/1"):
            raise ValueError(
                "prediction has no scored plan-next decision; "
                "independent Python predictions are inspection only")
        if (decision.task_id != prediction.candidate.task_id
                or decision.episode_id != prediction.candidate.episode_id):
            raise ValueError(
                "prediction and its U decision have conflicting task/episode ownership")
        rows = [row for row in outcome.get("candidates") or []
                if row.get("prediction_id") == prediction.prediction_id]
        if len(rows) != 1:
            raise ValueError("prediction is not uniquely among this decision candidates")
        row = rows[0]
        score = row.get("score") or {}
        utility = score.get("utility")
        if (not isinstance(utility, (int, float)) or isinstance(utility, bool)
                or not math.isfinite(utility)
                or score.get("prediction_id") != prediction.prediction_id):
            raise ValueError(
                "candidate has no available finite U; "
                "call plan-next with a usable G prediction")
        if (outcome.get("selected_prediction_id") != prediction.prediction_id
                or not row.get("execution_allowed")
                or outcome.get("suggestion_withheld")
                or outcome.get("selection_rejected")):
            raise ValueError(
                "candidate is not selected or is not approved for execution "
                "by this U decision")
        if digest(row.get("candidate_ref")) != digest(prediction.candidate.to_dict()):
            raise ValueError(
                "candidate method, solver, config or scope changed since its U decision")
        adopted = outcome.get("adopted_by_action_id")
        if adopted is not None and adopted != action_id:
            raise ValueError(
                "this U decision is already bound/consumed by execution action "
                + str(adopted))
        return decision, row

    def _resolve_execution_candidate(
            self, prediction_id: str, task: Dict[str, Any],
            episode_id: Optional[str], *,
            strategy_id: Optional[str], solver: Optional[str],
            code_path: str, workspace: str,method=None,config=None) -> CandidateRef:
        """Check a prediction-driven execution BEFORE anything runs.

        Everything here is STATIC (no solver, no model call, no action):
        the goal is that a path error, an identity conflict or a claim on a
        prediction already consumed by another attempt is reported to the
        caller before the prediction is spent and before a single second of
        sandbox time is used.

        Returns the candidate whose identity the execution will run under.
        Raises ValueError with the specific reason otherwise — a refusal is
        never a silent substitution.
        """
        prediction = self.strategy_predictions.get(prediction_id)
        if prediction is None:
            raise ValueError(
                f"unknown prediction_id {prediction_id!r}: pass the id "
                "plan-next selected for this decision")
        candidate = prediction.candidate
        self._scored_decision(prediction)
        if str(task.get('task_id',''))!=candidate.task_id:
            raise ValueError('execution task_id contradicts the scored prediction task_id')
        if method is not None and normalize_method(method)!=(normalize_method(candidate.method) or None):
            raise ValueError('explicit method contradicts the scored candidate method')
        if config is not None and config!=candidate.config:
            raise ValueError('explicit config contradicts the scored candidate configuration')
        info = prediction.trace.model_info or {}
        # (a) The prediction must be USABLE. A failed/invalid prediction has
        # no candidate content to test, and a non-execute candidate is not
        # something this command can run.
        if prediction.status != "valid":
            raise ValueError(
                f"prediction {prediction_id!r} has status "
                f"{prediction.status!r}: a non-valid prediction is not a "
                "candidate to execute. Re-predict the candidate (and read "
                "`failure.kind` if it failed) before running it")
        if candidate.action_type != "execute_strategy":
            raise ValueError(
                f"prediction {prediction_id!r} is about a "
                f"{candidate.action_type!r} action: `execute` runs an "
                "execute_strategy candidate only")
        # (b) Problem identity: the prediction was made for a specific
        # problem. Writing the model afterwards is NOT a change (the
        # identity excludes post-strategy artifacts), but a different task
        # text, data or CIR is — reusing the prediction across a changed
        # problem would score an unrelated result against it. Both sides
        # come from frozen records: the prediction recorded the digest it
        # was made under, and the task in hand is digests the same way.
        from or_harness.world_model.conditions import prediction_conditions
        predicted_digest = prediction_conditions(self.store, prediction).get("problem_identity_digest")
        if predicted_digest:
            current_digest = problem_identity_version(task)
            if current_digest != predicted_digest:
                raise ValueError(
                    "the problem changed since this prediction was made "
                    f"(problem identity {current_digest} != "
                    f"{predicted_digest}): this is a different problem "
                    "input, so the prediction is not about it. Post-strategy "
                    "artifacts such as the `model` field do NOT count as a "
                    "change — a real change (task text, data, parameters or "
                    "CIR) does. Predict the candidate again for the current "
                    "problem")
        # (c) One attempt per candidate: a prediction already claimed by a
        # real execution must not be silently re-used — that would put two
        # executions under one forecast and double-count the sample. Two
        # sources are consulted, because the claim is written to BOTH sides:
        # the prediction carries the forward link, and the action carries the
        # prediction id. Reading only the prediction would miss a claim whose
        # prediction write had not landed, and reading only the action would
        # miss a legacy bind. The check is repeated inside the claim
        # transaction, which is what actually prevents a race.
        bound = info.get("bound_action_id")
        if bound:
            raise ValueError(
                f"prediction {prediction_id!r} is already bound to action "
                f"{bound!r}: a prediction corresponds to ONE real attempt. "
                "Predict the corrected candidate again if you re-solve")
        claiming_action = self._prediction_claimed_by_action(prediction_id)
        if claiming_action:
            raise ValueError(
                f"prediction {prediction_id!r} is already claimed by action "
                f"{claiming_action!r}: a prediction corresponds to ONE real "
                "attempt. Predict the corrected candidate again if you "
                "re-solve")
        # (d) Explicit arguments must AGREE with the candidate. A caller
        # that re-types strategy/solver and gets it wrong is told now, not
        # after the run (where it would be a binding mismatch and a wasted
        # attempt).
        conflicts: List[str] = []
        if strategy_id is not None and str(strategy_id) != "" \
                and candidate.strategy_id is not None \
                and str(strategy_id) != candidate.strategy_id:
            conflicts.append(
                f"strategy_id: argument {strategy_id!r} != candidate "
                f"{candidate.strategy_id!r}")
        if solver is not None and str(solver) != "" \
                and candidate.solver is not None \
                and str(solver) != candidate.solver:
            conflicts.append(
                f"solver: argument {solver!r} != candidate {candidate.solver!r}")
        if episode_id is not None and str(episode_id) != candidate.episode_id:
            conflicts.append(
                f"episode_id: argument {episode_id!r} != candidate "
                f"{candidate.episode_id!r}")
        if conflicts:
            raise ValueError(
                "the explicit arguments contradict the prediction's "
                "candidate: " + "; ".join(conflicts)
                + ". Omit them (they are taken from the candidate) or pass "
                "the candidate's own values")
        # (e) Static path/scope error: the solve script must live inside the
        # workspace (the executor's own policy), checked here so a mistyped
        # path does not first consume the prediction.
        if not str(code_path or "").strip():
            raise ValueError("code_path is required (a solve.py inside the "
                             "workspace)")
        if not str(workspace or "").strip():
            raise ValueError("workspace is required")
        code = Path(code_path).resolve()
        work = Path(workspace).resolve()
        if work not in code.parents and code != work:
            raise ValueError(
                f"solve script {str(code)!r} is outside its workspace "
                f"{str(work)!r}: the executor refuses a script that could "
                "read or write outside it, and the prediction is not spent "
                "on a run that would be rejected")
        return candidate

    def _claim_prediction_for_action(self, prediction_id: str,
                                     action_id: str) -> None:
        """Claim a prediction for an action, ATOMICALLY or not at all.

        Runs in ONE locked transaction (``store.locked()`` is re-entrant on
        the thread lock, so the nested ``_save``/``_update`` calls cannot
        deadlock): the prediction's forward link and the action's
        ``prediction_id`` parameter are written together. Ordering is chose
        so that a failure can never leave a HALF association that a later
        call would read as a free prediction:

        1. the ACTION is stamped first — the claim that blocks a second
           execution lives on the action, and every reader scans actions;
        2. the prediction's forward link is written second.

        Any exception rolls the whole transaction back and is PROPAGATED:
        the caller must not start the execution, because an unrecorded
        association means the attempt could not be attributed. The check
        that the prediction is unclaimed happens inside the same lock, so
        two writers cannot both see it free.

        A newly created action has no prediction claim. The transaction
        selects one winner across competing actions; refused contenders
        cannot leave a second adoption or start an executor.
        """
        store = self.strategy_predictions.store
        with store.transaction():
            action = self.actions.get(action_id)
            if action is None:
                raise StorageError(f"unknown action_id {action_id!r}")
            existing = (action.params or {}).get("prediction_id")
            if existing not in (None, str(prediction_id)):
                raise StorageError(
                    f"action {action_id!r} already claims prediction "
                    f"{existing!r}: an action corresponds to one prediction")
            prediction = self.strategy_predictions.get(prediction_id)
            if prediction is None:
                raise StorageError(
                    f"prediction {prediction_id!r} disappeared while claiming "
                    "it for this action")
            info = prediction.trace.model_info
            decision,row=self._scored_decision(prediction,action_id=action_id)
            other=self._prediction_claimed_by_action(prediction_id)
            if other not in (None,action_id):raise ValueError('prediction is already claimed by another real attempt')
            bound = info.get("bound_action_id")
            if bound not in (None, str(action_id)):
                raise StorageError(
                    f"prediction {prediction_id!r} is already claimed by "
                    f"action {bound!r}: a prediction corresponds to ONE real "
                    "attempt")
            # (1) the action's claim, then (2) the prediction's forward link.
            # Reuse a compatible choice intent; otherwise record one automatic
            # choice with the existing entry point. Only a real execute claims it.
            choices=[a for a in self.actions.query(task_id=action.task_id,episode_id=action.episode_id,action_type='select_strategy')
                     if a.parent_action_id==decision.action_id and (a.outcome or {}).get('prediction_id')==prediction_id
                     and not (a.outcome or {}).get('adopted_by_action_id')]
            if choices:choice=choices[-1]
            else:
                chosen=self.choose_next(decision.action_id,prediction_id=prediction_id)
                choice=self.actions.get(chosen['action_id'])
            choice.outcome.update(adopted_by_action_id=action_id,adopted_at=time.time(),adoption_source='execute')
            self.actions._update(choice)
            action.params.update(decision_action_id=decision.action_id,decision_choice_action_id=choice.action_id,
                                 prediction_id=prediction_id,prediction_bound_at=time.time(),
                                 scored_adoption={'prediction_id':prediction_id,'utility':row['score']['utility'],
                                                 'score':copy.deepcopy(row['score'])})
            decision.outcome['adopted_by_action_id']=action_id
            self.actions._update(decision)
            self.actions._update(action)
            info.setdefault("bound_action_id", str(action_id))
            info["bound_action_id_before_execution"] = str(action_id)
            info["association_phase"] = "linked_before_execution"
            self.strategy_predictions._save(prediction)

    def _prediction_claimed_by_action(self, prediction_id: str
                                      ) -> Optional[str]:
        """The OTHER action that already claims a prediction, or None.

        Scans the action log for any action carrying this prediction id in
        its params (the pre-execution link), EXCLUDING nothing — the caller
        passes its own new action id separately and compares. Used as the
        reader-side check that makes the claim survive a race: the writer
        stamps its own action first, so a loser of the race sees the
        winner's stamp here even though the prediction payload may still
        read as free.
        """
        wanted = str(prediction_id)
        for action in self.actions.query():
            if (action.params or {}).get("prediction_id") == wanted:
                return action.action_id
        return None

    def _release_prediction_claim(self, prediction_id: str,
                                  action_id: str, *, reason: str) -> None:
        """Compatibility recovery for an old claim with no scored adoption.

        Current scored attempts retain their claim, including failed attempts.
        A correction or retry is proposed and scored as a fresh candidate."""
        action=self.actions.get(action_id)
        if action and (action.params or {}).get('scored_adoption'):
            return  # a started attempt, including failure, consumes this decision
        try:
            self.actions.amend_action_params(
                action_id, prediction_id=None, prediction_released_at=None)
            self.actions.amend_action_params(
                action_id, prediction_released_at=time.time(),
                prediction_release_reason=str(reason))
        except Exception:  # noqa: BLE001 - never mask the caller's error
            return
        try:
            prediction = self.strategy_predictions.get(prediction_id)
            if prediction is None:
                return
            info = prediction.trace.model_info
            if info.get("bound_action_id") != action_id:
                return
            info.pop("bound_action_id", None)
            info.pop("bound_action_id_before_execution", None)
            info["association_phase"] = "released_no_execution"
            info["claim_released"] = {
                "action_id": action_id,
                "reason": str(reason),
                "released_at": time.time(),
            }
            self.strategy_predictions._save(prediction)
        except Exception:  # noqa: BLE001 - never mask the caller's error
            return

    def _observe_retry_increment(self, record, action, *, task_id, episode_id):
        """Measure this attempt's 0/1 retry increment across methods/solvers."""
        from or_harness.world_model.contracts import RETRIES_INCREMENT_VERSION
        from or_harness.world_model.cost_signal import execution_attempts
        from or_harness.world_model.episode_closeout import _episode_executions
        record.cost.measured=record.cost.measured_dims()-{'retries'}
        record.execution_features.get('cost_provenance',{}).pop('retries',None)
        if record.execution_features.get('executed') is False:
            return
        actions=self.actions.query(task_id=str(task_id))
        records=_episode_executions(self,str(task_id),episode_id)
        earlier=[a for a in actions if a.action_id!=action.action_id and
                 (a.started_at is None or a.started_at<=action.started_at)]
        prior,gaps=execution_attempts(earlier,records,task_id=str(task_id),episode_id=episode_id)
        if gaps:
            record.execution_features['retries_proof']='retry increment unknown: '+gaps[0]['reason']
            return
        record.cost.retries=1.0 if prior else 0.0
        record.cost.mark_measured('retries')
        record.execution_features['retries_proof']=(
            'an earlier execute_strategy attempt of this (task, episode) exists: retries=1 is observed'
            if prior else 'no earlier execute_strategy attempt of this (task, episode) exists: retries=0 is observed')
        record.execution_features.setdefault('cost_provenance',{})['retries']={
            'source':'agent_observed','basis':'per_attempt_retry_increment',
            'observation_rule_version':RETRIES_INCREMENT_VERSION,'action_id':action.action_id,
            'prior_execution_ids':[a.linked_execution_id for a in prior]}

    def _stage_interrupted_attempt(self, action, exc: BaseException,
                                   *, profile, strategy_id: str,
                                   solver: Optional[str],
                                   method_planned: Optional[Dict[str, Any]],
                                   task_id: str, episode_id: Optional[str],
                                   task_text_ver: Optional[str],
                                   prediction_id: Optional[str] = None) -> None:
        """Stage a REAL execution fact for a run that raised before producing
        one.

        A started attempt is a real event even when the executor blew up: the
        spend (the latency up to the exception) and the failure really
        happened, and the user's rule is that "已经发生的尝试成本和实现失败仍应
        尽可能保留". Recording an HONEST failure fact (``status="error"``,
        ``feasible=False``, a ``FailureRecord``) keeps the failure visible to
        the calibration: the retry is a SEPARATE attempt, the failure is its
        own observation, and nothing is fabricated (no objective, no gap, no
        solver runtime were observed).

        Best-effort: a staging failure must never mask the original
        exception.
        """
        try:
            from or_harness.strategy.triggers import classify_failure
            action = self.actions.get(action.action_id) or action
            record = ExecutionRecord(
                execution_id=ExecutionRecord.new_id(),
                task_id=str(task_id),
                strategy_id=str(strategy_id),
                profile_snapshot=profile,
                quality={"status": "error", "feasible": False},
                cost=CostVector(measured=set()),
                solver={"name": str(solver) if solver else None},
                source="executed",
            )
            # The attempt really consumed wall-clock time before it blew up.
            # That latency is MEASURABLE (the action's own start stamp is on
            # record), so it is recorded as a real measurement — never left
            # as an unmeasured zero, which would read as "this attempt cost
            # nothing". Only latency is observable this way; no token count,
            # tool call or solver runtime is invented.
            try:
                elapsed_s = max(0.0, time.time() - float(action.started_at))
                record.cost.latency_s = round(elapsed_s, 6)
                record.cost.mark_measured("latency_s")
                record.execution_features["interrupted_latency_provenance"] = (
                    "wall-clock from the action's started_at to the "
                    "executor's exception: a real measurement of the time "
                    "this attempt consumed, not a placeholder")
            except Exception:  # noqa: BLE001 - never mask the original error
                pass
            record.quality["error"] = f"{type(exc).__name__}: {exc}"
            record.failures.append(FailureRecord(
                attempt=1, error=f"{type(exc).__name__}: {exc}",
                recovery_action=None))
            record.failures[0].error_class = classify_failure(record)
            record.execution_features["interrupted_before_record"] = (
                "the executor raised before producing a record; this fact is "
                "the attempt that really started and really failed (no "
                "objective/gap/runtime were observed)")
            record.execution_features["baseline_measurement"] = (
                "staged before the run began: latency reflects the real time "
                "the attempt consumed up to the exception")
            for key in ("artifacts", "input_resources"):
                if (action.params or {}).get(key):
                    record.execution_features[key] = copy.deepcopy(action.params[key])
            # A scored attempt stays consumed. Its failed fact retains the
            # original prediction; a repaired attempt uses a fresh decision.
            if prediction_id:
                record.execution_features["predicted_for"] = str(prediction_id)
                record.execution_features["prediction_id"] = str(prediction_id)
            if method_planned is not None:
                record.method_planned = method_planned
            if task_text_ver is not None:
                record.task_text_digest = task_text_ver
            record.action_id = action.action_id
            self._observe_retry_increment(record,action,task_id=task_id,episode_id=episode_id)
            self.bank.stage_pending(record)
            self.actions.end_action(
                action.action_id, status="failed",
                outcome={"error": f"{type(exc).__name__}: {exc}",
                         "phase": "executor",
                         "execution_status": "error",
                         "interrupted": True},
                linked_execution_id=record.execution_id, rollup="reference")
        except Exception:  # noqa: BLE001 - never mask the original error
            return

    def execute(self, task: Dict[str, Any], strategy_id: Optional[str] = None,
                code_path: str = "", workspace: str = "", *, solver: Optional[str] = None,
                episode_id: Optional[str] = None,
                prediction_id: Optional[str] = None,
                method: Optional[Dict[str, Any]] = None,
                config: Optional[Dict[str, Any]] = None,
                adapted_from: Optional[Sequence[str]] = None,
                adaptation: Optional[str] = None,
                used_entry_ids: Optional[Sequence[Any]] = None
                ) -> ExecutionRecord:
        """Execute the selected finite-U prediction from plan-next.

        Resolve score, permission, task/episode, method, solver, config and
        unused claim before launch. Atomically record the actual adoption and
        prediction binding before running. Every real attempt, including a
        staged failure, consumes its prediction. Repair needs a fresh plan.
        The result remains staged: check the original task and record it."""
        # ---- (1) RESOLVE THE CANDIDATE BEFORE ANYTHING RUNS ----------------
        # Static errors (an unknown prediction, a wrong task identity, a
        # conflict with an explicit argument, a candidate already claimed)
        # are refused HERE, before an action exists and before the sandbox
        # is touched — they must not consume the prediction or any budget.
        candidate = None
        if not prediction_id:
            raise ValueError('execute requires the selected scored prediction from plan-next; bare strategy/solver and independent forecasts cannot authorize execution')
        forecast = self.strategy_predictions.get(prediction_id)
        if forecast is not None:
            task = execution_task_inputs(task, (forecast.trace.model_info or {}).get("input_resources") or [])
        if prediction_id:
            candidate = self._resolve_execution_candidate(
                prediction_id, task, episode_id,
                strategy_id=strategy_id, solver=solver,
                code_path=code_path, workspace=workspace,method=method,config=config)
            strategy_id = candidate.strategy_id
            solver = candidate.solver or solver
            episode_id = candidate.episode_id
        # ---- (1b) THE BUDGET GATE -----------------------------------------
        # A stopped task must not start NEW solving work. Only an
        # EPISODE-scoped overrun (or a recorded host cancel) refuses here:
        # an attempt-level overrun is a failure to keep and retry around.
        # The refusal happens BEFORE the action exists, so it consumes no
        # prediction and no budget. Close-out, checks and archiving are NOT
        # routed through this gate — the host's finish work is never blocked.
        gate = self._budget_gate(str(task.get("task_id", "")), episode_id)
        if not gate["allowed"]:
            raise BudgetExhausted(gate["reason"], gate.get("view"))
        if not str(strategy_id or "").strip():
            raise ValueError(
                "strategy_id is required: the record must name the method "
                "that actually ran (there is no default strategy). Pass it "
                "explicitly, or pass --prediction to take it from the "
                "candidate")
        if not str(solver or "").strip():
            raise ValueError(
                "solver is required: the record must name the solver that "
                "actually ran. Pass it explicitly, or pass --prediction to "
                "take it from the candidate")
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
        pre = self.snapshot(task, episode_id, include_memory=False)
        # The execution config the FRAMEWORK knows before the run: the
        # strategy and solver it is about to use. Everything else (a time
        # limit applied inside solve.py, a seed) is NOT observable here and
        # is filled in AFTER the run from the executor's own report — never
        # copied from the candidate.
        exec_params: Dict[str, Any] = {
            "strategy_id": strategy_id, "solver": solver}
        if task.get("input_resources"):
            exec_params["input_resources"] = copy.deepcopy(task["input_resources"])
        if prediction_id:
            # The PRE-EXECUTION association: persisted with the action the
            # moment it exists, so the prediction→action link survives an
            # interruption. ``prediction_bound_at`` distinguishes "bound
            # before the run" (the target case) from a later bind.
            # The actual claim and adoption are written together in the
            # transaction below, after all static validation succeeds.
            forecast = self.strategy_predictions.get(prediction_id)
            exec_params["prediction_context_id"] = forecast.trace.model_info.get("prediction_context_id")
            if forecast.trace.model_info.get('hplus'):
                exec_params['hplus_verification']={'prediction_id':prediction_id,
                    'conditions':copy.deepcopy(forecast.trace.model_info['hplus'].get('verification_conditions') or [])}
        declared_memory = None
        reuse_versions = {}
        if adapted_from:
            from or_harness.world_model.conditions import put_value
            with self.store.transaction() as conn:
                for execution_id in adapted_from:
                    previous = self.bank.get(str(execution_id))
                    reuse_versions[str(execution_id)] = (put_value(
                        conn, self.store, "execution_evidence", previous.to_dict())
                        if previous is not None else None)
            exec_params["reuse_versions"] = reuse_versions
        if used_entry_ids is not None:
            declaration = ExecutionRecord(execution_id="declaration", task_id=str(task["task_id"]),
                                          strategy_id=str(strategy_id), profile_snapshot=profile)
            self._record_used_entries(declaration, used_entry_ids)
            declared_memory = declaration.execution_features["used_entries"]
            exec_params["used_entries"] = copy.deepcopy(declared_memory)
        action = self.actions.begin_action(
            "execute_strategy", str(task["task_id"]), episode_id,
            pre_snapshot=pre, params=exec_params)
        if prediction_id and candidate is not None:
            # ATOMIC PRE-EXECUTION ASSOCIATION. The claim is written in ONE
            # transaction BEFORE the executor starts, so it is consistent
            # and recoverable. If ANY part fails the exception propagates
            # and the execution is REFUSED — an attempt whose association
            # could not be recorded must not run, or its result would be
            # unattributable (and a second run of the same prediction could
            # slip through). The action is ended first so no action is left
            # ``running`` (which would block the episode close-out).
            try:
                self._claim_prediction_for_action(prediction_id,
                                                 action.action_id)
            except Exception as exc:
                try:
                    self.actions.end_action(
                        action.action_id, status="failed",
                        outcome={"error": f"{type(exc).__name__}: {exc}",
                                 "phase": "association",
                                 "note": ("the prediction could not be "
                                          "associated with this action, so "
                                          "the execution was NOT started")})
                except Exception:  # noqa: BLE001
                    pass
                raise
        # The method the outer agent PLANNED for this attempt: from the
        # candidate (a prediction-driven run) or from the explicit
        # ``method`` argument (an unpredicted run where the agent still
        # wants its method on record). It is a PLAN — the executor writes it
        # to ``method_planned`` and never to ``method_actual``; the method
        # that really ran comes only from the script's own receipt.
        planned_candidate = candidate.method if candidate is not None else None
        method_planned = (normalize_method(planned_candidate)
                          or normalize_method(method))
        if method_planned is not None:
            method_planned["source"] = "candidate" if planned_candidate \
                else "harness_declared"
            # The plan travels with the action so the binding can compare it
            # with the performed method without re-reading the candidate.
            self.actions.amend_action_params(
                action.action_id, method_planned=method_planned)
        try:
            from or_harness.core.artifacts import capture
            artifacts = {}
            submitted = Path(code_path).resolve()
            work_root = Path(workspace).resolve()
            if work_root in submitted.parents and submitted.is_file():
                artifacts["submitted_script"] = capture(
                    self.home, submitted.read_bytes(), kind="script",
                    source="framework-submitted script before attempt; execution status is separate")
                self.actions.amend_action_params(action.action_id, artifacts=artifacts)
            record = self._run_executor(
                Path(code_path), Path(workspace), solver=str(solver),
                task_id=str(task["task_id"]), strategy_id=str(strategy_id),
                profile=profile, action_id=action.action_id,
                method_planned=method_planned, input_resources=task.get("input_resources"))
            try:
                for key, value in getattr(record, "_captured_output", {}).items():
                    if value:
                        artifacts[key] = capture(self.home, value, kind=key,
                            source="executor captured output clip; may be truncated")
                result_bytes = getattr(record, "_result_bytes", None)
                if record.execution_features.get("executed") and result_bytes is not None:
                    artifacts["workspace_result"] = capture(self.home, result_bytes, kind="result",
                        source="file observed after attempt; acceptance is governed by execution checks")
                if record.cir_snapshot is not None:
                    artifacts["cir"] = capture(self.home, self.store.dumps(record.cir_snapshot), kind="cir",
                        source="internal representation supplied for this attempt")
                if artifacts:
                    record.execution_features["artifacts"] = artifacts
            except (OSError, StorageError) as exc:
                # Artifact storage cannot turn an observed execution into a
                # fabricated failed attempt or discard its measured cost.
                record.execution_features["artifact_capture_missing"] = str(exc)

        except BaseException as exc:
            # The PRE snapshot is already bound and the action is already
            # persisted as ``running``. An exception between here and
            # ``end_action`` (a solve script outside its workspace, a
            # filesystem error, an interrupt) would leave that action
            # running FOREVER — and a running action blocks episode
            # close-out, so a single mistyped path could make the whole
            # episode unclosable. End the action honestly instead: it
            # really did fail, and the reason is recorded. The exception
            # is re-raised unchanged (this is bookkeeping, not error
            # swallowing).
            #
            # The attempt itself is NOT discarded: a started attempt that
            # raised before producing a record is a REAL failure with a real
            # (if partial) cost, so an HONEST failure fact is staged and
            # linked to the action. On its own that stage gives the action a
            # ``linked_execution_id``, which is what makes the close-out see
            # the failure as its own observation rather than a gap. A scored
            # attempt keeps its prediction claim, even on this failure path;
            # a repaired submission needs a fresh plan-next selection. The
            # legacy release helper below leaves scored adoptions consumed.
            self._stage_interrupted_attempt(
                action, exc, profile=profile, strategy_id=str(strategy_id),
                solver=solver, method_planned=method_planned,
                task_id=str(task["task_id"]), episode_id=episode_id,
                task_text_ver=task_text_ver, prediction_id=prediction_id)
            if prediction_id:
                self._release_prediction_claim(
                    str(prediction_id), action.action_id,
                    reason=f"{type(exc).__name__}: {exc}")
            raise
        # A legacy executor (one written before the method fields existed)
        # cannot accept ``method_planned``; attach it here so the plan is
        # never silently dropped on the way to the record.
        if method_planned is not None and record.method_planned is None:
            record.method_planned = method_planned
        from or_harness.strategy.rule_versions import declared_task_identity
        record.execution_features["task_identity"] = declared_task_identity(task)
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
        self._observe_retry_increment(record,action,task_id=str(task['task_id']),episode_id=episode_id)
        # REUSE TRACE (lightweight, opt-in). When the agent states that this
        # attempt was ADAPTED FROM specific past executions (the recall hits
        # it actually read), the citation is recorded on the fact. It is a
        # citation, not a claim of benefit: "adapted_from ex_..." means the
        # agent LOOKED AT that case, NOT that reusing it succeeded. Whether
        # the reuse worked is decided independently by THIS attempt's task
        # check and by later calibration — never by the citation itself, and
        # never counted as "reuse succeeded". Unknown ids are kept verbatim
        # (the citation is the agent's, the framework does not validate
        # existence) so a later reader can retrace it.
        adapted = [str(e) for e in (adapted_from or []) if str(e).strip()]
        if adapted:
            reuse: Dict[str, Any] = {
                "adapted_from": adapted,
                "versions": reuse_versions,
                "version_source": "past execution value observed before this attempt; missing refs remain unknown",
                "note": ("citations of past executions this attempt was "
                         "adapted from — a record that the agent READ them, "
                         "NOT a claim that reuse helped; its effect is judged "
                         "by this attempt's own task check and calibration"),
            }
            if adaptation is not None and str(adaptation).strip():
                reuse["adaptation"] = str(adaptation).strip()
            record.execution_features["reuse_trace"] = reuse
        # USED KNOWLEDGE (entry numbers the agent declares it ACTUALLY
        # adopted). Distinct from BOTH recall (``last_consulted_at``) and the
        # adapted-from citation: this is the agent naming the entries this
        # attempt's method relied on. ``[]`` is VALID — "no prior knowledge
        # was adopted" — and is recorded as such rather than left absent, so
        # a later reader can tell "adopted nothing" from "did not say". An
        # UNSTATED adoption (``None``) writes nothing: absence is not an
        # empty declaration.
        if declared_memory is not None:
            record.execution_features["used_entries"] = copy.deepcopy(declared_memory)
        record.execution_features["preconditions_id"] = pre.snapshot_id
        if prediction_id:
            record.execution_features["prediction_id"] = prediction_id
            if exec_params.get('hplus_verification'):
                record.execution_features['hplus_verification_plan']=copy.deepcopy(exec_params['hplus_verification'])
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
                     "failure_stage": (record.execution_features.get("technical_outcome") or {}).get("failure_stage"),
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
        post = self.snapshot(task, episode_id, task_progress=progress, include_memory=False)
        self.actions._bind_post_snapshot(action.action_id, post.snapshot_id)
        record.action_id = action.action_id
        # The configuration that REALLY took effect is now known (the
        # executor reported it, and the solve script may have read back its
        # own effective values). Attach it to the ACTION so the binding can
        # compare the candidate's prediction against reality rather than
        # against itself. A rejected script (never ran) reports no config —
        # it stays absent, never fabricated.
        observed_config = record.execution_features.get("execution_config")
        if observed_config:
            self.actions.amend_action_params(
                action.action_id, execution_config=observed_config)
        # The association ALREADY exists (it was persisted before the run).
        # What happens now is the SECOND HALF: the real RESULT and the real
        # configuration OBSERVATION are added to it, through the SAME
        # identity-checked, idempotent binding the explicit path uses — no
        # name guessing, no second count, no re-interpretation of the
        # candidate from scratch. A binding failure is REPORTED on the
        # record, never raised: the execution really happened and its
        # evidence must survive a bookkeeping problem. ``bind-strategy``
        # remains the recovery and manual path.
        if prediction_id:
            binding: Dict[str, Any] = {"prediction_id": str(prediction_id),
                                       "bound": False}
            try:
                bound = self.bind_strategy_outcome(prediction_id,
                                                   action.action_id)
                info = dict(bound.trace.model_info or {})
                binding = {
                    "prediction_id": str(prediction_id),
                    "bound": bool(info.get("bound_action_id")
                                  == action.action_id),
                    "comparable": bound.trace.comparable,
                    # The config the candidate ASKED FOR vs the config that
                    # really took effect, now that the run reported it. This
                    # is the point of the two-phase shape: the prediction is
                    # linked before the run, and the OBSERVATION is attached
                    # after it — the predicted config is never copied in as
                    # if it had been observed.
                    "config_observed": info.get("config_observed"),
                    "config_unknown": info.get("config_unknown"),
                    "trace": {
                        "bound_action_id": info.get("bound_action_id"),
                        "binding_mismatch": info.get("binding_mismatch"),
                        "binding_unknown": info.get("binding_unknown"),
                    },
                }
            except Exception as exc:  # noqa: BLE001 - never lose evidence
                binding["reason"] = (f"{type(exc).__name__}: {exc}; bind it "
                                     "explicitly with `orx bind-strategy`")
            record.execution_features["prediction_binding"] = binding
        return record

    def _run_executor(self, code_path: Path, workspace: Path, *, solver: str,
                      task_id: str, strategy_id: str, profile: ProblemProfile,
                      action_id: Optional[str],
                      method_planned: Optional[Dict[str, Any]] = None,
                      input_resources=None,
                      ) -> ExecutionRecord:
        """Call the executor, passing ``action_id`` only when it accepts it.

        The action id is the config receipt's stamp (see
        ``execution/executor.py``). An injected executor — a test double, or
        a custom one written against the earlier signature — may not take
        it; in that case the executor simply receives no stamp and its
        scripts cannot report an attributable configuration. That is
        reported as UNKNOWN by the binding, never silently accepted, and it
        must not stop an unrelated execution from running.

        ``method_planned`` is the candidate's own method description (the
        outer agent's PLAN). It is passed only when the executor accepts it,
        and it never lands on the record as the performed method: the
        executor writes it to ``method_planned`` and fills
        ``method_actual`` only from the script's ``method_performed``
        receipt.
        """
        import inspect
        try:
            parameters = inspect.signature(
                self.executor.execute).parameters
            accepts = ("action_id" in parameters
                       or any(p.kind == inspect.Parameter.VAR_KEYWORD
                              for p in parameters.values()))
            accepts_method = ("method_planned" in parameters
                              or any(p.kind == inspect.Parameter.VAR_KEYWORD
                                     for p in parameters.values()))
        except (TypeError, ValueError):
            parameters = {}
            accepts = False
            accepts_method = False
        extra: Dict[str, Any] = {}
        if accepts:
            extra["action_id"] = action_id
        elif method_planned is not None:
            # The planned method matters even when a legacy executor cannot
            # take the action-id stamp: it is attached by the CALLER below
            # instead, so the plan is never silently dropped.
            pass
        if accepts_method:
            extra["method_planned"] = method_planned
        if "input_home" in parameters or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
            extra["input_home"] = self.home
        if input_resources:
            if "input_resources" not in parameters and not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
                raise ValueError("injected executor does not support declared task inputs")
            extra.update(input_resources=input_resources, input_home=self.home)
        return self.executor.execute(
            code_path, workspace, solver=solver, task_id=task_id,
            strategy_id=strategy_id, profile=profile, **extra)

    def record(self, record: ExecutionRecord,
               override: Optional[Dict[str, float]] = None,
               *,
               override_mode: str = "replace",
               override_source: str = "agent_estimate",
               override_force: bool = False,
               host_usage: Optional[Dict[str, Any]] = None,
               prediction: Optional[PredictionSnapshot] = None,
               method: Optional[Dict[str, Any]] = None,
               method_actual: Optional[Dict[str, Any]] = None,
               used_entry_ids: Optional[Sequence[Any]] = None
               ) -> Dict[str, Any]:
        """Append a fact, then run the automatic chain:
        frozen quality checks -> cost backfill -> cost feedback ->
        induction-pattern hints.

        The chain is EVIDENCE-ONLY: it never promotes, demotes, or awakens a
        Strategic Knowledge entry. Quality checks are written onto the fact
        (``execution_features.quality_feedback``); the next explicit
        ``induce`` replays them offline (``InductionEngine.revise``).

        ``prediction`` is the pre-execution cost prediction ACTUALLY used
        (from :meth:`predict_cost`); it is frozen onto the record as the
        snapshot that feedback is computed against — feedback never re-reads
        a post-hoc current estimate. ``override`` backfills harness-owned
        host-owned dimensions (llm_tokens and extra tool calls) or sourced
        legacy measurements with explicit accounting. Current executions
        automatically measure their retry increment. Backfill uses replace
        (default, idempotent —
        re-applying the same measurement never double-counts) or increment.

        ``override_mode`` / ``prediction`` are keyword-only so the historical
        positional call ``record(record, override, retain_reason)`` keeps its
        original meaning.

        ``method`` / ``method_actual`` are keyword-only declarations for a
        record assembled OUTSIDE ``execute`` (the harness appending a fact
        directly). ``method`` is the agent's PLAN (written to
        ``method_planned`` with ``source="harness_declared"``);
        ``method_actual`` is the method the harness declares was ACTUALLY
        performed (``source="harness_declared"``). Neither is ever copied
        over a value the record already carries, so a script-reported
        performance is never overwritten by a later declaration.

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
        # Freeze the pre-execution prediction snapshot when the harness
        # supplies one and the record does not already carry it.
        if prediction is not None and record.prediction_snapshot is None:
            record.prediction_snapshot = prediction
        # Method declarations for a record assembled outside ``execute``.
        # A value already on the record WINS (a script's own performance
        # report is never overwritten by a later harness declaration), and
        # an empty declaration is not a method — it stays absent.
        if method is not None and record.method_planned is None:
            declared = normalize_method(method)
            if declared is not None:
                declared.setdefault("source", "harness_declared")
                record.method_planned = declared
        if method_actual is not None and record.method_actual is None:
            declared_actual = normalize_method(method_actual)
            if declared_actual is not None:
                declared_actual.setdefault("source", "harness_declared")
                record.method_actual = declared_actual
        # ADOPTION DECLARATION for a record assembled outside ``execute``:
        # the entry numbers the attempt relied on. A value already on the
        # record (from ``execute``) WINS — a later re-declaration never
        # rewrites what the run recorded. An explicit list (including ``[]``)
        # is recorded; ``None`` leaves it unstated.
        if used_entry_ids is not None \
                and "used_entries" not in record.execution_features:
            self._record_used_entries(record, used_entry_ids)
        # Frozen quality checks are computed against the interval in force
        # RIGHT NOW and persisted with the fact — nothing downstream
        # re-scores a later execution against a post-hoc interval.
        prediction_checks = self._check_predictions(record)
        if prediction_checks:
            record.execution_features["quality_feedback"] = prediction_checks
        # Carry forward a TASK-RESULT check already written to the STAGED
        # payload. ``check-task`` works on a staged execution ("the
        # annotation survives ``record --from-staged`` verbatim"), so a
        # caller that checks, then records the SAME in-memory object it got
        # back from ``execute``, would otherwise overwrite the staged
        # annotation with its stale copy and silently lose the verdict —
        # the answer would re-enter the close-out as UNCHECKED. The staged
        # annotation is the newer fact, so it wins; a check on the incoming
        # record is never overwritten (it is at least as new).
        staged = self.bank.get_pending(record.execution_id)
        if staged is not None:
            for key, value in (staged.execution_features or {}).items():
                if key == "task_check" \
                        and key not in record.execution_features:
                    record.execution_features[key] = value
        self.bank.append(record)
        # HOST usage first (the REAL attempt spend the outer framework
        # measured), then any explicit override. A host report is recorded
        # with the full口径 and marked ``provider_usage`` provenance (the
        # host really measured it), so it is distinguishable from a declared
        # estimate and protected from a later silent overwrite.
        if host_usage:
            from or_harness.world_model.usage import (
                host_report_dimensions,
                host_usage_cost_vector,
                is_host_usage_report,
            )
            if is_host_usage_report(host_usage):
                # A versioned HostUsageReport (or-host-usage/1): its
                # ``measured`` whitelist governs which dimensions may enter
                # cost_measured, and its per-dimension ``provenance`` map
                # says where each number came from (llm_tokens from the
                # provider, tool_calls observed by the host). A dimension
                # outside the whitelist stays UNKNOWN — never zero.
                dimensions, info = host_report_dimensions(host_usage)
                for dim, value in (dimensions.items() if info["scope"] == "attempt" else []):
                    self.bank.update_cost(
                        record.execution_id,
                        source=info["provenance"].get(
                            dim, "provider_usage"),
                        basis=(info.get("basis") if dim == "llm_tokens" else
                               info.get("output_tokens_basis") if dim == "output_tokens" else None),
                        **{dim: float(value)})
                self.bank.annotate_features(
                    record.execution_id, {"host_usage": info})
            else:
                vector, breakdown = host_usage_cost_vector(host_usage)
                if vector is not None and "llm_tokens" in \
                        vector.measured_dims():
                    self.bank.update_cost(
                        record.execution_id, source="provider_usage",
                        llm_tokens=float(vector.llm_tokens))
                    # Record the口径 basis beside the value, so a legacy
                    # completion-only record and a full-口径 total are never
                    # pooled as if they were the same unit.
                    self.bank.annotate_features(
                        record.execution_id,
                        {"host_usage": breakdown} if breakdown else
                        {"host_usage": {"note": "host reported no token "
                                                "usage"}})
                    stored = self.bank.get(record.execution_id)
                    prov = dict(stored.execution_features.get(
                        "cost_provenance") or {})
                    if isinstance(prov.get("llm_tokens"), dict) \
                            and breakdown and breakdown.get("basis"):
                        prov["llm_tokens"]["basis"] = breakdown["basis"]
                        self.bank.annotate_features(
                            record.execution_id, {"cost_provenance": prov})
            record = self.bank.get(record.execution_id)
        if override:
            # Backfilled dimensions are DECLARATIONS by default (the harness
            # reports its own spend): provenance says so, and the amend
            # refuses to clobber a framework-measured dimension without an
            # explicit force.
            self.bank.update_cost(record.execution_id, mode=override_mode,
                                  source=override_source, force=override_force,
                                  **override)
            record = self.bank.get(record.execution_id)
        self.bank.clear_pending(record.execution_id)

        cost_feedback = compute_cost_feedback(
            record.prediction_snapshot, record.strategy_id,
            record.measurement_scope, record.cost)
        if cost_feedback is not None:
            self.bank.set_cost_feedback(record.execution_id, cost_feedback)
        unrecorded = [p.execution_id for p in
                      self.bank.pending(task_id=record.task_id)]
        result = {
            "execution_id": record.execution_id,
            "recorded": True,
            "prediction_checks": prediction_checks,
        }
        completeness = self._cost_completeness(record)
        if completeness is not None:
            result["cost_completeness"] = completeness
        if cost_feedback is not None:
            result["cost_feedback"] = cost_feedback
        # Index sync happens AFTER the fact is durable: the memory exists
        # whether or not the embedding call works, and the outcome is always
        # reported (never a silent divergence between bank and index).
        result["index_sync"] = self.index_sync.sync_execution(record)
        if unrecorded:
            result["unrecorded_staged_executions"] = unrecorded
        dependencies=self._refresh_rule_evidence(record.execution_id)
        if dependencies['affected_versions']:result['rule_validation']=dependencies
        return result

    def record_session(self, task_id: str,
                       episode_id: Optional[str] = None, *,
                       reason: str, cancelled: bool = False,
                       attempted_execution_id: Optional[str] = None,
                       cost: Optional[Dict[str, float]] = None,
                       ) -> Dict[str, Any]:
        """Archive a SESSION-LEVEL stop (host cancel / hard budget stop).

        Called by the HOST after it has CONFIRMED the working process is
        gone (it cancelled it, or it exited). The framework does NOT cancel
        anything here — it records the interruption as a real fact, so the
        episode can be closed and calibrated without fabricating an outcome.

        Three shapes, decided by what actually exists:

        1. An already-STAGED attempt (``attempted_execution_id`` names a
           staged execution) -> the attempt fact already exists: nothing is
           rewritten, and the session stop is recorded as an annotation
           (``session_interrupted``) on it. Its observed cost and partial
           result stay as they are.
        2. A ``running`` execute action with NO staged fact (a kill that
           landed BEFORE staging) -> that action is ended as an honest
           interruption (``status="failed"``, ``outcome.interrupted=True``)
           and the session cost is recorded; result fields that could not be
           observed are marked ``unobservable`` — the real cost is kept, no
           solve result is invented.
        3. NO execute attempt at all (the session spent its budget editing
           code / calling the world model) -> a non-execution action records
           the real termination and cost. No ``execute_strategy`` is created.

        In every shape the reason carries the HOST's stated cause verbatim
        (e.g. "tool_calls reached 100"), and an ``episode_termination`` mark
        is written so a later :meth:`_budget_gate` refuses new solving work
        for this task — reaching a cap and refusing the NEXT call leaves the
        count AT the limit, so the stop cannot be inferred from ``> limit``.

        IDEMPOTENT: re-recording the SAME termination (same reason/cancel/
        cost) returns the stored mark and writes nothing twice, so a retried
        archive never double-counts.

        Read-only with respect to the budget gate: this is the FINISH path
        the host runs after stopping; it is never gated.
        """
        key = f"episode_termination|{task_id}|{episode_id or ''}"
        fingerprint = {
            "reason": str(reason), "cancelled": bool(cancelled),
            "attempted_execution_id": attempted_execution_id,
            "cost": ({d: float(v) for d, v in (cost or {}).items()}
                     if cost else None),
        }
        existing = self._episode_termination(task_id, episode_id)
        recorded: List[Dict[str, Any]] = []

        # IDEMPOTENT REPLAY runs FIRST: if the SAME stop was already
        # recorded, return the stored mark and write nothing (a repeated
        # archive — e.g. the host retries — must not spawn a second
        # terminal action or double-count its cost).
        if existing is not None and existing.get("fingerprint") == fingerprint:
            self._annotate_staged_interruption(attempted_execution_id,
                                               fingerprint, recorded)
            return {"termination": existing, "recorded": recorded,
                    "already_recorded": True}

        # (1) A staged attempt already carries the fact.
        if self._annotate_staged_interruption(attempted_execution_id,
                                              fingerprint, recorded) is None:
            # (2) A running execute action with no staged fact, or (3) none.
            running = [a for a in self.actions.query(
                task_id=task_id, episode_id=episode_id, status="running")
                if a.action_type == "execute_strategy"]
            cost_vector = self._cost_from_dict(cost)
            if running:
                for act in running:
                    self.actions.end_action(
                        act.action_id, status="failed",
                        outcome={"interrupted": True,
                                 "reason": fingerprint["reason"],
                                 "cancelled": fingerprint["cancelled"],
                                 "result": "unobservable",
                                 "note": ("the run was stopped before its "
                                          "result could be observed; the "
                                          "cost incurred is kept, no solve "
                                          "result is fabricated")},
                        cost=cost_vector)
                    recorded.append({"action_id": act.action_id,
                                     "shape": "running_action_interrupted"})
            else:
                # (3) No attempt: record the real termination + cost as a
                # non-execution action, never a fabricated execution.
                report = self.actions.report_action(
                    "finish_task", task_id, episode_id,
                    status="failed",
                    outcome={"session_termination": True,
                             "reason": fingerprint["reason"],
                             "cancelled": fingerprint["cancelled"],
                             "no_execution_attempt": True,
                             "note": ("the session ended without an execute "
                                      "attempt: the real spend is recorded, "
                                      "no solve execution is invented")},
                    cost=cost_vector)
                recorded.append({"action_id": report.action_id,
                                 "shape": "session_only_terminal"})

        mark = {"task_id": task_id, "episode_id": episode_id,
                "reason": fingerprint["reason"],
                "cancelled": fingerprint["cancelled"],
                "attempted_execution_id": attempted_execution_id,
                "cost": fingerprint["cost"],
                "recorded_at": time.time(),
                "fingerprint": fingerprint}
        with self.store.transaction() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?,?)",
                (key, self.store.dumps(mark)))
        return {"termination": mark, "recorded": recorded,
                "already_recorded": False}

    def _annotate_staged_interruption(
            self, attempted_execution_id: Optional[str],
            fingerprint: Dict[str, Any],
            recorded: List[Dict[str, Any]]) -> Optional[str]:
        """Annotate a STAGED attempt with the session stop, when one exists.

        Returns the execution id when a staged fact was annotated (so the
        caller stops there), or None when there is no staged attempt to
        annotate.

        The host's stated REAL cost is ALSO backfilled onto the staged fact
        (via ``amend_pending_cost``) when one was supplied: without it the
        stopped attempt's real spend was invisible to the budget ledger, so
        the episode could read ``unconfirmed`` forever and no budget label
        could be produced even with a confirmed stop (the r23 gap). The
        numbers are the host's receipt, marked measured with provenance;
        nothing is fabricated and no session total is split."""
        if not attempted_execution_id:
            return None
        staged = self.bank.get_pending(attempted_execution_id)
        if staged is None:
            return None
        self.bank.annotate_features(attempted_execution_id, {
            "session_interrupted": {
                "reason": fingerprint["reason"],
                "cancelled": fingerprint["cancelled"],
            }})
        receipt_cost = fingerprint.get("cost") or {}
        if receipt_cost:
            try:
                self.bank.amend_pending_cost(
                    attempted_execution_id,
                    {d: float(v) for d, v in receipt_cost.items()},
                    source="agent_observed")
            except Exception:  # noqa: BLE001 - the stop itself must land
                pass
        recorded.append({"execution_id": attempted_execution_id,
                         "shape": "staged_attempt_annotated"})
        return attempted_execution_id

    def induce(self, *, induction_id=None, review=None, submission=None, **kwargs):
        """Authorize and settle binary H+ work around the existing writer.

        ``submission`` is the compact agent entry (see
        ``world_model.induction_submission``): one JSON object with the
        learning result, when/do/expect, one evidence-and-argument basis and
        an optional publication choice.  It is normalised here into the SAME
        ``relations`` / ``review`` payload used by the legacy ``--relation``
        / ``--review`` path, then committed through the existing versioned
        writer and H+ receipt.  The two interfaces cannot be mixed in one
        call.
        """
        from or_harness.world_model.hplus import mode
        if mode(self)=='frozen':raise ValueError('the strategy library is frozen for testing')
        if submission is not None:
            forbidden = [name for name, value in (
                ('review', review),
                ('relations', kwargs.get('relations')),
                ('notes', kwargs.get('notes')),
                ('verify', kwargs.get('verify')),
                ('effect_attributions', kwargs.get('effect_attributions')),
                ('strategy_id', kwargs.get('strategy_id')),
            ) if value not in (None, [], {})]
            if forbidden:
                raise ValueError('compact submission cannot be combined with '
                                 + ', '.join('--' + name.replace('_', '-')
                                             for name in forbidden)
                                 + '; use one induction interface per call')
            from or_harness.world_model.induction_submission import normalize_submission
            compact = normalize_submission(self, submission,
                                           induction_id=induction_id)
            review = compact['review']
            kwargs = {**kwargs, 'relations': compact['relations']}
        relations=kwargs.get('relations') or []
        body_keys={'when','do','expect','claim','text','method','conditions','predicates','effect_scope','merge_entry_ids'}
        existing_only=bool(relations) and all(
            (r.get('entry_id') or r.get('target_entry') or r.get('target_entry_id')) and
            r.get('operation','add_evidence')=='add_evidence' and not(body_keys & set(r)) and
            self.sbank.get(str(r.get('entry_id') or r.get('target_entry') or r.get('target_entry_id'))) is not None
            for r in relations)
        associated=induction_id is not None or any(r.get('induction_id') or r.get('material_id') for r in relations)
        if (existing_only or not relations and kwargs.get('effect_attributions')) and not associated and review is None:
            return self._maintain_entries(**kwargs)
        from or_harness.world_model.hplus import (
            reserve_action, complete, get_job, completed_result, put_submission,submission_conflict,
            submission_fingerprint,submission_matches,validate_submission,
        )
        from or_harness.world_model.hplus import submission_job
        job=submission_job(self,induction_id,kwargs.get('relations'),submission_fingerprint(kwargs,review))
        if job is None:
            return self._induce_impl(**kwargs)
        if job['state'] == 'completed':
            if not submission_matches(self,job,kwargs,review):
                raise submission_conflict(self,job,kwargs,review)
            if kwargs.get('dry_run'):
                return {'dry_run':True,'induction_id':job['induction_id'],'already_completed':True,
                    'receipt_ref':job['receipt_ref'],'would_complete':False,'would_create':0,'would_update':[],
                    'saved':0,'published':0,'relations':[],'validation':{'valid':True}}
            result = completed_result(self, job)
            result['already_completed'] = True
            self._sync_induction_result(result)
            return result
        if not kwargs.get('relations') and not (review or kwargs.get('notes')):
            raise ValueError('empty review requires a reason or structured review receipt')
        if not (job.get('material_sessions') or job.get('material_ref')):
            raise ValueError('read induction-material for this job before submitting its review')
        support=validate_submission(self,job,review,relations)
        if kwargs.get('dry_run'):
            self._active_induction_job=job
            try:
                result=self._induce_impl(**kwargs)
            finally:
                self._active_induction_job=None
            rows=result.get('relations') or []
            valid=not relations or any(r.get('would_create') or r.get('would_update') for r in rows)
            return {**result,'dry_run':True,'induction_id':job['induction_id'],
                'would_complete':bool(valid),'would_create':sum(bool(r.get('would_create')) for r in rows),
                'would_update':list(dict.fromkeys(r['would_update'] for r in rows if r.get('would_update'))),
                'validation':{'valid':bool(valid),'comparison_support':support,
                              'relation_errors':[{'field':f'relations[{i}]','reason':r['skipped']}
                                                 for i,r in enumerate(rows) if r.get('skipped')]}}
        reserve_action(self, job)
        try:
            with self.store.transaction():
                current=get_job(self,job['induction_id'])
                fingerprint=submission_fingerprint(kwargs,review)
                if current['state']=='completed':
                    if not submission_matches(self,current,kwargs,review):
                        raise submission_conflict(self,current,kwargs,review)
                    result=completed_result(self,current)
                    result['already_completed']=True
                else:
                    job=current
                    self._active_induction_job=job
                    result = self._induce_impl(**kwargs)
                    if kwargs.get('relations') and not result.get('saved'):
                        raise ValueError('induction submission refused; job stays pending: '+str(result.get('relations')))
                    job['submission_digest']=fingerprint
                    job['submission_ref']=put_submission(self,kwargs,review)
                    complete(self,job,result,review)
            self._sync_induction_result(result)
            return result
        finally:
            self._active_induction_job = None

    def induce_submission(self, submission, *, induction_id=None,
                          dry_run=False, force=False):
        """Submit one compact induction review.

        Thin convenience over :meth:`induce`; the compact protocol is
        documented in ``references/induction.md`` and validated by
        ``world_model.induction_submission``.  It commits through the same
        versioned rule writer, H+ authorisation, receipt and idempotency
        path as ``induce(relations=..., review=...)``.
        """
        return self.induce(induction_id=induction_id, submission=submission,
                           dry_run=dry_run, force=force)

    def _sync_induction_result(self,result):
        """Repair only committed vector intents; receipt/action identity stays fixed."""
        from or_harness.strategy.rule_versions import sync_rule_entries
        affected=set(result.get('affected_entry_ids') or [])|set(result.get('index_entry_ids') or [])
        result['index_sync']=sync_rule_entries(self,[],affected)
        result['committed']=not self.store.conn.in_transaction

    def _maintain_entries(self,*,effect_attributions=None,relations=None,dry_run=False,strategy_id=None,**kwargs):
        """Manage selected existing entries without an induction action or receipt."""
        from or_harness.world_model.state import KnowledgeRef
        plans=[self._validate_effect_attribution(v) for v in effect_attributions or []]
        ids=sorted({p['entry_id'] for p in plans}|{str(r.get('entry_id') or r.get('target_entry') or r.get('target_entry_id')) for r in relations or []})
        ids=sorted(set(ids)|{e.entry_id for e in self.sbank.list() if (e.rule or {}).get('merge_pending_into') in ids})
        def state():
            return {eid:KnowledgeRef.from_entry(entry).to_dict() for eid in ids if (entry:=self.sbank.get(eid)) is not None}
        with self.store.transaction():
            from or_harness.world_model.hplus import mode
            if mode(self)=='frozen':raise ValueError('the strategy library is frozen for testing')
            before=state();count_before=self.sbank.count()
            self._existing_entry_maintenance=True
            try:
                submitted=self._submit_relations(relations,dry_run=dry_run,force=False,verify=kwargs.get('verify'),notes=kwargs.get('notes')) if relations else {}
            finally:self._existing_entry_maintenance=False
            attributed=self._record_validated_attributions(plans,dry_run=dry_run)
            revisions=self.induction.revise(dry_run=dry_run,entry_ids=set(ids))
            after=state()
            index_changed=set(submitted.get('index_entry_ids') or [])|{p['entry_id'] for p in plans if p['lifecycle_action'] or p['item']['verdict']=='refuting'}
            index_changed.update(eid for a in attributed for eid in a.get('merged_entry_ids') or [])
            if not dry_run:
                from or_harness.strategy.rule_versions import sync_rule_entries
                sync_rule_entries(self,index_changed,ids)
        fields=('predicates','expected_quality_hat','quality_interval','expected_cost_hat','cost_interval',
                'failure_prob','support_n','status','verification_state','rule','claim')
        changes=[]
        for eid in sorted(set(before)&set(after)):
            changed={f:{'before':before[eid].get(f),'after':after[eid].get(f)} for f in fields if before[eid].get(f)!=after[eid].get(f)}
            if changed:changes.append({'entry_id':eid,'changed':changed})
        delta={'entries_created':[],'entries_removed':[],'entry_count_before':count_before,
               'entry_count_after':self.sbank.count(),'entry_changes':changes}
        result={**submitted,'maintenance_only':True,'saved':submitted.get('saved',0),'effect_attributions':attributed,
                'revisions':revisions,'affected_entry_ids':ids,'knowledge_delta':delta,
                'knowledge_after':{'entries':list(after.values()),'entry_count':self.sbank.count(),'scope':'affected_entries'},
                'business_result':'revised' if changes else 'relation_updated' if any(r.get('changed') for r in submitted.get('relations') or []) else 'unchanged',
                'execution_ids':sorted({p['item']['execution_id'] for p in plans if p['item'].get('execution_id')})}
        if not dry_run:
            from or_harness.strategy.rule_versions import sync_rule_entries
            result['index_sync']=sync_rule_entries(self,[],set(ids)|index_changed)
        return result

    def _induce_impl(self, *, dry_run: bool = False, force: bool = False,
               notes: Optional[List[str]] = None,
               verify: Optional[Dict[str, Any]] = None,
               strategy_id: Optional[str] = None,
               relations: Optional[Sequence[Dict[str, Any]]] = None,
               effect_attributions: Optional[Sequence[Dict[str, Any]]] = None
               ) -> Dict[str, Any]:
        """Submit new strategies / strategy revisions as STRUCTURED claims.

        This is the ONE place knowledge changes. There is no statistical
        path that writes a strategy from a cell's means: the framework will
        not turn a strategy name and a mean into a technique. Instead the
        agent reads the material (``orx induction-material``) and submits a
        structured strategy (condition -> how -> consequence -> boundary) with
        the executions that established it, using ``relations``.

        Each ``relations`` entry references real executions with a ROLE and
        declares what is checkable. The framework derives the evidence
        identity (tasks, family, cell, strategy ids) from the recorded facts —
        a caller submits only ids and roles, never a second identity. It
        saves the new strategy and (with a ``check`` block or ``verify``)
        computes its verdict. There is NO strategy-id requirement: a claim's
        own optional ``subject`` names a free-form subject (e.g.
        ``principle:cross_period_state``) so cross-task knowledge that belongs
        to no single strategy still has a create/save/verify/recall path.

        Different tasks, strategy ids and cells may be cited TOGETHER — there
        is no "same method name / same cell" bar. Whether a shared mechanism
        exists is the agent's judgment from the content; the framework
        REPORTS the distinct-task count and lets the publication gate speak.
        A transaction is not created by splicing two tasks together to reach
        a count.

        ``notes`` are harness-written applicability notes (free text): kept
        for the reader, never scored.

        Submitting NO relations writes no knowledge, but it is a REAL review:
        the maintenance lifecycle (``revise``) STILL runs, so an entry's
        forward checks, promotion/demotion and dormancy management keep
        working even when the agent decided to add nothing. The review is
        recorded as a maintenance action fact (its scope and reason), so the
        existing attribution chain sees that a review happened and concluded
        "no new knowledge". The framework never runs a statistical induction
        behind the agent's back — it only maintains the EXISTING knowledge.
        """
        if not relations:
            result = self._record_empty_review(
                dry_run=dry_run, reason=notes, strategy_id=strategy_id,
                effect_attributions=effect_attributions)
            return result
        # A relation write is a knowledge write like any other: it gets a
        # maintenance action with PRE and POST knowledge state, a knowledge
        # transition and an index result.
        #
        # ORDER (r21): VALIDATE the effect attributions FIRST (a bad payload
        # must not leave a half-written state), begin the action, then submit
        # the relations, RECORD the attributions (before ``revise`` so the
        # management events are replayed and the entry is written once),
        # run the lifecycle maintenance, and only THEN snapshot the POST
        # knowledge state — so the delta / business result / index all see the
        # SAME state that the management just produced.
        validated_attributions = [
            self._validate_effect_attribution(raw)
            for raw in (effect_attributions or [])]
        maintenance = None
        if not dry_run:
            maintenance = self._begin_induce_action(strategy_id, False)
            self.actions.amend_action_params(
                maintenance["action_id"], knowledge_shape="relations")
        try:
            result = self._submit_relations(
                relations, dry_run=dry_run, force=force, verify=verify,
                notes=notes)
        except Exception:
            if maintenance is not None:
                import traceback
                self.actions.end_action(
                    maintenance["action_id"], status="failed",
                    outcome={"error": traceback.format_exc(limit=3)})
            raise
        # Record the agent's OWN use-effect attributions (and their optional
        # management requests) BEFORE the lifecycle pass, so a promote/demote
        # is replayed THIS time and the entry is written once. ``revise``
        # applies the status change; nothing is applied twice.
        attributed = self._record_validated_attributions(
            validated_attributions, dry_run=dry_run)
        if attributed:
            result["effect_attributions"] = attributed
            ids={a["entry_id"] for a in attributed}|{eid for a in attributed for eid in a.get('merged_entry_ids') or []}
            result["index_entry_ids"]=sorted(set(result.get("index_entry_ids") or [])|ids)
            result["affected_entry_ids"]=sorted(set(result.get("affected_entry_ids") or [])|ids)
        if maintenance is not None:
            # Offline LIFECYCLE maintenance: calibration is replayed and any
            # pending management event is applied (promotion/demotion,
            # dormancy wakeup). This is management of EXISTING knowledge.
            try:
                revisions = self.induction.revise(strategy_id=strategy_id)
            except Exception as exc:  # never fail a real write for this
                revisions = None
                result["revisions_error"] = f"{type(exc).__name__}: {exc}"
            if revisions is not None:
                result["revisions"] = revisions
            result["action"] = self._end_induce_action(maintenance, result)
            # Surface the transition at the TOP level too, so a caller (and
            # ``accept_capability_operation``) reads the same
            # ``business_result`` / ``knowledge_delta`` shape on every write.
            result.setdefault("business_result",
                              result["action"]["business_result"])
            result.setdefault("knowledge_delta",
                              result["action"]["knowledge_delta"])
            result.setdefault("knowledge_after",
                              result["action"]["knowledge_after"])
        if not dry_run:
            from or_harness.strategy.rule_versions import sync_rule_entries
            result["index_sync"] = sync_rule_entries(self,result.get("index_entry_ids") or [],result.get("affected_entry_ids") or [])
        return result

    def _submit_relations(self, relations: Sequence[Dict[str, Any]], *,
                          dry_run: bool, force: bool,
                          verify: Optional[Dict[str, Any]],
                          notes: Optional[List[str]] = None
                          ) -> Dict[str, Any]:
        """Write short rule versions and append declared evidence/checks.

        IDs and scopes come from real facts. Legacy relation assertions
        remain audit records; sourced transfer and boundary checks determine
        publication of new versions. Synchronize only affected documents.
        """
        results = []
        created: List[str] = []
        for raw in relations:
            relation = dict(raw)
            try:
                outcome = self.induction.submit_relation(
                    relation, dry_run=dry_run, force=force,
                    verify=verify, notes=notes)
            except ValueError as exc:
                outcome = {"saved": None, "skipped": f"invalid relation: {exc}"}
            results.append(outcome)
            entry_id = outcome.get("created_entry") or outcome.get("saved")
            if entry_id and not dry_run:
                created.append(entry_id)
        index_sync = None
        saved = len([r for r in results
                     if r.get("saved") or r.get("created_entry")])
        out: Dict[str, Any] = {
            "relations": results,
            "saved": saved,
            "published": len([r for r in results
                              if (r.get("publication") or {}).get("published")]),
            "index_entry_ids": sorted({i for r in results for i in r.get("index_entry_ids") or []}),
            "affected_entry_ids": sorted({i for r in results for i in r.get("affected_entry_ids") or []}),
        }
        # A dry run writes nothing, index included — and does not report an
        # index result it never produced.
        if not dry_run:
            out["index_sync"] = index_sync
        return out

    def _record_empty_review(self, *, dry_run: bool, reason: Optional[Any],
                             strategy_id: Optional[str],
                             effect_attributions: Optional[
                                 Sequence[Dict[str, Any]]] = None
                             ) -> Dict[str, Any]:
        """Record an induction review that concluded NO new knowledge.

        A review with no submission is still a real maintenance event: the
        offline lifecycle (``revise``) runs so existing knowledge's calibration
        is replayed and any management request is applied, and a maintenance
        action is written naming the review's scope and reason so the
        attribution chain can see it. NO entry is created and NO fabricated
        knowledge is written. The agent may ALSO attach use-effect
        attributions (and management requests) to this review, since utility
        management of existing knowledge is exactly what a no-new-knowledge
        review is for.
        """
        # VALIDATE first (a bad payload writes nothing).
        validated_attributions = [
            self._validate_effect_attribution(raw)
            for raw in (effect_attributions or [])]
        result: Dict[str, Any] = {
            "relations": [], "saved": 0, "published": 0,
            "no_new_knowledge": True,
            "note": ("no relations were submitted: this is a review that "
                     "decided the existing knowledge needs no addition. The "
                     "lifecycle maintenance below still ran."),
        }
        if dry_run:
            return result
        maintenance = self._begin_induce_action(strategy_id, False)
        self.actions.amend_action_params(
            maintenance["action_id"], knowledge_shape="review_only")
        # Record the agent's attributions BEFORE the lifecycle pass, so any
        # management event is replayed THIS time and the entry is written once.
        attributed = self._record_validated_attributions(validated_attributions)
        if attributed:
            result["effect_attributions"] = attributed
        revised: List[Dict[str, Any]] = []
        try:
            revised = self.induction.revise(strategy_id=strategy_id)
        except Exception as exc:  # never fail a review for this
            result["revisions_error"] = f"{type(exc).__name__}: {exc}"
        result["revisions"] = revised
        # The action's POST snapshot and delta see the state the management
        # just produced. It is ended ONCE, with the review's reason folded in.
        result["action"] = self._end_induce_action(
            maintenance, result, extra_outcome={
                "review": True,
                "reason": [str(r) for r in (reason or []) if str(r).strip()],
                "note": ("a review with no new knowledge: the lifecycle was "
                         "maintained over the EXISTING entries; nothing was "
                         "created and no statistical induction ran"),
            })
        result["business_result"] = result["action"]["business_result"]
        result["knowledge_delta"] = result["action"]["knowledge_delta"]
        result["knowledge_after"] = result["action"].get("knowledge_after")
        if not dry_run:
            from or_harness.strategy.rule_versions import sync_rule_entries
            ids=sorted({p['entry_id'] for p in validated_attributions}|{eid for a in attributed for eid in a.get('merged_entry_ids') or []})
            result['affected_entry_ids']=result['index_entry_ids']=ids
            result["index_sync"] = sync_rule_entries(self,ids,ids)
        return result


    def _begin_induce_action(self, strategy_id: Optional[str],
                             all_: bool = False) -> Dict[str, Any]:
        """Begin the maintenance-scope induce action with a real PRE        snapshot: the knowledge state (entries + verification layers) as it
        stood BEFORE induction."""
        job = getattr(self, '_active_induction_job', None)
        if job and job.get('action_id'):
            return {'action_id':job['action_id'], 'episode_id':job['maintenance_episode_id'],
                    'knowledge_before':copy.deepcopy(job['knowledge_before'])}
        episode_id = f"maint_{int(time.time())}"
        from or_harness.strategy.rule_versions import maintenance_references
        knowledge_before = {
            "entries": maintenance_references(self),
            "entry_count": self.sbank.count(),
        }
        action = self.actions.begin_action(
            "induce", MAINTENANCE_TASK_ID, episode_id,
            params={"scope": "maintenance",
                    "strategy_id": strategy_id,
                    "all": bool(all_),
                    "knowledge_before": knowledge_before})
        return {"action_id": action.action_id, "episode_id": episode_id,
                "knowledge_before": knowledge_before}

    def _end_induce_action(self, maintenance: Dict[str, Any],
                           result: Dict[str, Any],
                           extra_outcome: Optional[Dict[str, Any]] = None
                           ) -> Dict[str, Any]:
        """End the maintenance induce action: business result, verification
        outcomes, and the knowledge delta (before vs after), plus the POST
        snapshot of the knowledge state.

        ``extra_outcome`` merges caller-specific fields (e.g. a review's
        ``reason``) into the persisted outcome, so the action is ended ONCE
        with the full picture.

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
        # RELATION submissions travel in their own result shape: the entry
        # they created/updated is named by ``created_entry`` / ``updated_entry``
        # and their publication state by ``publication``. They are a knowledge
        # write like any other, so they must produce the same kind of business
        # result and delta rather than an empty one.
        relation_rows = result.get("relations") or []
        relations_saved = [r for r in relation_rows if r.get("saved")]
        relation_created_entries = [
            r.get("created_entry") for r in relations_saved
            if r.get("created_entry")]
        revised = result.get("revisions") or []
        if created_verified:
            business = "created"
        elif created_unverified:
            business = "created_unverified"
        elif updated:
            business = "updated"
        elif relation_created_entries:
            # Report only the IDs and version changes this write committed.
            business = "relation_created"
        elif any(r.get('changed',True) for r in relations_saved):
            business = "relation_updated"
        elif revised:
            business = "revised"
        elif refused or (relation_rows and not relations_saved):
            business = "refused"
        else:
            business = "unchanged"
        from or_harness.strategy.rule_versions import maintenance_references
        knowledge_after = {
            "entries": maintenance_references(self),
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
        #
        # The stated CLAIM is part of the entry's knowledge, so it is
        # diffed too — field by field, not as one opaque blob: revising an
        # assertion or re-scoping its conditions are different events, and a
        # feedback that read "nothing changed" while the claim was rewritten
        # would misjudge the induction entirely.
        entry_changes = []
        for entry_id in sorted(before_ids & after_ids):
            before, after = before_by_id[entry_id], after_by_id[entry_id]
            changed_fields = {}
            for field in ("predicates", "expected_quality_hat",
                          "quality_interval", "expected_cost_hat",
                          "cost_interval", "failure_prob", "support_n",
                          "status", "verification_state", "rule"):
                if before.get(field) != after.get(field):
                    changed_fields[field] = {
                        "before": before.get(field),
                        "after": after.get(field),
                    }
            claim_change = ({'before_ref':before.get('claim_ref'),'after_ref':after.get('claim_ref')}
                            if before.get('claim_ref')!=after.get('claim_ref') else None)
            if claim_change:
                changed_fields["claim"] = claim_change
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
            "relations_saved": len(relations_saved),
            # The REAL executions the induction CITED (derived from each
            # saved relation's verification scope, never re-typed). This is
            # what lets a later reader associate the knowledge change with
            # the online predictions made about those very executions —
            # without defaulting to "the bound action was the induction".
            "execution_ids": sorted({str(eid) for r in relations_saved for eid in r.get("execution_ids") or []} |
                {str(a['execution_id']) for a in result.get('effect_attributions') or [] if a.get('execution_id')} | {
                str(e.get("execution_id"))
                for r in relations_saved
                for e in ((r.get("verification") or {}).get("scope") or {}
                          ).get("evidence") or []
                if isinstance(e, dict) and e.get("execution_id")}),
            "verification_results": [
                {"entry": r.get("created") or r.get("updated"),
                 "verification": r.get("verification")}
                for r in results if r.get("verification")] + [
                {"entry":r['saved'], "version":r.get('version'), "body_ref":r.get('body_ref'),
                 "publication":r.get('publication'),
                 "check_refs":[v['check_id'] for v in r.get('validation') or []]}
                for r in relations_saved],
            "knowledge_delta": {
                "entries_created": sorted(after_ids - before_ids),
                "entries_removed": sorted(before_ids - after_ids),
                "entry_count_before":
                    maintenance["knowledge_before"]["entry_count"],
                "entry_count_after": knowledge_after["entry_count"],
                "entry_changes": entry_changes,
                "rule_updates": [{k:r.get(k) for k in ('saved','version','operation','body_ref','new_version')}
                                 for r in relations_saved if r.get('changed',True)],
            },
            # The full AFTER state (value-copied entry refs): the frozen
            # record can reconstruct what the knowledge looked like after
            # this induction, not just which ids moved.
            "knowledge_after": knowledge_after,
        }
        if extra_outcome:
            outcome.update(extra_outcome)
        job = getattr(self, '_active_induction_job', None)
        if job:
            outcome['induction_id'] = job['induction_id']
            outcome['capability_prediction_id'] = job['prediction_id']
            outcome['execution_ids'] = sorted(set(outcome['execution_ids']) | set(job['execution_ids']))
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

    def inspect(self, *, bank: str = "experience",
                task_id: Optional[str] = None,
                strategy_id: Optional[str] = None,
                status: Optional[str] = None,
                episode_id: Optional[str] = None,
                evaluation_id: Optional[str] = None,
                prediction_id: Optional[str] = None,
                entry_id: Optional[str] = None,
                execution_id: Optional[str] = None,
                rule_version=None,
                value_ref=None,
                ) -> Dict[str, Any]:
        """Query the memory layers (read-only).

        One entry point for every read: the two banks, the derived layers
        and the calibration/feedback views. Each ``bank`` value answers a
        different question and none of them substitutes for another:

        - ``experience`` / ``strategic`` / ``archive`` / ``actions`` /
          ``snapshots`` / ``texts``: the stored layers, filtered by
          ``task_id`` / ``strategy_id`` / ``status`` / ``episode_id``;
        - ``predictions``: EVERY prediction generation under one roof —
          legacy ``OutcomePrediction`` (``legacy_predictions``),
          strategy-outcome (``strategy_predictions``) and
          capability-evolution (``capability_predictions``). Pass
          ``prediction_id`` to read ONE prediction across all three
          generations instead of listing them;
        - ``evaluations``: the stored post-hoc strategy-outcome
          evaluations; ``evaluation_id`` reads one of them;
        - ``retention``: the three retention scopes (window / late-check
          grace / archive caps) and the online vs archive counts;
        """
        if bank == 'values':
            if not value_ref:
                raise ValueError('inspect --bank values requires --value REF')
            from or_harness.world_model.conditions import get_value
            return {'bank':'values','value_ref':value_ref,'value':get_value(self.store,value_ref)}
        if bank == "experience":
            if execution_id is not None:
                from or_harness.strategy.evidence_access import resolve_execution,validity
                record=resolve_execution(self,execution_id,include_excluded=True)
                return {"bank":"experience","count":int(record is not None),"records":[record.to_dict()] if record else [],
                        "evidence_validity":validity(self,execution_id),
                        "storage":"live" if self.bank.get(execution_id) or self.bank.get_pending(execution_id) else "retained_reference"}
            records = self.bank.query(task_id=task_id, strategy_id=strategy_id)
            return {"bank": "experience", "count": len(records),
                    "records": [r.to_dict() for r in records]}
        if bank == "strategic":
            if entry_id is not None:
                entry = self.sbank.get(entry_id, version=rule_version)
                from or_harness.strategy.rule_versions import (
                    inspect_versions, utility_summary,
                )
                result = {"bank": "strategic",
                          "count": int(entry is not None),
                          "entries": [entry.to_dict()] if entry else [],
                          "rule_history": inspect_versions(
                              self, entry_id, version=rule_version)}
                if entry is not None:
                    # THREE different questions, kept apart in ONE view:
                    # is it currently offered (publication), how far was its
                    # CLAIM checked (verification), and how did AGENT-
                    # JUDGED USE actually turn out (usage). "validated" is
                    # the second, never the third.
                    utility = result["rule_history"].get("utility") or \
                        utility_summary(self, entry_id, version=rule_version)
                    result["utility"] = utility
                    result["status_view"] = {
                        "publication": utility.get("status"),
                        "legacy_status": entry.status,
                        "verification": utility.get("verification"),
                        "usage": {
                            "source": utility.get("source"),
                            "totals": utility.get("totals"),
                            "latest_management": utility.get(
                                "latest_management"),
                        },
                        "note": (
                            "publication says whether the entry is currently "
                            "offered; verification says how far the CLAIM "
                            "was checked; usage says how the agent judged "
                            "actual adoptions. validated / fact_checked / "
                            "published / an adoption count are NOT proof "
                            "that using the entry helps"),
                    }
                    if entry.rule:
                        track = entry.prediction_track
                        result["forward_calibration"] = {
                            "n_predictions": int(track.n_predictions),
                            "hit_rate": round(track.hit_rate, 4),
                            "n_adoptions": int(track.n_adoptions),
                            "note": (
                                "forward calibration of a DECLARED prediction "
                                "track, NOT a utility statistic; n_predictions "
                                "= 0 means not estimated, never a measured "
                                "zero effect. Read the usage summary instead"),
                        }
                return result
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
            # Asking for ONE prediction is a KEY LOOKUP, so it is answered
            # as one: listing every prediction in three logs and then
            # appending the target made a single read cost the whole
            # history, and buried the answer in unrelated output.
            if prediction_id is not None:
                single = self.read_prediction(prediction_id)
                if single is None:
                    raise ValueError(
                        f"unknown prediction_id {prediction_id!r}")
                return {"bank": "predictions", "count": 1,
                        "prediction": single}
            # Listing mode: EVERY prediction generation, each under its own
            # key: the legacy M2 OutcomePrediction log, the wm-so/1
            # strategy-outcome channel that calibration aggregates, and the
            # historical capability payloads. The retired channel remains
            # a raw read and never invokes a service or updates its records.
            legacy = self.predictions_query(task_id=task_id,
                                            episode_id=episode_id)
            strategy = self.strategy_predictions.query(
                task_id=task_id, episode_id=episode_id)
            capability = [self.store.loads(raw) for raw in self.store.capability_predictions_for(
                task_id=task_id, episode_id=episode_id)]
            return {
                "bank": "predictions",
                "count": (len(legacy) + len(strategy) + len(capability)),
                "legacy_predictions": [p.to_dict() for p in legacy],
                "strategy_predictions": [p.to_dict() for p in strategy],
                "capability_predictions": capability,
            }
        if bank == "evaluations":
            if evaluation_id is not None:
                single = self.get_strategy_evaluation(evaluation_id)
                if single is None:
                    raise ValueError(
                        f"unknown evaluation_id {evaluation_id!r}")
                return {"bank": "evaluations", "count": 1,
                        "evaluation": single}
            evaluations = self.strategy_prediction_evaluations(
                task_id=task_id, episode_id=episode_id)
            return {"bank": "evaluations", "count": len(evaluations),
                    "evaluations": evaluations}
        if bank == "retention":
            return {"bank": "retention", **self.calibration_retention()}
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
                         "actions|snapshots|predictions|evaluations|"
                         "retention|texts")

    def read_prediction(self, prediction_id: str
                        ) -> Optional[Dict[str, Any]]:
        """Read ONE prediction, whichever generation it belongs to.

        The three prediction stores are separate logs, so the lookup is
        tried in a fixed order and the result names its own generation —
        an id is never silently resolved to a different kind of record.
        """
        stored = self.strategy_predictions.get(prediction_id)
        if stored is not None:
            return {"kind": "strategy_outcome",
                    "prediction": stored.to_dict()}
        raw_capability = self.store.get_capability_prediction(prediction_id)
        if raw_capability is not None:
            return {"kind": "capability_evolution", "retired": True,
                    "prediction": self.store.loads(raw_capability)}
        legacy = self.get_prediction(prediction_id)
        if legacy is not None:
            return {"kind": "legacy_outcome",
                    "prediction": legacy.to_dict()}
        return None

    def _apply_effect_attributions(
            self, attributions: Optional[Sequence[Dict[str, Any]]], *,
            dry_run: bool = False,
            defer_lifecycle: bool = False) -> List[Dict[str, Any]]:
        """Record the agent's OWN use-effect attributions on entries.

        Each item is ``{"entry_id", "verdict", "note"?, "execution_id"?,
        "lifecycle_action"?, "evidence_execution_ids"?}``. The verdict is one
        of ``helped|neutral|unrelated|refuting`` and the author is stamped
        ``by="agent"`` — the framework NEVER writes one. This is the channel
        that lets a qualitative technique (no numeric prediction) carry a
        real judgement of how its use turned out, distinct from
        ``verification_state`` (how far the CLAIM was checked).

        r21 MANAGEMENT: an OPTIONAL ``lifecycle_action``
        (``promote|demote|keep``) is the agent's explicit maturity request.
        When present, the framework performs an ADMINISTRATIVE check only:

        - the entry must exist;
        - a named ``execution_id`` must be an actually recorded execution;
        - every ``evidence_execution_ids`` must have DECLARED adoption of
          this entry number (a recall-only or shared-name reference is
          refused);
        - the request must carry a non-empty ``note`` (the agent's reason)
          AND at least one real execution reference.

        The framework never certifies that the accumulated evidence really
        supports the move — that is the agent's judgement. It never rewrites
        ``verification_state`` and never touches the claim/number/evidence.

        ``defer_lifecycle``: when the caller will run ``revise`` right after
        (which replays the attributions and applies the management events in
        chronological order), the live status change is DEFERRED to that
        pass so the entry is written once and the result/delta/snapshot see a
        single consistent state. Otherwise the action is applied here.

        Returns the recorded items (with the assigned time), or, on
        ``dry_run``, the items that WOULD be recorded.
        """
        recorded: List[Dict[str, Any]] = []
        validated = [self._validate_effect_attribution(raw)
                     for raw in (attributions or [])]
        for plan in validated:
            item = plan["item"]
            recorded.append({"entry_id": plan["entry_id"], **item})
            if dry_run:
                continue
            entry = self.sbank.get(plan["entry_id"])
            if entry.rule:
                from or_harness.strategy.rule_versions import record_effect
                recorded[-1] = record_effect(self, entry.entry_id, item)
                continue
            entry.effect_attribution.append(item)
            self.sbank.update(entry)
            if plan["lifecycle_action"] and not defer_lifecycle:
                self.sbank.apply_lifecycle_action(
                    plan["entry_id"], plan["lifecycle_action"],
                    reason=item["note"])
        return recorded

    def _record_validated_attributions(
            self, validated: Sequence[Dict[str, Any]], *,
            dry_run: bool = False) -> List[Dict[str, Any]]:
        """Record PRE-VALIDATED attributions WITHOUT applying management.

        Used by the ``induce`` paths, which run ``revise`` right after: the
        recorded items are replayed there in chronological order (so the
        management event lands exactly once and the entry is written once).
        Returns the recorded items (or the would-be items on ``dry_run``).
        """
        recorded: List[Dict[str, Any]] = []
        for plan in validated:
            item = plan["item"]
            recorded.append({"entry_id": plan["entry_id"], **item})
            if dry_run:
                continue
            entry = self.sbank.get(plan["entry_id"])
            if entry is None:
                continue
            if entry.rule:
                from or_harness.strategy.rule_versions import record_effect
                recorded[-1] = record_effect(self, entry.entry_id, item)
                continue
            entry.effect_attribution.append(item)
            self.sbank.update(entry)
        return recorded

    def _record_attributions_at_close(
            self, validated: Sequence[Dict[str, Any]]
            ) -> List[Dict[str, Any]]:
        """Record the agent's utility review from a CLOSING call, once.

        Same channels as ``induce --attribute-effect``: a versioned entry
        appends a ``record_effect`` event (and its management event), a legacy
        entry appends to ``effect_attribution`` and applies its management
        decision here (the offline ``revise`` pass would otherwise replay it
        later — the same decision, not a second one). An identical attribution
        submitted twice by an idempotent re-close is recorded once.
        """
        from or_harness.strategy.rule_versions import record_effect
        from or_harness.world_model.conditions import digest
        recorded: List[Dict[str, Any]] = []
        keys = ("verdict", "execution_id", "note", "lifecycle_action",
                "evidence_execution_ids")
        for plan in validated:
            item = plan["item"]
            entry = self.sbank.get(plan["entry_id"])
            if entry is None:
                continue
            if entry.rule:
                recorded.append(record_effect(self, entry.entry_id, item))
                continue
            signature = digest({key: item.get(key) for key in keys})
            previous = [raw for raw in entry.effect_attribution
                        if isinstance(raw, dict)
                        and str(raw.get("execution_id"))
                        == str(item.get("execution_id"))]
            if previous and digest(
                    {key: previous[-1].get(key) for key in keys}) == signature:
                recorded.append({"entry_id": entry.entry_id,
                                 **previous[-1], "idempotent": True})
                continue
            action = plan["lifecycle_action"]
            if action is None and item.get("verdict") == "refuting":
                # The existing default for a NEGATIVE judgement (same rule as
                # the offline lifecycle replay): a refuting verdict pauses the
                # version unless an explicit management action says otherwise.
                action = "demote"
            if action is not None:
                item = {**item, "applied_at": item.get("at")}
            entry.effect_attribution.append(item)
            self.sbank.update(entry)
            recorded.append({"entry_id": entry.entry_id, **item})
            if action is not None:
                entry, transitions = self.sbank.apply_lifecycle_action(
                    entry.entry_id, action,
                    reason=item.get("note") or "agent review at close-out")
                if transitions:
                    recorded[-1]["transitions"] = transitions
        return recorded

    def _validate_effect_attribution(self, raw: Dict[str, Any]
                                     ) -> Dict[str, Any]:
        """Validate ONE effect attribution (verdict + optional management).

        Raises ``ValueError`` (a correctable structural error) when the
        payload is malformed, the entry is unknown, a referenced execution is
        not recorded, or a management request lacks real adopted evidence.
        No logical certification is performed.
        """
        if not isinstance(raw, dict):
            raise ValueError("effect_attribution must be a JSON object")
        entry_id = str(raw.get("entry_id") or "").strip()
        verdict = str(raw.get("verdict") or "").strip()
        if not entry_id or not verdict:
            raise ValueError(
                "effect_attribution requires entry_id and verdict")
        if verdict not in EFFECT_ATTRIBUTION_VERDICTS:
            raise ValueError(
                f"unknown effect_attribution verdict {verdict!r} "
                f"(expected one of {list(EFFECT_ATTRIBUTION_VERDICTS)})")
        entry = self.sbank.get(entry_id)
        if entry is None:
            raise ValueError(f"unknown entry_id {entry_id!r}: an effect "
                             "attribution must name a real entry")
        execution_id = (str(raw["execution_id"]).strip()
                        if raw.get("execution_id") else None)
        if execution_id and self._recorded_execution(execution_id) is None:
            raise ValueError(
                f"execution_id {execution_id!r} is not a recorded execution: "
                "an effect attribution may only cite real facts")
        action = raw.get("lifecycle_action")
        action = str(action).strip() if action else None
        note = (str(raw.get("note")).strip() if raw.get("note") else None)
        evidence_ids = [str(e).strip() for e
                        in (raw.get("evidence_execution_ids") or [])
                        if str(e).strip()]
        if action is not None:
            if action not in LIFECYCLE_ACTIONS:
                raise ValueError(
                    f"unknown lifecycle_action {action!r} "
                    f"(expected one of {list(LIFECYCLE_ACTIONS)})")
            # A management request must rest on REAL adopted use, not on a
            # bare status string: it needs a reason and at least one cited
            # execution that DECLARED adoption of THIS entry.
            if not note:
                raise ValueError(
                    f"lifecycle_action {action!r} requires a non-empty note: "
                    "state the agent's reason from the accumulated use")
            if entry.rule and entry.rule.get('merged_into'):
                raise ValueError('entry was merged into '+str(entry.rule['merged_into'])+'; manage that merge target explicitly')
            cited = list(dict.fromkeys(evidence_ids
                                       + ([execution_id] if execution_id
                                          else [])))
            if not cited:
                raise ValueError(
                    f"lifecycle_action {action!r} requires execution_id or "
                    "evidence_execution_ids naming real adoptions")
            adopted = [e for e in cited
                       if self._declared_adoption(entry_id, e)]
            if not adopted:
                raise ValueError(
                    f"lifecycle_action {action!r} cites no execution that "
                    f"declared adoption of entry {entry_id!r}: a recall-only "
                    "or shared-name reference is not adoption (declare it "
                    "with `--used-entry-ids`)")
        item = {
            "by": "agent",
            "verdict": verdict,
            "execution_id": execution_id,
            "note": note,
            "at": time.time(),
        }
        if action is not None:
            item["lifecycle_action"] = action
            item["evidence_execution_ids"] = evidence_ids
        if entry.rule:
            from or_harness.strategy.rule_versions import adopted_version
            cited=list(dict.fromkeys(evidence_ids+([execution_id] if execution_id else [])))
            versions={adopted_version(self,r,entry_id) for eid in cited
                      if (r:=self._recorded_execution(eid)) is not None and self._declared_adoption(entry_id,eid)}-{None}
            requested=raw.get('version')
            if execution_id:
                primary=self._recorded_execution(execution_id)
                primary_version=adopted_version(self,primary,entry_id) if primary else None
                if requested is not None and requested!=primary_version:
                    raise ValueError('primary execution adoption version is '+str(primary_version)+'; auxiliary evidence cannot replace it')
                if action and primary_version is None:
                    raise ValueError('primary execution has no frozen adoption version for this entry')
                requested=primary_version
            if requested is not None and requested not in versions:
                raise ValueError('version must match one of the cited actual adoptions: '+str(sorted(versions)))
            if len(versions)>1 and requested is None:
                raise ValueError('multiple adopted versions; select one existing version: '+str(sorted(versions)))
            item['version']=primary_version if execution_id else requested if requested is not None else next(iter(versions),None)
            item['evidence_execution_ids']=cited
        return {"entry_id": entry_id, "item": item,
                "lifecycle_action": action}

    def _recorded_execution(self, execution_id: str):
        """The stored execution record for ``execution_id``, or None."""
        from or_harness.strategy.evidence_access import resolve_execution
        record=resolve_execution(self,execution_id)
        if record is None:
            return None
        if str(record.source) not in ("executed", "staged"):
            return None
        return record

    def _declared_adoption(self, entry_id: str, execution_id: str) -> bool:
        """True when ``execution_id`` DECLARED adoption of ``entry_id``."""
        record = self._recorded_execution(execution_id)
        if record is None:
            return False
        used = (record.execution_features or {}).get("used_entries")
        if not isinstance(used, dict):
            return False
        return str(entry_id) in [str(e) for e
                                 in (used.get("known_entry_ids") or [])]


    def retire(self, entry_id: str, reason: str, *,
               outcome: str = "retired",
               evidence_execution_ids: Optional[Sequence[str]] = None
               ) -> Dict[str, Any]:
        """Retire ONE entry to the cold archive on the agent's decision.

        The reason and any cited evidence are KEPT on the archive card (the
        framework never derives a utility verdict from them, and the old
        forward-calibration numbers are labelled as calibration — not as
        proof the rule was useful). The read-time usage summary is archived
        beside them so a later reader sees what the retirement was based on.
        """
        from or_harness.strategy.rule_versions import utility_summary
        cited = [str(item).strip() for item in
                 (evidence_execution_ids or []) if str(item).strip()]
        for execution_id in cited:
            if self._recorded_execution(execution_id) is None:
                raise ValueError(
                    f"evidence execution {execution_id!r} is not a recorded "
                    "execution: retirement evidence must cite real facts")
        try:
            usage = utility_summary(self, entry_id)
        except ValueError:
            usage = None
        card = self.sbank.retire(entry_id, reason=reason, outcome=outcome,
                                 evidence_execution_ids=cited, usage=usage)
        # The entry left the hot store, so its vector must leave the index
        # too: a vector that outlives its record would surface a claim that
        # no longer exists — and it would keep occupying a retrieval slot
        # even though recall filters the entry out afterwards.
        #
        # A bare ``sync_entries()`` is NOT enough: it walks the entries still
        # IN the bank, so a retired id never enters its removal list. The
        # retired id is passed EXPLICITLY through the same existing
        # interface, which removes a vector whose entry can no longer be
        # resolved (no second wrapper function is needed).
        self.index_sync.sync_entries([entry_id])
        return {"retired": entry_id, "cold_archive_card": card.to_dict(),
                "utility": usage, "evidence_execution_ids": cited}

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
        from or_harness.strategy.evidence_access import resolve_execution,set_validity
        record=resolve_execution(self,execution_id,include_excluded=True)
        if self.bank.get(execution_id) is not None:
            record=self.bank.exclude(execution_id,reason,superseded_by=superseded_by)
        elif record is not None and record.source!='excluded':
            if superseded_by and resolve_execution(self,superseded_by) is None:
                raise ValueError('superseded_by execution does not exist')
            record.source='excluded'
            record.execution_features['correction']={'excluded':True,'reason':str(reason),
                'superseded_by':superseded_by,'excluded_at':time.time()}
        else:
            raise ValueError('unknown or already excluded execution '+str(execution_id))
        evidence_validity=set_validity(self,record,False,reason)
        from or_harness.strategy.rule_versions import refresh_execution_support,sync_rule_entries
        dependencies=refresh_execution_support(self,execution_id)
        affected=[v['entry_id'] for v in dependencies['affected_versions']]
        if affected:sync_rule_entries(self,affected,affected)
        unindexed = None
        if self.embedding_index is not None:
            from or_harness.strategy.embedding_index import LAYER_EXECUTION
            try:
                unindexed = self.embedding_index.remove(LAYER_EXECUTION,
                                                        [execution_id])
            except Exception as exc:  # noqa: BLE001 - never block the exclusion
                unindexed = {"removed": 0,
                             "deferred": f"{type(exc).__name__}: {exc}"}
        # An exclusion can change a calibration sample (a withdrawn fact is
        # no longer evidence): republish when the affected episode is in
        # the window.
        republished = self._republish_calibration_if_in_window(
            str(record.task_id), self._episode_of(execution_id))
        result: Dict[str, Any] = {"excluded": execution_id,
                                  "reason": str(reason),
                                  "superseded_by": superseded_by,
                                  "correction": record.execution_features.get(
                                      "correction"),
                                  "index": unindexed}
        if republished is not None:
            result["calibration_republished"] = {
                "reason": "the exclusion can change a calibration sample",
                "n_evaluated": republished.get("n_evaluated")}
        result['rule_validation']=dependencies
        result['evidence_validity']=evidence_validity
        return result

    def restore_execution(self, execution_id: str, reason: str
                          ) -> Dict[str, Any]:
        """Reverse :meth:`exclude_execution`: the fact counts again.

        A second explicit statement (an exclusion can itself be wrong). The
        correction history is kept on the fact. Re-indexing is left to the
        caller via ``rebuild_index`` — a restore is rare and explicit, so it
        does not silently rewrite derived index state."""
        from or_harness.strategy.evidence_access import resolve_execution,set_validity
        record=resolve_execution(self,execution_id,include_excluded=True)
        if self.bank.get(execution_id) is not None:
            record=self.bank.restore(execution_id,reason)
        elif record is not None and record.source=='excluded':
            record.source='executed'
            record.execution_features.setdefault('correction',{}).update(restored=True,
                restore_reason=str(reason),restored_at=time.time())
        else:
            raise ValueError('unknown or not excluded execution '+str(execution_id))
        evidence_validity=set_validity(self,record,True,reason)
        from or_harness.strategy.rule_versions import refresh_execution_support,sync_rule_entries
        dependencies=refresh_execution_support(self,execution_id)
        affected=[v['entry_id'] for v in dependencies['affected_versions']]
        if affected:sync_rule_entries(self,affected,affected)
        republished = self._republish_calibration_if_in_window(
            str(record.task_id), self._episode_of(execution_id))
        result: Dict[str, Any] = {"restored": execution_id,
                                  "reason": str(reason),
                                  "correction": record.execution_features.get(
                                      "correction")}
        if republished is not None:
            result["calibration_republished"] = {
                "reason": "the restore can change a calibration sample",
                "n_evaluated": republished.get("n_evaluated")}
        result['rule_validation']=dependencies
        result['evidence_validity']=evidence_validity
        return result

    def _episode_of(self, execution_id: str) -> Optional[str]:
        """The episode an execution's action belongs to, or None."""
        action = self.actions.by_execution(execution_id)
        return action.episode_id if action is not None else None

    def _refresh_rule_evidence(self,execution_id):
        """Refresh referenced facts only; the agent owns utility/publication decisions."""
        from or_harness.strategy.evidence_access import resolve_execution,refresh_retained
        record=resolve_execution(self,execution_id,include_excluded=True)
        retained=refresh_retained(self,record) if record else {}
        return {'affected_versions':[],'index_entry_ids':[],'authority':'agent',**retained}

    def _republish_calibration_if_in_window(self, task_id: str,
                                            episode_id: Optional[str]
                                            ) -> Optional[Dict[str, Any]]:
        """Rebuild the published summary if a correction can affect it.

        The single trigger every correction channel calls. It is cheap when
        it does nothing (one indexed window lookup) and rebuilds only when
        the affected episode is actually in the calibration window.
        """
        from or_harness.world_model.episode_closeout import (
            republish_if_in_window,
        )
        try:
            return republish_if_in_window(self, task_id, episode_id)
        except Exception:  # a republish failure must never lose the fact
            return None

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

    def enforce_window(self, *, window_episodes: Optional[int] = None,
                       open_grace_days: Optional[float] = None,
                       dry_run: bool = False) -> Dict[str, Any]:
        """Bound the Execution Evidence Bank to a recent window of episodes.

        The Evidence Bank keeps a bounded RECENT window of COMPLETE episodes
        (default 800, ``OR_EVIDENCE_WINDOW_EPISODES``). Episodes inside the
        calibration window, episodes awaiting a task verdict within the
        late-check grace period, and young unclosed episodes are NEVER
        evicted. The whole episode leaves at once, so a contrast/repair chain
        is never split.

        Run this LAST in a maintenance sequence: calibration publish/archive
        and induction must have taken their opportunity first. ``dry_run``
        reports the candidates and writes nothing (no row, no vector, no
        text). Re-running is idempotent.

        The returned ``timer`` semantics are the point: an eviction is a
        historical source reference EXPIRING. It never refutes a claim and is
        never confused with ``exclude`` (a withdrawal). The evidence bound
        covers the Evidence Bank — it does NOT make the whole project
        directory bounded.
        """
        from or_harness.world_model.episode_closeout import (
            EvidenceWindowPolicy, enforce_evidence_window,
        )
        policy = EvidenceWindowPolicy.from_env(
            window_episodes=window_episodes, open_grace_days=open_grace_days)
        return enforce_evidence_window(self, policy=policy, dry_run=dry_run)

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
            "executor": self.executor.capabilities() if hasattr(self.executor, "capabilities") else None,
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

    def _record_used_entries(self, record: ExecutionRecord,
                             used_entry_ids: Optional[Sequence[Any]]
                             ) -> None:
        """Record the knowledge NUMBERS the agent declares it ADOPTED.

        Three states are kept apart, never inferred from one another:

        - **recalled**: an entry was surfaced to the agent (its
          ``last_consulted_at`` moved on the recall path);
        - **adopted**: the agent NAMES the entry numbers this attempt relied
          on (``used_entry_ids`` here) — a DECLARATION, not a guess from a
          shared name;
        - **outcome**: this attempt's own result (``quality`` / task check).

        An empty list is VALID and recorded as ``[]`` ("adopted no prior
        knowledge"), so "adopted nothing" is distinguishable from "did not
        say". A non-empty list is validated against the bank and
        de-duplicated by number; an id that names no entry is recorded as
        ``unknown`` (a citation, not a fabricated fact), and existence is
        the framework's ADMINISTRATIVE check — it does not judge whether the
        adoption helped (that is the outcome's job).
        """
        requested_versions = {}
        declared = []
        for raw in used_entry_ids or []:
            if isinstance(raw, dict):
                entry_id, version = str(raw.get("entry_id") or "").strip(), raw.get("version")
            else:
                entry_id, sep, version = str(raw).strip().partition("@")
                version = version if sep else None
            if entry_id:
                declared.append(entry_id)
                if version is not None:
                    requested_versions[entry_id] = int(version)
        # De-duplicate by number, preserving first-seen order.
        declared = list(dict.fromkeys(declared))
        known: List[str] = []
        unknown: List[str] = []
        versions = {}
        rule_versions = {}
        from or_harness.world_model.conditions import put_value
        with self.store.transaction() as conn:
            for entry_id in declared:
                entry = self.sbank.get(entry_id, version=requested_versions.get(entry_id))
                if entry is not None:
                    known.append(entry_id)
                    from or_harness.strategy.rule_versions import frozen_entry
                    versions[entry_id] = put_value(conn, self.store, "knowledge", frozen_entry(entry))
                    if entry.rule:
                        rule_versions[entry_id] = entry.rule["view_version"]
                else:
                    unknown.append(entry_id)
        note = ("entry numbers the agent declared this attempt ADOPTED — a "
                "declaration, not a guess from a shared name; whether the "
                "adoption helped is judged by this attempt's own task check "
                "and calibration, never by the declaration")
        if not declared:
            note = ("the agent declared that NO prior knowledge was adopted; "
                    "an empty adoption is a fact, distinct from recall and "
                    "from an unstated adoption")
        record.execution_features["used_entries"] = {
            "entry_ids": declared,
            "known_entry_ids": known,
            "unknown_entry_ids": unknown,
            "versions": versions,
            "rule_versions": rule_versions,
            "basis": "declared_frozen", "state": "declared" if declared else "declared_none",
        }

    def _check_predictions(self, record: ExecutionRecord) -> List[Dict[str, Any]]:
        """Frozen forward checks: this execution against the DECLARED
        predictions of the entries it ADOPTED, as EVIDENCE.

        Online the harness only accumulates: each check is computed against
        the interval in force at this moment and written onto the fact
        (``execution_features.quality_feedback``). No entry is promoted,
        demoted, tightened, or awakened here — ``InductionEngine.revise``
        replays these checks at the next offline induction.

        ATTRIBUTION — adoption, never a name:
        - a check exists ONLY for an entry the record DECLARED it adopted
          (``execution_features.used_entries.known_entry_ids``). A shared
          strategy name is not adoption, and being recalled is not adoption:
          an entry whose id differs from the executed strategy
          (``method:monotone_reduction`` vs ``milp_pulp_cbc``) is checked
          normally when the agent adopts it by NUMBER.
        - a record that did NOT declare its adoption produces NO forward
          checks. "Did not say" must not silently fall back to a name match.

        CALIBRATION — only against a DECLARED prediction:
        - an entry that never declared a quality prediction
          (``quality_estimated=False``) has an interval that is a default,
          not a forecast. No hit/miss is computed against it (its default
          interval would otherwise score EVERY observation as a hit). Its
          adoption, cost and failures are still recorded as evidence — the
          adoption outcome accumulates independently of prediction
          calibration, and whether a failure REFUTES the knowledge is the
          agent's analysis, never an automatic verdict.

        Isolation rules (unchanged):
        - only attempt-scope executions produce checks;
        - dormant entries are eligible (a matching execution is evidence
          about the pattern; waking is an offline decision).
        """
        if record.measurement_scope != "attempt":
            return []
        observed = quality_score(record)
        used = record.execution_features.get("used_entries")
        if not isinstance(used, dict):
            # Adoption was never declared: the record does not say what it
            # adopted, so there is no basis for a forward check. An entry is
            # never inferred from a shared strategy name.
            return []
        declared = [str(e) for e in (used.get("known_entry_ids") or [])]
        if not declared:
            # An explicit EMPTY adoption (``entry_ids: []``) is a fact:
            # nothing was adopted, so there is nothing to check.
            return []
        events: List[Dict[str, Any]] = []
        for entry_id in declared:
            ref = (used.get("versions") or {}).get(entry_id)
            if ref:
                from or_harness.world_model.conditions import get_value
                entry = StrategicEntry.from_dict(get_value(self.store, ref))
            else:
                entry = self.sbank.get(entry_id)  # legacy declaration without a frozen value
            if entry is None:
                continue  # an unknown number is a citation, not a fact
            if not entry.quality_estimated:
                # No DECLARED prediction: the interval is a default, not a
                # forecast. Record the observation WITHOUT a hit/miss so the
                # adoption outcome and cost stay visible while prediction
                # calibration stays unknown.
                events.append({
                    "entry_id": entry.entry_id,
                    "predicted": None,
                    "interval": None,
                    "observed": round(observed, 4),
                    "hit": None,
                    "adopted": True,
                    # The objective outcome of THIS adoption, so a maintenance
                    # consumer can weigh the use without a hit/miss.
                    "outcome": {
                        "observed_quality": round(observed, 4),
                        "task_check": task_check_state(record),
                        "status": (record.quality or {}).get("status"),
                        "failure_classes": (record.execution_features or {})
                        .get("failure_classes"),
                    },
                    "note": ("this entry declared no quality prediction, so "
                             "no interval hit/miss is computed (a default "
                             "interval is not a forecast); the observation "
                             "is recorded as ADOPTION evidence (usage fact, "
                             "not calibration)"),
                })
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
                "adopted": True,
                "outcome": {
                    "observed_quality": round(observed, 4),
                    "task_check": task_check_state(record),
                    "status": (record.quality or {}).get("status"),
                    "failure_classes": (record.execution_features or {})
                    .get("failure_classes"),
                },
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
          retries this attempt adds (observed from the task/episode's real
          execution chain, not a running total) — 0/1/1 totals two retries.
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
        # Wall-clock spans never appear in total_cost: attempts may overlap,
        # and end-to-end wait is a separate harness-supplied fact (see below).
        total_cost = {d: (round(total[d], 4) if n_measured[d] > 0 else None)
                      for d in COST_DIMENSIONS
                      if d not in NON_CUMULATIVE_DIMENSIONS}
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

    def _cost_completeness(self, record: ExecutionRecord
                           ) -> Optional[Dict[str, Any]]:
        """What this record's cost data can and cannot support.

        Reported at record time because that is the moment the harness still
        has the numbers at hand: ``llm_tokens`` and ``tool_calls`` are
        invisible to the sandbox and can only be declared by the caller, and
        a dimension left unmeasured here will not support a strategic-entry
        cost claim (``induce`` withholds an incomplete dimension) or a cost
        prediction. Returns None when every dimension is measured — there is
        nothing to warn about.

        This never blocks recording: the fact is honest as it stands, and an
        unmeasured dimension is recorded as UNKNOWN rather than fabricated.
        """
        measured = record.cost.measured_dims()
        missing = [d for d in COST_DIMENSIONS if d not in measured]
        if not missing:
            return None
        fixes = []
        if "llm_tokens" in missing:
            fixes.append("llm_tokens=<actual tokens the LLM call used>")
        if "tool_calls" in missing:
            fixes.append("tool_calls=<count of ALL tool invocations: shell "
                         "commands, file reads/writes, sandbox runs, solver "
                         "calls>")
        if "retries" in missing:
            fixes.append("retries=<observed 0/1 retry increment for this attempt>")
        note = ("unmeasured dimensions are recorded as UNKNOWN, never as zero. "
                "They will not support a strategic-entry cost claim or a cost "
                "prediction until every supporting record measures them.")
        if fixes:
            note += (" Backfill with `orx amend-cost <execution_id> --override "
                     + ",".join(fixes) + "` (amends an already-recorded fact "
                     "in place; re-applying is idempotent).")
        return {
            "measured": sorted(measured),
            "missing": missing,
            "note": note,
            "lower_bounds": {
                k: v for k, v in record.execution_features.items()
                if k == "tool_calls_lower_bound"},
        }

    def _prior_failures(self, record: ExecutionRecord) -> List[ExecutionRecord]:
        """Failed executions (bank + staged) for the same task, excluding
        this record itself.

        "Failed" means the ATTEMPT did not produce a usable answer, which is
        not the same as "the solver refused": a model can be written wrong,
        solved to a legal optimal status, and still FAIL the task check —
        two vehicles where the task needs the answer to cost 200. That case
        is the main modeling-repair story, so it counts here too. A record
        whose task check is ``failed`` is a prior failure even though its
        solver quality looked fine; a record whose check ``passed`` or was
        never run is not.

        ``insufficient`` is deliberately NOT a failure: a check that could
        not decide says nothing about whether the answer was usable, and
        treating it as failure would manufacture repair chains out of
        unverified attempts.
        """
        def _is_failed(rec: ExecutionRecord) -> bool:
            if not rec.quality.get("feasible", False):
                return True
            return task_check_state(rec) == "failed"

        prior = [r for r in self.bank.query(task_id=record.task_id)
                 if r.execution_id != record.execution_id and _is_failed(r)]
        prior += [p for p in self.bank.pending(task_id=record.task_id)
                  if p.execution_id != record.execution_id and _is_failed(p)]
        return prior

    # -- task-result checks (solver success vs. task correctness) ---------------

    def check_task_result(self, execution_id: str,
                          check: Optional[Dict[str, Any]] = None,
                          *, episode_id: Optional[str] = None,
                          verification: Optional[Dict[str, Any]] = None
                          ) -> Dict[str, Any]:
        """Check whether an execution's ANSWER satisfies the ORIGINAL task.

        The step the framework was missing. ``orx execute`` answers "did the
        solver solve the model it was given" (a legal status, a finite
        objective, a gap). It cannot answer "is this a valid answer to the
        task", and the difference is not academic: a relaxed LP answered with
        fractional values reports ``optimal`` with ``gap=0`` and would
        otherwise become a positive quality sample in recall, the conditional
        statistics, the world-model feedback and offline induction.

        ``check`` declares what the framework may verify (see
        :func:`or_harness.strategy.verification.verify_task_result`):
        ``reference_objective`` (+``tolerance``), ``reference_status``,
        ``integer`` (``{"variables": [...] | omitted, "tolerance": t}``),
        ``recompute_objective`` (``{"coefficients": {...}, "constant": k}``),
        ``semantic_probe`` (``{"path", equals|min|max|in}``), and an optional
        ``intent`` (``relaxation`` / ``intermediate``) marking an execution
        whose answer is deliberately NOT the task's answer.

        Optional top-level ``g_attribution`` is validated before writing and
        stored separately from that generic rule. Invalid declarations raise
        ``ValueError``; valid Unknown assessments can close normally. The
        returned ``g_assessment_ref`` is also linked by the verify action.

        Three verdicts, and none of them is a default:

        - ``passed`` — every DECLARED basis held on the recorded values. It
          covers those bases only: the fixed checker scope travels in
          ``report.scope.note`` and is not expanded into per-task findings.
          A pass does NOT block a new ``plan-next`` when the agent's own
          self-check finds a problem.
        - ``failed`` — a declared basis ran and did not hold. The execution
          is NOT demoted or rewritten: its observed quality and cost stand,
          and it stays in the evidence set (the cost is real, the failure is
          raw material). What changes is that it can no longer count as a
          success sample (see :func:`or_harness.core.schema.task_check_state`).
          The verdict does not establish the CAUSE (task reading vs model vs
          implementation vs the check itself) and does not by itself require
          a new solve: a corrected check or missing evidence may be supplied
          on the SAME execution.
        - ``insufficient`` — no basis declared, no solution vector recorded, a
          needed variable missing, or the execution produced no usable
          result. NOT a pass, NOT a failure, and NEVER a request that the
          user supply a reference.

        The agent owns the pre-submission self-check: reading the original
        task, verifying the model/code/solution, choosing a calculation or
        validation method, and deciding whether to commit, repair the check,
        or plan a new attempt. This method supplies the deterministic part and
        the material; it does not choose that decision for the agent.

        ``verification`` optionally carries the evidence of a verification the
        AGENT ran on this execution (an independent script, a manual
        recomputation, an enumeration): ``{"script": PATH, "output": PATH,
        "result": {"ran": bool, "exit_code": int, "assertions": [...]}}`` or
        ``result_path`` (a JSON file). Files are captured as IMMUTABLE
        artifacts and the normalised record is labelled ``agent_reported``
        and stored under ``execution_features.verification_evidence`` —
        separate from the framework's ``task_check``. It creates no solving
        attempt, never overwrites the solve ``result.json``, and a satisfied
        agent assertion is never reported as a framework check.

        Recording (the same two channels the rest of the harness uses):

        1. a ``verify`` ACTION — the framework really performed this check, so
           it is logged as an executed action (audit trail, per-call history);
        2. ``execution_features.task_check`` on the execution — a narrow
           annotation written through ``ExperienceBank.set_task_check``, which
           works for staged (unrecorded) and recorded executions alike, so a
           check may legitimately arrive AFTER the episode closed.

        Also returned: ``reflection_material`` — the self-check material for
        this execution (task text + digest with an inspect reference, the
        candidate and performed method, the captured script/result artifact
        references, the captured solution and its reference, the full check
        report, agent-reported verification evidence and prior attempts as
        references). Locating the cause is the agent's job — the framework
        does not classify the failure as a modeling mistake, does not rebuild
        the model, and never relaxes the task to match a reference value.
        """
        from or_harness.strategy.verification import verify_task_result
        from or_harness.strategy.evidence_access import resolve_execution
        record=resolve_execution(self,execution_id)
        staged=self.bank.get_pending(execution_id) is not None
        retained=record is not None and self.bank.get(execution_id) is None and not staged
        if record is None:
            raise ValueError(f'unknown execution_id {execution_id!r}: unavailable or withdrawn evidence')
        task_id = str(record.task_id)
        if check is not None and not isinstance(check, dict):
            raise ValueError('check must be a JSON object')
        check = check if check is not None else {}
        from or_harness.world_model.g_signal import (
            checked_attribution, require_g_attribution_location, save_g_assessment,
            g_observation_convention,
        )
        require_g_attribution_location(check)
        # Resolve the producing action, not the newest verify action linked
        # to the same execution. Explicit ownership claims must agree.
        execution_action = self.actions.get(record.action_id) if record.action_id else None
        if execution_action is None:
            execution_action = next((a for a in self.actions.query(task_id=task_id)
                if a.action_type == 'execute_strategy' and a.source != 'hypothetical'
                and a.linked_execution_id == execution_id), None)
        execution_action_id = execution_action.action_id if execution_action else record.action_id
        resolved_episode = episode_id
        if execution_action is not None:
            if episode_id is not None and episode_id != execution_action.episode_id:
                raise ValueError('episode_id does not match the action that produced this execution')
            resolved_episode = execution_action.episode_id
        attribution = None
        prediction_id = (execution_action.params or {}).get('prediction_id') if execution_action else None
        if 'g_attribution' in check:
            attribution = checked_attribution(record, check['g_attribution'])
            issues = list(attribution.get('issues') or [])
            for key, expected in (('episode_id', resolved_episode),
                                  ('execution_action_id', execution_action_id),
                                  ('strategy_id', record.strategy_id)):
                if key in attribution and attribution[key] != expected:
                    issues.append(f'{key} does not match this execution')
            supplied_prediction = attribution.get('prediction_id')
            if supplied_prediction is not None and supplied_prediction != prediction_id:
                declared = self.strategy_predictions.get(supplied_prediction) if isinstance(supplied_prediction, str) else None
                if (prediction_id is not None or declared is None or not execution_action_id or
                        declared.trace.model_info.get('bound_action_id') != execution_action_id):
                    issues.append('prediction_id is not bound to this execution')
                else:
                    prediction_id = supplied_prediction
            if issues:
                raise ValueError('invalid g_attribution: ' + '; '.join(issues) +
                    '. Use unknown fidelity/acceptance_scope and available evidence when '
                    'attribution is uncertain, or omit g_attribution for a generic task check.')
        # The AGENT's own verification evidence (a separate source, separate
        # storage). Validated and captured BEFORE any write so a bad bundle
        # cannot leave a half-written check behind.
        verification_block = (self._normalize_verification(record, verification)
                              if verification is not None else None)
        report = verify_task_result(record, check or {})
        from or_harness.strategy.rule_support import inherit_assessments
        inherit_assessments(record,report,check or {})
        if record.execution_features.get('task_check'):
            from or_harness.world_model.conditions import put_value
            with self.store.transaction() as conn:
                previous = copy.deepcopy(record.execution_features['task_check'])
                if 'g_attribution' in previous:
                    from or_harness.world_model.g_signal import save_g_assessment
                    save_g_assessment(self.store, record, previous, previous.pop('g_attribution'))
                report['previous_check_ref']=put_value(conn,self.store,'task_check',previous)
        # The report is a generic acceptance fact. Optional G-only evidence
        # belongs to the world-model log and is bound to this check revision.
        report["task_id"] = task_id
        report["episode_id"] = resolved_episode
        report["staged"] = staged
        # (1) The verify action — an executed action, because the framework
        # really ran these checks on real values. Its PRE snapshot is taken
        # from the task payload recovered for this episode (the same reader
        # ``_recover_task_text`` uses), never from a fabricated one-field
        # stub: the snapshot has to mean the same thing here as everywhere
        # else, and a stub would produce a snapshot of a different problem.
        task = self._task_payload_for(record, resolved_episode)
        snap = self.snapshot(task, resolved_episode)
        action = self.actions.begin_action(
            "verify", task_id, resolved_episode, pre_snapshot=snap,
            params={"execution_id": execution_id,
                    "basis": list(report["scope"]["basis"]),
                    "intent": report.get("intent"),
                    "verification": (None if verification_block is None else {
                        "state": verification_block["state"],
                        "source": verification_block["source"],
                        "artifacts": verification_block["artifacts"]})})
        report["action_id"] = action.action_id
        if verification_block is not None:
            verification_block["action_id"] = action.action_id
        g_assessment_ref = None
        with self.store.transaction():
            if attribution is not None:
                g_convention = g_observation_convention(
                    self.strategy_predictions.get(prediction_id)) if prediction_id else None
                g_assessment_ref = save_g_assessment(
                    self.store, record, report, attribution, prediction_id=prediction_id,
                    execution_action_id=execution_action_id,
                    observation_rule_version=(g_convention or {}).get("observation_rule_version"))
            self.actions.end_action(
                action.action_id, status="completed",
                outcome={"state": report["state"],
                         "execution_id": execution_id,
                         "checks_run": [c.get("check") for c in report["checks"]],
                         "n_diffs": len(report["diffs"]),
                         "unchecked": list(report["scope"]["unchecked"]),
                         **({'g_assessment_ref': g_assessment_ref} if g_assessment_ref else {})},
                linked_execution_id=execution_id, rollup="own")
            # (2) The fact annotation contains no G evaluation payload.
            if not retained:
                self.bank.set_task_check(execution_id,report)
            record.execution_features['task_check']=copy.deepcopy(report)
            if verification_block is not None:
                # A SEPARATE feature, never merged into task_check: the
                # framework's deterministic verdict and the agent's own
                # verification must stay distinguishable forever.
                evidence = [dict(item) for item in
                            (record.execution_features.get(
                                'verification_evidence') or [])
                            if isinstance(item, dict)]
                evidence.append(copy.deepcopy(verification_block))
                record.execution_features['verification_evidence'] = evidence
                if not retained:
                    self.bank.set_verification_evidence(execution_id, evidence)
            from or_harness.strategy.evidence_access import refresh_retained
            refresh_retained(self,record)
        dependencies={'affected_versions':[],'authority':'agent'}
        rule_validation = None
        if self.episode_closeout_record(task_id, resolved_episode) is not None:
            from or_harness.strategy.rule_versions import record_uses
            refreshed = record
            rule_validation = record_uses(self,[refreshed])
            if rule_validation.get("affected_entry_ids"):
                from or_harness.strategy.rule_versions import sync_rule_entries
                rule_validation["index_sync"] = sync_rule_entries(self,rule_validation["index_entry_ids"],rule_validation["affected_entry_ids"])
        # (3) REPUBLISH if this correction can change the calibration. A
        # late verdict on an episode still in the window changes a
        # calibration sample, so the published summary must be rebuilt —
        # otherwise a stale label would keep being served to later
        # predictions. An episode outside the window cannot change the
        # statistics and is skipped without a rebuild.
        republished = self._republish_calibration_if_in_window(
            task_id, resolved_episode)
        result: Dict[str, Any] = {
            "report": report,
            "state": report["state"],
            "execution_id": execution_id,
            "action_id": action.action_id,
            "staged": staged,
        }
        result['dependent_rule_validation']=dependencies
        if g_assessment_ref:
            result['g_assessment_ref'] = g_assessment_ref
        if rule_validation is not None:
            result["rule_validation"] = rule_validation
        if republished is not None:
            result["calibration_republished"] = {
                "reason": ("this task-result check can change a calibration "
                           "sample in the current window"),
                "n_evaluated": republished.get("n_evaluated"),
                "calibration_version": republished.get(
                    "calibration_version"),
            }
        # The SELF-CHECK MATERIAL is returned for every verdict: a pass also
        # needs the original task, the captured solution and the artifacts to
        # be reviewed by the agent before it commits. It is material with
        # references, not a diagnosis.
        result["reflection_material"] = self._reflection_material(
            record, report, resolved_episode)
        if verification_block is not None:
            result["verification"] = copy.deepcopy(verification_block)
        if report["state"] == "failed":
            result["next"] = (
                "a declared check found a violation; the framework does NOT "
                "infer the cause from it. Compare the original task, the "
                "model, the captured code and the check differences: decide "
                "whether the check or its evidence is wrong/incomplete "
                "(correct it and re-run check-task on THIS execution — no new "
                "attempt), or whether the model/implementation is wrong "
                "(obtain a fresh plan-next selection before a new attempt). "
                "The failed attempt stays recorded with its real cost and can "
                "be cited as contrast evidence. Before the answer is "
                "committed, use task-based evidence; after it is committed, "
                "reference answers and external verdicts are for offline "
                "analysis only — never re-solve to match a reference")
        elif report["state"] == "insufficient":
            result["next"] = (
                "the answer's validity is UNKNOWN, not confirmed: declare the "
                "deterministic check basis that applies to THIS task (do not "
                "declare `integer` for a problem that is genuinely "
                "continuous), or record the solution vector (result.json "
                "`variables`) so the declared check can run. An unchecked "
                "answer is not evidence that the task was solved, and a check "
                "that cannot run is not evidence that it was wrong")
        return result

    def _task_payload_for(self, record: ExecutionRecord,
                          episode_id: Optional[str]) -> Dict[str, Any]:
        """The task payload an action of this episode should snapshot.

        Priority: (1) the most recent real belief snapshot of this
        task/episode — it froze the task as it was; (2) the record's own
        profile and the stored task text; (3) a minimal payload carrying the
        identity, which is honest (``snapshot`` will report the coupling
        dimensions as unknown rather than invent them).

        This exists because a ``verify`` action needs a PRE snapshot and the
        caller of a task check supplies only an execution id: fabricating a
        two-field task stub would freeze a snapshot of a DIFFERENT problem.
        """
        from or_harness.world_model.conditions import latest_snapshot
        linked = next((action for action in self.actions.query(task_id=record.task_id)
                       if action.linked_execution_id == record.execution_id
                       and action.action_type == "execute_strategy"), None)
        best = self.get_snapshot(linked.pre_snapshot_id) if linked else None
        if best is None:
            best = latest_snapshot(self.store, record.task_id, episode_id,
                                   task_digest=record.task_text_digest, at=record.created_at)
        if best is not None:
            payload = ((best.problem_state or {}).get("task_input")
                       or (best.problem_state or {}).get("task_payload"))
            if isinstance(payload, dict) and payload.get("task_id"):
                return dict(payload)
        retained=record.execution_features.get('retained_task_payload')
        if isinstance(retained,dict) and retained.get('task_id'):
            return copy.deepcopy(retained)
        profile = record.profile_snapshot
        task: Dict[str, Any] = {
            "task_id": record.task_id,
            "family": getattr(profile, "family", "") or "",
        }
        text = None
        if record.task_text_digest:
            text = self.store.get_task_text(record.task_id,
                                            record.task_text_digest)
        if text:
            task["description"] = text
        return task

    def _normalize_verification(self, record: ExecutionRecord,
                                payload: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and CAPTURE the verification the AGENT ran on an execution.

        The harness does not execute this script and cannot certify its
        result. What it does is make the claim inspectable and honestly
        labelled: the script, output and result JSON are stored as immutable
        artifacts (never overwriting the solve ``result.json``), and the
        normalised record says ``source=agent_reported``. A satisfied
        assertion never becomes a framework check, and the task-check verdict
        is untouched — :meth:`check_task_result` still stores the framework's
        own deterministic result under ``execution_features.task_check``.

        Two failure modes are kept apart on purpose:

        - ``script_error`` — the script did not run / raised (``ran=false``,
          a recorded ``error``, or a non-zero ``exit_code``);
        - ``assertion_violations`` — the script ran and an agent-declared
          assertion was not satisfied.

        A bundle with neither files nor a result is refused: "I verified it"
        without any inspectable evidence is not verification.
        """
        from or_harness.core.artifacts import capture
        if not isinstance(payload, dict):
            raise ValueError("verification must be a JSON object")
        artifacts: Dict[str, Any] = {}

        def _capture_file(value: Any, key: str, kind: str, source: str) -> None:
            if value in (None, ""):
                return
            path = Path(str(value))
            if not path.is_file():
                raise ValueError(f"verification {key} file not found: "
                                 f"{value!r} (pass an existing path)")
            try:
                data = path.read_bytes()
            except OSError as exc:
                raise ValueError(f"verification {key} file could not be "
                                 f"read: {exc}") from exc
            artifacts[key] = capture(self.home, data, kind=kind, source=source)

        _capture_file(payload.get("script"), "script", "verification_script",
                      "agent-supplied verification script; the framework did "
                      "NOT execute it")
        _capture_file(payload.get("output"), "output", "verification_output",
                      "agent-captured verification output")
        result = payload.get("result")
        if result is None and payload.get("result_path"):
            path = Path(str(payload["result_path"]))
            if not path.is_file():
                raise ValueError(f"verification result_path not found: "
                                 f"{payload['result_path']!r}")
            try:
                result = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"verification result_path is not readable "
                                 f"JSON: {exc}") from exc
            artifacts["result"] = capture(
                self.home, path.read_bytes(), kind="verification_result",
                source="agent-supplied verification result; not a framework "
                       "verdict")
        if result is not None and not isinstance(result, dict):
            raise ValueError("verification result must be a JSON object")
        if not artifacts and result is None:
            raise ValueError(
                "verification needs at least one of `script`, `output`, "
                "`result` or `result_path`: a claim with no inspectable "
                "evidence is not a recorded verification")
        raw_assertions = (result or {}).get("assertions")
        if isinstance(raw_assertions, dict):
            raw_assertions = [raw_assertions]
        assertions: List[Dict[str, Any]] = []
        for index, item in enumerate(raw_assertions or []):
            if not isinstance(item, dict):
                continue
            raw_ok = item.get("ok")
            status = ("satisfied" if raw_ok is True
                      else "violated" if raw_ok is False
                      else "undecided")
            assertions.append({
                "name": str(item.get("name") or item.get("assertion")
                            or f"assertion[{index}]"),
                "status": status,
                "observed": copy.deepcopy(item.get("observed")),
                "expected": copy.deepcopy(item.get("expected")),
                "note": (None if item.get("note") is None
                         else str(item.get("note"))),
            })
        if len(assertions) > 200:
            raise ValueError("verification result carries more than 200 "
                             "assertions; split the verification")
        ran = (result or {}).get("ran")
        if ran is None:
            ran = payload.get("ran")
        error = (result or {}).get("error")
        exit_code = (result or {}).get("exit_code")
        if exit_code is None:
            exit_code = payload.get("exit_code")
        n_satisfied = sum(1 for item in assertions
                          if item["status"] == "satisfied")
        n_violated = sum(1 for item in assertions
                         if item["status"] == "violated")
        n_undecided = sum(1 for item in assertions
                          if item["status"] == "undecided")
        if error or ran is False:
            state = "script_error"
        elif assertions and n_violated:
            state = "assertion_violations"
        elif assertions and n_undecided:
            state = "assertions_undecided"
        elif assertions:
            state = "assertions_satisfied"
        elif result is not None:
            state = "no_assertions"
        else:
            state = "evidence_only"
        return {
            "source": "agent_reported",
            "state": state,
            "task_id": record.task_id,
            "execution_id": record.execution_id,
            "captured_at": time.time(),
            "ran": ran,
            "exit_code": exit_code,
            "error": (None if error is None else str(error)),
            "assertions": assertions,
            "n_satisfied": n_satisfied,
            "n_violated": n_violated,
            "n_undecided": n_undecided,
            "artifacts": artifacts,
            "note": (
                "the AGENT ran this verification and reported the result; the "
                "framework captured the files but did NOT execute the script "
                "and did not certify its output. This is evidence kept apart "
                "from report.checks: a satisfied assertion is not a framework "
                "verdict and never proves the answer correct, while a violated "
                "assertion or a script error is a signal to check the model, "
                "the code or the check — not a framework failure verdict"),
        }

    def _reflection_material(self, record: ExecutionRecord,
                             report: Dict[str, Any],
                             episode_id: Optional[str]) -> Dict[str, Any]:
        """The SELF-CHECK MATERIAL for one execution (material, not a diagnosis).

        Supplies the five things a pre-submission self-check needs, each tied
        to THIS execution and each preferring a REFERENCE over a repeated
        expansion: the original task text/version, the candidate and the
        method actually performed, the captured script/result artifacts, the
        captured solution (with a reference for the full record), and the
        check report with its differences. Agent-run verification evidence is
        included separately. The prior attempts are one-line references to
        earlier executions of the same task — the whole attempt chain is not
        re-expanded on every check.

        Deciding whether the problem is the task's interpretation, the model,
        the implementation, the recording or the check itself is the agent's
        judgment — the framework does not presume it is a modeling error and
        does not prescribe a rebuild.
        """
        from or_harness.world_model.state import task_text
        text = None
        if record.task_text_digest:
            text = self.store.get_task_text(record.task_id,
                                            record.task_text_digest)
        if text is None:
            text = task_text({"task_id": record.task_id})
        solution = record.execution_features.get("solution_variables")
        solution_variables = dict(solution) if isinstance(solution, dict) else solution
        artifacts = dict(record.execution_features.get("artifacts") or {})
        prior: List[Dict[str, Any]] = []
        for other in self.bank.query(task_id=record.task_id):
            if other.execution_id == record.execution_id:
                continue
            prior.append({
                "execution_id": other.execution_id,
                "strategy_id": other.strategy_id,
                "status": (other.quality or {}).get("status"),
                "objective": (other.quality or {}).get("objective"),
                "task_check": task_check_state(other),
                "inspect_ref": ("orx inspect --bank experience --execution "
                                + str(other.execution_id)),
            })
        prediction = getattr(record, "prediction_snapshot", None)
        prediction_id = (getattr(prediction, "prediction_id", None)
                         if prediction is not None else None)
        prediction_id = prediction_id or record.execution_features.get("prediction_id")
        from or_harness.world_model.implementation_facts import execution_implementation_evidence, compact_evidence
        implementation = compact_evidence(execution_implementation_evidence(self, record))
        return {
            "task_id": record.task_id,
            "implementation_evidence": implementation,
            "episode_id": episode_id,
            "task_text": text,
            "task_text_digest": record.task_text_digest,
            "task_ref": {
                "command": (f"orx inspect --bank texts --task "
                            f"{record.task_id}"),
                "text_digest": record.task_text_digest,
                "note": ("the original requirements THIS execution was "
                         "produced under; the same task_id with another "
                         "digest is a different task text"),
            },
            "strategy_id": record.strategy_id,
            "candidate": {
                "prediction_id": prediction_id,
                "strategy_id": record.strategy_id,
                "method_planned": copy.deepcopy(record.method_planned),
                "method_actual": copy.deepcopy(record.method_actual),
                "solver": dict(record.solver or {}),
                "ref": (("orx inspect --bank predictions --prediction "
                         + str(prediction_id)) if prediction_id else None),
            },
            "solver": dict(record.solver),
            "code_hash": (record.solver or {}).get("code_hash"),
            "artifacts": artifacts,
            "input_resources": copy.deepcopy(record.execution_features.get("input_resources") or []),
            "artifacts_ref": {
                "command": ("orx inspect --bank experience --execution "
                            + str(record.execution_id)),
                "note": ("immutable script/result/error artifacts captured for "
                         "this execution; the solve result.json is one of "
                         "them and is never overwritten by checks or "
                         "verification"),
            },
            "reported_quality": dict(record.quality or {}),
            "solution_variables": solution_variables,
            "solution_ref": {
                "command": ("orx inspect --bank experience --execution "
                            + str(record.execution_id)
                            + " #execution_features.solution_variables"),
                "n_variables": (len(solution) if isinstance(solution, dict)
                                else None),
                "note": ("the solution this execution captured; a check that "
                         "needs values can also read the result artifact"),
            },
            "check_report": report,
            "verification_evidence": copy.deepcopy(
                record.execution_features.get("verification_evidence") or []),
            "prior_attempts": prior,
            "guidance": (
                "self-check before committing: (1) read the original task and "
                "the candidate/method actually performed; (2) check the model "
                "against the task (variable meaning and domain, units and "
                "billing range, time horizon, objective meaning, key "
                "constraints, feasibility of the returned solution, and an "
                "optimality argument when the task needs one); (3) choose any "
                "applicable method — hand calculation, code review, an "
                "independent verification script, enumeration, bounds; (4) "
                "decide whether the problem is the solve, the implementation, "
                "the recording or the check itself. A passed check does not "
                "prove the model matches the task; a failed check does not "
                "prove the model is wrong; neither decides the next action for "
                "you. Correct a check or its evidence on this same execution "
                "when that is the problem; plan a new attempt only for a real "
                "model/implementation problem. Submit an optional G assessment "
                "once as top-level g_attribution in check-task --check, not an "
                "action report; use g_attribution.assessment in one to three "
                "sentences with necessary evidence references. Unknown is a "
                "normal outcome: do not repeat checks or add solves just to "
                "obtain a numeric G label. Use failure_source=reference_conflict "
                "for an unresolved reference discrepancy. Use reference answers and "
                "external correctness feedback only after the answer is "
                "committed, for offline analysis rather than a return to "
                "solving — never change the task to match a reference value"),
        }
