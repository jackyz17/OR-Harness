"""Heterogeneous parallel solver exploration with intra-branch sequential repair."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from ..config import ExperienceBankConfig
from ..experience.admission import AdmissionJudge, candidate_to_record
from ..experience.comparative_extractor import ComparativeSynthesisExtractor
from ..core.episode import EpisodeStore, build_episode
from ..experience.extractor import ExperienceExtractor
from ..experience.failure_buffer import FailureBuffer
from ..llm_client import LLMClient
from ..core.modeling_store import ModelingStore
from ..modeling.modeling_stage import StructuredModelingStage
from ..retrieval.query_builder import StageAwareQueryBuilder, sanitize_feedback
from .reflection import (
    DEFAULT_GOLD_TOLERANCE,
    GoldVerdict,
    ReflectionGenerator,
    evaluate_gold,
)
from .result_selector import ResultSelector
from ..retrieval.retrieval import ExperienceRetriever
from ..core.schemas import (
    AttemptRecord,
    BranchResult,
    BranchState,
    SolveResult,
    SolverExecutionResult,
    TerminationReason,
    ValidationLevel,
    utc_now,
)
from ..solvers.base import SolverAdapter
from ..solvers.registry import SolverRegistry
from ..core.store import AppendOnlyExperienceStore
from .trajectory import BranchStateSummarizer, TrajectoryStore
from .validator import ResultValidator


LOG = logging.getLogger("or_experience_bank")


class NoSolverAvailable(RuntimeError):
    pass


class ORExperienceOrchestrator:
    def __init__(
        self,
        config: ExperienceBankConfig,
        store: AppendOnlyExperienceStore,
        retriever: ExperienceRetriever,
        registry: SolverRegistry,
        llm_client: LLMClient,
        extractor: Optional[ExperienceExtractor] = None,
        modeling_retriever=None,
        utility_tracker=None,
    ):
        self.config = config
        self.store = store
        self.retriever = retriever
        self.registry = registry
        self.llm = llm_client
        self.extractor = extractor or ExperienceExtractor(llm_client)
        self.query_builder = StageAwareQueryBuilder()
        self.trajectory = TrajectoryStore(config.bank_home)
        self.summarizer = BranchStateSummarizer()
        self.result_validator = ResultValidator()
        self.selector = ResultSelector()
        self.modeling_stage = StructuredModelingStage(llm_client)
        # Phase 2: comparative synthesis + independent modeling bank (Option 3, D10).
        self.synthesis = ComparativeSynthesisExtractor(llm_client)
        self.judge = AdmissionJudge(llm_client)
        self.modeling_store = ModelingStore(config.bank_home)
        self.episode_store = EpisodeStore(config.bank_home)
        # Phase 4.1 (module 4.1): pattern reflow into online solving. Optional so existing
        # flows are untouched when absent. modeling_retriever recalls validated patterns +
        # realizations as planning priors; utility_tracker receives precise attribution for
        # explicitly cited patterns (and coarse attribution for recalled realizations).
        self.modeling_retriever = modeling_retriever
        self.utility_tracker = utility_tracker
        self._timeline: List[Dict[str, Any]] = []
        # Option A deferred state: populated by solve(), consumed by evaluate_with_gold().
        self._pending: Dict[str, Any] = {}
        # Per-solve failure buffer (Phase 2 step 1): consumed by comparative synthesis.
        self.failures: Optional[FailureBuffer] = None

    async def solve(
        self,
        problem: str,
        solvers: Optional[List[str]] = None,
        max_attempts: Optional[int] = None,
        auto_append: Optional[bool] = None,
        reference_objective: Optional[float] = None,
        semantic_validator=None,
        defer_extraction: bool = False,
    ) -> SolveResult:
        self.config.ensure_directories()
        spec = self._normalize_problem(problem)
        problem_id = "prob_" + hashlib.sha256(problem.encode("utf-8")).hexdigest()[:16]
        spec["problem_id"] = problem_id
        self.trajectory.initialize_problem(problem_id, spec)
        self._timeline = []
        self.failures = FailureBuffer(problem_id)
        self._event("problem_normalized", problem_id=problem_id, problem_family=spec["problem_family"])

        # Phase 1 (module 1.1, D17): structured modeling stage runs BEFORE any solver
        # branch. The <model> must pass the ModelingGate before branches are created.
        # Phase 4.1: recall planning priors (validated patterns + similar realizations)
        # from the Modeling Bank and inject them into the modeling prompt.
        planning_priors = None
        if self.modeling_retriever is not None:
            query_text = self.query_builder.modeling(spec, "produce a verified OR model")
            planning_priors = self.modeling_retriever.retrieve_priors(query_text)
            self._event(
                "planning_priors_recalled",
                problem_id=problem_id,
                records=len(planning_priors.records),
            )
        modeling_result = await self.modeling_stage.run(problem, planning_priors)
        spec["verified_model"] = modeling_result.model
        spec["modeling_think"] = modeling_result.think
        spec["cited_principle_ids"] = list(modeling_result.cited_principle_ids)
        if modeling_result.signature is not None:
            spec["structural_signature"] = modeling_result.signature.to_dict()
        self._event(
            "modeling_stage_finished",
            problem_id=problem_id,
            success=modeling_result.success,
            rounds_used=modeling_result.rounds_used,
            has_signature=modeling_result.signature is not None,
            cited_principle_ids=modeling_result.cited_principle_ids,
        )
        if not modeling_result.success:
            for issue in modeling_result.issues:
                self.failures.add("modeling", summary=issue.get("detail", "modeling gate failed"),
                                  context={"issue": issue})
            warnings = ["modeling stage failed to produce a verified model: " + json.dumps(modeling_result.issues, ensure_ascii=False)]
            self._event("solve_aborted", problem_id=problem_id, reason="modeling_gate_failed")
            return SolveResult(
                problem_id=problem_id,
                selected_branch_id=None,
                selection_reason="modeling gate failed after {} round(s)".format(modeling_result.rounds_used),
                branches=[],
                retrieved_experience_ids={"modeling": [], "implementation": [], "repair": [], "solving": []},
                appended_experience_ids=[],
                duplicate_experience_ids=[],
                validation_level=ValidationLevel.UNVERIFIED.value,
                warnings=warnings,
                timeline=list(self._timeline),
                objective_comparable=False,
                branch_discrepancies=[],
            )

        names = solvers or self.config.solvers
        adapters, unavailable = self.registry.available(names)
        warnings = ["{}: {}".format(name, result.reason) for name, result in unavailable.items()]
        if not adapters:
            raise NoSolverAvailable("No requested solver is available: " + "; ".join(warnings))
        if len(adapters) == 1:
            warnings.append("Only one solver branch is available; heterogeneous parallel comparison was not formed.")

        modeling_query = self.query_builder.modeling(spec, "construct a valid solver-independent formulation")
        modeling_hits = self.retriever.retrieve(
            "modeling", modeling_query,
            metadata_filters={"problem_family": spec["problem_family"], "generality": "solver_agnostic"},
            top_k=self.config.top_k["modeling"], min_score=self.config.min_similarity,
        )
        retrieved: Dict[str, List[str]] = {
            "modeling": [hit.experience_id for hit in modeling_hits],
            "implementation": [], "repair": [], "solving": [],
        }
        self._event("modeling_retrieved", ids=retrieved["modeling"])
        # NOTE (single-solver era): the orx CLI path has the agent choose ONE
        # solver per run; this programming API still accepts a list for
        # backward compatibility but no longer bounds concurrency by a
        # dedicated config knob — branches run unbounded via asyncio.gather.
        attempts_limit = max_attempts or self.config.max_attempts_per_branch

        async def guarded(adapter: SolverAdapter) -> BranchResult:
            return await self._run_branch(
                adapter, spec, problem, problem_id, attempts_limit, modeling_hits,
                reference_objective, semantic_validator,
            )

        branches = await asyncio.gather(*(guarded(adapter) for adapter in adapters))
        self._event("all_branches_finished", branch_ids=[branch.branch_id for branch in branches])

        for branch in branches:
            for attempt in branch.attempts:
                for layer, ids in attempt.retrieved_experience_ids.items():
                    retrieved[layer].extend(ids)
        retrieved = {key: list(dict.fromkeys(value)) for key, value in retrieved.items()}
        self._apply_cross_solver_validation(branches)

        candidates = []
        for branch in branches:
            candidates.extend(self.extractor.extract_intra_branch(branch, spec["problem_family"]))
        candidates.extend(self.extractor.extract_cross_branch(branches, spec["problem_family"]))
        self._event("experience_extracted", count=len(candidates))
        selection = self.selector.select(branches)
        selected = next((b for b in branches if b.branch_id == selection["selected_branch_id"]), None)
        validation_level = selected.validation.validation_level if selected else ValidationLevel.UNVERIFIED.value

        # Phase 2 step 6: record the problem-level Episode right after solving (gold pending).
        failure_count = self.failures.count() if self.failures is not None else 0
        solve_status = "success" if selected and selected.execution and selected.execution.status in {"optimal", "feasible"} else "failed"
        episode = build_episode(problem, problem_id, spec, branches, failure_count, solve_status)
        self.episode_store.record_episode(episode)
        self._event("episode_recorded", problem_id=problem_id, status=solve_status, failures=failure_count)

        if defer_extraction:
            # Option A: gold arrives after solving. Stash candidates + failures and let
            # the harness agent call evaluate_with_gold() to decide extraction/reflection.
            self._pending = {
                "problem": problem,
                "problem_id": problem_id,
                "spec": spec,
                "branches": branches,
                "candidates": candidates,
                "retrieved": retrieved,
                "auto_append": self.config.auto_append if auto_append is None else auto_append,
                # Unified attribution payload consumed by evaluate_with_gold on match.
                "cited_principle_ids": list(modeling_result.cited_principle_ids),
                "planning_experience_ids": (
                    [r.get("experience_id", "") for r in planning_priors.records]
                    if planning_priors is not None else []
                ),
            }
            self._event("extraction_deferred", problem_id=problem_id, candidates=len(candidates))
            return SolveResult(
                problem_id=problem_id,
                selected_branch_id=selection["selected_branch_id"],
                selection_reason=selection["selection_reason"],
                branches=branches,
                retrieved_experience_ids=retrieved,
                appended_experience_ids=[],
                duplicate_experience_ids=[],
                validation_level=validation_level,
                warnings=warnings,
                timeline=list(self._timeline),
                objective_comparable=selection["objective_comparable"],
                branch_discrepancies=selection["branch_discrepancies"],
            )

        appended_ids: List[str] = []
        duplicate_ids: List[str] = []
        should_append = self.config.auto_append if auto_append is None else auto_append
        if should_append:
            for candidate in candidates:
                if candidate.polarity == "positive" and not self.config.append_positive:
                    continue
                if candidate.polarity == "negative" and not self.config.append_negative:
                    continue
                if self.config.detect_near_duplicates and candidate.possible_duplicate_of is None:
                    near = self.retriever.retrieve(
                        candidate.layer, candidate.retrieval_text, top_k=1, min_score=0.92
                    )
                    if near:
                        candidate.possible_duplicate_of = near[0].experience_id
                result = self.store.append(candidate)
                if result.status == "appended":
                    appended_ids.append(result.experience_id)
                    self.retriever.rebuild(result.layer)
                elif result.duplicate_of:
                    duplicate_ids.append(result.duplicate_of)
        self._event("experience_append_finished", appended=appended_ids, duplicates=duplicate_ids)

        return SolveResult(
            problem_id=problem_id,
            selected_branch_id=selection["selected_branch_id"],
            selection_reason=selection["selection_reason"],
            branches=branches,
            retrieved_experience_ids=retrieved,
            appended_experience_ids=appended_ids,
            duplicate_experience_ids=duplicate_ids,
            validation_level=validation_level,
            warnings=warnings,
            timeline=list(self._timeline),
            objective_comparable=selection["objective_comparable"],
            branch_discrepancies=selection["branch_discrepancies"],
        )

    async def evaluate_with_gold(
        self,
        gold: Optional[float],
        tolerance: float = DEFAULT_GOLD_TOLERANCE,
    ) -> GoldVerdict:
        """Option A second step: judge the deferred solve result against the gold answer.

        On match: append the stashed candidates (comparative synthesis hooks in later
        modules) and mark ready_for_extraction. On mismatch: stash failures into the
        pending buffer so the harness agent can drive an outer reflection round.
        """
        if not self._pending:
            raise RuntimeError("evaluate_with_gold called without a deferred solve()")
        branches = self._pending["branches"]
        selection = self.selector.select(branches)
        selected = next((b for b in branches if b.branch_id == selection["selected_branch_id"]), None)
        validation_level = selected.validation.validation_level if selected else ValidationLevel.UNVERIFIED.value
        temp_result = SolveResult(
            problem_id=self._pending["problem_id"],
            selected_branch_id=selection["selected_branch_id"],
            selection_reason=selection["selection_reason"],
            branches=branches,
            retrieved_experience_ids=self._pending["retrieved"],
            appended_experience_ids=[],
            duplicate_experience_ids=[],
            validation_level=validation_level,
            warnings=[],
            timeline=[],
        )
        verdict = evaluate_gold(temp_result, gold, tolerance)

        if verdict.ready_for_extraction:
            # Utility attribution: the solve matched gold, so planning priors that
            # contributed get credited. Cited experiences (LLM declared [uses En]) get
            # precise attribution. This feeds the soft-delete scoring and induction triggers.
            if self.utility_tracker is not None:
                cited = self._pending.get("cited_principle_ids", [])
                recalled = self._pending.get("planning_experience_ids", [])
                all_ids = list(dict.fromkeys(cited + recalled))
                if all_ids:
                    self.utility_tracker.record_utilities(all_ids)
                self._event(
                    "utility_attributed",
                    cited_experiences=len(cited),
                    total_recalled=len(recalled),
                )
            appended_ids, duplicate_ids = await self._append_synthesis_candidates()
            self._event("experience_append_finished", appended=appended_ids, duplicates=duplicate_ids)
            self._pending["appended_experience_ids"] = appended_ids
            self._pending["duplicate_experience_ids"] = duplicate_ids
        else:
            self._event("gold_mismatch", reason=verdict.reason, gold=gold)

        # Phase 2 step 6 (b): append the gold supplement once gold has been judged.
        self.episode_store.record_gold_supplement(
            self._pending["problem_id"],
            gold,
            verdict.matched,
            self._pending.get("appended_experience_ids", []),
        )
        self._event("episode_gold_supplement", problem_id=self._pending["problem_id"], matched=verdict.matched)

        return verdict

    async def _append_synthesis_candidates(self) -> tuple:
        """Run comparative synthesis (success vs failures) and append bank-classified
        candidates through the admission gate. modeling -> ModelingStore; other layers ->
        shared flat store. Falls back to rule-based candidates when synthesis yields none."""
        appended_ids: List[str] = []
        duplicate_ids: List[str] = []
        if not self._pending["auto_append"]:
            return appended_ids, duplicate_ids

        spec = self._pending["spec"]
        branches = self._pending["branches"]
        branch_ids = [b.branch_id for b in branches]
        attempt_ids = [a.attempt_id for b in branches for a in b.attempts]
        signature = None
        if spec.get("structural_signature"):
            from ..core.modeling_schemas import StructuralSignature
            signature = StructuralSignature.from_dict(spec["structural_signature"])

        synthesis_candidates = await self.synthesis.synthesize(
            self._pending["problem"], branches, self.failures, spec.get("verified_model", "")
        )
        self._event("synthesis_candidates", count=len(synthesis_candidates))

        for candidate in synthesis_candidates:
            if candidate.get("polarity") == "positive" and not self.config.append_positive:
                continue
            if candidate.get("polarity") == "negative" and not self.config.append_negative:
                continue
            if not await self.judge.accept(candidate):
                continue
            try:
                record = candidate_to_record(
                    candidate,
                    problem_id=self._pending["problem_id"],
                    problem_family=spec.get("problem_family", "general_milp"),
                    branch_ids=branch_ids,
                    attempt_ids=attempt_ids,
                    signature=signature,
                )
            except Exception as exc:  # conversion/validation failure -> skip candidate
                self._event("candidate_rejected", layer=candidate.get("layer"), reason=str(exc))
                continue
            if candidate["layer"] == "modeling":
                result = self.modeling_store.append(record)
                if result["status"] == "appended":
                    appended_ids.append(result["experience_id"])
                elif result.get("duplicate_of"):
                    duplicate_ids.append(result["duplicate_of"])
            else:
                append_result = self.store.append(record)
                if append_result.status == "appended":
                    appended_ids.append(append_result.experience_id)
                    self.retriever.rebuild(append_result.layer)
                elif append_result.duplicate_of:
                    duplicate_ids.append(append_result.duplicate_of)
        return appended_ids, duplicate_ids

    def build_reflection_prompt(self, verdict: GoldVerdict) -> str:
        """Compose the outer-reflection prompt for the harness agent's LLM (Option A)."""
        if not self._pending:
            raise RuntimeError("build_reflection_prompt called without a deferred solve()")
        return ReflectionGenerator().build_reflection_prompt(
            self._pending["problem"], self._result_from_pending(), verdict
        )

    def _result_from_pending(self) -> SolveResult:
        branches = self._pending["branches"]
        selection = self.selector.select(branches)
        selected = next((b for b in branches if b.branch_id == selection["selected_branch_id"]), None)
        return SolveResult(
            problem_id=self._pending["problem_id"],
            selected_branch_id=selection["selected_branch_id"],
            selection_reason=selection["selection_reason"],
            branches=branches,
            retrieved_experience_ids=self._pending["retrieved"],
            appended_experience_ids=self._pending.get("appended_experience_ids", []),
            duplicate_experience_ids=self._pending.get("duplicate_experience_ids", []),
            validation_level=selected.validation.validation_level if selected else ValidationLevel.UNVERIFIED.value,
            warnings=[],
            timeline=[],
        )

    async def _run_branch(
        self,
        adapter: SolverAdapter,
        spec: Dict[str, Any],
        original_problem: str,
        problem_id: str,
        max_attempts: int,
        modeling_hits,
        reference_objective,
        semantic_validator,
    ) -> BranchResult:
        branch_id = "{}-{}".format(adapter.name, uuid4().hex[:8])
        workspace = self.config.bank_home / "runs" / problem_id / branch_id
        workspace.mkdir(parents=True, exist_ok=False)
        state = BranchState(problem_id=problem_id, branch_id=branch_id, solver=adapter.name, workspace=str(workspace))
        self._event("branch_started", branch_id=branch_id, solver=adapter.name)
        implementation_query = self.query_builder.implementation(
            spec["normalized_description"], adapter.name, adapter.solver_family, adapter.api,
            "implement all variables, objective, constraints, solve, and write result.json",
        )
        implementation_hits = self.retriever.retrieve(
            "implementation", implementation_query,
            metadata_filters={"solver": adapter.name, "solver_family": adapter.solver_family},
            top_k=self.config.top_k["implementation"], min_score=self.config.min_similarity,
        )
        next_repair_hits = []
        last_execution = SolverExecutionResult(status="unknown", solver=adapter.name)
        last_validation = self.result_validator.validate(last_execution)
        termination = TerminationReason.UNKNOWN.value
        previous_error = None
        previous_code_hash = None

        for number in range(1, max_attempts + 1):
            started = utc_now()
            retrieved_ids = {
                "modeling": [hit.experience_id for hit in modeling_hits],
                "implementation": [hit.experience_id for hit in implementation_hits],
                "repair": [hit.experience_id for hit in next_repair_hits],
                "solving": [],
            }
            prompt = self._generation_prompt(
                original_problem, spec, adapter, number, modeling_hits,
                implementation_hits, next_repair_hits, state,
            )
            code = await self.llm.generate_text(prompt, timeout=self.config.python_timeout_seconds)
            code = self._extract_python(code)
            code_path = workspace / "attempt_{}.py".format(number)
            code_path.write_text(code, encoding="utf-8")
            code_hash = hashlib.sha256(code.encode("utf-8")).hexdigest()
            state.current_code = code
            state.current_formulation = spec["normalized_description"]
            execution = await adapter.execute(code_path, workspace, number)
            execution = adapter.parse_result(execution)
            validation = self.result_validator.validate(execution, reference_objective, semantic_validator)
            last_execution, last_validation = execution, validation
            feedback = adapter.normalize_feedback(execution)
            state.latest_feedback = sanitize_feedback(feedback, 1500)
            if execution.normalized_error:
                state.unresolved_issues.append(execution.normalized_error)
                if self.failures is not None:
                    self.failures.add(
                        "branch_code",
                        summary="{} attempt {} error".format(adapter.name, number),
                        normalized_error=execution.normalized_error,
                        solver=adapter.name,
                        context={"status": execution.status, "stderr": execution.stderr[:500]},
                    )
            repair_action = ""
            attempt_termination = None

            performance_symptom = ""
            if execution.status == "timeout":
                performance_symptom = "timeout"
            elif execution.mip_gap is not None and execution.mip_gap > 0.05:
                performance_symptom = "large MIP gap"
            elif "numerical" in (execution.message or "").lower():
                performance_symptom = "numerical warning"
            if performance_symptom:
                solving_query = self.query_builder.solving(
                    spec["problem_family"], "unknown", None, None, adapter.name,
                    execution.status, execution.runtime_seconds, execution.mip_gap,
                    execution.objective_bound, execution.message, performance_symptom,
                )
                solving_hits = self.retriever.retrieve(
                    "solving", solving_query,
                    metadata_filters={"solver": adapter.name, "solver_family": adapter.solver_family},
                    top_k=self.config.top_k["solving"], min_score=self.config.min_similarity,
                )
                retrieved_ids["solving"] = [hit.experience_id for hit in solving_hits]

            if validation.valid and execution.status in {"optimal", "feasible"}:
                termination = TerminationReason.OPTIMAL.value if execution.status == "optimal" else TerminationReason.FEASIBLE.value
                attempt_termination = termination
                if state.unresolved_issues:
                    state.resolved_issues.append(state.unresolved_issues[-1])
            elif execution.status == "timeout":
                termination = TerminationReason.TIMEOUT.value
                attempt_termination = termination
            elif execution.status == "infeasible":
                termination = TerminationReason.INFEASIBLE.value
                attempt_termination = termination
            elif execution.status == "unbounded":
                termination = TerminationReason.UNBOUNDED.value
                attempt_termination = termination
            elif number >= max_attempts:
                termination = TerminationReason.MAX_ATTEMPTS.value
                attempt_termination = termination
            elif self.config.stop_on_repeated_error and execution.normalized_error and execution.normalized_error == previous_error:
                termination = TerminationReason.REPEATED_ERROR.value
                attempt_termination = termination
            elif self.config.stop_on_unchanged_code and previous_code_hash and code_hash == previous_code_hash:
                termination = TerminationReason.UNCHANGED_CODE.value
                attempt_termination = termination

            if attempt_termination is None:
                repair_query = self.query_builder.repair(
                    adapter.name, adapter.solver_family,
                    execution.normalized_error or execution.status,
                    execution.stderr, execution.status,
                    state.ineffective_repairs[-1] if state.ineffective_repairs else "initial implementation",
                    self.summarizer.summarize(state),
                )
                next_repair_hits = self.retriever.retrieve(
                    "repair", repair_query,
                    metadata_filters={"solver": adapter.name, "solver_family": adapter.solver_family},
                    top_k=self.config.top_k["repair"], min_score=self.config.min_similarity,
                )
                # Option (b): derived error-transition-graph guidance alongside hits.
                if execution.normalized_error:
                    guidance = self.retriever.repair_guidance(adapter.name, execution.normalized_error)
                    if guidance["actions"] or guidance["pitfalls"]:
                        state.latest_feedback += "\nRepair graph guidance: actions={} pitfalls={} path={}".format(
                            [a["action"] for a in guidance["actions"]][:3],
                            guidance["pitfalls"][:3],
                            guidance["repair_path"][:3],
                        )
                repair_action = "Use latest normalized feedback and retrieved Repair Bank records in the next attempt"

            attempt = AttemptRecord(
                attempt_id="att_" + uuid4().hex,
                problem_id=problem_id,
                branch_id=branch_id,
                solver=adapter.name,
                attempt_number=number,
                started_at=started,
                finished_at=utc_now(),
                retrieved_experience_ids=retrieved_ids,
                problem_summary=spec["normalized_description"],
                formulation_summary=state.current_formulation,
                code_path=str(code_path.relative_to(self.config.bank_home)),
                code_hash=code_hash,
                stdout_summary=execution.stdout,
                stderr_summary=execution.stderr,
                normalized_error=execution.normalized_error,
                solver_status=execution.status,
                objective_value=execution.objective_value,
                objective_bound=execution.objective_bound,
                mip_gap=execution.mip_gap,
                runtime_seconds=execution.runtime_seconds,
                validator_report={"valid": validation.valid, "errors": validation.errors, "warnings": validation.warnings},
                validation_level=validation.validation_level,
                repair_action_summary=repair_action,
                termination_reason=attempt_termination,
            )
            state.attempts.append(attempt)
            self.trajectory.append_attempt(attempt)
            self._event(
                "attempt_finished", branch_id=branch_id, solver=adapter.name,
                attempt_number=number, status=execution.status,
                termination_reason=attempt_termination, validation_level=validation.validation_level,
            )
            previous_error = execution.normalized_error
            previous_code_hash = code_hash
            if attempt_termination:
                break

        self._event("branch_finished", branch_id=branch_id, solver=adapter.name, termination_reason=termination)
        return BranchResult(
            branch_id=branch_id,
            solver=adapter.name,
            workspace=str(workspace.relative_to(self.config.bank_home)),
            attempts=state.attempts,
            execution=last_execution,
            validation=last_validation,
            termination_reason=termination,
        )

    def _generation_prompt(self, problem, spec, adapter, number, modeling_hits, implementation_hits, repair_hits, state):
        context = adapter.build_generation_context(spec, number)
        verified_model = spec.get("verified_model") or "(no verified model; formulate from the problem)"
        lines = [
            "Generate only executable Python code for an OR solver branch.",
            "Original problem: " + problem,
            "Verified mathematical model (implement THIS faithfully): " + verified_model,
            "Normalized problem: " + json.dumps(spec, ensure_ascii=False),
            "Solver context: " + json.dumps(context, ensure_ascii=False),
            "Required result: write result.json in the current directory with this schema:",
            '  {"status":"optimal|feasible|infeasible|unbounded|timeout|error|unknown",',
            '   "solver":"...", "objective_sense":"minimize|maximize|feasibility|unknown",',
            '   "objective_value":null, "objective_bound":null, "mip_gap":null,',
            '   "runtime_seconds":null, "variables":{}, "diagnostics":{}, "message":""}',
            "  NOTE: field is objective_value (NOT objective). status must be lowercase.",
            "Security: blocked: subprocess, socket, urllib, http, requests, pathlib, shutil.",
            "  Allowed: os, os.path, json, sys, math, itertools, collections, and the solver's own package.",
            "  Blocked os.* calls: os.system, os.popen, os.exec*, os.listdir, os.environ, etc.",
            "  Use open() only for result.json in the current directory. No network, no shell.",
            "Modeling experiences: " + json.dumps([h.record for h in modeling_hits], ensure_ascii=False),
            "Implementation experiences: " + json.dumps([h.record for h in implementation_hits], ensure_ascii=False),
        ]
        if number > 1:
            lines.extend(
                [
                    "This is sequential repair attempt {}.".format(number),
                    "Latest branch state: " + self.summarizer.summarize(state),
                    "Repair experiences: " + json.dumps([h.record for h in repair_hits], ensure_ascii=False),
                    "Return the complete latest code, not a patch.",
                ]
            )
        return "\n".join(lines)

    @staticmethod
    def _extract_python(text: str) -> str:
        match = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else text.strip()

    @staticmethod
    def _normalize_problem(problem: str) -> Dict[str, Any]:
        lower = problem.lower()
        # Order matters: more specific families precede generic ones. The fallback
        # "general_milp" is a last resort so that two structurally-different problems
        # are not falsely labelled as the same family (which would defeat the
        # cross-family induction trigger).
        families = [
            ("aircraft_landing", ["aircraft landing", "aircraft scheduling", "runway", "landing sequence", "航班降落", "飞机着陆", "跑道调度"]),
            ("job_shop_scheduling", ["job shop", "job-shop", "machine scheduling", "车间调度", "作业车间"]),
            ("nurse_rostering", ["nurse", "rostering", "roster", "shift scheduling", "护士排班", "轮班"]),
            ("crew_scheduling", ["crew scheduling", "crew pairing", "crew rostering", "机组排班", "乘务"]),
            ("facility_location", ["facility location", "warehouse location", "plant location", "选址", "设施选址"]),
            ("network_design", ["network design", "network flow", "网络设计", "网络流"]),
            ("cutting_stock", ["cutting stock", "cutting-stock", "下料", "切割"]),
            ("bin_packing", ["bin packing", "bin-packing", "装箱"]),
            ("tsp", ["tsp", "traveling salesman", "travelling salesman", "旅行商"]),
            ("cvrp", ["vrp", "vehicle routing", "车辆路径", "配送"]),
            ("assignment", ["assignment", "assign", "指派", "分配任务"]),
            ("scheduling", ["schedule", "scheduling", "排班", "调度"]),
            ("inventory", ["inventory", "库存", "补货", "订货"]),
            ("knapsack", ["knapsack", "背包"]),
            ("production_planning", ["production", "生产计划", "产能"]),
            ("set_covering", ["set covering", "set cover", "集合覆盖"]),
            ("portfolio", ["portfolio", "投资组合", "资产配置"]),
        ]
        family = next((name for name, words in families if any(word in lower for word in words)), "general_milp")
        objective = "maximize" if any(word in lower for word in ["maximize", "最大化", "最大"] ) else "minimize" if any(word in lower for word in ["minimize", "最小化", "最少", "最低"]) else "unknown"
        constraints = [word for word in ["capacity", "容量", "time window", "时间窗", "precedence", "需求", "demand"] if word in lower]
        return {
            "description": problem,
            "normalized_description": " ".join(problem.split()),
            "problem_family": family,
            "objective": objective,
            "entities": [],
            "constraints": constraints,
        }

    @staticmethod
    def _apply_cross_solver_validation(branches: List[BranchResult]) -> None:
        valid = [b for b in branches if b.validation.valid and b.execution.status in {"optimal", "feasible"} and b.execution.objective_value is not None]
        if len(valid) < 2:
            return
        senses = {b.execution.objective_sense for b in valid}
        if len(senses) != 1 or "unknown" in senses:
            for branch in valid:
                branch.validation.objective_comparable = False
            return
        values = [float(b.execution.objective_value) for b in valid]
        tolerance = 1e-6 * max(1.0, max(abs(value) for value in values))
        if max(values) - min(values) <= tolerance:
            for branch in valid:
                if branch.validation.validation_level != ValidationLevel.SEMANTIC_CHECKED.value:
                    branch.validation.validation_level = ValidationLevel.CROSS_SOLVER_CONSISTENT.value

    def _event(self, event: str, **fields: Any) -> None:
        item = {"timestamp": utc_now(), "event": event, **fields}
        self._timeline.append(item)
        LOG.info("%s", json.dumps(item, ensure_ascii=False))
