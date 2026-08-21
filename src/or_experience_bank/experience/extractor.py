"""Evidence-gated intra-branch and cross-branch experience extraction."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from ..llm_client import LLMClient
from ..core.schemas import (
    AttemptRecord,
    BranchResult,
    ExperienceEvidence,
    ExperienceGenerality,
    ExperienceLayer,
    ExperiencePolicy,
    ExperiencePolarity,
    ExperienceRecord,
    ExperienceScope,
    ExperienceTrigger,
    ProblemContext,
    ValidationLevel,
)
from ..solving.validator import ExperienceValidator


SUCCESS = {"optimal", "feasible"}


class ExperienceExtractor:
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client
        self.validator = ExperienceValidator()

    async def extract_with_llm(self, prompt: str) -> List[ExperienceRecord]:
        """Parse and validate strict JSON candidates, with one format-only retry."""
        if self.llm_client is None:
            return []
        raw: Any = None
        for attempt in range(2):
            try:
                raw = await self.llm_client.generate_object(
                    prompt if attempt == 0 else (
                        "Format repair only: return the previous candidate as a strict JSON array of complete "
                        "ExperienceRecord objects. Do not add prose or change semantics. Previous output: " + str(raw)[:6000]
                    )
                )
                if not isinstance(raw, list):
                    raise ValueError("extraction output is not a JSON array")
                records = []
                for candidate in raw:
                    if not isinstance(candidate, dict):
                        raise ValueError("candidate is not an object")
                    record = ExperienceRecord.from_dict(candidate)
                    data = record.to_dict()
                    from ..core.store import compute_content_hash

                    data["content_hash"] = compute_content_hash(data)
                    report = self.validator.validate(data)
                    if not report.valid:
                        raise ValueError("invalid experience: " + "; ".join(report.errors))
                    records.append(record)
                return records
            except (KeyError, TypeError, ValueError):
                if attempt == 1:
                    return []
        return []

    def extract_intra_branch(self, branch: BranchResult, problem_family: str) -> List[ExperienceRecord]:
        records: List[ExperienceRecord] = []
        attempts = branch.attempts
        for previous, current in zip(attempts, attempts[1:]):
            if previous.normalized_error and current.solver_status in SUCCESS:
                action = current.repair_action_summary or "Replace the failing implementation with the validated next-attempt implementation"
                records.append(
                    self._record(
                        layer=ExperienceLayer.REPAIR.value,
                        polarity=ExperiencePolarity.POSITIVE.value,
                        title="Repair {} error in {} branch".format(previous.normalized_error[:70], branch.solver),
                        retrieval_text=(
                            "When {} reports '{}', {}. The next attempt reached {} after this repair."
                            .format(branch.solver, previous.normalized_error, action, current.solver_status)
                        ),
                        problem_family=problem_family,
                        stage="repair",
                        solver=branch.solver,
                        solver_family=self._family(branch.solver),
                        normalized_error=previous.normalized_error,
                        solver_status=previous.solver_status,
                        diagnosis="The previous implementation produced a reproducible execution or solver failure.",
                        action=action,
                        rationale="The error disappeared and the immediately following attempt produced a solver-feasible result.",
                        evidence=ExperienceEvidence(
                            problem_id=previous.problem_id,
                            branch_ids=[branch.branch_id],
                            attempt_ids=[previous.attempt_id, current.attempt_id],
                            solver_feedback_summary="error-before / success-after",
                            validation_level=current.validation_level,
                            causal_confidence="high",
                        ),
                        generality=ExperienceGenerality.SOLVER_SPECIFIC.value,
                    )
                )
            elif previous.normalized_error and current.normalized_error == previous.normalized_error:
                records.append(
                    self._record(
                        layer=ExperienceLayer.REPAIR.value,
                        polarity=ExperiencePolarity.NEGATIVE.value,
                        title="Avoid ineffective repeated repair for {}".format(previous.normalized_error[:70]),
                        retrieval_text=(
                            "For {} error '{}', avoid repeating the unchanged repair action '{}'; "
                            "the same normalized error persisted in the next attempt."
                        ).format(branch.solver, previous.normalized_error, current.repair_action_summary or "unchanged repair"),
                        problem_family=problem_family,
                        stage="repair",
                        solver=branch.solver,
                        solver_family=self._family(branch.solver),
                        normalized_error=previous.normalized_error,
                        solver_status=current.solver_status,
                        diagnosis="The repair did not change the observed failure.",
                        action="Do not repeat the same repair; change the failing expression, index, or formulation before rerunning.",
                        rationale="Two consecutive attempts produced the same normalized error.",
                        evidence=ExperienceEvidence(
                            problem_id=previous.problem_id,
                            branch_ids=[branch.branch_id],
                            attempt_ids=[previous.attempt_id, current.attempt_id],
                            solver_feedback_summary="repeated normalized error",
                            validation_level=ValidationLevel.RUNTIME_ONLY.value,
                            causal_confidence="medium",
                        ),
                        generality=ExperienceGenerality.SOLVER_SPECIFIC.value,
                    )
                )
        return [record for record in records if self._valid(record)]

    def extract_cross_branch(self, branches: List[BranchResult], problem_family: str) -> List[ExperienceRecord]:
        successful = [b for b in branches if b.execution.status in SUCCESS and b.validation.valid]
        if len(successful) < 2:
            return []
        objective_values = [b.execution.objective_value for b in successful]
        comparable = all(value is not None for value in objective_values)
        consistent = comparable and max(float(v) for v in objective_values) - min(float(v) for v in objective_values) <= 1e-6 * max(1.0, max(abs(float(v)) for v in objective_values))
        if not consistent:
            return []
        branch_ids = [b.branch_id for b in successful]
        attempt_ids = [b.attempts[-1].attempt_id for b in successful if b.attempts]
        record = self._record(
            layer=ExperienceLayer.MODELING.value,
            polarity=ExperiencePolarity.POSITIVE.value,
            title="Preserve solver-independent semantics across {} formulations".format(problem_family),
            retrieval_text=(
                "For {}, preserve the same decision-variable meaning, objective sense, and constraint semantics "
                "across heterogeneous solver implementations before comparing objective values. Independent branches "
                "reached consistent feasible objectives."
            ).format(problem_family),
            problem_family=problem_family,
            stage="formulation",
            solver=None,
            solver_family=None,
            normalized_error=None,
            solver_status="optimal_or_feasible",
            diagnosis="Independent solver branches implemented comparable formulations.",
            action="Define a solver-independent formulation contract and keep variable, objective, and constraint semantics identical in each adapter.",
            rationale="At least two isolated solver branches returned consistent feasible objectives.",
            evidence=ExperienceEvidence(
                problem_id=successful[0].attempts[0].problem_id,
                branch_ids=branch_ids,
                attempt_ids=attempt_ids,
                solver_feedback_summary="cross-solver consistent objective and feasible statuses",
                validation_level=ValidationLevel.CROSS_SOLVER_CONSISTENT.value,
                causal_confidence="high",
            ),
            generality=ExperienceGenerality.SOLVER_AGNOSTIC.value,
        )
        return [record] if self._valid(record) else []

    def _record(
        self, layer: str, polarity: str, title: str, retrieval_text: str,
        problem_family: str, stage: str, solver: Optional[str], solver_family: Optional[str],
        normalized_error: Optional[str], solver_status: Optional[str], diagnosis: str,
        action: str, rationale: str, evidence: ExperienceEvidence, generality: str,
    ) -> ExperienceRecord:
        api = {
            "gurobi": "gurobipy", "scip": "pyscipopt", "highs": "highspy",
            "copt": "coptpy", "ortools": "ortools.cp_model", "pulp": "pulp", "pyomo": "pyomo",
        }.get(solver)
        return ExperienceRecord(
            layer=layer,
            polarity=polarity,
            title=title,
            retrieval_text=retrieval_text,
            problem_context=ProblemContext(
                problem_family=problem_family, objective_type="unknown", stage=stage,
                keywords=[problem_family, stage] + ([solver] if solver else []),
            ),
            scope=ExperienceScope(
                generality=generality, solver_family=solver_family, solver=solver,
                language="python", api=api if generality == ExperienceGenerality.SOLVER_SPECIFIC.value else None,
            ),
            trigger=ExperienceTrigger(
                situation=retrieval_text.split(".")[0], normalized_error=normalized_error,
                solver_status=solver_status,
            ),
            policy=ExperiencePolicy(diagnosis=diagnosis, action=action, rationale=rationale),
            evidence=evidence,
        )

    def _valid(self, record: ExperienceRecord) -> bool:
        data = record.to_dict()
        from ..core.store import compute_content_hash

        data["content_hash"] = compute_content_hash(data)
        return self.validator.validate(data).valid

    @staticmethod
    def _family(solver: str) -> str:
        return {
            "gurobi": "milp", "scip": "milp", "highs": "milp", "copt": "milp",
            "pulp": "milp", "pyomo": "milp", "ortools": "cp_sat",
        }.get(solver, "unknown")
