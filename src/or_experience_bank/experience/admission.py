"""Layer-aware conversion + admission gate for synthesis candidates (Phase 2 step 4).

The synthesis LLM self-classifies each lesson into a layer. At admission time we route
by that layer:
  - modeling                -> ModelingExperience (fused schema, signature required)
  - implementation/repair/solving -> flat ExperienceRecord

Then each candidate passes two gates before append:
  1. structural validation (ExperienceValidator for flat layers; ModelingExperience.validate)
  2. semantic judge (LLM-as-a-Judge: actionable / accurate / valuable) — D18: the
     framework composes the judge prompt and parses the verdict; the agent owns the LLM.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..llm_client import LLMClient
from ..core.modeling_schemas import (
    MathScope,
    MethodBody,
    ModelingEvidence,
    ModelingExperience,
    StructuralSignature,
)
from ..core.schemas import (
    ExperienceEvidence,
    ExperienceGenerality,
    ExperiencePolicy,
    ExperienceRecord,
    ExperienceScope,
    ExperienceTrigger,
    ProblemContext,
    ValidationLevel,
)
from ..core.store import compute_content_hash
from ..solving.validator import ExperienceValidator


class CandidateConversionError(ValueError):
    pass


def candidate_to_record(
    candidate: Dict[str, Any],
    problem_id: str,
    problem_family: str,
    branch_ids: List[str],
    attempt_ids: List[str],
    signature: Optional[StructuralSignature] = None,
    solver: Optional[str] = None,
    solver_family: Optional[str] = None,
) -> Any:
    """Route a synthesis candidate to the correct record type by its self-chosen layer."""
    layer = candidate.get("layer")
    if layer == "modeling":
        return _to_modeling_experience(candidate, problem_id, signature)
    if layer in ("implementation", "repair", "solving"):
        return _to_flat_record(candidate, problem_id, problem_family, branch_ids, attempt_ids, solver, solver_family)
    raise CandidateConversionError("unknown layer: " + str(layer))


def _to_modeling_experience(
    candidate: Dict[str, Any], problem_id: str, signature: Optional[StructuralSignature]
) -> ModelingExperience:
    record = ModelingExperience(
        title=candidate.get("title", ""),
        polarity=candidate.get("polarity", "positive"),
        retrieval_text=candidate.get("retrieval_text", candidate.get("title", "")),
        modeling_aspect=candidate.get("modeling_aspect", "constraint"),
    )
    if signature is not None:
        record.math_scope = MathScope(structural_signature=signature, exclusions=list(candidate.get("exclusions", [])))
    record.method = MethodBody(
        action_template=candidate.get("action", ""),
        wrong_form=candidate.get("wrong_form"),
        rationale=candidate.get("rationale", ""),
        derivation_ref=candidate.get("derivation_ref"),
    )
    record.evidence = ModelingEvidence(
        source_episodes=[problem_id],
        solver_feedback_summary=candidate.get("diagnosis", ""),
        validation_level=ValidationLevel.SOLVER_FEASIBLE.value,
        causal_confidence="medium",
    )
    record.validate()
    record.compute_content_hash()
    return record


def _to_flat_record(
    candidate: Dict[str, Any],
    problem_id: str,
    problem_family: str,
    branch_ids: List[str],
    attempt_ids: List[str],
    solver: Optional[str],
    solver_family: Optional[str],
) -> ExperienceRecord:
    generality = (
        ExperienceGenerality.SOLVER_SPECIFIC.value if solver
        else ExperienceGenerality.SOLVER_FAMILY.value if solver_family
        else ExperienceGenerality.SOLVER_AGNOSTIC.value
    )
    record = ExperienceRecord(
        layer=candidate["layer"],
        polarity=candidate.get("polarity", "positive"),
        title=candidate.get("title", ""),
        retrieval_text=candidate.get("retrieval_text", candidate.get("title", "")),
        problem_context=ProblemContext(
            problem_family=problem_family,
            objective_type="unknown",
            stage={"implementation": "implementation", "repair": "repair", "solving": "solving"}[candidate["layer"]],
            keywords=[problem_family],
        ),
        scope=ExperienceScope(generality=generality, solver_family=solver_family, solver=solver, language="python"),
        trigger=ExperienceTrigger(situation=candidate.get("diagnosis", "")[:200]),
        policy=ExperiencePolicy(
            diagnosis=candidate.get("diagnosis", ""),
            action=candidate.get("action", ""),
            rationale=candidate.get("rationale", ""),
        ),
        evidence=ExperienceEvidence(
            problem_id=problem_id,
            branch_ids=branch_ids,
            attempt_ids=attempt_ids,
            solver_feedback_summary=candidate.get("diagnosis", "")[:500],
            validation_level=ValidationLevel.SOLVER_FEASIBLE.value,
            causal_confidence="medium",
        ),
    )
    data = record.to_dict()
    data["content_hash"] = compute_content_hash(data)
    report = ExperienceValidator().validate(data)
    if not report.valid:
        raise CandidateConversionError("invalid flat record: " + "; ".join(report.errors))
    record.content_hash = data["content_hash"]
    return record


class AdmissionJudge:
    """Semantic gate (LLM-as-a-Judge). Framework composes prompt, agent owns LLM (D18)."""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client

    def build_judge_prompt(self, candidate: Dict[str, Any]) -> str:
        return (
            "Judge whether this candidate OR experience is (1) actionable, (2) accurate, "
            "(3) valuable for future solves. Answer with a single JSON object: "
            '{"accept": true|false, "reason": "..."}.\n\nCANDIDATE:\n' + str(candidate)
        )

    async def accept(self, candidate: Dict[str, Any]) -> bool:
        if self.llm is None:
            return True  # no judge available -> structural gate already passed
        try:
            verdict = await self.llm.generate_object(self.build_judge_prompt(candidate))
        except (AttributeError, TypeError, ValueError):
            return False
        if isinstance(verdict, dict):
            return bool(verdict.get("accept"))
        return False


__all__ = ["CandidateConversionError", "candidate_to_record", "AdmissionJudge"]
