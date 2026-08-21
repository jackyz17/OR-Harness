"""Experience and solver-result validation."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from ..retrieval.query_builder import ABSOLUTE_PATH, SECRET
from ..core.schemas import (
    ExperienceGenerality,
    ExperienceLayer,
    ExperiencePolarity,
    SolverExecutionResult,
    ValidationLevel,
    ValidationReport,
)
from ..core.store import compute_content_hash


VAGUE_ACTIONS = {
    "check constraints",
    "carefully check constraints",
    "debug the model",
    "检查约束",
    "仔细检查约束",
}


class ExperienceValidator:
    def validate(self, record: Dict[str, Any]) -> ValidationReport:
        errors = []
        required = {
            "schema_version", "experience_id", "created_at", "layer", "polarity",
            "title", "retrieval_text", "problem_context", "scope", "trigger",
            "policy", "evidence", "content_hash",
        }
        missing = sorted(required - set(record))
        if missing:
            errors.append("missing fields: " + ", ".join(missing))
            return ValidationReport(valid=False, errors=errors)
        try:
            ExperienceLayer(record["layer"])
        except ValueError:
            errors.append("invalid layer")
        try:
            ExperiencePolarity(record["polarity"])
        except ValueError:
            errors.append("invalid polarity")
        if not str(record.get("title", "")).strip():
            errors.append("title is empty")
        if not str(record.get("retrieval_text", "")).strip():
            errors.append("retrieval_text is empty")
        policy = record.get("policy", {})
        action = str(policy.get("action", "")).strip()
        if not action:
            errors.append("policy.action is empty")
        if action.lower() in VAGUE_ACTIONS:
            errors.append("policy.action is too vague")
        evidence = record.get("evidence", {})
        if not str(evidence.get("problem_id", "")).strip():
            errors.append("evidence.problem_id is empty")
        if not evidence.get("branch_ids") and not evidence.get("attempt_ids"):
            errors.append("evidence requires branch_ids or attempt_ids")
        if evidence.get("causal_confidence") not in {"low", "medium", "high"}:
            errors.append("invalid causal_confidence")
        try:
            ValidationLevel(evidence.get("validation_level", ""))
        except ValueError:
            errors.append("invalid validation_level")
        scope = record.get("scope", {})
        generality = scope.get("generality")
        try:
            ExperienceGenerality(generality)
        except ValueError:
            errors.append("invalid generality")
        if generality == ExperienceGenerality.SOLVER_SPECIFIC.value and not scope.get("solver"):
            errors.append("solver_specific experience requires scope.solver")
        if generality == ExperienceGenerality.SOLVER_FAMILY.value and not scope.get("solver_family"):
            errors.append("solver_family experience requires scope.solver_family")
        if generality == ExperienceGenerality.SOLVER_AGNOSTIC.value and scope.get("api"):
            errors.append("solver_agnostic experience must not name a solver API")
        serialized = str(record)
        if SECRET.search(serialized):
            errors.append("record contains a possible secret")
        if ABSOLUTE_PATH.search(serialized):
            errors.append("record contains an absolute user path")
        if len(str(record.get("trigger", {}).get("normalized_error", ""))) > 4000:
            errors.append("normalized_error is too long")
        if record.get("content_hash") and record["content_hash"] != compute_content_hash(record):
            errors.append("content_hash mismatch")
        return ValidationReport(valid=not errors, errors=errors)


class ResultValidator:
    VALID_STATUSES = {"optimal", "feasible", "infeasible", "unbounded", "timeout", "error", "unknown"}

    def validate(
        self,
        execution: SolverExecutionResult,
        reference_objective: Optional[float] = None,
        semantic_validator: Optional[Callable[[SolverExecutionResult], bool]] = None,
    ) -> ValidationReport:
        errors = []
        warnings = []
        # Normalize status to lowercase — PuLP returns "Optimal", Gurobi returns
        # GRB.OPTIMAL (integer → adapter converts). Different solvers use different
        # capitalization; the contract is case-insensitive.
        status_lower = (execution.status or "unknown").lower()
        if execution.exit_code not in (0, None) and status_lower not in {"timeout", "error"}:
            errors.append("execution process failed")
        if status_lower not in self.VALID_STATUSES:
            errors.append("invalid solver status: {}".format(execution.status))
        level = ValidationLevel.UNVERIFIED.value
        if execution.exit_code == 0:
            level = ValidationLevel.RUNTIME_ONLY.value
        if status_lower in {"optimal", "feasible"}:
            level = ValidationLevel.SOLVER_FEASIBLE.value
            if execution.objective_value is not None:
                try:
                    if not math.isfinite(float(execution.objective_value)):
                        errors.append("objective is not finite")
                except (TypeError, ValueError):
                    errors.append("objective is not numeric")
            if not isinstance(execution.variables, dict):
                errors.append("variables is not an object")
        if reference_objective is not None and execution.objective_value is not None:
            tolerance = 1e-6 * max(1.0, abs(reference_objective))
            if abs(float(execution.objective_value) - reference_objective) > tolerance:
                errors.append("objective does not match reference")
        if semantic_validator is not None and not errors:
            try:
                if semantic_validator(execution):
                    level = ValidationLevel.SEMANTIC_CHECKED.value
                else:
                    errors.append("problem-specific semantic validator failed")
            except Exception as exc:  # user validator failure is reported, not swallowed
                errors.append("semantic validator error: " + type(exc).__name__)
        if status_lower in {"infeasible", "unbounded"}:
            warnings.append("terminal solver status requires independent model review")
        return ValidationReport(valid=not errors, validation_level=level, errors=errors, warnings=warnings)

