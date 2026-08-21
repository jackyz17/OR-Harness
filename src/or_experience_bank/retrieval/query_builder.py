"""Stage-specific compact retrieval queries and feedback sanitization."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, Optional


ABSOLUTE_PATH = re.compile(r"(?:/Users|/home|/private|/tmp|[A-Za-z]:\\)[^\s:'\"]+")
MEMORY_ADDRESS = re.compile(r"0x[0-9a-fA-F]+")
SECRET = re.compile(r"(?i)(api[_-]?key|token|authorization|secret)\s*[:=]\s*[^\s,;]+")
TEMP_DIRECTORY = re.compile(r"(?:tmp|temp)[-_][A-Za-z0-9_.-]+")


def sanitize_feedback(text: str, max_chars: int = 4000) -> str:
    value = ABSOLUTE_PATH.sub("<path>", text or "")
    value = MEMORY_ADDRESS.sub("<memory-address>", value)
    value = SECRET.sub(lambda match: match.group(1) + "=<redacted>", value)
    value = TEMP_DIRECTORY.sub("<temp-dir>", value)
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    compact = "\n".join(lines[-40:])
    return compact[:max_chars]


class StageAwareQueryBuilder:
    def modeling(self, spec: Dict[str, Any], current_need: str) -> str:
        return "\n".join(
            [
                "Problem: " + str(spec.get("normalized_description", spec.get("description", ""))),
                "Problem family: " + str(spec.get("problem_family", "general_milp")),
                "Objective: " + str(spec.get("objective", "unknown")),
                "Key entities: " + ", ".join(spec.get("entities", [])),
                "Key constraints: " + ", ".join(spec.get("constraints", [])),
                "Current need: " + current_need,
            ]
        )

    def implementation(
        self,
        formulation_summary: str,
        solver: str,
        solver_family: str,
        api: str,
        need: str,
        language: str = "Python",
    ) -> str:
        return "\n".join(
            [
                "Solver: " + solver,
                "Solver family: " + solver_family,
                "Language: " + language,
                "API: " + api,
                "Model: " + formulation_summary,
                "Need: " + need,
            ]
        )

    def repair(
        self,
        solver: str,
        solver_family: str,
        normalized_error: str,
        traceback: str,
        solver_status: str,
        recent_action: str,
        current_summary: str,
        stage: str = "repair",
    ) -> str:
        return "\n".join(
            [
                "Solver: " + solver,
                "Solver family: " + solver_family,
                "Stage: " + stage,
                "Normalized error: " + sanitize_feedback(normalized_error, 800),
                "Traceback: " + sanitize_feedback(traceback, 2200),
                "Solver status: " + solver_status,
                "Recent action: " + sanitize_feedback(recent_action, 500),
                "Current model/code summary: " + sanitize_feedback(current_summary, 1000),
            ]
        )

    def solving(
        self,
        problem_family: str,
        instance_scale: str,
        variables: Optional[int],
        constraints: Optional[int],
        solver: str,
        solver_status: str,
        runtime: Optional[float],
        mip_gap: Optional[float],
        objective_bound: Optional[float],
        numerical_warning: str = "",
        performance_symptom: str = "",
    ) -> str:
        return "\n".join(
            [
                "Problem family: " + problem_family,
                "Instance scale: " + instance_scale,
                "Variables: " + str(variables),
                "Constraints: " + str(constraints),
                "Solver: " + solver,
                "Solver status: " + solver_status,
                "Runtime: " + str(runtime),
                "MIP gap: " + str(mip_gap),
                "Objective bound: " + str(objective_bound),
                "Numerical warning: " + sanitize_feedback(numerical_warning, 500),
                "Performance symptom: " + sanitize_feedback(performance_symptom, 500),
            ]
        )
