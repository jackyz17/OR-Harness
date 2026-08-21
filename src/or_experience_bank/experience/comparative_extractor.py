"""Comparative synthesis extractor (Phase 2 step 3): success-vs-failure contrast.

User decision: failure experiences are NOT appended on their own, and the experience
from a reflection-recovered success is treated as a plain success (it does not capture
the failure's value). Instead, once a solve succeeds, we feed the SUCCESS side plus ALL
buffered failures to a summarizer LLM, which contrasts them and emits bank-classified
experience candidates.

Branch rule:
  - failures present  -> comparative synthesis (success vs failure)
  - no failures        -> plain success channel

The framework composes the prompt and validates/parses output; the harness agent's LLM
writes the synthesis (D18). Falls back to the deterministic rule-based extractor when no
LLM is available, so behavior never regresses.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .failure_buffer import FailureBuffer
from ..llm_client import LLMClient
from ..core.schemas import BranchResult


BANK_LAYERS = ("modeling", "implementation", "repair", "solving")

SUCCESS = {"optimal", "feasible"}

_SYNTHESIS_INSTRUCTION = (
    "You are extracting reusable OR experiences by CONTRASTING a successful solve with "
    "the failures that preceded it. Identify the KEY DIFFERENCES that separate success "
    "from failure — what the success did right that the failures missed, and what each "
    "failure did wrong. For each distinct lesson, output one JSON object with keys: "
    "layer (one of modeling|implementation|repair|solving), title, retrieval_text, "
    "polarity (positive|negative), diagnosis, action, rationale. "
    "For modeling-layer experiences, also specify modeling_aspect: exactly one of "
    "constraint|objective|variable|classification|structure. "
    "Classify by WHERE the lesson applies: modeling=formulation semantics, "
    "implementation=solver/API mechanics, repair=error->fix path, solving=performance/"
    "solver-choice. Return a JSON array only."
)


class ComparativeSynthesisExtractor:
    """Contrasts success with buffered failures to produce bank-classified candidates."""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client

    def has_failures(self, failures: Optional[FailureBuffer]) -> bool:
        return failures is not None and not failures.is_empty()

    def build_synthesis_prompt(
        self,
        problem: str,
        success_summary: str,
        failures: FailureBuffer,
    ) -> str:
        """Compose the contrast prompt for the agent's LLM."""
        failure_lines = []
        for record in failures.all():
            location = record.solver or record.stage
            detail = record.normalized_error or record.summary
            failure_lines.append("- [{}] {}: {}".format(record.stage, location, detail))
        return (
            _SYNTHESIS_INSTRUCTION
            + "\n\nPROBLEM:\n" + problem
            + "\n\nSUCCESSFUL OUTCOME:\n" + success_summary
            + "\n\nFAILURES THAT PRECEDED SUCCESS:\n" + ("\n".join(failure_lines) or "(none)")
        )

    def build_success_only_prompt(self, problem: str, success_summary: str) -> str:
        """Plain success channel when there were no failures to contrast."""
        return (
            "Extract reusable OR experiences from this SUCCESSFUL solve. For each distinct "
            "lesson output one JSON object with keys: layer (modeling|implementation|repair|"
            "solving), title, retrieval_text, polarity, diagnosis, action, rationale. "
            "Return a JSON array only.\n\nPROBLEM:\n" + problem
            + "\n\nSUCCESSFUL OUTCOME:\n" + success_summary
        )

    def summarize_success(self, branches: List[BranchResult], verified_model: str = "") -> str:
        """Compact success-side summary: verified model + per-branch outcomes."""
        lines = []
        if verified_model:
            lines.append("Verified model:\n" + verified_model)
        for branch in branches:
            if branch.execution and branch.execution.status in SUCCESS:
                lines.append(
                    "solver={} status={} objective={} attempts={}".format(
                        branch.solver,
                        branch.execution.status,
                        branch.execution.objective_value,
                        len(branch.attempts),
                    )
                )
        return "\n".join(lines) or "(no successful branch)"

    def parse_candidates(self, raw: Any) -> List[Dict[str, Any]]:
        """Parse the LLM JSON array into normalized candidate dicts (bank-classified)."""
        candidates: List[Dict[str, Any]] = []
        if not isinstance(raw, list):
            return candidates
        for item in raw:
            if not isinstance(item, dict):
                continue
            layer = item.get("layer")
            if layer not in BANK_LAYERS:
                continue
            candidates.append(
                {
                    "layer": layer,
                    "title": str(item.get("title", "")).strip(),
                    "retrieval_text": str(item.get("retrieval_text", item.get("title", ""))).strip(),
                    "polarity": item.get("polarity", "positive"),
                    "diagnosis": str(item.get("diagnosis", "")).strip(),
                    "action": str(item.get("action", "")).strip(),
                    "rationale": str(item.get("rationale", "")).strip(),
                }
            )
        return [c for c in candidates if c["title"] and c["action"]]

    async def synthesize(
        self,
        problem: str,
        branches: List[BranchResult],
        failures: Optional[FailureBuffer],
        verified_model: str = "",
    ) -> List[Dict[str, Any]]:
        """Run contrast (or success-only) synthesis and return candidate dicts."""
        if self.llm is None:
            return []
        success_summary = self.summarize_success(branches, verified_model)
        if self.has_failures(failures):
            prompt = self.build_synthesis_prompt(problem, success_summary, failures)
        else:
            prompt = self.build_success_only_prompt(problem, success_summary)
        try:
            raw = await self.llm.generate_object(prompt)
        except (AttributeError, TypeError, ValueError):
            return []
        return self.parse_candidates(raw)


__all__ = ["BANK_LAYERS", "ComparativeSynthesisExtractor"]
