"""Gold-answer evaluation and outer reflective loop (Phase 1 module 1.2, Option A).

Harness constraint (user decision): the gold answer is NOT supplied with the problem;
it arrives AFTER solving, in an interactive dialogue. So solve() only runs
model -> branches and returns; it does NOT judge or extract. Judgement and the
reflection decision happen here, driven afterwards by the harness agent.

Two-step flow (Option A):
  1. result = await orchestrator.solve(problem)          # no gold, no extraction
  2. verdict = await orchestrator.evaluate_with_gold(...) # gold arrives -> judge + decide

On match: extract via comparative synthesis and append (done by caller / later module).
On mismatch: produce a reflection that feeds a fresh modeling round (<= max rounds).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..core.schemas import BranchResult, SolveResult, ValidationLevel


# Default relative tolerance for matching a selected objective to the gold answer.
DEFAULT_GOLD_TOLERANCE = 1e-4


@dataclass
class GoldVerdict:
    """Outcome of comparing the selected branch against the gold answer."""

    matched: bool
    gold: Optional[float] = None
    selected_objective: Optional[float] = None
    selected_branch_id: Optional[str] = None
    reason: str = ""
    # when matched, True means it is safe to run comparative synthesis + append
    ready_for_extraction: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "matched": self.matched,
            "gold": self.gold,
            "selected_objective": self.selected_objective,
            "selected_branch_id": self.selected_branch_id,
            "reason": self.reason,
            "ready_for_extraction": self.ready_for_extraction,
        }


def _selected_branch(result: SolveResult) -> Optional[BranchResult]:
    for branch in result.branches:
        if branch.branch_id == result.selected_branch_id:
            return branch
    return None


def evaluate_gold(
    result: SolveResult,
    gold: Optional[float],
    tolerance: float = DEFAULT_GOLD_TOLERANCE,
) -> GoldVerdict:
    """Compare the selected branch's objective to the gold answer.

    Gold is provided AFTER solving (Option A). When gold is None the caller has not
    supplied it; we fall back to cross-solver consistency as a weak criterion (D3) and
    never mark ready_for_extraction on a guess.
    """
    branch = _selected_branch(result)
    if branch is None or branch.execution is None:
        return GoldVerdict(matched=False, gold=gold, reason="no selected branch or execution")

    objective = branch.execution.objective_value
    selected_id = branch.branch_id

    if gold is None:
        # No gold: weak acceptance only if validation already reached a strong level.
        strong = branch.validation.validation_level in {
            ValidationLevel.CROSS_SOLVER_CONSISTENT.value,
            ValidationLevel.SEMANTIC_CHECKED.value,
        }
        return GoldVerdict(
            matched=strong,
            gold=None,
            selected_objective=objective,
            selected_branch_id=selected_id,
            reason="no gold supplied; weak accept via validation_level=" + branch.validation.validation_level,
            ready_for_extraction=strong,
        )

    if objective is None:
        return GoldVerdict(
            matched=False, gold=gold, selected_branch_id=selected_id,
            reason="selected branch produced no objective value",
        )

    diff = abs(float(objective) - float(gold))
    scale = max(1.0, abs(float(gold)))
    matched = diff <= tolerance * scale
    return GoldVerdict(
        matched=matched,
        gold=gold,
        selected_objective=objective,
        selected_branch_id=selected_id,
        reason="objective {} vs gold {} (diff {:.3g})".format(objective, gold, diff),
        ready_for_extraction=matched,
    )


class ReflectionGenerator:
    """Builds the reflection text that drives a fresh modeling round after a gold mismatch.

    The framework composes the prompt; the harness agent's LLM writes the reflection.
    """

    def build_reflection_prompt(self, problem: str, result: SolveResult, verdict: GoldVerdict) -> str:
        branch_summaries = []
        for branch in result.branches:
            branch_summaries.append(
                "- solver={} status={} objective={} termination={}".format(
                    branch.solver,
                    branch.execution.status if branch.execution else "unknown",
                    branch.execution.objective_value if branch.execution else None,
                    branch.termination_reason,
                )
            )
        return (
            "The OR model solved below did NOT match the gold answer. Analyze WHY the "
            "modeling direction was wrong (not the code), then state how the model should "
            "change. Be concrete about decision variables, objective, and constraints.\n\n"
            "PROBLEM:\n" + problem + "\n\n"
            "GOLD ANSWER: " + str(verdict.gold) + "\n"
            "SELECTED OBJECTIVE: " + str(verdict.selected_objective) + "\n\n"
            "BRANCH OUTCOMES:\n" + "\n".join(branch_summaries) + "\n\n"
            "Write a concise reflection and a corrected modeling direction."
        )


__all__ = ["DEFAULT_GOLD_TOLERANCE", "GoldVerdict", "evaluate_gold", "ReflectionGenerator"]
