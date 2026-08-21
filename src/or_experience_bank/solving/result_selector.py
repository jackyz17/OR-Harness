"""Select the strongest validated branch without unsafe objective comparison."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from ..core.schemas import BranchResult, ValidationLevel


LEVEL_RANK = {
    ValidationLevel.UNVERIFIED.value: 0,
    ValidationLevel.RUNTIME_ONLY.value: 1,
    ValidationLevel.SOLVER_FEASIBLE.value: 3,
    ValidationLevel.CROSS_SOLVER_CONSISTENT.value: 4,
    ValidationLevel.SEMANTIC_CHECKED.value: 5,
}
STATUS_RANK = {"error": 0, "unknown": 0, "timeout": 0, "infeasible": 1, "unbounded": 1, "feasible": 3, "optimal": 4}


class ResultSelector:
    def select(self, branches: List[BranchResult]) -> Dict:
        if not branches:
            return {
                "selected_branch_id": None, "selection_reason": "No solver branch completed",
                "discarded_branches": [], "branch_discrepancies": [], "objective_comparable": False,
            }
        successful = [b for b in branches if b.execution.status in {"optimal", "feasible"} and b.validation.valid]
        senses = {b.execution.objective_sense for b in successful}
        objectives_present = all(b.execution.objective_value is not None for b in successful)
        objective_comparable = bool(successful) and len(senses) == 1 and "unknown" not in senses and objectives_present
        discrepancies = []
        if len(senses) > 1:
            discrepancies.append("objective senses differ across branches")
        if successful and not objectives_present:
            discrepancies.append("one or more successful branches lack objective values")

        def key(branch: BranchResult):
            level = LEVEL_RANK.get(branch.validation.validation_level, 0)
            status = STATUS_RANK.get(branch.execution.status, 0)
            objective = branch.execution.objective_value
            objective_score = 0.0
            if objective_comparable and objective is not None:
                objective_score = -float(objective) if branch.execution.objective_sense == "minimize" else float(objective)
            gap_score = -float(branch.execution.mip_gap or 0.0)
            runtime_score = -float(branch.execution.runtime_seconds or 0.0)
            return (level, status, objective_score, gap_score, runtime_score, branch.branch_id)

        selected = max(branches, key=key)
        reason = "Selected highest validation level and solver status"
        if objective_comparable:
            reason += ", then best comparable objective/bound/gap/runtime"
        else:
            reason += "; objectives were not used for cross-branch ranking"
        return {
            "selected_branch_id": selected.branch_id,
            "selection_reason": reason,
            "discarded_branches": [b.branch_id for b in branches if b.branch_id != selected.branch_id],
            "branch_discrepancies": discrepancies,
            "objective_comparable": objective_comparable,
        }
