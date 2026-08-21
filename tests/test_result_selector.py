from __future__ import annotations

import unittest

from helpers import SRC_DIR
from or_experience_bank.solving.result_selector import ResultSelector
from or_experience_bank.core.schemas import BranchResult, SolverExecutionResult, ValidationReport


def branch(branch_id, level, objective, sense="minimize"):
    return BranchResult(
        branch_id=branch_id, solver=branch_id, workspace="runs/" + branch_id, attempts=[],
        execution=SolverExecutionResult(status="optimal", solver=branch_id, exit_code=0, objective_sense=sense, objective_value=objective, variables={}),
        validation=ValidationReport(True, level), termination_reason="optimal",
    )


class ResultSelectorTests(unittest.TestCase):
    def test_validation_precedes_objective_and_mismatched_sense_disables_comparison(self):
        selected = ResultSelector().select([
            branch("semantic", "semantic_checked", 10),
            branch("feasible", "solver_feasible", 1),
        ])
        self.assertEqual(selected["selected_branch_id"], "semantic")
        mismatch = ResultSelector().select([
            branch("min", "solver_feasible", 1, "minimize"),
            branch("max", "solver_feasible", 100, "maximize"),
        ])
        self.assertFalse(mismatch["objective_comparable"])
        self.assertTrue(mismatch["branch_discrepancies"])
