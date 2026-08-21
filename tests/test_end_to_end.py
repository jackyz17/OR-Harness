from __future__ import annotations

import asyncio
import tempfile
import unittest

from helpers import build_system
from or_experience_bank.core.schemas import SolverExecutionResult
from or_experience_bank.solvers.mock import MockSolverAdapter


class EndToEndTests(unittest.TestCase):
    def test_parallel_repair_extract_append_and_immediate_retrieve(self):
        with tempfile.TemporaryDirectory() as temp:
            adapters = [
                MockSolverAdapter("gurobi", [
                    SolverExecutionResult(status="error", solver="gurobi", exit_code=1, normalized_error="TypeError: list expression", stderr="TypeError: list expression"),
                    SolverExecutionResult(status="optimal", solver="gurobi", exit_code=0, objective_sense="minimize", objective_value=7, variables={"x_1": 1}),
                ], delay=0.01),
                MockSolverAdapter("scip", [
                    SolverExecutionResult(status="optimal", solver="scip", exit_code=0, objective_sense="minimize", objective_value=7, variables={"x_1": 1}),
                ], delay=0.01),
            ]
            orchestrator, store, retriever, _ = build_system(
                temp, adapters, ["# gurobi first", "# scip first", "# gurobi repaired"]
            )
            result = asyncio.run(orchestrator.solve(
                "Assign tasks to machines and minimize total cost", ["gurobi", "scip"], max_attempts=3, auto_append=True
            ))
            self.assertEqual(len(result.branches), 2)
            self.assertTrue(result.selected_branch_id)
            self.assertTrue(result.appended_experience_ids)
            self.assertIn(result.validation_level, {"cross_solver_consistent", "solver_feasible"})
            repair = retriever.retrieve(
                "repair", "TypeError list expression successful repair",
                {"solver": "gurobi", "solver_family": "milp"}, top_k=5,
            )
            modeling = retriever.retrieve("modeling", "solver independent assignment formulation", top_k=5)
            self.assertTrue(repair)
            self.assertTrue(modeling)
            self.assertTrue(store.validate_bank()["valid"])

