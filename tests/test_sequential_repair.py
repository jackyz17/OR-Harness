from __future__ import annotations

import asyncio
import tempfile
import unittest

from helpers import build_system, experience
from or_experience_bank.core.schemas import SolverExecutionResult
from or_experience_bank.solvers.mock import MockSolverAdapter


class SequentialRepairTests(unittest.TestCase):
    def test_failure_then_repair_success_uses_repair_bank(self):
        with tempfile.TemporaryDirectory() as temp:
            adapter = MockSolverAdapter("mock-a", [
                SolverExecutionResult(status="error", solver="mock-a", exit_code=1, normalized_error="TypeError: bad expression", stderr="TypeError: bad expression"),
                SolverExecutionResult(status="optimal", solver="mock-a", exit_code=0, objective_sense="minimize", objective_value=3, variables={"x": 1}),
            ])
            orchestrator, store, retriever, llm = build_system(temp, [adapter], ["# first code", "# changed repaired code"])
            store.append(experience(
                title="Replace bad expression with a solver linear expression", layer="repair",
                generality="solver_family", solver_family="mock", error="TypeError: bad expression",
            ))
            retriever.rebuild("repair")
            result = asyncio.run(orchestrator.solve("Assign tasks and minimize cost", ["mock-a"], auto_append=False))
            branch = result.branches[0]
            self.assertEqual(len(branch.attempts), 2)
            self.assertEqual(branch.termination_reason, "optimal")
            self.assertTrue(branch.attempts[1].retrieved_experience_ids["repair"])
            # The modeling stage prepends its own prompts, so locate the repair-attempt
            # code-generation prompt by content rather than by absolute index.
            code_prompts = [p for p in llm.prompts if "Generate only executable Python code" in p]
            repair_prompt = code_prompts[-1]
            self.assertIn("Latest branch state", repair_prompt)
            self.assertIn("TypeError: bad expression", repair_prompt)

    def test_repeated_error_unchanged_code_and_max_attempts_stop(self):
        with tempfile.TemporaryDirectory() as temp:
            repeated = MockSolverAdapter("repeat", [
                SolverExecutionResult(status="error", solver="repeat", exit_code=1, normalized_error="same error"),
                SolverExecutionResult(status="error", solver="repeat", exit_code=1, normalized_error="same error"),
            ])
            orchestrator, _, _, _ = build_system(temp, [repeated], ["# code one", "# code two"])
            result = asyncio.run(orchestrator.solve("generic minimize model", ["repeat"], auto_append=False))
            self.assertEqual(result.branches[0].termination_reason, "repeated_error")

        with tempfile.TemporaryDirectory() as temp:
            unchanged = MockSolverAdapter("unchanged", [
                SolverExecutionResult(status="error", solver="unchanged", exit_code=1, normalized_error="first"),
                SolverExecutionResult(status="error", solver="unchanged", exit_code=1, normalized_error="second"),
            ])
            orchestrator, _, _, _ = build_system(temp, [unchanged], ["# identical", "# identical"])
            result = asyncio.run(orchestrator.solve("generic minimize model", ["unchanged"], auto_append=False))
            self.assertEqual(result.branches[0].termination_reason, "unchanged_code")

        with tempfile.TemporaryDirectory() as temp:
            maximum = MockSolverAdapter("maximum", [
                SolverExecutionResult(status="error", solver="maximum", exit_code=1, normalized_error="one"),
                SolverExecutionResult(status="error", solver="maximum", exit_code=1, normalized_error="two"),
            ])
            orchestrator, _, _, _ = build_system(temp, [maximum], ["# one", "# two"])
            result = asyncio.run(orchestrator.solve("generic minimize model", ["maximum"], max_attempts=2, auto_append=False))
            self.assertEqual(result.branches[0].termination_reason, "max_attempts")

