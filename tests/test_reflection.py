"""Tests for Option A two-step solve + gold evaluation and outer reflection (module 1.2)."""

import asyncio
import tempfile
import unittest

from helpers import build_system
from or_experience_bank.solving.reflection import ReflectionGenerator, evaluate_gold
from or_experience_bank.core.schemas import SolverExecutionResult
from or_experience_bank.solvers.mock import MockSolverAdapter


def _adapters(obj=7):
    return [
        MockSolverAdapter("gurobi", [
            SolverExecutionResult(status="optimal", solver="gurobi", exit_code=0,
                                  objective_sense="minimize", objective_value=obj, variables={"x": 1}),
        ], delay=0.01),
        MockSolverAdapter("scip", [
            SolverExecutionResult(status="optimal", solver="scip", exit_code=0,
                                  objective_sense="minimize", objective_value=obj, variables={"x": 1}),
        ], delay=0.01),
    ]


class DeferExtractionTest(unittest.TestCase):
    def test_solve_deferred_does_not_append(self):
        with tempfile.TemporaryDirectory() as temp:
            orchestrator, store, retriever, _ = build_system(temp, _adapters(), ["# c1", "# c2"])
            result = asyncio.run(orchestrator.solve(
                "Assign tasks and minimize cost", ["gurobi", "scip"], defer_extraction=True
            ))
            # deferred: nothing appended yet
            self.assertEqual(result.appended_experience_ids, [])
            self.assertTrue(orchestrator._pending)

    def test_evaluate_gold_match_appends(self):
        with tempfile.TemporaryDirectory() as temp:
            orchestrator, store, retriever, _ = build_system(temp, _adapters(), ["# c1", "# c2"])
            asyncio.run(orchestrator.solve("Assign tasks and minimize cost", ["gurobi", "scip"], defer_extraction=True))
            verdict = asyncio.run(orchestrator.evaluate_with_gold(gold=7))
            self.assertTrue(verdict.matched)
            self.assertTrue(verdict.ready_for_extraction)
            self.assertTrue(orchestrator._pending["appended_experience_ids"])

    def test_evaluate_gold_mismatch_no_append(self):
        with tempfile.TemporaryDirectory() as temp:
            orchestrator, store, retriever, _ = build_system(temp, _adapters(), ["# c1", "# c2"])
            asyncio.run(orchestrator.solve("Assign tasks and minimize cost", ["gurobi", "scip"], defer_extraction=True))
            verdict = asyncio.run(orchestrator.evaluate_with_gold(gold=999))
            self.assertFalse(verdict.matched)
            self.assertFalse(verdict.ready_for_extraction)
            self.assertEqual(orchestrator._pending.get("appended_experience_ids", []), [])

    def test_evaluate_without_defer_raises(self):
        with tempfile.TemporaryDirectory() as temp:
            orchestrator, _, _, _ = build_system(temp, _adapters(), ["# c1"])
            with self.assertRaises(RuntimeError):
                asyncio.run(orchestrator.evaluate_with_gold(gold=7))

    def test_gold_none_weak_accept(self):
        with tempfile.TemporaryDirectory() as temp:
            orchestrator, store, retriever, _ = build_system(temp, _adapters(), ["# c1", "# c2"])
            asyncio.run(orchestrator.solve("Assign tasks and minimize cost", ["gurobi", "scip"], defer_extraction=True))
            verdict = asyncio.run(orchestrator.evaluate_with_gold(gold=None))
            # cross-solver consistent -> weak accept
            self.assertTrue(verdict.matched)


class ReflectionPromptTest(unittest.TestCase):
    def test_build_reflection_prompt_after_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            orchestrator, store, retriever, _ = build_system(temp, _adapters(), ["# c1", "# c2"])
            asyncio.run(orchestrator.solve("Assign tasks and minimize cost", ["gurobi", "scip"], defer_extraction=True))
            verdict = asyncio.run(orchestrator.evaluate_with_gold(gold=999))
            self.assertFalse(verdict.matched)
            prompt = orchestrator.build_reflection_prompt(verdict)
            self.assertIn("gold", prompt.lower())
            self.assertIn("999", prompt)
            self.assertIn("model", prompt.lower())


if __name__ == "__main__":
    unittest.main()
