"""Tests for induction/counterexample.py (module 3.5: solver-backed refutation)."""

from __future__ import annotations

import asyncio
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from or_experience_bank.induction.counterexample import (
    CounterexampleSearcher,
    LLMBackedCounterexampleSearcher,
    RefutationAttempt,
)
from or_experience_bank.induction.inducer import PrincipleHypothesis
from or_experience_bank.core.schemas import SolverExecutionResult
from or_experience_bank.llm_client import FakeLLMClient


def make_hypothesis():
    return PrincipleHypothesis(
        hypothesis_id="hyp_x",
        statement="Prioritize higher marginal-contribution decisions for a shared scarce resource.",
        roles_used=["resource_pool", "competing_decisions"],
        source_realization_ids=["e1", "e2"],
        applicability_conditions=["linear objective"],
        status="hypothesis",
    )


class StubExecutor:
    """Returns a canned SolverExecutionResult; records the program it was asked to run."""

    def __init__(self, stdout="", status="ok"):
        self.stdout = stdout
        self.status = status
        self.ran = []

    async def execute(self, code_path, workspace, solver):
        self.ran.append(Path(code_path).read_text())
        return SolverExecutionResult(status=self.status, solver=solver, stdout=self.stdout)


class CounterexampleSearcherTest(unittest.TestCase):
    def test_failure_conditions_prompt_mentions_structural_breaks(self):
        prompt = CounterexampleSearcher().build_failure_conditions_prompt(make_hypothesis())
        self.assertIn("fixed-charge", prompt)
        self.assertIn("nonconvex", prompt.lower())

    def test_parse_failure_conditions(self):
        s = CounterexampleSearcher()
        conds = s.parse_failure_conditions(["fixed setup cost", "nonconvex coupling"])
        self.assertEqual(conds, ["fixed setup cost", "nonconvex coupling"])

    def test_parse_verdict_reads_last_json_line(self):
        s = CounterexampleSearcher()
        stdout = "computing...\n{\"principle_failed\": true, \"evidence\": \"fixed charge dominates\"}\n"
        self.assertTrue(s.parse_verdict(stdout))
        self.assertIsNone(s.parse_verdict("no json here"))

    def test_executor_verdict_confirms_counterexample(self):
        # LLM program CLAIMS nothing; the EXECUTED output is what counts.
        executor = StubExecutor(stdout='{"principle_failed": true, "evidence": "setup cost"}')
        s = CounterexampleSearcher(executor=executor)
        attempt = RefutationAttempt(condition="fixed setup cost", program="print('x')")
        with tempfile.TemporaryDirectory() as tmp:
            out = asyncio.run(s.run_refutation(attempt, Path(tmp)))
        self.assertTrue(out.executed)
        self.assertTrue(out.is_counterexample)
        self.assertEqual(executor.ran, ["print('x')"])

    def test_survived_condition_not_counterexample(self):
        executor = StubExecutor(stdout='{"principle_failed": false}')
        s = CounterexampleSearcher(executor=executor)
        attempt = RefutationAttempt(condition="nonconvex", program="pass")
        with tempfile.TemporaryDirectory() as tmp:
            out = asyncio.run(s.run_refutation(attempt, Path(tmp)))
        self.assertFalse(out.is_counterexample)

    def test_executor_error_is_not_counterexample(self):
        # anti self-judgment: a crashed refutation proves nothing.
        executor = StubExecutor(stdout="", status="error")
        s = CounterexampleSearcher(executor=executor)
        attempt = RefutationAttempt(condition="min batch", program="boom")
        with tempfile.TemporaryDirectory() as tmp:
            out = asyncio.run(s.run_refutation(attempt, Path(tmp)))
        self.assertFalse(out.is_counterexample)

    def test_no_executor_defers_to_harness(self):
        s = CounterexampleSearcher(executor=None)
        attempt = RefutationAttempt(condition="c", program="p")
        out = asyncio.run(s.run_refutation(attempt, Path("/tmp/nope")))
        self.assertFalse(out.executed)
        self.assertIn("no executor", out.evidence)

    def test_aggregate_shrinks_applicability_and_records(self):
        s = CounterexampleSearcher()
        attempts = [
            RefutationAttempt(condition="fixed charge", executed=True, principle_failed=True, evidence="e"),
            RefutationAttempt(condition="nonconvex", executed=True, principle_failed=False),
        ]
        result = s.aggregate(make_hypothesis(), attempts)
        self.assertTrue(result.refuted)
        self.assertEqual(len(result.counterexamples), 1)
        self.assertEqual(result.counterexamples[0].summary, "fixed charge")
        self.assertIn("NOT when: fixed charge", result.shrunk_applicability)
        self.assertEqual(result.surviving_conditions, ["nonconvex"])


class LLMBackedCounterexampleSearcherTest(unittest.TestCase):
    def test_full_loop_executor_decides(self):
        # LLM proposes condition + program; stub executor "runs" it and reports failure.
        executor = StubExecutor(stdout='{"principle_failed": true, "evidence": "confirmed"}')
        searcher = CounterexampleSearcher(executor=executor)
        llm = FakeLLMClient(
            text_responses=["print('refutation code')"],
            object_responses=[["fixed setup cost"]],
        )
        driver = LLMBackedCounterexampleSearcher(searcher=searcher, llm_client=llm)
        with tempfile.TemporaryDirectory() as tmp:
            result = asyncio.run(driver.search(make_hypothesis(), Path(tmp)))
        self.assertTrue(result.refuted)
        self.assertEqual(len(result.counterexamples), 1)

    def test_requires_client(self):
        driver = LLMBackedCounterexampleSearcher(llm_client=None)
        with tempfile.TemporaryDirectory() as tmp:
            result = asyncio.run(driver.search(make_hypothesis(), Path(tmp)))
        self.assertFalse(result.refuted)


if __name__ == "__main__":
    unittest.main()
