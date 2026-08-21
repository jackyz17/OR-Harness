"""Tests for induction/validation.py (module 3.6: transfer validation + scoring)."""

from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from or_experience_bank.induction.counterexample import RefutationAttempt, RefutationResult
from or_experience_bank.induction.inducer import PrincipleHypothesis
from or_experience_bank.induction.validation import (
    LLMBackedValidator,
    PatternValidator,
    ScoringWeights,
)


def make_hypothesis(complexity=0.3):
    return PrincipleHypothesis(
        hypothesis_id="hyp_v",
        statement="Prioritize higher marginal-contribution decisions for a shared scarce resource.",
        roles_used=["resource_pool", "competing_decisions"],
        source_realization_ids=["e1", "e2", "e3"],
        applicability_conditions=["linear objective"],
        complexity=complexity,
        status="hypothesis",
    )


def make_transfer_solver(objectives):
    """objectives: dict[(task, with_principle)] -> objective value."""

    async def solve(task, principle):
        return objectives.get((task, principle is not None))

    return solve


class SourceConsistencyTest(unittest.TestCase):
    def test_coverage_fraction(self):
        v = PatternValidator()
        h = make_hypothesis()
        self.assertAlmostEqual(v.source_consistency(h, ["e1", "e2"]), 2 / 3)
        self.assertAlmostEqual(v.source_consistency(h, ["e1", "e2", "e3"]), 1.0)

    def test_no_sources_zero_coverage(self):
        h = make_hypothesis()
        h.source_realization_ids = []
        self.assertEqual(PatternValidator().source_consistency(h, []), 0.0)


class UnseenTransferTest(unittest.TestCase):
    def test_with_without_comparison_minimize(self):
        v = PatternValidator()
        solver = make_transfer_solver({
            ("taskA", True): 10.0, ("taskA", False): 15.0,   # principle helps
            ("taskB", True): 20.0, ("taskB", False): 20.0,   # neutral
        })
        h = make_hypothesis()
        tests = asyncio.run(v.unseen_transfer(h, ["taskA", "taskB"], solver, "minimize"))
        self.assertEqual(tests[0].improvement, "improved")
        self.assertEqual(tests[1].improvement, "neutral")
        self.assertAlmostEqual(v.transferability(tests), 0.5)

    def test_maximize_sense(self):
        v = PatternValidator()
        solver = make_transfer_solver({("t", True): 30.0, ("t", False): 25.0})
        tests = asyncio.run(v.unseen_transfer(make_hypothesis(), ["t"], solver, "maximize"))
        self.assertEqual(tests[0].improvement, "improved")

    def test_missing_objective_neutral(self):
        v = PatternValidator()
        solver = make_transfer_solver({("t", True): None, ("t", False): 5.0})
        tests = asyncio.run(v.unseen_transfer(make_hypothesis(), ["t"], solver))
        self.assertIsNone(tests[0].improvement)


class NoveltyTest(unittest.TestCase):
    def test_novel_when_not_in_any_source(self):
        v = PatternValidator()
        h = make_hypothesis()
        sources = ["allocate stock to warehouse", "assign jobs to machines"]
        self.assertEqual(v.novelty(h, sources), 1.0)

    def test_not_novel_when_restates_source(self):
        v = PatternValidator()
        h = make_hypothesis()
        h.statement = "allocate stock to warehouse"
        sources = ["allocate stock to warehouse carefully"]
        self.assertEqual(v.novelty(h, sources), 0.0)


class ScoringTest(unittest.TestCase):
    def test_score_formula(self):
        w = ScoringWeights(alpha=1, beta=1, gamma=1, delta=1, lam=1, mu=1)
        v = PatternValidator(weights=w)
        h = make_hypothesis(complexity=0.2)
        refutation = RefutationResult(
            hypothesis_id="hyp_v",
            attempts=[RefutationAttempt(condition="c", executed=True, principle_failed=False)],
        )
        solver = make_transfer_solver({("t", True): 8.0, ("t", False): 10.0})
        tests = asyncio.run(v.unseen_transfer(h, ["t"], solver))
        scoring = v.score(h, ["e1", "e2", "e3"], tests, refutation, ["src a", "src b"])
        # C=1.0, T=1.0, V=1.0, N=1.0, K=0.2, X=0 -> total 3.8
        self.assertAlmostEqual(scoring.coverage, 1.0)
        self.assertAlmostEqual(scoring.transferability, 1.0)
        self.assertAlmostEqual(scoring.novelty, 1.0)
        self.assertAlmostEqual(scoring.total, 3.8)

    def test_counterexample_penalty_subtracted(self):
        v = PatternValidator()
        h = make_hypothesis()
        refutation = RefutationResult(
            hypothesis_id="hyp_v",
            attempts=[RefutationAttempt(condition="fixed charge", executed=True, principle_failed=True)],
        )
        from or_experience_bank.core.modeling_schemas import CounterexampleRecord
        refutation.counterexamples.append(CounterexampleRecord(summary="fixed charge"))
        self.assertEqual(v.counterexample_penalty(refutation), 1.0)
        self.assertEqual(v.validation_strength(refutation), 0.0)


class DecideTest(unittest.TestCase):
    def _make_outcome(self, transfer_improved, total_ok=True, refuted=False):
        v = PatternValidator(validation_threshold=0.5)
        h = make_hypothesis()
        solver = make_transfer_solver({("t", True): 8.0, ("t", False): 10.0 if transfer_improved else 8.0})
        tests = asyncio.run(v.unseen_transfer(h, ["t"], solver))
        from or_experience_bank.core.modeling_schemas import PatternValidation
        validation = PatternValidation(source_consistency="3/3", transfer_tests=tests)
        refutation = None
        if refuted:
            refutation = RefutationResult(
                hypothesis_id="hyp_v",
                attempts=[RefutationAttempt(condition="c", executed=True, principle_failed=True)],
            )
        scoring = v.score(h, ["e1", "e2", "e3"], tests, refutation, ["a"])
        if not total_ok:
            scoring.total = 0.1
        return v.decide(h, scoring, validation, refutation)

    def test_validated_when_all_pass(self):
        self.assertEqual(self._make_outcome(transfer_improved=True).status, "validated")

    def test_refuted_when_no_transfer(self):
        # This is the Induction != Summary guard: no unseen transfer -> refuted.
        self.assertEqual(self._make_outcome(transfer_improved=False).status, "refuted")

    def test_refuted_when_below_threshold(self):
        self.assertEqual(self._make_outcome(transfer_improved=True, total_ok=False).status, "refuted")

    def test_refuted_kept_not_deleted(self):
        outcome = self._make_outcome(transfer_improved=False)
        self.assertEqual(outcome.status, "refuted")
        self.assertIsNotNone(outcome.scoring)  # record survives (append-only)


class LLMBackedValidatorTest(unittest.TestCase):
    def test_end_to_end_validate(self):
        solver = make_transfer_solver({("t", True): 9.0, ("t", False): 12.0})
        driver = LLMBackedValidator()
        outcome = asyncio.run(
            driver.validate(make_hypothesis(), ["e1", "e2", "e3"], ["t"], solver, None, ["x"])
        )
        self.assertEqual(outcome.status, "validated")
        self.assertIn("total=", outcome.rationale)


if __name__ == "__main__":
    unittest.main()
