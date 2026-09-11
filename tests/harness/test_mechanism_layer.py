"""Tests for cost prediction loop, failure classification persistence, and
the reflow stub. (Mechanism feature tests removed in P1 — the mechanism
machinery was deleted as redundant with coupling features.)"""
import unittest

from helpers import HarnessTestCase

from or_harness.core.schema import CostVector, FailureRecord
from or_harness.strategy.experience_bank import ExperienceBank
from or_harness.strategy.induction import InductionEngine, PatternReflowEngine
from or_harness.strategy.stats import ConditionalStats
from or_harness.strategy.strategic_bank import StrategicBank
from or_harness.api import ORHarness


class TestCostPredictionLoop(HarnessTestCase):
    def setUp(self):
        super().setUp()
        self.h = ORHarness(home=self.home)

    def _entry_with_cost(self, tokens_hat=1500.0):
        self.h.bank.append(self.make_record(
            execution_id="ex_c1", task_id="tc1", strategy_id="S01",
            cost=CostVector(llm_tokens=tokens_hat, tool_calls=2,
                            solver_runtime_s=1.0, retries=0, latency_s=1.0)))
        self.h.bank.append(self.make_record(
            execution_id="ex_c2", task_id="tc2", strategy_id="S01",
            cost=CostVector(llm_tokens=tokens_hat, tool_calls=2,
                            solver_runtime_s=1.0, retries=0, latency_s=1.0)))
        result = self.h.induce(strategy_id="S01")
        return result["results"][0]["created"]

    def test_cost_hit_recorded(self):
        entry_id = self._entry_with_cost(tokens_hat=1500.0)
        entry = self.h.sbank.get(entry_id)
        self.assertIn("llm_tokens", entry.cost_interval)
        # An execution within [0.5x, 2x] of the prediction.
        rec = self.make_record(task_id="tc3", strategy_id="S01",
                               cost=CostVector(llm_tokens=1800, tool_calls=2,
                                               solver_runtime_s=1.0, retries=0,
                                               latency_s=1.0))
        outcome = self.h.record(rec)
        checks = outcome["prediction_checks"]
        self.assertTrue(checks)
        cost_check = checks[0].get("cost_check")
        self.assertIsNotNone(cost_check)
        self.assertTrue(cost_check["hit"])
        entry = self.h.sbank.get(entry_id)
        self.assertEqual(entry.prediction_track.n_cost_predictions, 1)
        self.assertEqual(entry.prediction_track.n_cost_hits, 1)

    def test_cost_miss_warns_never_demotes(self):
        entry_id = self._entry_with_cost(tokens_hat=1500.0)
        # Three executions far outside the band (10x the prediction).
        for i in range(3):
            rec = self.make_record(task_id=f"tc{i + 10}", strategy_id="S01",
                                   cost=CostVector(llm_tokens=15000,
                                                   tool_calls=2,
                                                   solver_runtime_s=1.0,
                                                   retries=0, latency_s=1.0))
            self.h.record(rec)
        entry = self.h.sbank.get(entry_id)
        # Cost misses accumulated...
        self.assertEqual(entry.prediction_track.n_cost_predictions, 3)
        self.assertEqual(entry.prediction_track.n_cost_hits, 0)
        # ...but the entry is NOT demoted (quality-side semantics untouched).
        self.assertEqual(entry.status, "candidate")
        self.assertEqual(entry.prediction_track.consecutive_misses, 0)
        # And the selector warns about the uncalibrated cost estimate.
        recs = self.h.selector.recall(self.make_profile(problem_id="q"), top=10)
        s01 = next(r for r in recs if r.strategy.strategy_id == "S01")
        self.assertTrue(any("uncalibrated" in w for w in s01.risk_warnings))

    def test_quality_miss_still_demotes(self):
        """The quality loop is untouched by the cost loop."""
        entry_id = self._entry_with_cost(tokens_hat=1500.0)
        for i in range(3):
            rec = self.make_record(task_id=f"tq{i}", strategy_id="S01", gap=0.9,
                                   cost=CostVector(llm_tokens=1500))
            self.h.record(rec)
        entry = self.h.sbank.get(entry_id)
        self.assertEqual(entry.status, "suspect")


class TestFailureClassificationPersistence(HarnessTestCase):
    def test_error_class_persisted_at_record(self):
        h = ORHarness(home=self.home)
        rec = self.make_record(
            execution_id="ex_fc", task_id="tfc", feasible=False, status="error",
            failures=[FailureRecord(1, "security policy: blocked import subprocess")])
        h.record(rec)
        stored = h.bank.get("ex_fc")
        self.assertEqual(stored.failures[0].error_class, "environment")
        h.close()

    def test_model_error_classified_as_model(self):
        h = ORHarness(home=self.home)
        rec = self.make_record(
            execution_id="ex_fm", task_id="tfm", feasible=False, status="error",
            failures=[FailureRecord(1, "ValueError: bad shape")])
        h.record(rec)
        stored = h.bank.get("ex_fm")
        self.assertEqual(stored.failures[0].error_class, "model")
        h.close()


class TestReflowStub(HarnessTestCase):
    def test_stub_is_noop(self):
        bank = ExperienceBank(self.store)
        sbank = StrategicBank(self.store)
        stats = ConditionalStats(bank)
        engine = PatternReflowEngine(stats, sbank)
        self.assertEqual(engine.propose_reflow(["se_anything"]), [])


if __name__ == "__main__":
    unittest.main()
