"""r13: the cost observation span matches the prediction's declared cost
span, and a clearly measured wall-clock span IS calibratable.

Two P1 defects this file pins:

3. ``strategy_prediction`` asks the model to predict the cost from the
   decision to the END of the task, but the close-out observed only a
   single attempt. A prediction of 300 tokens (100 failed attempt + 200
   repaired retry) was compared against the FIRST attempt's 100, reporting
   a large over-estimate.
4. Every non-cumulative dimension's ``total`` was forced to ``None``, and
   the evaluation required a non-``None`` total — so a backfilled real
   ``remaining_latency_s`` was "completely measured" yet scored as
   "unobserved". "Cannot be summed" is not "cannot be compared": a single
   explicit span is a comparable observation.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from or_harness.world_model.episode_closeout import (  # noqa: E402
    _aggregate_costs, _aggregated_dim)
from or_harness.core.schema import CostVector  # noqa: E402


class TestWallClockIsComparable(unittest.TestCase):

    def test_single_span_has_a_comparable_value(self):
        dim = _aggregated_dim("remaining_latency_s", [30.0], 1)
        self.assertIsNone(dim["total"], "a span is never reported as a sum")
        self.assertEqual(dim["comparable"]["value"], 30.0)
        self.assertFalse(dim["comparable"]["is_sum"])
        self.assertTrue(dim["complete"])

    def test_distinct_spans_do_not_combine(self):
        dim = _aggregated_dim("remaining_latency_s", [10.0, 20.0], 2)
        self.assertIsNone(dim["comparable"]["value"])
        self.assertIn("never summed", dim["comparable"]["basis"])

    def test_identical_spans_collapse_to_one_value(self):
        dim = _aggregated_dim("latency_s", [5.0, 5.0], 2)
        self.assertEqual(dim["comparable"]["value"], 5.0)

    def test_aggregate_costs_reports_comparable_for_a_backfilled_span(self):
        c = CostVector(remaining_latency_s=30.0,
                       measured={"remaining_latency_s"})
        agg = _aggregate_costs([c])
        self.assertEqual(agg["remaining_latency_s"]["comparable"]["value"],
                         30.0)
        self.assertEqual(agg["remaining_latency_s"]["n_measured"], 1)


from tests.harness.test_episode_closeout import M4Case, _task  # noqa: E402


class TestRemainingCostObservation(M4Case):
    """The report's case: predicted 300 remaining tokens, real spend = a
    100-token failed attempt + a 200-token repaired retry. The observation
    must cover BOTH, not just the first attempt."""

    def test_remaining_cost_covers_retries(self):
        from pathlib import Path

        h = self.h
        # The provider predicts a REMAINING cost of 300 tokens.
        self.provider.payload = {
            "benefit": {"kind": "effective_completion",
                        "metric": "task_result_check_passed",
                        "unit": "boolean", "value": 0.8,
                        "baseline": {"kind": "declared", "value": 0.5}},
            "cost": {"llm_tokens": 300.0},
        }
        task = _task("t1")
        prediction = h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S04"}, "ep1")
        self.assertEqual(prediction.cost.scope, "remaining_to_task_end")
        # A failed first attempt then a repaired retry, both in the same
        # episode. The real spend is amended through the proper channel so
        # each record measures llm_tokens with a trusted source.
        goal = Path(self.home) / "ws_rem"
        goal.mkdir(parents=True, exist_ok=True)
        (goal / "solve.py").write_text(
            "raise RuntimeError('first attempt fails')\n",
            encoding="utf-8")
        first = h.execute(task, "S04", str(goal / "solve.py"), str(goal),
                          solver="highs", episode_id="ep1")
        h.record(first)
        h.bank.update_cost(first.execution_id, source="provider_usage",
                           llm_tokens=100.0)
        second = self.solve(task, strategy="S04", episode_id="ep1",
                            tag="rem2")
        h.bank.update_cost(second.execution_id, source="provider_usage",
                           llm_tokens=200.0)
        h.bind_strategy_outcome(prediction.prediction_id, second.action_id)
        result = h.close_episode("t1", "ep1")
        evaluation = result["evaluations"][0]
        entry = evaluation["cost"]["per_dim"].get("llm_tokens")
        self.assertIsNotNone(
            entry,
            "the remaining-cost prediction must be compared against the "
            "real remaining spend, and 300 == 100 + 200")
        self.assertEqual(entry["predicted"], 300.0)
        self.assertEqual(entry["actual"], 300.0)
        self.assertAlmostEqual(entry["log_ratio"], 0.0, places=5)

    def test_a_backfilled_remaining_latency_is_calibratable(self):
        """The report's case 4: a real ``remaining_latency_s`` backfilled on
        the record must produce a cost error, not 'unobserved'."""
        from pathlib import Path

        h = self.h
        # The model predicts a 30s remaining latency; the real operation
        # takes 600s (modelling, debugging, retries).
        self.provider.payload = {
            "benefit": {"kind": "effective_completion",
                        "metric": "task_result_check_passed",
                        "unit": "boolean", "value": 0.8,
                        "baseline": {"kind": "declared", "value": 0.5}},
            "cost": {"remaining_latency_s": 30.0},
        }
        task = _task("t1")
        prediction = h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S04"}, "ep1")
        record = self.solve(task, strategy="S04", episode_id="ep1",
                            tag="lat")
        # Backfill the REAL remaining span (a trusted measurement).
        h.bank.update_cost(record.execution_id, source="agent_observed",
                           remaining_latency_s=600.0)
        h.bind_strategy_outcome(prediction.prediction_id, record.action_id)
        result = h.close_episode("t1", "ep1")
        evaluation = result["evaluations"][0]
        entry = evaluation["cost"]["per_dim"].get("remaining_latency_s")
        self.assertIsNotNone(
            entry,
            "a clearly measured wall-clock span must participate in "
            "calibration: 'cannot be summed' is not 'cannot be compared'")
        self.assertEqual(entry["predicted"], 30.0)
        self.assertEqual(entry["actual"], 600.0)
        # log(600/30) = log(20) ~= 2.996 (the model predicted ten minutes
        # as half a minute — the error must be visible).
        self.assertAlmostEqual(entry["log_ratio"], 2.9957, places=3)


if __name__ == "__main__":
    unittest.main()
