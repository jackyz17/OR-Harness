"""r12 world-model changes: benefit = original-task completion, complete
remaining cost, risk coverage, compressed feedback and a configurable
context budget.

Each class pins ONE behaviour the r12 plan requires to be verifiable, using
NO real world model and NO new run batch (scriptable providers only):

A. the DEFAULT benefit yardstick is task completion, not the solver gap;
   an optimal solve that fails the task check is NOT an effective
   completion; a recovered intermediate failure is its own fact;
   the benefit-implied risk is not charged twice.
B. ``remaining_latency_s`` is a wall-clock dimension that is never summed;
   ``solver_runtime_s`` is not billed on top of a measured container.
C. a failure the executor could not classify stays UNKNOWN (never forced to
   ``model``) and is not recorded as "did not happen"; calibration reports
   risk coverage.
D. paired feedback keeps the full method steps, does not repeat the Brier
   rule per event, is ordered failures/recent first, and hoists identity;
   the context input budget is deployment-configured and enforced.
E. H+ structure is unchanged (no unified score).
"""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tests.harness.helpers import HarnessTestCase  # noqa: E402

from or_harness.api import ORHarness  # noqa: E402
from or_harness.core.schema import (  # noqa: E402
    COST_DIMENSIONS, NON_CUMULATIVE_DIMENSIONS, CostVector,
    chargeable_dimensions)
from or_harness.world_model.planner import (  # noqa: E402
    DEFAULT_BENEFIT_CONVENTION, PRIMARY_BENEFIT_CONVENTION,
    benefit_implied_risk_events, score_strategy_outcome_predictions,
    PlanLimits)


# ---------------------------------------------------------------------------
# A. benefit = original-task completion
# ---------------------------------------------------------------------------


class TestPrimaryBenefitIsCompletion(unittest.TestCase):

    def test_default_yardstick_is_task_completion(self):
        """The default convention is completion, NOT the solver gap: the
        main benefit of choosing a method is completing the ORIGINAL task."""
        self.assertEqual(DEFAULT_BENEFIT_CONVENTION,
                         PRIMARY_BENEFIT_CONVENTION)
        self.assertEqual(DEFAULT_BENEFIT_CONVENTION,
                         ("effective_completion", "task_result_check_passed"))

    def test_completion_implied_risk_is_task_check_failed(self):
        self.assertEqual(benefit_implied_risk_events("effective_completion"),
                         ("task_check_failed",))
        # Under solver quality a task-check failure is an INDEPENDENT loss.
        self.assertEqual(benefit_implied_risk_events("solution_quality"), ())

    def test_task_check_failed_is_not_charged_twice_under_completion(self):
        """A candidate that predicts completion AND task_check_failed risk
        must not have the same loss priced into both the benefit (via 1-p)
        and the risk term."""
        from or_harness.world_model.contracts import (
            BenefitEstimate, BaselineStatement, ExpectedCost, RiskEvent,
            RiskStatement, StrategyOutcomePrediction, PredictionTrace,
            CandidateRef)

        def _prediction(pid, benefit_kind, benefit_metric, benefit_value,
                        risk_event):
            candidate = CandidateRef(
                task_id="t1", episode_id="ep1", strategy_id="S01",
                action_type="execute_strategy")
            prediction = StrategyOutcomePrediction(
                prediction_id=pid, candidate=candidate)
            prediction.status = "valid"
            prediction.benefit = BenefitEstimate(
                kind=benefit_kind, metric=benefit_metric, unit="boolean",
                value=benefit_value,
                baseline=BaselineStatement(kind="declared", value=0.5))
            prediction.cost = ExpectedCost(expected=CostVector())
            prediction.risk = RiskStatement(events=[
                RiskEvent(event=risk_event, probability=0.3)])
            prediction.trace = PredictionTrace(prediction_kind="strategy")
            return prediction

        limits = PlanLimits(alpha=1.0, beta=0.0, gamma=1.0,
                            benefit_kind="effective_completion",
                            benefit_metric="task_result_check_passed")
        completion = _prediction("p1", "effective_completion",
                                 "task_result_check_passed", 0.8,
                                 "task_check_failed")
        scores = score_strategy_outcome_predictions([completion], limits)
        self.assertEqual(scores[0].risk_effective, 0.0,
                         "a task_check_failed already inside the completion "
                         "benefit is not charged again")
        # Its probability is still REPORTED for diagnosis.
        self.assertEqual(
            [e["event"] for e in scores[0].risk_reported],
            ["task_check_failed"])
        # Under solver quality the same event IS an independent risk.
        quality = _prediction("p2", "solution_quality",
                              "normalized_objective_gap", 0.8,
                              "task_check_failed")
        qlimits = PlanLimits(alpha=1.0, beta=0.0, gamma=1.0,
                             benefit_kind="solution_quality",
                             benefit_metric="normalized_objective_gap")
        qscores = score_strategy_outcome_predictions([quality], qlimits)
        self.assertEqual(qscores[0].risk_effective, 0.3)


from tests.harness.test_task_check import TaskCheckCase, _task  # noqa: E402


class TestOptimalButTaskFailed(TaskCheckCase):
    """solver optimal + task check failed => NOT an effective completion."""

    def test_quality_observation_carries_the_cross_fact(self):
        """The quality number stays the solver's own figure, and an explicit
        cross-fact records that it does NOT represent task completion."""
        h = self.h
        task = _task("t1")
        prediction = h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S01"}, "ep1")
        record = self.solve(task, variables={"x1": 2.5, "x2": 1.5},
                            objective=10750.0, tag="r12optfail")
        h.check_task_result(
            record.execution_id,
            {"integer": {"variables": ["x1", "x2"]}})
        h.bind_strategy_outcome(prediction.prediction_id, record.action_id)
        closed = h.close_episode("t1", "ep1")
        evaluation = closed["evaluations"][0]
        # The quality observation is the solver's own figure (1.0).
        self.assertEqual(evaluation["benefit"]["observed"], 1.0)
        tc = (evaluation["benefit"].get("task_check") or {})
        self.assertEqual(tc.get("state"), "failed")
        # r12: the explicit cross-fact marks it NOT an effective
        # completion, without rewriting the quality number.
        self.assertIs(tc.get("represents_effective_completion"), False)


# ---------------------------------------------------------------------------
# B. remaining cost and no double billing
# ---------------------------------------------------------------------------


class TestRemainingCostDimension(unittest.TestCase):

    def test_remaining_latency_is_a_cost_dimension(self):
        self.assertIn("remaining_latency_s", COST_DIMENSIONS)

    def test_wall_clock_dimensions_are_not_cumulative(self):
        self.assertIn("latency_s", NON_CUMULATIVE_DIMENSIONS)
        self.assertIn("remaining_latency_s", NON_CUMULATIVE_DIMENSIONS)

    def test_component_is_not_charged_on_top_of_its_container(self):
        both = {"solver_runtime_s", "remaining_latency_s"}
        # When the container is measured, the component is dropped.
        self.assertEqual(chargeable_dimensions(both),
                         {"remaining_latency_s"})
        # Without the container the component is the only measurement: kept.
        self.assertEqual(chargeable_dimensions({"solver_runtime_s"}),
                         {"solver_runtime_s"})

    def test_accumulate_never_sums_wall_clock(self):
        from or_harness.core.schema import accumulate_measured_costs
        c1 = CostVector(remaining_latency_s=10.0,
                        measured={"remaining_latency_s"})
        c2 = CostVector(remaining_latency_s=20.0,
                        measured={"remaining_latency_s"})
        total = {d: 0.0 for d in COST_DIMENSIONS}
        n = {d: 0 for d in COST_DIMENSIONS}
        accumulate_measured_costs([c1, c2], total, n)
        self.assertEqual(total["remaining_latency_s"], 0.0,
                         "overlapping remaining spans must never be summed")
        self.assertEqual(n["remaining_latency_s"], 2)


# ---------------------------------------------------------------------------
# C. risk coverage and honest failure classification
# ---------------------------------------------------------------------------


class TestFailureClassificationImports(unittest.TestCase):

    def test_unknown_is_a_valid_outcome(self):
        from or_harness.strategy.triggers import classify_failure

        class _Rec:
            def __init__(self, error):
                self.failures = [type("F", (), {"error": error})()]
                self.quality = {"status": "error"}

        # A stranger error is NOT forced into "model".
        self.assertEqual(classify_failure(_Rec("something odd happened")),
                         "unknown")
        # A Python exception type is an implementation failure.
        self.assertEqual(
            classify_failure(_Rec("FileExistsError: [Errno 17] exists")),
            "model")
        # A missing module is an environment failure.
        self.assertEqual(
            classify_failure(_Rec("ModuleNotFoundError: no module named 'x'")),
            "environment")
        # An unavailable backend is an environment failure.
        self.assertEqual(
            classify_failure(_Rec("solver not supported on this platform")),
            "environment")


# ---------------------------------------------------------------------------
# D. feedback compression and context budget
# ---------------------------------------------------------------------------


class TestMethodStepCompression(unittest.TestCase):

    def test_all_steps_are_kept_when_under_the_cap(self):
        from or_harness.world_model.episode_closeout import (
            _compact_method_steps)
        steps = [f"step {i}" for i in range(1, 7)]
        kept = _compact_method_steps(steps)
        self.assertEqual(len(kept), 6,
                         "a normal method's steps are kept WHOLE — no silent "
                         "'first three only' truncation")
        self.assertIn("step 6", " ".join(kept))

    def test_oversized_method_is_elided_in_the_middle_not_the_tail(self):
        from or_harness.world_model.episode_closeout import (
            _compact_method_steps)
        steps = [f"constraint {i}" for i in range(1, 21)]
        kept = _compact_method_steps(steps, max_steps=6)
        joined = " ".join(kept)
        # The FIRST and LAST steps (setup and final solve/bound) survive.
        self.assertIn("constraint 1", joined)
        self.assertIn("constraint 20", joined)
        self.assertTrue(any("elided" in s for s in kept))


class TestRiskReasonCompression(unittest.TestCase):

    def test_brier_rule_becomes_a_short_code(self):
        from or_harness.world_model.episode_closeout import (
            _risk_label_basis_code, RISK_REASON_LEGEND)
        code = _risk_label_basis_code(
            "the model did not predict this event: it is observed by the "
            "framework and counted in the occurrence rate, but no Brier "
            "score can be computed without a probability")
        self.assertEqual(code, "unpredicted_no_brier")
        # And the legend explains it ONCE.
        self.assertIn(code, RISK_REASON_LEGEND)


class TestContextBudgetIsConfigured(unittest.TestCase):

    def test_budget_comes_from_deployment_config(self):
        from or_harness.world_model import context as ctx
        prior = os.environ.get("OR_HARNESS_INPUT_BUDGET_TOKENS")
        os.environ["OR_HARNESS_INPUT_BUDGET_TOKENS"] = "32000"
        try:
            chars, basis = ctx._input_budget_chars()
        finally:
            if prior is None:
                os.environ.pop("OR_HARNESS_INPUT_BUDGET_TOKENS", None)
            else:
                os.environ["OR_HARNESS_INPUT_BUDGET_TOKENS"] = prior
        self.assertEqual(basis["budget_tokens"], 32000)
        self.assertLess(chars, 32000 * 4)  # output room reserved
        self.assertIn("ESTIMATE", basis["estimator"])
        self.assertIn("tokenizer", basis["estimator"])

    def test_default_budget_is_large_enough_for_a_256k_deployment(self):
        from or_harness.world_model import context as ctx
        for name in ("OR_HARNESS_INPUT_BUDGET_TOKENS",
                     "OR_HARNESS_CHARS_PER_TOKEN"):
            os.environ.pop(name, None)
        chars, basis = ctx._input_budget_chars()
        self.assertGreaterEqual(basis["budget_tokens"], 64000)
        self.assertGreater(chars, 120000,
                           "the default must exceed the old hardcoded "
                           "120k-character bound")
        self.assertLessEqual(basis["budget_tokens"], 128000,
                             "the deployable ceiling is 128k tokens")


if __name__ == "__main__":
    unittest.main()
