"""r23: the budget_exhausted prediction channel is connected end to end.

The r22 audit found the ``budget_exhausted`` risk event could never be
scored (a ``candidate.scope`` was compared with the ledger's EPISODE scope,
which no prediction scope can equal), the label misread a bare ``exceeded``
status (a single per-attempt latency overrun was labelled budget exhaustion),
and a stopped attempt's real spend could vanish from the ledger. This suite
pins the fix, one behaviour per test:

1. attempt- and strategy-window-scope candidates carrying an explicit episode
   budget risk ARE scored once real results exist (no more scope_mismatch);
2. a measured episode overrun with other dimensions missing is ``occurred``;
3. within limits but a relevant episode dimension missing keeps ``unknown``
   (no fabricated Brier);
4. a completed episode, fully measured within limits, is ``not_occurred``;
5. a single latency overrun is NOT episode exhaustion, and a missing
   per-attempt dimension does not block a complete episode verdict;
6. a hard cap with an explicit budget-termination receipt is ``occurred``,
   while an unexplained abort is not dressed up as budget exhaustion;
7. several predictions / cross-task episodes keep attribution and occurrence
   counting honest, and unexecuted candidates get no counterfactual label;
8. a budget that CHANGED after the prediction was frozen is non-comparable
   (the frozen condition is kept, the label withheld);
9. the version constants travel with the evaluation and the summary; and a
   stopped staged attempt's real cost reaches the ledger.

Every test uses a temporary home and the hermetic local embedding backend;
the running experiment database is never touched.
"""
import os
import sys
import unittest
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tests.harness.helpers import HarnessTestCase  # noqa: E402

from or_harness.api import ORHarness  # noqa: E402
from or_harness.core.schema import (  # noqa: E402
    CostVector,
    ExecutionRecord,
    ProblemProfile,
)
from or_harness.strategy.embedding_index import (  # noqa: E402
    LocalHashEmbeddingBackend,
)
from or_harness.world_model.provider import WorldModelProvider  # noqa: E402

EMBEDDING_ENV_KEYS = ("OR_EMBEDDING_BACKEND", "OR_EMBEDDING_BASE_URL",
                      "OR_EMBEDDING_MODEL", "OR_EMBEDDING_API_KEY")

REQ_TEXT = ("A distribution centre must be loaded before the delivery window "
            "opens; demand 100 units may not be deferred.")


def _task(task_id="t1", **coupling):
    values = {"resource_coupling": 0.3, "temporal_coupling": 0.1,
              "route_complexity": 0.2}
    values.update(coupling)
    return {"task_id": task_id, "family": "routing",
            "description": REQ_TEXT,
            "spec": {"n_vars": 100, "n_constraints": 50, "n_int_vars": 100},
            "annotations": {"coupling": {**values, "semantic_coupling": 0.5}}}


def _profile(problem_id="t1", family="routing"):
    return ProblemProfile(
        problem_id=problem_id, family=family,
        scale_features={"n_vars": 100.0, "n_constraints": 50.0,
                        "n_int_vars": 100.0, "density": 0.01},
        semantic_coupling=0.8, resource_coupling=0.3,
        temporal_coupling=0.2, route_complexity=0.8)


PAYLOAD = {
    "benefit": {"kind": "solution_quality",
                "metric": "normalized_objective_gap", "unit": "1-gap",
                "value": 0.8,
                "baseline": {"kind": "conditional_stats", "value": 0.7}},
    "cost": {"llm_tokens": 1200, "solver_runtime_s": 3.0},
    "risk": {"events": [{"event": "budget_exhausted", "probability": 0.5}]},
}


class StubProvider(WorldModelProvider):
    name = "stub-r23"

    def __init__(self, payload=PAYLOAD):
        self.payload = payload
        self.requests = []

    def predict(self, request, timeout_s=None):
        self.requests.append(request)
        return {"payload": self.payload,
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
                "error": None, "latency_s": 0.02}


class R23Case(HarnessTestCase):

    def setUp(self):
        super().setUp()
        saved = {key: os.environ.pop(key, None) for key in EMBEDDING_ENV_KEYS}

        def restore():
            for key, value in saved.items():
                if value is not None:
                    os.environ[key] = value
        self.addCleanup(restore)
        self.backend = LocalHashEmbeddingBackend()
        self.provider = StubProvider()
        self.h = ORHarness(home=self.home, world_model=self.provider,
                           embedding=self.backend)
        self.addCleanup(self.h.close)

    def record_execution(self, task, strategy="S04", episode_id="ep1", *,
                         llm_tokens=5.0, latency_s=None):
        """Begin+end a real execute_strategy action and append its record."""
        task_id = str(task["task_id"])
        begun = self.h.begin_action(
            "execute_strategy", task, episode_id,
            params={"strategy_id": strategy, "solver": "highs"})
        action_id = begun["action_id"]
        measured = set()
        cost = CostVector(measured=measured)
        if llm_tokens is not None:
            cost.llm_tokens = float(llm_tokens)
            measured.add("llm_tokens")
        if latency_s is not None:
            cost.latency_s = float(latency_s)
            measured.add("latency_s")
        record = ExecutionRecord(
            execution_id=f"ex_{uuid.uuid4().hex[:12]}", task_id=task_id,
            strategy_id=strategy, profile_snapshot=_profile(task_id),
            quality={"feasible": True, "objective": 1.0, "gap": 0.0,
                     "status": "optimal"},
            solver={"name": "highs"}, cost=cost)
        self.h.bank.append(record)
        self.h.end_action(action_id, status="completed",
                          outcome={"strategy_id": strategy},
                          linked_execution_id=record.execution_id,
                          rollup="reference")
        return record, action_id

    def predict(self, task, episode="ep1", *, scope="attempt",
                window_id=None, strategy="S04"):
        candidate = {"action_type": "execute_strategy",
                     "strategy_id": strategy}
        if scope != "attempt":
            candidate["scope"] = scope
            candidate["window_id"] = window_id
        return self.h.predict_strategy_outcome(task, candidate, episode)


class TestBudgetChannelScoring(R23Case):
    """1, 2, 3, 4: the channel is connected and the label is episode-scoped."""

    def test_attempt_scope_is_scored_on_a_real_budget_result(self):
        task = _task("t1")
        prediction = self.predict(task)
        record, action_id = self.record_execution(task, llm_tokens=5000)
        self.h.declare_budget("t1", {"llm_tokens": 10}, "ep1")
        self.h.bind_strategy_outcome(prediction.prediction_id, action_id)
        result = self.h.close_episode("t1", "ep1")
        risk = result["evaluations"][0]["risk"]
        scored = {e["event"]: e for e in risk["scored"]}
        self.assertIn("budget_exhausted", scored,
                      "the attempt-scope prediction is scored: the budget "
                      "event has its own range")
        self.assertEqual(scored["budget_exhausted"]["label"], "occurred")
        self.assertNotIn("scope_mismatch",
                         scored["budget_exhausted"]["label_basis"])

    def test_window_scope_is_scored_on_a_real_budget_result(self):
        task = _task("t1")
        prediction = self.predict(task, scope="strategy_window",
                                  window_id="win::t1::ep1::S04")
        record, action_id = self.record_execution(task, llm_tokens=5000)
        self.h.declare_budget("t1", {"llm_tokens": 10}, "ep1")
        self.h.bind_strategy_outcome(prediction.prediction_id, action_id)
        result = self.h.close_episode("t1", "ep1")
        risk = result["evaluations"][0]["risk"]
        scored = {e["event"]: e for e in risk["scored"]}
        self.assertIn("budget_exhausted", scored)
        self.assertEqual(scored["budget_exhausted"]["observation_range"],
                         "episode_to_task_end")

    def test_episode_overrun_with_missing_dimension_is_occurred(self):
        """An episode dimension over its limit is ``occurred`` even when
        ANOTHER declared episode dimension is unmeasured."""
        task = _task("t1")
        prediction = self.predict(task)
        record, action_id = self.record_execution(task, llm_tokens=5000)
        # llm_tokens is measured and over; tool_calls is never measured.
        self.h.declare_budget("t1", {"llm_tokens": 10, "tool_calls": 10},
                              "ep1")
        self.h.bind_strategy_outcome(prediction.prediction_id, action_id)
        result = self.h.close_episode("t1", "ep1")
        risk = result["evaluations"][0]["risk"]
        self.assertEqual(risk["observed_units"]["budget_exhausted"]["label"],
                         "occurred")

    def test_within_limits_with_missing_dimension_stays_unknown(self):
        """Known spend within limits but a declared episode dimension
        unmeasured: ``unknown`` (no fabricated Brier)."""
        task = _task("t1")
        prediction = self.predict(task)
        record, action_id = self.record_execution(task, llm_tokens=5)
        # tool_calls declared but never measured.
        self.h.declare_budget("t1", {"llm_tokens": 10000, "tool_calls": 100},
                              "ep1")
        self.h.bind_strategy_outcome(prediction.prediction_id, action_id)
        result = self.h.close_episode("t1", "ep1")
        risk = result["evaluations"][0]["risk"]
        self.assertEqual(risk["scored"], [])
        unscored = {e["event"]: e for e in risk["unscored"]}
        self.assertIn("tool_calls", unscored["budget_exhausted"]["reason"])
        self.assertIsNone(
            risk["observed_units"]["budget_exhausted"]["label"])

    def test_completed_within_limits_is_not_occurred(self):
        """The episode ended and every declared episode dimension was
        measured within its limit: ``not_occurred``."""
        task = _task("t1")
        prediction = self.predict(task)
        record, action_id = self.record_execution(task, llm_tokens=5)
        self.h.declare_budget("t1", {"llm_tokens": 10000}, "ep1")
        self.h.bind_strategy_outcome(prediction.prediction_id, action_id)
        result = self.h.close_episode("t1", "ep1")
        risk = result["evaluations"][0]["risk"]
        scored = {e["event"]: e for e in risk["scored"]}
        self.assertIn("budget_exhausted", scored)
        self.assertEqual(scored["budget_exhausted"]["label"], "not_occurred")


class TestLabelDiscrimination(R23Case):
    """5, 6: a latency overrun is not budget exhaustion; a cap receipt is."""

    def test_latency_only_overrun_is_not_budget_exhaustion(self):
        task = _task("t1")
        prediction = self.predict(task)
        record, action_id = self.record_execution(
            task, llm_tokens=5, latency_s=999.0)
        # Only a per-attempt latency budget is declared: no episode
        # dimension to exhaust.
        self.h.declare_budget("t1", {"latency_s": 1.0}, "ep1")
        self.h.bind_strategy_outcome(prediction.prediction_id, action_id)
        result = self.h.close_episode("t1", "ep1")
        risk = result["evaluations"][0]["risk"]
        self.assertEqual(risk["scored"], [])
        self.assertIsNone(
            risk["observed_units"]["budget_exhausted"]["label"])
        basis = risk["observed_units"]["budget_exhausted"]["label_basis"]
        self.assertIn("no EPISODE-scoped dimension", basis)

    def test_missing_per_attempt_dimension_does_not_block_episode_verdict(self):
        """A mixed budget where the only unmeasured dimension is the
        per-attempt ``latency_s``: the EPISODE verdict is still complete."""
        task = _task("t1")
        prediction = self.predict(task)
        # llm_tokens measured; latency_s declared but NOT measured.
        record, action_id = self.record_execution(task, llm_tokens=5)
        self.h.declare_budget("t1", {"llm_tokens": 10000, "latency_s": 999},
                              "ep1")
        self.h.bind_strategy_outcome(prediction.prediction_id, action_id)
        result = self.h.close_episode("t1", "ep1")
        risk = result["evaluations"][0]["risk"]
        self.assertEqual(risk["observed_units"]["budget_exhausted"]["label"],
                         "not_occurred",
                         "a missing per-attempt dimension is not an episode "
                         "gap")

    def test_reaching_a_cap_with_receipt_is_occurred(self):
        """A hard cap reached and the next call refused (the host's own
        recorded stop) is a confirmed budget termination."""
        task = _task("t1")
        prediction = self.predict(task)
        # The measured total sits AT the limit (not over it).
        record, action_id = self.record_execution(task, llm_tokens=100)
        self.h.declare_budget("t1", {"llm_tokens": 100}, "ep1")
        self.h.record_session("t1", "ep1",
                              reason="tool_calls reached 100",
                              cancelled=True, cost={"tool_calls": 100.0})
        self.h.bind_strategy_outcome(prediction.prediction_id, action_id)
        result = self.h.close_episode("t1", "ep1",
                                      terminal_state="budget_exhausted")
        risk = result["evaluations"][0]["risk"]
        self.assertEqual(risk["observed_units"]["budget_exhausted"]["label"],
                         "occurred")

    def test_unexplained_abort_is_not_budget_exhaustion(self):
        """An abort with no proactive-cancel receipt is never dressed up as
        budget exhaustion."""
        task = _task("t1")
        prediction = self.predict(task)
        record, action_id = self.record_execution(task, llm_tokens=5)
        self.h.declare_budget("t1", {"llm_tokens": 10000}, "ep1")
        # A NON-cancelled stop (the process exited on its own, no cause).
        self.h.record_session("t1", "ep1", reason="process exited",
                              cancelled=False)
        self.h.bind_strategy_outcome(prediction.prediction_id, action_id)
        result = self.h.close_episode("t1", "ep1", terminal_state="aborted")
        risk = result["evaluations"][0]["risk"]
        self.assertNotEqual(
            risk["observed_units"]["budget_exhausted"]["label"], "occurred")


class TestAttributionAndCounting(R23Case):
    """7: attribution, occurrence counting and counterfactual honesty."""

    def test_occurrence_is_counted_once_per_episode(self):
        task = _task("t1")
        predictions = [self.predict(task) for _ in range(3)]
        record, action_id = self.record_execution(task, llm_tokens=5000)
        self.h.declare_budget("t1", {"llm_tokens": 10}, "ep1")
        for prediction in predictions:
            self.h.bind_strategy_outcome(prediction.prediction_id,
                                         action_id)
        self.h.close_episode("t1", "ep1")
        summary = self.h.calibration_summary(min_samples=1)
        occurrence = summary["occurrence"]["budget_exhausted"]
        self.assertEqual(occurrence["observation_unit"], "episode")
        self.assertEqual(occurrence["n_occurred"], 1,
                         "three predictions over ONE episode count once")

    def test_cross_task_episodes_are_distinct_units(self):
        for index in range(2):
            task = _task(f"task{index}")
            prediction = self.predict(task)
            record, action_id = self.record_execution(task, llm_tokens=5000)
            self.h.declare_budget(f"task{index}", {"llm_tokens": 10}, "ep1")
            self.h.bind_strategy_outcome(prediction.prediction_id,
                                         action_id)
            self.h.close_episode(f"task{index}", "ep1")
        summary = self.h.calibration_summary(min_samples=1)
        occurrence = summary["occurrence"]["budget_exhausted"]
        self.assertEqual(occurrence["n_occurred"], 2,
                         "two tasks' ep1 are two episode units")

    def test_unexecuted_candidate_gets_no_counterfactual_label(self):
        task = _task("t1")
        unchosen = self.predict(task, strategy="S99")
        chosen = self.predict(task, strategy="S04")
        record, action_id = self.record_execution(task, strategy="S04",
                                                  llm_tokens=5000)
        self.h.declare_budget("t1", {"llm_tokens": 10}, "ep1")
        self.h.bind_strategy_outcome(chosen.prediction_id, action_id)
        result = self.h.close_episode("t1", "ep1")
        # The evaluation is keyed by prediction id: the CHOSEN prediction is
        # evaluated; the unchosen candidate has no bound action and no real
        # outcome, so it produced no evaluation at all (no counterfactual
        # label).
        by_prediction = {e["prediction_id"]: e["risk"]
                         for e in result["evaluations"]}
        self.assertIn(chosen.prediction_id, by_prediction)
        self.assertNotIn(unchosen.prediction_id, by_prediction,
                         "an unexecuted candidate gets no evaluation")
        # And a stored evaluation for the unchosen prediction, if any, is
        # not counted as a scored budget sample.
        all_evals = self.h.strategy_prediction_evaluations(task_id="t1")
        for entry in all_evals:
            if entry["prediction_id"] == unchosen.prediction_id:
                self.assertEqual(entry["risk"].get("scored") or [], [])


class TestComparabilityAndVersions(R23Case):
    """8, 9: a changed budget is non-comparable; versions travel."""

    def test_changed_budget_is_non_comparable(self):
        task = _task("t1")
        # Declare a budget BEFORE the prediction so it is FROZEN into the
        # context, then change it after the run.
        self.h.declare_budget("t1", {"llm_tokens": 100}, "ep1")
        prediction = self.predict(task)
        record, action_id = self.record_execution(task, llm_tokens=5000)
        # The declaration CHANGED (100 -> 10). The current verdict answers a
        # different question.
        self.h.declare_budget("t1", {"llm_tokens": 10}, "ep1")
        self.h.bind_strategy_outcome(prediction.prediction_id, action_id)
        result = self.h.close_episode("t1", "ep1")
        risk = result["evaluations"][0]["risk"]
        scored = {e["event"] for e in risk["scored"]}
        self.assertNotIn("budget_exhausted", scored,
                         "a changed budget is not silently re-based")
        unscored = {e["event"]: e for e in risk["unscored"]}
        self.assertIn("CHANGED", unscored["budget_exhausted"]["reason"])
        comp = risk["observed_units"]["budget_exhausted"]["comparability"]
        self.assertFalse(comp["comparable"])
        self.assertEqual(comp["frozen_budget"], {"llm_tokens": 100.0})

    def test_version_constants_travel(self):
        from or_harness.world_model.episode_closeout import (
            CALIBRATION_SUMMARY_VERSION,
            EVENT_VOCABULARY_VERSION,
            OBSERVATION_RULE_VERSION,
        )
        self.assertEqual(EVENT_VOCABULARY_VERSION, "wm-events/3")
        self.assertEqual(OBSERVATION_RULE_VERSION, "wm-obs/4")
        self.assertEqual(CALIBRATION_SUMMARY_VERSION, "wm-calib/3")
        task = _task("t1")
        prediction = self.predict(task)
        record, action_id = self.record_execution(task, llm_tokens=5000)
        self.h.declare_budget("t1", {"llm_tokens": 10}, "ep1")
        self.h.bind_strategy_outcome(prediction.prediction_id, action_id)
        result = self.h.close_episode("t1", "ep1")
        risk = result["evaluations"][0]["risk"]
        self.assertEqual(risk["event_vocabulary_version"], "wm-events/3")
        summary = self.h.calibration_summary(min_samples=1)
        self.assertEqual(summary["calibration_version"], "wm-calib/3")
        self.assertEqual(summary["event_vocabulary_version"], "wm-events/3")


class TestStagedReceipt(R23Case):
    """The stopped staged attempt's real cost reaches the ledger."""

    def test_staged_stop_receipt_backfills_real_cost(self):
        staged = ExecutionRecord(
            execution_id="ex_staged_r23", task_id="T", strategy_id="S",
            profile_snapshot=_profile("T"),
            quality={"feasible": True, "objective": 7.0, "gap": 0.0,
                     "status": "optimal"},
            solver={"name": "highs"})
        self.h.bank.stage_pending(staged)
        # Before the receipt, llm_tokens is unmeasured on the staged fact.
        pending = self.h.bank.get_pending("ex_staged_r23")
        self.assertNotIn("llm_tokens", pending.cost.measured_dims())
        self.h.record_session("T", "e1", reason="tool_calls reached 100",
                              cancelled=True,
                              attempted_execution_id="ex_staged_r23",
                              cost={"llm_tokens": 4321.0})
        after = self.h.bank.get_pending("ex_staged_r23")
        self.assertIn("llm_tokens", after.cost.measured_dims())
        self.assertEqual(after.cost.llm_tokens, 4321.0)
        provenance = after.execution_features["cost_provenance"]["llm_tokens"]
        self.assertEqual(provenance["channel"], "session_stop_receipt")
        # The observed result is NOT rewritten.
        self.assertEqual(after.quality.get("objective"), 7.0)


if __name__ == "__main__":
    unittest.main()
