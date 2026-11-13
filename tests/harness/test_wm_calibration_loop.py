"""End-to-end tests for the world-model calibration loop (M5).

These tests prove the REAL LINK, not the mere existence of fields:

- closed-episode history (successes AND failures, cross-cell and
  cross-strategy-name) really appears in the NEXT prediction's provider
  request, and the CURRENT episode's unclosed result does NOT leak in;
- the deterministic rule reminders derived from the measured statistics are
  consumed by the next prediction;
- the H+ follow-up state reaches the next context, with ``pending``
  distinguished from a verified effect;
- the observation rules (wm-obs/3) keep the solver's quality figure and
  carry the task check as a SEPARATE fact;
- a late check / repeated publication never double-counts.

They are SIMULATION tests: they prove the flow and the plumbing, never that
prediction accuracy improved.
"""

import json
import os
import unittest

from or_harness.api import ORHarness

from tests.harness.helpers import HarnessTestCase
from tests.harness.test_episode_closeout import (
    GOOD_PAYLOAD,
    M4Case,
    StubProvider,
    _task,
)

EMBEDDING_ENV_KEYS = ("OR_EMBEDDING_MODEL", "OR_EMBEDDING_BACKEND",
                      "OR_EMBEDDING_DIM")


def _payload_for(strategy_id, value):
    payload = json.loads(json.dumps(GOOD_PAYLOAD))
    payload["benefit"]["value"] = value
    payload["benefit"]["baseline"] = {"kind": "conditional_stats",
                                      "value": 0.5}
    # An H+ stance so the prediction is archived as an online gain.
    payload["capability_gain"] = {
        "assessment": "expected",
        "claim": f"{strategy_id} could teach a reusable pattern",
        "expected_changes": [{"metric": "normalized_objective_gap",
                              "direction": "increase"}],
    }
    return payload


class MultiPayloadProvider(StubProvider):
    """A provider returning a payload keyed by the candidate's strategy."""

    def __init__(self, payloads):
        super().__init__(payload=GOOD_PAYLOAD)
        self.payloads = payloads

    def predict(self, request, timeout_s=None):
        self.requests.append(request)
        candidate = request.get("candidate") or {}
        sid = candidate.get("strategy_id")
        payload = self.payloads.get(sid, GOOD_PAYLOAD)
        return {"payload": payload, "usage": self.usage,
                "error": None, "latency_s": 0.02}


class CalibrationLoopCase(M4Case):
    """Shared fixtures for the loop tests."""

    def solve(self, *args, **kwargs):
        from pathlib import Path
        task = args[0] if args else kwargs.get("task")
        strategy = kwargs.get("strategy", "S04")
        objective = kwargs.get("objective", 100.0)
        episode_id = kwargs.get("episode_id", "ep1")
        label = kwargs.get("tag") or (
            f"{task['task_id']}_{strategy}_{episode_id}")
        work = Path(self.home) / f"ws_{label}"
        work.mkdir(parents=True, exist_ok=True)
        script = work / "solve.py"
        script.write_text(
            "import json\n"
            "with open('result.json', 'w') as fh:\n"
            "    json.dump({'status': 'optimal', 'objective_value': "
            f"{objective}, 'objective_bound': {objective}"
            ", 'runtime_seconds': 0.01}, fh)\n", encoding="utf-8")
        record = self.h.execute(task, strategy, str(script), str(work),
                                solver="highs", episode_id=episode_id)
        self.h.record(record)
        return record


class TestPairedFeedbackReachesTheProvider(CalibrationLoopCase):

    def test_closed_history_appears_in_the_next_request(self):
        """A CLOSED episode's pair must reach the NEXT prediction request."""
        task = _task("t1")
        prediction = self.h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S04"}, "ep1", benefit_convention={'kind': 'solution_quality', 'metric': 'normalized_objective_gap', 'unit': '1-gap'})
        record = self.solve(task, strategy="S04")
        self.h.bind_strategy_outcome(prediction.prediction_id,
                                     record.action_id)
        self.h.close_episode("t1", "ep1")

        # A NEW decision: episode 2 builds a context and predicts.
        self.provider.requests.clear()
        self.h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S04"}, "ep2", benefit_convention={'kind': 'solution_quality', 'metric': 'normalized_objective_gap', 'unit': '1-gap'})
        request = self.provider.requests[-1]
        pairs = request["prediction_context"]["prediction_execution_pairs"]
        self.assertEqual(pairs["n_pairs_total"], 1)
        row = pairs["pairs"][0]
        self.assertEqual(row["task_id"], "t1")
        self.assertEqual(row["episode_id"], "ep1")
        # The ORIGINAL prediction and the REAL observation both travel.
        self.assertEqual(row["benefit"]["predicted"], 0.8)
        self.assertEqual(row["benefit"]["observed"], 1.0)
        self.assertEqual(row["benefit"]["signed_error"], 0.2)

    def test_current_episode_result_does_not_leak(self):
        """An OPEN episode's result must NOT appear in its own request."""
        task = _task("t1")
        prediction = self.h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S04"}, "ep1", benefit_convention={'kind': 'solution_quality', 'metric': 'normalized_objective_gap', 'unit': '1-gap'})
        record = self.solve(task, strategy="S04")
        self.h.bind_strategy_outcome(prediction.prediction_id,
                                     record.action_id)
        # Episode 1 is NOT closed yet: a new prediction's request carries no
        # pair from it.
        self.h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S04"}, "ep1", benefit_convention={'kind': 'solution_quality', 'metric': 'normalized_objective_gap', 'unit': '1-gap'})
        request = self.provider.requests[-1]
        pairs = request["prediction_context"]["prediction_execution_pairs"]
        self.assertEqual(pairs["n_pairs_total"], 0)

    def test_failure_is_included_and_cross_cell_survives(self):
        """A FAILED task check and a cross-strategy-name case both appear.

        Nothing filters by cell or strategy name: a failure is exactly the
        material a later prediction needs.
        """
        task_a = _task("t1")
        prediction = self.h.predict_strategy_outcome(
            task_a, {"action_type": "execute_strategy",
                     "strategy_id": "S99"}, "ep1", benefit_convention={'kind': 'solution_quality', 'metric': 'normalized_objective_gap', 'unit': '1-gap'})
        record = self.solve(task_a, strategy="S99", tag="fail")
        # Confirm the answer FAILS the task.
        self.h.check_task_result(record.execution_id,
                                 {"reference_objective": 999.0})
        self.h.bind_strategy_outcome(prediction.prediction_id,
                                     record.action_id)
        self.h.close_episode("t1", "ep1")

        # A different task / strategy name predicts next.
        task_b = _task("t2")
        self.h.predict_strategy_outcome(
            task_b, {"action_type": "execute_strategy",
                     "strategy_id": "S04"}, "ep1", benefit_convention={'kind': 'solution_quality', 'metric': 'normalized_objective_gap', 'unit': '1-gap'})
        request = self.provider.requests[-1]
        pairs = request["prediction_context"]["prediction_execution_pairs"]
        self.assertEqual(pairs["n_pairs_total"], 1)
        row = pairs["pairs"][0]
        # The failure is carried as a separate task-check fact; the quality
        # observation is the solver's own figure (1.0), NOT rewritten to 0.
        self.assertEqual(row["task_check"]["state"], "failed")
        self.assertEqual(row["benefit"]["observed"], 1.0)
        self.assertEqual(row["conditions"]["strategy_id"], "S99")

    def test_unexecuted_candidate_produces_no_pair(self):
        """A predicted-but-unexecuted candidate has NO pair (and no
        fabricated outcome)."""
        task = _task("t1")
        self.h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S04"}, "ep1", benefit_convention={'kind': 'solution_quality', 'metric': 'normalized_objective_gap', 'unit': '1-gap'})
        self.h.close_episode("t1", "ep1")
        self.h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S04"}, "ep2", benefit_convention={'kind': 'solution_quality', 'metric': 'normalized_objective_gap', 'unit': '1-gap'})
        request = self.provider.requests[-1]
        pairs = request["prediction_context"]["prediction_execution_pairs"]
        self.assertEqual(pairs["n_pairs_total"], 0)

    def test_other_models_pairs_are_withheld(self):
        """A cross-strategy-name case is kept, but a DIFFERENT MODEL's pairs
        are withheld — exactly as the calibration summary is filtered."""
        from or_harness.world_model.episode_closeout import (
            _filter_pairs_by_model,
        )
        block = {"pairs": [
            {"model_identity": "model-a@v1", "benefit": {"signed_error": 0.2}},
            {"model_identity": "other@v9", "benefit": {"signed_error": 0.3}},
        ]}
        filtered = _filter_pairs_by_model(block, "model-a@v1")
        self.assertEqual(len(filtered["pairs"]), 1)
        self.assertEqual(filtered["pairs"][0]["model_identity"],
                         "model-a@v1")
        self.assertEqual(filtered["n_pairs_withheld"], 1)
        self.assertIn("different model identity", filtered["filter_note"])

    def test_pair_carries_the_measured_cost_not_only_the_ratio(self):
        """The real per-dimension spend travels beside the error ratio: a
        ratio alone hides the scale."""
        task = _task("t1")
        prediction = self.h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S04"}, "ep1", benefit_convention={'kind': 'solution_quality', 'metric': 'normalized_objective_gap', 'unit': '1-gap'})
        record = self.solve(task, strategy="S04")
        self.h.amend_action_cost(record.action_id, llm_tokens=500.0,
                                 solver_runtime_s=2.0)
        self.h.bind_strategy_outcome(prediction.prediction_id,
                                     record.action_id)
        self.h.close_episode("t1", "ep1")
        self.provider.requests.clear()
        self.h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S04"}, "ep2", benefit_convention={'kind': 'solution_quality', 'metric': 'normalized_objective_gap', 'unit': '1-gap'})
        request = self.provider.requests[-1]
        row = request["prediction_context"][
            "prediction_execution_pairs"]["pairs"][0]
        self.assertIn("cost_measured", row)
        # The prediction claimed solver_runtime_s=3.0; the measured spend
        # travels as its own value beside the ratio.
        self.assertEqual(
            row["cost_measured"]["solver_runtime_s"]["predicted"], 3.0)
        self.assertEqual(
            row["cost_measured"]["solver_runtime_s"]["actual"], 0.01)

    def test_pair_carries_the_actual_risk_event_label(self):
        """A predicted risk event is paired with what actually happened,
        and an unknown label stays unknown."""
        task = _task("t1")
        prediction = self.h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S04"}, "ep1", benefit_convention={'kind': 'solution_quality', 'metric': 'normalized_objective_gap', 'unit': '1-gap'})
        record = self.solve(task, strategy="S04")
        self.h.bind_strategy_outcome(prediction.prediction_id,
                                     record.action_id)
        self.h.close_episode("t1", "ep1")
        self.provider.requests.clear()
        self.h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S04"}, "ep2", benefit_convention={'kind': 'solution_quality', 'metric': 'normalized_objective_gap', 'unit': '1-gap'})
        request = self.provider.requests[-1]
        row = request["prediction_context"][
            "prediction_execution_pairs"]["pairs"][0]
        self.assertIn("risk_actual", row)
        timeout = next(r for r in row["risk_actual"]
                       if r["event"] == "timeout")
        # The attempt finished optimal (no timeout observed).
        self.assertEqual(timeout["label"], "not_occurred")
        # The PREDICTED probability is carried ONCE, in ``risk_predicted``;
        # ``risk_actual`` only POINTS at it (r15) and never restates the
        # number as an occurrence.
        self.assertTrue(timeout["predicted"])
        self.assertEqual(timeout["probability_ref"],
                         "risk_predicted[timeout].probability")
        self.assertNotIn("predicted_probability", timeout)
        predicted = next(e for e in row["risk_predicted"]
                         if e["event"] == "timeout")
        self.assertEqual(predicted["probability"], 0.2)

    def test_pair_carries_the_actual_method_when_reported(self):
        """The method that ACTUALLY ran travels with the pair, separate from
        the plan; an unreported method stays absent."""
        task = _task("t1")
        prediction = self.h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S04",
                   "method": {"name": "planned method", "steps": ["a", "b"]}},
            "ep1", benefit_convention={'kind': 'solution_quality', 'metric': 'normalized_objective_gap', 'unit': '1-gap'})
        # Execute with a script that reports a DIFFERENT performed method.
        from pathlib import Path
        work = Path(self.home) / "ws_actual"
        work.mkdir(parents=True, exist_ok=True)
        script = work / "solve.py"
        script.write_text(
            "import json, os\n"
            "with open('result.json', 'w') as fh:\n"
            "    json.dump({'status': 'optimal', 'objective_value': 100.0,"
            " 'objective_bound': 100.0, 'runtime_seconds': 0.01,"
            " 'method_performed': {'action_id': os.environ['OR_ACTION_ID'],"
            " 'name': 'fallback enumeration', 'steps': ['enumerate']}}, fh)\n",
            encoding="utf-8")
        from or_harness.world_model.prediction import ActionSpec  # noqa
        record = self.h.execute(
            task, "S04", str(script), str(work), solver="highs",
            episode_id="ep1", prediction_id=prediction.prediction_id)
        self.h.record(record)
        self.h.bind_strategy_outcome(prediction.prediction_id,
                                     record.action_id)
        self.h.close_episode("t1", "ep1")
        self.provider.requests.clear()
        self.h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S04"}, "ep2", benefit_convention={'kind': 'solution_quality', 'metric': 'normalized_objective_gap', 'unit': '1-gap'})
        request = self.provider.requests[-1]
        row = request["prediction_context"][
            "prediction_execution_pairs"]["pairs"][0]
        self.assertEqual(row["method_planned"]["name"], "planned method")
        self.assertEqual(row["method_actual"]["name"],
                         "fallback enumeration")
        self.assertIn("method_deviation", row)


class TestErrorRunIsNotAFullScore(M4Case):
    """A failed run's stray gap=0 must never be read as a full-quality
    solution."""

    def test_error_execution_with_zero_gap_is_not_scored(self):
        task = _task("t1")
        prediction = self.h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S04"}, "ep1", benefit_convention={'kind': 'solution_quality', 'metric': 'normalized_objective_gap', 'unit': '1-gap'})
        # A script that FAILS (exit code 1) but leaves a result.json carrying
        # gap=0 — the exact shape that would otherwise score Q=1.0.
        from pathlib import Path
        work = Path(self.home) / "ws_err"
        work.mkdir(parents=True, exist_ok=True)
        script = work / "solve.py"
        script.write_text(
            "import json, sys\n"
            "with open('result.json', 'w') as fh:\n"
            "    json.dump({'status': 'optimal', 'objective_value': 100.0,"
            " 'objective_bound': 100.0, 'mip_gap': 0.0}, fh)\n"
            "sys.exit(1)\n", encoding="utf-8")
        record = self.h.execute(
            task, "S04", str(script), str(work), solver="highs",
            episode_id="ep1", prediction_id=prediction.prediction_id)
        self.assertEqual(record.quality.get("status"), "error")
        self.h.record(record)
        self.h.bind_strategy_outcome(prediction.prediction_id,
                                     record.action_id)
        result = self.h.close_episode("t1", "ep1")
        evaluation = result["evaluations"][0]
        # No benefit observation was manufactured from the failed run.
        self.assertNotEqual(evaluation["benefit"].get("observed"), 1.0)
        self.assertIn("eligibility",
                      evaluation.get("benefit", {}) or {"eligibility": None})


class TestRemindersReachTheProvider(CalibrationLoopCase):

    def test_reminders_are_derived_and_consumed(self):
        """Enough closed episodes yield a deterministic reminder that the
        NEXT request carries."""
        from or_harness.world_model.episode_closeout import (
            _prediction_reminders,
        )
        # Build several closed episodes so the benefit group is measured.
        for index in range(5):
            task = _task(f"t{index}")
            prediction = self.h.predict_strategy_outcome(
                task, {"action_type": "execute_strategy",
                       "strategy_id": "S04"}, "ep1", benefit_convention={'kind': 'solution_quality', 'metric': 'normalized_objective_gap', 'unit': '1-gap'})
            record = self.solve(task, strategy="S04",
                                episode_id="ep1", tag=f"rem{index}")
            self.h.bind_strategy_outcome(prediction.prediction_id,
                                         record.action_id)
            self.h.close_episode(f"t{index}", "ep1")
        summary = self.h.calibration_summary(min_samples=5)
        group = list(summary["groups"].values())[0]
        self.assertGreaterEqual(group["n_distinct_episodes"], 5)
        reminders = summary["reminders"]
        benefit_reminders = [r for r in reminders
                             if r["field"] == "benefit"]
        self.assertFalse(benefit_reminders, "G residuals must remain diagnostic")
        self.assertTrue(any(g.get("mean_benefit_signed_error") is not None
                            for g in summary["groups"].values()))
        self.assertFalse(any(r["field"].startswith("cost.") for r in reminders))
        self.assertTrue(any(g.get("mean_cost_log_ratio") for g in summary["groups"].values()))

        # The NEXT request carries the reminders.
        self.provider.requests.clear()
        self.h.predict_strategy_outcome(
            _task("next"), {"action_type": "execute_strategy",
                            "strategy_id": "S04"}, "ep1", benefit_convention={'kind': 'solution_quality', 'metric': 'normalized_objective_gap', 'unit': '1-gap'})
        request = self.provider.requests[-1]
        sent = request["prediction_context"]["prediction_reminders"]
        self.assertFalse(any(r["field"].startswith("cost.") for r in sent))

    def test_below_threshold_yields_no_reminder(self):
        """A single episode is NOT a lesson: no reminder below the
        threshold."""
        task = _task("t1")
        prediction = self.h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S04"}, "ep1", benefit_convention={'kind': 'solution_quality', 'metric': 'normalized_objective_gap', 'unit': '1-gap'})
        record = self.solve(task, strategy="S04")
        self.h.bind_strategy_outcome(prediction.prediction_id,
                                     record.action_id)
        self.h.close_episode("t1", "ep1")
        summary = self.h.calibration_summary(min_samples=5)
        self.assertEqual(summary["reminders"], [])


class TestHplusFeedbackReachesTheContext(CalibrationLoopCase):

    def _hplus_payload(self):
        payload = json.loads(json.dumps(GOOD_PAYLOAD))
        payload["capability_gain"] = {
            "assessment": "expected",
            "claim": "this strategy could teach a reusable pattern",
            "expected_changes": [{"metric": "normalized_objective_gap",
                                  "direction": "increase"}],
        }
        return payload

    def test_pending_hplus_is_not_a_settled_training_prior(self):
        """Unexecuted forecasts remain diagnostic, outside settled feedback."""
        task = _task("t1")
        provider = StubProvider(payload=self._hplus_payload())
        h = ORHarness(home=self.home, world_model=provider,
                      embedding=self.backend)
        self.addCleanup(h.close)
        h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S04"}, "ep1", benefit_convention={'kind': 'solution_quality', 'metric': 'normalized_objective_gap', 'unit': '1-gap'})
        provider.requests.clear()
        h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S04"}, "ep2", benefit_convention={'kind': 'solution_quality', 'metric': 'normalized_objective_gap', 'unit': '1-gap'})
        request = provider.requests[-1]
        hplus = request["prediction_context"]["hplus_feedback"]
        self.assertEqual(hplus['settled_examples'],[])
        self.assertGreaterEqual(h.online_capability_gains()['n_pending'],1)

    def test_absent_hplus_is_reported_as_absence(self):
        task = _task("t1")
        # A payload with no capability_gain block.
        provider = StubProvider(payload={"benefit":
                                         GOOD_PAYLOAD["benefit"]})
        h = ORHarness(home=self.home, world_model=provider,
                      embedding=self.backend)
        self.addCleanup(h.close)
        h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S04"}, "ep1", benefit_convention={'kind': 'solution_quality', 'metric': 'normalized_objective_gap', 'unit': '1-gap'})
        provider.requests.clear()
        h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S04"}, "ep2", benefit_convention={'kind': 'solution_quality', 'metric': 'normalized_objective_gap', 'unit': '1-gap'})
        request = provider.requests[-1]
        hplus = request["prediction_context"]["hplus_feedback"]
        self.assertEqual(hplus['settled_examples'],[])
        self.assertEqual(hplus['n_applicable'],0)
        self.assertEqual(h.online_capability_gains()['binary_signal_counts']['missing'],2)


class TestObservationRules(CalibrationLoopCase):

    def test_quality_survives_a_failed_task_check(self):
        """wm-obs/3 (kept in wm-obs/4): the solver's quality figure is
        never rewritten by a task check; the check is a separate fact."""
        task = _task("t1")
        prediction = self.h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S04"}, "ep1", benefit_convention={'kind': 'solution_quality', 'metric': 'normalized_objective_gap', 'unit': '1-gap'})
        record = self.solve(task, strategy="S04")
        self.h.check_task_result(record.execution_id,
                                 {"reference_objective": 999.0})
        self.h.bind_strategy_outcome(prediction.prediction_id,
                                     record.action_id)
        closed = self.h.close_episode("t1", "ep1")
        evaluation = closed["evaluations"][0]
        self.assertEqual(evaluation["benefit"]["observed"], 1.0)
        self.assertEqual(evaluation["benefit"]["task_check"]["state"],
                         "failed")
        self.assertEqual(evaluation["observation_rule_version"],
                         "wm-obs/4")

    def test_interval_kind_is_carried_and_versioned(self):
        """An interval's kind/coverage travel with it, and the group key
        is versioned so mixed rules never pool."""
        payload = json.loads(json.dumps(GOOD_PAYLOAD))
        payload["benefit"]["interval"] = [0.7, 0.9]
        payload["benefit"]["interval_kind"] = "outcome"
        payload["benefit"]["interval_coverage"] = 0.9
        provider = StubProvider(payload=payload)
        h = ORHarness(home=self.home, world_model=provider,
                      embedding=self.backend)
        self.addCleanup(h.close)
        task = _task("t1")
        prediction = h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S04"}, "ep1", benefit_convention={'kind': 'solution_quality', 'metric': 'normalized_objective_gap', 'unit': '1-gap'})
        self.assertEqual(prediction.benefit.interval_kind, "outcome")
        self.assertEqual(prediction.benefit.interval_coverage, 0.9)
        record = self.solve(task, strategy="S04")
        h.bind_strategy_outcome(prediction.prediction_id, record.action_id)
        closed = h.close_episode("t1", "ep1")
        evaluation = closed["evaluations"][0]
        self.assertEqual(evaluation["interval"]["predicted_interval"],
                         [0.7, 0.9])
        self.assertIn("interval_by_kind", list(
            h.calibration_summary()["groups"].values())[0])

    def test_completion_interval_without_kind_is_flagged(self):
        """A success-probability interval with no kind is reported, not
        silently accepted as a single-label interval."""
        payload = json.loads(json.dumps(GOOD_PAYLOAD))
        payload["benefit"] = {
            "kind": "effective_completion",
            "metric": "task_result_check_passed",
            "value": 0.8, "interval": [0.7, 0.9],
            "baseline": {"kind": "declared", "value": 0.5}}
        provider = StubProvider(payload=payload)
        h = ORHarness(home=self.home, world_model=provider,
                      embedding=self.backend)
        self.addCleanup(h.close)
        prediction = h.predict_strategy_outcome(
            _task("t1"), {"action_type": "execute_strategy",
                          "strategy_id": "S04"}, "ep1", benefit_convention={'kind': 'solution_quality', 'metric': 'normalized_objective_gap', 'unit': '1-gap'})
        self.assertTrue(any("interval_kind" in p
                            for p in prediction.notes)
                        or prediction.status == "invalid",
                        "a completion interval without a kind must be "
                        "flagged, not silently accepted")


class TestNoDoubleCounting(CalibrationLoopCase):

    def test_repeated_close_and_late_check_do_not_double_count(self):
        task = _task("t1")
        prediction = self.h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S04"}, "ep1", benefit_convention={'kind': 'solution_quality', 'metric': 'normalized_objective_gap', 'unit': '1-gap'})
        record = self.solve(task, strategy="S04")
        self.h.bind_strategy_outcome(prediction.prediction_id,
                                     record.action_id)
        self.h.close_episode("t1", "ep1")
        first = self.h.calibration_summary()
        self.assertEqual(first["n_evaluated"], 1)
        # A second close is idempotent.
        self.h.close_episode("t1", "ep1")
        second = self.h.calibration_summary()
        self.assertEqual(second["n_evaluated"], 1)
        # A late check republishes but does not add a sample.
        self.h.check_task_result(record.execution_id,
                                 {"reference_objective": 999.0})
        third = self.h.calibration_summary()
        self.assertEqual(third["n_evaluated"], 1)
        group = list(third["groups"].values())[0]
        self.assertEqual(group["n_samples"], 1)


from tests.harness.g_fixtures import GCase, SPLIT, COUPLED


class TestFaithfulGObservations(GCase):
    def test_real_execute_check_close_table(self):
        cases = [
            ("pass", COUPLED, False, False, "faithful", "unknown", "original_task", True, 1.0),
            ("method_omission", SPLIT, True, False, "faithful", "candidate_method", "original_task", True, 0.0),
            ("code_omission", COUPLED, True, False, "deviated", "implementation", "original_task", True, None),
            ("crash", COUPLED, False, True, "unknown", "implementation", "original_task", True, None),
            ("partial_check", COUPLED, False, False, "faithful", "unknown", "partial", True, None),
            ("no_audit", COUPLED, True, False, "faithful", "candidate_method", "original_task", False, None),
            ("cause_unknown", SPLIT, True, False, "faithful", "unknown", "original_task", True, None),
        ]
        for ep, method, split, crash, fidelity, failure, coverage, evidence, expected in cases:
            with self.subTest(ep=ep):
                p = self.predict_g(method, ep)
                r = self.run_g(p, split=split, crash=crash)
                self.check_g(r, fidelity=fidelity, failure=failure, coverage=coverage, evidence=evidence)
                b = self.close_g(ep)["evaluations"][0]["benefit"]
                self.assertEqual(b.get("observed"), expected)
                if expected is None:
                    self.assertEqual(b["label_eligibility"], "unknown")
                    self.assertTrue(b["reason"])
                else:
                    self.assertEqual(b["label_eligibility"], "attributable")
                    self.assertTrue(b["evidence_refs"])

    def test_other_method_pass_is_not_borrowed_by_original(self):
        p = self.predict_g(SPLIT)
        r = self.run_g(p, actual=COUPLED)
        self.check_g(r, fidelity="deviated")
        self.assertIsNone(self.close_g()["evaluations"][0]["benefit"].get("observed"))
        actual = self.predict_g(COUPLED, "ep2")
        r2 = self.run_g(actual)
        self.check_g(r2)
        self.assertEqual(self.close_g("ep2")["evaluations"][0]["benefit"]["observed"], 1.0)

    def test_code_repair_can_be_a_later_positive_for_the_same_method(self):
        first = self.predict_g()
        bad = self.run_g(first, split=True)
        self.check_g(bad, fidelity="deviated", failure="implementation")
        second = self.predict_g()
        good = self.run_g(second)
        self.check_g(good)
        outcomes = {e["prediction_id"]: e["benefit"] for e in self.close_g()["evaluations"]}
        self.assertIsNone(outcomes[first.prediction_id].get("observed"))
        self.assertEqual(outcomes[second.prediction_id]["observed"], 1.0)

    def test_audit_can_establish_fidelity_with_reworded_receipt(self):
        p = self.predict_g()
        r = self.run_g(p, actual={"name": "coupled enumeration", "steps": ["Enumerate a common BE/CFC factory."]})
        self.check_g(r)
        self.assertEqual(self.close_g()["evaluations"][0]["benefit"]["observed"], 1.0)

    def test_window_terminal_unknown_does_not_reuse_an_earlier_pass(self):
        from or_harness.world_model.contracts import CandidateRef
        candidate = CandidateRef(action_type="execute_strategy", strategy_id="S04",
            task_id=self.task["task_id"], episode_id="ep1", method=COUPLED,
            scope="strategy_window", window_id="win::factory::ep1::S04")
        p = self.h.predict_strategy_outcome(self.task, candidate, "ep1")
        first = self.run_g(p)
        self.check_g(first)
        # Another in-scope attempt has no audited verdict. Its result is the
        # terminal result of the window, so the old pass cannot decide G.
        p2 = self.predict_g()
        self.run_g(p2, crash=True)
        outcomes = {e["prediction_id"]:e["benefit"] for e in self.close_g()["evaluations"]}
        self.assertIsNone(outcomes[p.prediction_id].get("observed"))

    def test_single_executed_method_is_a_fact_without_counterfactual(self):
        p = self.predict_g()
        r = self.run_g(p)
        self.check_g(r)
        self.close_g()
        self.predict_g(episode="ep2")
        pairs = self.provider.requests[-1]["prediction_context"]["prediction_execution_pairs"]
        self.assertEqual(pairs["pairs"][0]["benefit"]["observed"], 1.0)
        self.assertFalse(pairs["same_task_comparisons"])

    def test_same_task_different_methods_reach_provider_with_sources(self):
        for ep, method, split in [("ep1", SPLIT, True), ("ep2", COUPLED, False)]:
            p = self.predict_g(method, ep)
            r = self.run_g(p, split=split)
            self.check_g(r, failure="candidate_method" if split else "unknown")
            self.close_g(ep)
        # A newer unknown attempt may occupy a displayed row. The actual
        # comparable partner remains a sourced side-by-side fact.
        p3 = self.predict_g(episode="ep3")
        r3 = self.run_g(p3, split=True)
        self.check_g(r3, fidelity="deviated", failure="implementation")
        self.close_g("ep3")
        self.predict_g(episode="ep4")
        comparison = self.provider.requests[-1]["prediction_context"][
            "prediction_execution_pairs"]["same_task_comparisons"][0]
        self.assertEqual({comparison["left"]["observed"], comparison["right"]["observed"]}, {0.0, 1.0})
        self.assertTrue(comparison["left"]["evidence_refs"])
        self.assertTrue(comparison["right"]["evaluation_id"])

    def test_two_passes_tie_and_different_acceptance_is_not_paired(self):
        for ep, method in [("ep1", SPLIT), ("ep2", COUPLED)]:
            p = self.predict_g(method, ep)
            r = self.run_g(p)
            self.check_g(r)
            self.close_g(ep)
        self.predict_g(episode="ep3")
        block = self.provider.requests[-1]["prediction_context"]["prediction_execution_pairs"]
        self.assertEqual(block["same_task_comparisons"][0]["acceptance_comparison"], "tie")
        from or_harness.world_model.g_signal import same_task_comparisons
        block["pairs"][0]["benefit"]["acceptance_id"] = "different-checker/v2"
        self.assertFalse(same_task_comparisons(block["pairs"]))

    def test_historical_forecast_keeps_error_currency_but_audited_fact_is_recalled(self):
        self.provider.payload = json.loads(json.dumps(GOOD_PAYLOAD))
        old = self.predict_g(SPLIT, benefit_convention={"kind": "solution_quality",
            "metric": "normalized_objective_gap", "unit": "1-gap"})
        r = self.run_g(old, split=True)
        self.check_g(r, failure="candidate_method")
        benefit = self.close_g()["evaluations"][0]["benefit"]
        self.assertEqual(benefit["metric"], "normalized_objective_gap")
        self.assertEqual(benefit["observed"], 1.0)
        self.assertAlmostEqual(benefit["abs_error"], 0.2)
        from tests.harness.g_fixtures import G_PAYLOAD
        self.provider.payload = json.loads(json.dumps(G_PAYLOAD))
        new = self.predict_g(episode="ep2")
        r2 = self.run_g(new)
        self.check_g(r2)
        self.close_g("ep2")
        self.predict_g(episode="ep3")
        pairs = self.provider.requests[-1]["prediction_context"]["prediction_execution_pairs"]
        historical = next(row for row in pairs["pairs"] if row["prediction_id"] == old.prediction_id)
        self.assertEqual(historical["benefit"]["observed"], 1.0)
        self.assertEqual(historical["g_observed_fact"]["observed"], 0.0)
        self.assertEqual(len(pairs["same_task_comparisons"]), 1)

    def test_legacy_no_result_completion_rule_remains_legacy(self):
        self.provider.payload["benefit"].update(metric="task_result_check_passed", unit="boolean")
        self.provider.payload["benefit"].pop("observation_rule_version")
        p = self.predict_g(benefit_convention={"kind": "effective_completion",
            "metric": "task_result_check_passed", "unit": "boolean"})
        self.run_g(p, crash=True)
        self.assertEqual(self.close_g()["evaluations"][0]["benefit"]["observed"], 0.0)

    def test_unresolved_evidence_references_leave_g_unknown(self):
        p = self.predict_g()
        r = self.run_g(p)
        self.check_g(r)
        check = {"semantic_probe": [{"path": "execution_features.solution_variables.CFC", "equals": 1}],
            "g_attribution": {"method_fidelity": "faithful", "failure_source": "unknown",
                "acceptance_scope": "original_task", "acceptance_id": "factory-coupling/v1",
                "source": "auditor", "assessment": "checked the code and answer",
                "evidence_refs": ["artifacts.not-present"]}}
        self.assertEqual(self.h.check_task_result(r.execution_id, check)["state"], "passed")
        b = self.close_g()["evaluations"][0]["benefit"]
        self.assertEqual(b["label_eligibility"], "unknown")
        self.assertNotIn("observed", b)

    def test_timeout_and_insufficient_checker_are_unknown(self):
        self.h.executor.timeout_seconds = 1
        p = self.predict_g()
        r = self.run_g(p, timeout=True)
        self.assertEqual(r.quality["status"], "timeout")
        self.assertNotIn("observed", self.close_g()["evaluations"][0]["benefit"])
        self.h.executor.timeout_seconds = 120
        p2 = self.predict_g(episode="ep2")
        r2 = self.run_g(p2)
        result = self.check_g(r2)
        audit = result["report"]["g_attribution"]
        self.h.check_task_result(r2.execution_id, {"integer": {"variables": ["missing"]},
            "g_attribution": audit})
        self.assertNotIn("observed", self.close_g("ep2")["evaluations"][0]["benefit"])


class TestGReviewReproductions(GCase):
    def next_feedback(self, episode="next"):
        self.predict_g(episode=episode)
        return self.provider.requests[-1]["prediction_context"]

    def test_self_referenced_or_unstated_objective_cannot_supply_g(self):
        for ep, source in [("self", "self_derived"), ("unstated", None)]:
            with self.subTest(source=source):
                p = self.predict_g(SPLIT, ep)
                r = self.run_g(p, split=True, execution_config={"time_limit": 60})
                spec = {"reference_objective": r.quality["objective"]}
                if source:
                    spec["reference_source"] = source
                checked = self.check_g(r, check_spec=spec)
                self.assertEqual(checked["state"], "passed")
                self.assertFalse(checked["report"]["reference_source"]["independent"])
                benefit = self.close_g(ep)["evaluations"][0]["benefit"]
                self.assertEqual(benefit["label_eligibility"], "unknown")
                self.assertIn("reference", benefit["reason"])
        view = self.next_feedback()
        self.assertTrue(all(g["passed"] == 0 for g in view["strategy_outcome_calibration"]["conditional_g"]))

    def test_weak_failure_cannot_borrow_a_semantic_probe_that_never_ran(self):
        p = self.predict_g(SPLIT)
        r = self.run_g(p, split=True, execution_config={"time_limit": 60})
        checked = self.check_g(r, failure="candidate_method", check_spec={
            "reference_objective": 999, "reference_source": "self_derived",
            "semantic_probe": {"path": "execution_features.solution_variables.CFC", "equals": 1}})
        self.assertEqual(checked["state"], "failed")
        self.assertFalse(any(c["check"] == "semantic_probe" for c in checked["report"]["checks"]))
        self.assertEqual(self.close_g()["evaluations"][0]["benefit"]["label_eligibility"], "unknown")
        self.assertFalse(self.next_feedback()["prediction_execution_pairs"]["same_task_comparisons"])

    def test_different_actual_time_limits_remain_individual_cases(self):
        for ep, method, limit, split in [("short", SPLIT, 15, True), ("long", COUPLED, 180, False)]:
            p = self.predict_g(method, ep)
            self.assertNotIn("time_limit", p.candidate.config)
            r = self.run_g(p, split=split, execution_config={"time_limit": limit})
            self.assertEqual(r.execution_features["execution_config"]["values"]["time_limit"], limit)
            self.check_g(r, failure="candidate_method" if split else "unknown")
            self.close_g(ep)
        pairs = self.next_feedback()["prediction_execution_pairs"]
        self.assertEqual(len(pairs["pairs"]), 2)
        self.assertFalse(pairs["same_task_comparisons"])

    def test_same_acceptance_id_cannot_hide_changed_probe_threshold(self):
        for ep, method, expected in [("a", SPLIT, 1), ("b", COUPLED, 2)]:
            p = self.predict_g(method, ep)
            r = self.run_g(p, execution_config={"time_limit": 60})
            self.check_g(r, failure="unknown" if expected == 1 else "candidate_method",
                check_spec={"semantic_probe": {"path": "execution_features.solution_variables.CFC", "equals": expected}})
            self.close_g(ep)
        pairs = self.next_feedback()["prediction_execution_pairs"]
        self.assertEqual(len(pairs["pairs"]), 2)
        self.assertFalse(pairs["same_task_comparisons"])

    def test_equivalent_rules_pair_even_when_failure_returns_early(self):
        fingerprints = []
        spec = {"checker_version": "factory-check/1", "semantic_probe": [
            {"path": "execution_features.solution_variables.CFC", "equals": 1},
            {"path": "execution_features.solution_variables.BE", "equals": 1}]}
        for ep, method, split in [("fail", SPLIT, True), ("pass", COUPLED, False)]:
            p = self.predict_g(method, ep)
            r = self.run_g(p, split=split, execution_config={"time_limit": 60})
            report = self.check_g(r, failure="candidate_method" if split else "unknown", check_spec=spec)["report"]
            fingerprints.append(report.get("acceptance_rule", {}).get("fingerprint"))
            self.close_g(ep)
        self.assertTrue(fingerprints[0])
        self.assertEqual(fingerprints[0], fingerprints[1])
        comparison = self.next_feedback()["prediction_execution_pairs"]["same_task_comparisons"][0]
        self.assertEqual({comparison["left"]["observed"], comparison["right"]["observed"]}, {0, 1})

    def test_cross_family_statistics_stay_in_audit_not_current_request(self):
        self.task["family"] = "scheduling"
        p = self.predict_g()
        r = self.run_g(p, execution_config={"time_limit": 60})
        self.check_g(r)
        self.close_g()
        self.assertEqual(self.h.calibration_summary()["conditional_g"][0]["conditions"]["family"], "scheduling")
        self.task["family"] = "assignment"
        self.task["task_id"] = "another"
        self.assertFalse(self.next_feedback()["strategy_outcome_calibration"]["conditional_g"])

    def test_three_tokens_do_not_split_identical_executed_conditions(self):
        self.provider.usage = {"prompt_tokens": 2, "completion_tokens": 1}
        self.h.declare_budget(self.task["task_id"], {"llm_tokens": 1000}, "ep1")
        first = self.predict_g()
        second = self.predict_g()
        for p, limit in ((first, 60), (second, 60.0)):
            r = self.run_g(p, execution_config={"time_limit": limit})
            self.check_g(r)
        self.close_g()
        groups = self.h.calibration_summary()["conditional_g"]
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["passed"], 2)
        self.h.declare_budget(self.task["task_id"], {"llm_tokens": 1000}, "next")
        self.assertEqual(self.next_feedback()["strategy_outcome_calibration"]["conditional_g"][0]["passed"], 2)

    def test_omitting_entire_benefit_still_observes_execution(self):
        self.provider.payload.pop("benefit")
        p = self.predict_g()
        self.assertIsNone(p.benefit)
        r = self.run_g(p, execution_config={"time_limit": 60})
        self.check_g(r)
        benefit = self.close_g()["evaluations"][0]["benefit"]
        self.assertEqual(benefit["eligibility"], "not_predicted")
        view = self.next_feedback()
        groups = view["strategy_outcome_calibration"]["conditional_g"]
        self.assertEqual(len(groups), 1)
        self.assertEqual((groups[0]["passed"], groups[0]["n_scored"]), (1, 0))
        self.assertEqual(groups[0]["probability_bins"], [])
        self.assertEqual(view["prediction_execution_pairs"]["pairs"][0]["g_observed_fact"]["observed"], 1)

    def test_missing_actual_limit_keeps_labelled_cases_unpaired(self):
        for ep, method, config in [("a", SPLIT, {}), ("b", COUPLED, {}),
                ("c", SPLIT, {"time_limit": None}), ("d", COUPLED, {"time_limit": None})]:
            p = self.predict_g(method, ep)
            r = self.run_g(p, execution_config=config)
            self.check_g(r)
            self.close_g(ep)
        pairs = self.next_feedback()["prediction_execution_pairs"]
        self.assertEqual(pairs["n_pairs_total"], 4)
        self.assertTrue(all(row["benefit"]["observed"] == 1 for row in pairs["pairs"]))
        self.assertFalse(pairs["same_task_comparisons"])

    def test_independent_reference_can_support_covered_objective_acceptance(self):
        method = {"name": "constant objective", "steps": ["Return objective 100."]}
        for ep, expected, label in [("equal", 100, 1), ("different", 999, 0)]:
            self.task["description"] = f"Return the specified objective {expected}; exact equality is the entire acceptance rule."
            p = self.predict_g(method, ep)
            r = self.run_g(p)
            self.check_g(r, failure="unknown" if label else "candidate_method", check_spec={
                "reference_objective": expected, "tolerance": 0, "reference_source": "independent",
                "reference_version": "objective-fixture/1"})
            self.assertEqual(self.close_g(ep)["evaluations"][0]["benefit"]["observed"], label)

    def test_condition_selection_precedes_size_trimming_and_keeps_audit(self):
        from unittest.mock import patch
        from or_harness.world_model import context as context_module
        for family in ("scheduling", "assignment"):
            self.task["family"] = family
            self.task["task_id"] = family
            p = self.predict_g()
            r = self.run_g(p)
            self.check_g(r)
            self.close_g()
        audit = self.h.calibration_summary()["conditional_g"]
        self.assertEqual(len(audit), 2)
        original = context_module._bound_request
        seen = []

        def bounded(view):
            groups = view["strategy_outcome_calibration"]["conditional_g"]
            self.assertEqual([g["conditions"]["family"] for g in groups], ["assignment"])
            seen.append(True)
            return original(view)

        with patch.object(context_module, "_bound_request", side_effect=bounded):
            view = self.next_feedback()
        self.assertTrue(seen)
        selected = view["strategy_outcome_calibration"]
        self.assertEqual(selected["conditional_g_selection"]["n_available"], 2)
        self.assertIn("tc", selected["conditional_g"][0]["applicability"]["unknown_fields"])
        self.assertIn("sparse", selected["conditional_g"][0]["support"])
        self.assertEqual(self.h.calibration_summary()["conditional_g"], audit)
        self.task["task_id"] = "low_coupling"
        self.task["annotations"]["coupling"]["resource_coupling"] = 0.1
        self.assertFalse(self.next_feedback("other")["strategy_outcome_calibration"]["conditional_g"])

    def test_checker_version_and_tolerance_are_part_of_rule_identity(self):
        for ep, method, tolerance, version in [
                ("a", SPLIT, 0, "v1"), ("b", COUPLED, 1, "v1"), ("c", COUPLED, 0, "v2")]:
            p = self.predict_g(method, ep)
            r = self.run_g(p)
            self.check_g(r, check_spec={"reference_objective": 100, "reference_source": "independent",
                "tolerance": tolerance, "checker_version": version,
                "semantic_probe": {"path": "execution_features.solution_variables.CFC", "equals": 1}})
            self.close_g(ep)
        self.assertFalse(self.next_feedback()["prediction_execution_pairs"]["same_task_comparisons"])


if __name__ == "__main__":
    unittest.main()
