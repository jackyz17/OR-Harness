"""End-to-end tests for the world-model calibration loop (M5).

These tests prove the REAL LINK, not the mere existence of fields:

- closed-episode history (successes AND failures, cross-cell and
  cross-strategy-name) really appears in the NEXT prediction's provider
  request, and the CURRENT episode's unclosed result does NOT leak in;
- the deterministic rule reminders derived from the measured statistics are
  consumed by the next prediction;
- the H+ follow-up state reaches the next context, with ``pending``
  distinguished from a verified effect;
- the observation rules (wm-obs/2) keep the solver's quality figure and
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
                   "strategy_id": "S04"}, "ep1")
        record = self.solve(task, strategy="S04")
        self.h.bind_strategy_outcome(prediction.prediction_id,
                                     record.action_id)
        self.h.close_episode("t1", "ep1")

        # A NEW decision: episode 2 builds a context and predicts.
        self.provider.requests.clear()
        self.h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S04"}, "ep2")
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
                   "strategy_id": "S04"}, "ep1")
        record = self.solve(task, strategy="S04")
        self.h.bind_strategy_outcome(prediction.prediction_id,
                                     record.action_id)
        # Episode 1 is NOT closed yet: a new prediction's request carries no
        # pair from it.
        self.h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S04"}, "ep1")
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
                     "strategy_id": "S99"}, "ep1")
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
                     "strategy_id": "S04"}, "ep1")
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
                   "strategy_id": "S04"}, "ep1")
        self.h.close_episode("t1", "ep1")
        self.h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S04"}, "ep2")
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
                       "strategy_id": "S04"}, "ep1")
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
        self.assertTrue(benefit_reminders,
                        "a measured benefit bias must yield a reminder")
        r = benefit_reminders[0]
        self.assertEqual(r["kind"], "derived_reminder")
        self.assertEqual(r["basis"], "measured_statistics")
        self.assertIn("observed_bias", r)
        self.assertIn("watch_next_time", r)
        self.assertTrue(r["support"])

        # The NEXT request carries the reminders.
        self.provider.requests.clear()
        self.h.predict_strategy_outcome(
            _task("next"), {"action_type": "execute_strategy",
                            "strategy_id": "S04"}, "ep1")
        request = self.provider.requests[-1]
        sent = request["prediction_context"]["prediction_reminders"]
        self.assertTrue(sent)
        self.assertEqual(sent[0]["kind"], "derived_reminder")

    def test_below_threshold_yields_no_reminder(self):
        """A single episode is NOT a lesson: no reminder below the
        threshold."""
        task = _task("t1")
        prediction = self.h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S04"}, "ep1")
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

    def test_pending_hplus_reaches_the_next_request(self):
        """An H+ claim archived as pending appears in the next request, and
        pending is NOT reported as verified."""
        task = _task("t1")
        provider = StubProvider(payload=self._hplus_payload())
        h = ORHarness(home=self.home, world_model=provider,
                      embedding=self.backend)
        self.addCleanup(h.close)
        h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S04"}, "ep1")
        provider.requests.clear()
        h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S04"}, "ep2")
        request = provider.requests[-1]
        hplus = request["prediction_context"]["hplus_feedback"]
        self.assertGreaterEqual(hplus["n_gains_total"], 1)
        self.assertGreaterEqual(hplus["n_pending"], 1)
        self.assertEqual(hplus["n_effect_verified"], 0)
        self.assertEqual(hplus["gains"][0]["assessment"], "expected")

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
                   "strategy_id": "S04"}, "ep1")
        provider.requests.clear()
        h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S04"}, "ep2")
        request = provider.requests[-1]
        hplus = request["prediction_context"]["hplus_feedback"]
        self.assertIn("missing", hplus)


class TestObservationRules(CalibrationLoopCase):

    def test_quality_survives_a_failed_task_check(self):
        """wm-obs/2: the solver's quality figure is never rewritten by a
        task check; the check is a separate fact."""
        task = _task("t1")
        prediction = self.h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S04"}, "ep1")
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
                         "wm-obs/2")

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
                   "strategy_id": "S04"}, "ep1")
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
                          "strategy_id": "S04"}, "ep1")
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
                   "strategy_id": "S04"}, "ep1")
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


if __name__ == "__main__":
    unittest.main()
