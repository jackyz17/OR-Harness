"""Phase-A acceptance tests: bounded attribution and calibration.

What this file asserts, one behaviour per class:

1. **A missing execution receipt is a caveat, not a discard.** An
   unconfirmed config key that names no execution information (an
   unreported ``time_limit``, ``seed``, ``solver_timeout_s``) blocks
   NOTHING: the cost sample is still compared, the benefit is still scored
   when it is observable, and the exclusion reason names the fields instead
   of saying "identity not established".
2. **An approach cannot be confirmed → benefit blocked, cost kept.** An
   unconfirmed key that names the APPROACH (``method``, ``relaxation``)
   blocks the benefit and preserves the measured cost. This is the exact
   regression the user reported: a missing receipt used to discard the
   WHOLE sample.
3. **A condition deviation is not the original condition's performance.**
   A different solver (or a different value for a config key the run
   really reported) blocks the outcome dimensions and preserves the real
   cost and any failure: the attempt happened, so its spend stays.
4. **A performed method that differs from the plan blocks the benefit
   only** and is reported on its own (``method_deviation``), never as an
   identity problem.
5. **The genuine attribution boundaries still refuse.** A wrong task, a
   wrong strategy, a wrong attempt and a post-hoc prediction still exclude
   the sample entirely; an unexecuted candidate still gets no fabricated
   outcome.
6. **Entry-point field discipline.** A METHOD in ``config`` is folded into
   the candidate's ``method`` field and recorded; a key that names no
   execution parameter (``task_id``, ``scope``) is refused before an
   attempt is spent.
7. **An interrupted executor releases its claim** so the prediction is not
   stranded on a dead action, and no sample is double-counted.
8. **A rule rebuild re-scores a stored ``excluded`` sample** without
   rewriting the prediction and without counting it twice.
"""
import json
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tests.harness.helpers import HarnessTestCase  # noqa: E402

from or_harness.api import ORHarness  # noqa: E402
from or_harness.strategy.embedding_index import (  # noqa: E402
    LocalHashEmbeddingBackend,
)
from or_harness.world_model.attribution import (  # noqa: E402
    binding_attribution,
    check_candidate_config,
)

EMBEDDING_ENV_KEYS = ("OR_EMBEDDING_BACKEND", "OR_EMBEDDING_BASE_URL",
                      "OR_EMBEDDING_MODEL", "OR_EMBEDDING_API_KEY")

REQ_TEXT = ("A distribution centre must be loaded before the delivery window "
            "opens; demand 100 units may not be deferred.")

TASK = {"task_id": "t1", "family": "routing",
        "description": REQ_TEXT,
        "spec": {"n_vars": 100, "n_constraints": 50, "n_int_vars": 100},
        "annotations": {"coupling": {"resource_coupling": 0.3,
                                     "temporal_coupling": 0.1,
                                     "route_complexity": 0.2,
                                     "semantic_coupling": 0.5}}}

GOOD_PAYLOAD = {
    "benefit": {"kind": "solution_quality",
                "metric": "normalized_objective_gap", "unit": "1-gap",
                "value": 0.8,
                "baseline": {"kind": "conditional_stats", "value": 0.7}},
    "cost": {"solver_runtime_s": 3.0},
}


class Base(HarnessTestCase):
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

    def _script(self, name, body):
        work = Path(self.home) / f"ws_{name}"
        work.mkdir(parents=True, exist_ok=True)
        script = work / "solve.py"
        script.write_text(body, encoding="utf-8")
        return script, work

    def _plain_script(self, name, status="optimal", objective=100.0,
                      config=None, method=None):
        payload = {"status": status, "objective_value": objective,
                   "objective_bound": objective, "runtime_seconds": 0.01}
        if config is not None:
            payload["config"] = config
        if method is not None:
            payload["method_performed"] = method
        return self._script(
            name,
            "import json\n"
            "with open('result.json', 'w') as fh:\n"
            f"    json.dump({json.dumps(payload)}, fh)\n")

    def _stamped_script(self, name, config=None, method=None):
        """A script that stamps its receipt with the action id."""
        extra = ""
        if config is not None:
            cfg = dict(config)
            cfg["action_id"] = "__ACTION__"
            extra += (f"cfg = {json.dumps(cfg)}\n"
                      "cfg['action_id'] = os.environ.get('OR_ACTION_ID')\n")
        if method is not None:
            m = dict(method)
            m["action_id"] = "__ACTION__"
            extra += (f"method = {json.dumps(m)}\n"
                      "method['action_id'] = os.environ.get('OR_ACTION_ID')\n")
        body = ("import json, os\n" + extra +
                "payload = {'status': 'optimal', 'objective_value': 100.0,\n"
                "           'objective_bound': 100.0, 'runtime_seconds': 0.01}\n"
                "if 'cfg' in dir(): payload['config'] = cfg\n"
                "if 'method' in dir(): payload['method_performed'] = method\n"
                "with open('result.json', 'w') as fh:\n"
                "    json.dump(payload, fh)\n")
        return self._script(name, body)


class StubProvider:
    """A minimal provider (not a WorldModelProvider subclass) that returns a
    fixed payload and records the requests it received."""

    name = "stub-a"

    def __init__(self, payload=None, usage=None):
        self.payload = payload if payload is not None else GOOD_PAYLOAD
        self.usage = usage if usage is not None else {
            "prompt_tokens": 100, "completion_tokens": 50}
        self.requests = []

    def describe(self):
        return {"provider_model": "stub-a", "provider_version": "1"}

    def predict(self, request, timeout_s=None):
        self.requests.append(request)
        return {"payload": self.payload, "usage": self.usage,
                "error": None, "latency_s": 0.02}


# ---------------------------------------------------------------------------
# 1. the attribution table (pure, no harness)
# ---------------------------------------------------------------------------


class TestAttributionTable(unittest.TestCase):

    def test_an_unreported_effort_key_blocks_nothing(self):
        attr = binding_attribution(
            {}, {"config": {"time_limit": {"predicted": 60, "actual": None}}})
        blocked = {k: v for k, v in attr["blocked"].items() if v}
        self.assertEqual(blocked, {},
                         "an unreported time_limit must not discard a sample")
        self.assertEqual(attr["fields"]["config.time_limit"]["blocks"], [])

    def test_an_unreported_approach_key_blocks_the_benefit_only(self):
        attr = binding_attribution(
            {}, {"config": {"relaxation": {"predicted": "lp",
                                           "actual": None}}})
        self.assertTrue(attr["blocked"]["benefit"])
        self.assertFalse(attr["blocked"]["cost"])
        self.assertFalse(attr["blocked"]["risk"])

    def test_an_unknown_episode_blocks_nothing(self):
        attr = binding_attribution({}, {"episode_id": {"predicted": "ep1",
                                                       "actual": None}})
        blocked = {k: v for k, v in attr["blocked"].items() if v}
        self.assertEqual(blocked, {})

    def test_a_solver_mismatch_preserves_cost(self):
        attr = binding_attribution(
            {"solver": {"predicted": "highs", "actual": "glpk"}}, {})
        self.assertTrue(attr["blocked"]["benefit"])
        self.assertTrue(attr["blocked"]["risk"])
        self.assertFalse(attr["blocked"]["cost"],
                         "a different solver is a deviation: the spend stays")

    def test_a_config_value_mismatch_preserves_cost(self):
        attr = binding_attribution(
            {"config": {"time_limit": {"predicted": 60, "actual": 30}}}, {})
        self.assertTrue(attr["blocked"]["benefit"])
        self.assertFalse(attr["blocked"]["cost"])

    def test_a_wrong_task_excludes_everything(self):
        attr = binding_attribution({"task_id": {"predicted": "t1",
                                                "actual": "t2"}}, {})
        for dim in ("benefit", "cost", "risk", "interval"):
            self.assertTrue(attr["blocked"][dim],
                            f"a wrong task must block {dim}")

    def test_a_wrong_strategy_excludes_everything(self):
        attr = binding_attribution({"strategy_id": {"predicted": "S01",
                                                    "actual": "S02"}}, {})
        for dim in ("benefit", "cost", "risk", "interval"):
            self.assertTrue(attr["blocked"][dim],
                            "a different strategy is not this prediction's "
                            "attempt at all")

    def test_a_post_hoc_prediction_excludes_everything(self):
        attr = binding_attribution({"timing": {"predicted_at": 2,
                                               "action_started_at": 1}}, {})
        self.assertTrue(attr["blocked"]["benefit"])
        self.assertTrue(attr["blocked"]["cost"])

    def test_a_method_deviation_blocks_the_benefit_only(self):
        observation = {
            "verdict": "mismatch",
            "planned_name": "benders", "actual_name": "direct milp",
            "planned_steps": ["master", "subproblem"], "actual_steps": ["lp"],
            "reason": "the performed steps are not a subsequence",
        }
        attr = binding_attribution({}, {}, observation)
        self.assertTrue(attr["blocked"]["benefit"])
        self.assertFalse(attr["blocked"]["cost"])
        self.assertIsNotNone(attr["deviation"])
        self.assertEqual(attr["deviation"]["actual_name"], "direct milp")

    def test_a_matching_method_is_not_a_deviation(self):
        attr = binding_attribution({}, {}, {"verdict": "match"})
        blocked = {k: v for k, v in attr["blocked"].items() if v}
        self.assertEqual(blocked, {})
        self.assertIsNone(attr["deviation"])

    def test_an_unknown_method_is_not_a_deviation(self):
        attr = binding_attribution({}, {}, {"verdict": "unknown",
                                            "reason": "no steps"})
        self.assertIsNone(attr["deviation"],
                          "an unobserved performance is not a deviation")


# ---------------------------------------------------------------------------
# 2. entry-point field discipline
# ---------------------------------------------------------------------------


class TestCandidateConfigEntry(unittest.TestCase):

    def test_a_method_in_config_is_folded_into_method(self):
        result = check_candidate_config({"method": "benders",
                                         "time_limit": 60})
        self.assertEqual(result["config"], {"time_limit": 60})
        self.assertEqual(result["method"], "benders")
        self.assertTrue(result["notes"])
        self.assertEqual(result["errors"], [])

    def test_a_method_in_config_does_not_override_an_existing_method(self):
        result = check_candidate_config(
            {"method": "benders"}, {"name": "direct", "steps": ["lp"]})
        self.assertEqual(result["config"], {})
        self.assertEqual(result["method"]["name"], "direct")
        self.assertTrue(result["notes"])

    def test_a_non_execution_key_is_refused(self):
        result = check_candidate_config({"task_id": "t1",
                                         "time_limit": 60})
        self.assertEqual(len(result["errors"]), 1)
        self.assertIn("task_id", result["errors"][0])
        # The valid key survives.
        self.assertEqual(result["config"], {"time_limit": 60})

    def test_effort_keys_pass_through_untouched(self):
        result = check_candidate_config({"time_limit": 60, "seed": 42,
                                         "mip_gap": 0.01})
        self.assertEqual(result["config"], {"time_limit": 60, "seed": 42,
                                            "mip_gap": 0.01})
        self.assertEqual(result["notes"], [])
        self.assertEqual(result["errors"], [])


# ---------------------------------------------------------------------------
# 3. end-to-end: a missing receipt no longer discards the sample
# ---------------------------------------------------------------------------


class TestMissingReceiptIsACaveat(Base):

    def _predict(self, **config):
        return self.h.predict_strategy_outcome(
            TASK, {"action_type": "execute_strategy", "strategy_id": "S01",
                   "solver": "highs", "config": config}, "ep1")

    def test_an_unreported_effort_key_keeps_the_cost_sample(self):
        """The exact regression the user reported: a config key the run
        never reports used to discard the WHOLE evaluation."""
        script, work = self._plain_script("caveat")
        prediction = self._predict(time_limit=60, seed=42)
        record = self.h.execute(TASK, "S01", str(script), str(work),
                                solver="highs", episode_id="ep1",
                                prediction_id=prediction.prediction_id)
        self.h.record(record)
        result = self.h.close_episode("t1", "ep1")
        evaluation = result["evaluations"][0]
        # The sample is NOT excluded: the benefit and the cost are scored.
        self.assertEqual(evaluation["state"], "evaluated")
        self.assertEqual(evaluation["benefit"]["eligibility"], "evaluable")
        self.assertEqual(evaluation["cost"]["eligibility"], "evaluable")
        self.assertIn("solver_runtime_s", evaluation["cost"]["per_dim"])
        # The unconfirmed keys are still REPORTED, but block nothing.
        self.assertEqual(evaluation.get("attribution") or {}, {})
        info = self.h.strategy_predictions.get(
            prediction.prediction_id).trace.model_info
        self.assertIn("time_limit", (info.get("binding_unknown") or {}).get(
            "config", {}))

    def test_an_unreported_approach_key_keeps_the_cost_but_blocks_benefit(self):
        script, work = self._plain_script("approach")
        prediction = self._predict(relaxation="lp")
        record = self.h.execute(TASK, "S01", str(script), str(work),
                                solver="highs", episode_id="ep1",
                                prediction_id=prediction.prediction_id)
        self.h.record(record)
        result = self.h.close_episode("t1", "ep1")
        evaluation = result["evaluations"][0]
        # The benefit is blocked (the approach could not be confirmed)...
        self.assertEqual(evaluation["benefit"]["eligibility"],
                         "identity_unknown")
        self.assertIn("config.relaxation",
                      evaluation["benefit"]["identity_fields"])
        # ...but the measured cost is preserved and still scored.
        self.assertEqual(evaluation["cost"]["eligibility"], "evaluable")
        self.assertIn("solver_runtime_s", evaluation["cost"]["per_dim"])
        self.assertEqual(evaluation["state"], "evaluated")


# ---------------------------------------------------------------------------
# 4. end-to-end: a condition deviation is reported and keeps the cost
# ---------------------------------------------------------------------------


class TestConditionDeviation(Base):

    def test_a_different_reported_config_value_preserves_cost(self):
        """The run really reported a DIFFERENT value for a config key: the
        quality is not the predicted condition's performance, but the spend
        really happened."""
        script, work = self._stamped_script(
            "deviation", config={"time_limit": 30})
        prediction = self.h.predict_strategy_outcome(
            TASK, {"action_type": "execute_strategy", "strategy_id": "S01",
                   "solver": "highs", "config": {"time_limit": 60}}, "ep1")
        # Run WITHOUT the prediction id (the run used a different value), and
        # bind explicitly — the honest deviation path.
        record = self.h.execute(TASK, "S01", str(script), str(work),
                                solver="highs", episode_id="ep1")
        self.h.record(record)
        bound = self.h.bind_strategy_outcome(prediction.prediction_id,
                                             record.action_id)
        self.assertTrue((bound.trace.model_info.get("binding_mismatch")
                         or {}).get("config"))
        result = self.h.close_episode("t1", "ep1")
        evaluation = result["evaluations"][0]
        # Benefit blocked (a deviation), cost preserved.
        self.assertEqual(evaluation["benefit"]["eligibility"],
                         "identity_mismatch")
        self.assertEqual(evaluation["cost"]["eligibility"], "evaluable")
        self.assertEqual(evaluation["state"], "evaluated")

    def test_a_different_solver_preserves_cost_and_reports_the_deviation(self):
        script, work = self._plain_script("solver_dev")
        prediction = self.h.predict_strategy_outcome(
            TASK, {"action_type": "execute_strategy", "strategy_id": "S01",
                   "solver": "glpk"}, "ep1")
        # The run really used highs; bind explicitly to record the deviation.
        record = self.h.execute(TASK, "S01", str(script), str(work),
                                solver="highs", episode_id="ep1")
        self.h.record(record)
        self.h.bind_strategy_outcome(prediction.prediction_id,
                                     record.action_id)
        result = self.h.close_episode("t1", "ep1")
        evaluation = result["evaluations"][0]
        self.assertEqual(evaluation["benefit"]["eligibility"],
                         "identity_mismatch")
        self.assertIn("solver", evaluation["benefit"]["identity_fields"])
        self.assertEqual(evaluation["cost"]["eligibility"], "evaluable")


# ---------------------------------------------------------------------------
# 5. end-to-end: genuine boundaries still refuse
# ---------------------------------------------------------------------------


class TestGenuineBoundariesStillRefuse(Base):

    def test_a_wrong_strategy_still_excludes_the_sample(self):
        script, work = self._plain_script("wrong_strategy")
        prediction = self.h.predict_strategy_outcome(
            TASK, {"action_type": "execute_strategy", "strategy_id": "S01"},
            "ep1")
        # A DIFFERENT strategy really ran; bind explicitly to record it.
        record = self.h.execute(TASK, "S02", str(script), str(work),
                                solver="highs", episode_id="ep1")
        self.h.record(record)
        self.h.bind_strategy_outcome(prediction.prediction_id,
                                     record.action_id)
        result = self.h.close_episode("t1", "ep1")
        evaluation = result["evaluations"][0]
        self.assertEqual(evaluation["state"], "excluded")
        self.assertFalse(evaluation["calibratable"])
        self.assertTrue(evaluation["exclusion_reasons"])

    def test_an_unexecuted_candidate_gets_no_outcome(self):
        prediction = self.h.predict_strategy_outcome(
            TASK, {"action_type": "execute_strategy", "strategy_id": "S01"},
            "ep1")
        script, work = self._plain_script("other_ran")
        record = self.h.execute(TASK, "S04", str(script), str(work),
                                solver="highs", episode_id="ep1")
        self.h.record(record)
        # Bind the unused prediction to the other run.
        self.h.bind_strategy_outcome(prediction.prediction_id,
                                     record.action_id)
        result = self.h.close_episode("t1", "ep1")
        evaluation = result["evaluations"][0]
        self.assertEqual(evaluation["state"], "excluded")


# ---------------------------------------------------------------------------
# 6. end-to-end: a method deviation is an evaluation fact
# ---------------------------------------------------------------------------


class TestMethodDeviationIsAnEvaluationFact(Base):

    def test_a_different_performed_method_blocks_the_benefit_only(self):
        plan = {"name": "benders decomposition",
                "steps": ["build master problem", "solve subproblem"]}
        performed = {"name": "direct milp",
                     "steps": ["build milp", "solve"]}
        script, work = self._stamped_script("method_dev", method=performed)
        prediction = self.h.predict_strategy_outcome(
            TASK, {"action_type": "execute_strategy", "strategy_id": "S01",
                   "solver": "highs", "method": plan}, "ep1")
        record = self.h.execute(TASK, "S01", str(script), str(work),
                                solver="highs", episode_id="ep1",
                                prediction_id=prediction.prediction_id)
        self.h.record(record)
        result = self.h.close_episode("t1", "ep1")
        evaluation = result["evaluations"][0]
        # The deviation is reported on its own block.
        self.assertIsNotNone(evaluation.get("method_deviation"))
        self.assertEqual(
            evaluation["method_deviation"]["actual_name"], "direct milp")
        # Benefit blocked (not the planned method's answer), cost kept.
        self.assertEqual(evaluation["benefit"]["eligibility"], "identity_unknown")
        self.assertEqual(evaluation["cost"]["eligibility"], "evaluable")
        self.assertTrue(any("planned method" in n
                            for n in evaluation.get("notes") or []))


# ---------------------------------------------------------------------------
# 7. entry-point refusal reaches the API
# ---------------------------------------------------------------------------


class TestEntryRefusalReachesTheApi(Base):

    def test_a_non_execution_config_key_is_refused_before_any_attempt(self):
        with self.assertRaises(ValueError) as caught:
            self.h.predict_strategy_outcome(
                TASK, {"action_type": "execute_strategy",
                       "strategy_id": "S01", "config": {"task_id": "t2"}},
                "ep1")
        self.assertIn("not an execution parameter", str(caught.exception))

    def test_a_method_in_config_is_normalized_and_recorded(self):
        prediction = self.h.predict_strategy_outcome(
            TASK, {"action_type": "execute_strategy", "strategy_id": "S01",
                   "config": {"method": "benders", "time_limit": 60}}, "ep1")
        self.assertEqual(prediction.candidate.config, {"time_limit": 60})
        # The bare string is normalized to the ONE method shape, never left
        # as a string (which crashed ``dict()`` on read).
        self.assertEqual(prediction.candidate.method,
                         {"name": "benders", "steps": []})
        notes = prediction.trace.model_info.get("config_normalized")
        self.assertTrue(notes)
        self.assertTrue(any("APPROACH" in n for n in notes))


# ---------------------------------------------------------------------------
# 8a. a string method survives the full save/read/execute round trip
# ---------------------------------------------------------------------------


class TestMethodRoundTrip(Base):
    """The reported regression: a string ``config.method`` saved fine but
    crashed ``CandidateRef.from_dict`` on the next read (``dict("...")``)."""

    def test_config_method_string_round_trips_and_executes(self):
        prediction = self.h.predict_strategy_outcome(
            TASK, {"action_type": "execute_strategy", "strategy_id": "S01",
                   "solver": "highs",
                   "config": {"method": "assignment MILP"}}, "ep1")
        self.assertEqual(prediction.candidate.method,
                         {"name": "assignment MILP", "steps": []})
        # Reading it back must not raise, and must carry the SAME shape.
        reloaded = self.h.get_strategy_outcome_prediction(
            prediction.prediction_id)
        self.assertIsNotNone(reloaded)
        self.assertEqual(reloaded.candidate.method,
                         {"name": "assignment MILP", "steps": []})
        self.assertEqual(reloaded.candidate.config, {})
        # And the full predict -> save -> read -> execute chain runs.
        script, work = self._plain_script("method_rt")
        record = self.h.execute(TASK, None, str(script), str(work),
                                solver=None, episode_id="ep1",
                                prediction_id=prediction.prediction_id)
        self.h.record(record)
        result = self.h.close_episode("t1", "ep1")
        self.assertEqual(result["evaluations"][0]["state"], "evaluated")

    def test_a_bare_string_method_is_read_as_the_name(self):
        prediction = self.h.predict_strategy_outcome(
            TASK, {"action_type": "execute_strategy", "strategy_id": "S02",
                   "solver": "highs", "method": "assignment MILP"}, "ep1")
        self.assertEqual(prediction.candidate.method,
                         {"name": "assignment MILP", "steps": []})
        reloaded = self.h.get_strategy_outcome_prediction(
            prediction.prediction_id)
        self.assertEqual(reloaded.candidate.method,
                         {"name": "assignment MILP", "steps": []})


# ---------------------------------------------------------------------------
# 8b. an interrupted attempt is preserved as its own failure fact
# ---------------------------------------------------------------------------


class TestInterruptedAttemptIsPreserved(Base):
    """The reported regression: an executor exception left no execution
    fact, so the first failure vanished from calibration and the retry was
    labelled ``not_occurred`` with ``retries=0``."""

    def setUp(self):
        super().setUp()
        self.script, self.work = self._plain_script("interrupted")

    def _interrupt_once(self, prediction_id):
        """A probe executor that raises on the FIRST call only."""
        real = self.h.executor

        class Probe:
            def __init__(self):
                self.n = 0

            def execute(self, *a, **k):
                self.n += 1
                if self.n == 1:
                    raise FileExistsError(
                        "[Errno 17] File exists: 'result.json'")
                return real.execute(*a, **k)

        self.h.executor = Probe()

    def test_the_failed_attempt_keeps_its_cost_and_failure_class(self):
        prediction = self.h.predict_strategy_outcome(
            TASK, {"action_type": "execute_strategy", "strategy_id": "S01",
                   "solver": "highs"}, "ep1")
        self._interrupt_once(prediction.prediction_id)
        with self.assertRaises(FileExistsError):
            self.h.execute(TASK, "S01", str(self.script), str(self.work),
                           solver="highs", episode_id="ep1",
                           prediction_id=prediction.prediction_id)
        # A REAL failure fact exists, staged, with a classification.
        staged = self.h.bank.pending(task_id="t1")
        self.assertEqual(len(staged), 1)
        failure = staged[0]
        self.assertEqual(failure.quality["status"], "error")
        self.assertFalse(failure.quality["feasible"])
        self.assertTrue(failure.failures)
        self.assertIsNotNone(failure.failures[0].error_class)
        self.assertIn("interrupted_before_record", failure.execution_features)
        # And it is LINKED to the failed action, so the close-out can see it.
        action = [a for a in self.h.actions.query(
            task_id="t1", episode_id="ep1", action_type="execute_strategy")
            if a.status == "failed"][0]
        self.assertEqual(action.linked_execution_id, failure.execution_id)

    def test_the_failure_is_an_observation_and_the_retry_is_not_a_zero(self):
        prediction = self.h.predict_strategy_outcome(
            TASK, {"action_type": "execute_strategy", "strategy_id": "S01",
                   "solver": "highs"}, "ep1")
        self._interrupt_once(prediction.prediction_id)
        with self.assertRaises(FileExistsError):
            self.h.execute(TASK, "S01", str(self.script), str(self.work),
                           solver="highs", episode_id="ep1",
                           prediction_id=prediction.prediction_id)
        # The retry is a SEPARATE attempt: it cannot claim a proven zero.
        retry = self.h.execute(TASK, "S01", str(self.script), str(self.work),
                               solver="highs", episode_id="ep1",
                               prediction_id=prediction.prediction_id)
        self.assertNotIn("retries", retry.cost.measured_dims())
        self.assertNotIn("retries_proof", retry.execution_features)
        self.h.record(retry)
        result = self.h.close_episode("t1", "ep1")
        # The EPISODE occurrence counts BOTH attempts: the failed one really
        # happened and is now an observation, so the failure cannot be
        # erased by the retry.
        occurrence = (result["calibration_summary"]["occurrence"]
                      ["implementation_failure"])
        self.assertEqual(occurrence["n_observation_units"], 2)
        self.assertEqual(occurrence["n_occurred"], 1)
        self.assertEqual(occurrence["unit_occurrence_rate"], 0.5)

    def test_a_failed_attempt_is_charged_to_the_episode_budget(self):
        """The failure really spent what it spent: the episode budget counts
        the interrupted attempt once, alongside the retry."""
        prediction = self.h.predict_strategy_outcome(
            TASK, {"action_type": "execute_strategy", "strategy_id": "S01",
                   "solver": "highs"}, "ep1")
        self._interrupt_once(prediction.prediction_id)
        with self.assertRaises(FileExistsError):
            self.h.execute(TASK, "S01", str(self.script), str(self.work),
                           solver="highs", episode_id="ep1",
                           prediction_id=prediction.prediction_id)
        retry = self.h.execute(TASK, "S01", str(self.script), str(self.work),
                               solver="highs", episode_id="ep1",
                               prediction_id=prediction.prediction_id)
        self.h.record(retry)
        view = self.h.budget.view("t1", "ep1")
        # Two attempts, one of them the abandoned failure.
        self.assertEqual(view["consumption"]["n_attempts"], 2)
        staged_ids = [r.execution_id
                      for r in self.h.bank.pending(task_id="t1")]
        # The failed attempt is counted (either recorded or staged), and the
        # retry is recorded — never double-counted.
        self.assertEqual(len(set(staged_ids)), len(staged_ids))
        self.assertEqual(retry.execution_id in staged_ids, False)


# ---------------------------------------------------------------------------
# 8. an interrupted executor releases its claim
# ---------------------------------------------------------------------------


class TestClaimRelease(Base):

    def test_an_interrupted_run_frees_the_prediction_for_a_real_attempt(self):
        script, work = self._plain_script("claim_release")
        prediction = self.h.predict_strategy_outcome(
            TASK, {"action_type": "execute_strategy", "strategy_id": "S01",
                   "solver": "highs"}, "ep1")

        class Probe:
            def __init__(self, real):
                self.real = real

            def execute(self, *a, **k):
                raise RuntimeError("interrupted before the sandbox")

        self.h.executor = Probe(self.h.executor)
        with self.assertRaises(RuntimeError):
            self.h.execute(TASK, "S01", str(script), str(work),
                           solver="highs", episode_id="ep1",
                           prediction_id=prediction.prediction_id)
        # The claim was released: the SAME prediction can be tested now.
        stored = self.h.strategy_predictions.get(prediction.prediction_id)
        self.assertIsNone(stored.trace.model_info.get("bound_action_id"))
        self.assertEqual(stored.trace.model_info.get("association_phase"),
                         "released_no_execution")
        # A real attempt now binds and evaluates normally.
        self.h.executor = self._real_executor()
        record = self.h.execute(TASK, "S01", str(script), str(work),
                                solver="highs", episode_id="ep1",
                                prediction_id=prediction.prediction_id)
        self.h.record(record)
        result = self.h.close_episode("t1", "ep1")
        evaluation = result["evaluations"][0]
        self.assertEqual(evaluation["state"], "evaluated")

    def _real_executor(self):
        from or_harness.execution.executor import SafePythonExecutor
        return SafePythonExecutor()


# ---------------------------------------------------------------------------
# 9. a rule rebuild re-scores a stored excluded sample
# ---------------------------------------------------------------------------


class TestRuleRebuild(Base):

    def test_a_stored_excluded_sample_is_re_scored_without_a_new_sample(self):
        """An evaluation stored EXCLUDED (under the old blanket rule) is
        re-derived at read time; the corrected result counts, the stored
        record is untouched, and repeated reads never add a second sample."""
        from or_harness.world_model.episode_closeout import (
            StrategyPredictionEvaluation,
            _live_evaluation,
        )
        script, work = self._plain_script("rule_rebuild")
        prediction = self.h.predict_strategy_outcome(
            TASK, {"action_type": "execute_strategy", "strategy_id": "S01",
                   "solver": "highs", "config": {"time_limit": 60}}, "ep1")
        record = self.h.execute(TASK, "S01", str(script), str(work),
                                solver="highs", episode_id="ep1",
                                prediction_id=prediction.prediction_id)
        self.h.record(record)
        self.h.close_episode("t1", "ep1")

        # Simulate a STORE written under the OLD rule: the same evaluation,
        # but marked excluded with blanket identity reasons.
        stored = self.h.strategy_predictions.get(prediction.prediction_id)
        legacy = StrategyPredictionEvaluation(
            prediction_id=prediction.prediction_id, task_id="t1",
            episode_id="ep1", state="excluded",
            benefit={"eligibility": "identity_mismatch",
                     "reason": "the binding identity is not established"},
            cost={"eligibility": "identity_mismatch",
                  "reason": "the binding identity is not established"},
            risk={"eligibility": "identity_mismatch", "scored": [],
                  "unscored": [], "observed_units": {}},
            interval={"eligibility": "not_predicted"},
        )
        derived, correction = _live_evaluation(self.h, legacy)
        # The corrected rules re-score the sample from the SAME facts.
        self.assertEqual(derived.state, "evaluated")
        self.assertIsNotNone(correction)
        self.assertEqual(correction["kind"], "rule_rebuild")
        self.assertEqual(correction["stored_state"], "excluded")
        self.assertEqual(correction["derived_state"], "evaluated")
        self.assertIn("state", correction["fields"])
        # The stored record is never mutated.
        self.assertEqual(legacy.state, "excluded")
        self.assertEqual(legacy.benefit["eligibility"], "identity_mismatch")
        # Repeating the derivation is idempotent: same result, same reason.
        derived2, correction2 = _live_evaluation(self.h, legacy)
        self.assertEqual(derived2.state, derived.state)
        self.assertEqual(correction2["kind"], correction["kind"])


if __name__ == "__main__":
    unittest.main()
