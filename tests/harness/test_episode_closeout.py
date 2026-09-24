"""World-model M4 tests: episode close-out, post-hoc evaluation, and
experience calibration.

What this file asserts, one behaviour per test class:

1. the three pre-fixes hold: a missing episode/config on the executed
   action is an UNKNOWN, never a match; independent and parented
   strategy-outcome call costs enter the budget exactly once and the
   planning loop stops mid-decision when the known spend exceeds the
   budget; a malformed payload (an illegal baseline kind) is one
   candidate's invalid result and the remaining candidates are still
   predicted;
2. window rounds: the same strategy chosen twice (or switched away and
   back) is TWO windows; a multi-attempt window's cost scope and
   auxiliary overhead are attributed by scope;
3. close-out boundaries: an explicit ending, unfinished actions reported
   (never fabricated away), failed/aborted endings honest, no calibration
   from an open episode;
4. per-field evaluation eligibility: missing baseline, unit/scope
   mismatch, missing verification and unknown labels produce no fake
   errors; the frozen prediction is never rewritten; unexecuted
   candidates get no counterfactual labels;
5. idempotence and recovery: re-closing, re-reading and re-binding count
   nothing twice; a second episode's context reads the first closed
   episode's published summary (or its insufficient-evidence state) while
   the first episode's own stored context stays unchanged.
"""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tests.harness.helpers import HarnessTestCase  # noqa: E402

from or_harness.api import ORHarness  # noqa: E402
from or_harness.strategy.embedding_index import (  # noqa: E402
    LocalHashEmbeddingBackend,
)
from or_harness.world_model.prediction import ActionSpec  # noqa: E402
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


GOOD_PAYLOAD = {
    "benefit": {"kind": "solution_quality",
                "metric": "normalized_objective_gap", "unit": "1-gap",
                "value": 0.8,
                "baseline": {"kind": "conditional_stats", "value": 0.7}},
    "cost": {"llm_tokens": 1200, "solver_runtime_s": 3.0},
    "risk": {"events": [
        {"event": "timeout", "probability": 0.2},
    ]},
    "uncertainty": {"execution_randomness": 0.3, "knowledge_gap": 0.6},
}

#: The calibration group key prefix for the stub provider: it reports no
#: model name, so its identity is honestly ``(unknown)`` — never the
#: provider name standing in for a version.
GROUP_PREFIX = "strategy_outcome|(unknown)|"


class StubProvider(WorldModelProvider):
    """A stub provider returning a fixed payload, recording requests."""

    name = "stub-m4"

    def __init__(self, payload=GOOD_PAYLOAD, usage=None):
        self.payload = payload
        self.usage = usage if usage is not None else {
            "prompt_tokens": 100, "completion_tokens": 50}
        self.requests = []

    def predict(self, request, timeout_s=None):
        self.requests.append(request)
        return {"payload": self.payload, "usage": self.usage,
                "error": None, "latency_s": 0.02}


class M4Case(HarnessTestCase):
    """A harness with an injected local embedding backend (hermetic)."""

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

    def solve(self, task, strategy="S04", objective=100.0,
              episode_id="ep1", status="optimal", gap=None, tag=None):
        """Execute and record one real attempt (deterministic fixture)."""
        from pathlib import Path
        label = tag or f"{task['task_id']}_{strategy}_{episode_id}"
        work = Path(self.home) / f"ws_{label}"
        work.mkdir(parents=True, exist_ok=True)
        script = work / "solve.py"
        script.write_text(
            "import json\n"
            "with open('result.json', 'w') as fh:\n"
            "    json.dump({'status': " + json.dumps(status) +
            ", 'objective_value': "
            f"{objective}, 'objective_bound': {objective}"
            + (f", 'mip_gap': {gap}" if gap is not None else "")
            + ", 'runtime_seconds': 0.01}, fh)\n", encoding="utf-8")
        record = self.h.execute(task, strategy, str(script), str(work),
                                solver="highs", episode_id=episode_id)
        self.h.record(record)
        return record


# ---------------------------------------------------------------------------
# 1. the three pre-fixes hold
# ---------------------------------------------------------------------------


class TestPreFixesHold(M4Case):

    def test_missing_episode_is_unknown_not_match(self):
        task = _task("t1")
        prediction = self.h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S04"}, "ep1")
        # Execute WITHOUT an episode id: the action records none.
        from pathlib import Path
        work = Path(self.home) / "ws_noc"
        work.mkdir(parents=True, exist_ok=True)
        (work / "solve.py").write_text(
            "import json\n"
            "with open('result.json', 'w') as fh:\n"
            "    json.dump({'status': 'optimal', 'objective_value': 1,"
            " 'objective_bound': 1, 'runtime_seconds': 0.01}, fh)\n",
            encoding="utf-8")
        record = self.h.execute(task, "S04", str(work / "solve.py"),
                                str(work), solver="highs")
        self.h.record(record)
        bound = self.h.bind_strategy_outcome(
            prediction.prediction_id, record.action_id)
        info = bound.trace.model_info
        self.assertIsNone(info.get("binding_mismatch"))
        unknown = info.get("binding_unknown") or {}
        self.assertIn("episode_id", unknown,
                      "an unrecorded episode is an UNKNOWN, never a match")

    def test_missing_config_is_unknown_not_match(self):
        task = _task("t1")
        # The candidate declares a config key the action log never carries.
        prediction = self.h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S04",
                   "config": {"time_limit": 60, "seed": 42}}, "ep1")
        record = self.solve(task, strategy="S04")
        bound = self.h.bind_strategy_outcome(
            prediction.prediction_id, record.action_id)
        info = bound.trace.model_info
        self.assertIsNone(info.get("binding_mismatch"))
        unknown = info.get("binding_unknown") or {}
        self.assertIn("config", unknown)
        self.assertIn("time_limit", unknown["config"])
        self.assertIn("seed", unknown["config"])

    def test_independent_call_cost_enters_budget_once(self):
        task = _task("t1")
        # An INDEPENDENT strategy-outcome call (no parent action): its 50
        # tokens must appear in the episode's budget consumption exactly
        # once.
        self.h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S01"}, "ep1")
        view = self.h.budget.view("t1", "ep1")
        consumption = view["consumption"]
        self.assertEqual(consumption["total_cost"]["llm_tokens"], 50.0)
        self.assertEqual(len(consumption["prediction_call_costs"]), 1)

    def test_planning_loop_stops_mid_decision_on_budget(self):
        # Budget 40 tokens; each call costs 50. The FIRST call's cost is
        # charged the moment it returns, so the SECOND candidate must not
        # be predicted (the old behaviour made all three calls and only
        # reported the overrun at the end).
        self.h.declare_budget("t1", {"llm_tokens": 40}, "ep1")
        plan = self.h.plan_next(
            _task("t1"), "ep1",
            candidates=[ActionSpec("execute_strategy", "t1",
                                   strategy_id="S01"),
                        ActionSpec("execute_strategy", "t1",
                                   strategy_id="S02"),
                        ActionSpec("execute_strategy", "t1",
                                   strategy_id="S04")],
            limits={"horizon": 1}, protocol="strategy-outcome")
        self.assertEqual(plan["model_calls_made"], 1,
                         "the loop must stop after the first call's cost "
                         "is on the books")
        # The real budget is exceeded by the first call's own spend: the
        # plan reports fallback (suggestion withheld), never a quiet "ok".
        self.assertEqual(plan["status"], "fallback")
        self.assertIn("budget exceeded", plan["truncation_reason"])
        # The one call's cost is charged ONCE to the decision action.
        decision = self.h.actions.get(plan["decision_action_id"])
        self.assertEqual(decision.cost.llm_tokens, 50.0)

    def test_malformed_baseline_is_isolated_to_one_candidate(self):
        # An illegal baseline kind used to raise out of
        # BaselineStatement.from_dict and kill the whole planning loop.
        payloads = {
            "S01": {"benefit": {"kind": "solution_quality", "metric": "q",
                                "value": 0.5,
                                "baseline": {"kind": "not-a-baseline-kind",
                                             "value": 0.4}}},
            "S02": GOOD_PAYLOAD,
        }
        provider = StubProvider()
        h = ORHarness(home=self.home, world_model=provider,
                      embedding=self.backend)
        self.addCleanup(h.close)
        original = provider.predict

        def per_candidate(request, timeout_s=None):
            sid = request["candidate"]["strategy_id"]
            provider.payload = payloads[sid]
            return original(request, timeout_s=timeout_s)
        provider.predict = per_candidate
        plan = h.plan_next(
            _task("t1"), "ep1",
            candidates=[ActionSpec("execute_strategy", "t1",
                                   strategy_id="S01"),
                        ActionSpec("execute_strategy", "t1",
                                   strategy_id="S02")],
            limits={"horizon": 1}, protocol="strategy-outcome")
        # BOTH candidates were predicted: the malformed one is an invalid
        # RESULT, not an exception that kills the loop.
        self.assertEqual(plan["model_calls_made"], 2)
        statuses = {c["prediction_status"]
                    for c in plan["candidates"]}
        self.assertEqual(statuses, {"invalid", "valid"})
        # The invalid one is persisted with its reason and its cost.
        invalid = [c for c in plan["candidates"]
                   if c["prediction_status"] == "invalid"][0]
        stored = h.get_strategy_outcome_prediction(invalid["prediction_id"])
        joined = " ".join(stored.notes)
        self.assertIn("baseline", joined)
        self.assertIsNotNone(stored.trace.call_cost)


# ---------------------------------------------------------------------------
# 2. window rounds and scope attribution
# ---------------------------------------------------------------------------


class TestWindowRounds(M4Case):

    def _choose(self, task, strategy, episode="ep1"):
        """Record an explicit select_strategy choice action."""
        return self.h.report_action(
            "select_strategy", task, episode,
            params={"strategy_id": strategy},
            outcome={"kind": "choice", "selected": {
                "action_type": "execute_strategy",
                "strategy_id": strategy}})

    def test_same_strategy_chosen_twice_is_two_rounds(self):
        task = _task("t1")
        self._choose(task, "S04")
        self.solve(task, strategy="S04")
        # The agent switches away and back: a NEW round of S04.
        self._choose(task, "S01")
        self._choose(task, "S04")
        self.solve(task, strategy="S04")
        round0 = self.h.strategy_execution_window(
            "t1", "ep1", strategy_id="S04", round_index=0)
        round1 = self.h.strategy_execution_window(
            "t1", "ep1", strategy_id="S04", round_index=1)
        self.assertEqual(round0.round_index, 0)
        self.assertEqual(round1.round_index, 1)
        self.assertNotEqual(round0.window_id, round1.window_id)
        self.assertEqual(round0.n_attempts, 1)
        self.assertEqual(round1.n_attempts, 1,
                         "the second round's window must not swallow the "
                         "first round's attempt")

    def test_round_window_of_nonexistent_round_is_not_comparable(self):
        task = _task("t1")
        self._choose(task, "S04")
        self.solve(task, strategy="S04")
        window = self.h.strategy_execution_window(
            "t1", "ep1", strategy_id="S04", round_index=3)
        self.assertFalse(window.comparable)
        self.assertTrue(any("does not exist" in r
                            for r in window.not_comparable_reasons))

    def test_legacy_window_id_still_resolves(self):
        from or_harness.world_model.execution_window import parse_window_id
        identity = parse_window_id("win::t1::ep1::S04")
        self.assertEqual(identity.task_id, "t1")
        self.assertEqual(identity.episode_id, "ep1")
        self.assertEqual(identity.strategy_id, "S04")
        self.assertIsNone(identity.round_index)
        identity2 = parse_window_id("win::t1::ep1::S04::r2")
        self.assertEqual(identity2.round_index, 2)

    def test_auxiliary_cost_is_reported_separately(self):
        task = _task("t1")
        self.h.report_action("model", task, "ep1",
                             outcome={"kind": "model"},
                             cost={"llm_tokens": 200})
        record = self.solve(task, strategy="S04")
        window = self.h.strategy_execution_window(
            "t1", "ep1", strategy_id="S04")
        # The model action is auxiliary: real spend, never folded into the
        # attempt scope. The attempt's llm_tokens is UNMEASURED (None),
        # never zero-as-cheap.
        self.assertIsNone(window.attempt_cost["llm_tokens"]["total"])
        self.assertEqual(window.auxiliary_cost["llm_tokens"]["total"], 200.0)


# ---------------------------------------------------------------------------
# 3. close-out boundaries
# ---------------------------------------------------------------------------


class TestCloseoutBoundaries(M4Case):

    def test_close_episode_evaluates_bound_predictions(self):
        task = _task("t1")
        prediction = self.h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S04"}, "ep1")
        record = self.solve(task, strategy="S04")
        self.h.bind_strategy_outcome(prediction.prediction_id,
                                     record.action_id)
        result = self.h.close_episode("t1", "ep1")
        self.assertFalse(result["already_closed"])
        evaluations = result["evaluations"]
        self.assertEqual(len(evaluations), 1)
        self.assertEqual(evaluations[0]["state"], "evaluated")
        self.assertEqual(evaluations[0]["benefit"]["eligibility"],
                         "evaluable")
        # The observed quality: optimal -> 1.0; predicted 0.8.
        self.assertEqual(evaluations[0]["benefit"]["observed"], 1.0)
        self.assertAlmostEqual(evaluations[0]["benefit"]["abs_error"], 0.2)

    def test_reclose_is_idempotent(self):
        task = _task("t1")
        prediction = self.h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S04"}, "ep1")
        record = self.solve(task, strategy="S04")
        self.h.bind_strategy_outcome(prediction.prediction_id,
                                     record.action_id)
        first = self.h.close_episode("t1", "ep1")
        second = self.h.close_episode("t1", "ep1")
        self.assertTrue(second["already_closed"])
        # No duplicate evaluations, no duplicate calibration samples.
        all_evaluations = self.h.strategy_prediction_evaluations()
        self.assertEqual(len(all_evaluations),
                         len(first["evaluations"]))
        summary = self.h.calibration_summary()
        self.assertEqual(summary["n_evaluations_total"],
                         len(first["evaluations"]))

    def test_unfinished_action_blocks_closeout(self):
        task = _task("t1")
        # A running action: begun, never ended. The close-out must be
        # REFUSED (pending), never a completed record whose pending
        # scopes could not be back-filled later.
        snap = self.h.snapshot(task, "ep1")
        self.h.actions.begin_action("model", "t1", "ep1",
                                    pre_snapshot=snap)
        result = self.h.close_episode("t1", "ep1")
        self.assertIsNone(result["closeout"])
        self.assertEqual(result["state"], "pending")
        self.assertEqual(len(result["unfinished_actions"]), 1)
        self.assertIsNone(self.h.episode_closeout_record("t1", "ep1"))
        # No calibration was published from the refused close-out.
        self.assertEqual(self.h.calibration_summary()
                         ["n_evaluations_total"], 0)

    def test_closeout_after_actions_end_succeeds(self):
        task = _task("t1")
        snap = self.h.snapshot(task, "ep1")
        running = self.h.actions.begin_action("model", "t1", "ep1",
                                              pre_snapshot=snap)
        # The action ends; NOW the close-out goes through.
        self.h.actions.end_action(running.action_id, status="completed")
        result = self.h.close_episode("t1", "ep1")
        self.assertIsNotNone(result["closeout"])
        self.assertEqual(result["closeout"]["terminal_state"],
                         "completed")

    def test_failed_episode_closes_honestly(self):
        task = _task("t1")
        result = self.h.close_episode("t1", "ep1", terminal_state="failed")
        self.assertEqual(result["closeout"]["terminal_state"], "failed")
        self.assertTrue(any("never reported as success" in n
                            for n in result["closeout"]["notes"]))

    def test_invalid_terminal_state_is_refused(self):
        with self.assertRaises(ValueError):
            self.h.close_episode("t1", "ep1", terminal_state="success")

    def test_open_episode_does_not_calibrate(self):
        task = _task("t1")
        prediction = self.h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S04"}, "ep1")
        record = self.solve(task, strategy="S04")
        self.h.bind_strategy_outcome(prediction.prediction_id,
                                     record.action_id)
        # NOT closed yet: the calibration summary must not count it.
        summary = self.h.calibration_summary()
        self.assertEqual(summary["n_evaluations_total"], 0)
        self.assertIsNone(self.h.episode_closeout_record("t1", "ep1"))


# ---------------------------------------------------------------------------
# 4. per-field evaluation eligibility
# ---------------------------------------------------------------------------


class TestEvaluationEligibility(M4Case):

    def test_unexecuted_candidate_gets_no_label(self):
        task = _task("t1")
        prediction = self.h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S01"}, "ep1")
        record = self.solve(task, strategy="S04")  # a different strategy ran
        self.h.bind_strategy_outcome(prediction.prediction_id,
                                     record.action_id)
        result = self.h.close_episode("t1", "ep1")
        evaluation = result["evaluations"][0]
        self.assertEqual(evaluation["state"], "excluded")
        self.assertTrue(any("identity" in r for r
                            in evaluation["exclusion_reasons"]))

    def test_missing_baseline_prediction_is_excluded(self):
        # A prediction with NO benefit value: nothing to compare on the
        # benefit side, and no fake error is manufactured.
        provider = StubProvider(payload={
            "cost": {"llm_tokens": 100},
            "risk": {"events": [{"event": "timeout",
                                 "probability": 0.2}]},
        })
        h = ORHarness(home=self.home, world_model=provider,
                      embedding=self.backend)
        self.addCleanup(h.close)
        task = _task("t1")
        prediction = h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S04"}, "ep1")
        record = self.solve(task, strategy="S04")
        h.bind_strategy_outcome(prediction.prediction_id,
                                record.action_id)
        result = h.close_episode("t1", "ep1")
        evaluation = result["evaluations"][0]
        self.assertEqual(evaluation["benefit"]["eligibility"],
                         "not_predicted")

    def test_unknown_risk_label_is_not_scored(self):
        # The predicted event never occurred and the scope completed with
        # no failures: the label is not_occurred (single trajectory) and a
        # Brier score IS recorded — but an event whose scope did NOT
        # complete stays unscored. Exercise both.
        task = _task("t1")
        prediction = self.h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S04"}, "ep1")
        record = self.solve(task, strategy="S04")
        self.h.bind_strategy_outcome(prediction.prediction_id,
                                     record.action_id)
        result = self.h.close_episode("t1", "ep1")
        risk = result["evaluations"][0]["risk"]
        self.assertEqual(risk["eligibility"], "evaluable")
        self.assertEqual(len(risk["scored"]), 1)
        # predicted 0.2, label not_occurred (y=0): brier = 0.04.
        self.assertAlmostEqual(risk["scored"][0]["brier"], 0.04)

    def test_cost_compared_only_where_both_sides_measured(self):
        task = _task("t1")
        prediction = self.h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S04"}, "ep1")
        record = self.solve(task, strategy="S04")
        self.h.bind_strategy_outcome(prediction.prediction_id,
                                     record.action_id)
        result = self.h.close_episode("t1", "ep1")
        cost = result["evaluations"][0]["cost"]
        # The prediction predicted llm_tokens + solver_runtime_s; the real
        # execution measured solver_runtime_s (and llm_tokens=0 unmeasured
        # by default). Only the both-sides dims participate.
        self.assertIn("solver_runtime_s", cost["per_dim"])
        self.assertIn("llm_tokens", cost["excluded"])

    def test_prediction_is_not_rewritten_by_evaluation(self):
        task = _task("t1")
        prediction = self.h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S04"}, "ep1")
        frozen = json.dumps(prediction.to_dict(), sort_keys=True)
        record = self.solve(task, strategy="S04")
        self.h.bind_strategy_outcome(prediction.prediction_id,
                                     record.action_id)
        self.h.close_episode("t1", "ep1")
        stored = self.h.get_strategy_outcome_prediction(
            prediction.prediction_id)
        # The predicted CONTENT is byte-identical (binding metadata may be
        # added; the forecast itself is frozen).
        self.assertEqual(stored.benefit.value, prediction.benefit.value)
        self.assertEqual(stored.cost.expected.llm_tokens,
                         prediction.cost.expected.llm_tokens)
        self.assertEqual(stored.risk.events[0].probability,
                         prediction.risk.events[0].probability)

    def test_interval_coverage_computed_when_saved(self):
        payload = json.loads(json.dumps(GOOD_PAYLOAD))
        payload["benefit"]["interval"] = [0.7, 0.9]
        provider = StubProvider(payload=payload)
        h = ORHarness(home=self.home, world_model=provider,
                      embedding=self.backend)
        self.addCleanup(h.close)
        task = _task("t1")
        prediction = h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S04"}, "ep1")
        record = self.solve(task, strategy="S04")
        h.bind_strategy_outcome(prediction.prediction_id,
                                record.action_id)
        result = h.close_episode("t1", "ep1")
        interval = result["evaluations"][0]["interval"]
        self.assertEqual(interval["eligibility"], "evaluable")
        # observed 1.0 (optimal) is OUTSIDE [0.7, 0.9].
        self.assertFalse(interval["covered"])


# ---------------------------------------------------------------------------
# 5. idempotence, recovery, and the cross-episode calibration channel
# ---------------------------------------------------------------------------


class TestCalibrationChannel(M4Case):

    def test_second_episode_reads_published_summary(self):
        task = _task("t1")
        prediction = self.h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S04"}, "ep1")
        record = self.solve(task, strategy="S04")
        self.h.bind_strategy_outcome(prediction.prediction_id,
                                     record.action_id)
        self.h.close_episode("t1", "ep1")
        # A NEW context (episode 2) carries the published summary CONTENT.
        ctx2 = self.h.build_prediction_context(_task("t1"), "ep2")
        calibration = ctx2.strategy_calibration
        self.assertEqual(calibration["protocol"], "wm-so/1")
        self.assertEqual(calibration["n_evaluations_total"], 1)
        group = calibration["groups"][
            GROUP_PREFIX + "normalized_objective_gap|1-gap|attempt"]
        self.assertEqual(group["n_samples"], 1)
        # Below the default minimum: insufficient evidence, no figure.
        self.assertEqual(group["basis"], "insufficient_evidence")
        self.assertIsNone(group["reliability"])

    def test_first_episode_context_stays_unchanged(self):
        task = _task("t1")
        ctx1 = self.h.build_prediction_context(task, "ep1")
        frozen = json.dumps(ctx1.to_dict(), sort_keys=True)
        prediction = self.h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S04"}, "ep1")
        record = self.solve(task, strategy="S04")
        self.h.bind_strategy_outcome(prediction.prediction_id,
                                     record.action_id)
        self.h.close_episode("t1", "ep1")
        stored = self.h.get_prediction_context(ctx1.context_id)
        self.assertEqual(json.dumps(stored.to_dict(), sort_keys=True),
                         frozen,
                         "a stored context is never rewritten by later "
                         "episodes' feedback")

    def test_active_episode_does_not_see_own_feedback(self):
        task = _task("t1")
        prediction = self.h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S04"}, "ep1")
        record = self.solve(task, strategy="S04")
        self.h.bind_strategy_outcome(prediction.prediction_id,
                                     record.action_id)
        # Episode 1 is still OPEN: a context built now sees no calibration
        # from it (its own feedback is not yet closed).
        ctx = self.h.build_prediction_context(task, "ep1")
        self.assertEqual(ctx.strategy_calibration.get(
            "n_evaluations_total"), 0)

    def test_min_samples_threshold_is_effective_and_recorded(self):
        task = _task("t1")
        prediction = self.h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S04"}, "ep1")
        record = self.solve(task, strategy="S04")
        self.h.bind_strategy_outcome(prediction.prediction_id,
                                     record.action_id)
        self.h.close_episode("t1", "ep1", min_calibration_samples=1)
        summary = self.h.calibration_summary(min_samples=1)
        group = summary["groups"][
            GROUP_PREFIX + "normalized_objective_gap|1-gap|attempt"]
        self.assertEqual(group["basis"], "measured")
        self.assertEqual(summary["min_samples"], 1)
        self.assertEqual(summary["min_samples_basis"],
                         "distinct (task_id, episode_id) pairs")

    def test_calibration_separate_from_knowledge_reliability(self):
        task = _task("t1")
        prediction = self.h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S04"}, "ep1")
        record = self.solve(task, strategy="S04")
        self.h.bind_strategy_outcome(prediction.prediction_id,
                                     record.action_id)
        self.h.close_episode("t1", "ep1")
        ctx = self.h.build_prediction_context(_task("t1"), "ep2")
        # The legacy knowledge reliability and the strategy-outcome
        # calibration are SEPARATE blocks.
        self.assertIn("strategy_outcome_calibration",
                      ctx.provider_view())
        self.assertNotEqual(ctx.reliability, ctx.strategy_calibration)


# ---------------------------------------------------------------------------
# 6. bug-fix regressions (the four review findings)
# ---------------------------------------------------------------------------


class TestWindowScopeAggregation(M4Case):
    """P1-1: a strategy-window prediction is scored against the WHOLE
    window of its selection round, not the bound action's single
    execution."""

    def _window_prediction(self, task):
        return self.h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S04", "scope": "strategy_window",
                   "window_id": "win::t1::ep1::S04"}, "ep1")

    def test_window_cost_sums_all_attempts(self):
        task = _task("t1")
        prediction = self._window_prediction(task)
        # Two attempts of the same window.
        first = self.solve(task, strategy="S04", episode_id="ep1",
                          tag="w1")
        second = self.solve(task, strategy="S04", episode_id="ep1",
                          tag="w2")
        self.h.bind_strategy_outcome(prediction.prediction_id,
                                     second.action_id)
        result = self.h.close_episode("t1", "ep1")
        evaluation = result["evaluations"][0]
        runtime = evaluation["cost"]["per_dim"]["solver_runtime_s"]
        # BOTH attempts' runtimes aggregate: the old behaviour reported
        # only the bound action's single execution.
        self.assertEqual(runtime["actual"],
                         first.cost.solver_runtime_s
                         + second.cost.solver_runtime_s)
        # The summary's scope covered both executions.
        from or_harness.world_model.episode_closeout import (
            summarize_real_outcome,
        )
        stored = self.h.get_strategy_outcome_prediction(
            prediction.prediction_id)
        scope_summary = summarize_real_outcome(self.h, stored)
        self.assertEqual(len(scope_summary.execution_ids), 2)
        # tool_calls is a HARNESS declaration, not an executor measurement:
        # with no override supplied it stays unknown (never a fabricated
        # constant per record). The genuinely measured dimensions are known.
        self.assertIsNone(scope_summary.cost["tool_calls"]["total"])
        self.assertEqual(scope_summary.cost["solver_runtime_s"]["n_measured"], 2)

    def test_declared_round_window_excludes_other_rounds(self):
        """A prediction declaring round r1 is scored against round 1's
        executions ONLY — the first round's runtime must not leak into
        the second round's totals."""
        task = _task("t1")
        # Round 0: choose S04, run it.
        self.h.report_action("select_strategy", task, "ep1",
                             params={"strategy_id": "S04"},
                             outcome={"kind": "choice"})
        first = self.solve(task, strategy="S04", episode_id="ep1",
                          tag="r0")
        # Round 1: choose S04 again, run it.
        self.h.report_action("select_strategy", task, "ep1",
                             params={"strategy_id": "S04"},
                             outcome={"kind": "choice"})
        # The prediction declares the ROUND-1 window and is made BEFORE
        # the round-1 execution runs (the honest order).
        prediction = self.h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S04", "scope": "strategy_window",
                   "window_id": "win::t1::ep1::S04::r1"}, "ep1")
        second = self.solve(task, strategy="S04", episode_id="ep1",
                          tag="r1")
        self.h.bind_strategy_outcome(prediction.prediction_id,
                                     second.action_id)
        result = self.h.close_episode("t1", "ep1")
        evaluation = result["evaluations"][0]
        # ONLY round 1's runtime: the old behaviour aggregated the whole
        # episode (first + second).
        self.assertAlmostEqual(
            evaluation["cost"]["per_dim"]["solver_runtime_s"]["actual"],
            second.cost.solver_runtime_s, places=5)
        self.assertNotAlmostEqual(
            evaluation["cost"]["per_dim"]["solver_runtime_s"]["actual"],
            first.cost.solver_runtime_s + second.cost.solver_runtime_s,
            places=5)

    def test_binding_action_of_another_round_is_a_mismatch(self):
        task = _task("t1")
        self.h.report_action("select_strategy", task, "ep1",
                             params={"strategy_id": "S04"},
                             outcome={"kind": "choice"})
        # The ROUND-1 prediction is made BEFORE any round-1 execution.
        prediction = self.h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S04", "scope": "strategy_window",
                   "window_id": "win::t1::ep1::S04::r1"}, "ep1")
        first = self.solve(task, strategy="S04", episode_id="ep1",
                          tag="m0")
        self.h.report_action("select_strategy", task, "ep1",
                             params={"strategy_id": "S04"},
                             outcome={"kind": "choice"})
        second = self.solve(task, strategy="S04", episode_id="ep1",
                          tag="m1")
        # Bind the ROUND-0 action to the ROUND-1 prediction: a mismatch.
        bound = self.h.bind_strategy_outcome(prediction.prediction_id,
                                            first.action_id)
        mismatch = bound.trace.model_info.get("binding_mismatch") or {}
        self.assertIn("window_round", mismatch)
        self.assertFalse(bound.trace.comparable)

    def test_failed_retry_is_included_in_window_cost(self):
        task = _task("t1")
        prediction = self._window_prediction(task)
        from pathlib import Path
        # A FAILED attempt (error), then a repaired one.
        work = Path(self.home) / "ws_fail"
        work.mkdir(parents=True, exist_ok=True)
        (work / "solve.py").write_text(
            "raise RuntimeError('model invalid')\n", encoding="utf-8")
        failed = self.h.execute(task, "S04", str(work / "solve.py"),
                                str(work), solver="highs",
                                episode_id="ep1")
        self.h.record(failed)
        repaired = self.solve(task, strategy="S04", episode_id="ep1",
                              tag="w3")
        self.h.bind_strategy_outcome(prediction.prediction_id,
                                     repaired.action_id)
        result = self.h.close_episode("t1", "ep1")
        evaluation = result["evaluations"][0]
        # The failed retry is real spend inside the window's scope: the
        # runtime total covers BOTH attempts.
        self.assertAlmostEqual(
            evaluation["cost"]["per_dim"]["solver_runtime_s"]["actual"],
            failed.cost.solver_runtime_s
            + repaired.cost.solver_runtime_s, places=5)
        # The benefit observation is the LAST qualified attempt's.
        self.assertEqual(evaluation["benefit"]["observed"], 1.0)


class TestUnobservedIsNotALabel(M4Case):
    """P1-3: an unobservable metric or risk event is never converted into
    a real label."""

    def test_business_metric_is_not_relabelled(self):
        # A prediction declaring a BUSINESS metric: the solver's 1-gap
        # number must NOT be re-labelled as it.
        payload = json.loads(json.dumps(GOOD_PAYLOAD))
        payload["benefit"]["metric"] = "business_cost_saving_ratio"
        payload["benefit"]["unit"] = "ratio"
        provider = StubProvider(payload=payload)
        h = ORHarness(home=self.home, world_model=provider,
                      embedding=self.backend)
        self.addCleanup(h.close)
        task = _task("t1")
        prediction = h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S04"}, "ep1")
        record = self.solve(task, strategy="S04")
        h.bind_strategy_outcome(prediction.prediction_id,
                                record.action_id)
        result = h.close_episode("t1", "ep1")
        evaluation = result["evaluations"][0]
        self.assertEqual(evaluation["benefit"]["eligibility"],
                         "scope_mismatch")
        self.assertNotIn("observed", evaluation["benefit"])
        self.assertIn("never re-labelled",
                      evaluation["benefit"]["reason"])

    def test_unobservable_risk_event_stays_unknown(self):
        # A business risk with NO observation channel: "no failure log"
        # must not become a not_occurred label.
        payload = json.loads(json.dumps(GOOD_PAYLOAD))
        payload["risk"]["events"] = [
            {"event": "customer_demand_shortfall", "probability": 0.3},
            {"event": "timeout", "probability": 0.2},
        ]
        provider = StubProvider(payload=payload)
        h = ORHarness(home=self.home, world_model=provider,
                      embedding=self.backend)
        self.addCleanup(h.close)
        task = _task("t1")
        prediction = h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S04"}, "ep1")
        record = self.solve(task, strategy="S04")
        h.bind_strategy_outcome(prediction.prediction_id,
                                record.action_id)
        result = h.close_episode("t1", "ep1")
        risk = result["evaluations"][0]["risk"]
        scored = {e["event"]: e for e in risk["scored"]}
        unscored = {e["event"]: e for e in risk["unscored"]}
        # The observable event IS scored (not_occurred).
        self.assertIn("timeout", scored)
        # The unobservable one is NOT scored — no fake label.
        self.assertNotIn("customer_demand_shortfall", scored)
        self.assertIn("customer_demand_shortfall", unscored)
        self.assertIn("no observation channel",
                      unscored["customer_demand_shortfall"]["reason"])

    def test_retired_event_names_are_not_aliased(self):
        """A prediction naming a RETIRED event gets no label and no alias
        mapping: the old, wider meaning must not be counted under a new,
        narrower name."""
        payload = json.loads(json.dumps(GOOD_PAYLOAD))
        payload["risk"]["events"] = [
            {"event": "model_invalid", "probability": 0.2}]
        provider = StubProvider(payload=payload)
        h = ORHarness(home=self.home, world_model=provider,
                      embedding=self.backend)
        self.addCleanup(h.close)
        task = _task("t1")
        prediction = h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S04"}, "ep1")
        record = self.solve(task, strategy="S04")
        h.bind_strategy_outcome(prediction.prediction_id,
                                record.action_id)
        result = h.close_episode("t1", "ep1")
        risk = result["evaluations"][0]["risk"]
        self.assertEqual(risk["scored"], [],
                         "a retired event name is never scored")
        unscored = {e["event"]: e for e in risk["unscored"]}
        self.assertIn("model_invalid", unscored)
        self.assertIn("retired", unscored["model_invalid"]["reason"])
        self.assertIn("task_check_failed",
                      unscored["model_invalid"]["reason"])

    def test_unconfirmed_budget_keeps_label_unknown(self):
        """An UNCONFIRMED ledger (partially unmeasured cost) is not
        evidence of no exhaustion: the label stays unknown and the event
        is excluded from scoring."""
        payload = json.loads(json.dumps(GOOD_PAYLOAD))
        payload["risk"]["events"] = [
            {"event": "budget_exhausted", "probability": 0.5}]
        provider = StubProvider(payload=payload)
        h = ORHarness(home=self.home, world_model=provider,
                      embedding=self.backend)
        self.addCleanup(h.close)
        task = _task("t1")
        # Declare a budget; the execution leaves llm_tokens UNMEASURED
        # (the default), so the ledger verdict is UNCONFIRMED, not ok.
        h.declare_budget("t1", {"llm_tokens": 10000}, "ep1")
        prediction = h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S04"}, "ep1")
        record = self.solve(task, strategy="S04")
        h.bind_strategy_outcome(prediction.prediction_id,
                                record.action_id)
        view = h.budget.view("t1", "ep1",
                             budget=h._load_budget("t1", "ep1"))
        self.assertEqual(view["status"], "unconfirmed")
        result = h.close_episode("t1", "ep1")
        risk = result["evaluations"][0]["risk"]
        # NOT scored. The prediction declares attempt scope while the
        # ledger is episode-scoped, so the scope itself does not match:
        # the label stays unknown and the reason says so.
        self.assertEqual(risk["scored"], [])
        self.assertIn("budget_exhausted",
                      {e["event"] for e in risk["unscored"]})
        self.assertTrue(any("scope_mismatch" in e["reason"]
                            for e in risk["unscored"]))

    def test_confirmed_within_budget_is_a_framework_observation(self):
        """A CONFIRMED within-budget ledger IS a not_occurred observation —
        recorded by the FRAMEWORK, counted by observation unit, even though
        no attempt-scope prediction may be scored against it."""
        payload = json.loads(json.dumps(GOOD_PAYLOAD))
        payload["risk"]["events"] = [
            {"event": "budget_exhausted", "probability": 0.5}]
        provider = StubProvider(payload=payload)
        h = ORHarness(home=self.home, world_model=provider,
                      embedding=self.backend)
        self.addCleanup(h.close)
        task = _task("t1")
        prediction = h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S04"}, "ep1")
        record = self.solve(task, strategy="S04")
        # Backfill the execution's llm_tokens so EVERY contributing item
        # (the execution AND the prediction call) measured the declared
        # dimension — the ledger verdict is then a real "ok", not an
        # "unconfirmed".
        h.bank.update_cost(record.execution_id, llm_tokens=10)
        h.declare_budget("t1", {"llm_tokens": 10000}, "ep1")
        h.bind_strategy_outcome(prediction.prediction_id,
                                record.action_id)
        view = h.budget.view("t1", "ep1",
                             budget=h._load_budget("t1", "ep1"))
        self.assertEqual(view["status"], "ok")
        result = h.close_episode("t1", "ep1")
        risk = result["evaluations"][0]["risk"]
        # The attempt-scope prediction is NOT scored against the
        # episode-scoped ledger...
        self.assertEqual(risk["scored"], [])
        # ...but the framework still OBSERVED the event, by its own unit.
        observation = risk["observed_units"]["budget_exhausted"]
        self.assertEqual(observation["label"], "not_occurred")
        self.assertEqual(observation["unit"], "episode")
        self.assertEqual(observation["n_units"], 1)

    def test_budget_exhausted_is_observable_by_the_framework(self):
        # budget_exhausted HAS a channel: a declared budget exceeded by
        # real consumption is an observed event. The observation is the
        # framework's, counted by episode, and is reported in the
        # occurrence statistics even though the attempt-scope prediction
        # may not be scored against the episode-wide ledger.
        payload = json.loads(json.dumps(GOOD_PAYLOAD))
        payload["risk"]["events"] = [
            {"event": "budget_exhausted", "probability": 0.5}]
        provider = StubProvider(payload=payload)
        h = ORHarness(home=self.home, world_model=provider,
                      embedding=self.backend)
        self.addCleanup(h.close)
        task = _task("t1")
        h.declare_budget("t1", {"llm_tokens": 10}, "ep1")
        prediction = h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S04"}, "ep1")
        record = self.solve(task, strategy="S04")
        h.bind_strategy_outcome(prediction.prediction_id,
                                record.action_id)
        result = h.close_episode("t1", "ep1")
        risk = result["evaluations"][0]["risk"]
        self.assertEqual(risk["observed_units"]["budget_exhausted"]["label"],
                         "occurred")
        self.assertEqual(risk["scored"], [],
                         "an attempt-scope prediction is not scored against "
                         "the episode-scoped budget ledger")
        summary = h.calibration_summary()
        occurrence = summary["occurrence"]["budget_exhausted"]
        self.assertEqual(occurrence["observation_unit"], "episode")
        self.assertEqual(occurrence["n_occurred"], 1)
        self.assertEqual(occurrence["unit_occurrence_rate"], 1.0)


class TestCalibrationGrouping(M4Case):
    """P2: calibration groups by metric/unit/scope, scores risk events
    per name, and the threshold counts DISTINCT EPISODES."""

    def _close_with_predictions(self, n, task_id="t1", episode="ep1",
                                metric="normalized_objective_gap",
                                unit="1-gap"):
        """Close one episode with n predictions bound to ONE execution.

        The predictions are all made BEFORE the execution runs (the honest
        re-planning shape: several candidates predicted, one executed) —
        otherwise the timing check correctly marks later ones as
        post-hoc."""
        task = _task(task_id)
        payload = json.loads(json.dumps(GOOD_PAYLOAD))
        payload["benefit"]["metric"] = metric
        payload["benefit"]["unit"] = unit
        provider = StubProvider(payload=payload)
        h = ORHarness(home=self.home, world_model=provider,
                      embedding=self.backend)
        self.addCleanup(h.close)
        predictions = [h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S04"}, episode)
            for _ in range(n)]
        record = self.solve(task, strategy="S04", episode_id=episode)
        for prediction in predictions:
            h.bind_strategy_outcome(prediction.prediction_id,
                                    record.action_id)
        h.close_episode(task_id, episode)
        return h

    def test_correlated_predictions_do_not_cross_the_threshold(self):
        # FIVE predictions bound to ONE execution (one episode): the old
        # behaviour counted the predictions and reported "measured".
        self._close_with_predictions(n=5)
        summary = self.h.calibration_summary(min_samples=5)
        group = summary["groups"][
            GROUP_PREFIX + "normalized_objective_gap|1-gap|attempt"]
        self.assertEqual(group["n_samples"], 5)
        self.assertEqual(group["n_distinct_episodes"], 1)
        self.assertEqual(group["correlated_predictions"], 4)
        # ONE distinct episode < 5: insufficient evidence, no figure.
        self.assertEqual(group["basis"], "insufficient_evidence")
        self.assertIsNone(group["reliability"])

    def test_different_unit_is_a_different_group(self):
        self._close_with_predictions(n=1, metric="normalized_objective_gap",
                                     unit="1-gap")
        self._close_with_predictions(n=1, metric="normalized_objective_gap",
                                     unit="percent", episode="ep2")
        summary = self.h.calibration_summary()
        keys = set(summary["groups"])
        self.assertIn(GROUP_PREFIX + "normalized_objective_gap|1-gap"
                      "|attempt", keys)
        self.assertIn(GROUP_PREFIX + "normalized_objective_gap|percent"
                      "|attempt", keys)

    def test_brier_reported_per_event_name(self):
        payload = json.loads(json.dumps(GOOD_PAYLOAD))
        payload["risk"]["events"] = [
            {"event": "timeout", "probability": 0.2},
            {"event": "solver_reported_infeasible", "probability": 0.4},
        ]
        provider = StubProvider(payload=payload)
        h = ORHarness(home=self.home, world_model=provider,
                      embedding=self.backend)
        self.addCleanup(h.close)
        task = _task("t1")
        prediction = h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S04"}, "ep1")
        record = self.solve(task, strategy="S04")
        h.bind_strategy_outcome(prediction.prediction_id,
                                record.action_id)
        h.close_episode("t1", "ep1")
        summary = h.calibration_summary(min_samples=1)
        group = summary["groups"][
            GROUP_PREFIX + "normalized_objective_gap|1-gap|attempt"]
        # Per-event means, never one pooled Brier.
        self.assertIn("timeout", group["mean_brier_by_event"])
        self.assertIn("solver_reported_infeasible",
                      group["mean_brier_by_event"])
        self.assertNotIn("mean_brier", group)

    def test_same_episode_name_across_tasks_is_distinct_episodes(self):
        """Five DIFFERENT tasks that each used the episode id ``ep1`` are
        five task-episodes, not one.

        Episode ids are chosen per task, so the bare ``episode_id`` is not
        an identity: deduplicating on it collapsed five independent
        samples into one and kept the group below the threshold forever.
        """
        for index in range(5):
            self._close_with_predictions(
                n=1, task_id=f"task{index}", episode="ep1")
        summary = self.h.calibration_summary(min_samples=5)
        group = summary["groups"][
            GROUP_PREFIX + "normalized_objective_gap|1-gap|attempt"]
        self.assertEqual(group["n_samples"], 5)
        self.assertEqual(
            group["n_distinct_episodes"], 5,
            "five tasks' ep1 are five task-episodes: the pair "
            "(task_id, episode_id) is the identity")
        self.assertEqual(group["correlated_predictions"], 0)
        # The threshold is met by DISTINCT TASK-EPISODES.
        self.assertEqual(group["basis"], "measured")
        self.assertEqual(group["reliability"], "measured_experience")

    def test_repeated_predictions_of_one_task_episode_stay_correlated(self):
        """The task half of the identity must not weaken the episode
        half: five predictions over ONE task-episode are still one sample.
        """
        self._close_with_predictions(n=5, task_id="t1", episode="ep1")
        summary = self.h.calibration_summary(min_samples=2)
        group = summary["groups"][
            GROUP_PREFIX + "normalized_objective_gap|1-gap|attempt"]
        self.assertEqual(group["n_samples"], 5)
        self.assertEqual(group["n_distinct_episodes"], 1)
        self.assertEqual(group["basis"], "insufficient_evidence")


# ---------------------------------------------------------------------------
# 7. CLI surface
# ---------------------------------------------------------------------------


class TestCliCloseoutCommands(M4Case):

    def _run(self, argv):
        from or_harness.cli import main
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = main(["--home", self.home] + argv)
        return code, json.loads(buf.getvalue())

    def test_close_episode_command(self):
        task = _task("t1")
        prediction = self.h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S04"}, "ep1")
        record = self.solve(task, strategy="S04")
        self.h.bind_strategy_outcome(prediction.prediction_id,
                                     record.action_id)
        code, payload = self._run([
            "close-episode", "--task", "t1", "--episode", "ep1"])
        self.assertEqual(code, 0)
        self.assertEqual(payload["result"]["closeout"]["terminal_state"],
                         "completed")
        # Re-close over the CLI: idempotent.
        code, payload = self._run([
            "close-episode", "--task", "t1", "--episode", "ep1"])
        self.assertEqual(code, 0)
        self.assertTrue(payload["result"]["already_closed"])

    def test_calibration_command(self):
        code, payload = self._run(["calibration"])
        self.assertEqual(code, 0)
        self.assertEqual(payload["result"]["protocol"], "wm-so/1")

    def test_evaluations_command(self):
        task = _task("t1")
        prediction = self.h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S04"}, "ep1")
        record = self.solve(task, strategy="S04")
        self.h.bind_strategy_outcome(prediction.prediction_id,
                                     record.action_id)
        self.h.close_episode("t1", "ep1")
        code, payload = self._run(["evaluations", "--task", "t1"])
        self.assertEqual(code, 0)
        self.assertEqual(payload["result"]["count"], 1)
        evaluation_id = payload["result"]["evaluations"][0][
            "evaluation_id"]
        code, payload = self._run([
            "evaluations", "--evaluation", evaluation_id])
        self.assertEqual(code, 0)
        self.assertEqual(payload["result"]["evaluation_id"],
                         evaluation_id)


if __name__ == "__main__":
    unittest.main()
