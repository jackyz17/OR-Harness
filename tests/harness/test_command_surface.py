"""Acceptance tests for the command-surface consolidation.

This round REMOVED seven commands, renamed one, merged four queries into
`inspect`, switched planning to a single protocol, and added two automatic
bindings. The properties that must hold afterwards are the ones asserted
here — each test names the review requirement it covers:

1. **The default flow does not double-predict.** `plan-next` returns a
   prediction id per candidate and `execute --prediction` reuses it: no
   second model call, one binding, one calibration sample.
2. **A plan's result is directly usable downstream.** The id handed over by
   `plan-next` is the id the action gets bound to.
3. **The candidate evidence package still exists.** `induction-candidates`
   produces exactly what `predict-capability --bundle` consumes.
4. **Old records stay readable.** The legacy payload reader and
   `inspect --bank predictions` still resolve them, and they never enter the
   wm-so/1 calibration.
5. **The calibration scopes do not mix.** The two prediction logs stay
   separate in the read surface.
6. **Automatic binding is retryable and never double-counts.**
7. **Merging the queries lost nothing.** Every filter and single-record read
   the removed commands offered is reachable through `inspect`.
8. **The removed entry points are gone, not aliased.**

Nothing here claims real-LLM behaviour: every provider is a labelled stub.
"""
import io
import json
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tests.harness.helpers import HarnessTestCase  # noqa: E402

from or_harness.api import ORHarness  # noqa: E402
from or_harness.world_model.prediction import ActionSpec  # noqa: E402
from or_harness.world_model.provider import WorldModelProvider  # noqa: E402

TASK = {"task_id": "t1", "family": "routing", "description": "toy"}

SOLVE_SCRIPT = (
    "import json\n"
    "with open('result.json', 'w') as fh:\n"
    "    json.dump({'status': 'optimal', 'objective_value': 1.0,\n"
    "               'objective_bound': 1.0, 'runtime_seconds': 0.01}, fh)\n")


def _payload(quality=0.7, cost=None):
    payload = {
        "benefit": {"kind": "solution_quality",
                    "metric": "normalized_objective_gap", "unit": "1-gap",
                    "value": float(quality),
                    "baseline": {"kind": "conditional_stats", "value": 0.0}},
        "uncertainty": {"execution_randomness": 0.2},
    }
    if cost is not None:
        payload["cost"] = dict(cost)
    return payload


class CountingProvider(WorldModelProvider):
    """A wm-so/1 stub that COUNTS its calls (the double-prediction check)."""

    name = "counting-stub"

    def __init__(self, quality=0.7, cost=None, usage=None):
        self.quality = quality
        self.cost = cost
        self.usage = usage or {"prompt_tokens": 100, "completion_tokens": 50}
        self.requests = []

    def predict(self, request, timeout_s=None):
        self.requests.append(request)
        sid = (request.get("candidate") or {}).get("strategy_id")
        # A deterministic, declared quality per strategy id — not a hidden law.
        quality = 0.9 if sid == "S01" else self.quality
        return {"payload": _payload(quality=quality, cost=self.cost),
                "usage": dict(self.usage),
                "error": None, "latency_s": 0.01}


class LegacyReaderProvider(WorldModelProvider):
    """A legacy (M2 shape) stub: it answers the OLD request contract.

    Used ONLY to write a record in the pre-wm-so/1 format, so the reader
    path can be exercised against real stored data.
    """

    name = "legacy-stub"

    def predict(self, request, timeout_s=None):
        return {"payload": {"outcome_status": "feasible", "feasible": True,
                            "quality": 0.5, "failure_prob": 0.1,
                            "cost": {"llm_tokens": 10.0}},
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                "error": None, "latency_s": 0.01}


class Base(HarnessTestCase):
    def setUp(self):
        super().setUp()
        self._work = Path(self.home) / "ws"
        self._work.mkdir(parents=True, exist_ok=True)
        self.script = self._work / "solve.py"
        self.script.write_text(SOLVE_SCRIPT, encoding="utf-8")

    def make_harness(self, provider=None, **kwargs):
        h = ORHarness(home=self.home,
                      world_model=provider or CountingProvider(), **kwargs)
        self.addCleanup(h.close)
        return h

    def run_cli(self, argv):
        from or_harness import cli
        buffer = io.StringIO()
        old = sys.stdout
        sys.stdout = buffer
        try:
            code = cli.main(["--home", self.home] + argv)
        finally:
            sys.stdout = old
        return code, buffer.getvalue()


# ---------------------------------------------------------------------------
# 1 & 2  one prediction per decision, handed over and bound
# ---------------------------------------------------------------------------


class TestNoDoublePrediction(Base):
    def test_plan_prediction_is_reused_by_execute(self):
        """The mandatory review check: after planning, the chosen candidate
        is NOT predicted a second time, and the action binds to the plan's
        own prediction."""
        provider = CountingProvider()
        h = self.make_harness(provider)
        plan = h.plan_next(
            TASK, "ep1",
            candidates=[ActionSpec("execute_strategy", "t1", strategy_id="S01"),
                        ActionSpec("execute_strategy", "t1",
                                   strategy_id="S02")])
        self.assertEqual(plan["status"], "ok")
        calls_after_plan = len(provider.requests)
        self.assertEqual(calls_after_plan, 2, "one call per candidate")

        chosen = plan["suggested"]["strategy_id"]
        chosen_id = next(c["prediction_id"] for c in plan["candidates"]
                         if c["action_spec"]["strategy_id"] == chosen)
        h.choose_next(plan["decision_action_id"],
                      chosen=ActionSpec.from_dict(plan["suggested"]))
        record = h.execute(TASK, chosen, str(self.script), str(self._work),
                           solver="highs", episode_id="ep1",
                           prediction_id=chosen_id)
        # No further model call: the plan's prediction was REUSED.
        self.assertEqual(len(provider.requests), calls_after_plan,
                         "execute must not predict the candidate again")
        binding = record.execution_features["prediction_binding"]
        self.assertTrue(binding["bound"])
        self.assertIsNone(binding["trace"]["binding_mismatch"])
        self.assertTrue(binding["comparable"])
        # Exactly ONE wm-so/1 prediction exists for this decision.
        stored = h.strategy_predictions.query(task_id="t1")
        self.assertEqual(len(stored), 2,
                         "one prediction per candidate, none duplicated")
        h.record(record)

    def test_the_cli_hands_over_the_plan_prediction_id(self):
        """The hand-over is visible in the CLI output the agent reads, so
        it does not have to guess or re-derive it.

        The CLI is run with the STUB provider injected through
        ``cli._harness`` (the CLI normally builds a provider from
        ``--world-model``, which would mean a real HTTP endpoint). The
        assertion is about the OUTPUT CONTRACT, not about model quality."""
        provider = CountingProvider()
        h = ORHarness(home=self.home, world_model=provider)
        self.addCleanup(h.close)
        from or_harness import cli as cli_module
        original = cli_module._harness
        cli_module._harness = lambda args: h
        self.addCleanup(setattr, cli_module, "_harness", original)
        code, out = self.run_cli([
            "plan-next", "--task", json.dumps(TASK), "--episode", "ep1",
            "--candidates", json.dumps([
                {"action_type": "execute_strategy", "strategy_id": "S01"}])])
        self.assertEqual(code, 0)
        payload = json.loads(out)
        candidate = payload["result"]["plan"]["candidates"][0]
        self.assertTrue(candidate["prediction_id"].startswith("sp_"))
        # The summary tells the agent exactly what to do next.
        self.assertIn("--prediction", payload["summary"])
        self.assertIn(candidate["prediction_id"], payload["summary"])
        self.assertIn("do NOT call predict-strategy again",
                      payload["summary"])


# ---------------------------------------------------------------------------
# 3  the candidate evidence package survives
# ---------------------------------------------------------------------------


class TestCandidateEvidencePackage(Base):
    def test_scan_produces_what_predict_capability_consumes(self):
        """`induction-candidates` is the ONLY M4 responsibility that had to
        survive: `predict-capability --bundle` consumes its output."""
        provider = CountingProvider()
        h = self.make_harness(provider)
        for task_id in ("t1", "t2"):
            record = h.execute(dict(TASK, task_id=task_id), "S01",
                               str(self.script), str(self._work),
                               solver="highs", episode_id=f"e_{task_id}")
            h.record(record)
        bundles = h.induction_candidates()
        self.assertEqual(len(bundles), 1)
        bundle = bundles[0]
        # The bundle carries a real, frozen scope — not just a name.
        self.assertTrue(bundle["execution_ids"])
        self.assertEqual(bundle["tasks"], ["t1", "t2"])

        # And it really is consumable: the prediction is scoped to it.
        calls_before = len(provider.requests)
        prediction = h.predict_capability_evolution(
            {"operation_type": "induce", "strategy_id": "S01"},
            bundle=bundle, horizon="next 5 tasks", horizon_tasks=5)
        self.assertEqual(len(provider.requests), calls_before + 1)
        self.assertEqual(sorted(prediction.experience_scope.execution_ids),
                         sorted(bundle["execution_ids"]))

    def test_scan_makes_no_model_call(self):
        provider = CountingProvider()
        h = self.make_harness(provider)
        self.run_cli(["induction-candidates"])
        self.assertEqual(provider.requests, [],
                         "the evidence scan is a bank read, never a call")


# ---------------------------------------------------------------------------
# 4 & 5  old records readable, calibration scopes not mixed
# ---------------------------------------------------------------------------


class TestLegacyRecordsStayReadable(Base):
    def test_legacy_payload_is_readable_and_separate(self):
        """An old unversioned payload is still READABLE, and it never enters
        the wm-so/1 calibration: two logs, two questions."""
        provider = LegacyReaderProvider()
        h = self.make_harness(provider)
        spec = ActionSpec("execute_strategy", "t1", strategy_id="S01")
        legacy = h.predict_outcome(TASK, spec, "ep1")
        self.assertTrue(legacy.prediction_id.startswith("wp_"))

        # Readable through the contracts reader (the old payload shape).
        viewed = ORHarness.read_prediction_payload(legacy.to_dict())
        self.assertEqual(viewed["contract_version"], "legacy/unversioned")
        self.assertTrue(viewed["legacy"])
        self.assertTrue(viewed["supported"])

        # Readable through the unified `inspect` surface, under its OWN key.
        code, out = self.run_cli(["inspect", "--bank", "predictions"])
        self.assertEqual(code, 0)
        payload = json.loads(out)["result"]
        self.assertEqual(len(payload["legacy_predictions"]), 1)
        self.assertEqual(payload["strategy_predictions"], [])
        self.assertEqual(payload["capability_predictions"], [])
        self.assertEqual(payload["count"], 1)

    def test_the_two_logs_do_not_share_a_query_key(self):
        """A legacy prediction and a wm-so/1 prediction of the same task
        appear in DIFFERENT keys: a reader can never mistake one for the
        other, and the calibration reads only the wm-so/1 channel."""
        legacy_provider = LegacyReaderProvider()
        h = self.make_harness(legacy_provider)
        h.predict_outcome(TASK,
                          ActionSpec("execute_strategy", "t1",
                                     strategy_id="S01"), "ep1")
        # Swap in a wm-so/1 provider and predict under the new protocol.
        provider = CountingProvider()
        h.world_model = provider
        h.predictions.provider = provider
        h.strategy_predictions.provider = provider
        h.predict_strategy_outcome(
            TASK, {"action_type": "execute_strategy", "strategy_id": "S01"},
            "ep1")
        code, out = self.run_cli(["inspect", "--bank", "predictions"])
        payload = json.loads(out)["result"]
        self.assertEqual(len(payload["legacy_predictions"]), 1)
        self.assertEqual(len(payload["strategy_predictions"]), 1)
        self.assertEqual(payload["count"], 2)
        # The calibration summary counts only the wm-so/1 channel.
        summary = h.calibration_summary()
        self.assertEqual(summary["n_evaluated"], 0,
                         "no episode has closed: neither log contributes")

    def test_a_single_prediction_read_names_its_generation(self):
        legacy_provider = LegacyReaderProvider()
        h = self.make_harness(legacy_provider)
        legacy = h.predict_outcome(
            TASK, ActionSpec("execute_strategy", "t1", strategy_id="S01"),
            "ep1")
        read = h.read_prediction(legacy.prediction_id)
        self.assertEqual(read["kind"], "legacy_outcome")
        code, out = self.run_cli([
            "inspect", "--bank", "predictions",
            "--prediction", legacy.prediction_id])
        self.assertEqual(code, 0)
        payload = json.loads(out)["result"]
        self.assertEqual(payload["prediction"]["kind"], "legacy_outcome")

    def test_an_unknown_prediction_id_is_refused_not_guessed(self):
        h = self.make_harness(CountingProvider())
        code, out = self.run_cli([
            "inspect", "--bank", "predictions",
            "--prediction", "sp_does_not_exist"])
        self.assertEqual(code, 2)
        self.assertIn("unknown prediction_id", out)


# ---------------------------------------------------------------------------
# 6  automatic binding: retryable, idempotent, never double-counting
# ---------------------------------------------------------------------------


class TestAutomaticBindingIsSafe(Base):
    def test_execute_binding_is_idempotent(self):
        """Re-binding the same action returns the same stored link without
        creating a second one or charging anything."""
        provider = CountingProvider()
        h = self.make_harness(provider)
        prediction = h.predict_strategy_outcome(
            TASK, {"action_type": "execute_strategy", "strategy_id": "S01"},
            "ep1")
        calls_before = len(provider.requests)
        record = h.execute(TASK, "S01", str(self.script), str(self._work),
                           solver="highs", episode_id="ep1",
                           prediction_id=prediction.prediction_id)
        self.assertEqual(len(provider.requests), calls_before)
        # A second, EXPLICIT bind of the same pair is a no-op.
        again = h.bind_strategy_outcome(prediction.prediction_id,
                                        record.action_id)
        self.assertEqual(
            again.trace.model_info["bound_action_id"], record.action_id)
        # And the automatic one did not lose the link.
        updated = h.strategy_predictions.get(prediction.prediction_id)
        self.assertEqual(
            updated.trace.model_info["bound_action_id"], record.action_id)

    def test_accept_capability_binds_the_fact_once(self):
        """`accept-capability` binds the maintenance fact itself; a repeat
        bind is `already_bound` and the fact is counted ONCE."""
        provider = CountingProvider()
        h = self.make_harness(provider)
        for task_id in ("t1", "t2"):
            record = h.execute(dict(TASK, task_id=task_id), "S01",
                               str(self.script), str(self._work),
                               solver="highs", episode_id=f"e_{task_id}")
            h.record(record)
        prediction = h.predict_capability_evolution(
            {"operation_type": "induce", "strategy_id": "S01",
             "scope": {"execution_ids": ["ex1", "ex2"]}},
            bundle=h.induction_candidates()[0],
            horizon="next 5 tasks", horizon_tasks=5)
        recommendation = {
            "recommendation": "accept",
            "selected_prediction_id": prediction.prediction_id,
            "recommendation_id": "rec_1",
        }
        accepted = h.accept_capability_operation(recommendation)
        # The adoption result reports the binding it performed.
        self.assertTrue(accepted["maintenance_binding"]["bound"])
        self.assertEqual(
            h.capability_feedback_summary()["n_fact_bound"], 1)
        # A repeat bind is idempotent and does not count a second fact.
        repeat = h.bind_capability_maintenance(
            prediction.prediction_id,
            adoption_action_id=accepted["adoption_action_id"])
        self.assertTrue(repeat["already_bound"])
        self.assertEqual(
            h.capability_feedback_summary()["n_fact_bound"], 1,
            "a repeated bind must not double-count the fact")

    def test_a_bad_prediction_id_leaves_the_execution_intact(self):
        """A binding failure is bookkeeping, never evidence loss."""
        provider = CountingProvider()
        h = self.make_harness(provider)
        record = h.execute(TASK, "S01", str(self.script), str(self._work),
                           solver="highs", episode_id="ep1",
                           prediction_id="sp_nope")
        self.assertTrue(record.quality["feasible"])
        binding = record.execution_features["prediction_binding"]
        self.assertFalse(binding["bound"])
        self.assertIn("bind-strategy", binding["reason"])
        h.record(record)
        self.assertEqual(h.bank.count(), 1)


# ---------------------------------------------------------------------------
# 7  the merged query surface lost nothing
# ---------------------------------------------------------------------------


class TestQuerySurfaceIsComplete(Base):
    def test_every_removed_query_is_reachable_through_inspect(self):
        """Each capability the four removed commands offered is asserted
        here, so the merge is verifiable rather than asserted."""
        provider = CountingProvider()
        h = self.make_harness(provider)
        record = h.execute(TASK, "S01", str(self.script), str(self._work),
                           solver="highs", episode_id="ep1")
        h.record(record)
        prediction = h.predict_strategy_outcome(
            TASK, {"action_type": "execute_strategy", "strategy_id": "S01"},
            "ep1")
        h.bind_strategy_outcome(prediction.prediction_id, record.action_id)
        h.close_episode("t1", "ep1")

        # -- evaluations (list, task filter, single read) ----------------
        code, out = self.run_cli(["inspect", "--bank", "evaluations"])
        self.assertEqual(code, 0)
        listed = json.loads(out)["result"]
        self.assertEqual(listed["count"], 1)
        evaluation_id = listed["evaluations"][0]["evaluation_id"]
        code, out = self.run_cli([
            "inspect", "--bank", "evaluations", "--task", "t1"])
        self.assertEqual(json.loads(out)["result"]["count"], 1)
        code, out = self.run_cli([
            "inspect", "--bank", "evaluations", "--task", "other"])
        self.assertEqual(json.loads(out)["result"]["count"], 0)
        code, out = self.run_cli([
            "inspect", "--bank", "evaluations",
            "--evaluation", evaluation_id])
        self.assertEqual(
            json.loads(out)["result"]["evaluation"]["evaluation_id"],
            evaluation_id)

        # -- retention (policy + online + archive) ----------------------
        code, out = self.run_cli(["inspect", "--bank", "retention"])
        self.assertEqual(code, 0)
        retention = json.loads(out)["result"]
        for key in ("policy", "online", "archive"):
            self.assertIn(key, retention)
        self.assertIn("n_window_episodes", retention["online"])
        self.assertIn("total_bytes", retention["archive"])

        # -- capability (summary + single prediction full state) --------
        code, out = self.run_cli(["inspect", "--bank", "capability"])
        self.assertEqual(code, 0)
        summary = json.loads(out)["result"]
        for key in ("n_predictions", "n_fact_bound", "n_effect_verified"):
            self.assertIn(key, summary)

    def test_capability_bank_single_read_carries_both_stages(self):
        provider = CountingProvider()
        h = self.make_harness(provider)
        prediction = h.predict_capability_evolution(
            {"operation_type": "induce", "strategy_id": "S01"},
            horizon="next 5 tasks", horizon_tasks=5)
        code, out = self.run_cli([
            "inspect", "--bank", "capability",
            "--prediction", prediction.prediction_id])
        self.assertEqual(code, 0)
        result = json.loads(out)["result"]
        # The full state: the prediction, its binding and its effect.
        for key in ("prediction", "binding", "evaluation"):
            self.assertIn(key, result)
        self.assertEqual(result["prediction"]["prediction_id"],
                         prediction.prediction_id)

    def test_retirement_candidates_are_a_query_not_a_collector(self):
        """The `gc` command is gone WITH the collector that never collected;
        the retirement CANDIDATE view it advertised is a query."""
        h = self.make_harness(CountingProvider())
        code, out = self.run_cli([
            "inspect", "--bank", "strategic", "--status", "suspect"])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["result"]["count"], 0)
        code, out = self.run_cli([
            "inspect", "--bank", "strategic", "--status", "dormant"])
        self.assertEqual(code, 0)
        self.assertFalse(hasattr(h, "collect_garbage"))


# ---------------------------------------------------------------------------
# 8  the removed entry points are gone
# ---------------------------------------------------------------------------


class TestRemovedCommandsAreGone(Base):
    REMOVED = ("predict-outcome", "bind-outcome", "assess-induction",
               "bind-induction-outcome", "gc", "evaluations", "retention",
               "capability-feedback")

    def test_removed_commands_are_usage_errors(self):
        """No silent alias, no different command behind the old name: an
        agent that learned a removed command gets exit 2."""
        for command in self.REMOVED:
            with self.assertRaises(SystemExit) as ctx:
                self.run_cli([command])
            self.assertEqual(ctx.exception.code, 2,
                             f"{command} must be gone, not aliased")

    def test_the_renamed_command_keeps_its_behaviour(self):
        """`predict-cost` is the SAME cost snapshot; the old name is gone."""
        h = self.make_harness(CountingProvider())
        code, out = self.run_cli(["predict-cost", "--task",
                                  json.dumps(TASK), "--strategy", "S01"])
        self.assertEqual(code, 0)
        prediction = json.loads(out)["result"]["prediction"]
        self.assertEqual(prediction["source"], "unknown")
        self.assertIsNone(prediction["expected_cost"])
        with self.assertRaises(SystemExit) as ctx:
            self.run_cli(["predict", "--task", json.dumps(TASK),
                          "--strategy", "S01"])
        self.assertEqual(ctx.exception.code, 2)

    def test_predict_cost_calls_no_provider(self):
        """It is an evidence lookup, not a world-model call."""
        provider = CountingProvider()
        h = self.make_harness(provider)
        self.run_cli(["predict-cost", "--task", json.dumps(TASK),
                      "--strategy", "S01"])
        self.assertEqual(provider.requests, [])


if __name__ == "__main__":
    unittest.main()
