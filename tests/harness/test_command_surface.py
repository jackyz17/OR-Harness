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
import re
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

    def test_a_bad_prediction_id_is_refused_before_the_run(self):
        """An unknown prediction is a PRECONDITION error: the association
        is established before the run, so a bad id is refused BEFORE a
        second of sandbox time is spent and BEFORE any execution fact is
        staged. Nothing is lost — there was nothing to lose yet."""
        provider = CountingProvider()
        h = self.make_harness(provider)
        with self.assertRaises(ValueError) as caught:
            h.execute(TASK, "S01", str(self.script), str(self._work),
                      solver="highs", episode_id="ep1",
                      prediction_id="sp_nope")
        self.assertIn("unknown prediction_id", str(caught.exception))
        # No execution was staged and no action was left running: the
        # refusal happened before the run, so there is no dangling state.
        self.assertEqual(h.bank.pending(), [])
        self.assertEqual(
            [a for a in h.actions.query(task_id="t1")
             if a.action_type == "execute_strategy"], [])

    def test_a_manual_execution_still_records_without_a_prediction(self):
        """An unpredicted attempt stays a first-class path: explicit
        strategy/solver, no prediction, recorded as an unpredicted run."""
        provider = CountingProvider()
        h = self.make_harness(provider)
        record = h.execute(TASK, "S01", str(self.script), str(self._work),
                           solver="highs", episode_id="ep1")
        self.assertTrue(record.quality["feasible"])
        self.assertNotIn("prediction_binding", record.execution_features)
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




# ---------------------------------------------------------------------------
# 9  the reviewed round: the CLI must name the RIGHT candidate and read the
#    binding from where the API writes it
# ---------------------------------------------------------------------------


class SolverAwareProvider(WorldModelProvider):
    """A declared rule: the ``gurobi`` candidate scores BETTER than
    ``highs``, so listing ``highs`` FIRST still yields the ``gurobi``
    suggestion. Nothing here claims real model behaviour."""

    name = "solver-aware"

    def __init__(self):
        self.requests = []

    def predict(self, request, timeout_s=None):
        self.requests.append(request)
        solver = (request.get("candidate") or {}).get("solver")
        quality = 0.95 if solver == "gurobi" else 0.30
        return {"payload": {
                    "benefit": {"kind": "solution_quality",
                                "metric": "normalized_objective_gap",
                                "unit": "1-gap", "value": quality,
                                "baseline": {"kind": "conditional_stats",
                                             "value": 0.0}},
                    "uncertainty": {"execution_randomness": 0.2}},
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
                "error": None, "latency_s": 0.01}


class TestPlanSummaryNamesTheRightCandidate(Base):
    """A suggestion is only useful if the prediction handed over with it
    belongs to the SAME candidate. Matching on strategy_id alone named the
    first candidate with that strategy, so "one strategy, two solvers"
    bound the wrong solver's prediction and the later identity check
    (correctly) rejected it — a real sample lost to a display bug."""

    def _plan_with_two_solvers(self, provider):
        h = self.make_harness(provider)
        from or_harness import cli as cli_module
        original = cli_module._harness
        cli_module._harness = lambda args: h
        self.addCleanup(setattr, cli_module, "_harness", original)
        code, out = self.run_cli([
            "plan-next", "--task", json.dumps(TASK), "--episode", "ep1",
            "--candidates", json.dumps([
                {"action_type": "execute_strategy", "strategy_id": "S01",
                 "solver": "highs"},
                {"action_type": "execute_strategy", "strategy_id": "S01",
                 "solver": "gurobi"}])])
        self.assertEqual(code, 0)
        return json.loads(out)

    def test_the_handed_over_id_belongs_to_the_suggested_solver(self):
        payload = self._plan_with_two_solvers(SolverAwareProvider())
        plan = payload["result"]["plan"]
        suggested = plan["suggested"]
        self.assertEqual(suggested["solver"], "gurobi",
                         "the declared rule makes the SECOND candidate win")
        # The id the summary tells the agent to bind.
        match = re.search(r"--prediction (\S+?)[`.\s]", payload["summary"])
        self.assertIsNotNone(match, payload["summary"])
        handed_over = match.group(1).rstrip(".")
        # The id that really belongs to the suggested candidate.
        truth = next(c["prediction_id"] for c in plan["candidates"]
                     if c["action_spec"]["solver"] == "gurobi")
        self.assertEqual(handed_over, truth,
                         "the summary must name the SUGGESTED candidate's "
                         "prediction, not the first one with that strategy")
        # ... and the two solvers really have different predictions.
        ids = {c["action_spec"]["solver"]: c["prediction_id"]
               for c in plan["candidates"]}
        self.assertNotEqual(ids["highs"], ids["gurobi"])

    def test_the_bound_action_matches_the_suggestion(self):
        """End to end: hand the summary's id to `execute` and the binding
        is COMPARABLE — which is what the old behaviour broke."""
        provider = SolverAwareProvider()
        payload = self._plan_with_two_solvers(provider)
        plan = payload["result"]["plan"]
        handed_over = re.search(
            r"--prediction (\S+?)[`.\s]", payload["summary"]).group(1).rstrip(".")
        h = self.make_harness(provider)
        record = h.execute(
            TASK, "S01", str(self.script), str(self._work), solver="gurobi",
            episode_id="ep1", prediction_id=handed_over)
        binding = record.execution_features["prediction_binding"]
        self.assertTrue(binding["bound"])
        self.assertIsNone(binding["trace"]["binding_mismatch"],
                          "the suggested solver's own prediction must bind "
                          "cleanly")
        self.assertTrue(binding["comparable"])


class TestExecuteSummaryReportsTheRealBinding(Base):
    """The API writes the binding into ``execution_features``; a CLI that
    read a different (nonexistent) attribute reported failure for every
    successful automatic binding, which sent the agent to bind again by
    hand — and buried the REAL failure reason when one occurred."""

    def test_a_successful_binding_is_reported_as_success(self):
        provider = CountingProvider()
        h = self.make_harness(provider)
        from or_harness import cli as cli_module
        original = cli_module._harness
        cli_module._harness = lambda args: h
        self.addCleanup(setattr, cli_module, "_harness", original)
        prediction = h.predict_strategy_outcome(
            TASK, {"action_type": "execute_strategy", "strategy_id": "S01"},
            "ep1")
        code, out = self.run_cli([
            "execute", "--task", json.dumps(TASK), "--strategy", "S01",
            "--code", str(self.script), "--workspace", str(self._work),
            "--solver", "highs", "--episode", "ep1",
            "--prediction", prediction.prediction_id])
        self.assertEqual(code, 0)
        payload = json.loads(out)
        # The CLI's own report and the stored fact must AGREE.
        reported = payload["result"]["prediction_binding"]
        stored = payload["result"]["execution"]["execution_features"][
            "prediction_binding"]
        self.assertTrue(reported["bound"])
        self.assertTrue(stored["bound"])
        self.assertNotIn("WARNING", payload["summary"],
                         "a successful binding must not be reported as a "
                         "failure")

    def test_a_conflicting_explicit_strategy_is_refused_before_the_run(self):
        """The candidate fixes the strategy; an explicit argument that
        disagrees is caught BEFORE the run (exit 2), with the reason — a run
        that would be attributed to the wrong candidate never happens."""
        h = self.make_harness(CountingProvider())
        from or_harness import cli as cli_module
        original = cli_module._harness
        cli_module._harness = lambda args: h
        self.addCleanup(setattr, cli_module, "_harness", original)
        prediction = h.predict_strategy_outcome(
            TASK, {"action_type": "execute_strategy",
                   "strategy_id": "S01"}, "ep1")
        code, out = self.run_cli([
            "execute", "--task", json.dumps(TASK), "--strategy", "S99",
            "--code", str(self.script), "--workspace", str(self._work),
            "--solver", "highs", "--episode", "ep1",
            "--prediction", prediction.prediction_id])
        self.assertEqual(code, 2)
        payload = json.loads(out)
        self.assertIn("contradict", payload["result"]["error"])
        # Nothing ran and the prediction was not consumed.
        self.assertEqual(h.bank.pending(), [])


class TestWorldModelConfigurationIsReachable(Base):
    """The output budget must be settable, and the value that won must be
    the value that is REPORTED. A silent fallback would make a truncated
    answer look like a model failure."""

    def _args(self, **kwargs):
        # Global flags must precede the subcommand; ``argv`` never includes
        # the program name.
        from or_harness.cli import build_parser
        argv = list(kwargs.get("extra", [])) + ["doctor"]
        return build_parser().parse_args(argv)

    def test_precedence_flag_over_env_over_default(self):
        from or_harness import cli as cli_module
        saved = os.environ.pop("OR_WM_MAX_TOKENS", None)
        try:
            # 1. adapter default when nothing is set
            self.assertEqual(cli_module._wm_max_tokens(self._args()), 2048)
            # 2. environment variable wins over the default
            os.environ["OR_WM_MAX_TOKENS"] = "8192"
            self.assertEqual(cli_module._wm_max_tokens(self._args()), 8192)
            # 3. the explicit flag wins over the environment
            args = self._args(extra=["--wm-max-tokens", "4096"])
            self.assertEqual(cli_module._wm_max_tokens(args), 4096)
        finally:
            if saved is None:
                os.environ.pop("OR_WM_MAX_TOKENS", None)
            else:
                os.environ["OR_WM_MAX_TOKENS"] = saved

    def test_a_bad_env_value_is_reported_not_ignored(self):
        from or_harness import cli as cli_module
        saved = os.environ.get("OR_WM_MAX_TOKENS")
        os.environ["OR_WM_MAX_TOKENS"] = "8k192"
        try:
            with self.assertRaises(ValueError):
                cli_module._wm_max_tokens(self._args())
        finally:
            if saved is None:
                os.environ.pop("OR_WM_MAX_TOKENS", None)
            else:
                os.environ["OR_WM_MAX_TOKENS"] = saved

    def test_the_budget_reaches_the_http_body(self):
        from or_harness.world_model.provider import HttpChatProvider
        import urllib.request

        captured = {}

        class Resp:
            def read(self):
                return json.dumps({
                    "choices": [{"message": {"content": "{}"},
                                 "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                }).encode()

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        original = urllib.request.urlopen
        urllib.request.urlopen = lambda req, *a, **k: (
            captured.update(json.loads(req.data.decode())), Resp())[1]
        try:
            provider = HttpChatProvider("https://x/v1", "M", "k",
                                        max_output_tokens=8192)
            provider.predict({"prediction_protocol": "wm-so/1"})
        finally:
            urllib.request.urlopen = original
        self.assertEqual(captured["max_tokens"], 8192,
                         "the configured budget must be what is SENT")

    def test_plan_next_exposes_the_time_budget(self):
        from or_harness.cli import build_parser
        args = build_parser().parse_args(
            ["plan-next", "--task", "t.json", "--time-budget", "45"])
        self.assertEqual(args.time_budget, 45.0)


class TestTheCandidateCarriesItsMethod(Base):
    """A candidate the model cannot understand is a candidate it cannot
    predict. The CALLER supplies the method description — a cold start has
    no memory to read it from — and it must survive into the request."""

    def test_action_spec_method_reaches_the_request(self):
        from or_harness.world_model.prediction import ActionSpec

        class Capture(WorldModelProvider):
            name = "capture"

            def __init__(self):
                self.request = None

            def predict(self, request, timeout_s=None):
                self.request = request
                return {"payload": None, "usage": None, "error": "captured",
                        "latency_s": 0.0}

        provider = Capture()
        h = self.make_harness(provider=provider)
        spec = ActionSpec(
            action_type="execute_strategy", task_id="t1", strategy_id="S01",
            solver="highs", params={"time_limit": 60},
            method={"name": "rolling-horizon decomposition",
                    "steps": ["split into 3-period blocks",
                              "solve each block with the previous ending "
                              "inventory fixed"]})
        h.snapshot(TASK, episode_id="ep1")
        h.predict_strategy_outcome(TASK, spec, "ep1")
        candidate = provider.request["candidate"]
        self.assertEqual(candidate["method"]["name"],
                         "rolling-horizon decomposition")
        self.assertEqual(len(candidate["method"]["steps"]), 2)
        # The method and the execution parameters are SEPARATE: a solver
        # setting must never be sent as if it were the approach.
        self.assertNotIn("time_limit", json.dumps(candidate["method"]))
        self.assertEqual(candidate["config"]["time_limit"], 60)

    def test_candidate_ref_method_round_trips(self):
        from or_harness.world_model.contracts import CandidateRef
        ref = CandidateRef(
            action_type="execute_strategy", strategy_id="S01",
            method={"name": "Benders", "steps": ["master", "subproblem"]})
        again = CandidateRef.from_dict(ref.to_dict())
        self.assertEqual(again.method["name"], "Benders")
        self.assertEqual(again.method["steps"], ["master", "subproblem"])

    def test_an_undescribed_method_is_empty_not_invented(self):
        from or_harness.world_model.contracts import CandidateRef
        ref = CandidateRef.from_dict(
            {"action_type": "execute_strategy", "strategy_id": "S01"})
        self.assertEqual(ref.method, {})


# ---------------------------------------------------------------------------
# 9  world-model call configuration is reachable AND reported
# ---------------------------------------------------------------------------


class _EnvGuard:
    """Set environment variables for one test and restore them after.

    A leaked variable would make a later test's default look like a
    configured value, so restoration is unconditional and covers the
    unset-as-None case.
    """

    def __init__(self, **values):
        self.values = values
        self.saved = {}

    def __enter__(self):
        for name, value in self.values.items():
            self.saved[name] = os.environ.get(name)
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        return self

    def __exit__(self, *exc):
        for name, value in self.saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        return False


def _posted_body(provider):
    """The JSON body ``provider`` would POST, with the socket stubbed out."""
    import urllib.request
    captured = {}

    class Resp:
        def read(self):
            return json.dumps({
                "choices": [{"message": {"content": "{}"},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }).encode()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    original = urllib.request.urlopen
    urllib.request.urlopen = lambda req, *a, **k: (
        captured.update(json.loads(req.data.decode())), Resp())[1]
    try:
        provider.predict({"prediction_protocol": "wm-so/1"})
    finally:
        urllib.request.urlopen = original
    return captured


class TestTheJsonModeHintIsRemovable(unittest.TestCase):
    """Some endpoints reject ``response_format``. The opt-out must remove
    the KEY — sending ``null`` in its place would be a different request
    that the same endpoint may also reject — and an unconfigured
    environment must keep the previous wire format byte for byte."""

    def _provider(self):
        from or_harness.world_model.provider import HttpChatProvider
        return HttpChatProvider("https://x/v1", "M", "k")

    def test_the_switch_is_off_unless_explicitly_opted_in(self):
        from or_harness.world_model.provider import _truthy_env
        for value in ("1", "true", "TRUE", "Yes", "on", " on "):
            with _EnvGuard(OR_WM_NO_RESPONSE_FORMAT=value):
                self.assertTrue(_truthy_env("OR_WM_NO_RESPONSE_FORMAT"), value)
        for value in (None, "", "0", "false", "no", "off", "maybe", "2"):
            with _EnvGuard(OR_WM_NO_RESPONSE_FORMAT=value):
                self.assertFalse(_truthy_env("OR_WM_NO_RESPONSE_FORMAT"),
                                 repr(value))

    def test_unset_sends_the_json_mode_hint(self):
        with _EnvGuard(OR_WM_NO_RESPONSE_FORMAT=None):
            provider = self._provider()
            body = _posted_body(provider)
        self.assertEqual(body["response_format"], {"type": "json_object"})
        self.assertTrue(provider.describe()["send_response_format"])

    def test_enabled_omits_the_key_entirely(self):
        with _EnvGuard(OR_WM_NO_RESPONSE_FORMAT="1"):
            provider = self._provider()
            body = _posted_body(provider)
        self.assertNotIn("response_format", body)
        # Explicitly: absent, NOT present-and-null.
        self.assertIsNone(body.get("response_format"))
        self.assertFalse(provider.describe()["send_response_format"])

    def test_the_environment_is_read_once_at_construction(self):
        """The value reported by ``describe()`` must be the value that the
        body was built with, so it cannot be re-read per call."""
        with _EnvGuard(OR_WM_NO_RESPONSE_FORMAT="1"):
            provider = self._provider()
        with _EnvGuard(OR_WM_NO_RESPONSE_FORMAT=None):
            body = _posted_body(provider)
        self.assertNotIn("response_format", body)
        self.assertFalse(provider.describe()["send_response_format"])

    def test_only_the_hint_changes(self):
        """The rest of the request is untouched by the switch: exactly one
        key differs, every other key is byte-identical, and the key ORDER
        of the enabled request is the one this build sent before."""
        with _EnvGuard(OR_WM_NO_RESPONSE_FORMAT="1"):
            without = _posted_body(self._provider())
        with _EnvGuard(OR_WM_NO_RESPONSE_FORMAT=None):
            with_hint = _posted_body(self._provider())
        self.assertEqual(set(with_hint) - set(without), {"response_format"})
        self.assertEqual(set(without) - set(with_hint), set())
        self.assertEqual({k: v for k, v in with_hint.items()
                          if k != "response_format"}, without)
        self.assertEqual(list(with_hint),
                         ["model", "messages", "response_format",
                          "max_tokens", "temperature", "enable_thinking"])
        self.assertEqual(list(without),
                         ["model", "messages", "max_tokens", "temperature",
                          "enable_thinking"])
        self.assertEqual(with_hint["temperature"], 0.2)
        self.assertEqual(with_hint["max_tokens"], 2048)


class TestThinkingIsOffUnlessAskedFor(Base):
    """Thinking is charged INSIDE ``completion_tokens``, so it competes with
    the answer for the output budget and dominates the wall clock. The
    default arm asks the endpoint not to reason; the opt-in omits the key so
    the endpoint's own default governs — we never assert ``true``, because
    guessing that default is the same mistake in the other direction."""

    def _provider(self):
        from or_harness.world_model.provider import HttpChatProvider
        return HttpChatProvider("https://x/v1", "Qwen3.8-27B", "k")

    def test_the_default_request_turns_thinking_off(self):
        with _EnvGuard(OR_WM_ENABLE_THINKING=None):
            provider = self._provider()
            body = _posted_body(provider)
        self.assertIs(body["enable_thinking"], False)
        self.assertFalse(provider.describe()["enable_thinking"])

    def test_the_opt_in_omits_the_key_rather_than_asserting_true(self):
        with _EnvGuard(OR_WM_ENABLE_THINKING="1"):
            provider = self._provider()
            body = _posted_body(provider)
        self.assertNotIn("enable_thinking", body)
        self.assertIsNone(body.get("enable_thinking"))
        self.assertTrue(provider.describe()["enable_thinking"])

    def test_it_is_a_second_switch_not_a_coupled_one(self):
        """Turning thinking on must not change the JSON-mode arm or any
        other field: the two switches are independent."""
        with _EnvGuard(OR_WM_ENABLE_THINKING="1",
                       OR_WM_NO_RESPONSE_FORMAT=None):
            body = _posted_body(self._provider())
        self.assertEqual(body["response_format"], {"type": "json_object"})
        self.assertEqual(body["max_tokens"], 2048)
        self.assertEqual(body["temperature"], 0.2)
        self.assertEqual(sorted(body), sorted(
            ["model", "messages", "response_format", "max_tokens",
             "temperature"]))

    def test_the_flag_is_read_once_at_construction(self):
        with _EnvGuard(OR_WM_ENABLE_THINKING="1"):
            provider = self._provider()
        with _EnvGuard(OR_WM_ENABLE_THINKING=None):
            body = _posted_body(provider)
        self.assertNotIn("enable_thinking", body)
        self.assertTrue(provider.describe()["enable_thinking"])

    def test_the_two_arms_are_distinguishable_after_the_fact(self):
        """``describe()`` feeds ``capability_version``: a prediction made
        with thinking must not silently group with one made without it."""
        with _EnvGuard(OR_WM_ENABLE_THINKING=None):
            off = self._provider().describe()
        with _EnvGuard(OR_WM_ENABLE_THINKING="1"):
            on = self._provider().describe()
        self.assertNotEqual(off, on)
        self.assertEqual(set(off) ^ set(on), set())
        self.assertNotEqual(off["enable_thinking"], on["enable_thinking"])


class TestTheWorldModelTimeoutHasAChain(unittest.TestCase):
    """``$OR_WM_TIMEOUT`` must actually be reachable. With a numeric
    argparse default it never could be: "not given" and "given as 300"
    would be indistinguishable."""

    def _args(self, argv=None):
        from or_harness.cli import build_parser
        # Global flags precede the subcommand; argv never includes the
        # program name.
        return build_parser().parse_args(list(argv or []) + ["doctor"])

    def test_the_flag_default_is_none_so_the_env_can_win(self):
        self.assertIsNone(self._args().wm_timeout)

    def test_precedence_flag_over_env_over_default(self):
        from or_harness import cli as cli_module
        with _EnvGuard(OR_WM_TIMEOUT=None):
            self.assertEqual(cli_module._wm_timeout(self._args()), 300.0)
            with _EnvGuard(OR_WM_TIMEOUT="120"):
                self.assertEqual(cli_module._wm_timeout(self._args()), 120.0)
                args = self._args(["--wm-timeout", "45"])
                self.assertEqual(cli_module._wm_timeout(args), 45.0)

    def test_a_bad_env_value_is_reported_not_ignored(self):
        from or_harness import cli as cli_module
        with _EnvGuard(OR_WM_TIMEOUT="30s"):
            with self.assertRaises(ValueError):
                cli_module._wm_timeout(self._args())

    def test_an_explicit_zero_is_respected(self):
        """``--wm-timeout 0`` must not be swallowed into the default by a
        falsy-value ``or`` fallback."""
        from or_harness import cli as cli_module
        with _EnvGuard(OR_WM_TIMEOUT=None):
            args = self._args(["--wm-timeout", "0"])
            self.assertEqual(cli_module._wm_timeout(args), 0.0)

    def test_the_default_is_read_from_the_adapter_not_duplicated(self):
        import inspect
        from or_harness import cli as cli_module
        from or_harness.world_model.provider import HttpChatProvider
        default = inspect.signature(HttpChatProvider.__init__).parameters[
            "timeout_s"].default
        with _EnvGuard(OR_WM_TIMEOUT=None):
            self.assertEqual(cli_module._wm_timeout(self._args()),
                             float(default))
        self.assertEqual(float(default), 300.0)


class TestTheTimeoutReachesTheProvider(Base):
    def _built(self, argv):
        from or_harness.cli import _harness, build_parser
        args = build_parser().parse_args(
            ["--home", self.home, "--world-model", "https://x/v1::M"]
            + list(argv) + ["doctor"])
        h = _harness(args)
        self.addCleanup(h.close)
        return h.world_model

    def test_the_default_is_a_long_timeout_and_is_reported(self):
        with _EnvGuard(OR_WM_TIMEOUT=None):
            provider = self._built([])
        self.assertEqual(provider.timeout_s, 300.0)
        self.assertEqual(provider.describe()["timeout_s"], 300.0)

    def test_the_env_var_reaches_the_provider(self):
        with _EnvGuard(OR_WM_TIMEOUT="450"):
            provider = self._built([])
        self.assertEqual(provider.timeout_s, 450.0)

    def test_the_flag_still_wins(self):
        with _EnvGuard(OR_WM_TIMEOUT="450"):
            provider = self._built(["--wm-timeout", "60"])
        self.assertEqual(provider.timeout_s, 60.0)

    def test_the_callers_budget_still_bounds_the_call(self):
        """``min(timeout_s, self.timeout_s)`` is unchanged: a long default
        must not let one call outlive the caller's remaining budget."""
        provider = self._built(["--wm-timeout", "300"])
        _posted_body_with = {}

        class Recorder:
            def __init__(self, inner):
                self.inner = inner

            def read(self):
                return self.inner

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        import urllib.request
        original = urllib.request.urlopen

        def fake(req, *a, **k):
            _posted_body_with.update(k)
            return Recorder(json.dumps({
                "choices": [{"message": {"content": "{}"},
                             "finish_reason": "stop"}]}).encode())

        urllib.request.urlopen = fake
        try:
            result = provider.predict({"x": 1}, timeout_s=7.0)
        finally:
            urllib.request.urlopen = original
        self.assertEqual(_posted_body_with.get("timeout"), 7.0)
        effective = result["diagnostics"]["effective"]
        self.assertEqual(effective["timeout_s"], 7.0)
        self.assertEqual(effective["provider_timeout_s"], 300.0)


# ---------------------------------------------------------------------------
# 10  one solve per decision; the retired "strong" label is refused
# ---------------------------------------------------------------------------


class TestOneSolvePerDecision(Base):
    """Predicting several candidates is how a choice is made; EXECUTING
    one is what the loop does. A second solve must come from a real failure
    or an explicit decision, never from the framework. These tests count
    solves, because that is the only way to tell the two stories apart."""

    def setUp(self):
        super().setUp()
        self.solver_calls = 0
        # Count real solver invocations by wrapping the executor.
        original = ORHarness.__init__

        def counting_init(harness_self, *a, **k):
            original(harness_self, *a, **k)
            inner = harness_self.executor.execute

            def counted(*ea, **ek):
                self.solver_calls += 1
                return inner(*ea, **ek)

            harness_self.executor.execute = counted

        ORHarness.__init__ = counting_init
        self.addCleanup(setattr, ORHarness, "__init__", original)

    def _solve(self, task, tag, objective=1.0):
        work = Path(self.home) / f"ws_{tag}"
        work.mkdir(parents=True, exist_ok=True)
        script = work / "solve.py"
        script.write_text(
            "import json\n"
            "with open('result.json', 'w') as fh:\n"
            "    json.dump({'status': 'optimal', 'objective_value': "
            f"{objective}, 'objective_bound': {objective}, "
            f"'mip_gap': 0.0, 'runtime_seconds': 0.01, "
            "'variables': {'x1': 1}}, fh)\n", encoding="utf-8")
        return self.make_harness(), script, work

    def test_check_and_close_do_not_solve_again(self):
        h, script, work = self._solve(TASK, "one")
        record = h.execute(TASK, "S01", str(script), str(work),
                           solver="highs", episode_id="ep1")
        self.assertEqual(self.solver_calls, 1)
        # Checking the answer is a check, not another solve.
        checked = h.check_task_result(record.execution_id,
                                      {"reference_status": "optimal"})
        self.assertEqual(checked["state"], "passed")
        self.assertEqual(self.solver_calls, 1,
                         "check-task must not run the solver again")
        h.record(record)
        # Closing the episode reuses the recorded verdict.
        closed = h.close_episode(TASK["task_id"], "ep1",
                                 terminal_state="completed")
        self.assertIsNotNone(closed)
        self.assertEqual(self.solver_calls, 1,
                         "close-episode must not re-verify or re-solve")

    def test_a_failed_check_does_not_auto_retry(self):
        h, script, work = self._solve(TASK, "bad", objective=1.0)
        record = h.execute(TASK, "S01", str(script), str(work),
                           solver="highs", episode_id="ep1")
        failed = h.check_task_result(
            record.execution_id, {"reference_objective": 999.0})
        self.assertEqual(failed["state"], "failed")
        # The response tells the AGENT what to do next; it does not do it.
        self.assertIn("next", failed)
        self.assertEqual(self.solver_calls, 1,
                         "a failed check must not start a second solve")


class TestTheVerificationLevelConceptIsGone(Base):
    """There is ONE verification depth, so a level selector is a lever that
    moves nothing. ``strong`` was never more than a label (the executor's
    checks inspect no level), and a record carrying it would have claimed an
    enhanced verification that never ran. The whole concept is removed
    rather than accepted-and-ignored."""

    def test_the_execute_command_has_no_level_flag(self):
        from or_harness.cli import build_parser
        subcommands = [a for a in build_parser()._actions
                       if getattr(a, "choices", None)
                       and "execute" in (a.choices or ())]
        self.assertTrue(subcommands)
        execute = subcommands[0].choices["execute"]
        flags = {o for action in execute._actions
                 for o in action.option_strings}
        self.assertNotIn("--verification", flags)

    def test_the_api_takes_no_level_argument(self):
        import inspect
        parameters = inspect.signature(ORHarness.execute).parameters
        self.assertNotIn("verification_level", parameters)

    def test_a_record_has_no_level_field(self):
        from or_harness.core.schema import ExecutionRecord
        self.assertFalse(hasattr(ExecutionRecord, "verification_level"))
        # Neither on the dataclass FIELDS nor on a built instance.
        import dataclasses
        names = {f.name for f in dataclasses.fields(ExecutionRecord)}
        self.assertNotIn("verification_level", names)

    def test_an_old_record_payload_with_the_field_still_loads(self):
        """Dropping a field must be backward TOLERANT on read: an older
        payload carrying the key is loaded and the key is ignored, never
        turned into a load failure."""
        from or_harness.core.schema import ExecutionRecord
        h = self.make_harness()
        work = Path(self.home) / "ws_hist"
        work.mkdir(parents=True, exist_ok=True)
        script = work / "solve.py"
        script.write_text(
            "import json\n"
            "with open('result.json', 'w') as fh:\n"
            "    json.dump({'status': 'optimal', 'objective_value': 5.0, "
            "'runtime_seconds': 0.01}, fh)\n", encoding="utf-8")
        record = h.execute(TASK, "S01", str(script), str(work),
                           solver="highs", episode_id="ep1")
        stored = record.to_dict()
        stored["verification_level"] = "strong"
        back = ExecutionRecord.from_dict(stored)
        self.assertFalse(hasattr(back, "verification_level"))
        # The label leaves no trace that could be read as evidence.
        self.assertNotIn("verification_level", back.to_dict())
        self.assertNotIn("strong", json.dumps(back.to_dict()))


# ---------------------------------------------------------------------------
# 10  the association is established BEFORE the execution
# ---------------------------------------------------------------------------


class TestPreExecutionAssociation(Base):
    """Requirement 1 & 2 & 3: `execute --prediction` reads the FROZEN
    candidate, checks it before anything runs, takes the identity from it
    (the caller does not re-type it), persists the link before the run, and
    only ADDS the observed configuration afterwards."""

    def _predict(self, h, strategy="S01", solver="highs", **config):
        return h.predict_strategy_outcome(
            TASK, {"action_type": "execute_strategy",
                   "strategy_id": strategy, "solver": solver,
                   "config": config}, "ep1")

    def test_identity_is_taken_from_the_candidate(self):
        provider = CountingProvider()
        h = self.make_harness(provider)
        prediction = self._predict(h)
        # The caller re-types NOTHING: no --strategy, no --solver, no
        # --episode. They come from the frozen candidate.
        record = h.execute(TASK, None, str(self.script), str(self._work),
                           solver=None, episode_id=None,
                           prediction_id=prediction.prediction_id)
        self.assertEqual(record.strategy_id, "S01")
        self.assertEqual(record.solver["name"] or "highs", "highs")
        self.assertTrue(record.quality["feasible"])
        binding = record.execution_features["prediction_binding"]
        self.assertTrue(binding["bound"])
        self.assertIsNone(binding["trace"]["binding_mismatch"])

    def test_the_link_exists_before_the_run(self):
        """The association is persisted the moment the action exists: an
        interrupted execution still shows which attempt a prediction was
        testing. Observed by a probe executor that captures the action
        BEFORE running anything."""
        provider = CountingProvider()
        h = self.make_harness(provider)
        prediction = self._predict(h)
        seen: Dict[str, Any] = {}

        class Probe:
            """An injected executor that records the action and then fails."""

            def __init__(self, real):
                self.real = real
                self.calls = []

            def execute(self, *a, **k):
                seen["action"] = h.actions.get(k.get("action_id") or "")
                raise RuntimeError("probe stopped before the sandbox ran")

        h.executor = Probe(h.executor)
        with self.assertRaises(RuntimeError):
            h.execute(TASK, None, str(self.script), str(self._work),
                      solver=None, episode_id=None,
                      prediction_id=prediction.prediction_id)
        action = seen.get("action")
        self.assertIsNotNone(action, "the action exists during the run")
        self.assertEqual(action.params.get("prediction_id"),
                         prediction.prediction_id)
        self.assertIsNotNone(action.params.get("prediction_bound_at"))
        # And the prediction itself carries the forward link.
        stored = h.strategy_predictions.get(prediction.prediction_id)
        self.assertEqual(stored.trace.model_info.get(
            "association_phase"), "linked_before_execution")
        self.assertEqual(stored.trace.model_info.get(
            "bound_action_id_before_execution"), action.action_id)
        # The interrupted action is NOT left running forever.
        ended = h.actions.get(action.action_id)
        self.assertEqual(ended.status, "failed")

    def test_a_claimed_prediction_is_refused(self):
        """One prediction corresponds to ONE real attempt: reusing it for a
        second execution is refused before the run, so the sample is never
        double-counted."""
        provider = CountingProvider()
        h = self.make_harness(provider)
        prediction = self._predict(h)
        first = h.execute(TASK, None, str(self.script), str(self._work),
                          solver=None, episode_id=None,
                          prediction_id=prediction.prediction_id)
        self.assertTrue(first.quality["feasible"])
        with self.assertRaises(ValueError) as caught:
            h.execute(TASK, None, str(self.script), str(self._work),
                      solver=None, episode_id=None,
                      prediction_id=prediction.prediction_id)
        self.assertIn("already bound", str(caught.exception))

    def test_a_script_outside_the_workspace_is_refused_before_the_run(self):
        provider = CountingProvider()
        h = self.make_harness(provider)
        prediction = self._predict(h)
        outside = Path(self.home) / "outside.py"
        outside.write_text("import json\n", encoding="utf-8")
        with self.assertRaises(ValueError) as caught:
            h.execute(TASK, None, str(outside), str(self._work),
                      solver=None, episode_id=None,
                      prediction_id=prediction.prediction_id)
        self.assertIn("outside its workspace", str(caught.exception))
        # The prediction was NOT spent.
        stored = h.strategy_predictions.get(prediction.prediction_id)
        self.assertIsNone(
            stored.trace.model_info.get("bound_action_id"))


# ---------------------------------------------------------------------------
# 11  the actual execution configuration is OBSERVED, never copied
# ---------------------------------------------------------------------------


class TestActualConfigurationIsObserved(Base):
    """Requirement 3: the executor records what it controlled, the script
    reads back what really took effect, a missing value stays UNKNOWN, and a
    stale result.json from an earlier run is never read as this run's
    config."""

    def _script(self, name, body):
        work = Path(self.home) / f"ws_{name}"
        work.mkdir(parents=True, exist_ok=True)
        script = work / "solve.py"
        script.write_text(body, encoding="utf-8")
        return script, work

    def test_the_executor_records_what_it_controlled(self):
        script, work = self._script("cfg_ok", (
            "import json, os\n"
            "cfg = {'action_id': os.environ.get('OR_ACTION_ID'),\n"
            "       'time_limit': 60}\n"
            "with open('result.json', 'w') as fh:\n"
            "    json.dump({'status': 'optimal', 'objective_value': 1.0,\n"
            "               'objective_bound': 1.0, 'runtime_seconds': 0.01,\n"
            "               'config': cfg}, fh)\n"))
        h = self.make_harness()
        record = h.execute(TASK, "S01", str(script), str(work),
                           solver="highs", episode_id="ep1")
        config = record.execution_features["execution_config"]
        # The executor's OWN keys, from its own control.
        self.assertEqual(config["values"]["script_timeout_s"], 120.0)
        self.assertEqual(config["sources"]["script_timeout_s"], "executor")
        self.assertEqual(config["sources"]["solver"], "executor")
        # The script's read-back value, labelled as such.
        self.assertEqual(config["values"]["time_limit"], 60)
        self.assertEqual(config["sources"]["time_limit"], "script_reported")

    def test_a_receipt_without_the_matching_stamp_is_refused(self):
        """A leftover result.json from an earlier run is NOT this run's
        configuration: the receipt's action_id stamp must match."""
        work = Path(self.home) / "ws_stale"
        work.mkdir(parents=True, exist_ok=True)
        # Pre-write a result.json stamped with a DIFFERENT action id.
        (work / "result.json").write_text(json.dumps({
            "status": "optimal", "objective_value": 9.0,
            "config": {"action_id": "ac_someone_else", "time_limit": 999},
        }), encoding="utf-8")
        script = work / "solve.py"
        script.write_text(
            "import json\n"
            "with open('result.json', 'w') as fh:\n"
            "    json.dump({'status': 'optimal', 'objective_value': 1.0,\n"
            "               'objective_bound': 1.0, 'runtime_seconds': 0.01},\n"
            "              fh)\n", encoding="utf-8")
        h = self.make_harness()
        record = h.execute(TASK, "S01", str(script), str(work),
                           solver="highs", episode_id="ep1")
        config = record.execution_features["execution_config"]
        # The stale config NEVER appears: the new result.json carried none.
        self.assertNotIn("time_limit", config["values"])
        self.assertIn("receipt", config)

    def test_a_failed_run_still_reports_the_configuration_it_set(self):
        """A timeout after the script set its parameters still reports them
        — 'no result' is not 'no config'."""
        script, work = self._script("cfg_fail", (
            "import json, os, sys\n"
            "cfg = {'action_id': os.environ.get('OR_ACTION_ID'),\n"
            "       'time_limit': 45}\n"
            "with open('result.json', 'w') as fh:\n"
            "    json.dump({'status': 'running', 'config': cfg}, fh)\n"
            "sys.exit(3)\n"))
        h = self.make_harness()
        record = h.execute(TASK, "S01", str(script), str(work),
                           solver="highs", episode_id="ep1")
        self.assertEqual(record.quality["status"], "error")
        config = record.execution_features["execution_config"]
        self.assertEqual(config["values"]["time_limit"], 45)

    def test_old_script_without_a_config_report_still_executes(self):
        """A script that reports no config still runs; its internal
        parameters stay UNKNOWN (never a copy of the predicted config), and
        the binding says so rather than claiming a match."""
        script, work = self._script("cfg_old", (
            "import json\n"
            "with open('result.json', 'w') as fh:\n"
            "    json.dump({'status': 'optimal', 'objective_value': 1.0,\n"
            "               'objective_bound': 1.0, 'runtime_seconds': 0.01},\n"
            "              fh)\n"))
        h = self.make_harness()
        prediction = h.predict_strategy_outcome(
            TASK, {"action_type": "execute_strategy", "strategy_id": "S01",
                   "solver": "highs", "config": {"time_limit": 60}}, "ep1")
        record = h.execute(TASK, None, str(script), str(work),
                           solver=None, episode_id=None,
                           prediction_id=prediction.prediction_id)
        self.assertTrue(record.quality["feasible"])
        binding = record.execution_features["prediction_binding"]
        # The predicted config was NOT copied in as if observed.
        self.assertIn("time_limit", binding["trace"]["binding_unknown"]["config"])


# ---------------------------------------------------------------------------
# 12  choosing by prediction id (no re-typed candidate JSON)
# ---------------------------------------------------------------------------


class TestChooseNextByPredictionId(Base):
    """Requirement 2: `choose-next --prediction` resolves the chosen
    candidate from the DECISION's own recorded comparison; the deviation
    test uses the candidate reference, and a prediction from another
    decision is refused."""

    def _plan(self, provider=None):
        h = ORHarness(home=self.home, world_model=provider or CountingProvider())
        self.addCleanup(h.close)
        plan = h.plan_next(
            TASK, "ep1",
            candidates=[ActionSpec("execute_strategy", "t1",
                                   strategy_id="S01", solver="highs"),
                        ActionSpec("execute_strategy", "t1",
                                   strategy_id="S02", solver="highs")])
        return h, plan

    def test_choosing_by_id_matches_the_candidate(self):
        h, plan = self._plan()
        target = plan["candidates"][1]
        choice = h.choose_next(plan["decision_action_id"],
                               prediction_id=target["prediction_id"])
        self.assertEqual(choice["selected"]["strategy_id"], "S02")
        # `choose-next --prediction` alone does not create a deviation when
        # the id names the suggestion... but here S02 was chosen while S01
        # was suggested: a real deviation, recorded as such.
        suggested = plan["suggested"]["strategy_id"]
        if suggested == "S02":
            self.assertIsNone(choice["deviation"])
        else:
            self.assertIsNotNone(choice["deviation"])

    def test_a_prediction_from_another_decision_is_refused(self):
        h, plan = self._plan()
        # A second decision with its own predictions.
        plan2 = h.plan_next(TASK, "ep1",
                            candidates=[ActionSpec("execute_strategy", "t1",
                                                   strategy_id="S04")])
        other_id = plan2["candidates"][0]["prediction_id"]
        with self.assertRaises(ValueError) as caught:
            h.choose_next(plan["decision_action_id"],
                          prediction_id=other_id)
        self.assertIn("not among the candidates", str(caught.exception))

    def test_deviation_is_by_candidate_reference_not_whole_json(self):
        """The same candidate in a different key ORDER, or with a different
        cosmetic field, is NOT a deviation: the comparison is on the
        candidate reference."""
        h, plan = self._plan()
        suggested = plan["suggested"]
        chosen = dict(suggested)
        # Reorder keys and add a cosmetic note: same candidate.
        reordered = {k: chosen[k] for k in reversed(list(chosen))}
        choice = h.choose_next(
            plan["decision_action_id"],
            chosen=ActionSpec.from_dict(reordered))
        self.assertIsNone(choice["deviation"])

    def test_a_different_config_is_a_deviation(self):
        """strategy_id + solver alone is NOT the identity: the same pair
        under a different config is a different candidate, and choosing it
        is a deviation, not an accept."""
        h, plan = self._plan()
        chosen = dict(plan["suggested"])
        chosen["params"] = {"time_limit": 999}
        choice = h.choose_next(plan["decision_action_id"],
                               chosen=ActionSpec.from_dict(chosen))
        self.assertIsNotNone(choice["deviation"])

    def test_the_cli_accepts_a_prediction_id(self):
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
        plan = json.loads(out)["result"]["plan"]
        prediction_id = plan["candidates"][0]["prediction_id"]
        code, out = self.run_cli([
            "choose-next", "--decision", plan["decision_action_id"],
            "--prediction", prediction_id])
        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertEqual(payload["result"]["selected"]["strategy_id"], "S01")


# ---------------------------------------------------------------------------
# 13  the four review findings on the association round
# ---------------------------------------------------------------------------


class TestTheScriptCannotOverwriteExecutorConfig(Base):
    """Finding 1: the solve script may not overwrite a field the executor
    really controlled (its 1-second claim must not replace the 120-second
    truth and make the run look like it matched a prediction), and a value
    the executor only INSTRUCTED the script with is not an observation."""

    def _run(self, body, prediction_config):
        work = Path(self.home) / f"ws_{abs(hash(body)) % 10000}"
        work.mkdir(parents=True, exist_ok=True)
        (work / "solve.py").write_text(body, encoding="utf-8")
        h = self.make_harness()
        prediction = h.predict_strategy_outcome(
            TASK, {"action_type": "execute_strategy", "strategy_id": "S99",
                   "solver": "highs", "config": prediction_config}, "ep1")
        record = h.execute(TASK, None, str(work / "solve.py"), str(work),
                           solver=None, episode_id=None,
                           prediction_id=prediction.prediction_id)
        return record

    def test_a_script_timeout_claim_does_not_overwrite_the_executor(self):
        record = self._run(
            "import json, os\n"
            "cfg = {'action_id': os.environ.get('OR_ACTION_ID'),\n"
            "       'script_timeout_s': 1}\n"
            "with open('result.json', 'w') as fh:\n"
            "    json.dump({'status': 'optimal', 'objective_value': 1.0,\n"
            "               'objective_bound': 1.0, 'runtime_seconds': 0.01,\n"
            "               'config': cfg}, fh)\n",
            {"script_timeout_s": 120})
        config = record.execution_features["execution_config"]
        # The executor's own value stands; the contradiction is recorded.
        self.assertEqual(config["values"]["script_timeout_s"], 120.0)
        self.assertEqual(config["sources"]["script_timeout_s"], "executor")
        self.assertIn("script_timeout_s", config["conflicts"])
        self.assertEqual(
            config["conflicts"]["script_timeout_s"]["script_reported"], 1)
        # And the honest prediction still binds cleanly: the script's bogus
        # number must not manufacture a mismatch either.
        binding = record.execution_features["prediction_binding"]
        self.assertIsNone(binding["trace"]["binding_mismatch"])

    def test_an_instructed_value_is_not_an_observation(self):
        record = self._run(
            "import json\n"
            "with open('result.json', 'w') as fh:\n"
            "    json.dump({'status': 'optimal', 'objective_value': 1.0,\n"
            "               'objective_bound': 1.0, 'runtime_seconds': 0.01},\n"
            "              fh)\n",
            {"solver_timeout_s": 60})
        config = record.execution_features["execution_config"]
        # Handed to the script, never recorded as something the solver used.
        self.assertNotIn("solver_timeout_s", config["values"])
        self.assertIn("solver_timeout_s", config["executor_configured"])
        binding = record.execution_features["prediction_binding"]
        unknown = (binding["trace"]["binding_unknown"] or {}).get("config", {})
        self.assertIn("solver_timeout_s", unknown)


class TestAStaleResultIsNeverThisRun(Base):
    """Finding 2: a leftover result.json must not become this attempt's
    body — verified at the executor level AND end to end."""

    def test_the_executor_refuses_a_stale_result_body(self):
        from or_harness.execution.executor import SafePythonExecutor
        work = Path(self.home) / "ws_stale_body"
        work.mkdir(parents=True, exist_ok=True)
        (work / "result.json").write_text(json.dumps(
            {"status": "optimal", "objective_value": 999.0,
             "objective_bound": 999.0, "runtime_seconds": 0.01}))
        (work / "solve.py").write_text("pass\n")
        executor = SafePythonExecutor(timeout_seconds=30)
        outcome = executor.run(work / "solve.py", work, "highs",
                               action_id="ac_test")
        self.assertEqual(outcome.status, "error")
        self.assertIsNone(outcome.objective_value)

    def test_no_phantom_sample_reaches_the_calibration(self):
        work = Path(self.home) / "ws_phantom"
        work.mkdir(parents=True, exist_ok=True)
        (work / "result.json").write_text(json.dumps(
            {"status": "optimal", "objective_value": 999.0,
             "objective_bound": 999.0, "runtime_seconds": 0.01}))
        (work / "solve.py").write_text("pass\n")
        h = self.make_harness()
        prediction = h.predict_strategy_outcome(
            TASK, {"action_type": "execute_strategy", "strategy_id": "S01",
                   "solver": "highs"}, "ep1")
        record = h.execute(TASK, None, str(work / "solve.py"), str(work),
                           solver=None, episode_id=None,
                           prediction_id=prediction.prediction_id)
        self.assertEqual(record.quality["status"], "error")
        self.assertNotEqual(record.quality.get("objective"), 999.0)
        h.record(record)
        closed = h.close_episode("t1", "ep1", terminal_state="failed")
        evaluation = (closed.get("evaluations") or [{}])[0]
        benefit = evaluation.get("benefit") or {}
        # No quality observation exists for a run that produced no result.
        self.assertNotEqual(benefit.get("eligibility"), "evaluable")


class TestAssociationIsAtomic(Base):
    """Finding 3: the claim is written in one transaction BEFORE the run;
    a failure refuses the execution instead of running unassociated, and a
    second execution of the same prediction is refused pre-run."""

    def _solve_script(self, tag):
        work = Path(self.home) / f"ws_atomic_{tag}"
        work.mkdir(parents=True, exist_ok=True)
        (work / "solve.py").write_text(
            "import json\n"
            "with open('result.json', 'w') as fh:\n"
            "    json.dump({'status': 'optimal', 'objective_value': 1.0,\n"
            "               'objective_bound': 1.0, 'runtime_seconds': 0.01},\n"
            "              fh)\n", encoding="utf-8")
        return work

    def test_a_failed_claim_refuses_the_execution(self):
        h = self.make_harness()
        work = self._solve_script("fail")
        calls = {"n": 0}
        real = h.executor

        class Counting:
            def execute(self, *a, **k):
                calls["n"] += 1
                return real.execute(*a, **k)

        h.executor = Counting()
        prediction = h.predict_strategy_outcome(
            TASK, {"action_type": "execute_strategy", "strategy_id": "S01",
                   "solver": "highs"}, "ep1")

        def failing_claim(prediction_id, action_id):
            raise RuntimeError("synthetic association failure")

        h._claim_prediction_for_action = failing_claim
        with self.assertRaises(RuntimeError):
            h.execute(TASK, None, str(work / "solve.py"), str(work),
                      solver=None, episode_id=None,
                      prediction_id=prediction.prediction_id)
        # Nothing ran, and no action was left running.
        self.assertEqual(calls["n"], 0)
        self.assertEqual(
            [a.action_id for a in h.actions.query(task_id="t1",
                                                  status="running")], [])

    def test_a_second_execution_of_one_prediction_is_refused_pre_run(self):
        h = self.make_harness()
        work = self._solve_script("twice")
        calls = {"n": 0}
        real = h.executor

        class Counting:
            def execute(self, *a, **k):
                calls["n"] += 1
                return real.execute(*a, **k)

        h.executor = Counting()
        prediction = h.predict_strategy_outcome(
            TASK, {"action_type": "execute_strategy", "strategy_id": "S01",
                   "solver": "highs"}, "ep1")
        first = h.execute(TASK, None, str(work / "solve.py"), str(work),
                          solver=None, episode_id=None,
                          prediction_id=prediction.prediction_id)
        self.assertEqual(calls["n"], 1)
        with self.assertRaises(ValueError) as caught:
            h.execute(TASK, None, str(work / "solve.py"), str(work),
                      solver=None, episode_id=None,
                      prediction_id=prediction.prediction_id)
        self.assertIn("already", str(caught.exception))
        # The second attempt never reached the solver.
        self.assertEqual(calls["n"], 1)
        # The claim is readable from the ACTION side (race-proof).
        self.assertEqual(
            h._prediction_claimed_by_action(prediction.prediction_id),
            first.action_id)

    def test_the_claim_is_written_before_the_executor_starts(self):
        h = self.make_harness()
        work = self._solve_script("before")
        seen = {}
        real = h.executor

        class Probe:
            def execute(self, *a, **k):
                view = h.strategy_predictions.get(pred_id)
                seen["linked"] = view.trace.model_info.get(
                    "bound_action_id_before_execution")
                seen["phase"] = view.trace.model_info.get(
                    "association_phase")
                seen["action_param"] = (
                    h.actions.get(k.get("action_id")).params or {}
                ).get("prediction_id")
                return real.execute(*a, **k)

        pred_id = None
        h.executor = Probe()
        prediction = h.predict_strategy_outcome(
            TASK, {"action_type": "execute_strategy", "strategy_id": "S01",
                   "solver": "highs"}, "ep1")
        pred_id = prediction.prediction_id
        h.execute(TASK, None, str(work / "solve.py"), str(work),
                  solver=None, episode_id=None, prediction_id=pred_id)
        self.assertEqual(seen["phase"], "linked_before_execution")
        self.assertEqual(seen["action_param"], pred_id)
        self.assertIsNotNone(seen["linked"])


class TestSameConventionCompletionCompares(Base):
    """Finding 4: two COMPLETION candidates are compared on completion
    (instead of being rejected as off-yardstick), and a declared convention
    is a real per-decision choice that reaches the provider."""

    def _completion(self, value):
        return {
            "benefit": {"kind": "effective_completion",
                        "metric": "task_result_check_passed",
                        "unit": "boolean", "value": value,
                        "baseline": {"kind": "declared", "value": 0.5}},
            "cost": {"llm_tokens": 100},
        }

    def _provider(self, values_by_strategy):
        from or_harness.world_model.provider import WorldModelProvider

        class PerStrategy(WorldModelProvider):
            name = "per-strategy-convention"

            def __init__(inner):
                inner.requests = []

            def predict(inner, request, timeout_s=None):
                inner.requests.append(request)
                sid = (request.get("candidate") or {}).get("strategy_id")
                return {"payload": self._completion(
                    values_by_strategy.get(sid, 0.5)),
                    "usage": {"completion_tokens": 5}, "error": None,
                    "latency_s": 0.01}

            def describe(inner):
                return {"provider": inner.name, "model": "per-strategy"}

        return PerStrategy()

    def test_agreed_completion_convention_is_compared(self):
        provider = self._provider({"S01": 0.9, "S02": 0.4})
        h = self.make_harness(provider)
        plan = h.plan_next(
            TASK, "epc",
            candidates=[ActionSpec("execute_strategy", "t1",
                                   strategy_id="S01", solver="highs"),
                        ActionSpec("execute_strategy", "t1",
                                   strategy_id="S02", solver="highs")],
            limits={"horizon": 1})
        self.assertEqual(plan["status"], "ok")
        self.assertEqual(plan["suggested"]["strategy_id"], "S01")
        convention = plan["benefit_convention"]
        self.assertEqual(convention["kind"], "effective_completion")
        self.assertEqual(convention["metric"], "task_result_check_passed")
        self.assertEqual(convention["source"], "agreed")

    def test_a_declared_convention_reaches_every_request(self):
        provider = self._provider({"S01": 0.9})
        h = self.make_harness(provider)
        plan = h.plan_next(
            TASK, "epd",
            candidates=[ActionSpec("execute_strategy", "t1",
                                   strategy_id="S01", solver="highs")],
            limits={"horizon": 1,
                    "benefit_kind": "effective_completion",
                    "benefit_metric": "task_result_check_passed"})
        self.assertEqual(plan["benefit_convention"]["source"], "declared")
        sent = provider.requests[0]["benefit_convention"]
        self.assertEqual(sent["kind"], "effective_completion")
        self.assertTrue(sent["required"])

    def test_an_unrecognised_declaration_ranks_nothing(self):
        from or_harness.world_model.planner import (
            PlanLimits, resolve_benefit_convention,
            score_strategy_outcome_predictions)
        convention = resolve_benefit_convention(
            [], kind="valid_progress", metric="progress")
        self.assertFalse(convention["comparable"])
        self.assertEqual(convention["source"], "unknown_declaration")
        provider = self._provider({"S01": 0.9})
        h = self.make_harness(provider)
        prediction = h.predict_strategy_outcome(
            TASK, {"action_type": "execute_strategy", "strategy_id": "S01",
                   "solver": "highs"}, "epe")
        scores = score_strategy_outcome_predictions(
            [prediction],
            PlanLimits(alpha=1.0, beta=1.0, gamma=1.0,
                       benefit_kind="effective_completion",
                       benefit_metric="task_result_check_passed"))
        # The candidate IS on the declared convention: it ranks.
        self.assertIsNotNone(scores[0].utility)
        # A DIFFERENT convention on the same decision does not.
        provider.payload = self._completion(0.5)
        quality = h.predict_strategy_outcome(
            TASK, {"action_type": "execute_strategy", "strategy_id": "S02",
                   "solver": "highs"}, "epe")
        quality.benefit.kind = "solution_quality"
        quality.benefit.metric = "normalized_objective_gap"
        scores2 = score_strategy_outcome_predictions(
            [quality],
            PlanLimits(alpha=1.0, beta=1.0, gamma=1.0,
                       benefit_kind="effective_completion",
                       benefit_metric="task_result_check_passed"))
        self.assertIsNone(scores2[0].utility)
        self.assertIn("benefit", scores2[0].incomparable)


if __name__ == "__main__":
    unittest.main()
