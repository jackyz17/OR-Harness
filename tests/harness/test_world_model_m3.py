"""M3 tests: bounded planning and the plan -> choose -> execute -> bind loop.

Semantic-risk focused (no per-field mirror tests). The provider fixtures
here are LABELLED synthetic stubs: they answer from a declared map, never
from a hidden quality law. Nothing in this file claims real-LLM behaviour.

The comparison protocol is the wm-so/1 strategy-outcome one (the ONLY one
planning speaks). The legacy ``OutcomePrediction`` rollout — horizon 2, the
imagined successor state, the old ``outcome_status``/``quality`` payload —
was REMOVED, so the fixtures now return wm-so/1 payloads
(``benefit``/``cost``/``risk``/``uncertainty``) and the tests assert the
properties that matter for THIS protocol: one frozen context per decision,
a conservative yardstick, the prediction id handed over for binding, and
honest failure/truncation reporting.
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
from or_harness.core.storage import StorageError  # noqa: E402
from or_harness.world_model.prediction import ActionSpec  # noqa: E402
from or_harness.world_model.provider import WorldModelProvider  # noqa: E402

TASK = {"task_id": "t1", "family": "routing", "description": "toy"}


def _spec(strategy_id, task_id="t1", episode_id="ep1", **params):
    return ActionSpec(action_type="execute_strategy", task_id=task_id,
                      episode_id=episode_id, strategy_id=strategy_id,
                      params=params)


def _payload(quality=0.5, cost=None, risk=None, extra=None):
    """A well-formed wm-so/1 payload from declared numbers.

    ``quality`` becomes the benefit VALUE with a fixed baseline, so the
    comparison's one comparable currency (normalized solution quality) is
    what the fixture varies. ``cost`` is a per-dimension dict, ``risk`` a
    probability (None = "no risk predicted", which the yardstick charges
    in full).
    """
    payload = {
        "benefit": {"kind": "solution_quality",
                    "metric": "normalized_objective_gap",
                    "unit": "1-gap",
                    "value": float(quality),
                    "baseline": {"kind": "conditional_stats", "value": 0.0}},
        "uncertainty": {"execution_randomness": 0.2, "knowledge_gap": 0.3},
    }
    if cost is not None:
        payload["cost"] = dict(cost)
    if risk is not None:
        payload["risk"] = {"events": [{"event": "timeout",
                                       "probability": float(risk),
                                       "basis": "declared fixture"}]}
    if extra:
        payload.update(extra)
    return payload


class ScriptableProvider(WorldModelProvider):
    """Synthetic wm-so/1 provider with explicit, labelled rules.

    Rules (deterministic, declared in the test):
    - per strategy id: benefit value, cost dimensions, risk probability;
    - ids in ``fail`` return no payload (a recorded failed call, with the
      usage it consumed);
    - ids in ``invalid`` return a structurally broken payload, so the
      invalid-output path is exercised without a provider exception.
    """

    name = "scriptable-so"

    def __init__(self, per_strategy=None, fail=(), invalid=(),
                 usage=None):
        self.per_strategy = per_strategy or {}
        self.fail = set(fail)
        self.invalid = set(invalid)
        self.usage = usage if usage is not None else {
            "prompt_tokens": 100, "completion_tokens": 50}
        self.requests = []

    def predict(self, request, timeout_s=None):
        self.requests.append(request)
        sid = (request.get("candidate") or {}).get("strategy_id")
        if sid in self.fail:
            return {"payload": None, "usage": {"completion_tokens": 10},
                    "error": "synthetic timeout", "latency_s": 0.02}
        if sid in self.invalid:
            return {"payload": {"benefit": {"metric": "normalized_gap",
                                            "value": 5.0}},
                    "usage": {"completion_tokens": 30},
                    "error": None, "latency_s": 0.01}
        entry = self.per_strategy.get(sid, {})
        return {"payload": _payload(quality=entry.get("quality", 0.5),
                                    cost=entry.get("cost"),
                                    risk=entry.get("risk")),
                "usage": dict(self.usage),
                "error": None, "latency_s": 0.01}


class TestM2Closeout(HarnessTestCase):
    """F1-F3 regressions: identity, replay, strict binding."""

    def setUp(self):
        super().setUp()
        # The legacy M2 API is exercised THROUGH its Python entry points:
        # the CLI/agent surface is the wm-so/1 channel, but the stored-payload
        # reader and the knowledge-channel verdicts still go through the
        # legacy objects, so their identity discipline is still tested.
        self.h = ORHarness(home=self.home,
                           world_model=ScriptableProvider(
                               per_strategy={"S01": {"quality": 0.8}}))
        self.addCleanup(self.h.close)

    def test_prediction_episode_identity_no_budget_leak(self):
        """F1: a prediction made under ep1 must not appear in ep2's budget."""
        spec = _spec("S01", episode_id=None)  # caller forgot the episode
        prediction = self.h.predict_outcome(TASK, spec, "ep1")
        # Identity reconciled: the spec now carries ep1.
        self.assertEqual(prediction.action_spec.episode_id, "ep1")
        self.assertEqual(prediction.action_spec.task_id, "t1")
        ep2 = self.h.budget_view("t1", "ep2")
        self.assertEqual(ep2["consumption"]["prediction_call_costs"], [])
        ep1 = self.h.budget_view("t1", "ep1")
        self.assertEqual(len(ep1["consumption"]["prediction_call_costs"]), 1)

    def test_unattributable_prediction_reported_not_charged(self):
        """F1: a legacy prediction with episode=None is reported separately,
        never wildcard-charged to every episode."""
        spec = _spec("S01", episode_id=None)
        prediction = self.h.predictions.predict_outcome(
            TASK, spec, self.h.snapshot(TASK, None))  # episode-less call
        self.assertIsNone(prediction.action_spec.episode_id)
        view = self.h.budget_view("t1", "ep1")
        self.assertEqual(view["consumption"]["prediction_call_costs"], [])
        unattributed = view["consumption"]["unattributed_prediction_costs"]
        self.assertEqual(len(unattributed), 1)
        self.assertEqual(unattributed[0]["prediction_id"],
                         prediction.prediction_id)

    def test_end_action_replay_preserves_amended_cost(self):
        """F2: begin -> predict(parent) -> end(None) -> end(None) replays
        idempotently instead of conflicting on the amended cost."""
        begun = self.h.begin_action("select_strategy", TASK, "ep1")
        action_id = begun["action_id"]
        self.h.predict_outcome(TASK, _spec("S01"), "ep1",
                               parent_action_id=action_id)
        first = self.h.end_action(action_id, outcome={"note": "done"})
        second = self.h.end_action(action_id, outcome={"note": "done"})
        self.assertEqual(first["action_id"], second["action_id"])
        record = self.h.actions.get(action_id)
        self.assertGreater(record.cost.llm_tokens, 0)  # amended cost kept
        # A DIFFERENT outcome still conflicts.
        with self.assertRaises(StorageError):
            self.h.end_action(action_id, outcome={"note": "changed"})

    def test_bind_scope_both_directions(self):
        """F3: an attempt-scope prediction is never scored against a
        task-scope record, and vice versa (both directions)."""
        spec = _spec("S01")
        spec.measurement_scope = "task"
        prediction = self.h.predict_outcome(TASK, spec, "ep1")
        record = self.make_record(task_id="t1")
        record.measurement_scope = "attempt"
        self.h.bank.append(record)
        action = self.h.actions.begin_action(
            "execute_strategy", "t1", "ep1",
            pre_snapshot=self.h.snapshot(TASK, "ep1"),
            params={"strategy_id": "S01"})
        self.h.actions.end_action(
            action.action_id, status="completed",
            outcome={"execution_status": "optimal"},
            cost=record.cost, linked_execution_id=record.execution_id,
            rollup="reference")
        bound = self.h.bind_outcome(prediction.prediction_id,
                                    action.action_id)
        self.assertIn("measurement_scope", bound.binding_mismatch)


class TestPlanning(HarnessTestCase):
    """Candidates, one shared context, conservative comparison, the
    suggestion/choice split, and honest failure reporting."""

    def test_suggestion_and_cost_attribution(self):
        provider = ScriptableProvider(per_strategy={
            "S01": {"quality": 0.9, "cost": {"solver_runtime_s": 10.0}},
            "S02": {"quality": 0.6, "cost": {"solver_runtime_s": 1.0}},
        })
        h = ORHarness(home=self.home, world_model=provider)
        self.addCleanup(h.close)
        plan = h.plan_next(TASK, "ep1",
                           candidates=[_spec("S01"), _spec("S02")],
                           limits={"horizon": 1})
        self.assertEqual(plan["status"], "ok")
        self.assertEqual(plan["model_calls_made"], 2)
        self.assertIsNotNone(plan["suggested"])
        # ONE frozen context for the whole decision: the comparison's
        # candidates were all conditioned on the same input.
        self.assertTrue(plan["prediction_context_id"])
        # Every compared candidate carries its prediction id — this is what
        # the caller hands to `execute --prediction` instead of predicting
        # the chosen candidate a second time.
        ids = [c["prediction_id"] for c in plan["candidates"]]
        self.assertEqual(len(ids), 2)
        self.assertTrue(all(i and i.startswith("sp_") for i in ids))
        # Real planning spend charged ONCE to the decision action.
        decision = h.actions.get(plan["decision_action_id"])
        self.assertEqual(decision.cost.llm_tokens, 100.0)  # 2 x 50
        self.assertEqual(plan["planning_cost"]["cost"]["llm_tokens"], 100.0)
        # Planning spend is NOT part of any candidate's utility.
        for candidate in plan["candidates"]:
            self.assertNotIn("llm_tokens",
                             json.dumps(candidate["score"]))

    def test_horizon_two_is_refused(self):
        """There is ONE protocol and it plans at horizon=1. A two-step
        request raises with the replacement path named — never a silent
        downgrade to a different comparison."""
        provider = ScriptableProvider(per_strategy={"S01": {"quality": 0.8}})
        h = ORHarness(home=self.home, world_model=provider)
        self.addCleanup(h.close)
        with self.assertRaises(ValueError) as ctx:
            h.plan_next(TASK, "ep1", candidates=[_spec("S01")],
                        limits={"horizon": 2})
        self.assertIn("horizon=1 only", str(ctx.exception))
        self.assertIn("REAL observation", str(ctx.exception))
        # No model call was made for the refused request.
        self.assertEqual(provider.requests, [])

    def test_suggestion_is_not_a_selection(self):
        """Only an explicit choose_next writes X.selected_plan."""
        provider = ScriptableProvider(per_strategy={"S01": {"quality": 0.8}})
        h = ORHarness(home=self.home, world_model=provider)
        self.addCleanup(h.close)
        plan = h.plan_next(TASK, "ep1", candidates=[_spec("S01")])
        progress = h._episode_progress("t1", "ep1")
        self.assertNotIn("selected_plan", progress)
        choice = h.choose_next(plan["decision_action_id"],
                               chosen=ActionSpec.from_dict(
                                   plan["suggested"]))
        self.assertEqual(choice["selected"]["strategy_id"], "S01")
        progress = h._episode_progress("t1", "ep1")
        self.assertIn("selected_plan", progress)
        self.assertEqual(progress["selected_plan"]["value"]["selected"]
                         ["strategy_id"], "S01")

    def test_choice_deviation_and_rejection(self):
        provider = ScriptableProvider(per_strategy={
            "S01": {"quality": 0.9}, "S02": {"quality": 0.3}})
        h = ORHarness(home=self.home, world_model=provider)
        self.addCleanup(h.close)
        plan = h.plan_next(TASK, "ep1",
                           candidates=[_spec("S01"), _spec("S02")])
        # Deviation: the agent picks the non-suggested candidate.
        choice = h.choose_next(plan["decision_action_id"],
                               chosen=_spec("S02"),
                               deviation_note="domain reason")
        self.assertIsNotNone(choice["deviation"])
        # Rejection on a fresh decision.
        plan2 = h.plan_next(TASK, "ep1",
                            candidates=[_spec("S01"), _spec("S02")])
        rejected = h.choose_next(plan2["decision_action_id"], rejected=True,
                                 deviation_note="budget too tight")
        self.assertTrue(rejected["rejected"])

    def test_rejection_preserves_earlier_selection(self):
        """R3 regression: choosing S01, then receiving an S04 suggestion
        and REJECTING it must leave X.selected_plan = S01. The rejection
        is recorded as its own queryable event, never as a selection
        overwrite."""
        provider = ScriptableProvider(per_strategy={
            "S01": {"quality": 0.9}, "S04": {"quality": 0.85}})
        h = ORHarness(home=self.home, world_model=provider)
        self.addCleanup(h.close)
        plan1 = h.plan_next(TASK, "ep1", candidates=[_spec("S01")])
        h.choose_next(plan1["decision_action_id"], chosen=_spec("S01"))
        snap1 = h.snapshot(TASK, "ep1")
        selected = (snap1.task_progress.get("selected_plan") or {}) \
            .get("value", {}).get("selected", {})
        self.assertEqual(selected.get("strategy_id"), "S01")
        plan2 = h.plan_next(TASK, "ep1", candidates=[_spec("S04")])
        rejected = h.choose_next(plan2["decision_action_id"],
                                 rejected=True,
                                 deviation_note="keeping S01")
        self.assertTrue(rejected["rejected"])
        snap2 = h.snapshot(TASK, "ep1")
        selected_after = (snap2.task_progress.get("selected_plan") or {}) \
            .get("value", {}).get("selected", {})
        self.assertEqual(selected_after.get("strategy_id"), "S01")
        rejection = (snap2.task_progress.get("last_rejected_suggestion")
                     or {})
        self.assertEqual(rejection.get("provenance"), "agent_reported")
        self.assertTrue((rejection.get("value") or {}).get("rejected"))

    def test_budget_bounds_and_unknown_not_claimed_ok(self):
        provider = ScriptableProvider(
            per_strategy={f"S0{i}": {"quality": 0.5} for i in range(1, 7)})
        h = ORHarness(home=self.home, world_model=provider)
        self.addCleanup(h.close)
        # max_model_calls bounds the evaluation (and max_root_candidates
        # bounds how many candidates are even considered).
        plan = h.plan_next(TASK, "ep1",
                           candidates=[_spec(f"S0{i}") for i in range(1, 5)],
                           limits={"max_model_calls": 2})
        self.assertEqual(plan["model_calls_made"], 2)
        self.assertEqual(plan["status"], "truncated")
        # Unknown budget is never reported as confirmed-ok.
        self.assertEqual(plan["budget_confirmation"], "unknown")
        # Exceeded REAL budget stops planning before any model call.
        h.declare_budget("t1", {"llm_tokens": 10}, episode_id="ep1")
        calls_before = len(provider.requests)
        plan2 = h.plan_next(TASK, "ep1", candidates=[_spec("S01")])
        self.assertEqual(plan2["status"], "fallback")
        self.assertEqual(len(provider.requests), calls_before)

    def test_no_candidates_and_unsupported_types(self):
        provider = ScriptableProvider()
        h = ORHarness(home=self.home, world_model=provider)
        self.addCleanup(h.close)
        # verify/finish_task etc. are honestly reported as not plannable.
        spec = ActionSpec(action_type="verify", task_id="t1",
                          episode_id="ep1")
        plan = h.plan_next(TASK, "ep1", candidates=[spec])
        self.assertEqual(plan["status"], "no_candidates")
        # planning=False disables the planner explicitly.
        h2 = ORHarness(home=self.home, world_model=provider,
                       planning=False)
        self.addCleanup(h2.close)
        plan2 = h2.plan_next(TASK, "ep1", candidates=[_spec("S01")])
        self.assertEqual(plan2["status"], "disabled")

    def test_failed_call_cost_preserved(self):
        """A failed model call still records its known usage (10 tokens)
        and the candidate is reported, not silently dropped."""
        provider = ScriptableProvider(
            per_strategy={"S01": {"quality": 0.8}}, fail={"S02"})
        h = ORHarness(home=self.home, world_model=provider)
        self.addCleanup(h.close)
        plan = h.plan_next(TASK, "ep1",
                           candidates=[_spec("S01"), _spec("S02")],
                           limits={"horizon": 1})
        scores = {c["action_spec"]["strategy_id"]: c["score"]
                  for c in plan["candidates"]}
        self.assertIsNone(scores["S02"]["utility"])
        self.assertIn("prediction", scores["S02"]["incomparable"])
        self.assertEqual(plan["suggested"]["strategy_id"], "S01")
        decision = h.actions.get(plan["decision_action_id"])
        # 50 (S01) + 10 (failed S02) = 60 tokens of REAL spend.
        self.assertEqual(decision.cost.llm_tokens, 60.0)

    def test_all_predictions_invalid_reports_honestly(self):
        """When EVERY candidate's prediction is unusable, status must NOT
        be "ok" — the caller would mistake "model output unusable" for a
        quiet "no recommendation". Reported as no_valid_predictions with
        the real statuses; the calls still happened and their cost is
        recorded."""
        provider = ScriptableProvider(fail={"S01", "S02"})
        h = ORHarness(home=self.home, world_model=provider)
        self.addCleanup(h.close)
        plan = h.plan_next(TASK, "ep1",
                           candidates=[_spec("S01"), _spec("S02")],
                           limits={"horizon": 1})
        self.assertEqual(plan["status"], "no_valid_predictions")
        self.assertIsNone(plan["suggested"])
        self.assertIn("no candidate carried a usable", plan["truncation_reason"])
        self.assertEqual(plan["model_calls_made"], 2)
        decision = h.actions.get(plan["decision_action_id"])
        self.assertEqual(decision.cost.llm_tokens, 20.0)  # 2 x 10 (failed)

    def test_malformed_payload_becomes_invalid_not_a_crash(self):
        """A structurally valid JSON payload whose content violates the
        contract (a benefit value outside [0,1] for normalized quality)
        becomes an explicit invalid prediction with its known cost kept —
        it never crashes the planning loop or loses the other candidate."""
        provider = ScriptableProvider(per_strategy={"S01": {"quality": 0.9}},
                                      invalid={"S02"})
        h = ORHarness(home=self.home, world_model=provider)
        self.addCleanup(h.close)
        plan = h.plan_next(TASK, "ep1",
                           candidates=[_spec("S01"), _spec("S02")],
                           limits={"horizon": 1})
        self.assertEqual(plan["model_calls_made"], 2)
        statuses = sorted(c["prediction_status"] for c in plan["candidates"])
        self.assertEqual(statuses, ["invalid", "valid"])
        decision = h.actions.get(plan["decision_action_id"])
        self.assertNotEqual(decision.status, "running")
        self.assertEqual(decision.status, "completed")
        self.assertEqual(decision.cost.llm_tokens, 80.0)  # 50 + 30
        # The valid candidate is still suggested.
        self.assertEqual(plan["suggested"]["strategy_id"], "S01")

    def test_provider_exception_is_a_recorded_provider_error(self):
        """A provider that RAISES is a recorded prediction failure, never
        an exception escaping into the caller: the decision ends (not
        running), and the other candidate still yields a suggestion."""

        class CrashProvider(WorldModelProvider):
            name = "crash"

            def __init__(self):
                self.calls = 0

            def predict(self, request, timeout_s=None):
                self.calls += 1
                if self.calls == 2:
                    raise RuntimeError("synthetic internal crash")
                return {"payload": _payload(quality=0.9),
                        "usage": {"completion_tokens": 50},
                        "error": None, "latency_s": 0.01}

        provider = CrashProvider()
        h = ORHarness(home=self.home, world_model=provider)
        self.addCleanup(h.close)
        plan = h.plan_next(TASK, "ep1",
                           candidates=[_spec("S01"), _spec("S02")],
                           limits={"horizon": 1})
        decision = h.actions.get(plan["decision_action_id"])
        self.assertNotEqual(decision.status, "running")
        statuses = sorted(c["prediction_status"] for c in plan["candidates"])
        self.assertEqual(statuses, ["provider_error", "valid"])

    def test_shadow_mode_withholds_suggestion(self):
        provider = ScriptableProvider(per_strategy={"S01": {"quality": 0.8}})
        h = ORHarness(home=self.home, world_model=provider,
                      plan_mode="shadow")
        self.addCleanup(h.close)
        plan = h.plan_next(TASK, "ep1", candidates=[_spec("S01")])
        self.assertIsNone(plan["suggested"])
        self.assertTrue(any(c["score"]["utility"] is not None
                            for c in plan["candidates"]))


class TestExecutionReplan(HarnessTestCase):
    """Closed loop: plan -> choose -> real executor run -> record -> re-plan
    sees the new fact. The prediction is bound AUTOMATICALLY from the id
    `plan-next` returned — the candidate is never predicted twice."""

    def test_replan_uses_new_real_state_and_binds_the_plan_prediction(self):
        provider = ScriptableProvider(per_strategy={
            "S01": {"quality": 0.5}, "S02": {"quality": 0.6}})
        h = ORHarness(home=self.home, world_model=provider)
        self.addCleanup(h.close)
        plan = h.plan_next(TASK, "ep1",
                           candidates=[_spec("S01"), _spec("S02")])
        h.choose_next(plan["decision_action_id"],
                      chosen=ActionSpec.from_dict(plan["suggested"]))
        executed_strategy = plan["suggested"]["strategy_id"]
        # The prediction id for the CHOSEN candidate, taken from the plan —
        # NOT re-predicted.
        chosen_pred_id = next(
            c["prediction_id"] for c in plan["candidates"]
            if c["action_spec"]["strategy_id"] == executed_strategy)
        calls_before = len(provider.requests)
        # REAL executor run (synthetic solve script — a labelled fixture,
        # not a real OR solve): writes a feasible result.
        workspace = Path(self.home) / "ws"
        workspace.mkdir()
        script = workspace / "solve.py"
        script.write_text(
            "import json\n"
            "json.dump({'status': 'feasible', 'objective_value': 120.0},\n"
            "          open('result.json', 'w'))\n")
        record = h.execute(TASK, executed_strategy,
                           str(script), str(workspace), solver="highs",
                           episode_id="ep1", prediction_id=chosen_pred_id)
        self.assertTrue(record.quality["feasible"])
        # NO further model call: the plan's prediction was reused.
        self.assertEqual(len(provider.requests), calls_before)
        # The binding happened automatically and is on the record.
        binding = record.execution_features["prediction_binding"]
        self.assertTrue(binding["bound"])
        self.assertIsNone(binding["trace"]["binding_mismatch"])
        self.assertTrue(binding["comparable"])
        # The stored prediction really points at the real action.
        stored = h.read_prediction(chosen_pred_id)
        self.assertEqual(
            stored["prediction"]["trace"]["model_info"]["bound_action_id"],
            record.action_id)
        h.record(record)
        # Re-plan: the new root snapshot carries the real current_solution.
        plan2 = h.plan_next(TASK, "ep1",
                            candidates=[_spec("S01"), _spec("S02")])
        root = h.get_snapshot(plan2["root_snapshot_id"])
        self.assertIn("current_solution", root.task_progress)
        self.assertEqual(root.task_progress["current_solution"]
                         ["epistemic"], "fact")

    def test_a_mismatched_prediction_is_recorded_not_scored(self):
        """Automatic binding is IDENTITY-CHECKED, never name-guessed: an
        attempt that names another strategy's prediction binds with a
        mismatch, and the execution is still recorded."""
        provider = ScriptableProvider(per_strategy={
            "S01": {"quality": 0.5}, "S02": {"quality": 0.6}})
        h = ORHarness(home=self.home, world_model=provider)
        self.addCleanup(h.close)
        prediction = h.predict_strategy_outcome(
            TASK, {"action_type": "execute_strategy", "strategy_id": "S02"},
            "ep1")
        workspace = Path(self.home) / "ws2"
        workspace.mkdir()
        script = workspace / "solve.py"
        script.write_text(
            "import json\n"
            "json.dump({'status': 'feasible', 'objective_value': 120.0},\n"
            "          open('result.json', 'w'))\n")
        record = h.execute(TASK, "S01",   # a DIFFERENT strategy ran
                           str(script), str(workspace), solver="highs",
                           episode_id="ep1",
                           prediction_id=prediction.prediction_id)
        binding = record.execution_features["prediction_binding"]
        self.assertTrue(binding["bound"])
        self.assertIn("strategy_id", binding["trace"]["binding_mismatch"])
        self.assertFalse(binding["comparable"])

    def test_a_bad_prediction_id_never_loses_the_execution(self):
        """A binding problem is BOOKKEEPING: the execution really happened
        and its evidence must survive. The failure is reported on the
        record and `bind-strategy` can bind later."""
        provider = ScriptableProvider(per_strategy={"S01": {"quality": 0.5}})
        h = ORHarness(home=self.home, world_model=provider)
        self.addCleanup(h.close)
        workspace = Path(self.home) / "ws3"
        workspace.mkdir()
        script = workspace / "solve.py"
        script.write_text(
            "import json\n"
            "json.dump({'status': 'feasible', 'objective_value': 120.0},\n"
            "          open('result.json', 'w'))\n")
        record = h.execute(TASK, "S01", str(script), str(workspace),
                           solver="highs", episode_id="ep1",
                           prediction_id="wp_does_not_exist")
        self.assertTrue(record.quality["feasible"])
        binding = record.execution_features["prediction_binding"]
        self.assertFalse(binding["bound"])
        self.assertIn("bind-strategy", binding["reason"])
        # The execution is still recordable.
        h.record(record)
        self.assertEqual(h.bank.count(), 1)


class TestReviewFixes(HarnessTestCase):
    """Regression tests for reviewed defects in the comparison."""

    def test_cost_weights_inherited_from_harness(self):
        """Planning inherits the harness's alpha/beta/gamma and cost
        weights; an expensive candidate loses to a cheap one when quality
        and risk tie."""
        provider = ScriptableProvider(per_strategy={
            "S01": {"quality": 0.8, "cost": {"solver_runtime_s": 100.0},
                    "risk": 0.1},
            "S02": {"quality": 0.8, "cost": {"solver_runtime_s": 1.0},
                    "risk": 0.1},
        })
        h = ORHarness(home=self.home, world_model=provider,
                      beta=1.0, cost_weights={"solver_runtime_s": 1.0})
        self.addCleanup(h.close)
        plan = h.plan_next(TASK, "ep1",
                           candidates=[_spec("S01"), _spec("S02")],
                           limits={"horizon": 1})
        self.assertEqual(plan["limits"]["cost_weights"],
                         {"solver_runtime_s": 1.0})
        self.assertEqual(plan["suggested"]["strategy_id"], "S02")
        scores = {c["action_spec"]["strategy_id"]: c["score"]
                  for c in plan["candidates"]}
        self.assertGreater(scores["S01"]["cost_normalized"],
                           scores["S02"]["cost_normalized"])
        # Explicit override wins over inheritance.
        plan2 = h.plan_next(TASK, "ep1",
                            candidates=[_spec("S01"), _spec("S02")],
                            limits={"horizon": 1, "cost_weights": {}})
        self.assertEqual(plan2["limits"]["cost_weights"], {})
        self.assertEqual(plan2["suggested"]["strategy_id"], "S01")

    def test_unknown_risk_and_cost_never_auto_win(self):
        """Unknown failure risk is a deficit (charged full gamma), and an
        unpredicted cost dimension is not a predicted zero."""
        provider = ScriptableProvider(per_strategy={
            "S01": {"quality": 0.8, "risk": 0.1},
            "S02": {"quality": 0.8},          # no risk predicted
        })
        h = ORHarness(home=self.home, world_model=provider)
        self.addCleanup(h.close)
        plan = h.plan_next(TASK, "ep1",
                           candidates=[_spec("S01"), _spec("S02")],
                           limits={"horizon": 1})
        self.assertEqual(plan["suggested"]["strategy_id"], "S01")
        scores = {c["action_spec"]["strategy_id"]: c["score"]
                  for c in plan["candidates"]}
        self.assertIn("risk", scores["S02"]["incomparable"])
        self.assertEqual(scores["S02"]["risk_effective"], 1.0)
        # Unknown cost: S01 predicts only tokens, S03 predicts nothing.
        provider2 = ScriptableProvider(per_strategy={
            "S03": {"quality": 0.8, "cost": {"llm_tokens": 10.0}, "risk": 0.1},
            "S04": {"quality": 0.8, "risk": 0.1},   # no cost predicted
        })
        h2 = ORHarness(home=self.home, world_model=provider2,
                       cost_weights={"llm_tokens": 1.0})
        self.addCleanup(h2.close)
        plan2 = h2.plan_next(TASK, "ep1",
                             candidates=[_spec("S03"), _spec("S04")],
                             limits={"horizon": 1})
        self.assertEqual(plan2["suggested"]["strategy_id"], "S03")
        scores2 = {c["action_spec"]["strategy_id"]: c["score"]
                   for c in plan2["candidates"]}
        self.assertIn("cost", scores2["S04"]["incomparable"])

    def test_unknown_cost_never_erases_measured_differences(self):
        """The A/B/C case: A and B have comparable measured costs (A
        expensive, B cheap), C predicts NO cost at all. The union basis
        keeps A/B's measured difference, and C is charged the peak
        normalized share (never free).

        Quality and risk tie across all three; the ONLY differentiator is
        cost, so B must win regardless of C's presence or ordering."""
        provider = ScriptableProvider(per_strategy={
            "S01": {"quality": 0.8, "risk": 0.1,
                    "cost": {"solver_runtime_s": 100.0}},   # A: expensive
            "S02": {"quality": 0.8, "risk": 0.1,
                    "cost": {"solver_runtime_s": 1.0}},     # B: cheap
            "S03": {"quality": 0.8, "risk": 0.1},           # C: cost unknown
        })
        h = ORHarness(home=self.home, world_model=provider,
                      beta=1.0, cost_weights={"solver_runtime_s": 1.0})
        self.addCleanup(h.close)
        plan_ab = h.plan_next(TASK, "ep1",
                              candidates=[_spec("S01"), _spec("S02")],
                              limits={"horizon": 1})
        self.assertEqual(plan_ab["suggested"]["strategy_id"], "S02")
        for ordering in (("S01", "S02", "S03"), ("S03", "S02", "S01")):
            plan = h.plan_next(TASK, "ep1",
                               candidates=[_spec(s) for s in ordering],
                               limits={"horizon": 1})
            self.assertEqual(plan["suggested"]["strategy_id"], "S02",
                             f"ordering {ordering}: C must not win by "
                             "staying silent on cost")
            scores = {c["action_spec"]["strategy_id"]: c["score"]
                      for c in plan["candidates"]}
            self.assertGreater(scores["S01"]["cost_normalized"],
                               scores["S02"]["cost_normalized"])
            self.assertGreater(scores["S03"]["cost_normalized"],
                               scores["S02"]["cost_normalized"])
            self.assertAlmostEqual(scores["S03"]["cost_normalized"],
                                   scores["S01"]["cost_normalized"],
                                   places=3)
            self.assertIn("cost", scores["S03"]["incomparable"])
        # An EXPLICIT predicted zero stays a measured zero (distinct from
        # unknown): a candidate predicting solver_runtime_s=0 is cheaper
        # than B and wins.
        provider_zero = ScriptableProvider(per_strategy={
            "S01": {"quality": 0.8, "risk": 0.1,
                    "cost": {"solver_runtime_s": 100.0}},
            "S02": {"quality": 0.8, "risk": 0.1,
                    "cost": {"solver_runtime_s": 0.0}},
        })
        h_zero = ORHarness(home=self.home, world_model=provider_zero,
                           beta=1.0, cost_weights={"solver_runtime_s": 1.0})
        self.addCleanup(h_zero.close)
        plan_zero = h_zero.plan_next(TASK, "ep1",
                                     candidates=[_spec("S01"), _spec("S02")],
                                     limits={"horizon": 1})
        self.assertEqual(plan_zero["suggested"]["strategy_id"], "S02")
        zero = next(c for c in plan_zero["candidates"]
                    if c["action_spec"]["strategy_id"] == "S02")
        self.assertEqual(zero["score"]["cost_normalized"], 0.0)
        self.assertNotIn("cost", zero["score"]["incomparable"])

    def test_budget_rechecked_and_exceeded_withholds_suggestion(self):
        """Planning re-checks the REAL budget per call and after the spend
        lands; an exceeded budget never returns ok."""
        provider = ScriptableProvider(
            per_strategy={f"S0{i}": {"quality": 0.5, "risk": 0.1}
                          for i in range(1, 5)})
        h = ORHarness(home=self.home, world_model=provider)
        self.addCleanup(h.close)
        # 3 candidates x 50 tokens each = 150 > 60 declared.
        h.declare_budget("t1", {"llm_tokens": 60.0}, episode_id="ep1")
        plan = h.plan_next(TASK, "ep1",
                           candidates=[_spec("S01"), _spec("S02"),
                                       _spec("S03")],
                           limits={"horizon": 1})
        self.assertNotEqual(plan["status"], "ok")
        self.assertEqual(plan["budget_confirmation"], "exceeded")
        self.assertIsNone(plan["suggested"])
        view = h.budget_view("t1", "ep1")
        self.assertEqual(view["status"], "exceeded")
        self.assertGreater(plan["model_calls_made"], 0)

    def test_time_budget_truncates(self):
        """A near-zero time budget truncates evaluation instead of
        reporting ok after slow calls."""
        class SlowProvider(ScriptableProvider):
            def predict(self, request, timeout_s=None):
                import time as _time
                _time.sleep(0.05)
                return super().predict(request, timeout_s)
        provider = SlowProvider(per_strategy={"S01": {"quality": 0.8},
                                              "S02": {"quality": 0.7}})
        h = ORHarness(home=self.home, world_model=provider)
        self.addCleanup(h.close)
        plan = h.plan_next(TASK, "ep1",
                           candidates=[_spec("S01"), _spec("S02")],
                           limits={"horizon": 1, "time_budget_s": 0.01})
        self.assertEqual(plan["status"], "truncated")
        self.assertIn("time", plan["truncation_reason"])

    def test_single_candidate_timeout_reports_truncated(self):
        """ONE candidate whose single call exceeds the time budget must NOT
        report ok — the post-call deadline check catches the over-budget
        return and reports truncation with the cost preserved."""
        class SlowProvider(ScriptableProvider):
            def predict(self, request, timeout_s=None):
                import time as _time
                _time.sleep(0.06)
                return super().predict(request, timeout_s)
        provider = SlowProvider(per_strategy={"S01": {"quality": 0.8}})
        h = ORHarness(home=self.home, world_model=provider)
        self.addCleanup(h.close)
        plan = h.plan_next(TASK, "ep1", candidates=[_spec("S01")],
                           limits={"horizon": 1, "time_budget_s": 0.01})
        self.assertEqual(plan["status"], "truncated")
        self.assertIn("time budget", plan["truncation_reason"])
        self.assertEqual(plan["model_calls_made"], 1)
        decision = h.actions.get(plan["decision_action_id"])
        self.assertEqual(decision.cost.llm_tokens, 50.0)

    def test_last_candidate_timeout_reports_truncated(self):
        """The LAST candidate's call exceeding the budget is caught by the
        post-call check (no further loop iteration would catch it)."""
        class SlowProvider(ScriptableProvider):
            def predict(self, request, timeout_s=None):
                import time as _time
                _time.sleep(0.06)
                return super().predict(request, timeout_s)
        provider = SlowProvider(per_strategy={"S01": {"quality": 0.8},
                                              "S02": {"quality": 0.7}})
        h = ORHarness(home=self.home, world_model=provider)
        self.addCleanup(h.close)
        plan = h.plan_next(TASK, "ep1",
                           candidates=[_spec("S01"), _spec("S02")],
                           limits={"horizon": 1, "time_budget_s": 0.08})
        self.assertEqual(plan["status"], "truncated")
        self.assertEqual(plan["model_calls_made"], 2)

    def test_candidate_identity_conflict_rejected(self):
        """A candidate naming another task/episode is rejected, never
        planned under this root state."""
        provider = ScriptableProvider(per_strategy={"S01": {"quality": 0.8}})
        h = ORHarness(home=self.home, world_model=provider)
        self.addCleanup(h.close)
        foreign = _spec("S01", task_id="other", episode_id="other_ep")
        plan = h.plan_next(TASK, "ep1", candidates=[foreign, _spec("S01")],
                           limits={"horizon": 1})
        self.assertIn("identity conflict", plan["truncation_reason"])
        self.assertEqual(len(plan["candidates"]), 1)
        self.assertEqual(
            plan["candidates"][0]["action_spec"]["task_id"], "t1")
        # All-conflicting => no candidates at all.
        plan2 = h.plan_next(TASK, "ep1", candidates=[foreign])
        self.assertEqual(plan2["status"], "no_candidates")

    def test_planning_basis_persisted_and_recoverable_after_reopen(self):
        """Closing and re-opening the harness must fully recover the
        planning decision — root snapshot, candidates with their scores,
        prediction refs, comparison context and suggestion. Later
        knowledge changes must NOT rewrite the historical judgment."""
        provider = ScriptableProvider(per_strategy={
            "S01": {"quality": 0.9, "risk": 0.1,
                    "cost": {"solver_runtime_s": 10.0}},
            "S02": {"quality": 0.6, "risk": 0.1,
                    "cost": {"solver_runtime_s": 1.0}},
        })
        h = ORHarness(home=self.home, world_model=provider,
                      beta=1.0, cost_weights={"solver_runtime_s": 1.0})
        plan = h.plan_next(TASK, "ep1",
                           candidates=[_spec("S01"), _spec("S02")],
                           limits={"horizon": 1})
        decision_id = plan["decision_action_id"]
        h.close()

        # Re-open with a FRESH harness instance (different alpha).
        h2 = ORHarness(home=self.home, alpha=99.0)
        self.addCleanup(h2.close)
        decision = h2.plan_decision(decision_id)
        self.assertIsNotNone(decision)
        outcome = decision.get("outcome") or {}
        self.assertEqual(outcome.get("plan_id"), plan["plan_id"])
        self.assertEqual(outcome.get("root_snapshot_id"),
                         plan["root_snapshot_id"])
        self.assertEqual(outcome.get("status"), "ok")
        self.assertEqual(len(outcome.get("candidates") or []), 2)
        # Prediction references are recoverable and point at stored records.
        first = next(c for c in outcome["candidates"]
                     if c["action_spec"]["strategy_id"] == "S01")
        self.assertTrue(first["prediction_id"].startswith("sp_"))
        self.assertIsNotNone(h2.read_prediction(first["prediction_id"]))
        # The frozen comparison scores are preserved as computed then.
        self.assertEqual(first["score"]["benefit_value"], 0.9)
        self.assertEqual(first["score"]["cost_normalized"], 1.0)
        # The shared input context is referenced, and still resolvable.
        self.assertTrue(outcome.get("prediction_context_id"))
        self.assertIsNotNone(
            h2.get_prediction_context(outcome["prediction_context_id"]))
        # Suggestion and its basis string.
        self.assertEqual((outcome.get("suggested") or {}).get("strategy_id"),
                         "S02")
        self.assertTrue(outcome.get("suggestion_basis"))


class TestPlanNextCli(HarnessTestCase):
    """The CLI hands the chosen candidate's prediction id over, so the
    agent never calls `predict-strategy` again after planning."""

    def _run(self, argv):
        from or_harness import cli
        buffer = io.StringIO()
        old = sys.stdout
        sys.stdout = buffer
        try:
            code = cli.main(["--home", self.home] + argv)
        finally:
            sys.stdout = old
        return code, buffer.getvalue()

    def test_horizon_two_is_refused_at_the_parser(self):
        code, _out = self._run([
            "plan-next", "--task", json.dumps(TASK),
            "--candidates", json.dumps([{"action_type": "execute_strategy",
                                         "strategy_id": "S01"}]),
            "--horizon", "2"])
        self.assertEqual(code, 2)

    def test_protocol_flag_is_gone(self):
        """The protocol selector was removed WITH the second protocol: a
        stale `--protocol` must fail loudly rather than be ignored."""
        with self.assertRaises(SystemExit) as ctx:
            self._run([
                "plan-next", "--task", json.dumps(TASK),
                "--candidates", json.dumps([
                    {"action_type": "execute_strategy",
                     "strategy_id": "S01"}]),
                "--protocol", "strategy-outcome"])
        self.assertEqual(ctx.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
