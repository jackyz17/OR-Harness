"""M3 tests: bounded planning, state-conditioned rollout, budget honesty.

Semantic-risk focused (no per-field mirror tests). The FakeProvider here is
a LABELLED synthetic fixture: it answers from the request content, never
from a hidden quality law. Nothing in this file claims real-LLM behaviour.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tests.harness.helpers import HarnessTestCase  # noqa: E402

from or_harness.api import ORHarness  # noqa: E402
from or_harness.core.storage import StorageError  # noqa: E402
from or_harness.world_model.prediction import ActionSpec  # noqa: E402
from or_harness.world_model.planner import PlanLimits  # noqa: E402
from or_harness.world_model.provider import WorldModelProvider  # noqa: E402


class ScriptableProvider(WorldModelProvider):
    """Synthetic provider with explicit, labelled rules (test fixture).

    Rules (deterministic, declared in the test):
    - quality/cost per strategy id from the constructor map;
    - when the request state already carries a current_solution (the
      hypothetical successor of a first step), the second-step quality is
      the map's ``step2_quality`` — proving the prediction READ the
      hypothetical post-state, not just the root.
    """

    name = "scriptable"

    def __init__(self, per_strategy=None, step2_quality=None, fail=()):
        self.per_strategy = per_strategy or {}
        self.step2_quality = step2_quality
        self.fail = set(fail)
        self.requests = []

    def predict(self, request):
        self.requests.append(request)
        spec = request["action_spec"]
        sid = spec.get("strategy_id")
        if sid in self.fail:
            return {"payload": None, "usage": {"completion_tokens": 10},
                    "error": "synthetic timeout", "latency_s": 0.02}
        entry = self.per_strategy.get(sid, {})
        has_incumbent = bool(
            (request["state"].get("task_progress") or {})
            .get("current_solution"))
        quality = entry.get("quality", 0.5)
        if has_incumbent and self.step2_quality is not None:
            quality = self.step2_quality.get(sid, quality)
        payload = {
            "outcome_status": "feasible", "feasible": True,
            "quality": quality,
            "failure_prob": entry.get("failure_prob", 0.1),
            "state_changes": {"current_solution": {
                "status": "feasible", "strategy_id": sid,
                "objective": entry.get("objective", 100.0)}},
        }
        if entry.get("cost"):
            payload["cost"] = dict(entry["cost"])
        return {"payload": payload,
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
                "error": None, "latency_s": 0.01}


TASK = {"task_id": "t1", "family": "routing", "description": "toy"}


def _spec(strategy_id, task_id="t1", episode_id="ep1", **params):
    return ActionSpec(action_type="execute_strategy", task_id=task_id,
                      episode_id=episode_id, strategy_id=strategy_id,
                      params=params)


class TestM2Closeout(HarnessTestCase):
    """F1-F3 regressions: identity, replay, strict binding."""

    def setUp(self):
        super().setUp()
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

    def test_bind_rejects_hypothetical_input(self):
        """F3/H2: a prediction conditioned on a hypothetical snapshot is a
        conditional outlook — binding it records a mismatch, never a real
        one-step feedback sample."""
        plan = self.h.plan_next(
            TASK, "ep1", candidates=[_spec("S01")],
            limits={"horizon": 2})
        hypo_preds = [p for p in self.h.predictions_query(task_id="t1")
                      if self.h.get_snapshot(
                          p.input_snapshot_id).hypothetical]
        self.assertTrue(hypo_preds)
        record = self.make_record(task_id="t1")
        self.h.bank.append(record)
        action = self.h.actions.begin_action(
            "execute_strategy", "t1", "ep1",
            pre_snapshot=self.h.snapshot(TASK, "ep1"),
            params={"strategy_id": "S01"})
        self.h.actions.end_action(
            action.action_id, status="completed",
            outcome={"execution_status": "optimal"}, cost=record.cost,
            linked_execution_id=record.execution_id, rollup="reference")
        bound = self.h.bind_outcome(hypo_preds[0].prediction_id,
                                    action.action_id)
        self.assertIn("input_snapshot", bound.binding_mismatch)
        # The comparison runs but is recorded as NOT compared — a
        # conditional outlook never becomes a real one-step feedback
        # sample.
        compared = self.h.compare_prediction(bound.prediction_id)
        self.assertFalse(compared.feedback["compared"])


class TestPlanning(HarnessTestCase):
    """P/H/A: candidates, horizon-1/2 evaluation, suggestion/choice split."""

    def test_horizon1_suggestion_and_cost_attribution(self):
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
        # Both steps share the same frozen root snapshot.
        self.assertTrue(all(
            s["prediction_id"] for p in plan["paths"] for s in p["steps"]))
        # Real planning spend charged ONCE to the decision action.
        decision = h.actions.get(plan["decision_action_id"])
        self.assertEqual(decision.cost.llm_tokens, 100.0)  # 2 x 50
        self.assertEqual(plan["planning_cost"]["cost"]["llm_tokens"], 100.0)
        # Planning spend is NOT part of any path utility.
        for path in plan["paths"]:
            self.assertNotIn("llm_tokens", str(path["c_path"]))

    def test_horizon2_terminal_evaluation_changes_root_choice(self):
        """The KEY two-step test: the first step's predicted post-state
        changes the second step's prediction, and the TERMINAL evaluation
        (not a sum of independent root scores) decides the root action.

        Rule (declared fixture): one-step qualities favour S02 (0.9 vs 0.4),
        but S01's successor enables a strong second step (0.95) while S02's
        does not (0.3). A terminal-quality comparison must pick S01 even
        though S01's OWN one-step quality is worse."""
        provider = ScriptableProvider(
            per_strategy={"S01": {"quality": 0.4, "failure_prob": 0.0},
                          "S02": {"quality": 0.9, "failure_prob": 0.0}},
            step2_quality={"S01": 0.95, "S02": 0.3})
        h = ORHarness(home=self.home, world_model=provider)
        self.addCleanup(h.close)
        plan = h.plan_next(TASK, "ep1",
                           candidates=[_spec("S01"), _spec("S02")],
                           limits={"horizon": 2})
        self.assertEqual(plan["suggested"]["strategy_id"], "S01")
        paths = {p["steps"][0]["action_spec"]["strategy_id"]: p
                 for p in plan["paths"]}
        self.assertEqual(len(paths["S01"]["steps"]), 2)
        self.assertEqual(len(paths["S02"]["steps"]), 2)
        self.assertIsNotNone(paths["S01"]["hypothetical_snapshot_id"])
        # The second step was predicted FROM the hypothetical successor:
        # the provider saw the imagined incumbent.
        second_requests = [r for r in provider.requests
                           if (r["state"].get("task_progress") or {})
                           .get("current_solution")]
        self.assertEqual(len(second_requests), 2)
        # Horizon=1 alone would have picked S02 — proving the terminal
        # evaluation, not a one-step reordering, drove the answer.
        plan1 = h.plan_next(TASK, "ep1",
                            candidates=[_spec("S01"), _spec("S02")],
                            limits={"horizon": 1})
        self.assertEqual(plan1["suggested"]["strategy_id"], "S02")

    def test_hypothetical_isolation(self):
        """Imagined progress never enters the real state chain, episode
        inheritance, or knowledge views."""
        provider = ScriptableProvider(
            per_strategy={"S01": {"quality": 0.8}})
        h = ORHarness(home=self.home, world_model=provider)
        self.addCleanup(h.close)
        h.plan_next(TASK, "ep1", candidates=[_spec("S01")],
                    limits={"horizon": 2})
        # Real episode progress has NO current_solution (only imagined).
        progress = h._episode_progress("t1", "ep1")
        self.assertNotIn("current_solution", progress)
        hypos = [s for s in h.snapshots(task_id="t1") if s.hypothetical]
        self.assertTrue(hypos)
        for snap in hypos:
            solution = snap.task_progress["current_solution"]
            self.assertEqual(solution["epistemic"], "inferred")
            self.assertTrue(solution["hypothetical"])

    def test_suggestion_is_not_a_selection(self):
        """Only an explicit choose_next writes X.selected_plan."""
        provider = ScriptableProvider(
            per_strategy={"S01": {"quality": 0.8}})
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

    def test_budget_bounds_and_unknown_not_claimed_ok(self):
        provider = ScriptableProvider(
            per_strategy={f"S0{i}": {"quality": 0.5} for i in range(1, 7)})
        h = ORHarness(home=self.home, world_model=provider)
        self.addCleanup(h.close)
        # max_model_calls bounds the evaluation.
        plan = h.plan_next(TASK, "ep1",
                           candidates=[_spec(f"S0{i}") for i in range(1, 5)],
                           limits={"horizon": 2, "max_model_calls": 3})
        self.assertEqual(plan["model_calls_made"], 3)
        self.assertEqual(plan["status"], "truncated")
        # Unknown budget is never reported as confirmed-ok.
        self.assertEqual(plan["budget_confirmation"], "unknown")
        # Exceeded REAL budget stops planning before any model call.
        h.declare_budget("t1", {"llm_tokens": 10}, episode_id="ep1")
        provider2_calls_before = len(provider.requests)
        plan2 = h.plan_next(TASK, "ep1", candidates=[_spec("S01")])
        self.assertEqual(plan2["status"], "fallback")
        self.assertEqual(len(provider.requests), provider2_calls_before)

    def test_no_candidates_and_unsupported_types(self):
        provider = ScriptableProvider()
        h = ORHarness(home=self.home, world_model=provider)
        self.addCleanup(h.close)
        # verify/finish_task etc. are honestly reported as not plannable.
        spec = ActionSpec(action_type="verify", task_id="t1",
                          episode_id="ep1")
        plan = h.plan_next(TASK, "ep1", candidates=[spec])
        self.assertEqual(plan["status"], "no_candidates")
        # planning=False restores exact M2 behaviour.
        h2 = ORHarness(home=self.home, world_model=provider,
                       planning=False)
        self.addCleanup(h2.close)
        plan2 = h2.plan_next(TASK, "ep1", candidates=[_spec("S01")])
        self.assertEqual(plan2["status"], "disabled")

    def test_failed_call_cost_preserved(self):
        """A failed model call still records its known usage (10 tokens)
        and the path is reported, not silently dropped."""
        provider = ScriptableProvider(
            per_strategy={"S01": {"quality": 0.8}}, fail={"S02"})
        h = ORHarness(home=self.home, world_model=provider)
        self.addCleanup(h.close)
        plan = h.plan_next(TASK, "ep1",
                           candidates=[_spec("S01"), _spec("S02")],
                           limits={"horizon": 1})
        paths = {p["steps"][0]["action_spec"]["strategy_id"]: p
                 for p in plan["paths"]}
        self.assertEqual(paths["S02"]["steps"][0]["status"],
                         "invalid_output")
        self.assertEqual(plan["suggested"]["strategy_id"], "S01")
        decision = h.actions.get(plan["decision_action_id"])
        # 50 (S01) + 10 (failed S02) = 60 tokens of REAL spend.
        self.assertEqual(decision.cost.llm_tokens, 60.0)

    def test_all_predictions_invalid_reports_honestly(self):
        """When EVERY candidate's prediction is unusable, status must NOT
        be "ok" — the caller would mistake "model output unusable" for a
        quiet "no recommendation". Reported as no_valid_predictions with
        the real statuses; the calls still happened and their cost is
        recorded. (Surfaced by the first real-endpoint M4 run: MiniMax
        returned non-JSON for one candidate and illegal cost values for
        the other.)"""
        provider = ScriptableProvider(fail={"S01", "S02"})
        h = ORHarness(home=self.home, world_model=provider)
        self.addCleanup(h.close)
        plan = h.plan_next(TASK, "ep1",
                           candidates=[_spec("S01"), _spec("S02")],
                           limits={"horizon": 1})
        self.assertEqual(plan["status"], "no_valid_predictions")
        self.assertIsNone(plan["suggested"])
        self.assertIn("no candidate carried a usable prediction",
                      plan["truncation_reason"])
        self.assertIn("invalid_output", plan["truncation_reason"])
        # The calls still happened and their real spend is recorded.
        self.assertEqual(plan["model_calls_made"], 2)
        decision = h.actions.get(plan["decision_action_id"])
        self.assertEqual(decision.cost.llm_tokens, 20.0)  # 2 x 10 (failed)

    def test_shadow_mode_withholds_suggestion(self):
        provider = ScriptableProvider(
            per_strategy={"S01": {"quality": 0.8}})
        h = ORHarness(home=self.home, world_model=provider,
                      plan_mode="shadow")
        self.addCleanup(h.close)
        plan = h.plan_next(TASK, "ep1", candidates=[_spec("S01")])
        self.assertIsNone(plan["suggested"])
        self.assertTrue(any(p["utility"] is not None
                            for p in plan["paths"]))


class TestExecutionReplan(HarnessTestCase):
    """Closed loop: plan -> choose -> real executor run -> record -> bind
    -> re-plan sees the new fact."""

    def test_replan_uses_new_real_state(self):
        import json
        from pathlib import Path
        provider = ScriptableProvider(per_strategy={
            "S01": {"quality": 0.5}, "S02": {"quality": 0.6}})
        h = ORHarness(home=self.home, world_model=provider)
        self.addCleanup(h.close)
        plan = h.plan_next(TASK, "ep1",
                           candidates=[_spec("S01"), _spec("S02")])
        h.choose_next(plan["decision_action_id"],
                      chosen=ActionSpec.from_dict(plan["suggested"]))
        # REAL executor run (synthetic solve script — labelled fixture,
        # not a real OR solve): writes a feasible result.
        workspace = Path(self.home) / "ws"
        workspace.mkdir()
        script = workspace / "solve.py"
        script.write_text(
            "import json\n"
            "json.dump({'status': 'feasible', 'objective_value': 120.0},\n"
            "          open('result.json', 'w'))\n")
        executed_strategy = plan["suggested"]["strategy_id"]
        record = h.execute(TASK, executed_strategy,
                           str(script), str(workspace), solver="highs",
                           episode_id="ep1")
        self.assertTrue(record.quality["feasible"])
        h.record(record)
        # Bind the EXECUTED strategy's first-step prediction to the real
        # action (only the truly corresponding prediction is compared).
        executed_path = next(
            p for p in plan["paths"]
            if p["steps"][0]["action_spec"]["strategy_id"]
            == executed_strategy)
        first_pred = h.get_prediction(
            executed_path["steps"][0]["prediction_id"])
        bound = h.bind_outcome(first_pred.prediction_id, record.action_id)
        self.assertIsNone(bound.binding_mismatch)
        compared = h.compare_prediction(bound.prediction_id)
        self.assertIsNotNone(compared.feedback)
        # Re-plan: the new root snapshot carries the real current_solution.
        plan2 = h.plan_next(TASK, "ep1",
                            candidates=[_spec("S01"), _spec("S02")])
        root = h.get_snapshot(plan2["root_snapshot_id"])
        self.assertIn("current_solution", root.task_progress)
        self.assertEqual(root.task_progress["current_solution"]
                         ["epistemic"], "fact")


class TestReviewFixes(HarnessTestCase):
    """Regression tests for the four reviewed defects (2026-09-15)."""

    def test_cost_weights_inherited_from_harness(self):
        """Bug 1: planning inherits the harness's alpha/beta/gamma and
        cost weights; an expensive candidate loses to a cheap one when
        quality and risk tie."""
        provider = ScriptableProvider(per_strategy={
            "S01": {"quality": 0.8, "cost": {"solver_runtime_s": 100.0}},
            "S02": {"quality": 0.8, "cost": {"solver_runtime_s": 1.0}},
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
        paths = {p["steps"][0]["action_spec"]["strategy_id"]: p
                 for p in plan["paths"]}
        self.assertGreater(paths["S01"]["c_path"], paths["S02"]["c_path"])
        # Explicit override wins over inheritance.
        plan2 = h.plan_next(TASK, "ep1",
                            candidates=[_spec("S01"), _spec("S02")],
                            limits={"horizon": 1, "cost_weights": {}})
        self.assertEqual(plan2["limits"]["cost_weights"], {})
        self.assertEqual(plan2["suggested"]["strategy_id"], "S01")

    def test_unknown_risk_and_cost_never_auto_win(self):
        """Bug 2: unknown failure risk is a deficit (charged full gamma),
        and an unpredicted cost dimension is not a predicted zero."""
        provider = ScriptableProvider(per_strategy={
            "S01": {"quality": 0.8, "failure_prob": 0.1},
            "S02": {"quality": 0.8, "failure_prob": None},  # risk unknown
        })
        h = ORHarness(home=self.home, world_model=provider)
        self.addCleanup(h.close)
        plan = h.plan_next(TASK, "ep1",
                           candidates=[_spec("S01"), _spec("S02")],
                           limits={"horizon": 1})
        self.assertEqual(plan["suggested"]["strategy_id"], "S01")
        paths = {p["steps"][0]["action_spec"]["strategy_id"]: p
                 for p in plan["paths"]}
        self.assertIn("risk", paths["S02"]["incomparable"])
        self.assertIsNone(paths["S02"]["r_terminal"])
        # Unknown cost: S01 predicts only tokens, S02 predicts tokens AND
        # solver time. The shared basis is tokens only — but a candidate
        # that predicted NOTHING comparable must not score as cheap.
        provider2 = ScriptableProvider(per_strategy={
            "S03": {"quality": 0.8, "cost": {"llm_tokens": 10.0}},
            "S04": {"quality": 0.8},  # no cost predicted at all
        })
        h2 = ORHarness(home=self.home, world_model=provider2,
                       cost_weights={"llm_tokens": 1.0})
        self.addCleanup(h2.close)
        plan2 = h2.plan_next(TASK, "ep1",
                             candidates=[_spec("S03"), _spec("S04")],
                             limits={"horizon": 1})
        self.assertEqual(plan2["suggested"]["strategy_id"], "S03")
        paths2 = {p["steps"][0]["action_spec"]["strategy_id"]: p
                  for p in plan2["paths"]}
        self.assertIn("cost", paths2["S04"]["incomparable"])

    def test_budget_rechecked_and_exceeded_withholds_suggestion(self):
        """Bug 3: planning re-checks the REAL budget per call and after
        the spend lands; an exceeded budget never returns ok."""
        provider = ScriptableProvider(per_strategy={
            f"S0{i}": {"quality": 0.5} for i in range(1, 5)})
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
        # The real spend still landed on the books (never lost).
        view = h.budget_view("t1", "ep1")
        self.assertEqual(view["status"], "exceeded")
        self.assertGreater(plan["model_calls_made"], 0)

    def test_time_budget_truncates(self):
        """Bug 3 (time): a near-zero time budget truncates evaluation
        instead of reporting ok after slow calls."""
        class SlowProvider(ScriptableProvider):
            def predict(self, request):
                import time as _time
                _time.sleep(0.05)
                return super().predict(request)
        provider = SlowProvider(per_strategy={"S01": {"quality": 0.8},
                                              "S02": {"quality": 0.7}})
        h = ORHarness(home=self.home, world_model=provider)
        self.addCleanup(h.close)
        plan = h.plan_next(TASK, "ep1",
                           candidates=[_spec("S01"), _spec("S02")],
                           limits={"horizon": 1, "time_budget_s": 0.01})
        self.assertEqual(plan["status"], "truncated")
        self.assertIn("time", plan["truncation_reason"])

    def test_candidate_identity_conflict_rejected(self):
        """Bug 4: a candidate naming another task/episode is rejected,
        never planned under this root state."""
        provider = ScriptableProvider(
            per_strategy={"S01": {"quality": 0.8}})
        h = ORHarness(home=self.home, world_model=provider)
        self.addCleanup(h.close)
        foreign = _spec("S01", task_id="other", episode_id="other_ep")
        plan = h.plan_next(TASK, "ep1", candidates=[foreign, _spec("S01")],
                           limits={"horizon": 1})
        self.assertIn("identity conflict", plan["truncation_reason"])
        # Only the matching candidate was evaluated.
        self.assertEqual(len(plan["paths"]), 1)
        self.assertEqual(
            plan["paths"][0]["steps"][0]["action_spec"]["task_id"], "t1")
        # All-conflicting => no candidates at all.
        plan2 = h.plan_next(TASK, "ep1", candidates=[foreign])
        self.assertEqual(plan2["status"], "no_candidates")


if __name__ == "__main__":
    unittest.main()
