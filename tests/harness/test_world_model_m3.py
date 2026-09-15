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

    def test_rejection_preserves_earlier_selection(self):
        """R3 regression: choosing S01, then receiving an S04 suggestion
        and REJECTING it must leave X.selected_plan = S01. The rejection
        is recorded as its own queryable event, never as a selection
        overwrite."""
        provider = ScriptableProvider(per_strategy={
            "S01": {"quality": 0.9}, "S04": {"quality": 0.85}})
        h = ORHarness(home=self.home, world_model=provider)
        self.addCleanup(h.close)
        # 1. Plan + explicitly choose S01.
        plan1 = h.plan_next(TASK, "ep1", candidates=[_spec("S01")])
        h.choose_next(plan1["decision_action_id"], chosen=_spec("S01"))
        snap1 = h.snapshot(TASK, "ep1")
        selected = (snap1.task_progress.get("selected_plan") or {}) \
            .get("value", {}).get("selected", {})
        self.assertEqual(selected.get("strategy_id"), "S01")
        # 2. A new plan suggests S04; the agent REJECTS it.
        plan2 = h.plan_next(TASK, "ep1", candidates=[_spec("S04")])
        rejected = h.choose_next(plan2["decision_action_id"],
                                 rejected=True,
                                 deviation_note="keeping S01")
        self.assertTrue(rejected["rejected"])
        # 3. The current selection is STILL S01.
        snap2 = h.snapshot(TASK, "ep1")
        selected_after = (snap2.task_progress.get("selected_plan") or {}) \
            .get("value", {}).get("selected", {})
        self.assertEqual(selected_after.get("strategy_id"), "S01")
        # 4. The rejection event is recorded and queryable.
        rejection = (snap2.task_progress.get("last_rejected_suggestion")
                     or {})
        self.assertEqual(rejection.get("provenance"), "agent_reported")
        self.assertTrue((rejection.get("value") or {}).get("rejected"))

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

    def test_malformed_fields_become_invalid_output_not_crash(self):
        """R2 regression: a structurally valid JSON payload whose
        ``unsupported_fields`` is a STRING (not an object) used to escape
        validation and crash the prediction-object construction with
        ValueError — aborting plan_next mid-loop, losing the second call
        from the log, and leaving the decision action running forever.
        Now: the malformed payload is validated BEFORE construction,
        becomes an explicit invalid_output with its known cost kept, the
        decision ends failed (not running, not completed)."""

        class MalformedProvider(WorldModelProvider):
            """Returns valid JSON with a malformed unsupported_fields on
            the SECOND call (the first is clean)."""
            name = "malformed"

            def __init__(self):
                self.calls = 0

            def predict(self, request):
                self.calls += 1
                if self.calls == 2:
                    return {"payload": {
                                "outcome_status": "feasible",
                                "quality": 0.7,
                                "unsupported_fields": "retries"},
                            "usage": {"completion_tokens": 30},
                            "error": None, "latency_s": 0.01}
                return {"payload": {"outcome_status": "feasible",
                                    "quality": 0.9},
                        "usage": {"completion_tokens": 50},
                        "error": None, "latency_s": 0.01}

        provider = MalformedProvider()
        h = ORHarness(home=self.home, world_model=provider)
        self.addCleanup(h.close)
        plan = h.plan_next(TASK, "ep1",
                           candidates=[_spec("S01"), _spec("S02")],
                           limits={"horizon": 1})
        # Both calls are traceable in the prediction log.
        self.assertEqual(provider.calls, 2)
        self.assertEqual(plan["model_calls_made"], 2)
        # The malformed second call is an explicit invalid_output, not a
        # crash — its known cost is preserved.
        predictions = h.predictions_query(task_id="t1")
        self.assertEqual(len(predictions), 2)
        statuses = sorted(p.status for p in predictions)
        self.assertEqual(statuses, ["invalid_output", "valid"])
        bad = next(p for p in predictions if p.status == "invalid_output")
        self.assertIn("unsupported_fields", bad.error)
        self.assertEqual(bad.call_cost.llm_tokens, 30.0)
        # The decision action did NOT stay running and was NOT silently
        # completed: S01's valid prediction still yields a suggestion
        # (the invalid one is incomparable), and the decision completes.
        decision = h.actions.get(plan["decision_action_id"])
        self.assertNotEqual(decision.status, "running")
        self.assertEqual(decision.status, "completed")
        self.assertEqual(decision.cost.llm_tokens, 80.0)  # 50 + 30

    def test_planning_exception_ends_decision_failed(self):
        """R2 (closing path): when the prediction machinery itself raises
        mid-loop (not a payload problem — an internal error), plan_next
        must not leave the decision running: the completed part is kept,
        known cost is charged, and the decision ends failed."""

        class CrashProvider(WorldModelProvider):
            """Second call raises an unexpected internal error."""
            name = "crash"

            def __init__(self):
                self.calls = 0

            def predict(self, request):
                self.calls += 1
                if self.calls == 2:
                    raise RuntimeError("synthetic internal crash")
                return {"payload": {"outcome_status": "feasible",
                                    "quality": 0.9},
                        "usage": {"completion_tokens": 50},
                        "error": None, "latency_s": 0.01}

        provider = CrashProvider()
        h = ORHarness(home=self.home, world_model=provider)
        self.addCleanup(h.close)
        # A provider-raised exception is caught by the service layer as
        # provider_error (a prediction record), so planning continues —
        # but if the service layer itself failed, the planning-level
        # safety net ends the decision failed. Both layers keep the cost.
        plan = h.plan_next(TASK, "ep1",
                           candidates=[_spec("S01"), _spec("S02")],
                           limits={"horizon": 1})
        decision = h.actions.get(plan["decision_action_id"])
        self.assertNotEqual(decision.status, "running")
        predictions = h.predictions_query(task_id="t1")
        self.assertEqual(len(predictions), 2)
        statuses = sorted(p.status for p in predictions)
        self.assertEqual(statuses, ["provider_error", "valid"])

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

    def test_unknown_cost_never_erases_measured_differences(self):
        """R1 regression: the A/B/C case — A and B have comparable
        measured costs (A expensive, B cheap), C predicts NO cost at all.
        The old intersection basis collapsed to empty, zeroing every
        candidate's cost term and letting the input order decide. Now:
        the union basis keeps A/B's measured difference, and C is
        charged the peak normalized share (never free).

        Quality and risk tie across all three; the ONLY differentiator
        is cost. B must win regardless of C's presence or ordering."""
        provider = ScriptableProvider(per_strategy={
            "S01": {"quality": 0.8, "failure_prob": 0.1,
                    "cost": {"solver_runtime_s": 100.0}},   # A: expensive
            "S02": {"quality": 0.8, "failure_prob": 0.1,
                    "cost": {"solver_runtime_s": 1.0}},     # B: cheap
            "S03": {"quality": 0.8, "failure_prob": 0.1},   # C: cost unknown
        })
        h = ORHarness(home=self.home, world_model=provider,
                      beta=1.0, cost_weights={"solver_runtime_s": 1.0})
        self.addCleanup(h.close)
        # Baseline: without C, B wins on cost.
        plan_ab = h.plan_next(TASK, "ep1",
                              candidates=[_spec("S01"), _spec("S02")],
                              limits={"horizon": 1})
        self.assertEqual(plan_ab["suggested"]["strategy_id"], "S02")
        # With C added (in BOTH orderings): B still wins; A/B's cost
        # difference is preserved; C never gets a free cost pass.
        for ordering in (("S01", "S02", "S03"), ("S03", "S02", "S01")):
            plan = h.plan_next(TASK, "ep1",
                               candidates=[_spec(s) for s in ordering],
                               limits={"horizon": 1})
            self.assertEqual(plan["suggested"]["strategy_id"], "S02",
                             f"ordering {ordering}: C must not win by "
                             "staying silent on cost")
            paths = {p["steps"][0]["action_spec"]["strategy_id"]: p
                     for p in plan["paths"]}
            # A's measured cost difference vs B is preserved (not zeroed).
            self.assertGreater(paths["S01"]["c_path"],
                               paths["S02"]["c_path"])
            # C is charged the peak share (== A's normalized cost, the
            # peak) — strictly more than B, never zero.
            self.assertGreater(paths["S03"]["c_path"],
                               paths["S02"]["c_path"])
            self.assertAlmostEqual(paths["S03"]["c_path"],
                                   paths["S01"]["c_path"], places=3)
            self.assertIn("cost", paths["S03"]["incomparable"])
        # Explicit predicted zero stays a zero (measured), distinct from
        # unknown: a candidate predicting solver_runtime_s=0 is cheaper
        # than B and wins.
        provider_zero = ScriptableProvider(per_strategy={
            "S01": {"quality": 0.8, "failure_prob": 0.1,
                    "cost": {"solver_runtime_s": 100.0}},
            "S02": {"quality": 0.8, "failure_prob": 0.1,
                    "cost": {"solver_runtime_s": 0.0}},  # explicit zero
        })
        h_zero = ORHarness(home=self.home, world_model=provider_zero,
                           beta=1.0, cost_weights={"solver_runtime_s": 1.0})
        self.addCleanup(h_zero.close)
        plan_zero = h_zero.plan_next(TASK, "ep1",
                                     candidates=[_spec("S01"), _spec("S02")],
                                     limits={"horizon": 1})
        self.assertEqual(plan_zero["suggested"]["strategy_id"], "S02")
        zero_path = next(p for p in plan_zero["paths"]
                         if p["steps"][0]["action_spec"]["strategy_id"]
                         == "S02")
        self.assertEqual(zero_path["c_path"], 0.0)
        self.assertNotIn("cost", zero_path["incomparable"])

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

    def test_single_candidate_timeout_reports_truncated(self):
        """R5 regression: ONE candidate whose single call exceeds the
        time budget must NOT report ok — the post-call deadline check
        catches the over-budget return and reports truncation with the
        cost preserved."""
        class SlowProvider(ScriptableProvider):
            def predict(self, request):
                import time as _time
                _time.sleep(0.06)
                return super().predict(request)
        provider = SlowProvider(per_strategy={"S01": {"quality": 0.8}})
        h = ORHarness(home=self.home, world_model=provider)
        self.addCleanup(h.close)
        plan = h.plan_next(TASK, "ep1", candidates=[_spec("S01")],
                           limits={"horizon": 1, "time_budget_s": 0.01})
        # The call happened (cost is real) but the result is truncated,
        # never a quiet ok.
        self.assertEqual(plan["status"], "truncated")
        self.assertIn("time budget", plan["truncation_reason"])
        self.assertEqual(plan["model_calls_made"], 1)
        decision = h.actions.get(plan["decision_action_id"])
        self.assertEqual(decision.cost.llm_tokens, 50.0)

    def test_last_candidate_timeout_reports_truncated(self):
        """R5 regression: the LAST candidate's call exceeding the budget
        is caught by the post-call check (no further loop iteration
        would catch it)."""
        class SlowProvider(ScriptableProvider):
            def predict(self, request):
                import time as _time
                _time.sleep(0.06)
                return super().predict(request)
        provider = SlowProvider(per_strategy={"S01": {"quality": 0.8},
                                              "S02": {"quality": 0.7}})
        h = ORHarness(home=self.home, world_model=provider)
        self.addCleanup(h.close)
        # Budget fits exactly one call; the second call's return is
        # already over budget.
        plan = h.plan_next(TASK, "ep1",
                           candidates=[_spec("S01"), _spec("S02")],
                           limits={"horizon": 1, "time_budget_s": 0.08})
        self.assertEqual(plan["status"], "truncated")
        self.assertEqual(plan["model_calls_made"], 2)

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

    def test_second_step_identity_conflict_rejected(self):
        """R4 regression: a second-step candidate explicitly naming
        ANOTHER task must not produce a model call or a misattributed
        prediction — the conflict is rejected before the provider runs,
        and the caller's spec object is never mutated."""
        provider = ScriptableProvider(per_strategy={
            "S01": {"quality": 0.8}})
        h = ORHarness(home=self.home, world_model=provider)
        self.addCleanup(h.close)
        foreign_second = ActionSpec(action_type="execute_strategy",
                                    task_id="other_task",
                                    episode_id="ep1",
                                    strategy_id="S01")
        plan = h.plan_next(TASK, "ep1", candidates=[_spec("S01")],
                           second_step=[foreign_second],
                           limits={"horizon": 2})
        # The root step ran (1 call), but the conflicting second step
        # produced NO model call and NO prediction record.
        self.assertEqual(plan["model_calls_made"], 1)
        predictions = h.predictions_query(task_id="t1")
        self.assertEqual(len(predictions), 1)
        foreign_preds = h.predictions_query(task_id="other_task")
        self.assertEqual(len(foreign_preds), 0)
        # The conflict is reported, not silently overwritten.
        self.assertIn("second-step identity conflict",
                      plan["truncation_reason"])
        # The caller's spec object was not mutated.
        self.assertEqual(foreign_second.task_id, "other_task")

    def test_planning_basis_persisted_and_recoverable_after_reopen(self):
        """R6 regression: closing and re-opening the harness must fully
        recover the planning decision — root snapshot, candidates, paths
        with decomposition, prediction refs, comparison basis/norms,
        budget confirmation, and suggestion. Later knowledge changes
        must NOT rewrite the historical judgment."""
        provider = ScriptableProvider(per_strategy={
            "S01": {"quality": 0.9, "cost": {"solver_runtime_s": 10.0}},
            "S02": {"quality": 0.6, "cost": {"solver_runtime_s": 1.0}},
        })
        h = ORHarness(home=self.home, world_model=provider,
                      beta=1.0, cost_weights={"solver_runtime_s": 1.0})
        plan = h.plan_next(TASK, "ep1",
                           candidates=[_spec("S01"), _spec("S02")],
                           limits={"horizon": 1})
        decision_id = plan["decision_action_id"]
        h.close()

        # Re-open with a FRESH harness instance (different config / alpha).
        h2 = ORHarness(home=self.home, alpha=99.0)
        self.addCleanup(h2.close)
        decision = h2.plan_decision(decision_id)
        self.assertIsNotNone(decision)
        outcome = decision.get("outcome") or {}
        self.assertEqual(outcome.get("plan_id"), plan["plan_id"])
        self.assertEqual(outcome.get("root_snapshot_id"),
                         plan["root_snapshot_id"])
        self.assertEqual(outcome.get("status"), "ok")
        self.assertEqual(len(outcome.get("paths") or []), 2)
        # Prediction references are recoverable.
        path_s1 = next(p for p in outcome["paths"]
                       if p["steps"][0]["action_spec"]["strategy_id"] == "S01")
        pred_id = path_s1["steps"][0]["prediction_id"]
        self.assertTrue(pred_id.startswith("wp_"))
        stored_pred = h2.get_prediction(pred_id)
        self.assertIsNotNone(stored_pred)
        # Utility decomposition is preserved as computed at the time.
        self.assertEqual(path_s1["q_terminal"], 0.9)
        self.assertEqual(path_s1["c_path"], 1.0)
        # Comparison basis and norms are preserved.
        self.assertEqual(outcome.get("cost_basis"), ["solver_runtime_s"])
        self.assertEqual(outcome.get("cost_norms"),
                         {"solver_runtime_s": 10.0, "_max_c_path": 1.0})
        # Suggestion and its basis string.
        self.assertEqual((outcome.get("suggested") or {}).get("strategy_id"),
                         "S02")
        self.assertTrue(outcome.get("suggestion_basis"))


if __name__ == "__main__":
    unittest.main()
