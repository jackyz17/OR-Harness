"""Review regressions: cold-start runner inputs and contradictory host meters."""

from decision_fixtures import execute_scored, planned_prediction

import contextlib
import copy
import io
import json
import os
import time
from pathlib import Path
from unittest.mock import patch

from or_harness.api import ORHarness
from or_harness.strategy.embedding_index import LocalHashEmbeddingBackend
from tests.harness.c_fixtures import CCase, REUSE

RUNNER = {"phase": "initial_modelling", "source": "test runner configuration"}


class TestFirstProposal(CCase):
    def history_without_seed_actions(self):
        self.h = ORHarness(home=self.home, world_model=self.provider,
            embedding=LocalHashEmbeddingBackend())
        self.addCleanup(self.h.close)
        p = planned_prediction(self.h,self.task, {
            "action_type": "execute_strategy", "strategy_id": "S04",
            "solver": "highs", "method": REUSE}, "history")
        self.assertEqual(
            self.h.actions.query(task_id=self.task["task_id"],
                                 action_type="model"), [])
        r = self.run_g(p)
        self.check_g(r)
        self.finish_c(p)

    def call_cli(self, new_task, *, explicit=None):
        from or_harness import cli
        args = ["--home", self.home, "--world-model",
                "http://fixture.invalid::predicting-model",
                "--wm-timeout", "1", "--wm-max-tokens", "256"]
        if explicit is not None:
            args += ["--execution-context", json.dumps(explicit)]
        args += ["plan-next", "--task", json.dumps(new_task), "--episode",
                 "first", "--candidates",
                 json.dumps([{"action_type": "execute_strategy",
                              "strategy_id": "S04", "method": REUSE}])]
        def harness(**kwargs):
            return ORHarness(embedding=LocalHashEmbeddingBackend(), **kwargs)
        output = io.StringIO()
        with patch.object(cli, "ORHarness", side_effect=harness), patch(
                "or_harness.world_model.provider.HttpChatProvider",
                return_value=self.provider), \
                contextlib.redirect_stdout(output):
            status = cli.main(args)
        self.assertEqual(status, 0, output.getvalue())
        return self.provider.requests[-1]["prediction_context"]

    def test_first_cli_request_receives_cost_history_without_agent_identity(self):
        self.history_without_seed_actions()
        task = dict(self.task, task_id="brand-new-task")
        self.assertFalse(self.h.actions.query(task_id=task["task_id"]))
        view = self.call_cli(task)
        e = view["cost_evidence"]
        self.assertEqual(e["version"], "wm-c-history/1")
        # The unified retrieval surfaced the closed execution as a LOCAL
        # case: the four remaining-task actuals are there and the window is
        # the comparable one.
        self.assertGreaterEqual(len(e["cases"]), 1)
        case = e["cases"][0]
        self.assertEqual(case["scope"], "remaining_to_task_end")
        self.assertEqual(case["grade"], "strict")
        self.assertIn("tool_calls", case["cost"])
        self.assertNotIn("phase", e["current_conditions"])
        self.assertNotIn("phase_basis", e["current_conditions"])
        self.assertFalse(e["current_conditions"]["phase_used"])
        self.assertNotIn("execution_host", e["current_conditions"])
        self.assertNotIn("execution_host", view["solving_context"])
        actions = self.h.actions.query(task_id=task["task_id"])
        self.assertEqual(len(actions), 2)
        choices = [a for a in actions if (a.outcome or {}).get("kind") == "plan_next_choice"]
        actions = [a for a in actions if (a.params or {}).get("kind") == "plan_next"]
        self.assertEqual(len(actions), 1)
        self.assertEqual(len(choices), 1)
        self.assertEqual(choices[0].parent_action_id, actions[0].action_id)
        self.assertEqual(actions[0].action_type, "select_strategy")
        self.assertEqual(actions[0].params["kind"], "plan_next")
        self.assertEqual(actions[0].status, "completed")
        self.assertIsNone(actions[0].linked_execution_id)
        self.assertIsNotNone(actions[0].outcome["selected_prediction_id"])

    def test_runner_phase_input_no_longer_changes_cost_evidence(self):
        self.history_without_seed_actions()
        first = self.call_cli(dict(self.task, task_id="explicit"),
                              explicit={"phase": "initial_modelling"})
        second = self.call_cli(dict(self.task, task_id="explicit-2"),
                               explicit={"phase": "existing_model"})
        third = self.call_cli(dict(self.task, task_id="unknown"))
        for view in (first, second, third):
            current = view["cost_evidence"]["current_conditions"]
            self.assertNotIn("phase", current)
            self.assertNotIn("phase_basis", current)
            self.assertFalse(current["phase_used"])
            # The historical cases cannot depend on a phase classification
            # either: the same comparable windows are selected.
            self.assertEqual(
                [case["execution_id"] for case in view["cost_evidence"]["cases"]],
                [case["execution_id"] for case in first["cost_evidence"]["cases"]])

    def test_model_artifact_is_optional_context_not_a_phase(self):
        first = self.predict_g(episode="first")
        first_conditions = first.trace.model_info["cost_conditions"]
        self.assertNotIn("phase", first_conditions)
        self.assertFalse(first_conditions["reusable_artifacts"])
        self.h.report_action("model", self.task, "first",
                             outcome={"artifact": "model.lp",
                                      "status": "available"})
        later = self.predict_g(episode="first")
        later_conditions = later.trace.model_info["cost_conditions"]
        self.assertNotIn("phase", later_conditions)
        self.assertNotIn("phase_basis", later_conditions)
        # The artifact is kept as ORDINARY context for cost prediction; it
        # is never converted into a model phase.
        self.assertTrue(later_conditions["reusable_artifacts"])

    def test_cost_history_is_not_split_by_usage_report_model(self):
        self.history_without_seed_actions()
        p = planned_prediction(self.h, self.task, {
            "action_type": "execute_strategy", "strategy_id": "S04",
            "solver": "highs", "method": REUSE}, "second-history")
        r = self.run_g(p)
        self.check_g(r)
        self.finish_c(p, model="another-execution-model")
        view = self.call_cli(dict(self.task, task_id="new-task"))
        evidence = view["cost_evidence"]
        # Both closed episodes are recalled as local cases: the usage-report
        # model identity is not a cost condition.
        self.assertGreaterEqual(len(evidence["cases"]), 2)
        for case in evidence["cases"]:
            self.assertNotIn("execution_host", case)
            self.assertEqual(case["scope"], "remaining_to_task_end")
        tool = evidence["global_cost"]["dimensions"].get("tool_calls") or {}
        self.assertGreaterEqual(tool.get("n", 0), 2)
        self.assertIsNotNone(tool.get("p50"))


class TestHostMeterIntegrity(CCase):
    def settle(self, *, tools=8, tokens=200, details=None, complete=None, counters=None,
               own_bound=None, extra_execution=False, second_report=False):
        p = self.predict_c(cost={"tool_calls": 8})
        r = self.run_g(p)
        self.check_g(r)
        if extra_execution:
            work = Path(self.home) / p.prediction_id
            extra = execute_scored(self.h,self.task, "S04", str(work / "solve.py"), str(work), solver="highs", episode_id="ep1")
            self.h.record(extra)
        stop = time.time()
        report = {"schema": "or-host-usage/1", "host": "test-host", "model": "executor-model/v1",
            "scope": "remaining_to_task_end", "measured": ["tool_calls", "llm_tokens"],
            "window": {"started_at": self.h.actions.get(p.trace.model_info["decision_action_id"]).started_at, "ended_at": stop, "complete": True}}
        if tools is not None:
            report["tool_calls"] = tools
        if tokens is not None:
            report["tokens"] = {"prompt_tokens": tokens, "completion_tokens": 0}
        if details is not None:
            report["call_details"] = [{"started_at": self.h.actions.get(p.trace.model_info["decision_action_id"]).started_at, "ended_at": stop, **d} for d in details]
        if complete is not None:
            report["call_details_complete"] = complete
        if counters is not None:
            report["counters"] = counters
        if own_bound is not None:
            report["tool_calls_lower_bound"] = own_bound
        self.h.report_action("finish_task", self.task, "ep1", started_at=stop, ended_at=stop, host_usage=report)
        if second_report:
            good = copy.deepcopy(report)
            good.pop("call_details", None)
            self.h.report_action("verify", self.task, "ep1", started_at=stop, ended_at=stop, host_usage=good)
        result = self.close_g()
        return result["evaluations"][0], self.h.calibration_summary()["conditional_cost"]

    def test_aggregate_8_tools_and_200_tokens_cannot_be_replaced_by_one_tool_and_zero_tokens(self):
        evaluation, stats = self.settle(details=[{"id": "t1", "kind": "tool"}])
        fact = evaluation["cost"]["observed_fact"]
        self.assertNotIn("tool_calls", fact["values"])
        self.assertEqual(fact["values"]["llm_tokens"], 200)
        self.assertTrue(any("aggregate disagrees" in s for s in fact["boundary_exclusions"]))
        self.assertEqual(stats["coverage"]["tool_calls"]["n_actual"], 0)
        self.assertEqual(stats["coverage"]["tool_calls"]["n_error"], 0)
        self.assertEqual(evaluation["benefit"]["observed"], 1)
        self.predict_c("next")
        e = self.provider.requests[-1]["prediction_context"]["cost_evidence"]
        self.assertTrue(e["cases"])
        case = e["cases"][0]
        # The contradictory aggregate is NOT a complete tool_calls
        # observation: it must not appear as a usable cost number, and the
        # missing dimension is named instead of zero-filled.
        self.assertNotIn("tool_calls", case["cost"])
        self.assertIn("tool_calls", case["missing_dimensions"])

    def test_token_conflict_keeps_legitimate_tool_total(self):
        e, _ = self.settle(details=[{"id": "m1", "kind": "model", "prompt_tokens": 60, "completion_tokens": 40}])
        self.assertNotIn("llm_tokens", e["cost"]["observed_fact"]["values"])
        self.assertEqual(e["cost"]["per_dim"]["tool_calls"]["actual"], 8)

    def test_tool_only_details_do_not_prove_zero_tokens(self):
        e, _ = self.settle(tools=None, tokens=None, details=[{"id": "t1", "kind": "tool"}])
        self.assertEqual(e["cost"]["per_dim"]["tool_calls"]["actual"], 1)
        self.assertNotIn("llm_tokens", e["cost"]["observed_fact"]["values"])

    def test_partial_details_are_bounds_not_complete_totals(self):
        e, _ = self.settle(details=[{"id": "t1", "kind": "tool"}], complete=[])
        self.assertEqual(e["cost"]["per_dim"]["tool_calls"]["actual"], 8)
        self.assertEqual(
            e["cost"]["observed_fact"]["values"]["llm_tokens"], 200)

    def test_explicit_empty_complete_token_ledger_conflicts_with_positive_total(self):
        e, _ = self.settle(tools=1, details=[{"id": "t1", "kind": "tool"}],
                           complete=["tool_calls", "llm_tokens"])
        self.assertNotIn("llm_tokens", e["cost"]["observed_fact"]["values"])
        self.assertEqual(e["cost"]["per_dim"]["tool_calls"]["actual"], 1)

    def test_executor_lower_bound_rejects_zero_but_preserves_tokens(self):
        e, stats = self.settle(tools=0)
        self.assertNotIn("tool_calls", e["cost"]["observed_fact"]["values"])
        self.assertEqual(
            e["cost"]["observed_fact"]["values"]["llm_tokens"], 200)
        self.assertEqual(stats["coverage"]["tool_calls"]["n_actual"], 0)
        self.assertTrue(any("lower bound" in s for s in e["cost"]["observed_fact"]["boundary_exclusions"]))

    def test_two_execution_floors_are_added_once(self):
        e, _ = self.settle(tools=1, extra_execution=True)
        self.assertNotIn("tool_calls", e["cost"]["observed_fact"]["values"])
        self.assertEqual(e["cost"]["observed_fact"]["work_facts"]["execution_attempts"], 2)

    def test_same_window_counter_mismatch_is_not_silently_preferred(self):
        e, _ = self.settle(counters={"start": {"tool_calls": 0, "prompt_tokens": 0, "completion_tokens": 0},
            "end": {"tool_calls": 1, "prompt_tokens": 200, "completion_tokens": 0}})
        self.assertNotIn("tool_calls", e["cost"]["observed_fact"]["values"])
        self.assertEqual(
            e["cost"]["observed_fact"]["values"]["llm_tokens"], 200)

    def test_report_own_floor_is_a_per_dimension_unknown(self):
        e, _ = self.settle(tools=0, own_bound=1)
        self.assertNotIn("tool_calls", e["cost"]["observed_fact"]["values"])
        self.assertEqual(
            e["cost"]["observed_fact"]["values"]["llm_tokens"], 200)

    def test_another_report_does_not_hide_a_conflict(self):
        e, _ = self.settle(details=[{"id": "t1", "kind": "tool"}], second_report=True)
        self.assertNotIn("tool_calls", e["cost"]["observed_fact"]["values"])
        self.assertEqual(
            e["cost"]["observed_fact"]["values"]["llm_tokens"], 200)

    def test_matching_summary_and_deduplicated_details_remain_actuals(self):
        tool = {"id": "t1", "kind": "tool"}
        model = {"id": "m1", "kind": "model", "prompt_tokens": 120, "completion_tokens": 80}
        e, stats = self.settle(tools=1, details=[tool, model, dict(tool), dict(model)],
                              complete=["tool_calls", "llm_tokens"])
        self.assertEqual(e["cost"]["per_dim"]["tool_calls"]["actual"], 1)
        self.assertEqual(
            e["cost"]["observed_fact"]["values"]["llm_tokens"], 200)
        self.assertEqual(stats["coverage"]["tool_calls"]["n_actual"], 1)

    def test_partial_ledger_lower_bound_can_refute_a_total(self):
        e, _ = self.settle(tools=1, details=[{"id": "t1", "kind": "tool"},
            {"id": "t2", "kind": "tool"}], complete=[])
        self.assertNotIn("tool_calls", e["cost"]["observed_fact"]["values"])
        self.assertEqual(
            e["cost"]["observed_fact"]["values"]["llm_tokens"], 200)

    def test_explicit_complete_zero_tokens_is_a_valid_raw_error(self):
        e, stats = self.settle(tools=1, tokens=0, details=[{"id": "t1", "kind": "tool"}],
            complete=["tool_calls", "llm_tokens"])
        # The old raw-token observation is still recorded, but it is not an
        # active C prediction dimension and never gets a new C error.
        self.assertNotIn("llm_tokens", e["cost"]["per_dim"])
        self.assertEqual(
            e["cost"]["observed_fact"]["values"]["llm_tokens"], 0)
        self.assertEqual(stats["coverage"]["llm_tokens"]["n_actual"], 1)
        self.assertEqual(stats["coverage"]["llm_tokens"]["n_log_ratio"], 0)

    def test_late_check_does_not_move_the_executors_proven_call(self):
        p = self.predict_c(cost={"tool_calls": 0})
        r = self.run_g(p)
        stop = time.time()
        self.h.report_action("finish_task", self.task, "ep1", started_at=stop, ended_at=stop,
            host_usage={"schema": "or-host-usage/1", "host": "test-host", "model": "executor-model/v1",
                "scope": "remaining_to_task_end", "measured": ["tool_calls"], "tool_calls": 0,
                "window": {"started_at": self.h.actions.get(p.trace.model_info["decision_action_id"]).started_at, "ended_at": stop, "complete": True}})
        self.check_g(r)  # A later reference action must not replace the execute timestamp.
        e = self.close_g()["evaluations"][0]
        self.assertNotIn("tool_calls", e["cost"]["observed_fact"]["values"])
        self.assertTrue(any("lower bound 1" in s for s in e["cost"]["observed_fact"]["boundary_exclusions"]))
