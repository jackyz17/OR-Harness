"""Review regressions: cold-start runner inputs and contradictory host meters."""
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

RUNNER = {"host": "test-host", "model": "executor-model/v1",
          "phase": "initial_modelling", "source": "test runner configuration"}


class TestFirstProposal(CCase):
    def history_without_seed_actions(self):
        self.h = ORHarness(home=self.home, world_model=self.provider,
            embedding=LocalHashEmbeddingBackend(), execution_context=RUNNER)
        self.addCleanup(self.h.close)
        p = self.h.predict_strategy_outcome(self.task, {
            "action_type": "execute_strategy", "strategy_id": "S04", "method": REUSE}, "history")
        self.assertEqual(self.h.actions.query(task_id=self.task["task_id"]), [])
        r = self.run_g(p)
        self.check_g(r)
        self.finish_c(p)

    def call_cli(self, new_task, *, explicit=None):
        from or_harness import cli
        args = ["--home", self.home, "--world-model", "http://fixture.invalid::predicting-model",
                "--wm-timeout", "1", "--wm-max-tokens", "256"]
        if explicit is not None:
            args += ["--execution-context", json.dumps(explicit)]
        args += ["predict-strategy", "--task", json.dumps(new_task), "--episode", "first",
            "--candidate", json.dumps({"action_type": "execute_strategy", "strategy_id": "S04", "method": REUSE})]
        # Substitute only network/provider construction. _harness, snapshot,
        # CLI parsing, published retrieval and the predictor are real.
        def harness(**kwargs):
            return ORHarness(embedding=LocalHashEmbeddingBackend(), **kwargs)
        output = io.StringIO()
        with patch.object(cli, "ORHarness", side_effect=harness), patch(
                "or_harness.world_model.provider.HttpChatProvider", return_value=self.provider), \
                contextlib.redirect_stdout(output):
            status = cli.main(args)
        self.assertEqual(status, 0, output.getvalue())
        return self.provider.requests[-1]["prediction_context"]

    def test_ordinary_first_cli_request_receives_history_from_runner_environment(self):
        self.history_without_seed_actions()
        task = dict(self.task, task_id="brand-new-task")
        self.assertFalse(self.h.actions.query(task_id=task["task_id"]))
        with patch.dict(os.environ, {"OR_EXECUTION_CONTEXT": json.dumps(RUNNER)}):
            view = self.call_cli(task)
        e = view["cost_evidence"]
        self.assertEqual((len(e["reference_ranges"]), len(e["cases"])), (1, 1))
        self.assertEqual(e["current_conditions"]["phase"], "initial_modelling")
        self.assertEqual(e["current_conditions"]["execution_host"]["model"], RUNNER["model"])
        self.assertIn("execution_context", e["current_conditions"]["phase_basis"])
        self.assertFalse(self.h.actions.query(task_id=task["task_id"]), "no fake model/usage actions needed")

    def test_explicit_runner_flag_overrides_environment_and_unknown_inputs_stay_unknown(self):
        self.history_without_seed_actions()
        with patch.dict(os.environ, {"OR_EXECUTION_CONTEXT": json.dumps(dict(RUNNER, model="other-host-model"))}):
            view = self.call_cli(dict(self.task, task_id="explicit"), explicit=RUNNER)
            self.assertEqual(len(view["cost_evidence"]["cases"]), 1)
        with patch.dict(os.environ, {"OR_EXECUTION_CONTEXT": ""}):
            view = self.call_cli(dict(self.task, task_id="unknown"))
        e = view["cost_evidence"]
        self.assertFalse(e["reference_ranges"])
        self.assertEqual(e["current_conditions"]["phase"], "unknown")
        self.assertIn("execution-host model identity unavailable", e["missing"])

    def test_file_configuration_and_frozen_context_do_not_read_future_runner_state(self):
        config = Path(self.home) / "runner.json"
        config.write_text(json.dumps(RUNNER))
        with patch.dict(os.environ, {"OR_EXECUTION_CONTEXT": str(config)}):
            h = ORHarness(home=self.home, world_model=self.provider, embedding=LocalHashEmbeddingBackend())
        self.addCleanup(h.close)
        snapshot = h.snapshot(self.task, "first")
        ctx = h.build_prediction_context(self.task, "first")
        h.execution_context.update(model="later-model", phase="existing_model")
        p = h.predict_strategy_outcome(self.task, {"action_type": "execute_strategy", "strategy_id": "S04", "method": REUSE},
            "first", context=h.get_prediction_context(ctx.context_id))
        e = self.provider.requests[-1]["prediction_context"]["cost_evidence"]
        self.assertEqual(e["current_conditions"]["execution_host"]["model"], RUNNER["model"])
        self.assertEqual(e["current_conditions"]["phase"], "initial_modelling")
        historical = h.build_prediction_context(self.task, "first", snapshot=snapshot)
        self.assertEqual(historical.solving_context["execution_host"]["model"], RUNNER["model"])
        self.assertEqual(p.trace.model_info["cost_conditions"]["phase"], "initial_modelling")

    def test_real_model_progress_advances_the_initial_phase(self):
        self.h.execution_context = copy.deepcopy(RUNNER)
        first = self.predict_g(episode="first")
        self.assertEqual(first.trace.model_info["cost_conditions"]["phase"], "initial_modelling")
        self.h.report_action("model", self.task, "first", outcome={"artifact": "model.lp", "status": "available"})
        later = self.predict_g(episode="first")
        self.assertEqual(later.trace.model_info["cost_conditions"]["phase"], "existing_model")


class TestHostMeterIntegrity(CCase):
    def settle(self, *, tools=8, tokens=200, details=None, complete=None, counters=None,
               own_bound=None, extra_execution=False, second_report=False):
        p = self.predict_c(cost={"tool_calls": 8, "llm_tokens": 200})
        r = self.run_g(p)
        self.check_g(r)
        if extra_execution:
            work = Path(self.home) / p.prediction_id
            extra = self.h.execute(self.task, "S04", str(work / "solve.py"), str(work), solver="highs", episode_id="ep1")
            self.h.record(extra)
        stop = time.time()
        report = {"schema": "or-host-usage/1", "host": "test-host", "model": "executor-model/v1",
            "scope": "remaining_to_task_end", "measured": ["tool_calls", "llm_tokens"],
            "window": {"started_at": p.trace.created_at, "ended_at": stop, "complete": True}}
        if tools is not None:
            report["tool_calls"] = tools
        if tokens is not None:
            report["tokens"] = {"prompt_tokens": tokens, "completion_tokens": 0}
        if details is not None:
            report["call_details"] = [{"started_at": p.trace.created_at, "ended_at": stop, **d} for d in details]
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
        self.assertNotIn("tool_calls", e["reference_ranges"][0]["dimensions"])
        self.assertNotIn("tool_calls", e["cases"][0]["values"])

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
        self.assertEqual(e["cost"]["per_dim"]["llm_tokens"]["actual"], 200)

    def test_explicit_empty_complete_token_ledger_conflicts_with_positive_total(self):
        e, _ = self.settle(tools=1, details=[{"id": "t1", "kind": "tool"}],
                           complete=["tool_calls", "llm_tokens"])
        self.assertNotIn("llm_tokens", e["cost"]["observed_fact"]["values"])
        self.assertEqual(e["cost"]["per_dim"]["tool_calls"]["actual"], 1)

    def test_executor_lower_bound_rejects_zero_but_preserves_tokens(self):
        e, stats = self.settle(tools=0)
        self.assertNotIn("tool_calls", e["cost"]["observed_fact"]["values"])
        self.assertEqual(e["cost"]["per_dim"]["llm_tokens"]["actual"], 200)
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
        self.assertEqual(e["cost"]["per_dim"]["llm_tokens"]["actual"], 200)

    def test_report_own_floor_is_a_per_dimension_unknown(self):
        e, _ = self.settle(tools=0, own_bound=1)
        self.assertNotIn("tool_calls", e["cost"]["observed_fact"]["values"])
        self.assertEqual(e["cost"]["per_dim"]["llm_tokens"]["actual"], 200)

    def test_another_report_does_not_hide_a_conflict(self):
        e, _ = self.settle(details=[{"id": "t1", "kind": "tool"}], second_report=True)
        self.assertNotIn("tool_calls", e["cost"]["observed_fact"]["values"])
        self.assertEqual(e["cost"]["per_dim"]["llm_tokens"]["actual"], 200)

    def test_matching_summary_and_deduplicated_details_remain_actuals(self):
        tool = {"id": "t1", "kind": "tool"}
        model = {"id": "m1", "kind": "model", "prompt_tokens": 120, "completion_tokens": 80}
        e, stats = self.settle(tools=1, details=[tool, model, dict(tool), dict(model)],
                              complete=["tool_calls", "llm_tokens"])
        self.assertEqual(e["cost"]["per_dim"]["tool_calls"]["actual"], 1)
        self.assertEqual(e["cost"]["per_dim"]["llm_tokens"]["actual"], 200)
        self.assertEqual(stats["coverage"]["tool_calls"]["n_actual"], 1)

    def test_partial_ledger_lower_bound_can_refute_a_total(self):
        e, _ = self.settle(tools=1, details=[{"id": "t1", "kind": "tool"},
            {"id": "t2", "kind": "tool"}], complete=[])
        self.assertNotIn("tool_calls", e["cost"]["observed_fact"]["values"])
        self.assertEqual(e["cost"]["per_dim"]["llm_tokens"]["actual"], 200)

    def test_explicit_complete_zero_tokens_is_a_valid_raw_error(self):
        e, stats = self.settle(tools=1, tokens=0, details=[{"id": "t1", "kind": "tool"}],
            complete=["tool_calls", "llm_tokens"])
        error = e["cost"]["per_dim"]["llm_tokens"]
        self.assertEqual((error["actual"], error["abs_error"], error["signed_error"]), (0, 200, -200))
        self.assertIsNone(error["log_ratio"])
        self.assertEqual(stats["coverage"]["llm_tokens"]["n_actual"], 1)
        self.assertEqual(stats["coverage"]["llm_tokens"]["n_log_ratio"], 0)

    def test_late_check_does_not_move_the_executors_proven_call(self):
        p = self.predict_c(cost={"tool_calls": 0})
        r = self.run_g(p)
        stop = time.time()
        self.h.report_action("finish_task", self.task, "ep1", started_at=stop, ended_at=stop,
            host_usage={"schema": "or-host-usage/1", "host": "test-host", "model": "executor-model/v1",
                "scope": "remaining_to_task_end", "measured": ["tool_calls"], "tool_calls": 0,
                "window": {"started_at": p.trace.created_at, "ended_at": stop, "complete": True}})
        self.check_g(r)  # A later reference action must not replace the execute timestamp.
        e = self.close_g()["evaluations"][0]
        self.assertNotIn("tool_calls", e["cost"]["observed_fact"]["values"])
        self.assertTrue(any("lower bound 1" in s for s in e["cost"]["observed_fact"]["boundary_exclusions"]))
