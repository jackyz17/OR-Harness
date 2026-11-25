"""r13: the cost observation span matches the prediction's declared cost
span, and a clearly measured wall-clock span IS calibratable.

Two P1 defects this file pins:

3. ``strategy_prediction`` asks the model to predict the cost from the
   decision to the END of the task, but the close-out observed only a
   single attempt. A prediction of 300 tokens (100 failed attempt + 200
   repaired retry) was compared against the FIRST attempt's 100, reporting
   a large over-estimate.
4. Every non-cumulative dimension's ``total`` was forced to ``None``, and
   the evaluation required a non-``None`` total — so a backfilled real
   ``remaining_latency_s`` was "completely measured" yet scored as
   "unobserved". "Cannot be summed" is not "cannot be compared": a single
   explicit span is a comparable observation.
"""
from decision_fixtures import bind_legacy, execute_scored, planned_prediction, prediction_anchor


import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from or_harness.world_model.episode_closeout import (  # noqa: E402
    _aggregate_costs, _aggregated_dim)
from or_harness.core.schema import CostVector  # noqa: E402


class TestWallClockIsComparable(unittest.TestCase):

    def test_single_span_has_a_comparable_value(self):
        dim = _aggregated_dim("remaining_latency_s", [30.0], 1)
        self.assertIsNone(dim["total"], "a span is never reported as a sum")
        self.assertEqual(dim["comparable"]["value"], 30.0)
        self.assertFalse(dim["comparable"]["is_sum"])
        self.assertTrue(dim["complete"])

    def test_distinct_spans_do_not_combine(self):
        dim = _aggregated_dim("remaining_latency_s", [10.0, 20.0], 2)
        self.assertIsNone(dim["comparable"]["value"])
        self.assertIn("never summed", dim["comparable"]["basis"])

    def test_identical_spans_collapse_to_one_value(self):
        dim = _aggregated_dim("latency_s", [5.0, 5.0], 2)
        self.assertEqual(dim["comparable"]["value"], 5.0)

    def test_aggregate_costs_reports_comparable_for_a_backfilled_span(self):
        c = CostVector(remaining_latency_s=30.0,
                       measured={"remaining_latency_s"})
        agg = _aggregate_costs([c])
        self.assertEqual(agg["remaining_latency_s"]["comparable"]["value"],
                         30.0)
        self.assertEqual(agg["remaining_latency_s"]["n_measured"], 1)


from tests.harness.test_episode_closeout import M4Case, _task  # noqa: E402


class TestRemainingCostObservation(M4Case):
    """The report's case: predicted 300 remaining tokens, real spend = a
    100-token failed attempt + a 200-token repaired retry. The observation
    must cover BOTH, not just the first attempt."""

    def test_remaining_cost_covers_retries(self):
        from pathlib import Path

        h = self.h
        # The provider predicts a REMAINING cost of 300 tokens.
        self.provider.payload = {
            "benefit": {"kind": "effective_completion",
                        "metric": "task_result_check_passed",
                        "unit": "boolean", "value": 0.8,
                        "baseline": {"kind": "declared", "value": 0.5}},
            "cost": {"output_tokens": 300.0},
        }
        task = _task("t1")
        prediction = planned_prediction(h,
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S04"}, "ep1")
        self.assertEqual(prediction.cost.scope, "remaining_to_task_end")
        # A failed first attempt then a repaired retry, both in the same
        # episode. The real spend is amended through the proper channel so
        # each record measures llm_tokens with a trusted source.
        goal = Path(self.home) / "ws_rem"
        goal.mkdir(parents=True, exist_ok=True)
        (goal / "solve.py").write_text(
            "raise RuntimeError('first attempt fails')\n",
            encoding="utf-8")
        first = execute_scored(h,task, "S04", str(goal / "solve.py"), str(goal),
                          solver="highs", episode_id="ep1")
        h.record(first)
        h.bank.update_cost(first.execution_id, source="provider_usage",
                           llm_tokens=100.0)
        second = self.solve(task, strategy="S04", episode_id="ep1",
                            tag="rem2")
        h.bank.update_cost(second.execution_id, source="provider_usage",
                           llm_tokens=200.0)
        bind_legacy(h,prediction.prediction_id, second.action_id)
        # Attempt receipts alone are not the complete remaining window.
        # The host reports the shared-anchor -> real-termination total.
        import time
        end = time.time()
        h.report_action("finish_task", task, "ep1", started_at=end, ended_at=end,
            host_usage={"schema": "or-host-usage/1", "host": "test-host",
                "model": "executor/v1", "scope": "remaining_to_task_end",
                "tokens": {"prompt_tokens": 200, "completion_tokens": 300},
                "measured": ["llm_tokens"], "window": {
                    "started_at": prediction_anchor(self.h,prediction), "ended_at": end,
                    "complete": True, "end_event": "finish_task"}})
        result = h.close_episode("t1", "ep1")
        evaluation = result["evaluations"][0]
        entry = evaluation["cost"]["per_dim"].get("output_tokens")
        self.assertIsNotNone(
            entry,
            "the remaining-cost prediction must be compared against the "
            "real remaining spend, and 300 == 100 + 200")
        self.assertEqual(entry["predicted"], 300.0)
        self.assertEqual(entry["actual"], 300.0)
        self.assertAlmostEqual(entry["log_ratio"], 0.0, places=5)

    def test_a_backfilled_remaining_latency_is_calibratable(self):
        """The report's case 4: a real ``remaining_latency_s`` backfilled on
        the record must produce a cost error, not 'unobserved'."""
        from pathlib import Path

        h = self.h
        # The model predicts a 30s remaining latency; the real operation
        # takes 600s (modelling, debugging, retries).
        self.provider.payload = {
            "benefit": {"kind": "effective_completion",
                        "metric": "task_result_check_passed",
                        "unit": "boolean", "value": 0.8,
                        "baseline": {"kind": "declared", "value": 0.5}},
            "cost": {"remaining_latency_s": 30.0},
        }
        task = _task("t1")
        prediction = planned_prediction(h,
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S04"}, "ep1")
        record = self.solve(task, strategy="S04", episode_id="ep1",
                            tag="lat")
        # Backfill the REAL remaining span (a trusted measurement).
        h.bank.update_cost(record.execution_id, source="agent_observed",
                           remaining_latency_s=600.0)
        bind_legacy(h,prediction.prediction_id, record.action_id)
        result = h.close_episode("t1", "ep1")
        evaluation = result["evaluations"][0]
        entry = evaluation["cost"]["per_dim"].get("remaining_latency_s")
        self.assertIsNotNone(
            entry,
            "a clearly measured wall-clock span must participate in "
            "calibration: 'cannot be summed' is not 'cannot be compared'")
        self.assertEqual(entry["predicted"], 30.0)
        self.assertEqual(entry["actual"], 600.0)
        # log(600/30) = log(20) ~= 2.996 (the model predicted ten minutes
        # as half a minute — the error must be visible).
        self.assertAlmostEqual(entry["log_ratio"], 2.9957, places=3)


from tests.harness.c_fixtures import CCase, REUSE, REBUILD


class TestHostCostWindow(CCase):
    def test_counter_delta_and_duplicate_report_cover_full_window(self):
        prediction = self.predict_c(cost={"tool_calls": 8, "output_tokens": 0})
        record = self.run_g(prediction, record=False)
        self.check_g(record)
        result, _ = self.finish_c(prediction, counters=True, duplicate_record=record)
        cost = result["evaluations"][0]["cost"]
        self.assertEqual(cost["per_dim"]["tool_calls"]["actual"], 8)
        self.assertEqual(cost["per_dim"]["output_tokens"]["actual"], 0)
        self.assertEqual(cost["observed_fact"]["work_facts"]["host_rounds"], 2)
        self.assertEqual(cost["observed_fact"]["work_facts"]["execution_attempts"], 1)
        # A remaining-window total is not rewritten as the attempt's spend.
        self.assertNotIn("llm_tokens", self.h.bank.get(record.execution_id).cost.measured_dims())

    def test_timed_calls_count_input_output_and_each_id_once(self):
        import time
        prediction = self.predict_c(cost={"tool_calls": 2, "output_tokens": 80})
        record = self.run_g(prediction, record=False)
        self.check_g(record)
        start, stop = prediction_anchor(self.h,prediction), time.time()
        calls = [{"id": "host-1", "kind": "model", "started_at": start,
            "ended_at": stop, "prompt_tokens": 120, "completion_tokens": 80,
            "reasoning_tokens": 60, "cached_tokens": 90},
            {"id": "tool-1", "kind": "tool", "started_at": start, "ended_at": stop},
            {"id": "tool-2", "kind": "tool", "started_at": start, "ended_at": stop}]
        result, _ = self.finish_c(prediction, calls=calls + [dict(calls[1])], duplicate_record=record)
        cost = result["evaluations"][0]["cost"]
        self.assertEqual(cost["per_dim"]["tool_calls"]["actual"], 2)
        self.assertEqual(cost["per_dim"]["output_tokens"]["actual"], 80)
        self.assertEqual(cost["observed_fact"]["work_facts"]["host_rounds"], 1)

    def test_ambiguous_attempt_totals_and_last_action_are_lower_bounds(self):
        from or_harness.world_model.episode_closeout import summarize_real_outcome
        prediction = self.predict_c(cost={"tool_calls": 8, "remaining_latency_s": 1})
        record = self.run_g(prediction)
        self.check_g(record)
        self.h.bank.update_cost(record.execution_id, source="provider_usage", llm_tokens=300, tool_calls=8)
        result = self.close_g()
        cost = result["evaluations"][0]["cost"]
        self.assertFalse(cost.get("per_dim"))
        summary = summarize_real_outcome(self.h, self.h.strategy_predictions.get(prediction.prediction_id))
        self.assertFalse(summary.cost["remaining_latency_s"]["complete"])
        self.assertIn("LOWER BOUND", summary.cost["remaining_latency_s"]["boundary_note"])

    def test_failed_and_aborted_terminal_windows_have_spent_cost(self):
        for terminal in ("failed", "aborted"):
            with self.subTest(terminal=terminal):
                prediction = self.predict_c(terminal, cost={"tool_calls": 8})
                record = self.run_g(prediction, crash=True)
                result, _ = self.finish_c(prediction, terminal=terminal)
                cost = result["evaluations"][0]["cost"]
                self.assertEqual(cost["per_dim"]["tool_calls"]["actual"], 8)
                self.assertIn("remaining_latency_s", cost["observed_fact"]["values"])

    def test_missing_entire_cost_still_has_observation_and_baseline(self):
        from or_harness.world_model.episode_closeout import build_calibration_summary
        prediction = self.predict_c()
        record = self.run_g(prediction)
        self.check_g(record)
        result, _ = self.finish_c(prediction)
        cost = result["evaluations"][0]["cost"]
        self.assertEqual(cost["eligibility"], "not_predicted")
        self.assertEqual(cost["observed_fact"]["values"]["tool_calls"], 8)
        before = build_calibration_summary(self.h)["conditional_cost"]
        coverage = before["coverage"]["tool_calls"]
        self.assertEqual(coverage, {"n_actual": 1, "n_predicted": 0, "n_error": 0, "n_log_ratio": 0})
        self.close_g()
        self.assertEqual(build_calibration_summary(self.h)["conditional_cost"], before)

    def test_zero_errors_and_log_support_use_their_own_populations(self):
        # A real execution proves tool_calls >= 1. Tokens can legitimately
        # be zero when the host made no model calls.
        from or_harness.world_model.episode_closeout import build_calibration_summary
        # The host's default generation is 100 completion tokens. A zero
        # prediction is a REAL forecast and gets a finite absolute error; a
        # positive prediction also gets a log ratio.
        for episode, predicted, actual in (("zero", 0, 100), ("positive", 100, 100)):
            prediction = self.predict_c(episode, cost={"output_tokens": predicted})
            record = self.run_g(prediction)
            self.check_g(record)
            result, _ = self.finish_c(prediction)
            error = result["evaluations"][0]["cost"]["per_dim"]["output_tokens"]
            self.assertEqual(error["abs_error"], actual - predicted)
            self.assertEqual(error["signed_error"], actual - predicted)
        summary = build_calibration_summary(self.h, min_samples=2)
        group = next(g for g in summary["groups"].values()
                     if "output_tokens" in g["mean_cost_abs_error"])
        self.assertEqual(group["mean_cost_abs_error"]["output_tokens"], 50)
        self.assertEqual(group["cost_absolute_evidence"]["output_tokens"]["n_distinct_episodes"], 2)
        self.assertEqual(group["cost_evidence"]["output_tokens"]["n_distinct_episodes"], 1)
        self.assertEqual(group["cost_evidence"]["output_tokens"]["evidence"], "insufficient_evidence")
        self.assertEqual(summary["conditional_cost"]["coverage"]["output_tokens"],
            {"n_actual": 2, "n_predicted": 2, "n_error": 2, "n_log_ratio": 1})

    def test_remaining_latency_is_the_only_time_prediction(self):
        prediction = self.predict_c(cost={"remaining_latency_s": 10})
        record = self.run_g(prediction)
        self.check_g(record)
        result, _ = self.finish_c(prediction)
        per_dim = result["evaluations"][0]["cost"]["per_dim"]
        self.assertEqual(set(per_dim), {"remaining_latency_s"})
        self.assertNotIn("solver_runtime_s", per_dim)
        self.assertNotIn("latency_s", per_dim)

    def test_two_candidates_use_the_recorded_shared_decision_anchor(self):
        from or_harness.world_model.prediction import ActionSpec
        self.seed_host("shared")
        self.provider.payload["cost"] = {"tool_calls": 8}
        plan = self.h.plan_next(self.task, "shared", candidates=[
            ActionSpec("execute_strategy", self.task["task_id"], strategy_id="S04", method=REUSE),
            ActionSpec("execute_strategy", self.task["task_id"], strategy_id="S04", method=REBUILD)],
            limits={"horizon": 1})
        predictions = self.h.strategy_predictions.query(task_id=self.task["task_id"], episode_id="shared")
        self.assertEqual(len(predictions), 2)
        self.assertEqual({p.trace.model_info["decision_action_id"] for p in predictions},
                         {plan["decision_action_id"]})
        prediction = predictions[0]
        record = self.run_g(prediction)
        self.check_g(record)
        result, _ = self.finish_c(prediction)
        fact = result["evaluations"][0]["cost"]["observed_fact"]
        self.assertEqual(fact["window"]["anchor"], self.h.actions.get(plan["decision_action_id"]).started_at)
        self.assertEqual(fact["values"]["tool_calls"], 8)

    def test_complete_episode_call_ledger_is_sliced_to_actual_window(self):
        import time
        prediction = self.predict_c(cost={"tool_calls": 1})
        record = self.run_g(prediction)
        self.check_g(record)
        end = time.time()
        self.h.report_action("finish_task", self.task, "ep1", started_at=end, ended_at=end)
        stop = time.time()
        anchor = prediction_anchor(self.h,prediction)
        calls = [{"id": "before", "kind": "tool", "started_at": anchor - 2, "ended_at": anchor - 1},
                 {"id": "inside", "kind": "tool", "started_at": anchor, "ended_at": end},
                 {"id": "after", "kind": "tool", "started_at": (end + stop) / 2, "ended_at": stop}]
        self.h.report_action("verify", self.task, "ep1", started_at=stop, ended_at=stop,
            outcome={"host_usage": {"schema": "or-host-usage/1", "host": "test-host",
                "model": "executor-model/v1", "scope": "episode", "measured": ["tool_calls"],
                "window": {"started_at": anchor - 3, "ended_at": stop, "complete": True},
                "call_details": calls}})
        result = self.close_g()
        self.assertEqual(result["evaluations"][0]["cost"]["per_dim"]["tool_calls"]["actual"], 1)

    def test_late_full_window_report_republishes_without_rewriting_prediction(self):
        import time
        prediction = self.predict_c(cost={"tool_calls": 8})
        record = self.run_g(prediction)
        self.check_g(record)
        end = time.time()
        self.h.report_action("finish_task", self.task, "ep1", started_at=end, ended_at=end)
        result = self.close_g()
        self.assertNotIn("tool_calls", result["evaluations"][0]["cost"].get("per_dim") or {})
        self.h.report_action("finish_task", self.task, "ep1", started_at=end, ended_at=end,
            host_usage={"schema": "or-host-usage/1", "host": "test-host", "model": "executor-model/v1",
                "scope": "remaining_to_task_end", "tool_calls": 8, "measured": ["tool_calls"],
                "window": {"started_at": prediction_anchor(self.h,prediction), "ended_at": end, "complete": True}})
        summary = self.h.calibration_summary()
        self.assertEqual(summary["conditional_cost"]["coverage"]["tool_calls"]["n_actual"], 1)
        self.predict_c("next-late")
        evidence = self.provider.requests[-1]["prediction_context"]["cost_evidence"]
        self.assertEqual(evidence["cases"][0]["cost"]["tool_calls"], 8)
        self.assertEqual(self.h.strategy_predictions.get(prediction.prediction_id).cost.expected.tool_calls, 8)

    def test_scoped_cli_backfill_enters_next_request_without_becoming_attempt_cost(self):
        import contextlib
        import io
        import json
        import time
        from unittest.mock import patch
        from or_harness import cli
        prediction = self.predict_c(cost={"tool_calls": 8})
        record = self.run_g(prediction)
        self.check_g(record)
        end = time.time()
        self.h.report_action("finish_task", self.task, "ep1", started_at=end, ended_at=end)
        self.close_g()
        report = {"schema": "or-host-usage/1", "host": "test-host", "model": "executor-model/v1",
            "scope": "remaining_to_task_end", "tool_calls": 8, "measured": ["tool_calls"],
            "window": {"started_at": prediction_anchor(self.h,prediction), "ended_at": end, "complete": True}}
        with patch.object(cli, "_harness", return_value=self.h), patch.object(self.h, "close"), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(["amend-cost", record.execution_id, "--usage-file", json.dumps(report)]), 0)
        self.assertNotIn("tool_calls", self.h.bank.get(record.execution_id).cost.measured_dims())
        self.predict_c("next-cli")
        evidence = self.provider.requests[-1]["prediction_context"]["cost_evidence"]
        self.assertEqual(evidence["cases"][0]["cost"]["tool_calls"], 8)

    def test_wrong_numeric_window_remains_unknown_with_source_reason(self):
        import time
        prediction = self.predict_c(cost={"tool_calls": 8})
        record = self.run_g(prediction)
        self.check_g(record)
        end = time.time()
        self.h.report_action("finish_task", self.task, "ep1", started_at=end, ended_at=end,
            host_usage={"schema": "or-host-usage/1", "host": "test-host", "model": "executor-model/v1",
                "scope": "remaining_to_task_end", "tool_calls": 8, "measured": ["tool_calls"],
                "window": {"started_at": prediction_anchor(self.h,prediction) - 10, "ended_at": end, "complete": True}})
        result = self.close_g()
        cost = result["evaluations"][0]["cost"]
        self.assertNotIn("tool_calls", cost.get("per_dim") or {})
        self.assertTrue(cost["observed_fact"]["boundary_exclusions"])

    def test_script_wall_proxy_cannot_score_inner_solver_time(self):
        prediction = self.predict_c(cost={"remaining_latency_s": 1})
        self.run_g(prediction, crash=True)
        result, _ = self.finish_c(prediction, terminal="failed")
        cost = result["evaluations"][0]["cost"]
        self.assertNotIn("solver_runtime_s", cost["per_dim"])
        self.assertIn("remaining_latency_s", cost["per_dim"])

    def test_later_offline_execution_is_outside_original_cost_window(self):
        from pathlib import Path
        from or_harness.world_model.episode_closeout import build_calibration_summary
        prediction = self.predict_c(cost={"tool_calls": 8})
        record = self.run_g(prediction)
        self.check_g(record)
        self.finish_c(prediction)
        work = Path(self.home) / prediction.prediction_id
        later = execute_scored(self.h,self.task, "S04", str(work / "solve.py"), str(work),
                               solver="highs", episode_id="ep1")
        self.h.record(later)
        fact = build_calibration_summary(self.h)["conditional_cost"]["cases"][0]
        self.assertEqual(fact["execution_ids"], [record.execution_id])
        self.assertEqual(fact["work_facts"]["execution_attempts"], 1)
        self.assertEqual(fact["values"]["tool_calls"], 8)


class TestConditionalCostRequest(CCase):
    def make_history(self, episode, method, tools, **kwargs):
        prediction = self.predict_c(episode, method=method, **kwargs)
        record = self.run_g(prediction)
        self.check_g(record)
        self.finish_c(prediction, tools=tools)
        return prediction

    def test_request_contains_host_phase_ranges_and_candidate_work(self):
        from or_harness.world_model.episode_closeout import build_calibration_summary
        self.provider.usage = {"prompt_tokens": 0, "completion_tokens": 0}
        self.predict_c("reuse")  # trusted explicit zero before the first decision
        first = self.make_history("reuse", REUSE, 8)
        self.provider.usage = {"prompt_tokens": 1, "completion_tokens": 2}
        self.predict_c("rebuild", method=REBUILD)  # unexecuted warm-up spends exactly 3 tokens
        second = self.make_history("rebuild", REBUILD, 20)
        self.assertEqual(first.trace.model_info["frozen_budget_state"]["consumption"]["total_cost"]["llm_tokens"], 0)
        self.assertEqual(second.trace.model_info["frozen_budget_state"]["consumption"]["total_cost"]["llm_tokens"], 3)
        self.predict_c("next-A", method=REUSE)
        request_a = self.provider.requests[-1]
        self.predict_c("next-B", method=REBUILD)
        request_b = self.provider.requests[-1]
        for request in (request_a, request_b):
            evidence = request["prediction_context"]["cost_evidence"]
            self.assertNotIn("execution_host", evidence["current_conditions"])
            # C no longer classifies a modelling phase; the optional model
            # artifact is ordinary context only.
            self.assertNotIn("phase", evidence["current_conditions"])
            self.assertFalse(evidence["current_conditions"]["phase_used"])
            self.assertTrue(evidence["current_conditions"]["reusable_artifacts"])
            tool = evidence["global_cost"]["dimensions"]["tool_calls"]
            self.assertEqual(tool["p50"], 14)
            self.assertEqual(tool["n"], 2)
            self.assertEqual({c["cost"]["tool_calls"]
                              for c in evidence["cases"]}, {8, 20})
            self.assertTrue(all(c["source_refs"] for c in evidence["cases"]))
            self.assertNotIn("conditional_cost", request["prediction_context"]["strategy_outcome_calibration"])
        self.assertIn("Reuse", " ".join(request_a["candidate"]["method"]["steps"]))
        self.assertIn("three subproblems", " ".join(request_b["candidate"]["method"]["steps"]))
        # Different accumulated WM token spend did not fragment the group.
        groups = build_calibration_summary(self.h)["conditional_cost"]["groups"]
        self.assertEqual(len(groups), 1)

    def test_family_mismatch_is_a_weak_reference_and_artifact_is_not_a_phase(self):
        self.make_history("history", REUSE, 8)
        self.predict_c("other-host", model="other-executor")
        first = self.provider.requests[-1]["prediction_context"]["cost_evidence"]
        # The usage-report model identity is not a cost condition.
        self.assertGreaterEqual(len(first["cases"]), 1)
        self.assertEqual(first["cases"][0]["grade"], "strict")
        # A missing model artifact is NOT a phase: evidence stays available.
        self.predict_c("initial", existing=False)
        second = self.provider.requests[-1]["prediction_context"]["cost_evidence"]
        self.assertGreaterEqual(len(second["cases"]), 1)
        self.assertFalse(second["current_conditions"]["reusable_artifacts"])
        self.task["family"] = "scheduling"
        self.predict_c("other-family")
        third = self.provider.requests[-1]["prediction_context"]["cost_evidence"]
        # Same measurement scope and no KNOWN solving conflict: the case is a
        # broader_reference, never silently promoted to strict and never
        # dropped merely for a family label.
        self.assertEqual(third["reference_coverage"]["strict"]
                         ["n_cases_included"], 0)
        self.assertGreaterEqual(third["reference_coverage"]
                                ["broader_reference"]["n_cases_included"], 1)
        self.assertEqual(third["evidence_status"], "broader_reference_only")

    def test_open_episode_cost_never_leaks_into_next_request(self):
        prediction = self.predict_c()
        record = self.run_g(prediction)
        self.h.bank.update_cost(record.execution_id, source="agent_observed", tool_calls=99)
        self.predict_c("next")
        evidence = self.provider.requests[-1]["prediction_context"]["cost_evidence"]
        self.assertEqual(evidence["cases"], [])
        self.assertEqual(evidence["reference_coverage"]["strict"]
                         ["n_cases_included"], 0)
        self.assertEqual(evidence["evidence_status"], "insufficient_evidence")

    def test_actual_solving_limits_stay_separate_and_known_mismatch_is_withheld(self):
        self.make_history("short", REUSE, 8)
        slow = self.predict_c("long", method=REUSE)
        record = self.run_g(slow, execution_config={"time_limit": 180})
        self.check_g(record)
        self.finish_c(slow, tools=20)
        groups = self.h.calibration_summary()["conditional_cost"]["groups"]
        self.assertEqual(len(groups), 2)
        # Solving limits are NOT a primary grouping key any more. Without a
        # declared conflict the two historical windows are both comparable;
        # a KNOWN conflict with the candidate's declared time_limit makes
        # them unavailable, never silently relabelled.
        self.predict_c("next-open")
        open_cases = self.provider.requests[-1]["prediction_context"]["cost_evidence"]["cases"]
        self.assertEqual({case["cost"]["tool_calls"] for case in open_cases},
                         {8, 20})
        self.seed_host("next-limited")
        planned_prediction(self.h,self.task, {"action_type": "execute_strategy",
            "strategy_id": "S04", "method": REUSE, "solver": "highs", "config": {"time_limit": 60}},
            "next-limited")
        evidence = self.provider.requests[-1]["prediction_context"]["cost_evidence"]
        # Solving limits are a COMPARABILITY deduction, not an exclusion: the
        # 60-second window is strict, the 180-second window stays as a
        # broader_reference with the known difference recorded.
        by_tools = {case["cost"]["tool_calls"]: case
                    for case in evidence["cases"]}
        self.assertEqual(set(by_tools), {8, 20})
        self.assertEqual(by_tools[8]["grade"], "strict")
        self.assertEqual(by_tools[20]["grade"], "broader_reference")
        self.assertIn("solving config time_limit",
                      by_tools[20].get("differences") or [])
        self.assertLess(by_tools[20]["relevance_weight"], 1.0)
        self.assertEqual(evidence["reference_coverage"]["unavailable"], 0)

    def test_unreported_runner_phase_no_longer_limits_cost_evidence(self):
        self.make_history("history", REUSE, 8)
        self.h.report_action("verify", self.task, "unknown-phase",
            host_usage={"schema": "or-host-usage/1", "host": "test-host",
                        "model": "executor-model/v1", "measured": []})
        planned_prediction(self.h,self.task, {"action_type": "execute_strategy",
            "strategy_id": "S04", "method": REUSE}, "unknown-phase")
        evidence = self.provider.requests[-1]["prediction_context"]["cost_evidence"]
        # A missing runner phase is not a condition of C any more: the same
        # comparable remaining-task windows are still available.
        self.assertNotIn("phase", evidence["current_conditions"])
        self.assertFalse(evidence["current_conditions"]["phase_used"])
        self.assertGreaterEqual(len(evidence["cases"]), 1)
        self.assertEqual(evidence["cases"][0]["grade"], "strict")
        self.assertNotIn("decision phase unavailable",
                         " ".join(evidence["missing"]))


class TestOutputTokenCost(CCase):
    """Generated-token C through real decisions, execution and close-out."""

    def finish_output(self, prediction, **usage):
        import time
        end = time.time()
        report = {"schema": "or-host-usage/1", "host": "test-host",
            "model": "executor-model/v1", "scope": "remaining_to_task_end",
            "measured": ["llm_tokens"], "window": {
                "started_at": prediction_anchor(self.h, prediction), "ended_at": end,
                "anchor_action_id": prediction.trace.model_info["decision_action_id"],
                "complete": True, "end_event": "finish_task"}, **usage}
        finish = self.h.report_action("finish_task", self.task, prediction.candidate.episode_id,
            started_at=end, ended_at=end, host_usage=report)
        return self.h.close_episode(self.task["task_id"], prediction.candidate.episode_id,
            finish_action_id=finish["action_id"], min_calibration_samples=1)

    def test_completion_cost_is_separate_from_raw_usage_and_next_decision_scale(self):
        prediction = self.predict_c(cost={"output_tokens": 500})
        self.assertEqual(prediction.cost.measured, ["output_tokens"])
        self.assertEqual(self.provider.requests[-1]["cost_scope"]["token_dimension"], "output_tokens")
        record = self.run_g(prediction, record=False)
        self.check_g(record)
        self.h.record(record, host_usage={"schema": "or-host-usage/1", "host": "test-host",
            "model": "executor-model/v1", "scope": "attempt", "measured": ["llm_tokens"],
            "tokens": {"prompt_tokens": 10000, "completion_tokens": 400, "reasoning_tokens": 100}})
        raw = self.h.bank.get(record.execution_id).cost
        self.assertEqual(raw.llm_tokens, 10400)
        self.assertEqual(raw.output_tokens, 400)
        self.assertEqual(self.h.budget_view(self.task["task_id"], "ep1")["consumption"]["total_cost"]["llm_tokens"], 10400)
        result = self.finish_output(prediction, tokens={"prompt_tokens": 10000, "completion_tokens": 400})
        cost = result["evaluations"][0]["cost"]
        self.assertEqual(cost["per_dim"]["output_tokens"]["actual"], 400)
        self.assertEqual(cost["observed_fact"]["values"]["llm_tokens"], 10400)
        self.assertEqual(cost["per_dim"]["output_tokens"]["signed_error"], -100)

        later = self.predict_c("next", cost={"output_tokens": 300})
        decision = self.h.actions.get(later.trace.model_info["decision_action_id"])
        basis = decision.outcome["selected_score"]["scoring_basis"]
        self.assertEqual(basis["cost_dimensions"], ["output_tokens"])
        self.assertEqual(basis["cost_scales"]["output_tokens"], 400)
        self.assertEqual(basis["cost_scale_sources"]["output_tokens"]["source"], "pre_decision_measured_history")
        evidence = self.provider.requests[-1]["prediction_context"]["cost_evidence"]
        output = evidence["global_cost"]["dimensions"]["output_tokens"]
        self.assertEqual(output["p50"], 400)
        self.assertEqual({case["cost"].get("output_tokens")
                          for case in evidence["cases"]}, {400})

    def test_initial_and_retry_anchors_slice_one_complete_host_ledger(self):
        import time
        first = self.predict_c(cost={"output_tokens": 700})
        self.run_g(first, crash=True)
        first_end = time.time()
        retry = self.predict_c(cost={"output_tokens": 300})
        record = self.run_g(retry)
        self.check_g(record)
        second_end = time.time()
        before = {"id": "first-generation", "kind": "model",
            "started_at": prediction_anchor(self.h, first), "ended_at": first_end,
            "prompt_tokens": 10000, "completion_tokens": 400}
        after = {"id": "retry-generation", "kind": "model",
            "started_at": prediction_anchor(self.h, retry), "ended_at": second_end,
            "prompt_tokens": 20000, "completion_tokens": 300}
        result = self.finish_output(first, call_details=[before, after, dict(before)],
            call_details_complete=["llm_tokens"])
        costs = {e["prediction_id"]: e["cost"] for e in result["evaluations"]}
        self.assertEqual(costs[first.prediction_id]["per_dim"]["output_tokens"]["actual"], 700)
        self.assertEqual(costs[retry.prediction_id]["per_dim"]["output_tokens"]["actual"], 300)
        self.assertEqual(costs[first.prediction_id]["observed_fact"]["work_facts"]["host_rounds"], 2)
        self.assertEqual(costs[retry.prediction_id]["observed_fact"]["work_facts"]["host_rounds"], 1)

    def test_total_only_is_not_output_and_legacy_prediction_stays_readable(self):
        # A NEW prediction may not use the raw input-plus-output total.
        prediction = self.predict_c(cost={"llm_tokens": 10400})
        self.assertEqual(prediction.status, "invalid")
        joined = " ".join(prediction.notes)
        self.assertIn("historical observation dimension", joined)
        self.assertIn("remaining_latency_s", joined)

        # An ARCHIVED six-dimensional prediction remains READABLE and is not
        # converted into the new contract.
        from or_harness.world_model.contracts import (
            ExpectedCost, StrategyOutcomePrediction,
        )
        from or_harness.core.schema import CostVector
        old = StrategyOutcomePrediction(
            candidate=prediction.candidate,
            status="contract_only",
            cost=ExpectedCost(
                expected=CostVector(llm_tokens=10400.0,
                                    measured={"llm_tokens"}),
                measured=["llm_tokens"]),
            trace=prediction.trace)
        dumped = old.to_dict()
        from or_harness.world_model.contracts import StrategyOutcomePrediction as P
        restored = P.from_dict(dumped)
        self.assertEqual(restored.to_dict(), dumped)
        self.assertIsNone(restored.cost.prediction_contract)
        self.assertEqual(restored.cost.expected.llm_tokens, 10400.0)
        self.assertEqual(restored.cost.measured, ["llm_tokens"])

    def test_missing_prediction_still_provides_output_baseline(self):
        prediction = self.predict_c(cost=None)
        record = self.run_g(prediction)
        self.check_g(record)
        self.finish_output(prediction, tokens={"completion_tokens": 400})
        coverage = self.h.calibration_summary()["conditional_cost"]["coverage"]["output_tokens"]
        self.assertEqual((coverage["n_actual"], coverage["n_predicted"], coverage["n_error"]), (1, 0, 0))

    def test_counter_increments_preserve_zero_and_reject_completion_conflicts(self):
        cases = [(400, None, 400), (0, None, 0), (400, 500, None)]
        for index, (increment, aggregate, expected) in enumerate(cases):
            with self.subTest(increment=increment, aggregate=aggregate):
                prediction = self.predict_c("counter" + str(index), cost={"output_tokens": 400})
                record = self.run_g(prediction)
                self.check_g(record)
                report = {"counters": {"start": {"prompt_tokens": 20000, "completion_tokens": 800},
                    "end": {"prompt_tokens": 30000, "completion_tokens": 800 + increment}}}
                if aggregate is not None:
                    report["tokens"] = {"prompt_tokens": 10000, "completion_tokens": aggregate}
                result = self.finish_output(prediction, **report)
                cost = result["evaluations"][0]["cost"]
                self.assertEqual(cost["observed_fact"]["values"].get("output_tokens"), expected)
                if expected is None:
                    self.assertNotIn("output_tokens", cost.get("per_dim") or {})
                    self.assertTrue(any("output_tokens" in r for r in cost["observed_fact"]["boundary_exclusions"]))

    def test_incomplete_completion_ledger_and_attempt_only_cost_stay_unknown(self):
        import time
        prediction = self.predict_c(cost={"output_tokens": 400})
        record = self.run_g(prediction)
        self.check_g(record)
        self.h.bank.update_cost(record.execution_id, source="provider_usage", output_tokens=400)
        result = self.finish_output(prediction, call_details=[{
            "id": "missing-output", "kind": "model", "started_at": prediction_anchor(self.h, prediction),
            "ended_at": time.time(), "prompt_tokens": 10000}], call_details_complete=["output_tokens"])
        self.assertNotIn("output_tokens", result["evaluations"][0]["cost"].get("per_dim") or {})
        self.assertNotIn("output_tokens", result["evaluations"][0]["cost"]["observed_fact"]["values"])

    def test_legacy_normalized_window_can_reuse_recorded_completion_subfact(self):
        import time
        prediction = self.predict_c(cost={"output_tokens": 400})
        record = self.run_g(prediction)
        self.check_g(record)
        end = time.time()
        self.h.report_action("finish_task", self.task, "ep1", started_at=end, ended_at=end,
            outcome={"host_usage": {"schema": "or-host-usage/1", "host": "test-host",
                "model": "executor-model/v1", "scope": "remaining_to_task_end",
                "measured": ["llm_tokens"], "provenance": {"llm_tokens": "provider_usage"},
                "basis": "provider_total", "prompt_tokens": 10000, "completion_tokens": 400,
                "dimensions": {"llm_tokens": 10400}, "window": {
                    "started_at": prediction_anchor(self.h, prediction), "ended_at": end, "complete": True}}})
        result = self.close_g()
        self.assertEqual(result["evaluations"][0]["cost"]["per_dim"]["output_tokens"]["actual"], 400)

    def test_plan_prices_output_with_own_scale_and_preserves_missing_and_zero(self):
        for index, (cost, scales, expected) in enumerate([
                ({"output_tokens": 400}, {"output_tokens": 1000}, ["output_tokens"]),
                ({"output_tokens": 0}, {"output_tokens": 1000}, ["output_tokens"]),
                ({"output_tokens": 400}, {"llm_tokens": 10400}, [])]):
            with self.subTest(cost=cost, scales=scales):
                self.provider.payload["cost"] = cost
                result = self.h.plan_next(self.task, "scoring" + str(index), candidates=[{
                    "strategy_id": "S04", "solver": "highs", "method": REUSE}], limits={
                        "cost_weights": {"output_tokens": 1}, "cost_scales": scales})
                score = result["selected_score"]
                self.assertEqual(score["scoring_basis"]["cost_dimensions"], expected)
                if cost.get("output_tokens") == 0:
                    self.assertEqual(score["cost_normalized"], 0)
        # An old raw-token prediction is a CONTRACT ERROR, not a U term.
        self.provider.payload["cost"] = {"llm_tokens": 10400}
        result = self.h.plan_next(self.task, "scoring-legacy", candidates=[{
            "strategy_id": "S04", "solver": "highs", "method": REUSE}],
            limits={"cost_weights": {"llm_tokens": 1},
                    "cost_scales": {"llm_tokens": 20000}})
        self.assertEqual(result["status"], "no_valid_predictions")


class TestFutureRetryObservation(CCase):
    def plan_attempt(self, name, *, predicted=0, solver='python', episode='ep1'):
        self.seed_host(episode)
        self.provider.payload['cost']={'retries':predicted}
        # Archived formal-attempt contract regression. Current corrective
        # rounds (including host coverage) are tested in test_build64_implementation.
        from or_harness.world_model.contracts import RETRIES_FORMAL_VERSION
        prediction = planned_prediction(self.h,self.task,{'strategy_id':name,'solver':solver,'method':REUSE},episode)
        prediction.trace.model_info['cost_observation_rule_versions']['retries'] = RETRIES_FORMAL_VERSION
        self.h.strategy_predictions._save(prediction)
        return prediction

    def execute_attempt(self,prediction,*,crash=False,record=True):
        import json
        from pathlib import Path
        work=Path(self.home)/prediction.prediction_id;work.mkdir()
        code=work/'solve.py'
        result={'status':'optimal','objective_value':100,'objective_bound':100,
                'runtime_seconds':0.01,'variables':{'BE':1,'CFC':1},
                'method_performed':prediction.candidate.method,'config':{'time_limit':60}}
        code.write_text("raise RuntimeError('failed solve attempt')\n" if crash else
            'import json,os\nr='+repr(result)+"\nr['method_performed']['action_id']=os.environ['OR_ACTION_ID']\n"
            "r['config']['action_id']=os.environ['OR_ACTION_ID']\njson.dump(r,open('result.json','w'))\n")
        execution=self.h.execute(self.task,None,str(code),str(work),prediction_id=prediction.prediction_id)
        if record:
            self.h.record(execution,host_usage={'schema':'or-host-usage/1','host':'test-host',
                'model':'executor-model/v1','measured':[]})
        return execution

    def test_three_decisions_measure_increments_and_future_attempts_independently(self):
        from or_harness.world_model.contracts import RETRIES_FORMAL_VERSION as RETRIES_FUTURE_VERSION,RETRIES_INCREMENT_VERSION
        predictions=[];records=[]
        for index,solver in enumerate(('python','highs','python')):
            p=self.plan_attempt('ABC'[index],predicted=2.5 if index==0 else 0,solver=solver)
            predictions.append(p);records.append(self.execute_attempt(p,crash=index==0))
            if index==0:
                self.plan_attempt('unused-replan')  # prediction alone is not an attempt
        self.check_g(records[-1]);self.check_g(records[-1])
        self.assertEqual([r.cost.retries for r in records],[0,1,1])
        for r in records:
            self.assertIn('retries',r.cost.measured_dims())
            self.assertEqual(r.execution_features['cost_provenance']['retries']['observation_rule_version'],RETRIES_INCREMENT_VERSION)
        self.assertEqual(self.h.budget_view(self.task['task_id'],'ep1')['consumption']['total_cost']['retries'],2)
        closed=self.close_g()
        observed={e['prediction_id']:e['cost']['per_dim']['retries'] for e in closed['evaluations']}
        self.assertEqual([observed[p.prediction_id]['actual'] for p in predictions],[2,1,0])
        self.assertEqual(observed[predictions[0].prediction_id]['predicted'],2.5)
        self.assertEqual(observed[predictions[1].prediction_id]['abs_error'],1)
        self.assertEqual(observed[predictions[1].prediction_id]['signed_error'],1)
        self.assertEqual(observed[predictions[2].prediction_id]['abs_error'],0)
        self.assertTrue(all(observed[p.prediction_id]['log_ratio'] is None for p in predictions[1:]))
        self.assertTrue(all(observed[p.prediction_id]['observation_rule_version']==RETRIES_FUTURE_VERSION for p in predictions))
        facts={e['prediction_id']:e['cost']['observed_fact']['retries_observed_from'] for e in closed['evaluations']}
        self.assertEqual([facts[p.prediction_id]['baseline_execution_id'] for p in predictions],[r.execution_id for r in records])
        self.assertEqual([len(facts[p.prediction_id]['additional_execution_ids']) for p in predictions],[2,1,0])
        before=self.h.calibration_summary()['conditional_cost']['coverage']['retries']
        self.assertEqual((before['n_actual'],before['n_predicted'],before['n_error']),(0,0,0))
        from or_harness.core.storage import StorageError
        for r in records:
            with self.assertRaises(StorageError):self.h.record(r)
        self.assertTrue(self.close_g()['already_closed'])
        self.assertEqual(self.h.calibration_summary()['conditional_cost']['coverage']['retries'],before)
        self.assertEqual(self.h.budget_view(self.task['task_id'],'ep1')['consumption']['total_cost']['retries'],2)

    def test_executor_exception_is_a_real_retry_and_staged_receipt_is_counted_once(self):
        from unittest.mock import patch
        first=self.plan_attempt('A');a=self.execute_attempt(first)
        failed=self.plan_attempt('B')
        with patch.object(self.h,'_run_executor',side_effect=OSError('executor interrupted')):
            with self.assertRaises(OSError):self.execute_attempt(failed)
        staged=self.h.bank.pending(task_id=self.task['task_id'])
        self.assertEqual(len(staged),1)
        b=staged[0];self.assertEqual(b.cost.retries,1)
        final=self.plan_attempt('C');c=self.execute_attempt(final)
        self.assertEqual([a.cost.retries,b.cost.retries,c.cost.retries],[0,1,1])
        closed=self.close_g()
        costs={e['prediction_id']:e['cost']['per_dim']['retries']['actual'] for e in closed['evaluations']}
        self.assertEqual([costs[p.prediction_id] for p in (first,failed,final)],[2,1,0])
        self.h.record(b)
        from or_harness.core.storage import StorageError
        with self.assertRaises(StorageError):self.h.record(b)
        self.assertEqual(self.h.budget_view(self.task['task_id'],'ep1')['consumption']['total_cost']['retries'],2)

    def test_unclosed_or_incomplete_execution_chain_is_unknown(self):
        from or_harness.world_model.episode_closeout import summarize_real_outcome
        p=self.plan_attempt('A');self.execute_attempt(p)
        prediction=self.h.strategy_predictions.get(p.prediction_id)
        summary=summarize_real_outcome(self.h,prediction)
        self.assertFalse(summary.cost['retries']['complete'])
        self.assertIn('termination',summary.cost['retries']['comparable']['basis'])
        broken=self.h.actions.begin_action('execute_strategy',self.task['task_id'],'ep1')
        self.h.actions.end_action(broken.action_id,status='failed',outcome={'phase':'executor'})
        closed=self.close_g()
        cost=closed['evaluations'][0]['cost']
        self.assertNotIn('retries',cost.get('per_dim') or {})
        self.assertIn('execution chain is incomplete',cost['excluded']['retries'])

    def test_duplicate_execution_link_and_refused_association_do_not_add_attempts(self):
        p=self.plan_attempt('A');record=self.execute_attempt(p)
        alias=self.h.actions.begin_action('execute_strategy',self.task['task_id'],'ep1')
        self.h.actions.end_action(alias.action_id,status='completed',linked_execution_id=record.execution_id,rollup='reference')
        refused=self.h.actions.begin_action('execute_strategy',self.task['task_id'],'ep1')
        self.h.actions.end_action(refused.action_id,status='failed',outcome={'phase':'association'})
        later=self.plan_attempt('B');retry=self.execute_attempt(later)
        self.assertEqual(retry.cost.retries,1)
        closed=self.close_g()
        actual={e['prediction_id']:e['cost']['per_dim']['retries']['actual'] for e in closed['evaluations']}
        self.assertEqual((actual[p.prediction_id],actual[later.prediction_id]),(1,0))

    def test_old_predictions_keep_increment_sum_and_do_not_supply_new_retry_ranges(self):
        from or_harness.world_model.contracts import RETRIES_LEGACY_VERSION,RETRIES_FUTURE_VERSION
        predictions=[]
        for name in 'ABC':
            p=self.plan_attempt(name)
            if name=='B':
                p.trace.model_info.pop('cost_observation_rule_versions')
                self.h.strategy_predictions._save(p)
            predictions.append(p);self.execute_attempt(p)
        closed=self.close_g()
        observed={e['prediction_id']:e['cost']['per_dim']['retries'] for e in closed['evaluations']}
        self.assertEqual([observed[p.prediction_id]['actual'] for p in predictions],[2,2,0])
        self.assertEqual(observed[predictions[1].prediction_id]['observation_rule_version'],RETRIES_LEGACY_VERSION)
        self.plan_attempt('next-task',episode='next')
        evidence=self.provider.requests[-1]['prediction_context']['cost_evidence']
        self.assertFalse(any('retries' in c.get('cost', {}) for c in evidence['cases']))
        # Only the current retry rule may supply the retries target. A legacy
        # retry count is not re-read as the new remaining-task forecast.
        for case in evidence['cases']:
            if 'retries' in case['cost']:
                self.assertIn(case.get('scope'), (None, 'remaining_to_task_end'))
        retries = evidence['global_cost']['dimensions'].get('retries') or {}
        if retries.get('n'):
            self.assertGreaterEqual(retries['n'], 1)

    def test_new_and_legacy_retry_forecasts_do_not_share_a_utility_term(self):
        from or_harness.world_model.planner import PlanLimits,score_strategy_outcome_predictions
        a=self.plan_attempt('A',predicted=.25);b=self.plan_attempt('B',predicted=0)
        limits=PlanLimits(cost_weights={'retries':1},cost_scales={'retries':3})
        same=score_strategy_outcome_predictions([a,b],limits)
        self.assertTrue(all(s.scoring_basis['cost_dimensions']==['retries'] for s in same))
        self.assertGreater(same[0].cost_normalized,0)
        self.assertEqual(same[1].cost_normalized,0)
        b.trace.model_info.pop('cost_observation_rule_versions')
        mixed=score_strategy_outcome_predictions([a,b],limits)
        self.assertTrue(all(s.cost_normalized is None for s in mixed))
        self.assertTrue(all(s.scoring_basis['omitted_cost_dimensions']['retries']['reason']==
                            'different retry observation conventions' for s in mixed))


if __name__ == "__main__":
    unittest.main()
