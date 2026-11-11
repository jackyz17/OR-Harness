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
            "cost": {"llm_tokens": 300.0},
        }
        task = _task("t1")
        prediction = h.predict_strategy_outcome(
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
        first = h.execute(task, "S04", str(goal / "solve.py"), str(goal),
                          solver="highs", episode_id="ep1")
        h.record(first)
        h.bank.update_cost(first.execution_id, source="provider_usage",
                           llm_tokens=100.0)
        second = self.solve(task, strategy="S04", episode_id="ep1",
                            tag="rem2")
        h.bank.update_cost(second.execution_id, source="provider_usage",
                           llm_tokens=200.0)
        h.bind_strategy_outcome(prediction.prediction_id, second.action_id)
        # Attempt receipts alone are not the complete remaining window.
        # The host reports the shared-anchor -> real-termination total.
        import time
        end = time.time()
        h.report_action("finish_task", task, "ep1", started_at=end, ended_at=end,
            host_usage={"schema": "or-host-usage/1", "host": "test-host",
                "model": "executor/v1", "scope": "remaining_to_task_end",
                "tokens": {"prompt_tokens": 200, "completion_tokens": 100},
                "measured": ["llm_tokens"], "window": {
                    "started_at": prediction.trace.created_at, "ended_at": end,
                    "complete": True, "end_event": "finish_task"}})
        result = h.close_episode("t1", "ep1")
        evaluation = result["evaluations"][0]
        entry = evaluation["cost"]["per_dim"].get("llm_tokens")
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
        prediction = h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S04"}, "ep1")
        record = self.solve(task, strategy="S04", episode_id="ep1",
                            tag="lat")
        # Backfill the REAL remaining span (a trusted measurement).
        h.bank.update_cost(record.execution_id, source="agent_observed",
                           remaining_latency_s=600.0)
        h.bind_strategy_outcome(prediction.prediction_id, record.action_id)
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
        prediction = self.predict_c(cost={"tool_calls": 8, "llm_tokens": 300})
        record = self.run_g(prediction, record=False)
        self.check_g(record)
        result, _ = self.finish_c(prediction, counters=True, duplicate_record=record)
        cost = result["evaluations"][0]["cost"]
        self.assertEqual(cost["per_dim"]["tool_calls"]["actual"], 8)
        self.assertEqual(cost["per_dim"]["llm_tokens"]["actual"], 300)
        self.assertEqual(cost["observed_fact"]["work_facts"]["host_rounds"], 2)
        self.assertEqual(cost["observed_fact"]["work_facts"]["execution_attempts"], 1)
        # A remaining-window total is not rewritten as the attempt's spend.
        self.assertNotIn("llm_tokens", self.h.bank.get(record.execution_id).cost.measured_dims())

    def test_timed_calls_count_input_output_and_each_id_once(self):
        import time
        prediction = self.predict_c(cost={"tool_calls": 2, "llm_tokens": 200})
        record = self.run_g(prediction, record=False)
        self.check_g(record)
        start, stop = prediction.trace.created_at, time.time()
        calls = [{"id": "host-1", "kind": "model", "started_at": start,
            "ended_at": stop, "prompt_tokens": 120, "completion_tokens": 80,
            "reasoning_tokens": 60, "cached_tokens": 90},
            {"id": "tool-1", "kind": "tool", "started_at": start, "ended_at": stop},
            {"id": "tool-2", "kind": "tool", "started_at": start, "ended_at": stop}]
        result, _ = self.finish_c(prediction, calls=calls + [dict(calls[1])], duplicate_record=record)
        cost = result["evaluations"][0]["cost"]
        self.assertEqual(cost["per_dim"]["tool_calls"]["actual"], 2)
        self.assertEqual(cost["per_dim"]["llm_tokens"]["actual"], 200)
        self.assertEqual(cost["observed_fact"]["work_facts"]["host_rounds"], 1)

    def test_ambiguous_attempt_totals_and_last_action_are_lower_bounds(self):
        from or_harness.world_model.episode_closeout import summarize_real_outcome
        prediction = self.predict_c(cost={"tool_calls": 8, "llm_tokens": 300, "remaining_latency_s": 1})
        record = self.run_g(prediction)
        self.check_g(record)
        self.h.bank.update_cost(record.execution_id, source="provider_usage", llm_tokens=300, tool_calls=8)
        result = self.close_g()
        cost = result["evaluations"][0]["cost"]
        self.assertFalse(cost.get("per_dim"))
        summary = summarize_real_outcome(self.h, self.h.strategy_predictions.get(prediction.prediction_id))
        self.assertEqual(summary.cost["llm_tokens"]["observation_scope"], "execution_attempts")
        self.assertEqual(summary.cost["llm_tokens"]["total"], 300)
        self.assertFalse(summary.cost["remaining_latency_s"]["complete"])
        self.assertIn("LOWER BOUND", summary.cost["remaining_latency_s"]["boundary_note"])

    def test_failed_and_aborted_terminal_windows_have_spent_cost(self):
        for terminal in ("failed", "aborted"):
            with self.subTest(terminal=terminal):
                prediction = self.predict_c(terminal, cost={"tool_calls": 8, "llm_tokens": 300})
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
        from or_harness.world_model.episode_closeout import build_calibration_summary
        for episode, predicted, actual in (("zero", 0, 0), ("positive", 2, 4)):
            prediction = self.predict_c(episode, cost={"tool_calls": predicted})
            record = self.run_g(prediction)
            self.check_g(record)
            result, _ = self.finish_c(prediction, tools=actual)
            error = result["evaluations"][0]["cost"]["per_dim"]["tool_calls"]
            self.assertEqual(error["abs_error"], actual - predicted)
            self.assertEqual(error["signed_error"], actual - predicted)
        summary = build_calibration_summary(self.h, min_samples=2)
        group = next(g for g in summary["groups"].values() if "tool_calls" in g["mean_cost_abs_error"])
        self.assertEqual(group["mean_cost_abs_error"]["tool_calls"], 1)
        self.assertEqual(group["cost_absolute_evidence"]["tool_calls"]["n_distinct_episodes"], 2)
        self.assertEqual(group["cost_evidence"]["tool_calls"]["n_distinct_episodes"], 1)
        self.assertEqual(group["cost_evidence"]["tool_calls"]["evidence"], "insufficient_evidence")
        self.assertEqual(summary["conditional_cost"]["coverage"]["tool_calls"],
            {"n_actual": 2, "n_predicted": 2, "n_error": 2, "n_log_ratio": 1})

    def test_both_time_dimensions_keep_independent_errors(self):
        prediction = self.predict_c(cost={"solver_runtime_s": .005, "remaining_latency_s": 10})
        record = self.run_g(prediction)
        self.check_g(record)
        result, _ = self.finish_c(prediction)
        per_dim = result["evaluations"][0]["cost"]["per_dim"]
        self.assertEqual(set(per_dim), {"solver_runtime_s", "remaining_latency_s"})
        self.assertEqual(per_dim["solver_runtime_s"]["actual"], .01)

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
        anchor = prediction.trace.created_at
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
                "window": {"started_at": prediction.trace.created_at, "ended_at": end, "complete": True}})
        summary = self.h.calibration_summary()
        self.assertEqual(summary["conditional_cost"]["coverage"]["tool_calls"]["n_actual"], 1)
        self.predict_c("next-late")
        evidence = self.provider.requests[-1]["prediction_context"]["cost_evidence"]
        self.assertEqual(evidence["cases"][0]["values"]["tool_calls"], 8)
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
            "window": {"started_at": prediction.trace.created_at, "ended_at": end, "complete": True}}
        with patch.object(cli, "_harness", return_value=self.h), patch.object(self.h, "close"), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(["amend-cost", record.execution_id, "--usage-file", json.dumps(report)]), 0)
        self.assertNotIn("tool_calls", self.h.bank.get(record.execution_id).cost.measured_dims())
        self.predict_c("next-cli")
        evidence = self.provider.requests[-1]["prediction_context"]["cost_evidence"]
        self.assertEqual(evidence["cases"][0]["values"]["tool_calls"], 8)

    def test_wrong_numeric_window_remains_unknown_with_source_reason(self):
        import time
        prediction = self.predict_c(cost={"tool_calls": 8})
        record = self.run_g(prediction)
        self.check_g(record)
        end = time.time()
        self.h.report_action("finish_task", self.task, "ep1", started_at=end, ended_at=end,
            host_usage={"schema": "or-host-usage/1", "host": "test-host", "model": "executor-model/v1",
                "scope": "remaining_to_task_end", "tool_calls": 8, "measured": ["tool_calls"],
                "window": {"started_at": prediction.trace.created_at - 10, "ended_at": end, "complete": True}})
        result = self.close_g()
        cost = result["evaluations"][0]["cost"]
        self.assertNotIn("tool_calls", cost.get("per_dim") or {})
        self.assertTrue(cost["observed_fact"]["boundary_exclusions"])

    def test_script_wall_proxy_cannot_score_inner_solver_time(self):
        prediction = self.predict_c(cost={"solver_runtime_s": 1, "remaining_latency_s": 1})
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
        later = self.h.execute(self.task, "S04", str(work / "solve.py"), str(work),
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
            self.assertEqual(evidence["current_conditions"]["execution_host"]["model"], "executor-model/v1")
            self.assertEqual(evidence["current_conditions"]["phase"], "existing_model")
            self.assertEqual(len(evidence["reference_ranges"]), 1)
            group = evidence["reference_ranges"][0]
            self.assertEqual(group["dimensions"]["tool_calls"]["p50"], 14)
            self.assertEqual(group["dimensions"]["tool_calls"]["p75"], 17)
            self.assertEqual(group["dimensions"]["tool_calls"]["n"], 2)
            self.assertEqual({c["values"]["tool_calls"] for c in evidence["cases"]}, {8, 20})
            self.assertTrue(all(c["source_refs"] for c in evidence["cases"]))
            self.assertNotIn("conditional_cost", request["prediction_context"]["strategy_outcome_calibration"])
        self.assertIn("Reuse", " ".join(request_a["candidate"]["method"]["steps"]))
        self.assertIn("three subproblems", " ".join(request_b["candidate"]["method"]["steps"]))
        # Different accumulated WM token spend did not fragment the group.
        groups = build_calibration_summary(self.h)["conditional_cost"]["groups"]
        self.assertEqual(len(groups), 1)

    def test_host_family_phase_and_known_limits_mismatch_withhold_priors(self):
        self.make_history("history", REUSE, 8)
        for episode, options in (("other-host", {"model": "other-executor"}),
                                 ("initial", {"existing": False})):
            self.predict_c(episode, **options)
            self.assertEqual(self.provider.requests[-1]["prediction_context"]["cost_evidence"]["reference_ranges"], [])
        self.task["family"] = "scheduling"
        self.predict_c("other-family")
        self.assertEqual(self.provider.requests[-1]["prediction_context"]["cost_evidence"]["reference_ranges"], [])

    def test_open_episode_cost_never_leaks_into_next_request(self):
        prediction = self.predict_c()
        record = self.run_g(prediction)
        self.h.bank.update_cost(record.execution_id, source="agent_observed", tool_calls=99)
        self.predict_c("next")
        evidence = self.provider.requests[-1]["prediction_context"]["cost_evidence"]
        self.assertEqual(evidence["reference_ranges"], [])
        self.assertEqual(evidence["cases"], [])

    def test_actual_solving_limits_stay_separate_and_known_mismatch_is_withheld(self):
        self.make_history("short", REUSE, 8)
        slow = self.predict_c("long", method=REUSE)
        record = self.run_g(slow, execution_config={"time_limit": 180})
        self.check_g(record)
        self.finish_c(slow, tools=20)
        groups = self.h.calibration_summary()["conditional_cost"]["groups"]
        self.assertEqual(len(groups), 2)
        self.seed_host("next-limited")
        self.h.predict_strategy_outcome(self.task, {"action_type": "execute_strategy",
            "strategy_id": "S04", "method": REUSE, "solver": "highs", "config": {"time_limit": 60}},
            "next-limited")
        evidence = self.provider.requests[-1]["prediction_context"]["cost_evidence"]
        self.assertEqual(len(evidence["reference_ranges"]), 1)
        self.assertEqual(evidence["reference_ranges"][0]["dimensions"]["tool_calls"]["p50"], 8)
        self.assertEqual([case["values"]["tool_calls"] for case in evidence["cases"]], [8])

    def test_unreported_phase_stays_unknown(self):
        self.make_history("history", REUSE, 8)
        self.h.report_action("verify", self.task, "unknown-phase",
            host_usage={"schema": "or-host-usage/1", "host": "test-host",
                        "model": "executor-model/v1", "measured": []})
        self.h.predict_strategy_outcome(self.task, {"action_type": "execute_strategy",
            "strategy_id": "S04", "method": REUSE}, "unknown-phase")
        evidence = self.provider.requests[-1]["prediction_context"]["cost_evidence"]
        self.assertEqual(evidence["current_conditions"]["phase"], "unknown")
        self.assertEqual(evidence["reference_ranges"], [])
        self.assertIn("decision phase unavailable", evidence["missing"])


if __name__ == "__main__":
    unittest.main()
