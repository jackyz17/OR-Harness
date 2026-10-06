"""Budget stop / interruption semantics (r7 budget round).

Pins the behaviours this round introduced:

1. The budget gate is SCOPE-AWARE: an EPISODE-scoped overrun refuses new
   solving work, but an ATTEMPT-scoped overrun (latency on one attempt) does
   NOT — a single slow attempt must not permanently block retries.
2. A recorded session stop (host cancel) refuses new work even when the
   ledger's numbers do not read "over the limit" (reaching a cap and
   refusing the NEXT call leaves the count AT the limit).
3. ``record_session`` archives the stop in three honest shapes (staged
   attempt / running action with no staged fact / no attempt at all),
   keeps the real cost, invents no solve result, and is idempotent.
4. ``close-episode`` proceeds after a recorded interruption instead of being
   blocked by a still-"running" action.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tests.harness.helpers import HarnessTestCase  # noqa: E402

from or_harness.api import ORHarness  # noqa: E402
from or_harness.core.schema import (  # noqa: E402
    CostVector,
    ExecutionRecord,
    ProblemProfile,
)
from or_harness.core.storage import BudgetExhausted  # noqa: E402


def _profile(problem_id="t1", family="routing"):
    return ProblemProfile(
        problem_id=problem_id, family=family,
        scale_features={"n_vars": 100.0, "n_constraints": 50.0,
                        "n_int_vars": 100.0, "density": 0.01},
        semantic_coupling=0.8, resource_coupling=0.3,
        temporal_coupling=0.2, route_complexity=0.8)


class TestBudgetGateScope(HarnessTestCase):
    def test_episode_overrun_refuses_new_work(self):
        h = ORHarness(home=self.home)
        self.addCleanup(h.close)
        h.declare_budget("T", {"llm_tokens": 1.0})
        h.record(ExecutionRecord(
            execution_id="ex_1", task_id="T", strategy_id="S",
            profile_snapshot=_profile("T"),
            quality={"feasible": True, "objective": 1.0, "gap": 0.0,
                     "status": "optimal"},
            solver={"name": "highs"},
            cost=CostVector(llm_tokens=5000.0, measured={"llm_tokens"})))
        gate = h._budget_gate("T")
        self.assertFalse(gate["allowed"])
        self.assertTrue(gate["view"]["episode_exhausted"])

    def test_attempt_overrun_allows_a_retry(self):
        """A single slow attempt must NOT permanently block retries."""
        h = ORHarness(home=self.home)
        self.addCleanup(h.close)
        h.declare_budget("T", {"latency_s": 0.001})
        h.record(ExecutionRecord(
            execution_id="ex_1", task_id="T", strategy_id="S",
            profile_snapshot=_profile("T"),
            quality={"feasible": True, "objective": 1.0, "gap": 0.0,
                     "status": "optimal"},
            solver={"name": "highs"},
            cost=CostVector(latency_s=5.0, measured={"latency_s"})))
        view = h.budget_view("T")
        self.assertEqual(view["status"], "exceeded")
        self.assertTrue(view["attempt_limited"])
        self.assertFalse(view["episode_exhausted"])
        gate = h._budget_gate("T")
        self.assertTrue(gate["allowed"])

    def test_overrun_detail_names_dimension_and_scope(self):
        h = ORHarness(home=self.home)
        self.addCleanup(h.close)
        h.declare_budget("T", {"llm_tokens": 1.0, "latency_s": 0.001})
        h.record(ExecutionRecord(
            execution_id="ex_1", task_id="T", strategy_id="S",
            profile_snapshot=_profile("T"),
            quality={"feasible": True, "objective": 1.0, "gap": 0.0,
                     "status": "optimal"},
            solver={"name": "highs"},
            cost=CostVector(llm_tokens=5000.0, latency_s=5.0,
                            measured={"llm_tokens", "latency_s"})))
        dims = {(d["dimension"], d["scope"])
                for d in h.budget_view("T")["exceeded_dims"]}
        self.assertEqual(dims, {("llm_tokens", "episode"),
                                ("latency_s", "per_attempt")})


class TestSessionStop(HarnessTestCase):
    def test_stop_reaching_the_cap_still_refuses(self):
        """100 tools reached and the 101st refused leaves the count AT the
        limit; the recorded host reason is what stops the task."""
        h = ORHarness(home=self.home)
        self.addCleanup(h.close)
        h.declare_budget("T", {"tool_calls": 100.0})
        h.record(ExecutionRecord(
            execution_id="ex_1", task_id="T", strategy_id="S",
            profile_snapshot=_profile("T"),
            quality={"feasible": True, "objective": 1.0, "gap": 0.0,
                     "status": "optimal"},
            solver={"name": "highs"},
            cost=CostVector(tool_calls=100.0, measured={"tool_calls"})))
        # The ledger does NOT read "over" (100 is not > 100).
        self.assertNotEqual(h.budget_view("T")["status"],
                            "exceeded")
        # The host records the stop with its own reason.
        h.record_session("T", "e1", reason="tool_calls reached 100",
                         cancelled=True, cost={"tool_calls": 100.0})
        gate = h._budget_gate("T", "e1")
        self.assertFalse(gate["allowed"])
        self.assertIn("tool_calls reached 100", gate["reason"])

    def test_session_only_shape_records_cost_without_fabricating_an_execution(self):
        h = ORHarness(home=self.home)
        self.addCleanup(h.close)
        result = h.record_session(
            "T", "e1", reason="wall clock exceeded", cancelled=True,
            cost={"llm_tokens": 400.0, "tool_calls": 20.0})
        shapes = [x["shape"] for x in result["recorded"]]
        self.assertEqual(shapes, ["session_only_terminal"])
        # No execute_strategy action was invented.
        self.assertFalse([a for a in h.actions.query(task_id="T")
                          if a.action_type == "execute_strategy"])
        finish = [a for a in h.actions.query(task_id="T")
                  if a.action_type == "finish_task"]
        self.assertEqual(len(finish), 1)

    def test_running_action_without_staged_fact_is_interrupted(self):
        h = ORHarness(home=self.home)
        self.addCleanup(h.close)
        task = {"task_id": "T", "family": "routing"}
        pre = h.snapshot(task, "e1")
        h.actions.begin_action("execute_strategy", "T", "e1", pre_snapshot=pre,
                               params={"strategy_id": "S", "solver": "highs"})
        result = h.record_session("T", "e1", reason="killed before staging",
                                  cancelled=True, cost={"llm_tokens": 250.0})
        self.assertEqual([x["shape"] for x in result["recorded"]],
                         ["running_action_interrupted"])
        # No running execute action remains, and the interruption is honest.
        self.assertFalse([a for a in h.actions.query(
            task_id="T", episode_id="e1", status="running")])
        ended = [a for a in h.actions.query(task_id="T")
                 if a.action_type == "execute_strategy"][0]
        self.assertEqual(ended.status, "failed")
        self.assertTrue(ended.outcome.get("interrupted"))
        self.assertEqual(ended.outcome.get("result"), "unobservable")

    def test_staged_attempt_is_annotated_not_rewritten(self):
        h = ORHarness(home=self.home)
        self.addCleanup(h.close)
        h.bank.stage_pending(ExecutionRecord(
            execution_id="ex_staged", task_id="T", strategy_id="S",
            profile_snapshot=_profile("T"),
            quality={"feasible": True, "objective": 7.0, "gap": 0.0,
                     "status": "optimal"},
            solver={"name": "highs"}))
        result = h.record_session("T", "e1", reason="stop after staging",
                                  cancelled=True,
                                  attempted_execution_id="ex_staged")
        self.assertEqual([x["shape"] for x in result["recorded"]],
                         ["staged_attempt_annotated"])
        staged = h.bank.get_pending("ex_staged")
        self.assertIn("session_interrupted", staged.execution_features)
        # The observed result is NOT rewritten by the interruption.
        self.assertEqual(staged.quality.get("objective"), 7.0)

    def test_replay_is_idempotent(self):
        h = ORHarness(home=self.home)
        self.addCleanup(h.close)
        first = h.record_session("T", "e1", reason="cap", cancelled=True,
                                 cost={"tool_calls": 100.0})
        self.assertFalse(first["already_recorded"])
        n_after_first = len(h.actions.query(task_id="T"))
        second = h.record_session("T", "e1", reason="cap", cancelled=True,
                                  cost={"tool_calls": 100.0})
        self.assertTrue(second["already_recorded"])
        self.assertFalse(second["recorded"])
        self.assertEqual(len(h.actions.query(task_id="T")), n_after_first,
                         "an idempotent replay writes no second action")


class TestInterruptedCloseout(HarnessTestCase):
    def test_close_proceeds_after_session_stop(self):
        """A recorded stop ends the running action, so close-episode can
        evaluate instead of returning `pending` forever."""
        h = ORHarness(home=self.home)
        self.addCleanup(h.close)
        task = {"task_id": "T", "family": "routing"}
        pre = h.snapshot(task, "e1")
        h.actions.begin_action("execute_strategy", "T", "e1", pre_snapshot=pre,
                               params={"strategy_id": "S", "solver": "highs"})
        h.record_session("T", "e1", reason="tool_calls reached 100",
                        cancelled=True, cost={"tool_calls": 100.0})
        result = h.close_episode("T", "e1", terminal_state="budget_exhausted")
        self.assertFalse(result.get("already_closed"))
        self.assertEqual(result["closeout"]["terminal_state"],
                         "budget_exhausted")

    def test_gate_blocks_new_work_after_stop(self):
        h = ORHarness(home=self.home)
        self.addCleanup(h.close)
        h.record_session("T", "e1", reason="host stop", cancelled=True)
        with self.assertRaises(BudgetExhausted):
            h.execute({"task_id": "T", "family": "routing"}, "S",
                      "solve.py", ".", solver="highs", episode_id="e1")

    def test_cli_session_and_gate_surface(self):
        """The CLI archives a stop and then reports the stop (not a
        traceback) when a new attempt is refused."""
        import contextlib
        import io
        import json
        from or_harness import cli
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = cli.main(["--home", self.home, "record", "--session",
                             "--task", "T", "--episode", "e1",
                             "--reason", "tool_calls reached 100",
                             "--cancelled"])
        self.assertEqual(code, 0)
        self.assertIn("Session stop recorded",
                      json.loads(buf.getvalue())["summary"])
        # A new attempt via the CLI is refused cleanly.
        buf2 = io.StringIO()
        with contextlib.redirect_stdout(buf2):
            code2 = cli.main(["--home", self.home, "execute", "--task",
                              "{\"task_id\": \"T\", \"family\": \"routing\"}",
                              "--strategy", "S", "--code", "solve.py",
                              "--workspace", self.home, "--solver", "highs",
                              "--episode", "e1"])
        self.assertEqual(code2, 2)
        out = json.loads(buf2.getvalue())
        self.assertIn("task stopped before this attempt started",
                      out["summary"])


if __name__ == "__main__":
    unittest.main()
