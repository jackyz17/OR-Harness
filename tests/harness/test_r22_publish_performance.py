"""r22: close-episode calibration-publication performance.

Three defects and their fixes, each pinned by a behaviour test with a
COUNTING assertion (not a wall-clock assertion):

A. the budget ledger read the WHOLE action log per execution
   (``actions.query()`` with no task filter, called once per recorded /
   staged / unattributed execution). Fixed by reading the task's own
   actions ONCE and indexing them by linked execution.
B. one publication derived the same historical evaluation TWICE
   (``build_calibration_summary`` and ``build_paired_feedback`` each ran
   ``_live_evaluation`` / ``summarize_real_outcome``). Fixed by deriving the
   window's evaluations ONCE and sharing them.
C. ``_task_end_boundary`` scoped by ``episode_id`` alone, so a second task
   that reused an episode label leaked its ``finish_task`` boundary in.
   Fixed by scoping to ``(task_id, episode_id)``.

The tests COUNT calls / decodes and compare results on a fixed fixture, so
they do not depend on machine speed.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tests.harness.helpers import HarnessTestCase  # noqa: E402

from or_harness.api import ORHarness  # noqa: E402
from or_harness.strategy.embedding_index import LocalHashEmbeddingBackend  # noqa: E402
from or_harness.world_model import actions as actions_mod  # noqa: E402
from or_harness.world_model import budget as budget_mod  # noqa: E402
from or_harness.world_model import episode_closeout as ec_mod  # noqa: E402

from tests.harness.test_episode_closeout import StubProvider, _task  # noqa: E402


class _Counter:
    def __init__(self):
        self.query_calls = 0
        self.query_rows = 0          # ActionRecords deserialised by query()
        self.unfiltered = 0          # query() with no task AND no episode
        self.consumption = 0
        self.summarize = 0
        self.live = 0


class _Patch:
    """Count ActionLog.query / summarize_real_outcome / _live_evaluation."""

    def __init__(self):
        self.c = _Counter()

    def __enter__(self):
        c = self.c
        orig_query = actions_mod.ActionLog.query

        def query(log, **kw):
            rows = orig_query(log, **kw)
            c.query_calls += 1
            c.query_rows += len(rows)
            if kw.get("task_id") is None and kw.get("episode_id") is None:
                c.unfiltered += 1
            return rows
        actions_mod.ActionLog.query = query
        orig_sum = ec_mod.summarize_real_outcome

        def summ(*a, **k):
            c.summarize += 1
            return orig_sum(*a, **k)
        ec_mod.summarize_real_outcome = summ
        orig_live = ec_mod._live_evaluation

        def live(*a, **k):
            c.live += 1
            return orig_live(*a, **k)
        ec_mod._live_evaluation = live
        self._restore = (orig_query, orig_sum, orig_live)
        return self

    def __exit__(self, *exc):
        (actions_mod.ActionLog.query, ec_mod.summarize_real_outcome,
         ec_mod._live_evaluation) = self._restore
        return False


class ReproCase(HarnessTestCase):
    def setUp(self):
        super().setUp()
        self.provider = StubProvider()
        self.h = ORHarness(home=self.home, world_model=self.provider,
                           embedding=LocalHashEmbeddingBackend())
        self.addCleanup(self.h.close)

    def solve(self, task, strategy="S04", episode_id="ep1"):
        from pathlib import Path
        work = Path(self.home) / f"ws_{task['task_id']}_{episode_id}_{os.urandom(3).hex()}"
        work.mkdir(parents=True, exist_ok=True)
        (work / "solve.py").write_text(
            "import json\n"
            "with open('result.json', 'w') as fh:\n"
            "    json.dump({'status': 'optimal', 'objective_value': 100,"
            " 'objective_bound': 100, 'runtime_seconds': 0.01}, fh)\n",
            encoding="utf-8")
        rec = self.h.execute(task, strategy, str(work / "solve.py"),
                             str(work), solver="highs", episode_id=episode_id)
        self.h.record(rec)
        return rec

    def make_episodes(self, n_ep=6, per=2, declared_budget=True):
        for i in range(n_ep):
            tid = f"t{i}"
            task = _task(tid)
            preds = [self.h.predict_strategy_outcome(
                task, {"action_type": "execute_strategy",
                       "strategy_id": "S04"}, "ep1") for _ in range(per)]
            for a in range(per):
                rec = self.solve(task, episode_id="ep1")
                self.h.bind_strategy_outcome(preds[a].prediction_id,
                                             rec.action_id)
            if declared_budget:
                self.h.declare_budget(tid, {"llm_tokens": 1_000_000}, "ep1")
            self.h.close_episode(tid, "ep1")

    def add_noise(self, n):
        for i in range(n):
            a = self.h.actions.begin_action("model", f"noise_{i % 22}",
                                            f"oep{i}")
            self.h.actions.end_action(a.action_id, status="completed",
                                      outcome={"n": i}, cost=None)


class TestBudgetLedgerReadsPerTask(ReproCase):
    """A: the budget no longer rescans the whole action log per execution."""

    def test_no_unfiltered_action_scan_in_consumption(self):
        self.make_episodes()
        with _Patch() as patch:
            view = self.h.budget.view("t0", "ep1",
                                      budget={"llm_tokens": 1_000_000})
        self.assertIn(view["status"], ("ok", "unconfirmed", "exceeded"))
        # The budget read must not scan the WHOLE action log anywhere.
        self.assertEqual(patch.c.unfiltered, 0,
                         "consumption must not query actions without a task")

    def test_work_does_not_grow_with_out_of_window_history(self):
        self.make_episodes()
        with _Patch() as clean:
            budget_mod.BudgetLedger.consumption(
                self.h.budget, "t0", "ep1")
            baseline_rows = clean.c.query_rows
        self.add_noise(1500)
        with _Patch() as noisy:
            budget_mod.BudgetLedger.consumption(
                self.h.budget, "t0", "ep1")
            noisy_rows = noisy.c.query_rows
        # The number of action rows DESERIALISED by the budget read is
        # bounded by the task, not by the unrelated history. A little slack
        # for the (task-filtered) action read itself.
        self.assertLessEqual(noisy_rows, baseline_rows + 5,
                             "budget decode work must not grow with the "
                             "unrelated action history")

    def test_attribution_and_unattributed_still_correct(self):
        """The optimisation keeps the attribution rules AND the
        unattributed report."""
        task = _task("t0")
        # An execution WITH an episode linkage.
        pred = self.h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy", "strategy_id": "S04"},
            "ep1")
        rec = self.solve(task, episode_id="ep1")
        self.h.bind_strategy_outcome(pred.prediction_id, rec.action_id)
        # An execution with NO action linkage (directly recorded).
        rec2 = self.solve(task, episode_id="ep1")  # has a linkage too
        from or_harness.core.schema import (
            CostVector, ExecutionRecord, ProblemProfile)
        direct = ExecutionRecord(
            execution_id="ex_direct", task_id="t0", strategy_id="S9",
            profile_snapshot=ProblemProfile(problem_id="t0",
                                            family="planning"),
            quality={"status": "optimal"}, source="executed",
            cost=CostVector(llm_tokens=5, measured={"llm_tokens"}))
        self.h.record(direct)
        consum = self.h.budget.consumption("t0", "ep1")
        ids = {a["execution_id"] for a in consum["attempts"]}
        self.assertIn(rec.execution_id, ids)
        # The directly recorded execution is unattributed, not charged here.
        self.assertIsNotNone(consum["unattributed"])
        self.assertIn("ex_direct",
                      consum["unattributed"]["execution_ids"])


class TestPublishDerivesOnce(ReproCase):
    """B: one publication derives each evaluation exactly once."""

    def test_live_derivation_runs_once_per_evaluation(self):
        self.make_episodes(n_ep=6, per=2)  # 12 evaluations
        n_evals = len(ec_mod._evaluations_for_window(
            self.h, ec_mod.calibration_window(self.h)))
        self.assertEqual(n_evals, 12)
        with _Patch() as patch:
            ec_mod.republish_calibration(self.h)
        self.assertEqual(patch.c.live, n_evals,
                         "each window evaluation must be live-derived ONCE")
        self.assertEqual(patch.c.summarize, n_evals,
                         "summarize_real_outcome must run once per evaluation")

    def test_independent_builders_still_work(self):
        """Each builder can still be called on its own (no hidden state)."""
        self.make_episodes(n_ep=3, per=1)
        summary = ec_mod.build_calibration_summary(self.h)
        pairs = ec_mod.build_paired_feedback(self.h)
        self.assertGreaterEqual(summary["n_evaluations_total"], 3)
        self.assertIn("pairs", pairs)


class TestTaskEndBoundary(ReproCase):
    """C: the end boundary is scoped to (task_id, episode_id)."""

    def test_shared_episode_label_does_not_cross_tasks(self):
        import time
        a1 = self.h.actions.begin_action("finish_task", "t1", "ep1")
        self.h.actions.end_action(a1.action_id, status="completed")
        time.sleep(0.02)
        a2 = self.h.actions.begin_action("finish_task", "t2", "ep1")
        self.h.actions.end_action(a2.action_id, status="completed")
        t1 = self.h.actions.get(a1.action_id)
        t2 = self.h.actions.get(a2.action_id)
        e1, b1 = ec_mod._task_end_boundary(self.h, "t1", "ep1", anchor=None)
        self.assertEqual(b1, "finish_task")
        self.assertAlmostEqual(e1, t1.ended_at, places=6)
        self.assertNotAlmostEqual(e1, t2.ended_at, places=6)

    def test_none_episode_is_not_a_wildcard(self):
        a = self.h.actions.begin_action("finish_task", "t1", "ep1")
        self.h.actions.end_action(a.action_id, status="completed")
        # t1 has NO None-episode finish: querying (t1, None) must find
        # nothing, NOT fall through to "all episodes".
        e, basis = ec_mod._task_end_boundary(self.h, "t1", None, anchor=None)
        self.assertIsNone(e)
        self.assertEqual(basis, "no_end_boundary")
        # Its OWN None-episode finish IS found when present.
        b = self.h.actions.begin_action("finish_task", "t1", None)
        self.h.actions.end_action(b.action_id, status="completed")
        e2, basis2 = ec_mod._task_end_boundary(self.h, "t1", None, anchor=None)
        self.assertEqual(basis2, "finish_task")
        self.assertIsNotNone(e2)


class TestOptimizationPreservesResults(ReproCase):
    """Fixed facts: the published summary/pairs are unchanged by the
    optimisation (the boundary fix aside)."""

    def test_summary_is_stable_across_a_republish(self):
        self.make_episodes(n_ep=4, per=2)
        first = ec_mod.republish_calibration(self.h)
        second = ec_mod.republish_calibration(self.h)
        # Drop the publish timestamp, then compare the rest verbatim.
        first.pop("published_at", None)
        second.pop("published_at", None)
        self.assertEqual(first, second)

    def test_noise_does_not_change_the_summary(self):
        self.make_episodes(n_ep=4, per=2)
        before = ec_mod.republish_calibration(self.h)
        self.add_noise(800)
        after = ec_mod.republish_calibration(self.h)
        before.pop("published_at", None)
        after.pop("published_at", None)
        self.assertEqual(before, after,
                         "unrelated actions must not change the calibration")


if __name__ == "__main__":
    unittest.main()
