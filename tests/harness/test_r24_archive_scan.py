"""r24: the archive/evidence-window check must not rescan the action log.

The audit found ``_episode_awaits_check`` called ``ActionLog.by_execution``
once per execution, and ``by_execution`` reads the WHOLE action log (no
``WHERE``) and decodes every payload to find one execution's action — so a
close-episode's archive check re-read the entire action history once per
execution. This suite pins the fix (a task-scoped execution->action index):

1. ``_execution_action_index`` resolves the association for old records,
   new records, links established AFTER the action ended, a missing link,
   several actions on one execution (newest wins) and across tasks/episodes;
2. the archive check no longer calls ``by_execution`` (no unfiltered read)
   and the rows it does read do NOT grow with unrelated action history;
3. the archive result is unchanged: candidates, the held-for-late-check set,
   the archived objects, idempotence and late-check reprocessing;
4. ``enforce_evidence_window`` reuses the same index.

Every test uses a temporary home and the hermetic local embedding backend;
the running experiment database is never touched.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tests.harness.helpers import HarnessTestCase  # noqa: E402

import or_harness.world_model.actions as actions_mod  # noqa: E402
from or_harness.api import ORHarness  # noqa: E402
from or_harness.strategy.embedding_index import (  # noqa: E402
    LocalHashEmbeddingBackend,
)
from or_harness.world_model.provider import WorldModelProvider  # noqa: E402

EMBEDDING_ENV_KEYS = ("OR_EMBEDDING_BACKEND", "OR_EMBEDDING_BASE_URL",
                      "OR_EMBEDDING_MODEL", "OR_EMBEDDING_API_KEY")

REQ_TEXT = ("A distribution centre must be loaded before the delivery window "
            "opens; demand 100 units may not be deferred.")

PAYLOAD = {
    "benefit": {"kind": "solution_quality",
                "metric": "normalized_objective_gap", "unit": "1-gap",
                "value": 0.8,
                "baseline": {"kind": "conditional_stats", "value": 0.7}},
    "cost": {"llm_tokens": 1200, "solver_runtime_s": 3.0},
    "risk": {"events": [{"event": "timeout", "probability": 0.2}]},
}


def _task(task_id="t1", **coupling):
    values = {"resource_coupling": 0.3, "temporal_coupling": 0.1,
              "route_complexity": 0.2}
    values.update(coupling)
    return {"task_id": task_id, "family": "routing",
            "description": REQ_TEXT,
            "spec": {"n_vars": 100, "n_constraints": 50, "n_int_vars": 100},
            "annotations": {"coupling": {**values, "semantic_coupling": 0.5}}}


class StubProvider(WorldModelProvider):
    name = "stub-r24"

    def __init__(self, payload=PAYLOAD):
        self.payload = payload

    def predict(self, request, timeout_s=None):
        return {"payload": self.payload,
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
                "error": None, "latency_s": 0.02}


def _make_harness(home):
    saved = {key: os.environ.pop(key, None) for key in EMBEDDING_ENV_KEYS}
    h = ORHarness(home=home, world_model=StubProvider(),
                  embedding=LocalHashEmbeddingBackend())
    return h, saved


class ArchiveScanCase(HarnessTestCase):

    def setUp(self):
        super().setUp()
        saved = {key: os.environ.pop(key, None) for key in EMBEDDING_ENV_KEYS}

        def restore():
            for key, value in saved.items():
                if value is not None:
                    os.environ[key] = value
        self.addCleanup(restore)
        self.h = ORHarness(home=self.home, world_model=StubProvider(),
                           embedding=LocalHashEmbeddingBackend())
        self.addCleanup(self.h.close)

    def _close_one(self, h, index, *, unchecked=False, episodes=1):
        task_id = f"task{index}"
        task = _task(task_id)
        for ep_index in range(episodes):
            episode = f"ep{ep_index}"
            prediction = h.predict_strategy_outcome(
                task, {"action_type": "execute_strategy",
                       "strategy_id": "S04"}, episode)
            record = h.execute(task, "S04",
                               self._script(h, f"{task_id}_{episode}"),
                               self._workdir(h, f"{task_id}_{episode}"),
                               solver="highs", episode_id=episode)
            h.record(record)
            if not unchecked:
                h.check_task_result(record.execution_id,
                                    {"reference_objective": 100.0})
            h.bind_strategy_outcome(prediction.prediction_id, record.action_id)
            h.close_episode(task_id, episode)
        return task_id

    @staticmethod
    def _workdir(h, label):
        from pathlib import Path
        work = Path(h.home) / f"ws_{label}"
        work.mkdir(parents=True, exist_ok=True)
        return str(work)

    @staticmethod
    def _script(h, label):
        from pathlib import Path
        work = Path(h.home) / f"ws_{label}"
        work.mkdir(parents=True, exist_ok=True)
        script = work / "solve.py"
        script.write_text(
            "import json\n"
            "with open('result.json', 'w') as fh:\n"
            "    json.dump({'status': 'optimal', 'objective_value': 100.0,"
            " 'objective_bound': 100.0, 'runtime_seconds': 0.01}, fh)\n",
            encoding="utf-8")
        return str(script)


# ---------------------------------------------------------------------------
# 1. the index resolves the association exactly like by_execution
# ---------------------------------------------------------------------------


class TestExecutionActionIndex(ArchiveScanCase):

    def _link(self):
        from or_harness.world_model.episode_closeout import (
            _execution_action_index,
        )
        return _execution_action_index

    def test_index_matches_by_execution_for_new_records(self):
        idx = self._link()
        action = self.h.actions.report_action(
            "model", "T", "e1", status="completed",
            outcome={"strategy_id": "S"})
        self.h.actions.get(action.action_id)
        # Link a real execution via the normal end path.
        rec = self.h.actions.report_action(
            "verify", "T", "e1", status="completed", outcome={})
        # A verify action links an execution through api end_action params.
        self.h.actions.amend_action_params(rec.action_id,
                                           linked_execution_id=None)
        # Simulate a proper link on the model action through _update.
        stored = self.h.actions.get(action.action_id)
        stored.linked_execution_id = "ex_A"
        self.h.actions._update(stored)
        index = idx(self.h, "T")
        self.assertIsNotNone(index.get("ex_A"))
        direct = self.h.actions.by_execution("ex_A")
        self.assertEqual(index["ex_A"].action_id, direct.action_id)

    def test_missing_link_is_absent(self):
        idx = self._link()
        self.h.actions.report_action("model", "T", "e1", status="completed",
                                     outcome={})
        self.assertEqual(idx(self.h, "T"), {})

    def test_newest_action_wins_for_one_execution(self):
        idx = self._link()
        from or_harness.world_model.actions import ActionRecord
        # Two actions linking the SAME execution; the LATER one (by
        # started_at) must win — matching by_execution's DESC ordering.
        first = action = self.h.actions.report_action(
            "model", "T", "e1", status="completed", outcome={},
            started_at=100.0)
        second = self.h.actions.report_action(
            "verify", "T", "e1", status="completed", outcome={},
            started_at=200.0)
        for rec, ex in ((first, "ex_X"), (second, "ex_X")):
            stored = self.h.actions.get(rec.action_id)
            stored.linked_execution_id = ex
            self.h.actions._update(stored)
        index = idx(self.h, "T")
        self.assertEqual(index["ex_X"].action_id, second.action_id)
        self.assertEqual(index["ex_X"].action_id,
                         self.h.actions.by_execution("ex_X").action_id)

    def test_link_established_after_the_action_ended_is_seen(self):
        idx = self._link()
        rec = self.h.actions.report_action("model", "T", "e1",
                                           status="completed", outcome={})
        # No link yet.
        self.assertEqual(idx(self.h, "T"), {})
        # The association is established LATER (post-end): the index reads
        # the live payload, so it sees it.
        stored = self.h.actions.get(rec.action_id)
        stored.linked_execution_id = "ex_late"
        self.h.actions._update(stored)
        self.assertIn("ex_late", idx(self.h, "T"))

    def test_other_tasks_are_not_in_the_index(self):
        idx = self._link()
        rec = self.h.actions.report_action("model", "OTHER", "e1",
                                           status="completed", outcome={})
        stored = self.h.actions.get(rec.action_id)
        stored.linked_execution_id = "ex_o"
        self.h.actions._update(stored)
        self.assertEqual(idx(self.h, "T"), {})
        self.assertIn("ex_o", idx(self.h, "OTHER"))


# ---------------------------------------------------------------------------
# 2. no unfiltered scan; reads do not grow with unrelated history
# ---------------------------------------------------------------------------


class _Counters:
    def __init__(self):
        self.query_calls = 0
        self.query_rows = 0
        self.unfiltered_queries = 0
        self.by_execution_calls = 0
        self.decodes = 0


class _Patch:
    """Count action reads/decodes while the block runs."""

    def __init__(self):
        self.counters = _Counters()

    def __enter__(self):
        self._orig_query = actions_mod.ActionLog.query
        self._orig_by_execution = actions_mod.ActionLog.by_execution
        self._orig_decode = actions_mod.ActionLog._decode
        counters = self.counters
        orig_query = self._orig_query
        orig_by = self._orig_by_execution
        orig_decode = self._orig_decode

        def query(self, **kw):
            rows = orig_query(self, **kw)
            counters.query_calls += 1
            counters.query_rows += len(rows)
            if kw.get("task_id") is None:
                counters.unfiltered_queries += 1
            return rows

        def by_execution(self, execution_id):
            counters.by_execution_calls += 1
            return orig_by(self, execution_id)

        def decode(a, b=None):
            counters.decodes += 1
            return orig_decode(a) if b is None else orig_decode(b)

        actions_mod.ActionLog.query = query
        actions_mod.ActionLog.by_execution = by_execution
        actions_mod.ActionLog._decode = staticmethod(decode)
        return self.counters

    def __exit__(self, *exc):
        actions_mod.ActionLog.query = self._orig_query
        actions_mod.ActionLog.by_execution = self._orig_by_execution
        actions_mod.ActionLog._decode = staticmethod(self._orig_decode)
        return False


def _build_fixture(h, noise):
    """Identical fixture set; ``noise`` unrelated actions on other tasks."""
    case = ArchiveScanCase.__new__(ArchiveScanCase)  # helpers only
    case.h = h
    for index in range(3):
        # 2 checked (archivable), 1 unchecked (held by the grace period).
        case._close_one(h, index, unchecked=(index == 2))
    for i in range(noise):
        h.actions.report_action("model", f"noise{i}", f"ep{i}",
                                status="completed", outcome={"i": i})


class TestNoUnfilteredScan(ArchiveScanCase):

    def test_archive_check_never_calls_by_execution(self):
        _build_fixture(self.h, noise=0)
        from or_harness.world_model.episode_closeout import (
            CalibrationPolicy, archive_calibration_detail,
        )
        with _Patch() as counters:
            archive_calibration_detail(
                self.h, policy=CalibrationPolicy(
                    window=0, late_check_grace_days=30.0))
        self.assertEqual(counters.by_execution_calls, 0,
                         "the archive check must not use by_execution")
        self.assertEqual(counters.unfiltered_queries, 0,
                         "every action read must be task-scoped")

    def test_reads_do_not_grow_with_unrelated_history(self):
        # Two IDENTICAL fixtures, differing only in unrelated action count.
        from or_harness.world_model.episode_closeout import (
            CalibrationPolicy, archive_calibration_detail,
        )
        small_home = os.path.join(self.home, "small")
        big_home = os.path.join(self.home, "big")
        os.makedirs(small_home, exist_ok=True)
        os.makedirs(big_home, exist_ok=True)
        small, saved_s = _make_harness(small_home)
        big, saved_b = _make_harness(big_home)
        for key, value in {**saved_s, **saved_b}.items():
            if value is not None:
                os.environ[key] = value
        self.addCleanup(small.close)
        self.addCleanup(big.close)

        _build_fixture(small, noise=0)
        _build_fixture(big, noise=600)

        policy = CalibrationPolicy(window=0, late_check_grace_days=30.0)
        with _Patch() as c_small:
            r_small = archive_calibration_detail(small, policy=policy)
        with _Patch() as c_big:
            r_big = archive_calibration_detail(big, policy=policy)

        # 600 unrelated actions must NOT add to what the archive check reads.
        self.assertEqual(
            c_small.query_rows, c_big.query_rows,
            "the rows the archive check reads must be independent of "
            "unrelated action history")
        self.assertEqual(c_small.decodes, c_big.decodes,
                         "the payload decodes must be independent of "
                         "unrelated action history")
        # And the result is unchanged.
        self.assertEqual(r_small["candidate_episodes"],
                         r_big["candidate_episodes"])
        self.assertEqual(
            {x["task_id"] for x in r_small["held_for_late_check"]},
            {x["task_id"] for x in r_big["held_for_late_check"]})
        self.assertEqual(r_small["removed"], r_big["removed"])


# ---------------------------------------------------------------------------
# 3. archive semantics are unchanged
# ---------------------------------------------------------------------------


class TestArchiveSemanticsUnchanged(ArchiveScanCase):

    def policy(self):
        from or_harness.world_model.episode_closeout import CalibrationPolicy
        return CalibrationPolicy(window=0, late_check_grace_days=30.0)

    def test_held_set_and_archived_detail(self):
        self._close_one(self.h, 0, unchecked=False)
        self._close_one(self.h, 1, unchecked=False)
        self._close_one(self.h, 2, unchecked=True)
        from or_harness.world_model.episode_closeout import (
            archive_calibration_detail,
        )
        result = archive_calibration_detail(self.h, policy=self.policy())
        held = {x["task_id"] for x in result["held_for_late_check"]}
        self.assertEqual(held, {"task2"},
                         "only the UNCHECKED episode is held")
        # The two checked episodes are archived; the held one is not.
        self.assertEqual(result["removed"]["evaluations"], 2)
        self.assertEqual(result["candidate_episodes"], 3)

    def test_idempotent_and_late_check_reprocessing(self):
        self._close_one(self.h, 0, unchecked=True)
        from or_harness.world_model.episode_closeout import (
            CalibrationPolicy, archive_calibration_detail,
        )
        # A generous grace: the unchecked episode is HELD (nothing archived).
        held = archive_calibration_detail(
            self.h, policy=CalibrationPolicy(
                window=0, late_check_grace_days=30.0))
        self.assertEqual(held["removed"]["evaluations"], 0)
        self.assertTrue(held["held_for_late_check"])
        # Same run again: identical (idempotent).
        again = archive_calibration_detail(
            self.h, policy=CalibrationPolicy(
                window=0, late_check_grace_days=30.0))
        self.assertEqual(again["removed"], held["removed"])
        self.assertEqual(len(again["held_for_late_check"]),
                         len(held["held_for_late_check"]))
        # A late check lands: now the episode no longer awaits a verdict and
        # is archived on the next pass.
        records = self.h.bank.query(task_id="task0")
        self.h.check_task_result(records[0].execution_id,
                                 {"reference_objective": 100.0})
        after = archive_calibration_detail(
            self.h, policy=CalibrationPolicy(
                window=0, late_check_grace_days=30.0))
        self.assertFalse(after["held_for_late_check"])
        self.assertEqual(after["removed"]["evaluations"], 1)

    def test_evidence_window_reuses_the_index(self):
        for index in range(3):
            self._close_one(self.h, index, unchecked=False)
        from or_harness.world_model.episode_closeout import (
            EvidenceWindowPolicy, CalibrationPolicy,
            enforce_evidence_window,
        )
        with _Patch() as counters:
            enforce_evidence_window(
                self.h, policy=EvidenceWindowPolicy(),
                calibration_policy=CalibrationPolicy(
                    window=1, late_check_grace_days=0.0), dry_run=True)
        self.assertEqual(counters.by_execution_calls, 0,
                         "the evidence-window pass must not use by_execution")
        self.assertEqual(counters.unfiltered_queries, 0)


if __name__ == "__main__":
    unittest.main()
