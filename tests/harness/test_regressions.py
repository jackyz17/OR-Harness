"""Regressions for the defects reproduced against 121e29a.

Each test names the defect it pins, so a future reader can tell what the
behaviour is protecting rather than just what it asserts.
"""
import hashlib
import json
import sqlite3
import unittest
from pathlib import Path

from helpers import HarnessTestCase

from or_harness.api import ORHarness
from or_harness.core.schema import COST_DIMENSIONS, CostVector, group_key
from or_harness.strategy.experience_bank import ExperienceBank
from or_harness.strategy.stats import ConditionalStats
from or_harness.strategy.induction import InductionEngine
from or_harness.strategy.strategic_bank import StrategicBank


class _Case(HarnessTestCase):
    def setUp(self):
        super().setUp()
        self.h = ORHarness(home=self.home)
        self.addCleanup(self.h.close)

    def add(self, execution_id, task_id, *, rc=0.9, tc=0.1, rx=0.85, gap=0.02,
            strategy_id="S01", scope="attempt", feasible=True, status="optimal",
            tokens=None, retries=0, family="routing"):
        rec = self.make_record(
            execution_id=execution_id, task_id=task_id, strategy_id=strategy_id,
            gap=gap, feasible=feasible, status=status,
            cost=CostVector(llm_tokens=tokens if tokens is not None else 100,
                            tool_calls=2, solver_runtime_s=1.0, retries=retries,
                            latency_s=1.0, measured=set(COST_DIMENSIONS)),
            profile=self.make_profile(problem_id=task_id, family=family,
                                      resource_coupling=rc, temporal_coupling=tc,
                                      route_complexity=rx))
        rec.measurement_scope = scope
        self.h.bank.append(rec)
        return rec

    def snapshot(self):
        """Fingerprint of every table: rows + payload hashes."""
        conn = sqlite3.connect(str(Path(self.home) / "or_harness.db"))
        try:
            out = {}
            for table in ("executions", "strategic_entries", "cold_archive",
                          "pending_executions"):
                rows = conn.execute(
                    f"SELECT * FROM {table} ORDER BY 1").fetchall()
                out[table] = [
                    [str(c) for c in row[:-1]] + [hashlib.sha256(
                        str(row[-1]).encode()).hexdigest()] for row in rows]
            return out
        finally:
            conn.close()


class TestEvidenceScope(_Case):
    """One membership rule for every count: executed + attempt-scope + the
    target's structural cell. A task-scope total used to stand in for an
    independent attempt observation AND stretch the claim's range.

    The statistical PATH that consumed these counts is gone (knowledge is
    only written from an agent-formed strategy), but the underlying
    ``ConditionalStats.evidence`` membership rule — which recall and the world
    model still read — must keep excluding task-scope totals."""

    def test_task_scope_never_counts_toward_evidence(self):
        self.add("ex_p1", "ta")
        self.add("ex_p2", "tb")
        self.add("ex_ts", "tc", scope="task")
        profile = self.make_profile(problem_id="ta")
        self.assertEqual(len(self.h.stats.evidence(profile, "S01")), 2)


class TestBindOutcomeByExecutionId(_Case):
    """Defect: `bind-outcome` required the action id, so the natural
    predict -> execute -> bind loop failed whenever the caller only had the
    execution id `orx execute` prints (and the two sides' episode ids did
    not line up). Binding by execution id must resolve the action."""

    def _prediction_and_execution(self, pred_episode=None, action_episode=None):
        from or_harness.world_model.prediction import ActionSpec
        task = {"task_id": "t1", "family": "routing", "description": "route"}
        spec = ActionSpec(action_type="execute_strategy", task_id="t1",
                          strategy_id="S01")
        prediction = self.h.predict_outcome(task, spec, pred_episode)
        pre = self.h.snapshot(task, action_episode)
        action = self.h.actions.begin_action(
            "execute_strategy", "t1", action_episode, pre_snapshot=pre,
            params={"strategy_id": "S01", "solver": "highs"})
        execution_id = "ex_under_test"
        self.h.actions.end_action(action.action_id, status="completed",
                                  linked_execution_id=execution_id,
                                  rollup="reference")
        return prediction, action, execution_id

    def test_binding_by_execution_id_resolves_the_action(self):
        prediction, action, execution_id = self._prediction_and_execution(
            pred_episode=None, action_episode="EP_workforce_v2_001")
        bound = self.h.bind_outcome(prediction.prediction_id, execution_id)
        self.assertEqual(bound.bound_action_id, action.action_id)
        # No episode on the prediction side means no episode mismatch is
        # invented — an unknown identity is not a wrong one.
        self.assertIsNone(bound.binding_mismatch)

    def test_binding_by_action_id_still_works(self):
        prediction, action, _ = self._prediction_and_execution()
        bound = self.h.bind_outcome(prediction.prediction_id, action.action_id)
        self.assertEqual(bound.bound_action_id, action.action_id)

    def test_unknown_id_names_both_possibilities(self):
        from or_harness.core.storage import StorageError
        prediction, _, _ = self._prediction_and_execution()
        with self.assertRaises(StorageError) as ctx:
            self.h.bind_outcome(prediction.prediction_id, "ex_nope")
        self.assertIn("execution_id", str(ctx.exception))


class TestTaskTextField(_Case):
    """Defect: a task carrying its problem statement in `text` (the natural
    key) was reported as having NO text, so vector recall silently degraded
    to profile-only."""

    def test_text_field_is_read_as_task_text(self):
        from or_harness.world_model.state import task_text
        self.assertEqual(task_text({"task_id": "t1", "text": "route 3 trucks"}),
                         "route 3 trucks")

    def test_text_field_reaches_the_retrieval_document(self):
        text = self.h.capture_task_text(
            {"task_id": "t1", "text": "load the distribution centre first"})
        self.assertIsNotNone(text)
        self.assertIn("load the distribution centre",
                      self.h.store.get_task_text("t1", text))

    def test_text_and_description_both_contribute(self):
        from or_harness.world_model.state import task_text
        joined = task_text({"task_id": "t1", "text": "short form",
                            "description": "long form"})
        self.assertIn("short form", joined)
        self.assertIn("long form", joined)


class TestEvidenceExclusion(_Case):
    """Defect: a wrong execution fact could not be withdrawn (the bank is
    append-only), so it kept polluting statistics. Exclusion must preserve
    the row, record the reason, and drop it out of every evidence path."""

    def _seed_and_induce(self):
        self.add("ex_bad", "t1", tokens=99999)
        self.add("ex_good", "t2", tokens=100)
        # The facts are enough: exclusion/statistics do not need induction.
        return {"results": []}

    def test_excluded_fact_leaves_the_statistics(self):
        self._seed_and_induce()
        profile = self.h.bank.get("ex_bad").profile_snapshot
        from or_harness.strategy.stats import ConditionalStats
        stats = ConditionalStats(self.h.bank)
        before = stats.aggregate(group_key(profile), "S01",
                                 stats.evidence(profile, "S01"))
        self.h.exclude_execution("ex_bad", reason="wrong answer in toolrepair")
        after = stats.aggregate(group_key(profile), "S01",
                                stats.evidence(profile, "S01"))
        self.assertEqual(before.n, 2)
        self.assertEqual(after.n, 1)
        self.assertNotIn("ex_bad", after.execution_ids)

    def test_exclusion_preserves_the_row_and_records_the_reason(self):
        self._seed_and_induce()
        out = self.h.exclude_execution("ex_bad", reason="bad", superseded_by=None)
        record = self.h.bank.get("ex_bad")
        self.assertIsNotNone(record, "the fact must be preserved for audit")
        self.assertEqual(record.source, "excluded")
        self.assertEqual(record.execution_features["correction"]["reason"], "bad")
        self.assertEqual(out["excluded"], "ex_bad")

    def test_excluded_fact_is_not_evidence(self):
        from or_harness.strategy.stats import ConditionalStats
        self._seed_and_induce()
        self.h.exclude_execution("ex_bad", reason="bad")
        stats = ConditionalStats(self.h.bank)
        profile = self.h.bank.get("ex_bad").profile_snapshot
        ids = {r.execution_id for r in stats.evidence(profile, "S01")}
        self.assertNotIn("ex_bad", ids)

    def test_superseding_link_is_validated(self):
        from or_harness.core.storage import StorageError
        self._seed_and_induce()
        with self.assertRaises(StorageError):
            self.h.exclude_execution("ex_bad", reason="bad",
                                     superseded_by="ex_missing")

    def test_restore_returns_the_fact_to_evidence(self):
        self._seed_and_induce()
        self.h.exclude_execution("ex_bad", reason="bad")
        self.h.restore_execution("ex_bad", reason="the answer was right")
        record = self.h.bank.get("ex_bad")
        self.assertEqual(record.source, "executed")
        self.assertTrue(record.execution_features["correction"]["restored"])

    def test_double_exclusion_is_refused(self):
        from or_harness.core.storage import StorageError
        self._seed_and_induce()
        self.h.exclude_execution("ex_bad", reason="bad")
        with self.assertRaises(StorageError):
            self.h.exclude_execution("ex_bad", reason="bad again")


if __name__ == "__main__":
    unittest.main()
