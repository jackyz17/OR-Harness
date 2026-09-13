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
from or_harness.strategy.triggers import check_triggers


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


class TestDryRunWritesNothing(_Case):
    """All rehearsal paths must be read-only — including the ones that used to
    write: a force-lift of a cold-archive veto removed the card even when the
    call was a --dry-run."""

    def _retired_entry(self):
        self.add("ex_a", "ta")
        self.add("ex_b", "tb")
        created = self.h.induce(strategy_id="S01")["results"][0]["created"]
        entry = self.h.sbank.get(created)
        entry.status = "suspect"
        self.h.sbank.update(entry)
        self.h.sbank.retire(created, reason="bad generalization")
        return created

    def test_force_dry_run_keeps_the_archive_card(self):
        self._retired_entry()
        self.assertEqual(len(self.h.sbank.cold_archive()), 1)
        before = self.snapshot()
        out = self.h.induce(strategy_id="S01", dry_run=True, force=True)
        self.assertEqual(len(self.h.sbank.cold_archive()), 1,
                         "a rehearsal must not lift the veto")
        self.assertEqual(self.snapshot(), before)
        self.assertIn("would_create", out["results"][0])

    def test_force_dry_run_reports_the_intended_lift(self):
        self._retired_entry()
        out = self.h.induce(strategy_id="S01", dry_run=True, force=True)
        self.assertNotIn("vetoed", out["results"][0])
        self.assertIn("would_create", out["results"][0])

    def test_plain_dry_run_writes_nothing(self):
        self.add("ex_a", "ta")
        self.add("ex_b", "tb")
        before = self.snapshot()
        self.h.induce(strategy_id="S01", dry_run=True)
        self.assertEqual(self.snapshot(), before)

    def test_rebuild_dry_run_writes_nothing(self):
        self.add("ex_a", "ta")
        self.add("ex_b", "tb")
        self.h.induce(strategy_id="S01")
        before = self.snapshot()
        out = self.h.induce(rebuild=True, dry_run=True)
        self.assertIn("would_rebuild", out)
        self.assertEqual(self.snapshot(), before)

    def test_rebuild_force_dry_run_writes_nothing(self):
        self._retired_entry()
        before = self.snapshot()
        self.h.induct_rebuild_probe = self.h.induce(rebuild=True, dry_run=True,
                                                    force=True)
        self.assertIn("would_rebuild", self.h.induct_rebuild_probe)
        self.assertEqual(self.snapshot(), before)


class TestEvidenceScope(_Case):
    """One membership rule for every count: executed + attempt-scope + the
    target's structural cell. A task-scope total used to stand in for an
    independent attempt observation AND stretch the claim's range."""

    def test_task_scope_cannot_be_an_independent_task(self):
        """Two attempts of one task plus a task-scope row for a SECOND task
        used to satisfy the >=2-task gate (the task-scope row was counted as
        an independent task while being excluded from the statistics)."""
        self.add("ex_p1", "same", rc=0.20)
        self.add("ex_p2", "same", rc=0.20)
        self.add("ex_ts", "other", rc=0.20, scope="task")
        result = self.h.induce(strategy_id="S01")["results"][0]
        self.assertIsNone(result.get("created"))
        self.assertIn("needs independent evidence", result["skipped"])
        self.assertEqual(result["verification"]["tasks"], ["same"])

    def test_task_scope_never_enters_the_range(self):
        self.add("ex_p1", "ta", rc=0.20)
        self.add("ex_p2", "tb", rc=0.22)
        self.add("ex_ts", "tc", rc=0.95, scope="task")
        created = self.h.induce(strategy_id="S01")["results"][0]["created"]
        entry = self.h.sbank.get(created)
        self.assertEqual(entry.predicates["resource_coupling"], [0.0, 0.25])
        self.assertEqual(entry.support_n, 2)

    def test_task_scope_never_counts_toward_evidence(self):
        self.add("ex_p1", "ta")
        self.add("ex_p2", "tb")
        self.add("ex_ts", "tc", scope="task")
        profile = self.make_profile(problem_id="ta")
        self.assertEqual(len(self.h.stats.evidence(profile, "S01")), 2)


class TestDormantDedup(_Case):
    """A dormant claim used to be invisible to the dedup pass, so induction
    created a NEW entry and then revise woke the OLD one: two entries for one
    knowledge object."""

    def _dormant_entry(self):
        self.add("ex_a", "ta")
        self.add("ex_b", "tb")
        created = self.h.induce(strategy_id="S01")["results"][0]["created"]
        entry = self.h.sbank.get(created)
        entry.status = "dormant"
        self.h.sbank.update(entry)
        return created

    def test_new_evidence_reuses_the_dormant_entry_id(self):
        created = self._dormant_entry()
        self.add("ex_c", "tc")
        result = self.h.induce(strategy_id="S01")["results"][0]
        self.assertIsNone(result.get("created"))
        self.assertEqual(result.get("updated"), created)
        self.assertEqual(self.h.sbank.count(), 1)

    def test_dormant_entry_is_not_woken_by_a_no_op_refresh(self):
        """Re-inducing identical evidence must not flip the status: waking is
        the offline decision, made when the evidence actually changed."""
        created = self._dormant_entry()
        self.h.induce(strategy_id="S01")
        self.assertEqual(self.h.sbank.get(created).status, "dormant")

    def test_full_call_chain_leaves_one_entry(self):
        created = self._dormant_entry()
        self.add("ex_c", "tc")
        self.h.induce(strategy_id="S01")
        self.assertEqual([e.entry_id for e in self.h.sbank.list()], [created])


class TestTriggers(_Case):
    def test_c1_cost_contrast_not_swallowed_by_quality_dedup(self):
        """The reproduced defect: when both sides' QUALITY was already encoded
        by entries, the pair was skipped wholesale and a 5x token difference
        was never reported."""
        from or_harness.core.schema import StrategicEntry
        for sid in ("S01", "S02"):
            self.h.sbank.add(StrategicEntry(
                entry_id=f"se_{sid}", strategy_id=sid,
                pattern={"predicates": {"family": "routing"}},
                expected_quality_hat=0.90, quality_interval=(0.5, 1.0),
                support_n=9))
        for i in range(2):
            self.add(f"ex_S01_{i}", f"S01t{i}", strategy_id="S01", gap=0.1,
                     tokens=500)
            self.add(f"ex_S02_{i}", f"S02t{i}", strategy_id="S02", gap=0.1,
                     tokens=100)
        last = self.h.bank.get("ex_S02_1")
        expected = {e.strategy_id: {"quality": e.expected_quality_hat}
                    for e in self.h.sbank.matching(last.profile_snapshot)}
        hints = [h for h in check_triggers(last, self.h.stats, self.h.catalog,
                                           expected)
                 if h.criterion == "C1"]
        self.assertTrue(hints, "a 5x cost gap must still be reported")
        self.assertEqual(hints[0].evidence["kind"], "cost")

    def test_c6_requires_every_record_feasible(self):
        """The reproduced defect: an infeasible execution carries no failure
        record, so an empty ``failures`` list passed for success."""
        for i in range(2):
            self.add(f"ex_ok{i}", f"ok{i}", gap=0.0)
        for i in range(2):
            self.add(f"ex_inf{i}", f"inf{i}", gap=0.0, feasible=False,
                     status="infeasible")
        last = self.h.bank.get("ex_inf1")
        hints = [h for h in check_triggers(last, self.h.stats, self.h.catalog)
                 if h.criterion == "C6"]
        self.assertFalse(hints)

    def test_c6_still_fires_on_all_feasible(self):
        last = None
        for i in range(4):
            last = self.add(f"ex_ok{i}", f"ok{i}", gap=0.0)
        hints = [h for h in check_triggers(last, self.h.stats, self.h.catalog)
                 if h.criterion == "C6"]
        self.assertTrue(hints)
        self.assertEqual(hints[0].evidence["n_feasible"], 4)

    def test_triggers_read_the_cell_not_the_family(self):
        """C1/C2/C3/C6 must not mix evidence from another structural cell of
        the same family."""
        for i in range(2):
            self.add(f"ex_far{i}", f"far{i}", strategy_id="S07", rc=0.10,
                     gap=0.95)
        record = self.add("ex_near", "near", strategy_id="S01", rc=0.90,
                          gap=0.05)
        hints = check_triggers(record, self.h.stats, self.h.catalog)
        # C1 needs two eligible strategies IN THE CELL; S07's evidence is in
        # another cell, so no contrast may be claimed.
        self.assertNotIn("C1", {h.criterion for h in hints})
        self.assertNotIn("S07", [s for h in hints for s in h.strategy_ids])


class TestPrecision(_Case):
    def test_reloaded_entry_still_matches_its_supporting_task(self):
        """Precision must survive the persistence round trip, not just live
        in memory (``to_dict`` feeds the payload)."""
        third = 1.0 / 3.0
        self.add("ex_t1", "ta", rc=third)
        self.add("ex_t2", "tb", rc=third)
        created = self.h.induce(strategy_id="S01")["results"][0]["created"]
        # Reopen the bank from disk: a fresh object, same payload.
        self.h.close()
        h2 = ORHarness(home=self.home)
        try:
            entry = h2.sbank.get(created)
            profile = self.make_profile(problem_id="q", resource_coupling=third)
            self.assertTrue(entry.matches(profile))
            self.assertEqual(len(h2.stats.evidence(profile, "S01")), 2)
            h2.induce(strategy_id="S01")
            self.assertEqual(h2.sbank.count(), 1)
        finally:
            h2.close()

    def test_pattern_hash_is_stable_across_reload(self):
        self.add("ex_t1", "ta")
        self.add("ex_t2", "tb")
        created = self.h.induce(strategy_id="S01")["results"][0]["created"]
        before = self.h.sbank.get(created).predicates
        self.h.close()
        h2 = ORHarness(home=self.home)
        try:
            self.assertEqual(h2.sbank.get(created).predicates, before)
        finally:
            h2.close()


if __name__ == "__main__":
    unittest.main()
