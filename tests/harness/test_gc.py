"""GC tests: honest compaction deferral and purge planning.

Lossy evidence compaction is DEFERRED until the summary consumption contract
exists (statistics and induction ignore source="compacted" rows). Core
regressions:
  - ``gc compact`` never deletes or replaces raw execution facts;
  - conditional statistics and induction inputs are unchanged after GC
    (the 90-success/10-failure bias scenario);
  - no referential integrity between the Banks — knowledge admission never
    depends on evidence survival;
  - ``purge`` still lists retirement candidates without retiring them.
"""
import unittest

from helpers import HarnessTestCase

from or_harness.core.schema import FailureRecord, StrategicEntry
from or_harness.strategy.experience_bank import ExperienceBank
from or_harness.strategy.gc import COMPACTION_DEFERRED, GarbageCollector
from or_harness.strategy.stats import ConditionalStats
from or_harness.strategy.strategic_bank import StrategicBank


class GcCase(HarnessTestCase):
    def setUp(self):
        super().setUp()
        self.bank = ExperienceBank(self.store)
        self.sbank = StrategicBank(self.store)
        self.stats = ConditionalStats(self.bank)
        self.gc = GarbageCollector(self.bank, self.sbank, self.stats)

    def seed_cell(self, n_success=9, n_failure=1, strategy_id="S01"):
        """One (group, strategy) cell mixing successes and failures."""
        ids = []
        for i in range(n_success):
            rec = self.make_record(execution_id=f"ex_ok_{i}",
                                   task_id=f"tok{i}", strategy_id=strategy_id)
            self.bank.append(rec)
            ids.append(rec.execution_id)
        for i in range(n_failure):
            rec = self.make_record(execution_id=f"ex_fail_{i}",
                                   task_id=f"tfail{i}", strategy_id=strategy_id,
                                   feasible=False, status="error")
            rec.failures = [FailureRecord(attempt=1, error="boom")]
            self.bank.append(rec)
            ids.append(rec.execution_id)
        return ids


class TestCompactionDeferred(GcCase):
    def test_compact_defers_and_touches_nothing(self):
        ids = self.seed_cell()
        entry = StrategicEntry(
            entry_id="se_cover", strategy_id="S01",
            pattern={"scope_level": "L1", "predicates": {"family": "routing"}},
            expected_quality_hat=0.9, quality_interval=(0.5, 1.0),
            provenance=list(ids), support_n=len(ids))
        self.sbank.add(entry)
        before = {r.execution_id for r in self.bank.all()}
        result = self.gc.run("compact")
        self.assertEqual(result["deferred"], COMPACTION_DEFERRED)
        self.assertEqual(result["compacted_cells"], 0)
        after = {r.execution_id for r in self.bank.all()}
        self.assertEqual(before, after)  # nothing deleted or replaced
        kept = self.sbank.get("se_cover")
        self.assertIsNotNone(kept)
        self.assertEqual(kept.status, "candidate")

    def test_plan_compact_is_empty(self):
        self.seed_cell()
        self.assertEqual(self.gc.plan("compact"), [])

    def test_gc_does_not_bias_conditional_statistics(self):
        """The reproduced defect: 90 successes + 10 failures must stay a 10%
        failure rate after GC."""
        self.seed_cell(n_success=90, n_failure=10)
        group = self.bank.get("ex_ok_0").group_l1
        before = self.stats.cell(group, "S01")
        self.gc.run("compact")
        after = self.stats.cell(group, "S01")
        self.assertEqual(before.n, 100)
        self.assertEqual(after.n, 100)
        self.assertAlmostEqual(after.fail_rate, 0.1, places=6)
        self.assertEqual(before.execution_ids, after.execution_ids)

    def test_references_do_not_matter_and_rows_survive(self):
        """Deferral holds regardless of references — nothing is touched, so
        the no-referential-integrity behavior holds trivially."""
        ids = self.seed_cell()
        self.sbank.add(StrategicEntry(
            entry_id="se_cover", strategy_id="S01",
            pattern={"scope_level": "L1", "predicates": {"family": "routing"}},
            provenance=list(ids), support_n=len(ids)))
        self.gc.run("compact")
        self.assertEqual(self.bank.count(), len(ids))


class TestPurgeStillPlans(GcCase):
    def test_purge_lists_retirements_without_applying(self):
        self.sbank.add(StrategicEntry(
            entry_id="se_sus", strategy_id="S01",
            pattern={"scope_level": "L1", "predicates": {}},
            status="suspect"))
        plan = self.gc.plan("purge")
        self.assertEqual([a.kind for a in plan], ["retire"])
        self.assertEqual(plan[0].target, "se_sus")
        result = self.gc.run("purge")
        self.assertEqual(result["planned_retirements"], ["se_sus"])
        self.assertIsNotNone(self.sbank.get("se_sus"))  # not retired


if __name__ == "__main__":
    unittest.main()
