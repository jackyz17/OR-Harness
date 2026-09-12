"""GC compaction tests: retention classes, provenance independence, and the
enriched ledger line.

The corrected memory semantics: Strategic Knowledge is validated at induction
time; after admission it keeps only lightweight origin metadata. Compacting
or deleting old evidence never invalidates an admitted entry, so GC imposes
no referential integrity between the two Banks — provenance references do
NOT exempt evidence from compaction; retention classes do.
"""
import unittest

from helpers import HarnessTestCase

from or_harness.core.schema import FailureRecord, StrategicEntry
from or_harness.strategy.experience_bank import ExperienceBank
from or_harness.strategy.gc import (
    COMPACT_MIN_PER_CELL,
    RECENT_TASK_WINDOW,
    GarbageCollector,
)
from or_harness.strategy.stats import ConditionalStats
from or_harness.strategy.strategic_bank import StrategicBank


class GcCase(HarnessTestCase):
    def setUp(self):
        super().setUp()
        self.bank = ExperienceBank(self.store)
        self.sbank = StrategicBank(self.store)
        self.stats = ConditionalStats(self.bank)
        self.gc = GarbageCollector(self.bank, self.sbank, self.stats)

    def seed_recent_tasks(self, n=RECENT_TASK_WINDOW):
        """n recent executions with distinct timestamps (freshness anchor)."""
        for i in range(n):
            profile = self.make_profile(problem_id=f"trec{i}", family="recent")
            rec = self.make_record(execution_id=f"ex_recent_{i}",
                                   task_id=f"trec{i}", strategy_id="S09",
                                   profile=profile, created_at=2000.0 + i)
            self.bank.append(rec)

    def seed_old_cell(self, n=12, strategy_id="S01", created_at=1000.0,
                      retention_reason=None):
        ids = []
        for i in range(n):
            profile = self.make_profile(problem_id=f"told{i}", family="routing")
            rec = self.make_record(execution_id=f"ex_old_{strategy_id}_{i}",
                                   task_id=f"told{i}", strategy_id=strategy_id,
                                   profile=profile, created_at=created_at)
            rec.retention_reason = retention_reason
            self.bank.append(rec)
            ids.append(rec.execution_id)
        return ids

    def covering_entry(self, provenance=()):
        entry = StrategicEntry(
            entry_id="se_cover", strategy_id="S01",
            pattern={"scope_level": "L1",
                     "predicates": {"family": "routing"}},
            expected_quality_hat=0.9, quality_interval=(0.5, 1.0),
            provenance=list(provenance), support_n=len(provenance))
        self.sbank.add(entry)
        return entry


class TestCompactionEligibility(GcCase):
    def test_provenance_references_do_not_block_compaction(self):
        """Evidence referenced by an entry may be compacted — knowledge
        admission never depends on evidence survival."""
        self.seed_recent_tasks()
        ids = self.seed_old_cell()
        self.covering_entry(provenance=ids)
        plan = [a for a in self.gc.plan("compact") if a.kind == "compact"]
        self.assertEqual(len(plan), 1)
        self.assertEqual(set(plan[0].detail["execution_ids"]), set(ids))

    def test_retention_reason_rows_never_compacted(self):
        self.seed_recent_tasks()
        self.seed_old_cell(n=12, retention_reason="contrast")
        self.covering_entry()
        plan = [a for a in self.gc.plan("compact") if a.kind == "compact"]
        self.assertEqual(plan, [])

    def test_cell_at_k_threshold_stays_raw(self):
        self.seed_recent_tasks()
        self.seed_old_cell(n=COMPACT_MIN_PER_CELL)
        self.covering_entry()
        plan = [a for a in self.gc.plan("compact") if a.kind == "compact"]
        # Exactly K=10 rows: the cell-level gate (len(recs) > K) fails.
        self.assertEqual(plan, [])


class TestCompactionExecution(GcCase):
    def test_compaction_does_not_invalidate_referencing_entry(self):
        """Core regression: compacting referenced evidence leaves the
        admitted Strategic Knowledge entry intact and unchanged."""
        self.seed_recent_tasks()
        ids = self.seed_old_cell()
        entry = self.covering_entry(provenance=ids)
        result = self.gc.run("compact")
        self.assertEqual(result["compacted_cells"], 1)
        kept = self.sbank.get("se_cover")
        self.assertIsNotNone(kept)
        self.assertEqual(kept.status, entry.status)
        self.assertEqual(kept.expected_quality_hat, entry.expected_quality_hat)
        self.assertEqual(kept.support_n, entry.support_n)
        # Raw rows replaced by exactly one ledger line.
        ledgers = [r for r in self.bank.all() if r.source == "compacted"]
        self.assertEqual(len(ledgers), 1)
        remaining_ids = {r.execution_id for r in self.bank.all()}
        self.assertFalse(set(ids) & remaining_ids)

    def test_compacted_ledger_carries_strategic_summaries(self):
        self.seed_recent_tasks()
        for i in range(12):
            profile = self.make_profile(problem_id=f"tled{i}", family="routing")
            rec = self.make_record(execution_id=f"ex_led_{i}",
                                   task_id=f"tled{i}", strategy_id="S01",
                                   profile=profile, created_at=1000.0)
            if i % 3 == 0:
                rec.failures = [FailureRecord(
                    attempt=1, error="timeout",
                    recovery_action="switch solver",
                    error_class="environment")]
            rec.cost.llm_tokens = float(100 + i * 10)
            self.bank.append(rec)
        self.covering_entry()
        result = self.gc.run("compact")
        self.assertEqual(result["compacted_cells"], 1)
        ledger = next(r for r in self.bank.all()
                      if r.source == "compacted")
        self.assertIsNone(ledger.quality["gap"])  # no fabricated gap
        compacted = ledger.quality["compacted"]
        self.assertEqual(compacted["n"], 12)
        self.assertEqual(compacted["failures"], 4)
        self.assertEqual(compacted["failure_classes"]["environment"], 4)
        self.assertEqual(compacted["recovery_actions"]["switch solver"], 4)
        self.assertEqual(compacted["verification_levels"]["basic"], 12)
        # Cost spread survives for future cost learning.
        self.assertLess(compacted["cost_min"]["llm_tokens"],
                        compacted["cost_max"]["llm_tokens"])


if __name__ == "__main__":
    unittest.main()
