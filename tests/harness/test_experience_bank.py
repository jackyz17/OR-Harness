"""Experience Bank tests: append-only semantics, filters, cross-process
persistence, malformed rejection, and cost backfill."""
import unittest

from helpers import HarnessTestCase

from or_harness.core.schema import CostVector
from or_harness.core.storage import Store, StorageError
from or_harness.strategy.experience_bank import ExperienceBank


class TestExperienceBank(HarnessTestCase):
    def setUp(self):
        super().setUp()
        self.bank = ExperienceBank(self.store)

    def test_append_and_get(self):
        rec = self.make_record(execution_id="ex_a")
        self.assertEqual(self.bank.append(rec), "ex_a")
        self.assertEqual(self.bank.get("ex_a"), rec)
        self.assertEqual(self.bank.count(), 1)

    def test_append_only_rejects_duplicate(self):
        self.bank.append(self.make_record(execution_id="ex_dup"))
        with self.assertRaises(StorageError):
            self.bank.append(self.make_record(execution_id="ex_dup"))

    def test_append_rejects_malformed(self):
        rec = self.make_record(execution_id="ex_bad")
        rec.verification_level = "gold-plated"  # not a legal level
        with self.assertRaises(ValueError):
            self.bank.append(rec)
        self.assertEqual(self.bank.count(), 0)

    def test_query_filters(self):
        p2 = self.make_profile(problem_id="t2", family="scheduling",
                               route_complexity=None, resource_coupling=0.1)
        self.bank.append(self.make_record(execution_id="ex_1", task_id="t1"))
        self.bank.append(self.make_record(execution_id="ex_2", task_id="t2",
                                          strategy_id="S02", profile=p2))
        self.assertEqual([r.execution_id for r in self.bank.query(task_id="t1")], ["ex_1"])
        self.assertEqual([r.execution_id for r in self.bank.query(strategy_id="S02")], ["ex_2"])
        self.assertEqual([r.execution_id for r in self.bank.query(family="scheduling")], ["ex_2"])
        self.assertEqual(len(self.bank.query()), 2)

    def test_query_by_group(self):
        """One evidence set = one (family, structural cell, strategy) triple:
        a query for a cell returns that cell's facts, and the key is derived
        from each record's own profile rather than trusted from the index
        column."""
        rec = self.make_record(execution_id="ex_g")          # rc 0.9
        self.bank.append(rec)
        same_cell = self.make_record(execution_id="ex_g2", task_id="t2",
                                     profile=self.make_profile(
                                         problem_id="t2", resource_coupling=0.80))
        self.bank.append(same_cell)
        other_cell = self.make_record(execution_id="ex_g3", task_id="t3",
                                      profile=self.make_profile(
                                          problem_id="t3", resource_coupling=0.1))
        self.bank.append(other_cell)
        other_family = self.make_record(
            execution_id="ex_g4", task_id="t4",
            profile=self.make_profile(problem_id="t4", family="scheduling"))
        self.bank.append(other_family)
        rows = self.bank.query(group_l1=rec.group_l1)
        self.assertEqual({r.execution_id for r in rows}, {"ex_g", "ex_g2"})
        # The family alone is a wider view than one cell.
        self.assertEqual(len(self.bank.query(family="routing")), 3)

    def test_query_by_group_reads_legacy_index_formats(self):
        """group_l1's FORMAT has changed over time; membership is decided by
        the family column plus each record's derived key, so facts written
        under any historical index format stay visible."""
        conn = self.store.conn
        rec = self.make_record(execution_id="ex_legacy")
        self.bank.append(rec)
        conn.execute("UPDATE executions SET group_l1=? WHERE execution_id=?",
                     ("family=routing|sc[0.75,1.00]|rc[0.75,1.00]|tc[0.00,0.25]|"
                      "rx[0.75,1.00]", "ex_legacy"))
        conn.commit()
        rows = self.bank.query(group_l1=rec.group_l1)
        self.assertEqual([r.execution_id for r in rows], ["ex_legacy"])
        self.assertEqual(len(self.bank.query(family="routing")), 1)

    def test_cross_process_persistence(self):
        self.bank.append(self.make_record(execution_id="ex_p"))
        # Simulate a new process: fresh Store + bank on the same home.
        store2 = Store(self.home)
        try:
            bank2 = ExperienceBank(store2)
            got = bank2.get("ex_p")
            self.assertIsNotNone(got)
            self.assertEqual(got.task_id, "t1")
        finally:
            store2.close()

    def test_cost_backfill(self):
        rec = self.make_record(execution_id="ex_c", cost=CostVector())
        self.bank.append(rec)
        updated = self.bank.update_cost("ex_c", llm_tokens=4321.0, tool_calls=7.0)
        self.assertEqual(updated.cost.llm_tokens, 4321.0)
        self.assertEqual(self.bank.get("ex_c").cost.llm_tokens, 4321.0)
        # original dimensions untouched
        self.assertEqual(self.bank.get("ex_c").cost.solver_runtime_s, 0.0)

    def test_cost_backfill_rejects_unknown(self):
        self.bank.append(self.make_record(execution_id="ex_c2"))
        with self.assertRaises(StorageError):
            self.bank.update_cost("ex_c2", gpu_hours=3.0)
        with self.assertRaises(StorageError):
            self.bank.update_cost("ex_missing", llm_tokens=1.0)

    def test_append_only_rejects_duplicates(self):
        """The bank is append-only: re-appending an execution id is refused
        (no channel rewrites history)."""
        self.bank.append(self.make_record(execution_id="ex_r1"))
        with self.assertRaises(StorageError):
            self.bank.append(self.make_record(execution_id="ex_r1"))
        self.assertEqual(self.bank.count(), 1)


if __name__ == "__main__":
    unittest.main()
