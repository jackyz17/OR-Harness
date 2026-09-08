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
        rec = self.make_record(execution_id="ex_g")
        self.bank.append(rec)
        same_group = self.make_record(execution_id="ex_g2", task_id="t2")
        self.bank.append(same_group)
        other = self.make_record(execution_id="ex_g3", task_id="t3",
                                 profile=self.make_profile(problem_id="t3",
                                                           resource_coupling=0.1))
        self.bank.append(other)
        rows = self.bank.query(group_l1=rec.group_l1)
        self.assertEqual({r.execution_id for r in rows}, {"ex_g", "ex_g2"})

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

    def test_replace_all_round_trip(self):
        self.bank.append(self.make_record(execution_id="ex_r1"))
        self.bank.append(self.make_record(execution_id="ex_r2", task_id="t2"))
        compacted = self.make_record(execution_id="ex_r2", task_id="t2",
                                     source="compacted")
        self.bank.replace_all([self.bank.get("ex_r1"), compacted])
        self.assertEqual(self.bank.get("ex_r2").source, "compacted")
        self.assertEqual(self.bank.count(), 2)


if __name__ == "__main__":
    unittest.main()
