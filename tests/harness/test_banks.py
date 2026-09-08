"""Conditional statistics and Strategic Bank lifecycle tests."""
import unittest

from helpers import HarnessTestCase

from or_harness.core.schema import CostVector, StrategicEntry
from or_harness.core.storage import StorageError
from or_harness.strategy.experience_bank import ExperienceBank
from or_harness.strategy.stats import ConditionalStats, quality_score
from or_harness.strategy.strategic_bank import (
    DEMOTE_CONSECUTIVE_MISSES,
    PROMOTE_MIN_HIT_RATE,
    PROMOTE_MIN_PREDICTIONS,
    StrategicBank,
)


class TestConditionalStats(HarnessTestCase):
    def setUp(self):
        super().setUp()
        self.bank = ExperienceBank(self.store)
        self.stats = ConditionalStats(self.bank)

    def test_empty_group(self):
        cell = self.stats.cell("nowhere", "S01")
        self.assertEqual(cell.n, 0)
        self.assertEqual(cell.mean_quality, 0.0)

    def test_aggregation(self):
        for i, (gap, tokens) in enumerate([(0.0, 100), (0.1, 200), (0.2, 300)]):
            self.bank.append(self.make_record(
                execution_id=f"ex_{i}", task_id=f"t{i}", gap=gap,
                cost=CostVector(llm_tokens=tokens, solver_runtime_s=1.0)))
        cell = self.stats.cell(self.make_profile().family and
                               self.bank.get("ex_0").group_l1, "S01")
        self.assertEqual(cell.n, 3)
        self.assertAlmostEqual(cell.mean_quality, 0.9, places=6)
        self.assertAlmostEqual(cell.mean_cost.llm_tokens, 200.0)
        self.assertEqual(cell.n_feasible, 3)

    def test_infeasible_scores_zero(self):
        rec = self.make_record(feasible=False, status="infeasible")
        self.assertEqual(quality_score(rec), 0.0)

    def test_rebuild_consistency(self):
        for i in range(6):
            self.bank.append(self.make_record(
                execution_id=f"ex_{i}", task_id=f"t{i}",
                strategy_id="S01" if i % 2 == 0 else "S02",
                gap=0.05 * i))
        self.assertTrue(self.stats.rebuild_check())

    def test_cross_family_partition(self):
        p_a = self.make_profile(problem_id="a1", family="routing")
        p_b = self.make_profile(problem_id="b1", family="scheduling")
        self.bank.append(self.make_record(execution_id="ex_a", task_id="a1", profile=p_a))
        self.bank.append(self.make_record(execution_id="ex_b", task_id="b1", profile=p_b))
        cells = self.stats.cross_family(self.make_profile(problem_id="q"), "S01")
        self.assertEqual(len(cells), 2)
        self.assertTrue(all(c.n == 1 for c in cells))


class TestStrategicBank(HarnessTestCase):
    def setUp(self):
        super().setUp()
        self.sbank = StrategicBank(self.store)

    def make_entry(self, entry_id="se_1", strategy_id="S01", status="candidate",
                   scope="L1", predicates=None, support_n=2) -> StrategicEntry:
        return StrategicEntry(
            entry_id=entry_id, strategy_id=strategy_id,
            pattern={"scope_level": scope,
                     "predicates": predicates if predicates is not None
                     else {"family": "routing", "resource_coupling": [0.75, 1.0]}},
            expected_quality_hat=0.9, quality_interval=(0.5, 1.0),
            failure_prob=0.05, status=status, support_n=support_n,
            provenance=["ex_1", "ex_2"])

    def test_crud(self):
        self.sbank.add(self.make_entry())
        got = self.sbank.get("se_1")
        self.assertEqual(got.strategy_id, "S01")
        got.failure_prob = 0.2
        self.sbank.update(got)
        self.assertEqual(self.sbank.get("se_1").failure_prob, 0.2)
        self.assertEqual(self.sbank.count(), 1)

    def test_update_unknown_raises(self):
        with self.assertRaises(StorageError):
            self.sbank.update(self.make_entry())

    def test_bad_interval_rejected(self):
        entry = self.make_entry()
        entry.quality_interval = (0.9, 0.1)
        with self.assertRaises(StorageError):
            self.sbank.add(entry)

    def test_matching_respects_scope(self):
        self.sbank.add(self.make_entry(entry_id="se_l1", scope="L1"))
        self.sbank.add(self.make_entry(
            entry_id="se_l2", scope="L2",
            predicates={"resource_coupling": [0.75, 1.0]}))
        profile = self.make_profile(problem_id="q")
        self.assertEqual({e.entry_id for e in self.sbank.matching(profile)},
                         {"se_l1", "se_l2"})
        other_family = self.make_profile(problem_id="q2", family="scheduling")
        self.assertEqual([e.entry_id for e in self.sbank.matching(other_family)],
                         ["se_l2"])

    def test_promotion(self):
        self.sbank.add(self.make_entry())
        transitions = []
        for _ in range(PROMOTE_MIN_PREDICTIONS):
            entry, tr = self.sbank.record_prediction("se_1", hit=True)
            transitions.extend(tr)
        self.assertEqual(entry.status, "validated")
        self.assertIn("promoted:candidate->validated", transitions)
        self.assertGreaterEqual(entry.prediction_track.hit_rate, PROMOTE_MIN_HIT_RATE)

    def test_no_early_promotion(self):
        self.sbank.add(self.make_entry())
        for _ in range(PROMOTE_MIN_PREDICTIONS - 1):
            entry, _ = self.sbank.record_prediction("se_1", hit=True)
        self.assertEqual(entry.status, "candidate")

    def test_demotion_after_consecutive_misses(self):
        self.sbank.add(self.make_entry(status="validated"))
        for i in range(DEMOTE_CONSECUTIVE_MISSES):
            entry, tr = self.sbank.record_prediction("se_1", hit=False,
                                                     calibration_err=0.4)
        self.assertEqual(entry.status, "suspect")
        self.assertTrue(any(t.startswith("demoted") for t in tr))

    def test_dormant_and_wakeup(self):
        entry = self.make_entry()
        entry.created_at = 1.0  # long before the task window below
        self.sbank.add(entry)
        # 10 tasks after the entry's creation -> dormant.
        bank = ExperienceBank(self.store)
        for i in range(10):
            bank.append(self.make_record(execution_id=f"ex_t{i}",
                                         task_id=f"task_{i}",
                                         created_at=1000.0 + i))
        affected = self.sbank.age()
        self.assertEqual(affected, ["se_1"])
        self.assertEqual(self.sbank.get("se_1").status, "dormant")
        # A prediction event wakes it up.
        entry, tr = self.sbank.record_prediction("se_1", hit=True)
        self.assertEqual(entry.status, "candidate")
        self.assertIn("awakened:dormant->candidate", tr)

    def test_retire_moves_to_cold_archive(self):
        self.sbank.add(self.make_entry(status="suspect"))
        card = self.sbank.retire("se_1", reason="persistent misses")
        self.assertIsNone(self.sbank.get("se_1"))
        self.assertEqual(card.strategy_id, "S01")
        self.assertEqual(len(self.sbank.cold_archive()), 1)

    def test_archive_veto_and_force_revival(self):
        self.sbank.add(self.make_entry(status="suspect"))
        card = self.sbank.retire("se_1", reason="bad generalization")
        veto = self.sbank.archive_vetoes("S01", card.predicates)
        self.assertIsNotNone(veto)
        # Revival without force is refused.
        with self.assertRaises(StorageError):
            self.sbank.revive(card.pattern_hash)
        self.sbank.revive(card.pattern_hash, force=True)
        self.assertIsNone(self.sbank.archive_vetoes("S01", card.predicates))

    def test_consulted_marks(self):
        self.sbank.add(self.make_entry())
        self.sbank.mark_consulted(["se_1"], at=1234.0)
        self.assertEqual(self.sbank.get("se_1").last_consulted_at, 1234.0)


if __name__ == "__main__":
    unittest.main()
