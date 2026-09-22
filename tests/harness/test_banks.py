"""Conditional statistics and Strategic Bank lifecycle tests."""
import unittest

from helpers import HarnessTestCase

from or_harness.core.schema import CostVector, StrategicEntry
from or_harness.core.storage import StorageError
from or_harness.strategy import (
    ExecutionEvidenceBank,
    ExperienceBank,
    StrategicKnowledgeBank,
)
from or_harness.strategy.stats import ConditionalStats, quality_score
from or_harness.strategy.strategic_bank import (
    DEMOTE_CONSECUTIVE_MISSES,
    PROMOTE_MIN_HIT_RATE,
    PROMOTE_MIN_PREDICTIONS,
    StrategicBank,
)


class TestBankTerminologyAliases(HarnessTestCase):
    """Paper terminology: the aliases point at the same classes — no new
    storage, no new behavior."""

    def test_aliases_are_the_same_classes(self):
        self.assertIs(ExecutionEvidenceBank, ExperienceBank)
        self.assertIs(StrategicKnowledgeBank, StrategicBank)
        self.assertIsInstance(ExecutionEvidenceBank(self.store), ExperienceBank)
        self.assertIsInstance(StrategicKnowledgeBank(self.store), StrategicBank)


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
        cells = self.stats.cross_family("S01")
        self.assertEqual(len(cells), 2)
        self.assertTrue(all(c.n == 1 for c in cells))

    def test_stats_exclude_compacted_lines(self):
        # Facts are forever; compacted ledger lines are bookkeeping summaries
        # and must never re-enter conditional statistics.
        for i in range(3):
            self.bank.append(self.make_record(execution_id=f"ex_{i}",
                                              task_id=f"t{i}", gap=0.1))
        self.bank.append(self.make_record(execution_id="ex_comp", task_id="tC",
                                          gap=0.0, source="compacted"))
        cell = self.stats.cell(self.bank.get("ex_0").group_l1, "S01")
        self.assertEqual(cell.n, 3)
        self.assertNotIn("ex_comp", cell.execution_ids)


class TestRuntimeProvenanceSeparation(HarnessTestCase):
    """A script-reported runtime (the inner solve) and a wall-clock proxy
    (the whole process, imports included) are DIFFERENT quantities under one
    dimension name. Averaging them is a number about nothing."""

    def setUp(self):
        super().setUp()
        self.bank = ExperienceBank(self.store)
        self.stats = ConditionalStats(self.bank)

    def _add(self, name, runtime, provenance):
        rec = self.make_record(
            execution_id=name, task_id=f"t_{name}",
            cost=CostVector(llm_tokens=100.0, solver_runtime_s=runtime,
                            tool_calls=2.0, retries=0.0, latency_s=0.5,
                            measured={"llm_tokens", "tool_calls",
                                      "solver_runtime_s", "retries",
                                      "latency_s"}))
        rec.solver_runtime_provenance = provenance
        self.bank.append(rec)

    def _cell(self):
        return self.stats.cell(self.bank.get("a").group_l1, "S01")

    def test_single_provenance_is_comparable(self):
        self._add("a", 1.0, "reported")
        self._add("b", 3.0, "reported")
        cell = self._cell()
        self.assertEqual(cell.solver_runtime_by_provenance,
                         {"reported": {"n": 2.0, "mean": 2.0}})
        self.assertEqual(cell.measured_cost("solver_runtime_s"), 2.0)

    def test_mixed_provenance_is_not_comparable(self):
        self._add("a", 1.0, "reported")
        self._add("b", 9.0, "wall_proxy")
        cell = self._cell()
        self.assertEqual(set(cell.solver_runtime_by_provenance),
                         {"reported", "wall_proxy"})
        # Per-provenance means are available...
        self.assertEqual(cell.solver_runtime_by_provenance["reported"]["mean"], 1.0)
        self.assertEqual(cell.solver_runtime_by_provenance["wall_proxy"]["mean"], 9.0)
        # ...but the pooled mean is refused.
        self.assertIsNone(cell.measured_cost("solver_runtime_s"))
        self.assertIsNone(cell.comparable_cost("solver_runtime_s"))

    def test_legacy_record_without_provenance_counts_as_wall_proxy(self):
        rec = self.make_record(
            execution_id="a", task_id="ta",
            cost=CostVector(solver_runtime_s=2.0,
                            measured={"solver_runtime_s"}))
        self.assertIsNone(rec.solver_runtime_provenance)
        self.bank.append(rec)
        cell = self._cell()
        self.assertEqual(set(cell.solver_runtime_by_provenance), {"wall_proxy"})

    def test_other_dimensions_are_unaffected_by_provenance_mixing(self):
        self._add("a", 1.0, "reported")
        self._add("b", 9.0, "wall_proxy")
        cell = self._cell()
        self.assertEqual(cell.measured_cost("llm_tokens"), 100.0)


class TestCompleteDims(HarnessTestCase):
    """complete_dims is the bar for publishing a cost claim: measured on
    EVERY supporting record, not just one."""

    def setUp(self):
        super().setUp()
        self.bank = ExperienceBank(self.store)
        self.stats = ConditionalStats(self.bank)

    def test_partial_dimension_is_not_complete(self):
        self.bank.append(self.make_record(
            execution_id="a", task_id="ta",
            cost=CostVector(llm_tokens=100.0, solver_runtime_s=1.0,
                            measured={"llm_tokens", "solver_runtime_s"})))
        self.bank.append(self.make_record(
            execution_id="b", task_id="tb",
            cost=CostVector(solver_runtime_s=2.0,
                            measured={"solver_runtime_s"})))
        cell = self.stats.cell(self.bank.get("a").group_l1, "S01")
        self.assertEqual(cell.complete_dims(), {"solver_runtime_s"})
        self.assertIsNone(cell.comparable_cost("llm_tokens"))
        self.assertEqual(cell.comparable_cost("solver_runtime_s"), 1.5)
        # measured_cost is the weaker read: it answers "was it ever measured".
        self.assertEqual(cell.measured_cost("llm_tokens"), 100.0)

    def test_no_records_means_nothing_is_complete(self):
        self.assertEqual(self.stats.cell("nowhere", "S01").complete_dims(), set())


class TestStrategicBank(HarnessTestCase):
    def setUp(self):
        super().setUp()
        self.sbank = StrategicBank(self.store)

    def make_entry(self, entry_id="se_1", strategy_id="S01", status="candidate",
                   predicates=None, support_n=2, verified=False) -> StrategicEntry:
        return StrategicEntry(
            entry_id=entry_id, strategy_id=strategy_id,
            pattern={"predicates": predicates if predicates is not None
                     else {"family": "routing", "resource_coupling": [0.75, 1.0]}},
            expected_quality_hat=0.9, quality_interval=(0.5, 1.0),
            failure_prob=0.05, status=status, support_n=support_n,
            verification=({"state": "verified", "claim": "c",
                           "conclusion": "check passed"} if verified else {}),
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
        # A family-scoped claim and a family-free one; the latter matches
        # every family (it is cross-family by construction, not by a label).
        self.sbank.add(self.make_entry(entry_id="se_l1"))
        self.sbank.add(self.make_entry(
            entry_id="se_l2", predicates={"resource_coupling": [0.75, 1.0]}))
        profile = self.make_profile(problem_id="q")
        self.assertEqual({e.entry_id for e in self.sbank.matching(profile)},
                         {"se_l1", "se_l2"})
        other_family = self.make_profile(problem_id="q2", family="scheduling")
        self.assertEqual([e.entry_id for e in self.sbank.matching(other_family)],
                         ["se_l2"])

    def test_promotion(self):
        """Forward calibration promotes ONLY verified claims: five checks
        with a good hit rate are calibration evidence, not admission."""
        self.sbank.add(self.make_entry(verified=True))
        transitions = []
        for _ in range(PROMOTE_MIN_PREDICTIONS):
            entry, tr = self.sbank.record_prediction("se_1", hit=True)
            transitions.extend(tr)
        self.assertEqual(entry.status, "validated")
        self.assertIn("promoted:candidate->validated", transitions)
        self.assertGreaterEqual(entry.prediction_track.hit_rate, PROMOTE_MIN_HIT_RATE)

    def test_calibration_alone_never_promotes_unverified(self):
        """The reproduced defect: n>=5 hits used to promote regardless of
        admission verification, so a never-verified candidate could reach
        `validated` and be published."""
        self.sbank.add(self.make_entry())       # unverified
        for _ in range(PROMOTE_MIN_PREDICTIONS):
            entry, transitions = self.sbank.record_prediction("se_1", hit=True)
        self.assertEqual(entry.status, "candidate")
        self.assertNotIn("promoted:candidate->validated", transitions)
        self.assertEqual(entry.prediction_track.n_predictions,
                         PROMOTE_MIN_PREDICTIONS)

    def test_validated_requires_verified_state(self):
        with self.assertRaises(StorageError):
            self.sbank.add(self.make_entry(status="validated"))

    def test_refuted_claim_never_validated(self):
        entry = self.make_entry()
        entry.verification = {"state": "refuted", "claim": "c"}
        entry.status = "validated"
        with self.assertRaises(StorageError):
            self.sbank.add(entry)

    def test_no_early_promotion(self):
        self.sbank.add(self.make_entry())
        for _ in range(PROMOTE_MIN_PREDICTIONS - 1):
            entry, _ = self.sbank.record_prediction("se_1", hit=True)
        self.assertEqual(entry.status, "candidate")

    def test_demotion_after_consecutive_misses(self):
        self.sbank.add(self.make_entry(status="validated", verified=True))
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
