"""C1-C6 trigger tests: each criterion fires when it should and, critically,
stays silent when it should (n>=2 gate, prior-consistency, direction checks)."""
import unittest

from helpers import HarnessTestCase

from or_harness.core.schema import CostVector, FailureRecord
from or_harness.strategy.catalog import load_catalog
from or_harness.strategy.experience_bank import ExperienceBank
from or_harness.strategy.stats import ConditionalStats
from or_harness.strategy.triggers import check_triggers


class TriggerCase(HarnessTestCase):
    def setUp(self):
        super().setUp()
        self.bank = ExperienceBank(self.store)
        self.stats = ConditionalStats(self.bank)
        self.catalog = load_catalog()

    def check(self, record):
        return check_triggers(record, self.stats, self.catalog)

    def criteria(self, record):
        return {h.criterion for h in self.check(record)}

    def seed(self, strategy_id, n, gap, task_prefix="s", cost=None,
             cost_measured=None):
        last = None
        for i in range(n):
            last = self.make_record(
                execution_id=f"{task_prefix}_{strategy_id}_{i}",
                task_id=f"{task_prefix}{i}", strategy_id=strategy_id, gap=gap,
                cost=cost or CostVector(llm_tokens=100, solver_runtime_s=1.0),
                cost_measured=cost_measured)
            self.bank.append(last)
        return last


class TestC1Contrast(TriggerCase):
    def test_single_sample_never_triggers(self):
        # Q1 from the design doc: S02 first execution Q=0.98 vs S01's single
        # 0.85 -> no trigger (n=1 on both sides of the divergence check).
        self.seed("S01", 1, 0.15, task_prefix="a")
        new = self.seed("S02", 1, 0.02, task_prefix="b")
        self.assertNotIn("C1", self.criteria(new))

    def test_quality_contrast_fires(self):
        # S01 meanQ 0.60 vs S04 meanQ 0.98 — significant quality contrast.
        self.seed("S01", 2, 0.40, task_prefix="a")   # meanQ 0.60
        new = self.seed("S04", 2, 0.02, task_prefix="b")  # meanQ 0.98
        hints = [h for h in self.check(new) if h.criterion == "C1"]
        self.assertTrue(hints)
        self.assertEqual(set(hints[0].strategy_ids), {"S01", "S04"})

    def test_small_difference_silent(self):
        # Quality difference below threshold — no trigger.
        self.seed("S01", 2, 0.10, task_prefix="a")   # meanQ 0.90
        new = self.seed("S04", 2, 0.15, task_prefix="b")  # meanQ 0.85
        self.assertNotIn("C1", self.criteria(new))

    def test_cost_contrast_fires(self):
        # Quality tied (both ~0.80), S01 100% more expensive in tokens.
        cheap = CostVector(llm_tokens=100, solver_runtime_s=1.0)
        pricey = CostVector(llm_tokens=200, solver_runtime_s=1.0)
        self.seed("S06", 2, 0.20, task_prefix="a", cost=cheap)
        new = self.seed("S01", 2, 0.20, task_prefix="b", cost=pricey)
        hints = [h for h in self.check(new) if h.criterion == "C1"]
        self.assertTrue(hints)
        self.assertEqual(hints[0].evidence["kind"], "cost")


class TestC2ExtremePerformance(TriggerCase):
    def test_high_performance_fires(self):
        # S01 observed meanQ ~0.95 over n=2 — extreme high.
        new = self.seed("S01", 2, 0.05, task_prefix="a")
        self.assertIn("C2", self.criteria(new))

    def test_low_performance_fires(self):
        # S01 observed meanQ ~0.05 over n=2 — extreme low.
        new = self.seed("S01", 2, 0.95, task_prefix="a")
        self.assertIn("C2", self.criteria(new))

    def test_single_sample_silent(self):
        new = self.seed("S01", 1, 0.05, task_prefix="a")
        self.assertNotIn("C2", self.criteria(new))

    def test_moderate_performance_silent(self):
        # meanQ ~0.50 is neither high nor low.
        new = self.seed("S01", 3, 0.50, task_prefix="a")
        self.assertNotIn("C2", self.criteria(new))


class TestC3Drift(TriggerCase):
    def test_trend_fires(self):
        last = None
        for i, gap in enumerate([0.30, 0.20, 0.05]):
            last = self.make_record(execution_id=f"d{i}", task_id=f"dt{i}",
                                    strategy_id="S01", gap=gap)
            self.bank.append(last)
        self.assertIn("C3", self.criteria(last))

    def test_flat_series_silent(self):
        new = self.seed("S01", 4, 0.10, task_prefix="f")
        self.assertNotIn("C3", self.criteria(new))

    def test_two_samples_silent(self):
        for i, gap in enumerate([0.30, 0.05]):
            last = self.make_record(execution_id=f"e{i}", task_id=f"et{i}",
                                    strategy_id="S01", gap=gap)
            self.bank.append(last)
        self.assertNotIn("C3", self.criteria(last))


class TestC4FailureRecovery(TriggerCase):
    def test_recovery_fires(self):
        rec = self.make_record(
            execution_id="f1", task_id="ft1", feasible=True, status="feasible",
            failures=[FailureRecord(attempt=1, error="infeasible model",
                                    recovery_action="fallback:S01")])
        self.bank.append(rec)
        hints = [h for h in self.check(rec) if h.criterion == "C4"]
        self.assertTrue(hints)
        self.assertEqual(hints[0].evidence["execution_id"], "f1")

    def test_failure_without_recovery_silent(self):
        rec = self.make_record(
            execution_id="f2", task_id="ft2", feasible=False, status="error",
            failures=[FailureRecord(attempt=1, error="crash")])
        self.bank.append(rec)
        self.assertNotIn("C4", self.criteria(rec))

    def test_clean_run_silent(self):
        rec = self.make_record(execution_id="f3", task_id="ft3")
        self.bank.append(rec)
        self.assertNotIn("C4", self.criteria(rec))


class TestC5CrossFamily(TriggerCase):
    def test_reproduced_high_performance_fires(self):
        """Two families, same strategy, SAME structure (rc cell), same
        direction — the reproduction C5 exists to report."""
        last = None
        for fam in ("routing", "scheduling"):
            for i in range(2):
                profile = self.make_profile(problem_id=f"{fam}{i}", family=fam,
                                            resource_coupling=0.90)
                last = self.make_record(execution_id=f"c5_{fam}_{i}",
                                        task_id=f"{fam}{i}", strategy_id="S04",
                                        profile=profile, gap=0.05)
                self.bank.append(last)
        hints = [h for h in self.check(last) if h.criterion == "C5"]
        self.assertTrue(hints)
        self.assertEqual(set(hints[0].evidence["families"]),
                         {"routing", "scheduling"})
        self.assertEqual(hints[0].evidence["structure"]["resource_coupling"],
                         "[0.75,1.00]")

    def test_incomparable_structure_is_not_mixed_in(self):
        """The reproduced defect: a third family whose evidence comes from an
        unrelated structure used to be pooled into the same statistic."""
        last = None
        for fam, rc in (("routing", 0.90), ("scheduling", 0.90),
                        ("packing", 0.10)):
            for i in range(2):
                profile = self.make_profile(problem_id=f"{fam}{i}", family=fam,
                                            resource_coupling=rc)
                last = self.make_record(execution_id=f"mix_{fam}_{i}",
                                        task_id=f"{fam}{i}", strategy_id="S04",
                                        profile=profile, gap=0.05)
                self.bank.append(last)
        hits = [h for h in self.check(last) if h.criterion == "C5"]
        # packing is structurally different: not reported as reproduction.
        for h in hits:
            self.assertNotIn("packing", h.evidence["families"])

    def test_unrelated_family_cannot_veto_a_real_reproduction(self):
        """The reproduced defect (the other direction): an incomparable third
        family with opposite behaviour used to cancel a genuine reproduction
        between two comparable families."""
        scheduling = None
        orders = (("routing", 0.90, 0.05), ("scheduling", 0.90, 0.05),
                  ("packing", 0.10, 0.55))
        for fam, rc, gap in orders:
            for i in range(2):
                profile = self.make_profile(problem_id=f"{fam}{i}", family=fam,
                                            resource_coupling=rc)
                rec = self.make_record(execution_id=f"veto_{fam}_{i}",
                                       task_id=f"{fam}{i}", strategy_id="S04",
                                       profile=profile, gap=gap)
                self.bank.append(rec)
                if fam == "scheduling" and i == 1:
                    scheduling = rec
        hints = [h for h in self.check(scheduling) if h.criterion == "C5"]
        self.assertTrue(hints, "routing+scheduling reproduce")
        self.assertEqual(set(hints[0].evidence["families"]),
                         {"routing", "scheduling"})
        # packing is structurally incomparable AND opposite: it is neither
        # mixed into the statistic nor able to veto the reproduction.
        self.assertNotIn("packing", hints[0].evidence["families"])

    def test_unknown_structure_never_counts_as_similarity(self):
        """'Both sides unknown' is a shared absence of evidence, not evidence
        of structural similarity — so C5 stays silent."""
        last = None
        for fam in ("routing", "scheduling"):
            for i in range(2):
                profile = self.make_profile(problem_id=f"{fam}{i}", family=fam,
                                            resource_coupling=None)
                last = self.make_record(execution_id=f"unk_{fam}_{i}",
                                        task_id=f"{fam}{i}", strategy_id="S04",
                                        profile=profile, gap=0.05)
                self.bank.append(last)
        self.assertNotIn("C5", {h.criterion for h in self.check(last)})

    def test_single_family_silent(self):
        last = None
        for i in range(3):
            last = self.make_record(execution_id=f"c5s_{i}", task_id=f"t{i}",
                                    strategy_id="S04", gap=0.05)
            self.bank.append(last)
        self.assertNotIn("C5", self.criteria(last))


class TestC6StableSuccess(TriggerCase):
    def test_stable_success_fires(self):
        # Retries must be MEASURED zero on every record — unknown retries
        # never count as proof of stability.
        from or_harness.core.schema import COST_DIMENSIONS
        new = self.seed("S01", 4, 0.05, task_prefix="st",
                        cost_measured=tuple(COST_DIMENSIONS))
        self.assertIn("C6", self.criteria(new))

    def test_three_samples_silent(self):
        from or_harness.core.schema import COST_DIMENSIONS
        new = self.seed("S01", 3, 0.05, task_prefix="st",
                        cost_measured=tuple(COST_DIMENSIONS))
        self.assertNotIn("C6", self.criteria(new))

    def test_unmeasured_retries_never_claim_stability(self):
        """retries=0 but never measured: C6 must stay silent — unknown is
        not evidence of stability."""
        new = self.seed("S01", 4, 0.05, task_prefix="st")
        self.assertNotIn("C6", self.criteria(new))

    def test_retries_break_stability(self):
        last = None
        for i in range(4):
            last = self.make_record(
                execution_id=f"r{i}", task_id=f"rt{i}", strategy_id="S01", gap=0.05,
                cost=CostVector(llm_tokens=100, solver_runtime_s=1.0, retries=1))
            self.bank.append(last)
        self.assertNotIn("C6", self.criteria(last))


if __name__ == "__main__":
    unittest.main()
