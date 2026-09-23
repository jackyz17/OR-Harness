"""Induction-pattern tests: each of the four patterns fires when it should
and, critically, stays silent when it should (n>=2 gate, cell scope,
direction checks, unknown-is-not-similarity).

The four patterns are named for what they are — there are no criterion
numbers: strategy_contrast, intervention_recovery, structural_reproduction,
advantage_reversal.
"""
import inspect
import unittest

from helpers import HarnessTestCase

from or_harness.core.schema import COST_DIMENSIONS, CostVector, FailureRecord
from or_harness.strategy.experience_bank import ExperienceBank
from or_harness.strategy.stats import ConditionalStats
from or_harness.strategy.triggers import check_triggers


class TriggerCase(HarnessTestCase):
    def setUp(self):
        super().setUp()
        self.bank = ExperienceBank(self.store)
        self.stats = ConditionalStats(self.bank)

    def check(self, record, **kwargs):
        return check_triggers(record, self.stats, **kwargs)

    def patterns(self, record, **kwargs):
        return {h.pattern for h in self.check(record, **kwargs)}

    def seed(self, strategy_id, n, gap, task_prefix="s", cost=None,
             cost_measured=None, rc=None):
        last = None
        for i in range(n):
            kwargs = {}
            if rc is not None:
                kwargs["profile"] = self.make_profile(
                    problem_id=f"{task_prefix}{i}", resource_coupling=rc)
            last = self.make_record(
                execution_id=f"{task_prefix}_{strategy_id}_{i}",
                task_id=f"{task_prefix}{i}", strategy_id=strategy_id, gap=gap,
                cost=cost or CostVector(llm_tokens=100, solver_runtime_s=1.0),
                cost_measured=cost_measured, **kwargs)
            self.bank.append(last)
        return last


class TestStrategyContrast(TriggerCase):
    def test_single_sample_never_triggers(self):
        # Q1 from the design doc: S02 first execution Q=0.98 vs S01's single
        # 0.85 -> no trigger (n=1 on both sides of the divergence check).
        self.seed("S01", 1, 0.15, task_prefix="a")
        new = self.seed("S02", 1, 0.02, task_prefix="b")
        self.assertNotIn("strategy_contrast", self.patterns(new))

    def test_quality_contrast_fires(self):
        # S01 meanQ 0.60 vs S04 meanQ 0.98 — significant quality contrast.
        self.seed("S01", 2, 0.40, task_prefix="a")   # meanQ 0.60
        new = self.seed("S04", 2, 0.02, task_prefix="b")  # meanQ 0.98
        hints = [h for h in self.check(new)
                 if h.pattern == "strategy_contrast"]
        self.assertTrue(hints)
        self.assertEqual(set(hints[0].strategy_ids), {"S01", "S04"})

    def test_small_difference_silent(self):
        # Quality difference below threshold — no trigger.
        self.seed("S01", 2, 0.10, task_prefix="a")   # meanQ 0.90
        new = self.seed("S04", 2, 0.15, task_prefix="b")  # meanQ 0.85
        self.assertNotIn("strategy_contrast", self.patterns(new))

    def test_cost_contrast_fires(self):
        # Quality tied (both ~0.80), S01 100% more expensive in tokens.
        cheap = CostVector(llm_tokens=100, solver_runtime_s=1.0)
        pricey = CostVector(llm_tokens=200, solver_runtime_s=1.0)
        self.seed("S06", 2, 0.20, task_prefix="a", cost=cheap)
        new = self.seed("S01", 2, 0.20, task_prefix="b", cost=pricey)
        hints = [h for h in self.check(new)
                 if h.pattern == "strategy_contrast"]
        self.assertTrue(hints)
        self.assertEqual(hints[0].evidence["kind"], "cost")

    def test_the_lesson_is_the_relation_not_one_win(self):
        """A contrast is reported as a RELATION between two strategies in one
        structural cell — both sides and both sample counts are carried, so
        the reader sees a comparison, never a single winner."""
        self.seed("S01", 2, 0.40, task_prefix="a")
        new = self.seed("S04", 3, 0.02, task_prefix="b")
        hint = next(h for h in self.check(new)
                    if h.pattern == "strategy_contrast")
        self.assertEqual(set(hint.evidence["n"]), {"S01", "S04"})
        self.assertEqual(hint.evidence["n"]["S04"], 3)
        self.assertEqual(set(hint.evidence["observed_quality"]),
                         {"S01", "S04"})


class TestInterventionRecovery(TriggerCase):
    def test_recovery_fires(self):
        rec = self.make_record(
            execution_id="f1", task_id="ft1", feasible=True, status="feasible",
            failures=[FailureRecord(attempt=1, error="infeasible model",
                                    recovery_action="fallback:S01")])
        self.bank.append(rec)
        hints = [h for h in self.check(rec)
                 if h.pattern == "intervention_recovery"]
        self.assertTrue(hints)
        self.assertEqual(hints[0].evidence["execution_id"], "f1")

    def test_failure_without_recovery_silent(self):
        rec = self.make_record(
            execution_id="f2", task_id="ft2", feasible=False, status="error",
            failures=[FailureRecord(attempt=1, error="crash")])
        self.bank.append(rec)
        self.assertNotIn("intervention_recovery", self.patterns(rec))

    def test_clean_run_silent(self):
        rec = self.make_record(execution_id="f3", task_id="ft3")
        self.bank.append(rec)
        self.assertNotIn("intervention_recovery", self.patterns(rec))

    def test_cross_execution_solver_switch_fires(self):
        """A solver switch is an intervention: the change in real outcome
        emerges from two independent facts, never from a narrated record."""
        failed = self.make_record(execution_id="x1", task_id="xt", feasible=False,
                                  status="error", solver={"name": "pulp"})
        self.bank.append(failed)
        success = self.make_record(execution_id="x2", task_id="xt",
                                   solver={"name": "ortools"})
        hints = [h for h in self.check(success, prior_failures=[failed])
                 if h.pattern == "intervention_recovery"]
        self.assertTrue(hints)
        self.assertEqual(hints[0].evidence["kind"], "cross_execution_recovery")

    def test_same_solver_retry_is_not_an_intervention(self):
        failed = self.make_record(execution_id="y1", task_id="yt", feasible=False,
                                  status="error", solver={"name": "pulp"})
        success = self.make_record(execution_id="y2", task_id="yt",
                                   solver={"name": "pulp"})
        hints = [h for h in self.check(success, prior_failures=[failed])
                 if h.pattern == "intervention_recovery"]
        self.assertFalse(hints)

    def test_success_after_intervention_is_evidence_not_proof(self):
        """The hint names the change; it does not claim causation. The evidence
        carries both sides so the reader decides."""
        rec = self.make_record(
            execution_id="f4", task_id="ft4", feasible=True, status="feasible",
            failures=[FailureRecord(attempt=1, error="bad bounds",
                                    recovery_action="repair:bounds")])
        self.bank.append(rec)
        hint = next(h for h in self.check(rec)
                    if h.pattern == "intervention_recovery")
        self.assertIn("failures", hint.evidence)
        self.assertIn("final_status", hint.evidence)


class TestStructuralReproduction(TriggerCase):
    def test_reproduced_high_performance_fires(self):
        """Two families, same strategy, SAME structure (rc cell), same
        direction — the reproduction this pattern exists to report."""
        last = None
        for fam in ("routing", "scheduling"):
            for i in range(2):
                profile = self.make_profile(problem_id=f"{fam}{i}", family=fam,
                                            resource_coupling=0.90)
                last = self.make_record(execution_id=f"sr_{fam}_{i}",
                                        task_id=f"{fam}{i}", strategy_id="S04",
                                        profile=profile, gap=0.05)
                self.bank.append(last)
        hints = [h for h in self.check(last)
                 if h.pattern == "structural_reproduction"]
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
        hits = [h for h in self.check(last)
                if h.pattern == "structural_reproduction"]
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
        hints = [h for h in self.check(scheduling)
                 if h.pattern == "structural_reproduction"]
        self.assertTrue(hints, "routing+scheduling reproduce")
        self.assertEqual(set(hints[0].evidence["families"]),
                         {"routing", "scheduling"})
        # packing is structurally incomparable AND opposite: it is neither
        # mixed into the statistic nor able to veto the reproduction.
        self.assertNotIn("packing", hints[0].evidence["families"])

    def test_unknown_structure_never_counts_as_similarity(self):
        """'Both sides unknown' is a shared absence of evidence, not evidence
        of structural similarity — so the pattern stays silent."""
        last = None
        for fam in ("routing", "scheduling"):
            for i in range(2):
                profile = self.make_profile(problem_id=f"{fam}{i}", family=fam,
                                            resource_coupling=None)
                last = self.make_record(execution_id=f"unk_{fam}_{i}",
                                        task_id=f"{fam}{i}", strategy_id="S04",
                                        profile=profile, gap=0.05)
                self.bank.append(last)
        self.assertNotIn("structural_reproduction", self.patterns(last))

    def test_single_family_silent(self):
        """One family's own evidence is not reproduction: a single task's (or
        family's) observation is never transferable knowledge."""
        last = None
        for i in range(3):
            last = self.make_record(execution_id=f"srs_{i}", task_id=f"t{i}",
                                    strategy_id="S04", gap=0.05)
            self.bank.append(last)
        self.assertNotIn("structural_reproduction", self.patterns(last))


class TestAdvantageReversal(TriggerCase):
    def test_reversal_across_cells_fires(self):
        """The same strategy is strong at low coupling and weak at high
        coupling: the lesson is the BOUNDARY, not the success count."""
        last = None
        for i in range(2):
            self.bank.append(self.make_record(
                execution_id=f"rev_low_{i}", task_id=f"rl{i}",
                strategy_id="S01", gap=0.05,  # meanQ 0.95
                profile=self.make_profile(problem_id=f"rl{i}",
                                          resource_coupling=0.10)))
        for i in range(2):
            last = self.make_record(
                execution_id=f"rev_high_{i}", task_id=f"rh{i}",
                strategy_id="S01", gap=0.95,  # meanQ 0.05
                profile=self.make_profile(problem_id=f"rh{i}",
                                          resource_coupling=0.90))
            self.bank.append(last)
        hints = [h for h in self.check(last)
                 if h.pattern == "advantage_reversal"]
        self.assertTrue(hints)
        ev = hints[0].evidence
        self.assertEqual(ev["kind"], "advantage_reversal")
        self.assertIn("0.75,1.00", ev["adverse_cell"]["group_key"])
        self.assertIn("0.00,0.25", ev["advantageous_cell"]["group_key"])

    def test_consistent_advantage_is_not_a_reversal(self):
        """Strong everywhere is not a boundary — the pattern stays silent."""
        last = None
        for rc in (0.10, 0.90):
            for i in range(2):
                last = self.make_record(
                    execution_id=f"cons_{rc}_{i}", task_id=f"c{rc}{i}",
                    strategy_id="S01", gap=0.05,
                    profile=self.make_profile(problem_id=f"c{rc}{i}",
                                              resource_coupling=rc))
                self.bank.append(last)
        self.assertNotIn("advantage_reversal", self.patterns(last))

    def test_single_cell_silent(self):
        """One cell cannot demonstrate a boundary: there is nothing to
        compare against."""
        last = self.seed("S01", 4, 0.05, task_prefix="one", rc=0.90)
        self.assertNotIn("advantage_reversal", self.patterns(last))

    def test_thin_cell_silent(self):
        """A cell with n=1 is a single observation, not a condition."""
        self.bank.append(self.make_record(
            execution_id="thin_low", task_id="tl", strategy_id="S01", gap=0.05,
            profile=self.make_profile(problem_id="tl", resource_coupling=0.10)))
        last = None
        for i in range(2):
            last = self.make_record(
                execution_id=f"thin_high_{i}", task_id=f"th{i}",
                strategy_id="S01", gap=0.95,
                profile=self.make_profile(problem_id=f"th{i}",
                                          resource_coupling=0.90))
            self.bank.append(last)
        self.assertNotIn("advantage_reversal", self.patterns(last))

    def test_another_family_cannot_create_a_boundary(self):
        """A cross-family difference is a DIFFERENT question
        (structural_reproduction). Pooling families here would let a family's
        own structure masquerade as a boundary of this one."""
        self.bank.append(self.make_record(
            execution_id="fam_low", task_id="fl", strategy_id="S01", gap=0.05,
            profile=self.make_profile(problem_id="fl", family="routing",
                                      resource_coupling=0.10)))
        last = self.make_record(
            execution_id="fam_high", task_id="fh", strategy_id="S01", gap=0.95,
            profile=self.make_profile(problem_id="fh", family="scheduling",
                                      resource_coupling=0.90))
        self.bank.append(last)
        self.assertNotIn("advantage_reversal", self.patterns(last))

    def test_unknown_structure_is_not_a_condition(self):
        """An unmeasured dimension is an absence of evidence, not a
        structural condition to read a boundary off."""
        self.bank.append(self.make_record(
            execution_id="unk_low", task_id="ul", strategy_id="S01", gap=0.05,
            profile=self.make_profile(problem_id="ul", resource_coupling=None)))
        last = self.make_record(
            execution_id="unk_high", task_id="uh", strategy_id="S01", gap=0.95,
            profile=self.make_profile(problem_id="uh", resource_coupling=0.90))
        self.bank.append(last)
        self.assertNotIn("advantage_reversal", self.patterns(last))


class TestRetiredCriteriaLeftNoPath(unittest.TestCase):
    """The retired per-strategy criteria must leave NO reachable path.

    These were deleted because they were not relations: a lone strategy's
    extreme mean, a within-cell quality trend, and an accumulated success
    count. None of them may reappear as a hint or as a helper.
    """

    def test_no_historical_criterion_names_in_the_module(self):
        import or_harness.strategy.triggers as triggers
        source = inspect.getsource(triggers)
        for name in ("_c1_", "_c2_", "_c3_", "_c4_", "_c5_", "_c6_"):
            self.assertNotIn(name, source, f"{name} still exists")
        for constant in ("TREND_MIN_N", "STABLE_SUCCESS_MIN_N"):
            self.assertFalse(hasattr(triggers, constant),
                             f"{constant} still exists")

    def test_no_quality_trend_helper_remains(self):
        """The trend helper existed only for the retired drift criterion."""
        from or_harness.strategy.stats import GroupStats
        self.assertFalse(hasattr(GroupStats, "quality_trend"))

    def test_hint_exposes_pattern_not_criterion(self):
        from or_harness.strategy.triggers import InductionHint
        hint = InductionHint(pattern="strategy_contrast", strategy_ids=["S01"],
                             group_key="family=routing")
        self.assertEqual(hint.to_dict()["pattern"], "strategy_contrast")
        self.assertNotIn("criterion", hint.to_dict())

    def test_pure_success_accumulation_is_not_a_pattern(self):
        """Four clean runs, every dimension measured, no other strategy, one
        cell: nothing is a relation, so nothing fires. The old stable-success
        criterion would have fired here."""
        case = HarnessTestCase("run")
        case.setUp()
        try:
            bank = ExperienceBank(case.store)
            stats = ConditionalStats(bank)
            last = None
            for i in range(4):
                last = case.make_record(
                    execution_id=f"pure{i}", task_id=f"pt{i}", strategy_id="S01",
                    gap=0.05,
                    cost_measured=tuple(COST_DIMENSIONS))
                bank.append(last)
            patterns = {h.pattern for h in check_triggers(last, stats)}
            self.assertEqual(
                patterns - {"strategy_contrast", "intervention_recovery",
                            "structural_reproduction", "advantage_reversal"},
                set(), "only the four named patterns may be emitted")
        finally:
            case.tearDown()


if __name__ == "__main__":
    unittest.main()
