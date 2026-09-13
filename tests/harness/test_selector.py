"""Selector tests: fallback chain, scoring, ablation modes, cross-family
discount, suspect downweighting."""
import unittest

from helpers import HarnessTestCase

from or_harness.core.schema import CostVector, StrategicEntry
from or_harness.strategy.catalog import load_catalog
from or_harness.strategy.experience_bank import ExperienceBank
from or_harness.strategy.selector import Selector
from or_harness.strategy.stats import ConditionalStats
from or_harness.strategy.strategic_bank import StrategicBank


class SelectorCase(HarnessTestCase):
    def setUp(self):
        super().setUp()
        self.bank = ExperienceBank(self.store)
        self.sbank = StrategicBank(self.store)
        self.stats = ConditionalStats(self.bank)
        self.catalog = load_catalog()
        self.selector = Selector(self.catalog, self.sbank, self.stats)


class TestCatalog(SelectorCase):
    def test_catalog_loads(self):
        self.assertEqual(len(self.catalog), 10)
        s01 = self.catalog["S01"]
        self.assertEqual(s01.fallback, "S06")
        self.assertEqual(s01.solver_family, "milp")

    def test_applicability_filters(self):
        profile = self.make_profile(temporal_coupling=0.9, resource_coupling=0.1,
                                    route_complexity=0.1)
        recs = self.selector.recall(profile, top=20)
        ids = {r.strategy.strategy_id for r in recs}
        self.assertIn("S03", ids)         # temporal window applies
        self.assertNotIn("S02", ids)      # resource_coupling too low
        self.assertNotIn("S08", ids)      # route + resource requirements unmet


class TestFallbackChain(SelectorCase):
    def test_cold_start_no_evidence(self):
        recs = self.selector.recall(self.make_profile())
        self.assertTrue(all(r.evidence == "no_memory" for r in recs))
        self.assertTrue(all(r.score == float('-inf') for r in recs))
        self.assertTrue(all(r.confidence == 0.0 for r in recs))

    def test_stats_provide_first_evidence(self):
        # S04 performs exceptionally in this group; with 3 observations
        # the stats path provides the first evidence.
        for i in range(3):
            self.bank.append(self.make_record(
                execution_id=f"ex_{i}", task_id=f"t{i}", strategy_id="S04", gap=0.0))
        recs = self.selector.recall(self.make_profile(), top=5)
        by_id = {r.strategy.strategy_id: r for r in recs}
        self.assertEqual(by_id["S04"].evidence, "conditional_stats")
        self.assertAlmostEqual(by_id["S04"].expected_quality, 1.0)

    def test_entry_overrides_stats(self):
        for i in range(3):
            self.bank.append(self.make_record(
                execution_id=f"ex_{i}", task_id=f"t{i}", strategy_id="S01", gap=0.5))
        self.sbank.add(StrategicEntry(
            entry_id="se_boost", strategy_id="S01",
            pattern={"predicates": {"family": "routing"}},
            expected_quality_hat=0.99, quality_interval=(0.9, 1.0),
            failure_prob=0.0, status="validated", support_n=6,
            verification={"state": "verified", "claim": "S01 holds here"},
            provenance=["ex_0", "ex_1", "ex_2"]))
        recs = self.selector.recall(self.make_profile(), top=5)
        s01 = next(r for r in recs if r.strategy.strategy_id == "S01")
        self.assertEqual(s01.evidence, "strategic_entry")
        self.assertAlmostEqual(s01.expected_quality, 0.99)


class TestAblationModes(SelectorCase):
    def setUp(self):
        super().setUp()
        # Quality tied; S06 much cheaper than S01 in this group.
        for i in range(2):
            self.bank.append(self.make_record(
                execution_id=f"ex_a{i}", task_id=f"ta{i}", strategy_id="S01",
                gap=0.05, cost=CostVector(llm_tokens=2000, solver_runtime_s=30)))
            self.bank.append(self.make_record(
                execution_id=f"ex_b{i}", task_id=f"tb{i}", strategy_id="S06",
                gap=0.05, cost=CostVector(llm_tokens=200, solver_runtime_s=3)))

    def _winner(self, mode):
        recs = self.selector.recall(self.make_profile(), top=10, memory_mode=mode)
        return recs[0].strategy.strategy_id, recs[0].evidence

    def test_mode_none_returns_no_evidence(self):
        winner, evidence = self._winner("none")
        self.assertEqual(evidence, "no_memory")
        # With no priors, all scores are 0; alphabetical tie-break puts S01 first.
        self.assertEqual(winner, "S01")

    def test_mode_cases_ignores_cost_weights(self):
        # Cases mode: no cost scalarization; quality tied -> fail_rate 0 both,
        # tie broken by id order, and evidence is stats (not entries).
        # Cases mode: no cost scalarization; quality tied -> fail_rate 0 both,
        # tie broken by id order, and evidence is stats (not entries).
        winner, evidence = self._winner("cases")
        self.assertEqual(evidence, "conditional_stats")
        self.assertEqual(winner, "S01")  # quality-equal; lexicographic tie-break

    def test_mode_strategic_ignores_cost(self):
        winner, evidence = self._winner("strategic")
        self.assertEqual(winner, "S01")

    def test_mode_cost_aware_prefers_cheap(self):
        winner, evidence = self._winner("cost-aware")
        self.assertEqual(winner, "S06")  # same quality, ~10x cheaper


class TestEntryFlags(SelectorCase):
    def test_suspect_downweighted_and_warned(self):
        self.sbank.add(StrategicEntry(
            entry_id="se_sus", strategy_id="S07",
            pattern={"predicates": {"family": "routing"}},
            expected_quality_hat=1.0, quality_interval=(0.9, 1.0),
            failure_prob=0.0, status="suspect", support_n=8))
        recs = self.selector.recall(self.make_profile(), top=10)
        s07 = next(r for r in recs if r.strategy.strategy_id == "S07")
        self.assertTrue(any("suspect" in w for w in s07.risk_warnings))

    def test_cross_family_discounted(self):
        # A claim with no family predicate (cross-family by construction); its
        # evidence sits in 'routing' but it is queried from 'scheduling'.
        for i in range(2):
            self.bank.append(self.make_record(
                execution_id=f"ex_r{i}", task_id=f"tr{i}", strategy_id="S01"))
        self.sbank.add(StrategicEntry(
            entry_id="se_wide", strategy_id="S01",
            pattern={"predicates": {"resource_coupling": [0.75, 1.0]}},
            expected_quality_hat=0.95, quality_interval=(0.8, 1.0),
            failure_prob=0.0, status="validated", support_n=5,
            verification={"state": "verified", "claim": "rc>=0.75 holds"},
            provenance=["ex_r0", "ex_r1"]))
        other_family = self.make_profile(problem_id="q", family="scheduling")
        recs = self.selector.recall(other_family, top=10)
        s01 = next(r for r in recs if r.strategy.strategy_id == "S01")
        self.assertEqual(s01.evidence, "strategic_entry")
        self.assertTrue(s01.cross_family)
        self.assertLess(s01.confidence, 0.7)
        self.assertTrue(any("cross-family" in w for w in s01.risk_warnings))

    def test_consultation_marks_entries(self):
        self.sbank.add(StrategicEntry(
            entry_id="se_c", strategy_id="S01",
            pattern={"predicates": {"family": "routing"}},
            support_n=3))
        self.selector.recall(self.make_profile())
        self.assertIsNotNone(self.sbank.get("se_c").last_consulted_at)

    def test_exclude(self):
        recs = self.selector.recall(self.make_profile(), top=10,
                                   exclude=["S01", "S06"])
        self.assertNotIn("S01", {r.strategy.strategy_id for r in recs})
        self.assertNotIn("S06", {r.strategy.strategy_id for r in recs})


if __name__ == "__main__":
    unittest.main()
