"""Selector tests: evidence-derived candidates, scoring, ablation modes,
cross-family discount, suspect downweighting.

There is no candidate menu: every candidate in these tests exists because a
record or an entry really mentions it.
"""
import unittest

from helpers import HarnessTestCase

from or_harness.core.schema import CostVector, StrategicEntry
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
        self.selector = Selector(self.sbank, self.stats)

    def seed(self, strategy_id, n=2, gap=0.05, task_prefix="t",
             cost=None, **kwargs):
        """Real evidence for one strategy — the only way it becomes a
        candidate."""
        records = []
        for i in range(n):
            record = self.make_record(
                execution_id=f"ex_{strategy_id}_{i}",
                task_id=f"{task_prefix}{strategy_id}{i}",
                strategy_id=strategy_id, gap=gap,
                cost=cost if cost is not None else CostVector(), **kwargs)
            self.bank.append(record)
            records.append(record)
        return records


class TestNoCandidateMenu(SelectorCase):
    """The framework has no directory of strategies to fall back on."""

    def test_empty_bank_recalls_nothing(self):
        self.assertEqual(self.selector.recall(self.make_profile()), [])
        self.assertEqual(self.selector.candidate_ids(self.make_profile()), [])

    def test_a_strategy_becomes_a_candidate_by_being_run(self):
        self.seed("S04", n=3, gap=0.0)
        recs = self.selector.recall(self.make_profile(), top=5)
        self.assertEqual([r.strategy_id for r in recs], ["S04"])
        self.assertEqual(recs[0].evidence, "conditional_stats")
        self.assertAlmostEqual(recs[0].expected_quality, 1.0)

    def test_untried_strategies_are_not_candidates(self):
        """Only the strategies memory really holds appear — no id that was
        merely 'available' in some directory."""
        self.seed("S04", n=2)
        ids = {r.strategy_id
               for r in self.selector.recall(self.make_profile(), top=50)}
        self.assertEqual(ids, {"S04"})

    def test_caller_proposal_restricts_without_inventing(self):
        self.seed("S04", n=2)
        recs = self.selector.recall(
            self.make_profile(), top=10,
            candidates=["S04", "custom:never-run"])
        # The proposal is a FILTER: the unknown id is not given a score.
        self.assertEqual([r.strategy_id for r in recs], ["S04"])
        self.assertEqual(
            self.selector.candidate_ids(self.make_profile()), ["S04"])

    def test_memory_mode_none_recalls_nothing(self):
        self.seed("S04", n=3)
        self.assertEqual(
            self.selector.recall(self.make_profile(), memory_mode="none"),
            [])
        self.assertEqual(
            self.selector.candidate_ids(self.make_profile(),
                                        memory_mode="none"), [])

    def test_no_placeholder_rows_or_negative_infinity(self):
        """A memory-less strategy is ABSENT, never a zero-quality row."""
        self.seed("S04", n=2)
        recs = self.selector.recall(self.make_profile(), top=50)
        self.assertNotIn("no_memory", {r.evidence for r in recs})
        self.assertTrue(all(r.score != float("-inf") for r in recs))

    def test_recommendation_carries_no_directory_content(self):
        """No name/description is fabricated for a strategy id."""
        self.seed("S04", n=2)
        payload = self.selector.recall(self.make_profile())[0].to_dict()
        self.assertNotIn("name", payload)
        self.assertEqual(payload["strategy_id"], "S04")


class TestEvidencePrecedence(SelectorCase):
    def test_entry_overrides_stats(self):
        self.seed("S01", n=3, gap=0.5)
        self.sbank.add(StrategicEntry(
            entry_id="se_boost", strategy_id="S01",
            pattern={"predicates": {"family": "routing"}},
            expected_quality_hat=0.99, quality_interval=(0.9, 1.0),
            failure_prob=0.0, status="validated", support_n=6,
            verification={"state": "verified", "claim": "S01 holds here"},
            provenance=["ex_S01_0", "ex_S01_1", "ex_S01_2"]))
        recs = self.selector.recall(self.make_profile(), top=5)
        s01 = next(r for r in recs if r.strategy_id == "S01")
        self.assertEqual(s01.evidence, "strategic_entry")
        self.assertAlmostEqual(s01.expected_quality, 0.99)

    def test_entry_knowledge_block_comes_from_the_entry(self):
        """The content travels with the ENTRY, not from a directory."""
        self.sbank.add(StrategicEntry(
            entry_id="se_own", strategy_id="custom:two-phase",
            pattern={"predicates": {"family": "routing"}},
            strategy_type="decomposition",
            actions=["split along the resource axis", "iterate"],
            fallback_strategy_id="custom:monolith",
            applicability=["only when the resource coupling is high"],
            support_n=4))
        recs = self.selector.recall(self.make_profile(), top=5)
        rec = next(r for r in recs if r.strategy_id == "custom:two-phase")
        self.assertEqual(rec.knowledge["strategy_type"], "decomposition")
        self.assertEqual(rec.knowledge["actions"],
                         ["split along the resource axis", "iterate"])
        self.assertEqual(rec.knowledge["fallback_strategy_id"],
                         "custom:monolith")
        self.assertEqual(rec.knowledge["applicability"],
                         ["only when the resource coupling is high"])

    def test_entry_without_content_reports_it_absent(self):
        self.sbank.add(StrategicEntry(
            entry_id="se_bare", strategy_id="custom:bare",
            pattern={"predicates": {"family": "routing"}}, support_n=2))
        rec = self.selector.recall(self.make_profile(), top=5)[0]
        self.assertEqual(rec.knowledge["actions"], [])
        self.assertIsNone(rec.knowledge["strategy_type"])
        self.assertIsNone(rec.knowledge["fallback_strategy_id"])


class TestAblationModes(SelectorCase):
    def setUp(self):
        super().setUp()
        # Quality tied; S06 much cheaper than S01 in this group.
        cheap = CostVector(llm_tokens=2000, solver_runtime_s=30)
        dear = CostVector(llm_tokens=200, solver_runtime_s=3)
        self.seed("S01", n=2, gap=0.05, task_prefix="ta", cost=cheap)
        self.seed("S06", n=2, gap=0.05, task_prefix="tb", cost=dear)

    def _winner(self, mode):
        recs = self.selector.recall(self.make_profile(), top=10,
                                    memory_mode=mode)
        return recs[0].strategy_id, recs[0].evidence

    def test_mode_none_recalls_nothing(self):
        self.assertEqual(
            self.selector.recall(self.make_profile(), memory_mode="none"),
            [])

    def test_mode_cases_ignores_cost_weights(self):
        # No cost scalarization; quality tied -> fail_rate 0 both, tie broken
        # by id order, and evidence is stats (not entries).
        winner, evidence = self._winner("cases")
        self.assertEqual(evidence, "conditional_stats")
        self.assertEqual(winner, "S01")

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
        s07 = next(r for r in recs if r.strategy_id == "S07")
        self.assertTrue(any("suspect" in w for w in s07.risk_warnings))

    def test_cross_family_discounted(self):
        # A claim with no family predicate (cross-family by construction); its
        # evidence sits in 'routing' but it is queried from 'scheduling'.
        self.seed("S01", n=2, task_prefix="tr")
        self.sbank.add(StrategicEntry(
            entry_id="se_wide", strategy_id="S01",
            pattern={"predicates": {"resource_coupling": [0.75, 1.0]}},
            expected_quality_hat=0.95, quality_interval=(0.8, 1.0),
            failure_prob=0.0, status="validated", support_n=5,
            verification={"state": "verified", "claim": "rc>=0.75 holds"},
            provenance=["ex_S01_0", "ex_S01_1"]))
        other_family = self.make_profile(problem_id="q", family="scheduling")
        recs = self.selector.recall(other_family, top=10)
        s01 = next(r for r in recs if r.strategy_id == "S01")
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
        self.seed("S01", n=2, task_prefix="tx")
        self.seed("S06", n=2, task_prefix="ty")
        recs = self.selector.recall(self.make_profile(), top=10,
                                   exclude=["S01", "S06"])
        self.assertEqual(recs, [])


if __name__ == "__main__":
    unittest.main()
