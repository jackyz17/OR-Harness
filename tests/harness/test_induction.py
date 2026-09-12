"""Induction tests: consolidation, honest intervals, scope ladder widen/tighten,
restatement refusal, cold-archive veto, LLM condition citation binding, rebuild."""
import unittest

from helpers import HarnessTestCase

from or_harness.core.schema import StrategicEntry, min_interval_width
from or_harness.strategy.experience_bank import ExperienceBank
from or_harness.strategy.induction import InductionEngine
from or_harness.strategy.stats import ConditionalStats
from or_harness.strategy.strategic_bank import StrategicBank


class InductionCase(HarnessTestCase):
    def setUp(self):
        super().setUp()
        self.bank = ExperienceBank(self.store)
        self.sbank = StrategicBank(self.store)
        self.stats = ConditionalStats(self.bank)
        self.engine = InductionEngine(self.stats, self.sbank)

    def seed(self, strategy_id, gaps, family="routing", task_prefix="s"):
        ids = []
        for i, gap in enumerate(gaps):
            profile = self.make_profile(problem_id=f"{task_prefix}{i}", family=family)
            rec = self.make_record(execution_id=f"{task_prefix}_{strategy_id}_{i}",
                                   task_id=f"{task_prefix}{i}",
                                   strategy_id=strategy_id, profile=profile, gap=gap)
            self.bank.append(rec)
            ids.append(rec.execution_id)
        return ids


class TestInduce(InductionCase):
    def test_creates_candidate_entry(self):
        ids = self.seed("S01", [0.05, 0.10, 0.08])
        result = self.engine.induce(self.make_profile("q"), "S01")
        self.assertIsNotNone(result.get("created"))
        entry = self.sbank.get(result["created"])
        self.assertEqual(entry.status, "candidate")  # never born validated
        self.assertEqual(entry.scope_level, "L1")
        self.assertEqual(entry.predicates["family"], "routing")
        self.assertAlmostEqual(entry.expected_quality_hat, (0.95 + 0.90 + 0.92) / 3, places=3)
        self.assertEqual(sorted(entry.provenance), sorted(ids))

    def test_honest_interval_for_small_n(self):
        self.seed("S01", [0.0, 0.0])
        result = self.engine.induce(self.make_profile("q"), "S01")
        entry = self.sbank.get(result["created"])
        lo, hi = entry.quality_interval
        self.assertGreaterEqual(hi - lo, min_interval_width(2))
        # n=2 may not claim [0.95, 1.0]-style narrow intervals
        self.assertLessEqual(lo, 0.5)

    def test_single_sample_refused(self):
        self.seed("S01", [0.05])
        result = self.engine.induce(self.make_profile("q"), "S01")
        self.assertIsNone(result.get("created"))
        self.assertIn("fewer than 2", result["skipped"])

    def test_restatement_refused(self):
        self.seed("S01", [0.05, 0.10, 0.08])
        first = self.engine.induce(self.make_profile("q"), "S01")
        again = self.engine.induce(self.make_profile("q2"), "S01")
        self.assertIsNone(again.get("created"))
        self.assertIn("restatement", again["skipped"])
        self.assertEqual(again["entry_id"], first["created"])

    def test_update_when_evidence_changes(self):
        self.seed("S01", [0.05, 0.10])
        first = self.engine.induce(self.make_profile("q"), "S01")
        self.seed("S01", [0.60, 0.65], task_prefix="late")
        result = self.engine.induce(self.make_profile("q2"), "S01")
        self.assertEqual(result.get("updated"), first["created"])
        entry = self.sbank.get(first["created"])
        self.assertLess(entry.expected_quality_hat, 0.8)

    def test_dry_run_creates_nothing(self):
        self.seed("S01", [0.05, 0.10])
        result = self.engine.induce(self.make_profile("q"), "S01", dry_run=True)
        self.assertIn("would_create", result)
        self.assertEqual(self.sbank.count(), 0)


class TestColdArchiveVeto(InductionCase):
    def test_veto_blocks_and_force_overrides(self):
        self.seed("S01", [0.05, 0.10])
        created = self.engine.induce(self.make_profile("q"), "S01")["created"]
        entry = self.sbank.get(created)
        entry.status = "suspect"
        self.sbank.update(entry)
        card = self.sbank.retire(created, reason="bad generalization")
        # Same evidence, same pattern -> vetoed.
        blocked = self.engine.induce(self.make_profile("q2"), "S01")
        self.assertIsNone(blocked.get("created"))
        self.assertEqual(blocked["vetoed"]["pattern_hash"], card.pattern_hash)
        # Force overrides when the harness judges the environment drifted.
        forced = self.engine.induce(self.make_profile("q2"), "S01", force=True)
        self.assertIsNotNone(forced.get("created"))


class TestScopeLadder(InductionCase):
    def _wide_entry(self, support_families=("routing", "scheduling")):
        ids = []
        for fam in support_families:
            ids.extend(self.seed("S01", [0.05, 0.10], family=fam,
                                 task_prefix=f"w{fam[:2]}"))
        entry = StrategicEntry(
            entry_id="se_wide", strategy_id="S01",
            pattern={"scope_level": "L1",
                     "predicates": {"family": "routing",
                                    "resource_coupling": [0.75, 1.0]}},
            expected_quality_hat=0.9, quality_interval=(0.5, 1.0),
            failure_prob=0.0, provenance=ids, support_n=len(ids))
        self.sbank.add(entry)
        return entry

    def test_widen_on_cross_family_reproduction(self):
        self._wide_entry()
        result = self.engine.widen("se_wide")
        self.assertEqual(result.get("widened"), "se_wide")
        self.assertEqual(result["new_scope"], "L2")
        self.assertNotIn("family", result["predicates"])

    def test_widen_refused_when_advantage_diverges(self):
        # Second family performs badly -> advantage does not reproduce.
        self.seed("S01", [0.05, 0.10], family="routing", task_prefix="wr")
        self.seed("S01", [0.80, 0.85], family="scheduling", task_prefix="ws")
        ids = [r.execution_id for r in self.bank.all()]
        self.sbank.add(StrategicEntry(
            entry_id="se_div", strategy_id="S01",
            pattern={"scope_level": "L1",
                     "predicates": {"family": "routing",
                                    "resource_coupling": [0.75, 1.0]}},
            expected_quality_hat=0.9, quality_interval=(0.5, 1.0),
            failure_prob=0.0, provenance=ids, support_n=len(ids)))
        result = self.engine.widen("se_div")
        self.assertIn("error", result)

    def test_tighten_after_cross_family_miss(self):
        # Q3 from the design doc: L2 entry misses in a new family -> tighten
        # to L1 (range was wrong), not suspect (content may be right).
        ids = self.seed("S01", [0.05, 0.10], family="routing", task_prefix="tr")
        self.sbank.add(StrategicEntry(
            entry_id="se_l2", strategy_id="S01",
            pattern={"scope_level": "L2",
                     "predicates": {"resource_coupling": [0.75, 1.0]}},
            expected_quality_hat=0.9, quality_interval=(0.5, 1.0),
            failure_prob=0.0, provenance=ids, support_n=2))
        result = self.engine.tighten("se_l2")
        self.assertEqual(result.get("tightened"), "se_l2")
        entry = self.sbank.get("se_l2")
        self.assertEqual(entry.scope_level, "L1")
        self.assertEqual(entry.predicates["family"], "routing")
        self.assertEqual(entry.status, "candidate")  # not demoted


class TestLLMConditions(InductionCase):
    def test_valid_condition_accepted_unverified(self):
        ids = self.seed("S01", [0.05, 0.10])
        result = self.engine.induce(
            self.make_profile("q"), "S01",
            llm_conditions=[{"text": "performs above 90% here",
                             "supporting_execution_ids": ids}])
        entry = self.sbank.get(result["created"])
        self.assertEqual(len(entry.applicability), 1)
        self.assertFalse(entry.applicability[0].verified)  # never enters scoring yet
        self.assertEqual(result["conditions_rejected"], [])

    def test_forged_citation_rejected(self):
        self.seed("S01", [0.05, 0.10])
        result = self.engine.induce(
            self.make_profile("q"), "S01",
            llm_conditions=[{"text": "great", "supporting_execution_ids": ["ex_fake"]}])
        entry = self.sbank.get(result["created"])
        self.assertEqual(entry.applicability, [])
        self.assertEqual(len(result["conditions_rejected"]), 1)
        self.assertIn("does not exist", result["conditions_rejected"][0]["reason"])

    def test_numeric_claim_disagreement_rejected(self):
        ids = self.seed("S01", [0.05, 0.10])  # quality ~90-95%
        result = self.engine.induce(
            self.make_profile("q"), "S01",
            llm_conditions=[{"text": "achieves 30% quality",
                             "supporting_execution_ids": ids}])
        entry = self.sbank.get(result["created"])
        self.assertEqual(entry.applicability, [])
        self.assertIn("disagree", result["conditions_rejected"][0]["reason"])

    def test_uncited_condition_rejected(self):
        self.seed("S01", [0.05, 0.10])
        result = self.engine.induce(
            self.make_profile("q"), "S01",
            llm_conditions=[{"text": "trust me", "supporting_execution_ids": []}])
        entry = self.sbank.get(result["created"])
        self.assertEqual(entry.applicability, [])


class TestRebuild(InductionCase):
    """Re-induction from currently retained evidence. NOT exact
    reconstruction: the re-induced bank may legitimately differ from the
    previous one (induction logic and evidence sets evolve)."""

    def test_rebuild_regenerates_entries(self):
        self.seed("S01", [0.05, 0.10], task_prefix="a")
        self.seed("S04", [0.02, 0.04], task_prefix="b")
        plan = self.engine.rebuild(dry_run=True)
        self.assertEqual(plan["would_rebuild"], 2)
        result = self.engine.rebuild()
        self.assertEqual(result["rebuilt"], 2)
        self.assertEqual(self.sbank.count(), 2)

    def test_rebuild_preserves_cold_archive(self):
        self.seed("S01", [0.05, 0.10], task_prefix="a")
        created = self.engine.induce(self.make_profile("q"), "S01")["created"]
        self.sbank.retire(created, reason="x")
        self.engine.rebuild()
        self.assertEqual(len(self.sbank.cold_archive()), 1)


if __name__ == "__main__":
    unittest.main()
