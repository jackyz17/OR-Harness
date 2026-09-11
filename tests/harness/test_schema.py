"""Serialization round-trip and structural-grouping tests for core schemas."""
import unittest

from helpers import HarnessTestCase

from or_harness.core.schema import (
    COARSE_BIN_EDGES,
    CostVector,
    ColdArchiveCard,
    ExecutionRecord,
    ProblemProfile,
    StrategicEntry,
    Strategy,
    ApplicabilityCondition,
    group_key,
    min_interval_width,
    pattern_for,
    pattern_hash,
    predicates_cover,
    profile_matches,
    scope_of,
)


class TestSchemaRoundTrip(HarnessTestCase):
    def test_cost_vector_round_trip(self):
        cv = CostVector(llm_tokens=10, tool_calls=2, solver_runtime_s=0.5,
                        retries=1, latency_s=0.7)
        self.assertEqual(CostVector.from_dict(cv.to_dict()), cv)

    def test_cost_scalarize_configurable_weights(self):
        cv = CostVector(llm_tokens=100, retries=2)
        default = cv.scalarize()
        cheap_retry = cv.scalarize(weights={"llm_tokens": 1.0, "retries": 0.0})
        self.assertGreater(default, cheap_retry)
        normed = cv.scalarize(norms={"llm_tokens": 100.0})
        self.assertAlmostEqual(normed, 1.0 + 2 * 2.0, places=6)

    def test_profile_round_trip(self):
        p = self.make_profile(source="harness_supplied",
                              annotations={"note": "from upstream"})
        self.assertEqual(ProblemProfile.from_dict(p.to_dict()), p)

    def test_profile_requires_id_and_family(self):
        with self.assertRaises(ValueError):
            ProblemProfile.from_dict({"family": "x"})
        with self.assertRaises(ValueError):
            ProblemProfile.from_dict({"problem_id": "x"})

    def test_strategy_round_trip(self):
        s = Strategy(strategy_id="S01", name="monolithic",
                     applicability={"family": "routing"},
                     actions=["build", "solve"], fallback="S02",
                     solver_family="milp")
        s2 = Strategy.from_dict(s.to_dict())
        self.assertEqual(s2.strategy_id, "S01")
        self.assertEqual(s2.applicability, s.applicability)
        self.assertEqual(s2.fallback, "S02")
        self.assertEqual(s2.solver_family, "milp")

    def test_execution_record_round_trip(self):
        r = self.make_record(feasible=False, gap=0.12, status="feasible")
        self.assertEqual(ExecutionRecord.from_dict(r.to_dict()), r)

    def test_strategic_entry_round_trip(self):
        e = StrategicEntry(
            entry_id="se_001", strategy_id="S02",
            pattern={"scope_level": "L2",
                     "predicates": {"resource_coupling": [0.75, 1.0]}},
            expected_quality_hat=0.9, quality_interval=(0.7, 1.0),
            applicability=[ApplicabilityCondition("good here", False, ["ex_1"])],
            risk_conditions=["large scale"], provenance=["ex_1"], support_n=2,
        )
        self.assertEqual(StrategicEntry.from_dict(e.to_dict()), e)

    def test_entry_rejects_bad_scope(self):
        with self.assertRaises(ValueError):
            StrategicEntry.from_dict({
                "entry_id": "se_x", "strategy_id": "S01",
                "pattern": {"scope_level": "L9", "predicates": {}}})

    def test_cold_archive_card_round_trip(self):
        c = ColdArchiveCard(pattern_hash="abc", strategy_id="S01",
                            predicates={"resource_coupling": [0.75, 1.0]},
                            outcome="retired", reason="3 consecutive misses")
        self.assertEqual(ColdArchiveCard.from_dict(c.to_dict()), c)


class TestStructuralGrouping(HarnessTestCase):
    def test_l1_includes_family_l2_l3_do_not(self):
        p = self.make_profile()
        self.assertIn("family=routing", group_key(p, "L1"))
        self.assertIn("family=*", group_key(p, "L2"))
        self.assertIn("family=*", group_key(p, "L3"))

    def test_bin_boundaries(self):
        lo = self.make_profile(resource_coupling=0.1)
        hi = self.make_profile(resource_coupling=0.9)
        self.assertNotEqual(group_key(lo, "L1"), group_key(hi, "L1"))
        same = self.make_profile(resource_coupling=0.95)
        self.assertEqual(group_key(hi, "L1"), group_key(same, "L1"))

    def test_pattern_for_levels(self):
        p = self.make_profile()
        self.assertEqual(pattern_for(p, "L1")["family"], "routing")
        self.assertNotIn("family", pattern_for(p, "L2"))
        l3 = pattern_for(p, "L3")
        for f, (lo, hi) in l3.items():
            self.assertIn(lo, COARSE_BIN_EDGES)
            self.assertIn(hi, COARSE_BIN_EDGES)

    def test_profile_matches(self):
        p = self.make_profile()
        self.assertTrue(profile_matches(p, {"family": "routing"}))
        self.assertFalse(profile_matches(p, {"family": "scheduling"}))
        self.assertTrue(profile_matches(p, {"resource_coupling": [0.75, 1.0]}))
        self.assertFalse(profile_matches(p, {"resource_coupling": [0.0, 0.5]}))
        # Unknown coupling never matches a numeric predicate.
        q = self.make_profile(resource_coupling=None)
        self.assertFalse(profile_matches(q, {"resource_coupling": [0.0, 0.5]}))

    def test_scope_inference(self):
        self.assertEqual(scope_of({"family": "f", "resource_coupling": [0.75, 1.0]}), "L1")
        self.assertEqual(scope_of({"resource_coupling": [0.75, 1.0]}), "L2")
        self.assertEqual(scope_of({"resource_coupling": [0.0, 0.5]}), "L3")

    def test_predicates_cover(self):
        wide = {"resource_coupling": [0.5, 1.0]}
        narrow = {"resource_coupling": [0.75, 1.0]}
        self.assertTrue(predicates_cover(wide, narrow))
        self.assertFalse(predicates_cover(narrow, wide))
        self.assertFalse(predicates_cover({"family": "a"}, {"family": "b"}))

    def test_pattern_hash_stable(self):
        a = pattern_hash({"resource_coupling": [0.75, 1.0]}, "S01")
        b = pattern_hash({"resource_coupling": [0.75, 1.0]}, "S01")
        c = pattern_hash({"resource_coupling": [0.75, 1.0]}, "S02")
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_interval_honesty_floor(self):
        self.assertEqual(min_interval_width(2), 0.50)
        self.assertEqual(min_interval_width(5), 0.15)
        self.assertEqual(min_interval_width(50), 0.0)


if __name__ == "__main__":
    unittest.main()
