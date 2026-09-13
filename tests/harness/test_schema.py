"""Serialization round-trip and structural-grouping tests for core schemas."""
import unittest

from helpers import HarnessTestCase

from or_harness.core.schema import (
    CostVector,
    ColdArchiveCard,
    ExecutionRecord,
    ProblemProfile,
    StrategicEntry,
    Strategy,
    evidence_predicates,
    group_key,
    min_interval_width,
    pattern_hash,
    predicates_cover,
    profile_matches,
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

    def test_strategy_type_round_trip(self):
        s = Strategy(strategy_id="S01", name="monolithic",
                     strategy_type="modeling")
        self.assertEqual(Strategy.from_dict(s.to_dict()).strategy_type, "modeling")
        # Old catalog payloads without the field load with None.
        self.assertIsNone(Strategy.from_dict(
            {"strategy_id": "S02", "name": "x"}).strategy_type)

    def test_execution_record_actual_aliases(self):
        r = self.make_record(feasible=False, gap=0.12, status="feasible")
        # Facts store ACTUAL observations; aliases make this explicit.
        self.assertIs(r.actual_quality, r.quality)
        self.assertIs(r.actual_cost, r.cost)

    def test_execution_record_cir_snapshot_round_trip(self):
        cir = {"entities": [{"name": "R1", "kind": "resource", "attrs": {}}],
               "decisions": [{"name": "x", "kind": "production",
                              "indexes": ["i"], "attrs": {}}],
               "constraints": [], "relations": [], "coupling_groups": []}
        r = self.make_record()
        r.cir_snapshot = cir
        r2 = ExecutionRecord.from_dict(r.to_dict())
        self.assertEqual(r2.cir_snapshot, cir)
        # Records without a CIR round-trip to None (backward compatible).
        r3 = ExecutionRecord.from_dict(self.make_record().to_dict())
        self.assertIsNone(r3.cir_snapshot)

    def test_execution_record_round_trip(self):
        r = self.make_record(feasible=False, gap=0.12, status="feasible")
        self.assertEqual(ExecutionRecord.from_dict(r.to_dict()), r)

    def test_strategic_entry_round_trip(self):
        e = StrategicEntry(
            entry_id="se_001", strategy_id="S02",
            pattern={"predicates": {"family": "routing",
                                    "resource_coupling": [0.75, 1.0]}},
            expected_quality_hat=0.9, quality_interval=(0.7, 1.0),
            applicability=["good here"], risk_conditions=["large scale"],
            provenance=["ex_1"], support_n=2,
        )
        self.assertEqual(StrategicEntry.from_dict(e.to_dict()), e)

    def test_legacy_condition_objects_load_as_notes(self):
        """Payloads written when applicability was a list of condition
        objects keep their text; the never-populated verification flags are
        dropped."""
        entry = StrategicEntry.from_dict({
            "entry_id": "se_legacy", "strategy_id": "S01",
            "pattern": {"predicates": {}},
            "applicability": [{"text": "works at scale", "verified": False,
                               "supporting_execution_ids": ["ex_1"]},
                              "plain note"]})
        self.assertEqual(entry.applicability, ["works at scale", "plain note"])
        self.assertEqual(entry.verification, {})

    def test_strategic_entry_expected_aliases(self):
        e = StrategicEntry(
            entry_id="se_001", strategy_id="S02",
            pattern={"predicates": {}},
            expected_quality_hat=0.8, failure_prob=0.3,
            expected_cost_hat=CostVector(llm_tokens=5))
        # Knowledge stores EXPECTED quantities; aliases make this explicit.
        self.assertEqual(e.expected_quality, 0.8)
        self.assertEqual(e.expected_cost, e.expected_cost_hat)
        self.assertEqual(e.expected_failure_risk, 0.3)

    def test_strategic_entry_extension_fields_round_trip(self):
        e = StrategicEntry(
            entry_id="se_002", strategy_id="S04",
            pattern={"predicates": {}},
            strategy_type="decomposition",
            actions=["identify bottleneck", "decompose locals"],
            provenance=["ex_1"], support_n=2)
        e2 = StrategicEntry.from_dict(e.to_dict())
        self.assertEqual(e2.strategy_type, "decomposition")
        self.assertEqual(e2.actions, e.actions)
        # Old payloads without the extension fields load with defaults.
        e3 = StrategicEntry.from_dict({
            "entry_id": "se_003", "strategy_id": "S01",
            "pattern": {"predicates": {}}})
        self.assertIsNone(e3.strategy_type)
        self.assertEqual(e3.actions, [])

    def test_cold_archive_card_round_trip(self):
        c = ColdArchiveCard(pattern_hash="abc", strategy_id="S01",
                            predicates={"resource_coupling": [0.75, 1.0]},
                            outcome="retired", reason="3 consecutive misses")
        self.assertEqual(ColdArchiveCard.from_dict(c.to_dict()), c)


class TestStructuralGrouping(HarnessTestCase):
    def test_group_key_is_the_family_plus_cell(self):
        """One evidence set = one (family, structural cell, strategy) triple.
        Structure conditions the aggregation; semantic_coupling never does."""
        p = self.make_profile()        # rc 0.9, tc 0.1, rx 0.85
        self.assertEqual(group_key(p),
                         "family=routing|rc[0.75,1.00]|tc[0.00,0.25]|rx[0.75,1.00]")
        # A structurally different region is a DIFFERENT evidence set: pooling
        # them once averaged a Q=1.0 region and a Q=0.1 region into one claim.
        self.assertNotEqual(group_key(p),
                            group_key(self.make_profile(resource_coupling=0.1)))
        self.assertNotEqual(group_key(p),
                            group_key(self.make_profile(family="scheduling")))
        # Same cell wherever the (never-derived) semantic value points.
        self.assertEqual(group_key(p),
                         group_key(self.make_profile(semantic_coupling=0.2)))

    def test_unknown_is_its_own_cell(self):
        self.assertEqual(group_key(self.make_profile(resource_coupling=None)),
                         "family=routing|rc[unknown]|tc[0.00,0.25]|rx[0.75,1.00]")
        self.assertNotEqual(group_key(self.make_profile(resource_coupling=None)),
                            group_key(self.make_profile(resource_coupling=0.9)))

    def test_evidence_predicates_are_the_cell(self):
        """Predicates are the structural CELL the evidence occupies — not a
        cross-sample min/max, which could stretch across incomparable
        regions just because samples sat at both ends."""
        records = [
            self.make_record(execution_id="ex_a", task_id="a",
                             profile=self.make_profile(problem_id="a",
                                                       resource_coupling=0.80)),
            self.make_record(execution_id="ex_b", task_id="b",
                             profile=self.make_profile(problem_id="b",
                                                       resource_coupling=0.94)),
        ]
        predicates = evidence_predicates(records)
        self.assertEqual(predicates["family"], "routing")
        self.assertEqual(predicates["resource_coupling"], [0.75, 1.0])
        inside = self.make_profile(problem_id="q", resource_coupling=0.80)
        outside = self.make_profile(problem_id="q2", resource_coupling=0.55)
        self.assertTrue(profile_matches(inside, predicates))
        self.assertFalse(profile_matches(outside, predicates))

    def test_predicates_keep_full_precision(self):
        """Rounding predicates to 4 places once made a claim fail to match
        the very executions supporting it (rc=1/3 -> [0.3333, 0.3333])."""
        third = 1.0 / 3.0
        records = [self.make_record(execution_id="ex_third", task_id="t1",
                                    profile=self.make_profile(
                                        problem_id="t1", resource_coupling=third,
                                        temporal_coupling=None,
                                        route_complexity=None))]
        predicates = evidence_predicates(records)
        # The cell label is the interval; the boundary values are exact.
        self.assertEqual(predicates["resource_coupling"], [0.25, 0.5])
        self.assertTrue(profile_matches(
            self.make_profile(problem_id="q", resource_coupling=third,
                              temporal_coupling=None, route_complexity=None),
            predicates))

    def test_unknown_predicate_matches_only_unknown(self):
        """'Both sides unknown' is a shared absence of evidence, not evidence
        of similarity — an unknown-cell claim must not be applied to a task
        whose structure IS known (nor the reverse)."""
        records = [self.make_record(
            execution_id="ex_a", task_id="a",
            profile=self.make_profile(problem_id="a", resource_coupling=None))]
        predicates = evidence_predicates(records)
        self.assertEqual(predicates["resource_coupling"], "[unknown]")
        self.assertTrue(profile_matches(
            self.make_profile(problem_id="q", resource_coupling=None), predicates))
        self.assertFalse(profile_matches(
            self.make_profile(problem_id="q2", resource_coupling=0.9), predicates))

    def test_evidence_predicates_empty_without_records(self):
        self.assertEqual(evidence_predicates([]), {})

    def test_profile_matches(self):
        p = self.make_profile()
        self.assertTrue(profile_matches(p, {"family": "routing"}))
        self.assertFalse(profile_matches(p, {"family": "scheduling"}))
        self.assertTrue(profile_matches(p, {"resource_coupling": [0.75, 1.0]}))
        self.assertFalse(profile_matches(p, {"resource_coupling": [0.0, 0.5]}))
        # Unknown coupling never matches a numeric predicate.
        q = self.make_profile(resource_coupling=None)
        self.assertFalse(profile_matches(q, {"resource_coupling": [0.0, 0.5]}))

    def test_predicates_cover_with_unknown(self):
        self.assertTrue(predicates_cover({"family": "routing"},
                                         {"family": "routing"}))
        self.assertTrue(predicates_cover({"resource_coupling": "[unknown]"},
                                         {"resource_coupling": "[unknown]"}))
        self.assertFalse(predicates_cover({"resource_coupling": "[unknown]"},
                                          {"resource_coupling": [0.0, 1.0]}))
        self.assertFalse(predicates_cover({"resource_coupling": [0.0, 1.0]},
                                          {"resource_coupling": "[unknown]"}))

    def test_semantic_coupling_never_enters_the_evidence(self):
        """The one dimension that is never derived must not split the
        evidence or appear in a claim."""
        a = self.make_profile(semantic_coupling=0.8)
        b = self.make_profile(semantic_coupling=0.2)
        self.assertEqual(group_key(a), group_key(b))
        records = [self.make_record(execution_id="ex_a", task_id="a", profile=a),
                   self.make_record(execution_id="ex_b", task_id="b", profile=b)]
        self.assertNotIn("semantic_coupling", evidence_predicates(records))
