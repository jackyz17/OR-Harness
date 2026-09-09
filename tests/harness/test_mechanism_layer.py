"""Tests for the mechanism layer: derivation, kinship matching, entry
annotation, cost prediction loop, failure classification persistence."""
import textwrap
import unittest

from helpers import HarnessTestCase

from or_harness.core.schema import (
    CostVector,
    ExecutionRecord,
    FailureRecord,
    ProblemProfile,
    StrategicEntry,
)
from or_harness.profiling.model_syntax import mechanisms_from_model, parse_model
from or_harness.profiling.profiler import profile_task
from or_harness.strategy.experience_bank import ExperienceBank
from or_harness.strategy.induction import InductionEngine, PatternReflowEngine
from or_harness.strategy.stats import ConditionalStats
from or_harness.strategy.strategic_bank import StrategicBank
from or_harness.strategy.catalog import load_catalog
from or_harness.strategy.selector import Selector
from or_harness.api import ORHarness


SHARED_RESOURCE_MODEL = textwrap.dedent("""
    SETS:
     i in Projects = {p0, p1, p2}
     j in Resources = {r0, r1}
    PARAMETERS:
     E[i,j]
     limit[j]
    VARIABLES:
     x[i,j] continuous >= 0
    OBJECTIVE:
     maximize sum(i, sum(j, E[i,j] * x[i,j]))
    CONSTRAINTS:
     C1: sum(i, x[i,j]) <= limit[j]
     C2: sum(j, x[i,j]) <= 100
     C3: sum(i, sum(j, x[i,j])) <= 250
""")

DISCRETE_MODEL = textwrap.dedent("""
    SETS:
     i in Machines = {m0, m1, m2, m3}
     t in Periods = {1, 2, 3, 4}
    PARAMETERS:
     cost[i,t]
     demand[t]
    VARIABLES:
     y[i,t] binary
     x[i,t] continuous >= 0
    OBJECTIVE:
     minimize sum(t, sum(i, cost[i,t] * (y[i,t] + x[i,t])))
    CONSTRAINTS:
     C1: sum(i, x[i,t]) >= demand[t]
     C2: x[i,t] <= 100 * y[i,t]
""")


class TestMechanismDerivation(HarnessTestCase):
    def test_shared_resource_competition_detected(self):
        mech = mechanisms_from_model(parse_model(SHARED_RESOURCE_MODEL))
        self.assertGreater(mech["shared_resource_competition"], 0.0)
        # C3 spans all variables -> global propagation.
        self.assertEqual(mech["global_constraint_propagation"], 1.0)

    def test_discrete_shrinkage_detected(self):
        mech = mechanisms_from_model(parse_model(DISCRETE_MODEL))
        # half the variables are binary
        self.assertAlmostEqual(mech["discrete_feasibility_shrinkage"], 0.5)

    def test_temporal_propagation_detected(self):
        linking = textwrap.dedent("""
            SETS:
             t in Periods = {1, 2, 3}
             i in Items = {a, b}
            PARAMETERS:
             cap[t]
            VARIABLES:
             x[i,t] continuous >= 0
            OBJECTIVE:
             minimize sum(t, sum(i, x[i,t]))
            CONSTRAINTS:
             C1: x[i,t] + x[i,t+1] <= cap[t]
             C2: sum(i, x[i,t]) >= 1
        """)
        mech = mechanisms_from_model(parse_model(linking))
        self.assertGreater(mech["temporal_propagation"], 0.0)

    def test_sparse_model_returns_empty(self):
        self.assertEqual(mechanisms_from_model(parse_model("SETS:\n a in A = {1}")), {})

    def test_profile_carries_mechanisms(self):
        task = {"task_id": "t1", "family": "allocation",
                "model": SHARED_RESOURCE_MODEL}
        profile = profile_task(task)
        self.assertIn("shared_resource_competition", profile.mechanism_features)
        # No model -> no mechanisms (never fabricated).
        bare = profile_task({"task_id": "t2", "family": "allocation", "spec": {}})
        self.assertEqual(bare.mechanism_features, {})

    def test_old_records_without_mechanisms_parse(self):
        # Backward compatibility: pre-mechanism JSON round-trips.
        legacy = {
            "problem_id": "t3", "family": "routing",
            "scale_features": {}, "semantic_coupling": 0.5,
            "resource_coupling": 0.8, "temporal_coupling": None,
            "route_complexity": None, "risk_features": {},
            "source": "derived", "annotations": {},
        }
        profile = ProblemProfile.from_dict(legacy)
        self.assertEqual(profile.mechanism_features, {})
        self.assertEqual(profile.resource_coupling, 0.8)


class TestMechanismKinship(HarnessTestCase):
    def setUp(self):
        super().setUp()
        self.bank = ExperienceBank(self.store)
        self.sbank = StrategicBank(self.store)
        self.stats = ConditionalStats(self.bank)
        self.catalog = load_catalog()
        self.selector = Selector(self.catalog, self.sbank, self.stats)
        self.engine = InductionEngine(self.stats, self.sbank)

    def _seed_and_induce(self, family, strategy_id="S02", n=3):
        mech = {"shared_resource_competition": 0.9,
                "global_constraint_propagation": 0.5,
                "temporal_propagation": 0.0,
                "discrete_feasibility_shrinkage": 0.0}
        for i in range(n):
            profile = self.make_profile(problem_id=f"{family}{i}", family=family)
            profile.mechanism_features = dict(mech)
            self.bank.append(self.make_record(
                execution_id=f"ex_{family}_{i}", task_id=f"{family}{i}",
                strategy_id=strategy_id, profile=profile, gap=0.05))
        result = self.engine.induce(self.make_profile(problem_id="q", family=family),
                                    strategy_id)
        return result

    def test_induce_annotates_mechanism(self):
        result = self._seed_and_induce("routing")
        entry = self.sbank.get(result["created"])
        self.assertAlmostEqual(
            entry.mechanism.features["shared_resource_competition"], 0.9)

    def test_first_contact_kinship_match(self):
        """The core scenario: routing-learned entry matches a scheduling
        problem at FIRST CONTACT via mechanism kinship — no exploration
        tuition in the target family required."""
        result = self._seed_and_induce("routing")
        self.assertIsNotNone(result.get("created"))
        # A scheduling problem with the SAME mechanism signature.
        scheduling = self.make_profile(problem_id="s1", family="scheduling")
        scheduling.mechanism_features = {
            "shared_resource_competition": 0.85,
            "global_constraint_propagation": 0.55,
            "temporal_propagation": 0.0,
            "discrete_feasibility_shrinkage": 0.0}
        recs = self.selector.recommend(scheduling, top=10)
        s02 = next((r for r in recs if r.strategy.strategy_id == "S02"), None)
        self.assertIsNotNone(s02)
        self.assertEqual(s02.evidence, "mechanism_entry")
        self.assertTrue(s02.cross_family)
        self.assertLess(s02.confidence, 0.7)  # discounted
        self.assertTrue(any("mechanism" in w for w in s02.risk_warnings))

    def test_different_mechanism_no_match(self):
        result = self._seed_and_induce("routing")
        # A problem with a DIFFERENT mechanism signature: no kinship.
        other = self.make_profile(problem_id="s2", family="scheduling")
        other.mechanism_features = {
            "temporal_propagation": 0.9,
            "discrete_feasibility_shrinkage": 0.8,
            "shared_resource_competition": 0.0,
            "global_constraint_propagation": 0.0}
        recs = self.selector.recommend(other, top=10)
        s02 = next((r for r in recs if r.strategy.strategy_id == "S02"), None)
        if s02 is not None:
            self.assertNotEqual(s02.evidence, "mechanism_entry")

    def test_mechanism_statistics_fallback(self):
        """Without an entry, cross-family mechanism statistics still inform
        the recommendation (kinship at the statistics level)."""
        # Seed facts but do NOT induce — no entry exists, so the statistics
        # path is what should fire.
        mech = {"shared_resource_competition": 0.9,
                "global_constraint_propagation": 0.5,
                "temporal_propagation": 0.0,
                "discrete_feasibility_shrinkage": 0.0}
        for i in range(3):
            profile = self.make_profile(problem_id=f"r{i}", family="routing")
            profile.mechanism_features = dict(mech)
            self.bank.append(self.make_record(
                execution_id=f"ex_ms_{i}", task_id=f"ms{i}",
                strategy_id="S04", profile=profile, gap=0.05))
        scheduling = self.make_profile(problem_id="s3", family="scheduling")
        scheduling.mechanism_features = {
            "shared_resource_competition": 0.9,
            "global_constraint_propagation": 0.5,
            "temporal_propagation": 0.0,
            "discrete_feasibility_shrinkage": 0.0}
        recs = self.selector.recommend(scheduling, top=10)
        s04 = next((r for r in recs if r.strategy.strategy_id == "S04"), None)
        self.assertIsNotNone(s04)
        self.assertEqual(s04.evidence, "conditional_stats")
        self.assertTrue(s04.cross_family)
        self.assertIn("mechanism", s04.basis)

    def test_structural_match_beats_mechanism_match(self):
        """A structural (L1) match keeps full confidence; mechanism kinship
        only fills the gap when structure doesn't match."""
        self._seed_and_induce("routing")
        routing_query = self.make_profile(problem_id="q2", family="routing")
        routing_query.mechanism_features = {
            "shared_resource_competition": 0.9,
            "global_constraint_propagation": 0.5,
            "temporal_propagation": 0.0,
            "discrete_feasibility_shrinkage": 0.0}
        recs = self.selector.recommend(routing_query, top=10)
        s02 = next(r for r in recs if r.strategy.strategy_id == "S02")
        self.assertEqual(s02.evidence, "strategic_entry")  # structural path wins
        self.assertFalse(s02.cross_family)


class TestCostPredictionLoop(HarnessTestCase):
    def setUp(self):
        super().setUp()
        self.h = ORHarness(home=self.home)

    def _entry_with_cost(self, tokens_hat=1500.0):
        self.h.bank.append(self.make_record(
            execution_id="ex_c1", task_id="tc1", strategy_id="S01",
            cost=CostVector(llm_tokens=tokens_hat, tool_calls=2,
                            solver_runtime_s=1.0, retries=0, latency_s=1.0)))
        self.h.bank.append(self.make_record(
            execution_id="ex_c2", task_id="tc2", strategy_id="S01",
            cost=CostVector(llm_tokens=tokens_hat, tool_calls=2,
                            solver_runtime_s=1.0, retries=0, latency_s=1.0)))
        result = self.h.induce(strategy_id="S01")
        return result["results"][0]["created"]

    def test_cost_hit_recorded(self):
        entry_id = self._entry_with_cost(tokens_hat=1500.0)
        entry = self.h.sbank.get(entry_id)
        self.assertIn("llm_tokens", entry.cost_interval)
        # An execution within [0.5x, 2x] of the prediction.
        rec = self.make_record(task_id="tc3", strategy_id="S01",
                               cost=CostVector(llm_tokens=1800, tool_calls=2,
                                               solver_runtime_s=1.0, retries=0,
                                               latency_s=1.0))
        outcome = self.h.record(rec)
        checks = outcome["prediction_checks"]
        self.assertTrue(checks)
        cost_check = checks[0].get("cost_check")
        self.assertIsNotNone(cost_check)
        self.assertTrue(cost_check["hit"])
        entry = self.h.sbank.get(entry_id)
        self.assertEqual(entry.prediction_track.n_cost_predictions, 1)
        self.assertEqual(entry.prediction_track.n_cost_hits, 1)

    def test_cost_miss_warns_never_demotes(self):
        entry_id = self._entry_with_cost(tokens_hat=1500.0)
        # Three executions far outside the band (10x the prediction).
        for i in range(3):
            rec = self.make_record(task_id=f"tc{i + 10}", strategy_id="S01",
                                   cost=CostVector(llm_tokens=15000,
                                                   tool_calls=2,
                                                   solver_runtime_s=1.0,
                                                   retries=0, latency_s=1.0))
            self.h.record(rec)
        entry = self.h.sbank.get(entry_id)
        # Cost misses accumulated...
        self.assertEqual(entry.prediction_track.n_cost_predictions, 3)
        self.assertEqual(entry.prediction_track.n_cost_hits, 0)
        # ...but the entry is NOT demoted (quality-side semantics untouched).
        self.assertEqual(entry.status, "candidate")
        self.assertEqual(entry.prediction_track.consecutive_misses, 0)
        # And the selector warns about the uncalibrated cost estimate.
        recs = self.h.selector.recommend(self.make_profile(problem_id="q"), top=10)
        s01 = next(r for r in recs if r.strategy.strategy_id == "S01")
        self.assertTrue(any("uncalibrated" in w for w in s01.risk_warnings))

    def test_quality_miss_still_demotes(self):
        """The quality loop is untouched by the cost loop."""
        entry_id = self._entry_with_cost(tokens_hat=1500.0)
        for i in range(3):
            rec = self.make_record(task_id=f"tq{i}", strategy_id="S01", gap=0.9,
                                   cost=CostVector(llm_tokens=1500))
            self.h.record(rec)
        entry = self.h.sbank.get(entry_id)
        self.assertEqual(entry.status, "suspect")


class TestFailureClassificationPersistence(HarnessTestCase):
    def test_error_class_persisted_at_record(self):
        h = ORHarness(home=self.home)
        rec = self.make_record(
            execution_id="ex_fc", task_id="tfc", feasible=False, status="error",
            failures=[FailureRecord(1, "security policy: blocked import subprocess")])
        h.record(rec)
        stored = h.bank.get("ex_fc")
        self.assertEqual(stored.failures[0].error_class, "environment")
        h.close()

    def test_model_error_classified_as_model(self):
        h = ORHarness(home=self.home)
        rec = self.make_record(
            execution_id="ex_fm", task_id="tfm", feasible=False, status="error",
            failures=[FailureRecord(1, "ValueError: bad shape")])
        h.record(rec)
        stored = h.bank.get("ex_fm")
        self.assertEqual(stored.failures[0].error_class, "model")
        h.close()


class TestReflowStub(HarnessTestCase):
    def test_stub_is_noop(self):
        bank = ExperienceBank(self.store)
        sbank = StrategicBank(self.store)
        stats = ConditionalStats(bank)
        engine = PatternReflowEngine(stats, sbank)
        self.assertEqual(engine.propose_reflow(["se_anything"]), [])


class TestHierarchyFields(HarnessTestCase):
    def test_hierarchy_fields_round_trip(self):
        entry = StrategicEntry(
            entry_id="se_h", strategy_id="S01",
            pattern={"scope_level": "L1", "predicates": {"family": "routing"}},
            parent_pattern_id="se_parent", mechanism_id="mech_src")
        data = entry.to_dict()
        self.assertEqual(data["parent_pattern_id"], "se_parent")
        self.assertEqual(data["mechanism_id"], "mech_src")
        parsed = StrategicEntry.from_dict(data)
        self.assertEqual(parsed.parent_pattern_id, "se_parent")
        self.assertEqual(parsed.mechanism_id, "mech_src")


if __name__ == "__main__":
    unittest.main()
