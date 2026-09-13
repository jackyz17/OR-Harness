"""Offline admission verification and the publishing gate.

The framework never treats a program's own printed verdict as a verification:
a candidate passes only when a framework-side check holds on real execution
evidence. Failed executions are ``insufficient_evidence`` (not checked), never
``refuted`` (checked and disproven). Unpublished candidates are recorded but
not recommended, and cannot reach a prediction either.
"""
import unittest

from helpers import HarnessTestCase

from or_harness.api import ORHarness
from or_harness.core.schema import COST_DIMENSIONS, CostVector, ExecutionRecord
from or_harness.strategy.verification import (INSUFFICIENT, REFUTED, VERIFIED,
                                              verify_candidate)


def _record(execution_id, *, feasible=True, status="optimal", objective=100.0,
            tokens=100, measured=True, task_id="t1", strategy_id="S01"):
    from or_harness.core.schema import ProblemProfile
    return ExecutionRecord(
        execution_id=execution_id, task_id=task_id, strategy_id=strategy_id,
        profile_snapshot=ProblemProfile(problem_id=task_id, family="routing",
                                        resource_coupling=0.9,
                                        temporal_coupling=0.1,
                                        route_complexity=0.85),
        quality={"feasible": feasible, "objective": objective, "gap": 0.0,
                 "status": status},
        cost=CostVector(llm_tokens=tokens, tool_calls=2.0, solver_runtime_s=1.0,
                        retries=0.0, latency_s=1.0,
                        measured=set(COST_DIMENSIONS) if measured else {"tool_calls"}),
    )


class TestVerificationChecks(HarnessTestCase):
    """The check functions in isolation: each verdict must follow only from
    framework-side, checkable facts."""

    # -- rule claims ---------------------------------------------------------

    def test_rule_verified_on_matching_objective(self):
        report = verify_candidate(
            "rule", "S01 reaches the optimal objective at rc>=0.75",
            check={"reference_objective": 100.0, "tolerance": 1e-6},
            executions=[_record("ex_ok")],
            supporting=[_record("ex_new", task_id="t2")])
        self.assertEqual(report["state"], VERIFIED)
        self.assertEqual(report["purpose"], "rule")
        self.assertEqual(report["claim"],
                         "S01 reaches the optimal objective at rc>=0.75")
        self.assertEqual(report["evidence"], ["ex_ok", "ex_new"])
        self.assertTrue(any(c["check"] == "objective_within_tolerance"
                            for c in report["checks"]))
        self.assertTrue(report["conclusion"])

    def test_rule_refuted_on_wrong_objective(self):
        report = verify_candidate(
            "rule", "c", check={"reference_objective": 100.0},
            executions=[_record("ex_bad", objective=300.0)])
        self.assertEqual(report["state"], REFUTED)

    def test_rule_refuted_on_semantic_check_failure(self):
        report = verify_candidate(
            "rule", "c", check={"semantic_ok": False},
            executions=[_record("ex_sem")])
        self.assertEqual(report["state"], REFUTED)

    def test_program_printed_verdict_is_not_evidence(self):
        """A record claims success but the framework's own check says the
        result is wrong: the candidate is refuted regardless of anything the
        program printed about itself."""
        record = _record("ex_liar")
        record.execution_features["self_report"] = "principle_failed=false"
        report = verify_candidate(
            "rule", "c", check={"reference_objective": 100.0},
            executions=[record])
        record.quality["objective"] = 999.0   # the ACTUAL result disagrees
        report = verify_candidate(
            "rule", "c", check={"reference_objective": 100.0},
            executions=[record])
        self.assertEqual(report["state"], REFUTED)
        self.assertNotIn("principle_failed", str(report["checks"]))

    def test_failed_execution_is_not_a_refutation(self):
        report = verify_candidate(
            "rule", "c", check={"reference_objective": 100.0},
            executions=[_record("ex_err", feasible=False, status="error",
                                objective=None)])
        self.assertEqual(report["state"], INSUFFICIENT)
        self.assertIn("not refuted", report["conclusion"])

    def test_no_executions_is_insufficient(self):
        report = verify_candidate("rule", "c", executions=[])
        self.assertEqual(report["state"], INSUFFICIENT)

    def test_independent_comparison_that_failed_is_not_verified(self):
        report = verify_candidate(
            "rule", "c", check={"reference_objective": 100.0},
            executions=[_record("ex_ok")],
            supporting=[_record("ex_cmp", feasible=False, status="error",
                                objective=None)])
        self.assertEqual(report["state"], INSUFFICIENT)

    # -- repair claims -------------------------------------------------------

    def test_repair_verified_when_fix_succeeds_where_original_failed(self):
        report = verify_candidate(
            "repair", "the fallback fixes the infeasible model",
            executions=[_record("ex_fixed")],
            supporting=[_record("ex_broken", feasible=False, status="infeasible")])
        self.assertEqual(report["state"], VERIFIED)

    def test_repair_without_a_recorded_failure_is_insufficient(self):
        report = verify_candidate("repair", "c",
                                  executions=[_record("ex_ok")], supporting=[])
        self.assertEqual(report["state"], INSUFFICIENT)
        self.assertIn("no recorded failure", report["conclusion"])

    def test_repair_that_still_fails_is_insufficient_not_refuted(self):
        report = verify_candidate(
            "repair", "c", executions=[_record("ex_still", feasible=False,
                                              status="error", objective=None)],
            supporting=[_record("ex_broken", feasible=False, status="infeasible")])
        self.assertEqual(report["state"], INSUFFICIENT)

    # -- cost-saving claims --------------------------------------------------

    def test_cost_saving_verified_when_quality_met_and_cost_lower(self):
        report = verify_candidate(
            "cost_saving", "S02 matches S01's quality for a fifth of the tokens",
            check={"dimension": "llm_tokens", "quality_floor": 0.9},
            executions=[_record("ex_cheap", tokens=200)],
            supporting=[_record("ex_pricey", tokens=1000)])
        self.assertEqual(report["state"], VERIFIED)
        self.assertIn("1000", report["conclusion"])

    def test_cost_saving_refuted_when_cost_is_not_lower(self):
        report = verify_candidate(
            "cost_saving", "c", check={"dimension": "llm_tokens"},
            executions=[_record("ex_same", tokens=1000)],
            supporting=[_record("ex_ref", tokens=1000)])
        self.assertEqual(report["state"], REFUTED)

    def test_cost_saving_refuted_when_quality_below_floor(self):
        """Cheap is not enough: the candidate's own quality must still meet
        the floor (expressed in the same [0, 1] quality space the statistics
        use)."""
        weak = _record("ex_cheap", tokens=200)
        weak.quality["gap"] = 0.9          # quality 0.10
        report = verify_candidate(
            "cost_saving", "c",
            check={"dimension": "llm_tokens", "quality_floor": 0.9},
            executions=[weak],
            supporting=[_record("ex_pricey", tokens=1000)])
        self.assertEqual(report["state"], REFUTED)
        self.assertIn("below the declared floor", report["conclusion"])

    def test_cost_saving_refuted_when_results_not_comparable(self):
        report = verify_candidate(
            "cost_saving", "c", check={"dimension": "llm_tokens"},
            executions=[_record("ex_cheap", tokens=200, objective=10.0)],
            supporting=[_record("ex_pricey", tokens=1000, objective=1000.0)])
        self.assertEqual(report["state"], REFUTED)

    def test_unmeasured_cost_can_never_prove_a_saving(self):
        report = verify_candidate(
            "cost_saving", "c", check={"dimension": "llm_tokens"},
            executions=[_record("ex_cheap", tokens=0, measured=False)],
            supporting=[_record("ex_pricey", tokens=1000)])
        self.assertEqual(report["state"], INSUFFICIENT)
        self.assertIn("unmeasured", report["conclusion"])


class TestPublishingGate(HarnessTestCase):
    def setUp(self):
        super().setUp()
        self.h = ORHarness(home=self.home)
        self.addCleanup(self.h.close)

    def _seed_two_tasks(self, strategy="S01", gap=0.02, tokens=100):
        last = None
        for i in range(2):
            rec = self.make_record(
                execution_id=f"ex_{strategy}_{i}", task_id=f"t{i}",
                strategy_id=strategy, gap=gap,
                cost=CostVector(llm_tokens=tokens, tool_calls=2,
                                solver_runtime_s=1.0, retries=0, latency_s=1.0,
                                measured=set(COST_DIMENSIONS)))
            self.h.bank.append(rec)
            last = rec
        return last

    def test_unverified_candidate_is_not_published(self):
        self._seed_two_tasks()
        result = self.h.induce(strategy_id="S01")["results"][0]
        self.assertIsNotNone(result.get("created"))
        self.assertIn("not published", result["skipped"])
        entry = self.h.sbank.get(result["created"])
        self.assertEqual(entry.verification_state, "unverified")
        # Recall does not present it as strategic knowledge...
        recs = self.h.selector.recall(self.make_profile(problem_id="q"), top=5)
        s01 = next(r for r in recs if r.strategy.strategy_id == "S01")
        self.assertEqual(s01.evidence, "conditional_stats")
        # ...and it cannot drive a prediction either (the second door): the
        # snapshot falls back down the provenance ladder instead of quoting
        # the unpublished entry.
        snapshot = self.h.predict_cost({"task_id": "q", "family": "routing"},
                                       "S01")
        self.assertNotEqual(snapshot.source, "entry")
        self.assertNotIn(entry.entry_id, snapshot.evidence_refs)

    def test_verified_candidate_is_published(self):
        self._seed_two_tasks()
        verify = {"purpose": "rule", "claim": "S01 holds in this cell",
                  "check": {"reference_objective": 100.0},
                  "executions": [self.make_record(execution_id="ex_v",
                                                  task_id="t_verify")]}
        result = self.h.induce(strategy_id="S01", verify=verify)["results"][0]
        entry = self.h.sbank.get(result["created"])
        self.assertEqual(entry.verification_state, "verified")
        self.assertNotIn("skipped", result)
        recs = self.h.selector.recall(self.make_profile(problem_id="q"), top=5)
        s01 = next(r for r in recs if r.strategy.strategy_id == "S01")
        self.assertEqual(s01.evidence, "strategic_entry")

    def test_refuted_candidate_is_not_published(self):
        self._seed_two_tasks()
        verify = {"purpose": "rule", "claim": "S01 reaches objective 100",
                  "check": {"reference_objective": 100.0},
                  "executions": [self.make_record(execution_id="ex_v",
                                                  task_id="t_verify",
                                                  objective=999.0)]}
        result = self.h.induce(strategy_id="S01", verify=verify)["results"][0]
        entry = self.h.sbank.get(result["created"])
        self.assertEqual(entry.verification_state, "refuted")
        recs = self.h.selector.recall(self.make_profile(problem_id="q"), top=5)
        s01 = next(r for r in recs if r.strategy.strategy_id == "S01")
        self.assertEqual(s01.evidence, "conditional_stats")

    def test_insufficient_evidence_does_not_masquerade_as_verified(self):
        self._seed_two_tasks()
        verify = {"purpose": "rule", "claim": "S01 holds elsewhere",
                  "check": {"reference_objective": 100.0},
                  "executions": [self.make_record(execution_id="ex_v",
                                                  task_id="t_verify",
                                                  feasible=False,
                                                  status="error",
                                                  objective=None)]}
        result = self.h.induce(strategy_id="S01", verify=verify)["results"][0]
        entry = self.h.sbank.get(result["created"])
        self.assertEqual(entry.verification_state, "insufficient_evidence")
        self.assertFalse(entry.is_published)

    def test_offline_view_can_still_return_candidates(self):
        self._seed_two_tasks()
        entry_id = self.h.induce(strategy_id="S01")["results"][0]["created"]
        recs = self.h.selector.recall(self.make_profile(problem_id="q"), top=5,
                                      include_unverified=True)
        s01 = next(r for r in recs if r.strategy.strategy_id == "S01")
        self.assertEqual(s01.evidence, "strategic_entry")
        self.assertIn(entry_id, s01.evidence_refs)

    def test_reinducing_without_a_verdict_keeps_the_verdict(self):
        """Refreshing statistics must not silently demote a verified claim."""
        self._seed_two_tasks()
        verify = {"purpose": "rule", "claim": "c",
                  "check": {"reference_objective": 100.0},
                  "executions": [self.make_record(execution_id="ex_v",
                                                  task_id="t_verify")]}
        entry_id = self.h.induce(strategy_id="S01", verify=verify)["results"][0]["created"]
        self.h.bank.append(self.make_record(execution_id="ex_more", task_id="t3",
                                            strategy_id="S01", gap=0.05))
        self.h.induce(strategy_id="S01")
        self.assertEqual(self.h.sbank.get(entry_id).verification_state, "verified")

    def test_no_verified_knowledge_still_offers_the_catalog(self):
        """Gating publishing must not stop the harness from trying."""
        recs = self.h.selector.recall(self.make_profile(problem_id="q"), top=3)
        self.assertTrue(recs)
        self.assertTrue(all(r.evidence in ("no_memory", "conditional_stats")
                            for r in recs))


if __name__ == "__main__":
    unittest.main()
