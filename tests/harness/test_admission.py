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
            tokens=100, measured=True, task_id="t1", strategy_id="S01",
            family="routing"):
    from or_harness.core.schema import ProblemProfile
    return ExecutionRecord(
        execution_id=execution_id, task_id=task_id, strategy_id=strategy_id,
        profile_snapshot=ProblemProfile(problem_id=task_id, family=family,
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

    # -- the payload must state something checkable --------------------------

    def test_feasibility_alone_is_not_a_check(self):
        """The reproduced defect: ``{"quality": {"feasible": true}}`` with no
        execution id, objective or declared criterion returned `verified`."""
        report = verify_candidate("rule", "c", executions=[
            {"quality": {"feasible": True}}])
        self.assertEqual(report["state"], INSUFFICIENT)
        self.assertIn("execution_id", report["conclusion"])

    def test_identified_record_without_a_basis_is_insufficient(self):
        report = verify_candidate("rule", "c", executions=[_record("ex_ok")])
        self.assertEqual(report["state"], INSUFFICIENT)
        self.assertIn("no check basis", report["conclusion"])

    def test_agent_declared_boolean_alone_cannot_verify(self):
        """A bare boolean the harness asserts is recorded and labelled, but
        the framework cannot re-derive it, so it does not carry a verdict."""
        report = verify_candidate("rule", "c", check={"semantic_ok": True},
                                  executions=[_record("ex_ok")])
        self.assertEqual(report["state"], INSUFFICIENT)
        declared = [c for c in report["checks"]
                    if c["check"] == "problem_semantic_check"]
        self.assertEqual(declared[0]["source"], "agent-declared")

    def test_framework_probe_verifies_a_semantic_check(self):
        """The framework-side counterpart: it reads a real observed value."""
        report = verify_candidate(
            "rule", "c",
            check={"semantic_probe": {"path": "quality.objective",
                                      "max": 200.0}},
            executions=[_record("ex_ok")])
        self.assertEqual(report["state"], VERIFIED)
        probe = [c for c in report["checks"] if c["check"] == "semantic_probe"]
        self.assertEqual(probe[0]["source"], "framework")
        self.assertTrue(probe[0]["ok"])

    def test_framework_probe_can_refute(self):
        report = verify_candidate(
            "rule", "c",
            check={"semantic_probe": {"path": "quality.objective", "max": 50.0}},
            executions=[_record("ex_ok")])
        self.assertEqual(report["state"], REFUTED)

    def test_status_can_be_the_declared_basis(self):
        report = verify_candidate("rule", "c",
                                  check={"reference_status": "optimal"},
                                  executions=[_record("ex_ok")])
        self.assertEqual(report["state"], VERIFIED)
        report = verify_candidate("rule", "c",
                                  check={"reference_status": "infeasible"},
                                  executions=[_record("ex_ok")])
        self.assertEqual(report["state"], REFUTED)

    # -- every record is checked ---------------------------------------------

    def test_counterexample_anywhere_in_the_batch_refutes(self):
        """The reproduced defect: only the first usable record was evaluated,
        so put the good one first and the claim passed."""
        good = _record("ex_good")
        bad = _record("ex_bad", objective=999.0)
        for order in ([good, bad], [bad, good]):
            report = verify_candidate("rule", "c",
                                      check={"reference_objective": 100.0},
                                      executions=order)
            self.assertEqual(report["state"], REFUTED,
                             "record order must not decide the verdict")
            self.assertIn("ex_bad", report["conclusion"])

    # -- the comparison set must be independent ------------------------------

    def test_repeated_execution_is_not_an_independent_comparison(self):
        same = _record("ex_same")
        report = verify_candidate("rule", "c",
                                  check={"reference_objective": 100.0},
                                  executions=[same], supporting=[same])
        self.assertEqual(report["state"], INSUFFICIENT)
        self.assertIn("more than once", report["conclusion"])

    def test_comparison_on_the_inducing_tasks_is_not_independent(self):
        report = verify_candidate(
            "rule", "c", check={"reference_objective": 100.0},
            executions=[_record("ex_a", task_id="t1")],
            supporting=[_record("ex_b", task_id="t1")])
        self.assertEqual(report["state"], INSUFFICIENT)
        self.assertIn("not an independent check", report["conclusion"])

    def test_comparison_on_a_new_task_verifies(self):
        report = verify_candidate(
            "rule", "c", check={"reference_objective": 100.0},
            executions=[_record("ex_a", task_id="t1")],
            supporting=[_record("ex_b", task_id="t2")])
        self.assertEqual(report["state"], VERIFIED)

    # -- evidence must correspond to the candidate ---------------------------

    def test_evidence_for_another_strategy_does_not_verify(self):
        """The reproduced defect: an S04/scheduling execution verified an
        S01/routing candidate."""
        report = verify_candidate(
            "rule", "c", check={"reference_objective": 100.0},
            executions=[_record("ex_s04", strategy_id="S04")],
            strategy_id="S01", family="routing")
        self.assertEqual(report["state"], INSUFFICIENT)
        self.assertIn("does not correspond", report["conclusion"])
        self.assertIn("S04", report["conclusion"])

    def test_evidence_for_another_family_does_not_verify(self):
        report = verify_candidate(
            "rule", "c", check={"reference_objective": 100.0},
            executions=[_record("ex_sch", family="scheduling")],
            strategy_id="S01", family="routing")
        self.assertEqual(report["state"], INSUFFICIENT)
        self.assertIn("scheduling", report["conclusion"])

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
        """A repair needs a failure to repair: a success with nothing failing
        is not a repair, it is an ordinary successful run."""
        report = verify_candidate(
            "repair", "c", executions=[_record("ex_ok")],
            supporting=[_record("ex_also_ok", task_id="t1")])
        self.assertEqual(report["state"], INSUFFICIENT)
        self.assertIn("no recorded failure", report["conclusion"])

    def test_repair_across_different_tasks_is_not_a_repair(self):
        """The reproduced defect: a failure on one task plus a success on an
        unrelated task was read as a working repair."""
        report = verify_candidate(
            "repair", "c",
            executions=[_record("ex_other_ok", task_id="other_task")],
            supporting=[_record("ex_broken", task_id="task_a",
                                feasible=False, status="infeasible")])
        self.assertEqual(report["state"], INSUFFICIENT)
        self.assertIn("different tasks", report["conclusion"])

    def test_repair_without_both_sides_is_insufficient(self):
        report = verify_candidate("repair", "c",
                                  executions=[_record("ex_ok")], supporting=[])
        self.assertEqual(report["state"], INSUFFICIENT)

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

    def test_cost_saving_across_different_tasks_is_not_comparable(self):
        report = verify_candidate(
            "cost_saving", "c", check={"dimension": "llm_tokens"},
            executions=[_record("ex_cheap", tokens=100, task_id="t_new")],
            supporting=[_record("ex_pricey", tokens=1000, task_id="t_base")])
        self.assertEqual(report["state"], INSUFFICIENT)
        self.assertIn("different tasks", report["conclusion"])

    def test_attempt_cost_is_not_compared_against_a_task_total(self):
        """The reproduced defect: an attempt cost was compared against a
        task-scope total and reported as a saving — different quantities."""
        cheap = _record("ex_attempt", tokens=100, task_id="t1")
        cheap.measurement_scope = "attempt"
        total = _record("ex_task_total", tokens=1000, task_id="t1")
        total.measurement_scope = "task"
        report = verify_candidate(
            "cost_saving", "c", check={"dimension": "llm_tokens"},
            executions=[cheap], supporting=[total])
        self.assertEqual(report["state"], INSUFFICIENT)
        self.assertIn("different scopes", report["conclusion"])

    def test_same_execution_on_both_sides_is_insufficient(self):
        record = _record("ex_same", tokens=100)
        report = verify_candidate(
            "cost_saving", "c", check={"dimension": "llm_tokens"},
            executions=[record], supporting=[record])
        self.assertEqual(report["state"], INSUFFICIENT)
        self.assertIn("more than once", report["conclusion"])

    def test_json_and_object_inputs_agree(self):
        """The reproduced defect: the same evidence verified through the
        Python API but reported "the cost dimension could not be read" through
        the CLI, because JSON costs arrive as dicts."""
        obj_report = verify_candidate(
            "cost_saving", "c", check={"dimension": "llm_tokens"},
            executions=[_record("ex_cand", tokens=100, task_id="t1")],
            supporting=[_record("ex_base", tokens=1000, task_id="t1")])
        json_report = verify_candidate(
            "cost_saving", "c", check={"dimension": "llm_tokens"},
            executions=[{"execution_id": "ex_cand", "task_id": "t1",
                         "strategy_id": "S01", "measurement_scope": "attempt",
                         "cost": {"llm_tokens": 100},
                         "cost_measured": ["llm_tokens"],
                         "quality": {"feasible": True, "objective": 100.0,
                                     "status": "optimal"}}],
            supporting=[{"execution_id": "ex_base", "task_id": "t1",
                         "strategy_id": "S01", "measurement_scope": "attempt",
                         "cost": {"llm_tokens": 1000},
                         "cost_measured": ["llm_tokens"],
                         "quality": {"feasible": True, "objective": 100.0,
                                     "status": "optimal"}}])
        self.assertEqual(obj_report["state"], VERIFIED)
        self.assertEqual(json_report["state"], obj_report["state"],
                         json_report["conclusion"])
        self.assertIn("1000", json_report["conclusion"])


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
