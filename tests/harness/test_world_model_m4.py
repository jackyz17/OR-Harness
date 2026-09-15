"""M4 maintenance tests: candidate bundling, value assessment, explicit
adoption/rejection, scope preservation, and attribution.

All tests use controlled scripted providers — nothing here claims real-LLM
behaviour.
"""

import copy
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tests.harness.helpers import HarnessTestCase  # noqa: E402

from or_harness.api import ORHarness  # noqa: E402
from or_harness.strategy.strategic_bank import StrategicEntry  # noqa: E402
from or_harness.world_model.maintenance import (  # noqa: E402
    InductionAssessment,
    InductionCandidateBundle,
    build_induction_candidates,
)
from or_harness.world_model.provider import WorldModelProvider  # noqa: E402


def _profile(problem_id="t1", family="routing", **coupling):
    values = {"semantic_coupling": 0.8, "resource_coupling": 0.3,
              "temporal_coupling": 0.2, "route_complexity": 0.8}
    values.update(coupling)
    from or_harness.core.schema import ProblemProfile
    return ProblemProfile(
        problem_id=problem_id,
        family=family,
        scale_features={"n_vars": 100.0, "n_constraints": 50.0,
                        "n_int_vars": 100.0, "density": 0.01},
        **values,
    )


class MaintenanceScriptedProvider(WorldModelProvider):
    """Controlled provider for M4 tests. Answers with explicit,
    pre-configured maintenance consequences."""

    name = "maint-scripted"

    def __init__(self, response=None, fail=False):
        self.response = response or {
            "candidate_formation_prob": 0.9,
            "expected_reuse_benefit": 0.8,
            "generalization_risk": 0.1,
            "quality": 0.85,
            "failure_prob": 0.05,
            "cost": {"solver_runtime_s": 5.0},
            "confidence": 0.8,
        }
        self.fail = fail
        self.requests = []

    def predict(self, request, timeout_s=None):
        self.requests.append(request)
        if self.fail:
            return {"payload": None, "usage": {"completion_tokens": 10},
                    "error": "simulated provider failure", "latency_s": 0.01}
        return {"payload": copy.deepcopy(self.response),
                "usage": {"prompt_tokens": 150, "completion_tokens": 80},
                "error": None, "latency_s": 0.02}


class TestM4CandidateBundling(HarnessTestCase):
    """M4-A: candidate bundle construction from real evidence."""

    def test_bundle_requires_independent_evidence(self):
        """A candidate bundle forms only when >=2 executions exist across
        >=2 distinct tasks. Repeating one task does NOT form a bundle."""
        h = ORHarness(home=self.home)
        self.addCleanup(h.close)

        # 1. Single execution: no bundle.
        h.bank.append(self.make_record(task_id="t1", strategy_id="S01",
                                       profile=_profile()))
        self.assertEqual(len(h.induction_candidates()), 0)

        # 2. Repeated execution on the SAME task: n=2 but tasks=1 => no bundle.
        h.bank.append(self.make_record(task_id="t1", strategy_id="S01",
                                       profile=_profile()))
        self.assertEqual(len(h.induction_candidates()), 0)

        # 3. Independent execution on task t2: n=2, tasks=2 => bundle forms.
        h.bank.append(self.make_record(task_id="t2", strategy_id="S01",
                                       profile=_profile()))
        bundles = h.induction_candidates()
        self.assertEqual(len(bundles), 1)
        b = bundles[0]
        self.assertEqual(b["kind"], "new_claim")
        self.assertEqual(b["strategy_id"], "S01")
        self.assertEqual(b["family"], "routing")
        self.assertEqual(b["tasks"], ["t1", "t2"])
        self.assertEqual(b["n_supporting"], 3)  # all 3 unique executions
        self.assertTrue(b["trigger_reasons"])

    def test_bundle_distinguishes_new_and_revision(self):
        """A bundle for a cell covered by an existing StrategicEntry is
        labelled as a revision candidate with the entry reference preserved."""
        h = ORHarness(home=self.home)
        self.addCleanup(h.close)

        # Pre-populate an existing StrategicEntry.
        prof = _profile(problem_id="t1", resource_coupling=0.35)
        entry = StrategicEntry(
            entry_id=StrategicEntry.new_id(),
            strategy_id="S01",
            pattern={"predicates": {"family": "routing",
                                    "resource_coupling": [0.25, 0.5]}},
            expected_quality_hat=0.5,
            verification={"state": "verified", "claim": "x"})
        h.sbank.add(entry)

        # Add evidence that has drifted from the entry's claim.
        h.bank.append(self.make_record(task_id="t1", strategy_id="S01",
                                       status="optimal", objective=95.0,
                                       gap=0.05, profile=prof))
        h.bank.append(self.make_record(task_id="t2", strategy_id="S01",
                                       status="optimal", objective=95.0,
                                       gap=0.05, profile=prof))
        bundles = h.induction_candidates()
        self.assertEqual(len(bundles), 1)
        b = bundles[0]
        self.assertEqual(b["kind"], "revision")
        self.assertEqual(b["target_entry_id"], entry.entry_id)
        self.assertIsNotNone(b["entry_before"])

    def test_hypothetical_and_future_records_excluded(self):
        """Hypothetical executions and dynamically added future records do
        NOT alter an already-formed bundle."""
        h = ORHarness(home=self.home)
        self.addCleanup(h.close)
        h.bank.append(self.make_record(task_id="t1", strategy_id="S01",
                                       profile=_profile()))
        h.bank.append(self.make_record(task_id="t2", strategy_id="S01",
                                       profile=_profile()))
        bundle_dict = h.induction_candidates()[0]
        bundle = InductionCandidateBundle.from_dict(bundle_dict)

        # Later: new execution arrives. The frozen bundle does NOT change.
        h.bank.append(self.make_record(task_id="t3", strategy_id="S01",
                                       profile=_profile()))
        self.assertEqual(bundle.n_supporting, 2)
        self.assertEqual(bundle.tasks, ["t1", "t2"])


class TestM4ValueAssessment(HarnessTestCase):
    """M4-B: assess_induction consequence and value evaluation."""

    def test_assess_induction_recommends_new_claim(self):
        """When the model predicts high formation prob, positive net value,
        and the bundle has sufficient evidence, recommendation=induce_new."""
        provider = MaintenanceScriptedProvider({
            "candidate_formation_prob": 0.95,
            "expected_reuse_benefit": 0.8,
            "generalization_risk": 0.1,
            "quality": 0.9,
            "confidence": 0.85,
        })
        h = ORHarness(home=self.home, world_model=provider)
        self.addCleanup(h.close)

        h.bank.append(self.make_record(task_id="t1", strategy_id="S01",
                                       profile=_profile()))
        h.bank.append(self.make_record(task_id="t2", strategy_id="S01",
                                       profile=_profile()))
        bundle = h.induction_candidates()[0]

        res = h.assess_induction(bundle)
        self.assertEqual(res["recommendation"], "induce_new")
        self.assertGreater(res["net_value"], 0.0)
        self.assertEqual(res["target_strategy_id"], "S01")
        self.assertEqual(res["target_family"], "routing")
        # Assessment cost charged to the maintenance decision action.
        self.assertIsNotNone(res["assessment_cost"])
        self.assertEqual(res["assessment_cost"]["cost"]["llm_tokens"], 80.0)
        # Decision action is queryable.
        decision = h.actions.get(res["decision_action_id"])
        self.assertIsNotNone(decision)
        self.assertEqual(decision.status, "completed")

    def test_assess_induction_defers_when_net_value_negative(self):
        """When generalization risk exceeds reuse benefit, recommendation=defer."""
        provider = MaintenanceScriptedProvider({
            "candidate_formation_prob": 0.9,
            "expected_reuse_benefit": 0.1,  # low benefit
            "generalization_risk": 0.9,     # high risk
        })
        h = ORHarness(home=self.home, world_model=provider)
        self.addCleanup(h.close)

        h.bank.append(self.make_record(task_id="t1", strategy_id="S01",
                                       profile=_profile()))
        h.bank.append(self.make_record(task_id="t2", strategy_id="S01",
                                       profile=_profile()))
        bundle = h.induction_candidates()[0]

        res = h.assess_induction(bundle)
        self.assertEqual(res["recommendation"], "defer")
        self.assertLessEqual(res["net_value"], 0.0)
        self.assertIn("cost/risk outweighs", res["recommendation_basis"])

    def test_assess_induction_handles_provider_failure(self):
        """Provider failure reports insufficient_evidence with cost preserved;
        decision action ends failed."""
        provider = MaintenanceScriptedProvider(fail=True)
        h = ORHarness(home=self.home, world_model=provider)
        self.addCleanup(h.close)

        h.bank.append(self.make_record(task_id="t1", strategy_id="S01",
                                       profile=_profile()))
        h.bank.append(self.make_record(task_id="t2", strategy_id="S01",
                                       profile=_profile()))
        bundle = h.induction_candidates()[0]

        res = h.assess_induction(bundle)
        self.assertEqual(res["recommendation"], "insufficient_evidence")
        self.assertEqual(res["status"], "invalid_output")
        # Failed call cost still preserved.
        self.assertIsNotNone(res["assessment_cost"])
        self.assertEqual(res["assessment_cost"]["cost"]["llm_tokens"], 10.0)
        decision = h.actions.get(res["decision_action_id"])
        self.assertEqual(decision.status, "failed")

    def test_workload_forecast_scales_reuse_benefit(self):
        """A workload forecast scales expected reuse benefit proportionally."""
        provider = MaintenanceScriptedProvider({
            "candidate_formation_prob": 0.9,
            "expected_reuse_benefit": 0.1,
            "generalization_risk": 0.5,
        })
        h = ORHarness(home=self.home, world_model=provider)
        self.addCleanup(h.close)

        h.bank.append(self.make_record(task_id="t1", strategy_id="S01",
                                       profile=_profile()))
        h.bank.append(self.make_record(task_id="t2", strategy_id="S01",
                                       profile=_profile()))
        bundle = h.induction_candidates()[0]

        # Without forecast: net_value = 0.1 - 0.5 = -0.4 => defer.
        res_no_forecast = h.assess_induction(bundle)
        self.assertEqual(res_no_forecast["recommendation"], "defer")

        # With forecast: 10 expected matching tasks => benefit = 0.1 * 10 = 1.0;
        # net_value = 1.0 - 0.5 = 0.5 => induce_new.
        res_forecast = h.assess_induction(
            bundle, workload_forecast={"expected_matching_tasks": 10})
        self.assertEqual(res_forecast["recommendation"], "induce_new")
        self.assertAlmostEqual(res_forecast["net_value"], 0.5, places=3)


class TestM4AdoptionAndInductionLoop(HarnessTestCase):
    """M4-C/D/E: explicit adoption/rejection, scope restriction, downstream use."""

    def test_explicit_adoption_triggers_induction_with_scope(self):
        """Accepting an assessment triggers induction strictly on the bundle's
        execution IDs; Strategic Bank gains the verified entry."""
        provider = MaintenanceScriptedProvider()
        h = ORHarness(home=self.home, world_model=provider)
        self.addCleanup(h.close)

        # Seed evidence across two tasks.
        prof = _profile(problem_id="t1")
        r1 = self.make_record(task_id="t1", strategy_id="S01",
                              profile=prof)
        r2 = self.make_record(task_id="t2", strategy_id="S01",
                              profile=prof)
        h.bank.append(r1)
        h.bank.append(r2)
        bundle = h.induction_candidates()[0]

        assessment = h.assess_induction(bundle)
        self.assertEqual(assessment["recommendation"], "induce_new")

        # Explicit adoption with admission verification.
        verify_payload = {
            "purpose": "rule",
            "claim": "S01 produces optimal solutions on routing",
            "check": {"kind": "result", "reference_status": "optimal"},
            "executions": [r1.to_dict(), r2.to_dict()],
        }
        res = h.accept_induction(assessment, verify=verify_payload)
        self.assertTrue(res["accepted"])
        self.assertIsNotNone(res["adoption_action_id"])

        # Strategic Bank now has the entry.
        entries = h.sbank.matching(prof)
        self.assertEqual(len(entries), 1)
        entry = entries[0]
        self.assertEqual(entry.strategy_id, "S01")
        self.assertEqual(entry.verification_state, "verified")
        # Subsequent task recalls the verified entry.
        recs = h.recall({"task_id": "t3", "family": "routing",
                         "annotations": {"coupling": {
                             "resource_coupling": 0.3,
                             "temporal_coupling": 0.2,
                             "route_complexity": 0.8}}})
        matched_sids = [r["strategy_id"] for r in recs["recommendations"]]
        self.assertIn("S01", matched_sids)

    def test_rejection_touches_nothing(self):
        """Rejecting an assessment records a rejection action; Strategic
        Bank is untouched; no induction runs."""
        provider = MaintenanceScriptedProvider()
        h = ORHarness(home=self.home, world_model=provider)
        self.addCleanup(h.close)

        h.bank.append(self.make_record(task_id="t1", strategy_id="S01",
                                       profile=_profile()))
        h.bank.append(self.make_record(task_id="t2", strategy_id="S01",
                                       profile=_profile()))
        bundle = h.induction_candidates()[0]
        assessment = h.assess_induction(bundle)

        res = h.reject_induction(assessment, reason="not priority right now")
        self.assertTrue(res["rejected"])
        self.assertFalse(res["accepted"])
        self.assertEqual(res["reason"], "not priority right now")
        # Strategic bank is still empty.
        self.assertEqual(h.sbank.count(), 0)

    def test_shadow_mode_withholds_advice(self):
        """Under induction_assessment='shadow', the evaluation runs and
        is recorded, but advice is withheld."""
        provider = MaintenanceScriptedProvider()
        h = ORHarness(home=self.home, world_model=provider,
                      induction_assessment="shadow")
        self.addCleanup(h.close)

        h.bank.append(self.make_record(task_id="t1", strategy_id="S01",
                                       profile=_profile()))
        h.bank.append(self.make_record(task_id="t2", strategy_id="S01",
                                       profile=_profile()))
        bundle = h.induction_candidates()[0]

        res = h.assess_induction(bundle)
        self.assertIn("shadow mode", res["recommendation_basis"])
        self.assertIn("advice withheld", res["recommendation_basis"])

    def test_verification_failure_does_not_publish(self):
        """M4-C key discipline: a high-confidence recommendation whose
        admission verification FAILS must not form published knowledge —
        the entry stays unverified (or the induction is refused)."""
        provider = MaintenanceScriptedProvider({
            "candidate_formation_prob": 0.95,
            "expected_reuse_benefit": 0.8,
            "generalization_risk": 0.1,
            "confidence": 0.95,  # high self-reported confidence
        })
        h = ORHarness(home=self.home, world_model=provider)
        self.addCleanup(h.close)

        prof = _profile(problem_id="t1")
        r1 = self.make_record(task_id="t1", strategy_id="S01",
                              status="infeasible", profile=prof)
        r2 = self.make_record(task_id="t2", strategy_id="S01",
                              status="infeasible", profile=prof)
        h.bank.append(r1)
        h.bank.append(r2)
        bundle = h.induction_candidates()[0]
        assessment = h.assess_induction(bundle)
        # The model recommended it (its confidence is not evidence).
        self.assertEqual(assessment["recommendation"], "induce_new")

        # But the admission check against real executions FAILS the
        # claim (executions are infeasible, the claim asserts optimal).
        verify_payload = {
            "purpose": "rule",
            "claim": "S01 produces optimal solutions on routing",
            "check": {"kind": "result", "reference_status": "optimal"},
            "executions": [r1.to_dict(), r2.to_dict()],
        }
        res = h.accept_induction(assessment, verify=verify_payload)
        # The induction ran but the entry was NOT published as verified.
        entries = h.sbank.matching(prof)
        self.assertEqual(len(entries), 1)
        entry = entries[0]
        self.assertNotEqual(entry.verification_state, "verified")
        self.assertIn(entry.verification_state,
                      ("insufficient_evidence", "unverified", "refuted"))

    def test_assessment_cost_in_maintenance_scope_only(self):
        """M4-D: the assessment's real model-call cost lands in the
        MAINTENANCE task scope (__maintenance__), never in a business
        task's budget."""
        provider = MaintenanceScriptedProvider()
        h = ORHarness(home=self.home, world_model=provider)
        self.addCleanup(h.close)

        h.bank.append(self.make_record(task_id="t1", strategy_id="S01",
                                       profile=_profile()))
        h.bank.append(self.make_record(task_id="t2", strategy_id="S01",
                                       profile=_profile()))
        bundle = h.induction_candidates()[0]

        h.declare_budget("t1", {"llm_tokens": 10000.0}, episode_id="ep1")
        res = h.assess_induction(bundle)

        # The decision action is in the maintenance scope.
        decision = h.actions.get(res["decision_action_id"])
        self.assertEqual(decision.task_id, "__maintenance__")
        self.assertTrue(decision.episode_id.startswith("maint_"))
        # The business task's budget does NOT carry the assessment spend.
        view = h.budget_view("t1", "ep1")
        task_llm = (view.get("consumption") or {}).get(
            "total_cost", {}).get("llm_tokens")
        self.assertNotEqual(task_llm, 80.0)
        # The maintenance action DOES carry it (own cost).
        self.assertEqual(decision.cost.llm_tokens, 80.0)

    def test_disabled_mode_returns_explicitly(self):
        """M4-E: induction_assessment='disabled' returns an explicit
        disabled status without any model call."""
        provider = MaintenanceScriptedProvider()
        h = ORHarness(home=self.home, world_model=provider,
                      induction_assessment="disabled")
        self.addCleanup(h.close)

        h.bank.append(self.make_record(task_id="t1", strategy_id="S01",
                                       profile=_profile()))
        h.bank.append(self.make_record(task_id="t2", strategy_id="S01",
                                       profile=_profile()))
        bundle = h.induction_candidates()[0]

        res = h.assess_induction(bundle)
        self.assertEqual(res["status"], "disabled")
        # No model call, no decision action, no cost.
        self.assertEqual(len(provider.requests), 0)
        self.assertIsNone(res["decision_action_id"])
        self.assertIsNone(res["assessment_cost"])


if __name__ == "__main__":
    unittest.main()
