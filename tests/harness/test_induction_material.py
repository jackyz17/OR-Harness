"""Offline review chain: material entry, trigger demotion, and the two
verification-syntax fixes.

This pins the behaviours this round introduced:

1. ``review-material`` reads a BATCH of completed tasks straight from the
   evidence bank — no detector candidate, no sample-count gate — so failed,
   cross-cell and cross-method-name material are all visible; a same-task
   retry is marked non-independent; missing fields are marked, not dropped.
2. A thin cell / single repeated task is a ``cell_observation`` with an
   ``admission_note`` (visible), NOT a silent drop, and the old
   ``single_observation`` trigger category is gone.
3. A predicate key the framework cannot evaluate never defaults to
   ``applies`` (``profile_matches`` False, ``classify_applicability``
   ``unknown``).
4. A ``check`` block INSIDE a ``--relation`` payload is honoured, and when
   both an embedded check and ``--verify`` are supplied the override is
   reported.
5. ``entry.knowledge.estimate`` separates "not estimated" from a measured 0.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tests.harness.helpers import HarnessTestCase  # noqa: E402

from or_harness.api import ORHarness  # noqa: E402
from or_harness.core.schema import (  # noqa: E402
    ProblemProfile,
    profile_matches,
)
from or_harness.strategy.vector_recall import classify_applicability  # noqa: E402


def _profile(problem_id="t1", family="routing", **coupling):
    values = {"semantic_coupling": 0.8, "resource_coupling": 0.3,
              "temporal_coupling": 0.2, "route_complexity": 0.8}
    values.update(coupling)
    return ProblemProfile(
        problem_id=problem_id, family=family,
        scale_features={"n_vars": 100.0, "n_constraints": 50.0,
                        "n_int_vars": 100.0, "density": 0.01},
        **values)


class TestReviewMaterialEntry(HarnessTestCase):
    """The default material entry point needs no detector and no count gate."""

    def _seed(self, h, records):
        for rec in records:
            h.record(rec)

    def test_failed_and_cross_name_material_are_visible(self):
        h = ORHarness(home=self.home)
        self.addCleanup(h.close)
        self._seed(h, [
            self.make_record(execution_id="ex_ok", task_id="T_ok",
                             strategy_id="milp_named_a",
                             profile=_profile("T_ok")),
            self.make_record(execution_id="ex_bad", task_id="T_bad",
                             strategy_id="flow_named_b",
                             feasible=False, status="error",
                             profile=_profile("T_bad", route_complexity=0.95)),
        ])
        result = h.induction_material()
        ids = {m["execution_id"] for m in result["material"]}
        self.assertEqual(ids, {"ex_ok", "ex_bad"})
        bad = [m for m in result["material"]
               if m["execution_id"] == "ex_bad"][0]
        self.assertFalse(bad["outcome"]["feasible"])
        # A never-checked record reports that explicitly, not as a pass.
        self.assertEqual(bad["task_check_state"], "never_checked")

    def test_retry_is_marked_non_independent(self):
        h = ORHarness(home=self.home)
        self.addCleanup(h.close)
        for i in range(2):
            h.record(self.make_record(execution_id=f"ex_r{i}", task_id="T_rep",
                                      strategy_id="S", profile=_profile("T_rep")))
        result = h.induction_material(task_id="T_rep")
        self.assertEqual(result["n_distinct_tasks"], 1)
        for m in result["material"]:
            self.assertEqual(m["attempts_of_task"], 2)
            self.assertFalse(m["independent_task"])

    def test_missing_fields_are_marked_not_dropped(self):
        h = ORHarness(home=self.home)
        self.addCleanup(h.close)
        # A record with no predicted method and no task text digest.
        rec = self.make_record(execution_id="ex_bare", task_id="T_bare",
                               strategy_id="S", profile=_profile("T_bare"))
        h.record(rec)
        result = h.induction_material()
        entry = result["material"][0]
        # The method is present as an explicit basis, not silently absent.
        self.assertEqual(entry["method"]["basis"], "none")
        # The task text has no retained version -> an explicit unknown marker.
        self.assertIsInstance(entry["task_text"], dict)
        self.assertIn("unknown", entry["task_text"])

    def test_budget_eviction_is_reported(self):
        h = ORHarness(home=self.home)
        self.addCleanup(h.close)
        for i in range(6):
            h.record(self.make_record(execution_id=f"ex_b{i}", task_id=f"T_b{i}",
                                      strategy_id="S", profile=_profile(f"T_b{i}")))
        old = os.environ.get("OR_HARNESS_INDUCTION_MATERIAL_CHARS")
        os.environ["OR_HARNESS_INDUCTION_MATERIAL_CHARS"] = "1500"
        try:
            result = h.induction_material()
        finally:
            if old is None:
                os.environ.pop("OR_HARNESS_INDUCTION_MATERIAL_CHARS", None)
            else:
                os.environ["OR_HARNESS_INDUCTION_MATERIAL_CHARS"] = old
        self.assertTrue(result["budget"]["truncated_by_budget"])
        self.assertTrue(result["budget"]["omitted_execution_ids"])
        self.assertLess(result["count"], 6)

    def test_method_why_and_fallback_are_carried(self):
        """The material keeps the method's ``why`` and ``fallback`` — the
        operative knowledge a reviewer judges a mechanism and its premises
        with — not only the name and steps."""
        h = ORHarness(home=self.home)
        self.addCleanup(h.close)
        rec = self.make_record(execution_id="ex_m", task_id="T_m",
                               strategy_id="S", profile=_profile("T_m"))
        rec.method_actual = {
            "name": "bound-and-enumerate",
            "steps": ["bound Y", "enumerate X"],
            "why": "the bound makes the range finite",
            "fallback": "fall back to a full MILP when the bound fails"}
        h.record(rec)
        entry = h.induction_material()["material"][0]
        actual = entry["method"]["actual"]
        self.assertEqual(actual["why"], "the bound makes the range finite")
        self.assertIn("full MILP", actual["fallback"])

    def test_material_reports_the_banks_memory_state(self):
        """Cold start is reported as a STATE, never a gate."""
        h = ORHarness(home=self.home)
        self.addCleanup(h.close)
        empty = h.induction_material()
        self.assertEqual(empty["memory_state"]["state"], "both_banks_empty")
        h.bank.append(self.make_record(
            execution_id="ex_s", task_id="T_s", strategy_id="S",
            profile=_profile("T_s")))
        out = h.induction_material()
        self.assertEqual(out["memory_state"]["state"],
                         "strategic_bank_empty")
        self.assertFalse(out["memory_state"]["evidence_bank_empty"])

    def test_material_entry_has_an_inspect_hint(self):
        """A bounded excerpt names how to read the FULL record by id."""
        h = ORHarness(home=self.home)
        self.addCleanup(h.close)
        h.bank.append(self.make_record(
            execution_id="ex_h", task_id="T_h", strategy_id="S",
            profile=_profile("T_h")))
        entry = h.induction_material()["material"][0]
        self.assertIn("ex_h", entry["inspect_hint"])


class TestTriggerDemotion(HarnessTestCase):
    """A single execution is visible material, never a silent drop — and
    there is no ``cell_observation`` verdict any more: the batch reports the
    FACTS (one task, one attempt), and the agent decides."""

    def test_single_execution_is_visible_material(self):
        h = ORHarness(home=self.home)
        self.addCleanup(h.close)
        h.record(self.make_record(execution_id="ex_one", task_id="T1",
                                  strategy_id="S1", profile=_profile("T1")))
        material = h.induction_material(strategy_id="S1")
        self.assertEqual(material["count"], 1)
        self.assertEqual(material["n_distinct_tasks"], 1)
        # No candidate verdict anywhere: the facts are the report.
        blob = str(material)
        self.assertNotIn("cell_observation", blob)
        self.assertNotIn("admission_note", blob)
        self.assertNotIn("single_observation", blob)


class TestPredicateHonesty(HarnessTestCase):
    """An unevaluable predicate key never defaults to 'applies'."""

    def test_unsupported_predicate_is_not_a_match(self):
        profile = _profile("t", family="anything", resource_coupling=0.1,
                           temporal_coupling=0.9, route_complexity=0.9)
        predicates = {"problem_class": "scheduling", "task_family": "x",
                      "structure": {"a": 1},
                      "resource_coupling": [0.0, 0.25]}
        self.assertFalse(profile_matches(profile, predicates))
        state, reusable, reason = classify_applicability(profile, predicates)
        self.assertEqual(state, "unknown")
        self.assertFalse(reusable)
        self.assertIn("problem_class", reason)

    def test_supported_predicates_still_decide(self):
        profile = _profile("t", family="sched", resource_coupling=0.1,
                           temporal_coupling=0.9, route_complexity=0.9)
        self.assertTrue(profile_matches(
            profile, {"family": "sched", "resource_coupling": [0.0, 0.25]}))
        state, reusable, reason = classify_applicability(
            profile, {"family": "sched", "resource_coupling": [0.0, 0.25]})
        self.assertEqual((state, reusable), ("applies", True))
        # A KNOWN contradiction is 'conflicts', distinct from 'unknown'.
        self.assertEqual(classify_applicability(profile, {"family": "other"})[0],
                         "conflicts")


class TestRelationCheckSyntax(HarnessTestCase):
    """An embedded ``check`` is read; a conflicting ``--verify`` is reported."""

    def _seed_pair(self, h):
        before = self.make_record(execution_id="ex_before", task_id="T1",
                                  strategy_id="S", gap=0.4,
                                  profile=_profile("T1"))
        after = self.make_record(execution_id="ex_after", task_id="T1",
                                 strategy_id="S", gap=0.0,
                                 profile=_profile("T1"))
        h.record(before)
        h.record(after)
        return ["ex_before", "ex_after"]

    def test_embedded_check_is_honoured(self):
        h = ORHarness(home=self.home)
        self.addCleanup(h.close)
        self._seed_pair(h)
        out = h.induce(relations=[{
            "subject": "principle:embedded",
            "claim": "the fix raised quality on this task",
            "evidence": [{"execution_id": "ex_before", "role": "before"},
                         {"execution_id": "ex_after", "role": "after"}],
            "check": {"assertions": [
                {"kind": "comparison", "metric": "quality",
                 "roles_a": ["after"], "roles_b": ["before"],
                 "direction": "higher", "min_gap": 0.2, "mode": "paired"}]},
        }])
        rel = out["relations"][0]
        self.assertIsNotNone(rel["saved"])
        self.assertEqual(rel["check_note"]["check_source"],
                         "embedded_relation_check")
        self.assertEqual(h.sbank.get(rel["saved"]).verification_state,
                         "verified")

    def test_verify_arg_wins_and_reports_the_override(self):
        h = ORHarness(home=self.home)
        self.addCleanup(h.close)
        self._seed_pair(h)
        out = h.induce(
            relations=[{
                "subject": "principle:conflict",
                "claim": "the fix raised quality on this task",
                "evidence": [{"execution_id": "ex_before", "role": "before"},
                             {"execution_id": "ex_after", "role": "after"}],
                # An embedded check that would REFUTE if it were used.
                "check": {"assertions": [
                    {"kind": "probe", "roles": ["after"],
                     "path": "quality.feasible", "equals": False}]},
            }],
            verify={"claim": "ok", "check": {"assertions": [
                {"kind": "comparison", "metric": "quality",
                 "roles_a": ["after"], "roles_b": ["before"],
                 "direction": "higher", "min_gap": 0.2, "mode": "paired"}]}})
        rel = out["relations"][0]
        self.assertEqual(rel["check_note"]["check_source"], "verify_arg")
        self.assertIn("overridden", rel["check_note"]["note"])
        # The standalone verdict won: the claim is verified.
        self.assertEqual(h.sbank.get(rel["saved"]).verification_state,
                         "verified")


class TestEstimateHonesty(HarnessTestCase):
    """``knowledge.estimate`` keeps 'not estimated' apart from a measured 0."""

    def test_claim_only_entry_reports_not_estimated(self):
        h = ORHarness(home=self.home)
        self.addCleanup(h.close)
        before = self.make_record(execution_id="ex_cb", task_id="T1",
                                  strategy_id="S", gap=0.4,
                                  profile=_profile("T1"))
        after = self.make_record(execution_id="ex_ca", task_id="T1",
                                 strategy_id="S", gap=0.0,
                                 profile=_profile("T1"))
        h.record(before)
        h.record(after)
        out = h.induce(relations=[{
            "subject": "principle:no_estimate",
            "kind": "conditional_fact",
            "claim": "the fix raised quality on this task",
            "evidence": [{"execution_id": "ex_cb", "role": "before"},
                         {"execution_id": "ex_ca", "role": "after"}],
            "check": {"assertions": [
                {"kind": "comparison", "metric": "quality",
                 "roles_a": ["after"], "roles_b": ["before"],
                 "direction": "higher", "min_gap": 0.2, "mode": "paired"}]},
        }])
        entry = h.sbank.get(out["relations"][0]["saved"])
        self.assertEqual(entry.support_n, 0)
        knowledge = h.selector._entry_knowledge(entry)
        estimate = knowledge["estimate"]
        self.assertFalse(estimate["estimated"])
        self.assertIsNone(estimate["quality_hat"])
        self.assertEqual(estimate["evidence_executions"], 2)
        self.assertEqual(estimate["distinct_tasks"], 1)


if __name__ == "__main__":
    unittest.main()
