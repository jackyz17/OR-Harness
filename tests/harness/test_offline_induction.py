"""Offline induction: detector hints reach the candidate bundle, and a
candidate's METHOD material is readable.

The gap this closes: the four detectors fired ONLINE (at record time) and
their evidence was returned to the caller and then thrown away, so the
offline candidate builder re-derived candidates from bare counts and never
saw the detector's cross-execution references. These tests pin the joined
chain: detect -> persist the hint -> reuse it offline -> carry the method
material -> refuse to abstract a technique from a name and a mean.
"""
import io
import json
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tests.harness.helpers import HarnessTestCase  # noqa: E402

from or_harness.api import ORHarness  # noqa: E402
from or_harness.core.schema import (  # noqa: E402
    CostVector,
    FailureRecord,
)
from or_harness.strategy.triggers import classify_failure, solver_advisories  # noqa: E402


def _profile(problem_id="t1", family="routing", **coupling):
    from or_harness.core.schema import ProblemProfile
    values = {"semantic_coupling": 0.8, "resource_coupling": 0.3,
              "temporal_coupling": 0.2, "route_complexity": 0.8}
    values.update(coupling)
    return ProblemProfile(
        problem_id=problem_id, family=family,
        scale_features={"n_vars": 100.0, "n_constraints": 50.0,
                        "n_int_vars": 100.0, "density": 0.01},
        **values)


class TestNoAutomaticHints(HarnessTestCase):
    """Recording no longer produces or persists induction labels: the
    framework does not interpret the fact it just stored. The material is
    read later with ``review-material`` and the agent abstracts it."""

    def test_record_does_not_produce_hints(self):
        h = ORHarness(home=self.home)
        self.addCleanup(h.close)
        failed = self.make_record(execution_id="f1", task_id="ft",
                                  feasible=False, status="error",
                                  solver={"name": "pulp"},
                                  profile=_profile("ft"))
        h.record(failed)
        success = self.make_record(execution_id="f2", task_id="ft",
                                   solver={"name": "ortools"},
                                   profile=_profile("ft"))
        result = h.record(success)
        self.assertNotIn("induction_hints", result)
        stored = h.bank.get("f2")
        self.assertNotIn("induction_hints", stored.execution_features)


class TestCellCandidates(HarnessTestCase):
    """The candidate builder produces structural-cell LEADS only."""

    def _run_cli(self, h, argv):
        from or_harness import cli
        buffer = io.StringIO()
        old = sys.stdout
        sys.stdout = buffer
        try:
            code = cli.main(["--home", self.home] + argv)
        finally:
            sys.stdout = old
        return code, buffer.getvalue()

    def test_only_cell_kinds_are_emitted(self):
        h = ORHarness(home=self.home)
        self.addCleanup(h.close)
        cheap = CostVector(llm_tokens=100, solver_runtime_s=1.0,
                           measured={"llm_tokens", "solver_runtime_s"})
        for prefix, sid, gap in (("a", "S01", 0.40), ("b", "S04", 0.02)):
            for i in range(2):
                h.record(self.make_record(
                    execution_id=f"{prefix}_{i}", task_id=f"{prefix}{i}",
                    strategy_id=sid, gap=gap, cost=cheap,
                    profile=_profile(f"{prefix}{i}", resource_coupling=0.30)))
        bundles = h.induction_candidates()
        self.assertTrue(bundles)
        kinds = {b["kind"] for b in bundles}
        self.assertTrue(kinds <= {"new_claim", "revision",
                                  "cell_observation"})
        for b in bundles:
            self.assertNotIn("pattern", b)
            self.assertNotIn("purpose", b)

    def test_a_candidate_carrying_no_method_is_reported(self):
        """No execution reports a method: the material report NAMES the
        missing content rather than inventing a technique from a name and a
        mean. It is a report, not an admission verdict."""
        h = ORHarness(home=self.home)
        self.addCleanup(h.close)
        for i in range(2):
            h.record(self.make_record(task_id=f"t{i}", strategy_id="S01",
                                      profile=_profile(f"t{i}")))
        bundles = h.induction_candidates()
        self.assertTrue(bundles)
        material = h.induction_material()
        self.assertTrue(material["material"])
        for item in material["material"]:
            report = item["material_report"]
            self.assertFalse(report["basis"] == "performed")
            self.assertTrue(any("method_performed" in m
                                for m in report["missing"]))
        code, out = self._run_cli(h, ["induction-material"])
        self.assertEqual(code, 0)
        self.assertIn("missing", out)

    def test_unknown_bundle_is_refused(self):
        from or_harness.core.storage import StorageError
        h = ORHarness(home=self.home)
        self.addCleanup(h.close)
        with self.assertRaises(StorageError):
            h.induction_material(bundle_id="cb_does_not_exist")


class TestRetainedFactsReachable(unittest.TestCase):
    def test_failure_classification_is_available(self):
        self.assertTrue(callable(classify_failure))
        self.assertTrue(callable(solver_advisories))


class TestRelationWriteBookkeeping(HarnessTestCase):
    """A relation submission is a knowledge WRITE, so it must be traceable
    exactly like a statistical induction: a maintenance action with a PRE
    state, a knowledge delta, and the index result. It must also not be
    counted twice."""

    def _seed(self, h, method=False):
        ids = []
        for i in range(3):
            rec = self.make_record(execution_id=f"r{i}", task_id=f"T{i}",
                                   strategy_id="S01", gap=0.02,
                                   profile=_profile(f"T{i}",
                                                    temporal_coupling=0.7,
                                                    resource_coupling=0.2))
            if method:
                rec.method_actual = {"name": "keep cross-period state",
                                     "steps": ["carry the state", "solve"]}
            h.record(rec)
            ids.append(rec.execution_id)
        return ids

    def test_a_relation_write_records_a_maintenance_action_and_delta(self):
        h = ORHarness(home=self.home)
        self.addCleanup(h.close)
        ids = self._seed(h)
        before_actions = len(h.actions.query())
        relation = {
            "subject": "principle:cross_period_state",
            "claim": "on temporally coupled tasks the cross-period state must "
                     "be preserved; dropping it cost quality",
            "conditions": {"predicates": {"family": "routing",
                                          "temporal_coupling": [0.5, 1.0]}},
            "evidence": [{"execution_id": eid, "role": "preserved"}
                         for eid in ids],
        }
        result = h.induce(relations=[relation])
        # A maintenance action was recorded with the relation shape.
        self.assertIn("action", result)
        action = h.actions.get(result["action"]["action_id"])
        self.assertIsNotNone(action)
        self.assertEqual(action.action_type, "induce")
        self.assertEqual(action.params.get("knowledge_shape"), "relations")
        self.assertEqual(action.status, "completed")
        # The delta names the created entry and reports a real transition.
        delta = result["action"]["knowledge_delta"]
        self.assertTrue(delta["entries_created"])
        self.assertEqual(delta["entry_count_before"], 0)
        self.assertEqual(delta["entry_count_after"], 1)
        self.assertEqual(len(h.actions.query()), before_actions + 1)
        # The index result is REPORTED, not discarded.
        self.assertIn("index_sync", result)

    def test_the_relation_write_is_not_counted_twice(self):
        """Two identical submissions produce two actions but ONE entry, and
        the second reports 'updated', never a second creation."""
        h = ORHarness(home=self.home)
        self.addCleanup(h.close)
        ids = self._seed(h)
        relation = {
            "subject": "principle:cross_period_state",
            "claim": "preserve the cross-period state",
            "evidence": [{"execution_id": eid, "role": "preserved"}
                         for eid in ids],
        }
        first = h.induce(relations=[relation])
        second = h.induce(relations=[relation])
        self.assertTrue(first["action"]["knowledge_delta"]["entries_created"])
        self.assertFalse(second["action"]["knowledge_delta"]["entries_created"])
        self.assertEqual(second["action"]["business_result"],
                         "relation_updated")
        self.assertEqual(h.sbank.count(), 1)

    def test_a_dry_run_relation_write_persists_nothing(self):
        h = ORHarness(home=self.home)
        self.addCleanup(h.close)
        ids = self._seed(h)
        before_actions = len(h.actions.query())
        before_entries = h.sbank.count()
        result = h.induce(relations=[{
            "subject": "principle:dry",
            "claim": "x",
            "evidence": [{"execution_id": eid, "role": "preserved"}
                         for eid in ids]}], dry_run=True)
        self.assertNotIn("action", result)
        self.assertEqual(len(h.actions.query()), before_actions)
        self.assertEqual(h.sbank.count(), before_entries)

    def test_a_method_less_relation_is_reported_not_refused(self):
        """The material report is a WARNING: the claim is saved, and the
        outcome says the framework did not (and will not) derive a technique
        from the numbers."""
        h = ORHarness(home=self.home)
        self.addCleanup(h.close)
        ids = self._seed(h)
        result = h.induce(relations=[{
            "subject": "principle:no_method",
            "claim": "S01 works better here",
            "evidence": [{"execution_id": eid, "role": "preserved"}
                         for eid in ids]}])
        outcome = result["relations"][0]
        self.assertIsNotNone(outcome["saved"])
        self.assertIsNotNone(outcome["material"])
        self.assertIn("derive a technique", outcome["material"]["reason"])

    def test_a_relation_that_reports_a_method_carries_no_warning(self):
        h = ORHarness(home=self.home)
        self.addCleanup(h.close)
        ids = self._seed(h, method=True)
        result = h.induce(relations=[{
            "subject": "principle:with_method",
            "claim": "preserve the cross-period state, then solve",
            "evidence": [{"execution_id": eid, "role": "preserved"}
                         for eid in ids]}])
        self.assertIsNone(result["relations"][0]["material"])

    def test_a_claim_may_cite_a_bundle_instead_of_execution_ids(self):
        h = ORHarness(home=self.home)
        self.addCleanup(h.close)
        self._seed(h)
        bundles = h.induction_candidates()
        self.assertTrue(bundles)
        bundle = max(bundles, key=lambda b: len(b["execution_ids"]))
        bundle_id = bundle["bundle_id"]
        self.assertEqual(len(bundle["execution_ids"]), 3)
        result = h.induce(relations=[{
            "subject": "principle:from_bundle",
            "claim": "the bundle's evidence supports this",
            "evidence": [{"bundle_id": bundle_id, "role": "preserved"}]}])
        outcome = result["relations"][0]
        self.assertIsNotNone(outcome.get("saved"))
        # Every execution in the bundle was cited.
        stored = h.sbank.get(outcome["saved"]).claim
        self.assertEqual(len(stored["evidence"]), 3)

    def test_an_unknown_bundle_is_refused_with_a_reason(self):
        h = ORHarness(home=self.home)
        self.addCleanup(h.close)
        self._seed(h)
        result = h.induce(relations=[{
            "subject": "principle:bad_bundle",
            "claim": "x",
            "evidence": [{"bundle_id": "cb_missing", "role": "preserved"}]}])
        self.assertIn("unknown induction bundle",
                      result["relations"][0]["skipped"])


class TestColdStartObservation(HarnessTestCase):
    """A lone verified execution is no longer a special trigger CATEGORY: it
    is simply a THIN CELL, reported by the same cell path with an
    ``admission_note`` that says a transferable claim is not yet admissible.
    Its material is ALWAYS visible (the sample count limits what a claim may
    assert, never what may be read)."""

    def _verified(self, h, task_id, method_name, strategy_id=None):
        rec = self.make_record(execution_id=f"ex_{task_id}", task_id=task_id,
                               strategy_id=strategy_id or f"s_{task_id}",
                               profile=_profile(task_id))
        rec.method_planned = {"name": method_name, "steps": ["do the thing"]}
        h.record(rec)
        h.check_task_result(rec.execution_id, {"reference_objective": 100.0})
        return rec

    def test_verified_distinct_task_is_a_visible_thin_cell(self):
        h = ORHarness(home=self.home)
        self.addCleanup(h.close)
        # Three DISTINCT cold-start tasks, each its own method, each
        # verified: no contrast/repair/reproduction fires, yet each is real,
        # VISIBLE material — a thin ``cell_observation``, not a silent drop.
        for i in range(3):
            self._verified(h, f"t{i}", f"method-{i}")
        bundles = h.induction_candidates()
        obs = [b for b in bundles if b["kind"] == "cell_observation"]
        self.assertEqual(len(obs), 3)
        for b in obs:
            self.assertEqual(len(b["execution_ids"]), 1)
            self.assertIsNotNone(b.get("admission_note"))
            self.assertIn("admissible", b["admission_note"])
        # The material report names what the evidence carries: a plan plus a
        # passed check, with no performed method recorded.
        material = h.induction_material()
        reports = {m["bundle_id"]: m["material_report"]
                   for m in material["material"]}
        for b in obs:
            self.assertEqual(reports[b["bundle_id"]]["basis"],
                             "planned_only")

    def test_unverified_or_methodless_record_is_visible_but_insufficient(self):
        h = ORHarness(home=self.home)
        self.addCleanup(h.close)
        # No task check -> not verified.
        rec = self.make_record(execution_id="ex_nv", task_id="t_nv",
                               strategy_id="s_nv", profile=_profile("t_nv"))
        rec.method_planned = {"name": "m", "steps": ["x"]}
        h.record(rec)
        # Verified but no method content -> not abstractable.
        rec2 = self.make_record(execution_id="ex_nm", task_id="t_nm",
                                strategy_id="s_nm", profile=_profile("t_nm"))
        h.record(rec2)
        h.check_task_result("ex_nm", {"reference_objective": 100.0})
        bundles = h.induction_candidates()
        # Both are VISIBLE thin cells (material is never hidden), but the
        # methodless/unverified one reports its missing content so nothing
        # is invented from a name and a mean.
        ids = {b["strategy_id"] for b in bundles
               if b["kind"] == "cell_observation"}
        self.assertEqual(ids, {"s_nv", "s_nm"})
        material = h.induction_material()
        reports = {m["strategy_id"]: m["material_report"]
                   for m in material["material"]}
        self.assertEqual(reports["s_nm"]["basis"], "none")
        self.assertTrue(any("method_performed" in x
                            for x in reports["s_nm"]["missing"]))

    def test_repeated_runs_of_one_task_are_not_independent(self):
        """Two runs of ONE task are repetition, not reproduction: the cell is
        visible but its ``admission_note`` says so — never disguised as
        cross-task support."""
        h = ORHarness(home=self.home)
        self.addCleanup(h.close)
        # Two runs of the SAME task (distinct execution ids) — a retry.
        for i in range(2):
            rec = self.make_record(execution_id=f"ex_c_same_{i}",
                                   task_id="c_same", strategy_id="s_shared",
                                   profile=_profile("c_same"))
            rec.method_planned = {"name": "same-method",
                                  "steps": ["do the thing"]}
            h.record(rec)
            h.check_task_result(rec.execution_id,
                                {"reference_objective": 100.0})
        bundles = [b for b in h.induction_candidates()
                   if b["strategy_id"] == "s_shared"]
        self.assertTrue(bundles)
        cell = bundles[0]
        self.assertEqual(cell["kind"], "cell_observation")
        self.assertIn("single task", cell["admission_note"])


class TestConditionalFactPublication(HarnessTestCase):
    """W4: a ``conditional_fact`` claim publishes on ONE verified
    observation, but is stamped unproven — the fact/transfer distinction."""

    def test_single_observation_fact_publishes_but_is_unproven(self):
        h = ORHarness(home=self.home)
        self.addCleanup(h.close)
        rec = self.make_record(execution_id="ex_one", task_id="T1",
                               strategy_id="m1", profile=_profile("T1"))
        rec.method_planned = {"name": "direct upper-bound",
                              "steps": ["set the bound"]}
        h.record(rec)
        h.check_task_result("ex_one", {"reference_objective": 100.0})
        result = h.induce(relations=[{
            "subject": "principle:single_fact",
            "kind": "conditional_fact",
            "claim": ("on independent upper-bound problems, setting each "
                      "variable to its bound gave a checked answer"),
            "method": {"name": "direct upper-bound", "steps": ["set the bound"]},
            "evidence": [{"execution_id": "ex_one", "role": "sole_evidence"}]}],
            verify={"claim": "the answer passed its check",
                    "check": {"assertions": [
                        {"kind": "status", "roles": ["sole_evidence"],
                         "status": "optimal"}]}})
        rel = result["relations"][0]
        pub = rel["publication"]
        self.assertTrue(pub["published"])
        self.assertEqual(pub["support_scope"], "single_observation")
        self.assertEqual(pub["transferability"], "unproven")
        from or_harness.strategy.selector import is_publishable
        self.assertTrue(is_publishable(h.sbank.get(rel["saved"])))

    def test_transferable_claim_still_needs_two_tasks(self):
        """The rule is NOT relaxed for transferable (non-fact) claims."""
        h = ORHarness(home=self.home)
        self.addCleanup(h.close)
        rec = self.make_record(execution_id="ex_two", task_id="T1",
                               strategy_id="m1", profile=_profile("T1"))
        h.record(rec)
        h.check_task_result("ex_two", {"reference_objective": 100.0})
        result = h.induce(relations=[{
            "subject": "principle:rule",
            "kind": "rule",
            "claim": "this method always works",
            "evidence": [{"execution_id": "ex_two", "role": "evidence"}]}],
            verify={"claim": "ok", "check": {"assertions": [
                {"kind": "status", "roles": ["evidence"],
                 "status": "optimal"}]}})
        pub = result["relations"][0]["publication"]
        self.assertFalse(pub["published"])
        self.assertIn("independent tasks", " ".join(pub["reasons"]))


if __name__ == "__main__":
    unittest.main()
