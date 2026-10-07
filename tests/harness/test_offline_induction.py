"""Offline induction: recording writes EVIDENCE only, the material is read
directly, and a relation submission is a traceable knowledge write.

The gap this pinned: the framework used to build induction CANDIDATES from
counts and let a claim cite a bundle. That generator is gone — the outer
agent reads a BATCH of completed tasks (``induction-material``, no candidate,
no sample-count gate) and submits the strategy itself. These tests pin the
converged chain: record -> material (a batch, never a candidate verdict) ->
``induce --relation`` -> a maintenance action with a real knowledge delta.
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
    read later with ``induction-material`` and the agent abstracts it."""

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


class TestInductionMaterialEntry(HarnessTestCase):
    """The material reads a BATCH of completed tasks — no candidate, no
    sample-count gate, no ``new_claim`` / ``cell_observation`` verdict."""

    def _seed(self, h):
        cheap = CostVector(llm_tokens=100, solver_runtime_s=1.0,
                           measured={"llm_tokens", "solver_runtime_s"})
        for prefix, sid, gap in (("a", "S01", 0.40), ("b", "S04", 0.02)):
            for i in range(2):
                h.record(self.make_record(
                    execution_id=f"{prefix}_{i}", task_id=f"{prefix}{i}",
                    strategy_id=sid, gap=gap, cost=cheap,
                    profile=_profile(f"{prefix}{i}", resource_coupling=0.30)))
        return h

    def test_material_reads_a_batch_with_no_candidate_verdict(self):
        h = self._seed(ORHarness(home=self.home))
        self.addCleanup(h.close)
        material = h.induction_material()
        self.assertEqual(material["count"], 4)
        self.assertEqual(material["n_distinct_tasks"], 4)
        # No candidate-shaped keys anywhere: this is FACTS, not a verdict.
        blob = json.dumps(material)
        for forbidden in ("new_claim", "cell_observation", "bundle_id",
                          "admission_note", "material_report"):
            self.assertNotIn(forbidden, blob)
        # Cross-task affordance reports the real distinct-task count.
        hint = material["cross_task_hint"]
        self.assertEqual(hint["n_distinct_tasks_in_batch"], 4)
        self.assertGreaterEqual(hint["distinct_tasks_across_history"], 4)

    def test_a_batch_with_no_method_is_still_readable(self):
        """A batch whose evidence reports no method is VISIBLE, not hidden:
        the missing field is what the agent goes and reads/records — it is
        never an admission verdict."""
        h = ORHarness(home=self.home)
        self.addCleanup(h.close)
        for i in range(2):
            h.record(self.make_record(task_id=f"t{i}", strategy_id="S01",
                                      profile=_profile(f"t{i}")))
        material = h.induction_material()
        self.assertEqual(material["count"], 2)
        for entry in material["material"]:
            self.assertEqual(entry["method"]["basis"], "none")
            self.assertIsNone(entry["method"]["planned"])
            self.assertIsNone(entry["method"]["actual"])

    def test_paging_walks_the_whole_history(self):
        """A page never stands in for the bank: the cursor reads OLDER
        material, so a long history is walked in distinct batches."""
        h = ORHarness(home=self.home)
        self.addCleanup(h.close)
        for i in range(6):
            rec = self.make_record(execution_id=f"p{i}", task_id=f"T{i}",
                                   strategy_id="S01",
                                   profile=_profile(f"T{i}"))
            rec.created_at = 1000.0 + i      # distinct, increasing
            h.record(rec)
        seen = set()
        cursor = None
        for _ in range(20):
            page = h.induction_material(limit=2, cursor=cursor)
            if not page["material"]:
                break
            seen.update(m["execution_id"] for m in page["material"])
            if len(page["material"]) < 2:
                break
            cursor = page["material"][0]["cursor"]  # oldest entry returned
        self.assertEqual(seen, {f"p{i}" for i in range(6)})

    def test_cli_help_points_at_the_converged_flow(self):
        from or_harness import cli
        buffer = io.StringIO()
        old = sys.stdout
        sys.stdout = buffer
        try:
            code = cli.main(["--home", self.home, "induction-material"])
        finally:
            sys.stdout = old
        self.assertEqual(code, 0)
        self.assertIn("induce --relation", buffer.getvalue())


class TestRetainedFactsReachable(unittest.TestCase):
    def test_failure_classification_is_available(self):
        self.assertTrue(callable(classify_failure))
        self.assertTrue(callable(solver_advisories))


class TestRelationWriteBookkeeping(HarnessTestCase):
    """A relation submission is a knowledge WRITE, so it must be traceable
    with a maintenance action carrying a PRE state, a knowledge delta and the
    index result."""

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
        self.assertIn("action", result)
        action = h.actions.get(result["action"]["action_id"])
        self.assertIsNotNone(action)
        self.assertEqual(action.action_type, "induce")
        self.assertEqual(action.params.get("knowledge_shape"), "relations")
        self.assertEqual(action.status, "completed")
        delta = result["action"]["knowledge_delta"]
        self.assertTrue(delta["entries_created"])
        self.assertEqual(delta["entry_count_before"], 0)
        self.assertEqual(delta["entry_count_after"], 1)
        self.assertEqual(len(h.actions.query()), before_actions + 1)
        self.assertIn("index_sync", result)

    def test_the_relation_write_is_additive_each_time(self):
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
        # Knowledge is ADDITIVE: each submission creates its OWN numbered
        # entry (a re-submission is never collapsed into the first).
        self.assertTrue(first["action"]["knowledge_delta"]["entries_created"])
        self.assertTrue(second["action"]["knowledge_delta"]["entries_created"])
        self.assertNotEqual(first["relations"][0]["saved"],
                            second["relations"][0]["saved"])
        self.assertEqual(second["action"]["business_result"],
                         "relation_created")
        self.assertEqual(h.sbank.count(), 2)

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

    def test_submitting_no_relation_writes_no_knowledge_but_maintains(self):
        """The framework never runs a statistical induction behind the
        agent's back: no relations means no knowledge write. But a review is
        still a REAL maintenance event — the lifecycle runs and a review
        fact is recorded, so utility maintenance is decoupled from creation.
        """
        h = ORHarness(home=self.home)
        self.addCleanup(h.close)
        self._seed(h)
        before_entries = h.sbank.count()
        before_actions = len(h.actions.query())
        result = h.induce()
        self.assertEqual(result["saved"], 0)
        self.assertTrue(result["no_new_knowledge"])
        # No entry was created...
        self.assertEqual(h.sbank.count(), before_entries)
        # ...but the review ran and is recorded as a maintenance action.
        self.assertEqual(len(h.actions.query()), before_actions + 1)
        self.assertEqual(result["business_result"], "unchanged")
        self.assertIn("revisions", result)

    def test_a_method_less_relation_is_reported_not_refused(self):
        """The material report is a WARNING: the strategy is saved, and the
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

    def test_a_bundle_citation_is_no_longer_supported(self):
        """There is no candidate generator to resolve a ``bundle_id``: the
        evidence must be cited by explicit execution id, and a bundle-shaped
        citation is refused with a reason rather than silently resolved."""
        h = ORHarness(home=self.home)
        self.addCleanup(h.close)
        self._seed(h)
        result = h.induce(relations=[{
            "subject": "principle:bad_bundle",
            "claim": "x",
            "evidence": [{"bundle_id": "cb_missing", "role": "preserved"}]}])
        skipped = result["relations"][0]["skipped"]
        self.assertIn("execution_id", skipped)


class TestConditionalFactPublication(HarnessTestCase):
    """W4: a ``conditional_fact`` strategy publishes on ONE verified
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

    def test_transferable_claim_publishes_with_scope_stated(self):
        """A verified transfer claim publishes; its task span is a reported
        FACT, not a threshold (two tasks are not a proof, one task can still
        reveal a conditional method)."""
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
        self.assertTrue(pub["published"])
        self.assertEqual(pub["distinct_tasks"], 1)
        # Nothing in the publication claims a task-count rule.
        self.assertNotIn("required_tasks", pub)
        self.assertNotIn("independent tasks", " ".join(pub["reasons"]))


if __name__ == "__main__":
    unittest.main()
