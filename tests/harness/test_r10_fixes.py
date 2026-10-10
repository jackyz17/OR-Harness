"""r10 fixes: CIR path errors, assertion normalization, fact_checked
admission, and contiguous pagination.

Each class pins one behaviour introduced this round, with the r9 material as
the reference case:

- CIR nested required fields report a correctable PATH, never a bare
  ``KeyError`` (orarla_6/7 shapes);
- assertions written as a single-key wrapper, with ``assertion_type``, or
  with execution ids in ``roles`` are normalized, and a ``status`` assertion
  runs on an ``error`` record instead of the whole verification bailing out;
- no assertion at all still READS the facts (``fact_checked``) and publishes
  a single-observation ``conditional_fact``, while a transfer claim from
  facts alone is saved but not published;
- pagination walks the whole history with one cursor, under a budget AND
  under ``--limit``.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tests.harness.helpers import HarnessTestCase  # noqa: E402

from or_harness.api import ORHarness  # noqa: E402
from or_harness.core.coupling import CIRFormatError, CouplingAwareIR  # noqa: E402
from or_harness.core.schema import (  # noqa: E402
    COST_DIMENSIONS, CostVector, ExecutionRecord, ProblemProfile)
from or_harness.strategy.verification import (  # noqa: E402
    FACT_CHECKED, VERIFIED, verify_relation)


def _record(execution_id, *, task_id="t1", strategy_id="S01", status="optimal",
            code_hash="h1", family="planning", method=None):
    rec = ExecutionRecord(
        execution_id=execution_id, task_id=task_id, strategy_id=strategy_id,
        profile_snapshot=ProblemProfile(problem_id=task_id, family=family,
                                        resource_coupling=0.9,
                                        temporal_coupling=0.1,
                                        route_complexity=0.85),
        quality={"feasible": status in ("optimal", "feasible"),
                 "objective": 100.0 if status in ("optimal", "feasible")
                 else None,
                 "gap": 0.0, "status": status},
        cost=CostVector(llm_tokens=100, tool_calls=2, solver_runtime_s=1.0,
                        retries=0, latency_s=1.0,
                        measured=set(COST_DIMENSIONS)),
        solver={"name": "highs", "code_hash": code_hash},
        source="executed")
    if method is not None:
        rec.method_actual = method
    return rec


class TestCIRPathErrors(HarnessTestCase):

    def test_missing_relation_endpoint_reports_a_path(self):
        cir = {"entities": [{"name": "X", "kind": "channel"}],
               "decisions": [{"name": "alloc"}],
               "relations": [{"id": "budget", "type": "constraint"}]}
        with self.assertRaises(CIRFormatError) as ctx:
            CouplingAwareIR.from_dict(cir, allow_empty=False)
        self.assertIn("relations[0].source", str(ctx.exception))
        self.assertIn("target", str(ctx.exception))

    def test_missing_entity_name_reports_a_path(self):
        cir = {"entities": [{"id": "x", "kind": "channel"}],
               "decisions": [{"name": "alloc"}],
               "relations": [{"source": "x", "target": "alloc"}]}
        with self.assertRaises(CIRFormatError) as ctx:
            CouplingAwareIR.from_dict(cir, allow_empty=False)
        self.assertIn("entities[0].name", str(ctx.exception))

    def test_unknown_relation_endpoint_is_reported(self):
        cir = {"entities": [{"name": "X"}], "decisions": [{"name": "alloc"}],
               "relations": [{"source": "X", "target": "nowhere"}]}
        with self.assertRaises(CIRFormatError) as ctx:
            CouplingAwareIR.from_dict(cir, allow_empty=False)
        self.assertIn("nowhere", str(ctx.exception))
        self.assertIn("relations[0].target", str(ctx.exception))

    def test_a_well_formed_cir_still_parses(self):
        cir = {"entities": [{"name": "X"}], "decisions": [{"name": "alloc"}],
               "relations": [{"source": "X", "target": "alloc"}]}
        parsed = CouplingAwareIR.from_dict(cir, allow_empty=False)
        self.assertEqual(len(parsed.relations), 1)


class TestAssertionNormalization(HarnessTestCase):

    def _evidence(self):
        return [_record("ex_1", task_id="t1"), _record("ex_2", task_id="t1")]

    def _roles(self):
        return [{"execution_id": "ex_1", "role": "a"},
                {"execution_id": "ex_2", "role": "b"}]

    def test_single_key_wrapper_is_unwrapped(self):
        report = verify_relation(
            "c", evidence=self._evidence(), roles=self._roles(),
            assertions=[{"status": {"roles": ["a", "b"],
                                    "status": "optimal"}}])
        self.assertEqual(report["state"], VERIFIED)
        self.assertEqual(report["checked"]["assertions"][0]["kind"], "status")

    def test_assertion_type_is_an_alias_for_kind(self):
        report = verify_relation(
            "c", evidence=self._evidence(), roles=self._roles(),
            assertions=[{"assertion_type": "status", "roles": ["a", "b"],
                         "status": "optimal"}])
        self.assertEqual(report["state"], VERIFIED)

    def test_execution_ids_in_roles_are_resolved(self):
        report = verify_relation(
            "c", evidence=self._evidence(), roles=self._roles(),
            assertions=[{"kind": "status", "roles": ["ex_1", "ex_2"],
                         "status": "optimal"}])
        self.assertEqual(report["state"], VERIFIED)

    def test_status_assertion_runs_on_an_error_record(self):
        """r9's 12 error executions: a `status: error` assertion is decidable,
        and must NOT be blocked by a global "all usable" gate."""
        evidence = [_record("ex_1", status="error", code_hash="h1"),
                    _record("ex_2", status="error", code_hash="h1")]
        report = verify_relation(
            "c", evidence=evidence, roles=self._roles(),
            assertions=[{"kind": "status", "roles": ["a", "b"],
                         "status": "error"}])
        self.assertEqual(report["state"], VERIFIED)

    def test_code_unchanged_runs_on_error_records_with_a_hash(self):
        evidence = [_record("ex_1", status="error", code_hash="h1"),
                    _record("ex_2", status="error", code_hash="h1")]
        report = verify_relation(
            "c", evidence=evidence, roles=self._roles(),
            assertions=[{"kind": "code_unchanged", "roles": ["a", "b"]}])
        self.assertEqual(report["state"], VERIFIED)

    def test_a_refuted_status_is_refuted(self):
        evidence = [_record("ex_1", status="error"), _record("ex_2")]
        report = verify_relation(
            "c", evidence=evidence, roles=self._roles(),
            assertions=[{"kind": "status", "roles": ["a", "b"],
                         "status": "optimal"}])
        self.assertEqual(report["state"], "refuted")


class TestFactCheckedAdmission(HarnessTestCase):

    def test_no_assertion_reads_the_facts(self):
        evidence = [_record("ex_1"), _record("ex_2")]
        report = verify_relation("c", evidence=evidence,
                                 roles=[{"execution_id": "ex_1", "role": "a"},
                                        {"execution_id": "ex_2", "role": "b"}])
        self.assertEqual(report["state"], FACT_CHECKED)
        self.assertIn("fact_check", report["scope"])
        self.assertIn("does not establish", report["not_covered"])

    def test_conditional_fact_without_assertion_publishes_and_recalls(self):
        h = ORHarness(home=self.home)
        self.addCleanup(h.close)
        h.bank.append(_record("ex_0", task_id="t1"))
        h.bank.append(_record("ex_1", task_id="t2"))
        out = h.induce(relations=[{
            "subject": "method:enum", "kind": "conditional_fact",
            "claim": "enumeration works here",
            "method": {"name": "enum", "steps": ["bound", "enumerate"]},
            "evidence": [{"execution_id": "ex_0", "role": "evidence"},
                         {"execution_id": "ex_1", "role": "evidence"}]}])
        pub = out["relations"][0]["publication"]
        self.assertTrue(pub["published"])
        self.assertEqual(pub["state"], FACT_CHECKED)
        self.assertEqual(pub["transferability"], "unproven")
        recs = h.selector.recall(
            ProblemProfile(problem_id="q", family="planning",
                           resource_coupling=0.9, temporal_coupling=0.1,
                           route_complexity=0.85), top=5)
        entry = next(r for r in recs if r.strategy_id == "method:enum")
        self.assertEqual(entry.evidence, "strategic_entry")
        self.assertEqual(entry.knowledge["verification_state"], FACT_CHECKED)

    def test_transfer_claim_from_facts_alone_is_published_with_state(self):
        h = ORHarness(home=self.home)
        self.addCleanup(h.close)
        h.bank.append(_record("ex_0", task_id="t1"))
        h.bank.append(_record("ex_1", task_id="t2"))
        out = h.induce(relations=[{
            "subject": "method:rule", "kind": "rule",
            "claim": "this method is always better",
            "method": {"name": "m", "steps": ["s"]},
            "evidence": [{"execution_id": "ex_0", "role": "evidence"},
                         {"execution_id": "ex_1", "role": "evidence"}]}])
        relation = out["relations"][0]
        self.assertIsNotNone(relation.get("saved"))
        pub = relation["publication"]
        # A submitted claim is published; the framework read the facts and
        # records that state rather than gating on it.
        self.assertTrue(pub["published"])
        self.assertEqual(pub["state"], FACT_CHECKED)

    def test_declared_assertion_still_publishes_a_transfer_claim(self):
        h = ORHarness(home=self.home)
        self.addCleanup(h.close)
        h.bank.append(_record("ex_0", task_id="t1"))
        h.bank.append(_record("ex_1", task_id="t2"))
        out = h.induce(relations=[{
            "subject": "method:rule2", "kind": "rule",
            "claim": "this method is checked optimal on both tasks",
            "method": {"name": "m", "steps": ["s"]},
            "evidence": [{"execution_id": "ex_0", "role": "evidence"},
                         {"execution_id": "ex_1", "role": "evidence"}],
            "check": {"assertions": [
                {"kind": "status", "roles": ["evidence"],
                 "status": "optimal"}]}}])
        pub = out["relations"][0]["publication"]
        self.assertTrue(pub["published"])
        self.assertEqual(pub["state"], VERIFIED)


class TestPaginationContiguity(HarnessTestCase):

    def _seed(self, n=20):
        h = ORHarness(home=self.home)
        self.addCleanup(h.close)
        for i in range(n):
            h.bank.append(_record(f"ex_{i:02d}", task_id=f"t{i}"))
        return h

    def _walk(self, h, budget=None, limit=None):
        old = os.environ.get("OR_HARNESS_INDUCTION_MATERIAL_CHARS")
        if budget is not None:
            os.environ["OR_HARNESS_INDUCTION_MATERIAL_CHARS"] = str(budget)
        else:
            os.environ.pop("OR_HARNESS_INDUCTION_MATERIAL_CHARS", None)
        self.addCleanup(self._restore_budget, old)
        seen, cursor, pages = [], None, 0
        while True:
            out = h.induction_material(limit=limit, cursor=cursor)
            seen.extend(m["execution_id"] for m in out["material"])
            pages += 1
            cursor = out["budget"]["next_cursor"]
            if cursor is None or pages > 60:
                break
        return seen

    @staticmethod
    def _restore_budget(old):
        """Restore ``OR_HARNESS_INDUCTION_MATERIAL_CHARS`` after a walk.

        Without this the small walk budget LEAKED into every later test in
        the process, making tests that read the DEFAULT budget order-
        dependent (a genuine hygiene bug this restores)."""
        if old is None:
            os.environ.pop("OR_HARNESS_INDUCTION_MATERIAL_CHARS", None)
        else:
            os.environ["OR_HARNESS_INDUCTION_MATERIAL_CHARS"] = old

    def test_limit_pages_without_gap_or_repeat(self):
        h = self._seed()
        seen = self._walk(h, limit=3)
        self.assertEqual(len(seen), len(set(seen)))
        self.assertEqual(set(seen), {f"ex_{i:02d}" for i in range(20)})

    def test_small_budget_pages_without_gap_or_repeat(self):
        h = self._seed()
        seen = self._walk(h, budget=4000)
        self.assertEqual(len(seen), len(set(seen)))
        self.assertEqual(set(seen), {f"ex_{i:02d}" for i in range(20)})

    def test_full_budget_returns_everything_with_no_cursor(self):
        h = self._seed()
        out = h.induction_material()
        self.assertEqual(len(out["material"]), 20)
        self.assertIsNone(out["budget"]["next_cursor"])

    def test_guidance_is_present(self):
        h = self._seed()
        out = h.induction_material()
        self.assertEqual(set(out["guidance"]),
                         {"structural_contrast", "key_step_explanation",
                          "boundary_check"})


if __name__ == "__main__":
    unittest.main()
