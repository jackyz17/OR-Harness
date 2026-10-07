"""r11 speedup: profile reuse, related-history discovery, and the removal of
the hard-wrap documentation check.

Each class pins one behaviour:

- ``recall(profile=...)`` reuses a caller-supplied profile (one CIR parse)
  and REFUSES a profile that does not describe the task;
- ``induction_material(related_top_k=N)`` surfaces semantically related
  history — cross-name hits, FAILED records included, unpublished entries
  included — and reports no-hit vs retrieval-failure distinctly; ``0``
  disables it;
- the query text is built from the batch's own recorded method.
"""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tests.harness.helpers import HarnessTestCase  # noqa: E402

from or_harness.api import ORHarness  # noqa: E402
from or_harness.core.schema import (  # noqa: E402
    COST_DIMENSIONS, CostVector, ExecutionRecord, ProblemProfile)
from or_harness.strategy.embedding_index import LocalHashEmbeddingBackend  # noqa: E402


def _task(task_id="t1", text="derive a bound then enumerate"):
    return {"task_id": task_id, "family": "planning", "description": text,
            "spec": {"n_vars": 2, "n_constraints": 3, "n_int_vars": 2}}


def _record(task_id, strategy_id, *, status="optimal", method=None,
            text="x"):
    from or_harness.world_model.state import task_text_digest
    task = _task(task_id, text)
    rec = ExecutionRecord(
        execution_id=f"ex_{task_id}_{strategy_id}",
        task_id=task_id, strategy_id=strategy_id,
        profile_snapshot=ProblemProfile(problem_id=task_id, family="planning",
                                        resource_coupling=0.9,
                                        temporal_coupling=0.1,
                                        route_complexity=0.85),
        quality={"feasible": status in ("optimal", "feasible"),
                 "objective": 100.0 if status in ("optimal", "feasible")
                 else None, "gap": 0.0, "status": status},
        cost=CostVector(llm_tokens=100, tool_calls=2, solver_runtime_s=1.0,
                        retries=0, latency_s=1.0,
                        measured=set(COST_DIMENSIONS)),
        solver={"name": "highs", "code_hash": "h1"}, source="executed",
        task_text_digest=task_text_digest(task))
    rec.method_actual = method or {"name": strategy_id, "steps": ["do it"]}
    return rec, task


class TestProfileReuse(HarnessTestCase):

    def test_recall_accepts_a_matching_profile(self):
        h = ORHarness(home=self.home)
        self.addCleanup(h.close)
        task = _task("t1")
        profile = h.profile(task)
        out = h.recall(task, profile=profile)
        self.assertEqual(out["profile"], profile.to_dict())

    def test_recall_refuses_a_mismatched_profile(self):
        h = ORHarness(home=self.home)
        self.addCleanup(h.close)
        profile = h.profile(_task("other"))
        with self.assertRaises(ValueError) as ctx:
            h.recall(_task("t1"), profile=profile)
        self.assertIn("not this task", str(ctx.exception))

    def test_cir_parsed_once_when_profile_is_reused(self):
        """The whole point: profile+recall parse the CIR once, not twice."""
        import or_harness.core.coupling as coupling
        original = coupling.CouplingAwareIR.from_dict
        calls = []

        def counting(data, **kw):
            calls.append(1)
            return original(data, **kw)
        coupling.CouplingAwareIR.from_dict = counting
        self.addCleanup(setattr, coupling.CouplingAwareIR, "from_dict",
                        original)
        h = ORHarness(home=self.home)
        self.addCleanup(h.close)
        task = dict(_task("t1"), coupling={
            "entities": [{"name": "X"}], "decisions": [{"name": "a"}],
            "relations": [{"source": "X", "target": "a"}]})
        profile = h.profile(task)
        before = len(calls)
        h.recall(task, profile=profile)
        self.assertEqual(len(calls), before)  # no re-parse inside recall


class TestRelatedHistory(HarnessTestCase):

    def setUp(self):
        super().setUp()
        saved = {k: os.environ.pop(k, None) for k in
                 ("OR_EMBEDDING_BACKEND", "OR_EMBEDDING_BASE_URL",
                  "OR_EMBEDDING_MODEL", "OR_EMBEDDING_API_KEY")}
        self.addCleanup(lambda: [os.environ.__setitem__(k, v)
                                 for k, v in saved.items() if v is not None])
        self.h = ORHarness(home=self.home,
                           embedding=LocalHashEmbeddingBackend())
        self.addCleanup(self.h.close)

    def _seed(self):
        h = self.h
        for tid, sid, status, text in [
                ("t1", "bound_then_monotone_enum", "optimal",
                 "derive bound from constraints then monotone enumerate"),
                ("t2", "a_different_name_same_idea", "optimal",
                 "bound the domain from constraints and enumerate one dim"),
                ("t3", "scipy_milp", "error",
                 "constraint matrix shape error in scipy")]:
            rec, task = _record(tid, sid, status=status, text=text)
            h.capture_task_text(task)
            h.record(rec)

    def test_related_history_finds_cross_name_and_failed(self):
        self._seed()
        out = self.h.induction_material(task_id="t1", related_top_k=5)
        rh = out["related_history"]
        self.assertTrue(rh["enabled"])
        ids = {e["execution_id"] for e in rh["executions"]}
        # A cross-NAME comparable method is discovered...
        self.assertIn("ex_t2_a_different_name_same_idea", ids)
        # ...and a FAILED record is NOT filtered out.
        statuses = {e["execution_id"]: e["status"] for e in rh["executions"]}
        self.assertIn("error", statuses.values())

    def test_query_is_built_from_the_recorded_method(self):
        self._seed()
        out = self.h.induction_material(task_id="t1", related_top_k=5)
        basis = out["related_history"]["query_basis"]
        self.assertIn("bound_then_monotone_enum", basis["text"])
        self.assertEqual(basis["basis"], "performed")

    def test_top_k_zero_disables_the_channel(self):
        self._seed()
        out = self.h.induction_material(task_id="t1", related_top_k=0)
        self.assertFalse(out["related_history"]["enabled"])

    def test_unavailable_backend_is_a_failure_not_no_hits(self):
        # A harness with NO embedding backend: retrieval cannot run.
        h = ORHarness(home=self.home)
        self.addCleanup(h.close)
        rec, task = _record("t1", "m", text="x")
        h.capture_task_text(task)
        h.record(rec)
        out = h.induction_material(task_id="t1", related_top_k=5)
        rh = out["related_history"]
        self.assertIsNotNone(rh["failure"])
        self.assertIsNone(rh["no_hits"])
        self.assertEqual(rh["executions"], [])


if __name__ == "__main__":
    unittest.main()
