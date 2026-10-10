"""r20: the induction material chain — unified knowledge bodies, a knowledge
retrieval document that carries the method, task-representative history
selection, and a de-duplicated query.

What this file asserts, one behaviour per test class:

A. the TWO knowledge channels are UNIONed by ``entry_id`` into ONE
   ``existing_knowledge`` list, each body sent ONCE (``related_history
   .knowledge`` points at it), a cross-family SEMANTIC hit is not vetoed by
   the same-family hint, and a full expansion never changes an entry's
   status/verification/support;
B. ``document_entry`` carries the claim's structured METHOD (steps/why/
   fallback) and its CONDITIONS note, so a change to either moves the digest
   and the knowledge layer must be rebuilt (and recovers after rebuild);
C. history selection is task-representative BEFORE the cut (one row per
   task, then a second same-task attempt preferring a failure/method
   change), the online default stays unchanged, and the batch's own
   executions stay excluded;
D. the query de-duplicates repeated steps, interleaves different attempts,
   carries the saved task text and structural coupling, and its length bound
   never clips the evidence/knowledge bodies.

No real world model is used; a local hashing embedding backend keeps the
retrieval hermetic.
"""
import copy
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tests.harness.helpers import HarnessTestCase  # noqa: E402

from or_harness.api import ORHarness  # noqa: E402
from or_harness.core.schema import (  # noqa: E402
    COST_DIMENSIONS, CostVector, ExecutionRecord, ProblemProfile,
    StrategicEntry)
from or_harness.strategy.embedding_index import (  # noqa: E402
    LocalHashEmbeddingBackend, document_digest, document_entry)
from or_harness.world_model.state import task_text_digest  # noqa: E402

EMBEDDING_ENV_KEYS = ("OR_EMBEDDING_BACKEND", "OR_EMBEDDING_BASE_URL",
                      "OR_EMBEDDING_MODEL", "OR_EMBEDDING_API_KEY")


def _profile(task_id="t1", family="planning"):
    return ProblemProfile(problem_id=task_id, family=family,
                          resource_coupling=0.9, temporal_coupling=0.1,
                          route_complexity=0.8)


def _task(task_id="t1", text="solve by bounding then enumerating"):
    return {"task_id": task_id, "family": "planning", "description": text,
            "spec": {"n_vars": 2, "n_constraints": 3, "n_int_vars": 2}}


def _record(task_id, strategy_id, *, status="optimal", method=None,
            failures=None, text=None, execution_id=None):
    task = _task(task_id, text or f"solve {task_id} by bounding then enumerating")
    rec = ExecutionRecord(
        execution_id=execution_id or f"ex_{task_id}_{strategy_id}",
        task_id=task_id, strategy_id=strategy_id,
        profile_snapshot=_profile(task_id),
        quality={"feasible": status in ("optimal", "feasible"),
                 "objective": 100.0 if status in ("optimal", "feasible")
                 else None, "gap": 0.0, "status": status},
        cost=CostVector(llm_tokens=100, tool_calls=2, solver_runtime_s=1.0,
                        retries=0, latency_s=1.0,
                        measured=set(COST_DIMENSIONS)),
        failures=failures or [],
        solver={"name": "highs", "code_hash": "h1"}, source="executed",
        task_text_digest=task_text_digest(task))
    rec.method_actual = method or {"name": strategy_id, "steps": ["do it"]}
    return rec, task


class KnowledgeCase(HarnessTestCase):
    def setUp(self):
        super().setUp()
        saved = {k: os.environ.pop(k, None) for k in EMBEDDING_ENV_KEYS}
        self.addCleanup(lambda: [os.environ.__setitem__(k, v)
                                 for k, v in saved.items() if v is not None])
        # Hermetic budget: a leaked small budget from another test must not
        # evict this file's records. Restored afterwards.
        saved_budget = os.environ.pop("OR_HARNESS_INDUCTION_MATERIAL_CHARS",
                                      None)
        self.addCleanup(self._restore_budget, saved_budget)
        self.h = ORHarness(home=self.home,
                           embedding=LocalHashEmbeddingBackend())
        self.addCleanup(self.h.close)

    @staticmethod
    def _restore_budget(old):
        if old is None:
            os.environ.pop("OR_HARNESS_INDUCTION_MATERIAL_CHARS", None)
        else:
            os.environ["OR_HARNESS_INDUCTION_MATERIAL_CHARS"] = old

    def _entry(self, *, strategy_id="S1", family="planning",
               claim_text="bound the domain then enumerate", method=None,
               conditions_note=None, status="candidate",
               verification=None, support_n=0):
        claim = {"text": claim_text, "kind": "method",
                 "conditions": {"predicates": {}, "note": conditions_note},
                 "evidence": [], "method": method}
        entry = StrategicEntry(
            entry_id="", strategy_id=strategy_id,
            pattern={"predicates": {"family": family}},
            claim=claim, status=status,
            verification=verification or {},
            support_n=support_n)
        eid = self.h.sbank.add(entry)
        # A direct sbank.add does not sync the vector index (only `induce`
        # does); the semantic channel needs the entry indexed.
        self.h.index_sync.sync_entries([eid])
        return eid


class TestUnifiedKnowledgeBodies(KnowledgeCase):
    """A: the two channels merge into one list, one body per entry."""

    def test_semantic_hit_cross_family_is_not_vetoed(self):
        """A generic entry (different family) reached only by text is added
        to existing_knowledge with related_by=['semantic'] — the same-family
        hint never vetoes it."""
        # A generic entry on a DIFFERENT family, whose text matches t1's
        # method strongly enough to be retrieved.
        eid = self._entry(
            strategy_id="method:generic_bound_enum", family="generic",
            claim_text=("derive a bound from constraints then monotone "
                        "enumerate the remaining dimension"),
            method={"name": "bound-then-enumerate",
                    "steps": ["derive bound", "enumerate one dim"]})
        rec, task = _record(
            "t1", "bound_then_monotone_enum",
            method={"name": "bound then enumerate",
                    "steps": ["derive bound from constraints",
                              "monotone enumerate"]})
        self.h.capture_task_text(task)
        self.h.record(rec)
        out = self.h.induction_material(task_id="t1", related_top_k=5)
        by_id = {k["entry_id"]: k for k in out["existing_knowledge"]}
        self.assertIn(eid, by_id)
        self.assertIn("semantic", by_id[eid]["related_by"])
        # Full body present (not just a projection).
        body = by_id[eid]["body"]
        self.assertEqual(body["claim"]["text"],
                         "derive a bound from constraints then monotone "
                         "enumerate the remaining dimension")
        self.assertIsNotNone(body["claim"]["method"])

    def test_same_id_from_both_channels_has_one_body(self):
        """An entry matched structurally AND semantically keeps ONE body with
        both reasons, and related_history.knowledge only references it."""
        eid = self._entry(
            strategy_id="bound_then_monotone_enum", family="planning",
            claim_text=("derive a bound from constraints then monotone "
                        "enumerate"),
            method={"name": "bound then enumerate",
                    "steps": ["derive bound", "monotone enumerate"]})
        rec, task = _record("t1", "bound_then_monotone_enum")
        self.h.capture_task_text(task)
        self.h.record(rec)
        out = self.h.induction_material(task_id="t1", related_top_k=5)
        ids = [k["entry_id"] for k in out["existing_knowledge"]]
        self.assertEqual(ids.count(eid), 1)
        item = next(k for k in out["existing_knowledge"]
                    if k["entry_id"] == eid)
        self.assertIn("strategy_id", item["related_by"])
        # The body is NOT repeated inside related_history.knowledge.
        refs = out["related_history"]["knowledge"]
        for ref in refs:
            self.assertNotIn("body", ref)
            self.assertEqual(ref["body_ref"], f"existing_knowledge[{eid}]")
            self.assertNotIn("claim_text", ref)

    def test_expansion_does_not_change_verification_or_support(self):
        eid = self._entry(claim_text="a claim", status="candidate",
                          verification={"state": "unverified"}, support_n=0)
        rec, task = _record("t1", "S1")
        self.h.capture_task_text(task)
        self.h.record(rec)
        out = self.h.induction_material(task_id="t1", related_top_k=5)
        item = next(k for k in out["existing_knowledge"]
                    if k["entry_id"] == eid)
        self.assertEqual(item["status"], "candidate")
        self.assertEqual(item["verification_state"], "unverified")
        self.assertEqual(item["support_n"], 0)
        stored = self.h.sbank.get(eid)
        self.assertEqual(stored.support_n, 0)
        self.assertEqual(stored.verification_state, "unverified")


class TestKnowledgeDocumentCarriesMethod(KnowledgeCase):
    """B: document_entry carries the method and conditions."""

    def test_method_and_conditions_are_in_the_document(self):
        eid = self._entry(
            claim_text="bound the domain then enumerate",
            conditions_note="only when the bound is independent of the objective",
            method={"name": "bound-then-enumerate",
                    "steps": ["derive Y <= 233", "scan integer Y"],
                    "why": "the bound makes the scan finite",
                    "fallback": "run the full MILP"})
        entry = self.h.sbank.get(eid)
        doc = document_entry(entry)
        self.assertIn("derive Y <= 233", doc)
        self.assertIn("scan integer Y", doc)
        self.assertIn("the bound makes the scan finite", doc)
        self.assertIn("run the full MILP", doc)
        self.assertIn("only when the bound is independent of the objective",
                      doc)

    def test_editing_the_method_moves_the_digest(self):
        eid = self._entry(claim_text="a claim",
                          method={"name": "m", "steps": ["one"]})
        before = document_digest(document_entry(self.h.sbank.get(eid)))
        entry = self.h.sbank.get(eid)
        entry.claim["method"]["steps"].append("two")
        self.h.sbank.update(entry)
        after = document_digest(document_entry(self.h.sbank.get(eid)))
        self.assertNotEqual(before, after)


class TestKnowledgeIndexRebuild(KnowledgeCase):
    """B: a changed document is stale, and a layered rebuild recovers it."""

    def test_stale_then_rebuild_recovers_and_preserves_entry(self):
        eid = self._entry(
            claim_text="derive a bound from constraints then enumerate",
            method={"name": "bound then enumerate",
                    "steps": ["derive bound", "enumerate"]})
        rec, task = _record("t1", "bound_then_enumerate")
        self.h.capture_task_text(task)
        self.h.record(rec)
        # Build the knowledge layer, then edit the METHOD so its document
        # changes and the stored vector becomes stale.
        self.h.rebuild_index(layer="strategic")
        entry = self.h.sbank.get(eid)
        entry.claim["method"]["steps"].append("recheck the boundary")
        self.h.sbank.update(entry)
        health = self.h.index_health()
        layer = health["layers"]["strategic_knowledge"]
        self.assertGreaterEqual(layer["stale"], 1)
        before = self.h.sbank.get(eid).to_dict()
        # Rebuild ONLY the knowledge layer.
        self.h.rebuild_index(layer="strategic")
        health2 = self.h.index_health()
        self.assertEqual(health2["layers"]["strategic_knowledge"]["stale"], 0)
        # The entry, its number, evidence and effect data did NOT change.
        self.assertEqual(self.h.sbank.get(eid).to_dict(), before)


class TestTaskRepresentativeHistory(KnowledgeCase):
    """C: history selection is task-representative before the cut."""

    def _seed_same_task_many(self):
        # FIVE near-identical attempts of ONE task (would fill every slot),
        # plus one attempt each of two OTHER tasks.
        for i in range(5):
            rec, task = _record(
                "t1", f"bound_then_enum_{i}",
                method={"name": "bound then enumerate",
                        "steps": ["same step"]},
                execution_id=f"ex_t1_{i}")
            self.h.capture_task_text(task)
            self.h.record(rec)
        for tid in ("t2", "t3"):
            rec, task = _record(tid, "bound_then_enum")
            self.h.capture_task_text(task)
            self.h.record(rec)

    def test_other_tasks_enter_the_window(self):
        self._seed_same_task_many()
        out = self.h.induction_material(task_id="t1", related_top_k=5)
        rh = out["related_history"]
        tasks = {e["task_id"] for e in rh["executions"]
                 if "task_id" in e}
        # t2 and t3 are NOT crowded out by t1's five hits.
        self.assertIn("t2", tasks)
        self.assertIn("t3", tasks)
        self.assertGreaterEqual(rh["execution_selection"]["n_distinct_tasks"],
                                2)

    def test_failure_representative_kept_when_slot_remains(self):
        # The BATCH is t1. Related history holds a task t2 with TWO attempts,
        # one failing; when a slot remains the failure is picked as t2's
        # second same-task attempt (a repair/contrast case).
        from or_harness.core.schema import FailureRecord
        rec_batch, task = _record("t1", "m_batch")
        self.h.capture_task_text(task)
        self.h.record(rec_batch)
        rec_ok, task2 = _record("t2", "m_ok")
        self.h.capture_task_text(task2)
        self.h.record(rec_ok)
        rec_fail, task3 = _record(
            "t2", "m_fail", status="error",
            failures=[FailureRecord(error="boom", error_class="ValueError",
                                    recovery_action="retry", attempt=1)],
            execution_id="ex_t2_fail")
        self.h.record(rec_fail)
        out = self.h.induction_material(task_id="t1", related_top_k=3)
        ids = {e.get("execution_id") for e in out["related_history"]
               ["executions"]}
        # Both of t2's attempts are present: the successful one and the
        # failure that shows the repair.
        self.assertIn("ex_t2_m_ok", ids)
        self.assertIn("ex_t2_fail", ids)

    def test_batch_own_executions_remain_excluded(self):
        self._seed_same_task_many()
        out = self.h.induction_material(task_id="t1", related_top_k=5)
        ids = {e.get("execution_id") for e in out["related_history"]
               ["executions"]}
        for i in range(5):
            self.assertNotIn(f"ex_t1_{i}", ids)

    def test_online_default_recall_is_undiversified(self):
        """The online path leaves diversify off: recall() shows no
        execution_selection and the default top-k is unchanged."""
        self._seed_same_task_many()
        task = _task("t1")
        out = self.h.recall(task, top=5)
        vr = out.get("vector_recall") or {}
        self.assertNotIn("execution_selection", vr)


class TestQueryText(KnowledgeCase):
    """D: the query de-duplicates, interleaves and carries the problem."""

    def _seed_repeated(self):
        for i in range(3):
            method = {"name": "bound_then_enum",
                      "steps": ["shared first step", "shared first step",
                                f"attempt {i} unique step"]}
            rec, task = _record("t1", f"S{i}", method=method,
                                execution_id=f"ex_t1_{i}")
            if i == 0:
                self.h.capture_task_text(task)
            self.h.record(rec)

    def test_repeated_steps_deduplicated_and_later_attempt_present(self):
        self._seed_repeated()
        out = self.h.induction_material(task_id="t1", related_top_k=5)
        basis = out["related_history"]["query_basis"]
        text = basis["text"]
        self.assertEqual(text.count("shared first step"), 1)
        # A later attempt's unique step is still expressed.
        self.assertIn("attempt 2 unique step", text)
        self.assertEqual(basis["basis"], "performed")

    def test_task_text_participates_and_missing_is_reported(self):
        rec, task = _record("t1", "S1")
        self.h.capture_task_text(task)
        self.h.record(rec)
        out = self.h.induction_material(task_id="t1", related_top_k=5)
        basis = out["related_history"]["query_basis"]
        self.assertIn("task ", basis["text"])
        self.assertIn("bounding then enumerating", basis["text"])
        self.assertNotIn("task_text", basis["missing"])

    def test_missing_method_and_text_are_reported(self):
        # A record with NO method and NO saved text.
        rec = ExecutionRecord(
            execution_id="ex_bare", task_id="t1", strategy_id="S1",
            profile_snapshot=_profile("t1"),
            quality={"status": "optimal", "feasible": True},
            cost=CostVector(llm_tokens=1, measured={"llm_tokens"}),
            solver={"name": "highs"}, source="executed",
            task_text_digest="")
        self.h.record(rec)
        out = self.h.induction_material(task_id="t1", related_top_k=5)
        basis = out["related_history"]["query_basis"]
        self.assertIn("method", basis["missing"])
        self.assertIn("task_text", basis["missing"])

    def test_query_bound_does_not_clip_bodies(self):
        # A record with a HUGE method step; the query is bounded but the
        # material body keeps the whole step.
        huge = "x" * 3000
        rec, task = _record(
            "t1", "S1", method={"name": "big", "steps": [huge]})
        self.h.capture_task_text(task)
        self.h.record(rec)
        out = self.h.induction_material(task_id="t1", related_top_k=5)
        basis = out["related_history"]["query_basis"]
        self.assertLessEqual(len(basis["text"]), 2400)
        # The BODY (material) keeps the whole step.
        step = out["material"][0]["method"]["actual"]["steps"][0]
        self.assertEqual(len(step), 3000)


class TestKnowledgeStaleReporting(KnowledgeCase):
    """A: a semantic hit whose entry vanished is reported, not dropped."""

    def test_stale_semantic_id_is_reported(self):
        # Directly exercise the merge with a knowledge ref to a missing id.
        from or_harness.world_model.maintenance import (
            _merge_existing_knowledge,
        )
        rec, task = _record("t1", "S1")
        self.h.capture_task_text(task)
        self.h.record(rec)
        out = _merge_existing_knowledge(
            self.h, [rec],
            [{"entry_id": "9999", "strategy_id": "gone",
              "similarity": 0.9, "structural_match": "unknown"}])
        ids = {k["entry_id"] for k in out}
        self.assertIn("9999", ids)
        stale = next(k for k in out if k["entry_id"] == "9999")
        self.assertIn("stale", stale)
        self.assertEqual(stale["related_by"], ["semantic"])


if __name__ == "__main__":
    unittest.main()
