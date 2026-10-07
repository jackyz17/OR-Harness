"""Numbered entries, additive knowledge, adoption feedback and cold start.

This file pins the round's core behaviours end to end:

1. Entry identity is a framework-assigned NUMBER (`1`, `2`, …), increasing,
   never reused or resequenced, and kept on the cold-archive card.
2. Recall/reference paths identify entries by their FIELD, not by an ``se_``
   prefix; every recall and reference entry point works with plain numbers.
3. Knowledge is ADDITIVE: a re-submission is a new numbered entry, never an
   in-place rewrite or merge; different structures never collide just because
   they share a solver.
4. Publication is the agent's decision: submission offers knowledge and the
   framework records the agent's own verification rather than gating on it.
5. Adoption feedback: declaring `used_entry_ids` ties the entry to the run's
   result; a mere recall never counts as a successful use; an empty adoption
   is a recorded fact, not an omission.
6. Utility maintenance runs even with NO new knowledge, and repeated runs do
   not double-count.
7. Cold start: the material reports the banks' `memory_state` and an empty
   bank never gates induction.
"""
import json
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tests.harness.helpers import HarnessTestCase  # noqa: E402

from or_harness.api import ORHarness  # noqa: E402
from or_harness.core.schema import CostVector, StrategicEntry  # noqa: E402
from or_harness.strategy.embedding_index import (  # noqa: E402
    LocalHashEmbeddingBackend,
)
from or_harness.strategy.selector import is_publishable  # noqa: E402


class NumberedCase(HarnessTestCase):
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

    def entry(self, strategy_id="S01", predicates=None, verified=True,
              verified_state="verified"):
        return StrategicEntry(
            entry_id="", strategy_id=strategy_id,
            pattern={"predicates": predicates or {"family": "routing"}},
            expected_quality_hat=0.9, quality_interval=(0.5, 1.0),
            expected_cost_hat=CostVector(llm_tokens=100.0,
                                         measured={"llm_tokens"}),
            failure_prob=0.1, support_n=2,
            verification=({"state": verified_state, "claim": "c",
                           "conclusion": "check"} if verified else {}))


class TestNumberAssignment(NumberedCase):
    def test_numbers_are_increasing_and_never_reused(self):
        a = self.h.sbank.add(self.entry("S01"))
        b = self.h.sbank.add(self.entry("S02"))
        self.assertEqual([a, b], ["1", "2"])
        self.h.retire(b, reason="test")
        c = self.h.sbank.add(self.entry("S03"))
        self.assertEqual(c, "3", "a retired number is never reused")

    def test_archive_card_keeps_the_number(self):
        eid = self.h.sbank.add(self.entry("S01"))
        card = self.h.sbank.retire(eid, reason="test")
        self.assertEqual(card.entry_id, eid)
        self.assertEqual(self.h.sbank.cold_archive()[0].entry_id, eid)

    def test_agent_supplied_id_is_ignored(self):
        """The agent cannot invent an identity: ``add`` overwrites it."""
        e = self.entry("S01")
        e.entry_id = "se_agent_invented"
        assigned = self.h.sbank.add(e)
        self.assertEqual(assigned, "1")
        self.assertEqual(self.h.sbank.get("1").entry_id, "1")

    def test_next_number_preview(self):
        self.assertEqual(self.h.sbank.next_number(), 1)
        self.h.sbank.add(self.entry("S01"))
        self.assertEqual(self.h.sbank.next_number(), 2)


class TestNumberedReferencesWork(NumberedCase):
    """Every recall/reference entry point works with a plain number."""

    def _record(self):
        return self.make_record(task_id="t1", strategy_id="S01")

    def test_recall_reports_the_number(self):
        eid = self.h.sbank.add(self.entry("S01"))
        self.h.index_sync.sync_entries([eid])
        out = self.h.recall({"task_id": "q", "family": "routing",
                             "annotations": {"coupling": {
                                 "resource_coupling": 0.9}}})
        recs = [r for r in out["recommendations"]
                if r["evidence"] == "strategic_entry"]
        self.assertTrue(recs)
        self.assertIn(eid, recs[0]["evidence_refs"])

    def test_vector_recall_keyed_on_the_number(self):
        eid = self.h.sbank.add(self.entry("S01"))
        self.h.index_sync.sync_entries([eid])
        out = self.h.recall({"task_id": "q", "family": "routing",
                             "description": "routing",
                             "annotations": {"coupling": {
                                 "resource_coupling": 0.9}}})
        ids = {k["entry_id"]
               for k in out["vector_recall"]["strategic_knowledge"]}
        self.assertIn(eid, ids)


class TestAdditiveKnowledge(NumberedCase):
    def test_different_structures_same_solver_are_separate_entries(self):
        """Two claims with the same strategy id but different predicates are
        SEPARATE entries: a shared solver name does not merge them."""
        a = self.h.sbank.add(self.entry(
            "S01", predicates={"family": "routing"}))
        b = self.h.sbank.add(self.entry(
            "S01", predicates={"family": "scheduling"}))
        self.assertNotEqual(a, b)
        self.assertEqual(self.h.sbank.count(), 2)

    def test_publication_is_not_a_content_gate(self):
        for state in ("verified", "fact_checked", "unverified",
                      "insufficient_evidence", "refuted"):
            e = self.entry(verified_state=state)
            self.assertTrue(is_publishable(e))
            self.assertTrue(e.is_published)


class TestAdoptionFeedback(NumberedCase):
    def _work(self, tag):
        work = Path(self.home) / f"ws_{tag}"
        work.mkdir(parents=True, exist_ok=True)
        script = work / "solve.py"
        script.write_text(
            "import json\n"
            "with open('result.json', 'w') as fh:\n"
            "    json.dump({'status': 'optimal', 'objective_value': 1.0,"
            " 'objective_bound': 1.0, 'runtime_seconds': 0.01}, fh)\n",
            encoding="utf-8")
        return str(script), str(work)

    def test_declared_adoption_is_recorded(self):
        eid = self.h.sbank.add(self.entry("S01"))
        code, ws = self._work("adopt")
        record = self.h.execute(
            {"task_id": "t1", "family": "routing",
             "annotations": {"coupling": {"resource_coupling": 0.9}}},
            "S01", code, ws, solver="highs", episode_id="ep1",
            used_entry_ids=[eid])
        used = record.execution_features["used_entries"]
        self.assertEqual(used["entry_ids"], [eid])
        self.assertEqual(used["known_entry_ids"], [eid])
        self.assertEqual(used["unknown_entry_ids"], [])

    def test_empty_adoption_is_a_recorded_fact(self):
        code, ws = self._work("empty")
        record = self.h.execute(
            {"task_id": "t1", "family": "routing",
             "annotations": {"coupling": {"resource_coupling": 0.9}}},
            "S01", code, ws, solver="highs", episode_id="ep1",
            used_entry_ids=[])
        used = record.execution_features["used_entries"]
        self.assertEqual(used["entry_ids"], [])
        self.assertEqual(used["known_entry_ids"], [])

    def test_unstated_adoption_is_not_recorded(self):
        code, ws = self._work("none")
        record = self.h.execute(
            {"task_id": "t1", "family": "routing",
             "annotations": {"coupling": {"resource_coupling": 0.9}}},
            "S01", code, ws, solver="highs", episode_id="ep1")
        self.assertNotIn("used_entries", record.execution_features)

    def test_recall_without_adoption_produces_no_forward_check(self):
        """Being RECALLED is not adoption: without a declared adoption there
        is no forward check for the entry's strategy."""
        eid = self.h.sbank.add(self.entry("S01"))
        self.h.index_sync.sync_entries([eid])
        # Recall it (marks last_consulted_at) but DO NOT adopt it.
        self.h.recall({"task_id": "q", "family": "routing",
                       "annotations": {"coupling": {"resource_coupling": 0.9}}})
        code, ws = self._work("recall_only")
        record = self.h.execute(
            {"task_id": "t1", "family": "routing",
             "annotations": {"coupling": {"resource_coupling": 0.9}}},
            "S01", code, ws, solver="highs", episode_id="ep1",
            used_entry_ids=[])
        checks = self.h._check_predictions(record)
        self.assertEqual(checks, [], "recall is not adoption")

    def test_unknown_number_is_kept_as_a_citation(self):
        code, ws = self._work("unknown")
        record = self.h.execute(
            {"task_id": "t1", "family": "routing",
             "annotations": {"coupling": {"resource_coupling": 0.9}}},
            "S01", code, ws, solver="highs", episode_id="ep1",
            used_entry_ids=["99"])
        used = record.execution_features["used_entries"]
        self.assertEqual(used["unknown_entry_ids"], ["99"])
        self.assertEqual(used["known_entry_ids"], [])


class TestUtilityMaintenanceWithoutCreation(NumberedCase):
    def test_empty_induce_runs_maintenance(self):
        eid = self.h.sbank.add(self.entry(
            "S01", verified_state="verified"))
        before_actions = len(self.h.actions.query())
        result = self.h.induce()
        self.assertTrue(result["no_new_knowledge"])
        self.assertEqual(self.h.sbank.count(), 1)
        self.assertEqual(len(self.h.actions.query()), before_actions + 1)
        self.assertIn("revisions", result)
        self.assertEqual(result["business_result"], "unchanged")

    def test_repeated_empty_induce_is_idempotent(self):
        self.h.sbank.add(self.entry("S01", verified_state="verified"))
        first = self.h.induce()
        second = self.h.induce()
        # Neither creates knowledge; both ran the maintenance pass.
        self.assertEqual(first["knowledge_delta"]["entries_created"], [])
        self.assertEqual(second["knowledge_delta"]["entries_created"], [])
        self.assertEqual(self.h.sbank.count(), 1)


class TestColdStart(NumberedCase):
    def test_both_banks_empty_is_reported(self):
        out = self.h.induction_material()
        state = out["memory_state"]
        self.assertEqual(state["state"], "both_banks_empty")
        self.assertTrue(state["both_empty"])

    def test_strategic_empty_evidence_present(self):
        self.h.bank.append(self.make_record(task_id="t1", strategy_id="S01"))
        out = self.h.induction_material()
        self.assertEqual(out["memory_state"]["state"], "strategic_bank_empty")
        self.assertFalse(out["memory_state"]["both_empty"])

    def test_empty_bank_does_not_gate_induction(self):
        """With no prior knowledge, a first claim can still be submitted."""
        self.h.bank.append(self.make_record(task_id="t1", strategy_id="S01"))
        rec = self.h.bank.query()[0]
        out = self.h.induce(relations=[{
            "subject": "method:first", "claim": "a first method",
            "evidence": [{"execution_id": rec.execution_id, "role": "e"}]}])
        self.assertIsNotNone(out["relations"][0]["saved"])
        self.assertEqual(self.h.sbank.count(), 1)


if __name__ == "__main__":
    unittest.main()
