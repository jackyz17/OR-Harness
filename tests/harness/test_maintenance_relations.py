"""Regression tests for the maintenance-path defects (2026-09-29), part 2.

Covers relation knowledge deltas, material content and retired-vector cleanup
through the ordinary induction and rule-maintenance interfaces.

    PYTHONPATH=src python -m unittest tests.harness.test_maintenance_relations -q
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tests.harness.helpers import HarnessTestCase  # noqa: E402

from or_harness.api import ORHarness  # noqa: E402
from or_harness.core.schema import ExecutionRecord  # noqa: E402
from or_harness.strategy.embedding_index import (  # noqa: E402
    LAYER_STRATEGIC,
    LocalHashEmbeddingBackend,
)
from or_harness.world_model.provider import WorldModelProvider  # noqa: E402

TASK = {"task_id": "transport", "family": "routing",
        "description": "ship boxes",
        "annotations": {"coupling": {"resource_coupling": 0.9,
                                     "temporal_coupling": 0.1}}}

INTEGER_CAR_METHOD = {
    "name": "integer vehicle variable",
    "steps": ["int car", "boxes <= 10 * car", "min 100 * car"]}



class StubProvider(WorldModelProvider):
    name = "stub"

    def __init__(self):
        self.requests = []

    def predict(self, request, timeout_s=None):
        self.requests.append(request)
        raise AssertionError("Rule maintenance must not request a capability prediction")


class MaintenanceCase(HarnessTestCase):
    def setUp(self):
        super().setUp()
        saved = {k: os.environ.pop(k, None) for k in
                 ("OR_EMBEDDING_BASE_URL", "OR_EMBEDDING_MODEL",
                  "OR_EMBEDDING_API_KEY", "OR_EMBEDDING_BACKEND")}

        def restore():
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v
        self.addCleanup(restore)
        self.backend = LocalHashEmbeddingBackend()
        self.provider = StubProvider()
        self.h = ORHarness(home=self.home, world_model=self.provider,
                           embedding=self.backend)
        self.addCleanup(self.h.close)

    def seed(self, ids=("ex1", "ex2"), strategy="S-decompose"):
        for i, eid in enumerate(ids):
            self.h.bank.append(ExecutionRecord(
                execution_id=eid, task_id=f"T{i}", strategy_id=strategy,
                profile_snapshot=self.h.profile(
                    dict(TASK, task_id=f"T{i}")),
                quality={"feasible": True, "objective": 200.0, "gap": 0.0,
                         "status": "optimal"},
                solver={"name": "highs"},
                method_actual=INTEGER_CAR_METHOD))




    # The fall-through (no relations declared -> the statistical induce runs)
    # is covered by test_capability_evolution.TestExplicitAcceptance, whose
    # fixtures already exercise the whole accept path; it is not repeated here.


class TestKnowledgeDeltaCoversRelations(MaintenanceCase):
    def test_relation_add_is_additive_and_recorded_in_the_delta(self):
        """A relation submission is a knowledge ADD: knowledge is additive,
        so a second submission (even with changed text) is a NEW numbered
        entry, and both creations appear in their actions' deltas."""
        self.seed()
        base = {
            "subject": "principle:state",
            "claim": "keep the inventory state",
            "evidence": [{"execution_id": "ex1", "role": "preserved"},
                         {"execution_id": "ex2", "role": "preserved"}],
        }
        first = self.h.induce(relations=[base])
        entry_id = first["relations"][0]["saved"]
        self.assertIn(entry_id,
                      first["knowledge_delta"]["entries_created"])
        # Submitting the same claim again with a CHANGED assertion is a NEW
        # entry (never an in-place rewrite), and the new creation is named in
        # the delta.
        revised = dict(base, claim="keep the inventory AND the deferral "
                                          "state")
        second = self.h.induce(relations=[revised])
        second_id = second["relations"][0]["saved"]
        self.assertNotEqual(second_id, entry_id)
        self.assertEqual(second["business_result"], "relation_created")
        self.assertIn(second_id,
                      second["knowledge_delta"]["entries_created"])
        self.assertNotIn(entry_id,
                         second["knowledge_delta"]["entries_created"])


class TestMaterialReport(MaintenanceCase):
    def test_a_method_less_cell_reports_its_missing_content(self):
        """A cell clearing the count gate does NOT turn a name and a mean
        into a technique. The batch material names the missing content
        (``method_performed``) rather than asserting a technique."""
        for i in range(2):
            self.h.bank.append(ExecutionRecord(
                execution_id=f"nm{i}", task_id=f"N{i}",
                strategy_id="S-decompose",
                profile_snapshot=self.h.profile(
                    dict(TASK, task_id=f"N{i}")),
                quality={"feasible": True, "objective": 200.0, "gap": 0.0,
                         "status": "optimal"},
                solver={"name": "highs"}))
        material = self.h.induction_material(strategy_id="S-decompose")
        self.assertEqual(material["count"], 2)
        for entry in material["material"]:
            self.assertEqual(entry["method"]["basis"], "none")
            self.assertIsNone(entry["method"]["planned"])
            self.assertIsNone(entry["method"]["actual"])


class TestRetiredVectorIsRemoved(MaintenanceCase):
    def test_retiring_an_entry_removes_its_vector(self):
        """#7: a retired entry's vector must leave the index, not merely be
        filtered at recall time."""
        self.seed()
        self.h.induce(relations=[{
            "subject": "principle:retire_me",
            "claim": "a claim that will be retired",
            "evidence": [{"execution_id": "ex1", "role": "preserved"},
                         {"execution_id": "ex2", "role": "preserved"}]}])
        entry_id = self.h.sbank.list()[0].entry_id
        self.assertIsNotNone(self.h.sbank.get(entry_id).claim)
        # Index the entry, then confirm it is present.
        self.h.index_sync.sync_entries()
        present = {item["id"] for item in
                   self.h.embedding_index.items(LAYER_STRATEGIC)}
        self.assertIn(entry_id, present)
        # Retire it: the vector must go.
        self.h.retire(entry_id, reason="repro")
        after = {item["id"] for item in
                 self.h.embedding_index.items(LAYER_STRATEGIC)}
        self.assertNotIn(entry_id, after,
                         "the retired entry's vector must be removed")


if __name__ == "__main__":
    unittest.main()
