"""Regression tests for the maintenance-path defects (2026-09-29), part 2.

Covers the accepted-operation relation dispatch (#1), the knowledge delta for
relation edits (#6) and the retired-vector cleanup (#7), plus the purpose
distinction between a statistical refresh and a method induction.

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

GOOD_PAYLOAD = {
    "expected_changes": [
        {"metric": "resource_cost", "unit": "s", "direction": "decrease",
         "value": -1.5, "beneficial_direction": "decrease",
         "value_kind": "absolute"},
        {"metric": "normalized_solution_quality", "unit": "1-gap",
         "direction": "unchanged", "beneficial_direction": "increase"},
    ],
    # The learning cost is in the SAVING's own unit and below it, so the
    # prediction is net-positive over the declared window.
    "learning_cost": {"solver_runtime_s": 0.2},
    "verification_conditions": [
        {"condition": "later tasks cost less", "evaluable": True}],
    "evidence_basis": ["capability_evidence.sources.m"],
}


class StubProvider(WorldModelProvider):
    name = "stub"

    def __init__(self):
        self.requests = []

    def predict(self, request, timeout_s=None):
        self.requests.append(request)
        return {"payload": GOOD_PAYLOAD,
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                "error": None, "latency_s": 0.01}


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

    def _accept(self, operation):
        # The prediction is scoped to the evidence behind a bundle the
        # framework builds from the seeded executions (the operation alone
        # declares no scope, and an unscoped acceptance is refused).
        bundles = [b for b in self.h.induction_candidates()
                   if b.get("strategy_id") == operation.get("strategy_id")
                   and b.get("kind") == "new_claim"]
        self.assertTrue(bundles, "the seeded evidence must form a bundle")
        # The framework's frozen, quality-safe saving: 1.5s per task over a
        # 5-task window against a 0.2s one-time cost — net positive.
        provider = StubProvider()
        h = ORHarness(home=self.home, world_model=provider,
                      embedding=self.backend)
        self.addCleanup(h.close)
        prediction = h.predict_capability_evolution(
            operation, task=dict(TASK), bundle=bundles[0],
            horizon="next 5 tasks", horizon_tasks=5)
        recommendation = h.compare_capability_evolution(
            [prediction.prediction_id], horizon_tasks=5)
        self.assertEqual(recommendation["recommendation"], "accept")
        return h.accept_capability_operation(recommendation)


class TestAcceptedOperationRunsDeclaredRelations(MaintenanceCase):
    def test_declared_relation_is_written_not_only_a_statistic(self):
        """#1: the relations an operation DECLARES are the operation. The
        accepted path must run them, not a bare statistical induction."""
        self.seed()
        operation = {
            "operation_type": "induce", "strategy_id": "S-decompose",
            "description": "record the integer-vehicle modeling rule",
            "config": {"relations": [{
                "subject": "principle:integer_vehicle_variable",
                "claim": ("when the cost is per vehicle and the vehicle count "
                          "must be integral, model an integer vehicle "
                          "variable and link load to capacity; do not fold "
                          "the per-vehicle cost into a linear per-unit rate"),
                "conditions": {"predicates": {"family": "routing"}},
                "evidence": [{"execution_id": "ex1", "role": "preserved"},
                             {"execution_id": "ex2", "role": "preserved"}],
            }]},
        }
        accepted = self._accept(operation)
        # A relation was REALLY written.
        entries = self.h.sbank.list()
        self.assertTrue(entries)
        relations = [r for e in entries for r in (e.relations or [])]
        self.assertTrue(relations,
                        "the declared relation must reach the bank")
        self.assertIn("integer vehicle variable", relations[0]["claim"])
        # The knowledge delta reports the created relation entry.
        delta = accepted["operation_result"]["knowledge_delta"]
        self.assertTrue(delta["entries_created"] or delta["entry_changes"])
        # And NOT a bare statistical claim on the strategy id.
        stats = [e for e in entries
                 if e.strategy_id == "S-decompose" and e.support_n > 0]
        self.assertFalse(stats, "a bare statistic was produced instead")

    # The fall-through (no relations declared -> the statistical induce runs)
    # is covered by test_capability_evolution.TestExplicitAcceptance, whose
    # fixtures already exercise the whole accept path; it is not repeated here.


class TestKnowledgeDeltaCoversRelations(MaintenanceCase):
    def test_relation_add_and_revise_both_appear_in_the_delta(self):
        """#6: a relation ADD and a relation REVISION are knowledge changes
        and must appear in the delta, by relation id."""
        self.seed()
        base = {
            "subject": "principle:state",
            "claim": "keep the inventory state",
            "evidence": [{"execution_id": "ex1", "role": "preserved"},
                         {"execution_id": "ex2", "role": "preserved"}],
        }
        first = self.h.induce(relations=[base])
        entry_id = first["relations"][0]["saved"]
        # A SECOND claim on the same entry needs a distinct kind (the id is
        # derived from subject + kind): an ADD.
        added = self.h.induce(relations=[{
            "subject": "principle:state",
            "claim": "a boundary claim under the same subject",
            "kind": "boundary",
            "evidence": [{"execution_id": "ex1", "role": "preserved"},
                         {"execution_id": "ex2", "role": "preserved"}]}])
        self.assertEqual(added["relations"][0]["saved"], entry_id)
        change = [c for c in added["knowledge_delta"]["entry_changes"]
                  if c["entry_id"] == entry_id][0]
        self.assertTrue(change["changed"]["relations"]["added"])
        # Rewriting the FIRST claim (same relation id) is a REVISION, and the
        # moved field is named.
        revised = dict(base, claim="keep the inventory AND the deferral "
                                          "state")
        second = self.h.induce(relations=[revised])
        self.assertEqual(second["business_result"], "relation_updated")
        change = second["knowledge_delta"]["entry_changes"][0][
            "changed"]["relations"]
        self.assertIn("claim",
                      change["revised"][0]["change"]["fields"])


class TestPurposeDistinction(MaintenanceCase):
    def test_a_method_less_cell_is_not_a_method_induction(self):
        """A cell clearing the count gate is a STATISTICAL refresh, never a
        method induction. Material is ``insufficient`` — nothing is
        abstracted from a name and a mean."""
        # Method-LESS evidence: exactly what the count gate does NOT turn
        # into a technique.
        for i in range(2):
            self.h.bank.append(ExecutionRecord(
                execution_id=f"nm{i}", task_id=f"N{i}",
                strategy_id="S-decompose",
                profile_snapshot=self.h.profile(
                    dict(TASK, task_id=f"N{i}")),
                quality={"feasible": True, "objective": 200.0, "gap": 0.0,
                         "status": "optimal"},
                solver={"name": "highs"}))
        bundles = [b for b in self.h.induction_candidates()
                   if b["strategy_id"] == "S-decompose"]
        self.assertTrue(bundles, "the cell must clear the sample-count gate")
        for bundle in bundles:
            self.assertEqual(bundle["kind"], "new_claim")
            self.assertEqual(bundle["purpose"], "statistical_refresh")
            state = self.h.induction_material(
                bundle_id=bundle["bundle_id"])["material"][0][
                    "material_state"]
            self.assertEqual(state["state"], "insufficient")


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
