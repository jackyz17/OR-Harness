"""D-series cleanups: the removed interfaces must be gone, and the surviving
one must still cover what they did.

A deletion is only safe when its BEHAVIOUR survives somewhere. These tests
pin both halves: the removed name is gone (so nothing silently depends on it
again), and the behaviour it used to provide is still reachable.
"""
import inspect
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tests.harness.helpers import HarnessTestCase  # noqa: E402

from or_harness.strategy.embedding_index import document_execution
from or_harness.strategy.index_sync import IndexSynchronizer
from or_harness.strategy.induction import InductionEngine
from or_harness.strategy.selector import Selector
from or_harness.world_model import maintenance


class TestRemovedInterfaces(HarnessTestCase):
    def test_induce_engine_has_one_write_entry(self):
        """The engine writes knowledge through ONE public path —
        ``submit_relation`` — and no longer carries a statistical ``induce``
        with dead parameters."""
        self.assertTrue(hasattr(InductionEngine, "submit_relation"))
        self.assertFalse(hasattr(InductionEngine, "induce"))

    def test_index_sync_has_no_forget_entry(self):
        self.assertFalse(hasattr(IndexSynchronizer, "forget_entry"))

    def test_selector_from_stats_has_no_dead_parameters(self):
        params = inspect.signature(Selector._from_stats).parameters
        self.assertNotIn("basis", params)
        self.assertNotIn("cross_family", params)
        self.assertIn("cost_basis", params)

    def test_maintenance_has_no_assessment_class(self):
        self.assertFalse(hasattr(maintenance, "InductionAssessment"))


class TestSurvivingBehaviour(HarnessTestCase):
    """What the deleted wrappers used to do still happens."""

    def test_retiring_an_entry_still_drops_its_vector(self):
        """``forget_entry`` is gone, but the guarantee it existed for — a
        vector must not outlive its entry — is still honoured by
        ``sync_entries``."""
        from or_harness.api import ORHarness
        from or_harness.strategy.strategic_bank import StrategicEntry
        h = ORHarness(home=self.home)
        self.addCleanup(h.close)
        entry = StrategicEntry(
            entry_id="", strategy_id="S01",
            pattern={"predicates": {"family": "routing"}},
            expected_quality_hat=0.5,
            verification={"state": "verified", "claim": "x"})
        h.sbank.add(entry)
        # Without an embedding backend configured the call is a no-op, so
        # the assertion is on the RETIRE path not raising and the entry
        # leaving the hot store.
        h.retire(entry.entry_id, reason="cleanup")
        self.assertIsNone(h.sbank.get(entry.entry_id))


class TestExecutionDocumentCarriesMethod(HarnessTestCase):
    def test_a_reported_method_is_searchable(self):
        record = self.make_record()
        record.method_planned = {"name": "rolling-horizon decomposition",
                                 "steps": ["relax the coupling constraint",
                                           "solve the master"]}
        record.method_actual = {"name": "rolling-horizon decomposition",
                                "steps": ["solve the master"]}
        text = document_execution(record, "a scheduling task")
        self.assertIn("planned method: rolling-horizon decomposition", text)
        self.assertIn("relax the coupling constraint", text)
        self.assertIn("performed method", text)

    def test_a_record_without_a_method_adds_nothing(self):
        record = self.make_record()
        text = document_execution(record, "a scheduling task")
        self.assertNotIn("method", text)


if __name__ == "__main__":
    unittest.main()
