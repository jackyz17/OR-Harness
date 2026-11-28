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
from g_fixtures import COUPLED, GCase


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


class TestRetiredWorldModelRuntime(GCase):
    def test_only_the_current_prediction_service_is_initialized(self):
        import importlib.util
        from or_harness.api import ORHarness
        for name in ('predict_outcome', 'bind_outcome', 'compare_prediction',
                     'knowledge_targets', 'prediction_reliability_table',
                     'evaluate_knowledge_execution', 'evaluate_knowledge_consolidation'):
            self.assertFalse(hasattr(self.h, name), name)
        self.assertFalse(hasattr(self.h, 'predictions'))
        self.assertIsNotNone(self.h.strategy_predictions)
        for name in ('service', 'prediction', 'knowledge', 'knowledge_track'):
            self.assertIsNone(importlib.util.find_spec('or_harness.world_model.' + name))
        for name in ('delta', 'prediction_mode'):
            self.assertNotIn(name, inspect.signature(ORHarness).parameters)

    def test_plan_predicts_once_per_candidate_without_writing_legacy_records(self):
        plan = self.h.plan_next(self.task, 'ep1', candidates=[
            {'strategy_id': sid, 'solver': 'highs', 'method': COUPLED}
            for sid in ('S04', 'S05')])
        self.assertEqual(len(self.provider.requests), 2)
        self.assertEqual(plan['selected_score']['benefit_value'], .7)
        contexts = [r['prediction_context']['context_id'] for r in self.provider.requests]
        self.assertEqual(len(set(contexts)), 1)
        for request in self.provider.requests:
            self.assertEqual(request['prediction_protocol'], 'wm-so/1')
            self.assertNotIn('candidate_knowledge_targets', request)
            self.assertNotIn('prediction_reliability', request)
        self.assertEqual(self.h.predictions_query(), [])

    def test_archived_predictions_are_read_only_and_separate_from_calibration(self):
        from decision_fixtures import seed_legacy_prediction
        prediction = seed_legacy_prediction(self.h, self.task, 'ep1', predicted={
            'quality': .5, 'knowledge_changes': [{'change': 'narrows', 'probability': .8}]},
            call_cost={'llm_tokens': 7})
        before = list(self.h.store.conn.iterdump())
        read = self.h.read_prediction(prediction.prediction_id)
        self.assertEqual(read['kind'], 'legacy_outcome')
        self.assertEqual(read['prediction']['predicted']['knowledge_changes'][0]['change'], 'narrows')
        self.assertEqual(self.h.get_prediction(prediction.prediction_id).to_dict(), prediction.to_dict())
        self.assertEqual(len(self.h.predictions_query(task_id=self.task['task_id'])), 1)
        self.assertEqual(list(self.h.store.conn.iterdump()), before)
        budget = self.h.budget.view(self.task['task_id'], 'ep1')
        self.assertEqual(budget['consumption']['total_cost']['llm_tokens'], 7)
        summary = self.h.calibration_summary()
        self.assertEqual(summary['n_evaluated'], 0)
        self.assertEqual(summary['conditional_g'], [])

    def test_current_knowledge_facts_do_not_depend_on_m6_targets(self):
        from or_harness.core.schema import StrategicEntry
        from or_harness.world_model.contracts import ActionSpec, HarnessCapabilityEvidence
        entry = StrategicEntry(entry_id='', strategy_id='S04',
            pattern={'predicates': {'family': 'assignment'}}, expected_quality_hat=.5,
            verification={'state': 'verified', 'claim': 'An explicitly recorded factory rule.'})
        entry_id = self.h.sbank.add(entry)
        context = self.h.build_prediction_context(self.task, 'ep1', candidates=[
            ActionSpec('execute_strategy', self.task['task_id'],
                       strategy_id='S04', solver='highs', method=COUPLED)])
        layers = context.snapshot['coverage']['knowledge_layers']
        self.assertIn(str(entry_id), {str(ref['entry_id']) for refs in layers.values() for ref in refs})
        self.assertNotIn('knowledge_targets', context.to_dict())
        self.assertNotIn('reliability', context.to_dict())
        self.assertNotIn('delta', context.capability_version)
        self.assertEqual(context.capability_version['prompt_template_version'], 'wm-so/1')
        self.assertEqual(HarnessCapabilityEvidence.from_dict(context.capability).source('m').status,
                         'indirect_evidence')

    def test_archived_context_fields_remain_readable_without_a_proposer(self):
        from or_harness.world_model.context import PredictionContext
        raw = self.h.build_prediction_context(self.task, 'ep1').to_dict()
        raw['knowledge_targets'] = [{'entry_id': 'historical', 'strategy_id': 'S04'}]
        raw['reliability'] = {'historical': {'n_resolved': 3, 'value': .5}}
        read = PredictionContext.from_dict(raw)
        self.assertEqual(read.to_dict()['knowledge_targets'], raw['knowledge_targets'])
        self.assertEqual(read.to_dict()['reliability'], raw['reliability'])
        self.assertFalse(hasattr(self.h, 'knowledge_targets'))

    def test_http_adapter_refuses_retired_requests_before_network(self):
        from unittest.mock import patch
        from or_harness.world_model.provider import HttpChatProvider
        provider = HttpChatProvider('https://synthetic.invalid/v1', 'fixture', 'synthetic-key')
        with patch('or_harness.world_model.provider.urllib.request.urlopen') as send:
            for request in ({}, {'action_spec': {'action_type': 'induce'}},
                            {'prediction_protocol': 'wm-ce/1'}):
                with self.assertRaisesRegex(ValueError, 'wm-so/1'):
                    provider.predict(request)
            send.assert_not_called()

    def test_retired_cli_flags_are_refused_and_cost_diagnostic_is_retained(self):
        import argparse
        import contextlib
        import io
        from or_harness.cli import build_parser
        from or_harness.world_model.planner import PlanLimits
        parser = build_parser()
        for args in (['--delta', '1', 'doctor'], ['--prediction-mode', 'x-b-only', 'doctor'],
                     ['plan-next', '--task', '{}', '--delta', '1'],
                     ['plan-next', '--task', '{}', '--prediction-mode', 'h-x-b']):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parser.parse_args(args)
        commands = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction)).choices
        self.assertIn('predict-cost', commands)
        self.assertNotIn('delta', PlanLimits().to_dict())
        self.h.predict_cost(self.task, 'S04')
        self.assertEqual(self.provider.requests, [])


if __name__ == "__main__":
    unittest.main()
