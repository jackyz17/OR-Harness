"""Versioned G observations on isolated execute/check/closeout evidence."""
import copy
from types import SimpleNamespace

from g_fixtures import GCase, SPLIT
from or_harness.world_model.conditions import get_value
from or_harness.world_model.contracts import (
    FAITHFUL_G_OBSERVATION_VERSION, LEGACY_FAITHFUL_G_OBSERVATION_VERSION,
    validate_strategy_outcome,
)
from or_harness.world_model.episode_closeout import (
    _live_evaluation, build_paired_feedback, get_evaluation,
)
from or_harness.world_model.g_signal import observe_faithful_g
from or_harness.world_model.history_retrieval import global_g_statistics


class TestGObservationV2(GCase):
    def observe(self, prediction, record):
        summary = SimpleNamespace(benefit={}, eligibility={})
        observe_faithful_g(summary, [record] if record else [], prediction, store=self.h.store)
        return summary

    def test_agent_insufficiency_and_external_evidence_gaps_are_distinct(self):
        for episode, status, label in (
                ('agent', 'agent_insufficient', 0),
                ('lost', 'unavailable', None),
                ('standard', 'invalid_standard', None),
                ('uncertain', 'unknown', None)):
            with self.subTest(episode=episode):
                p = self.predict_g(episode=episode)
                r = self.run_g(p)
                checked = self.check_g(r, coverage='partial', check_spec={}, evidence_status=status)
                self.assertEqual(checked['state'], 'insufficient')
                benefit = self.close_g(episode)['evaluations'][0]['benefit']
                self.assertEqual(benefit.get('observed'), label)
                self.assertEqual(benefit['attribution']['evidence_status'], status)
                self.assertTrue(benefit['reason'])

    def test_agent_insufficient_evidence_overrides_a_passing_partial_check(self):
        p = self.predict_g()
        r = self.run_g(p)
        checked = self.check_g(r, evidence_status='agent_insufficient')
        self.assertEqual(checked['state'], 'passed')
        self.assertEqual(self.close_g()['evaluations'][0]['benefit']['observed'], 0)

    def test_unresolved_failure_is_not_erased_by_a_passing_local_check(self):
        for cause in ('reference_conflict', 'candidate_method'):
            with self.subTest(cause=cause):
                p = self.predict_g(episode=cause)
                r = self.run_g(p)
                checked = self.check_g(r, failure=cause)
                self.assertEqual(checked['state'], 'passed')
                self.assertEqual(self.close_g(cause)['evaluations'][0]['benefit']['observed'], 0)

    def test_invalid_standard_can_be_replaced_by_supported_original_task_checks(self):
        p = self.predict_g()
        r = self.run_g(p)
        checked = self.check_g(r, failure='reference_conflict', evidence_status='invalid_standard',
            check_spec={'reference_objective': 999, 'reference_source': 'bench_declared'})
        self.assertEqual(checked['state'], 'failed')
        closed = self.close_g()
        stored = get_evaluation(self.h.store, closed['evaluations'][0]['evaluation_id'])
        self.assertNotIn('observed', stored.benefit)
        # New independent evidence resolves the faulty standard on the same
        # committed execution. Neither the solve nor its answer is rerun.
        fixed = self.check_g(r, evidence_status='sufficient', check_spec={
            'semantic_probe': [
                {'path': 'execution_features.solution_variables.BE', 'equals': 1},
                {'path': 'execution_features.solution_variables.CFC', 'equals': 1}],
            'reference_objective': 100, 'reference_source': 'independent',
            'reference_version': 'verified-factory-reference/1'})
        self.assertEqual(fixed['state'], 'passed')
        derived, correction = _live_evaluation(self.h, stored)
        self.assertEqual(derived.benefit['observed'], 1)
        self.assertEqual(derived.evaluation_id, stored.evaluation_id)
        self.assertTrue(correction)
        self.assertNotIn('observed', get_evaluation(self.h.store, stored.evaluation_id).benefit)
        self.assertEqual(len(self.h.strategy_prediction_evaluations()), 1)

    def test_missing_attribution_cannot_be_replaced_by_an_insufficient_check(self):
        p = self.predict_g()
        r = self.run_g(p)
        checked = self.check_g(r, check_spec={}, evidence=False)
        self.assertEqual(checked['state'], 'insufficient')
        benefit = self.close_g()['evaluations'][0]['benefit']
        self.assertNotIn('observed', benefit)
        self.assertIn('missing or invalid sourced G attribution', benefit['reason'])

    def test_framework_loss_or_revision_mismatch_never_fabricates_a_negative(self):
        p = self.predict_g()
        r = self.run_g(p)
        checked = self.check_g(r, coverage='partial', evidence_status='agent_insufficient')
        original = self.h.bank.get(r.execution_id)
        self.assertEqual(self.observe(p, original).benefit['observed'], 0)
        damaged = copy.deepcopy(original)
        damaged.execution_features['task_check']['checks'] = []
        self.assertNotIn('observed', self.observe(p, damaged).benefit)
        damaged = copy.deepcopy(original)
        damaged.execution_features['artifacts'].pop('workspace_result')
        self.assertNotIn('observed', self.observe(p, damaged).benefit)
        with self.h.store.transaction() as conn:
            conn.execute('DELETE FROM condition_values WHERE value_id=?', (checked['g_assessment_ref'],))
        self.assertNotIn('observed', self.observe(p, original).benefit)

    def test_unexecuted_candidate_and_invalid_method_binding_stay_unknown(self):
        p = self.predict_g()
        self.assertNotIn('observed', self.observe(p, None).benefit)
        r = self.run_g(p)
        self.check_g(r, coverage='partial')
        damaged = copy.deepcopy(self.h.bank.get(r.execution_id))
        damaged.method_planned = SPLIT
        self.assertNotIn('observed', self.observe(p, damaged).benefit)

    def test_unknown_cause_negative_reaches_errors_conditional_and_provider_feedback(self):
        p = self.predict_g(SPLIT)
        r = self.run_g(p, split=True)
        self.check_g(r, failure='unknown')
        closed = self.close_g()
        benefit = closed['evaluations'][0]['benefit']
        self.assertEqual(benefit['observed'], 0)
        self.assertAlmostEqual(benefit['abs_error'], .7)
        unknown = self.predict_g(episode='no-audit')
        self.run_g(unknown)
        self.close_g('no-audit')
        summary = self.h.calibration_summary()
        group = summary['conditional_g'][0]
        self.assertEqual((group['failed'], group['unknown'], group['n_scored']), (1, 1, 1))
        self.assertAlmostEqual(group['brier'], .49)
        self.assertTrue(group['unknown_reasons'])
        matching = [g for g in summary['groups'].values() if g.get('mean_benefit_abs_error') is not None]
        self.assertEqual(len(matching), 1)
        self.assertAlmostEqual(matching[0]['mean_benefit_abs_error'], .7)
        self.predict_g(episode='next')
        context = self.provider.requests[-1]['prediction_context']
        pair = next(row for row in context['prediction_execution_pairs']['pairs']
                    if row['prediction_id'] == p.prediction_id)
        self.assertEqual(pair['benefit']['observed'], 0)
        self.assertEqual(pair['benefit']['attribution']['failure_source'], 'unknown')
        case = next(row for row in context['g_candidate_evidence']['cases'] if row['execution_id'] == r.execution_id)
        self.assertEqual(case['g']['observed'], 0)
        unknown_case = next(row for row in context['g_candidate_evidence']['cases']
                            if row['execution_id'] != r.execution_id)
        self.assertIsNone(unknown_case['g']['observed'])
        self.assertEqual(unknown_case['g']['rule'], FAITHFUL_G_OBSERVATION_VERSION)
        global_stats = context['g_candidate_evidence']['global_history']
        self.assertEqual((global_stats['n_scored'], global_stats['unknown']['n']), (1, 1))
        self.assertAlmostEqual(global_stats['mae'], .7)
        self.assertAlmostEqual(global_stats['brier'], .49)
        count = len(self.h.strategy_prediction_evaluations())
        self.h.calibration_summary(rebuild=True)
        self.close_g()
        self.assertEqual(len(self.h.strategy_prediction_evaluations()), count)
        self.assertEqual(self.h.calibration_summary()['conditional_g'], summary['conditional_g'])

    def test_historical_rules_remain_readable_and_numerically_separate(self):
        for episode, legacy, coverage, label in (
                ('old-pass', True, 'original_task', 1),
                ('old-partial', True, 'partial', None),
                ('new-partial', False, 'partial', 0)):
            p = self.predict_g(episode=episode)
            r = self.run_g(p)
            if legacy:
                # A synthetic retained /1 forecast, never a migration of an
                # experiment bank. Current provider requests always use /2.
                p = self.h.strategy_predictions.get(p.prediction_id)
                p.benefit.observation_rule_version = LEGACY_FAITHFUL_G_OBSERVATION_VERSION
                p.trace.model_info['benefit_convention']['observation_rule_version'] = LEGACY_FAITHFUL_G_OBSERVATION_VERSION
                self.h.strategy_predictions._save(p)
                self.assertFalse(validate_strategy_outcome(p))
            checked = self.check_g(r, coverage=coverage)
            expected_version = LEGACY_FAITHFUL_G_OBSERVATION_VERSION if legacy else FAITHFUL_G_OBSERVATION_VERSION
            self.assertEqual(get_value(self.h.store, checked['g_assessment_ref'])['observation_rule_version'], expected_version)
            closed = self.close_g(episode)
            stored = get_evaluation(self.h.store, closed['evaluations'][0]['evaluation_id'])
            self.assertEqual(stored.benefit.get('observed'), label)
            self.assertEqual(_live_evaluation(self.h, stored)[0].benefit.get('observed'), label)
        groups = self.h.calibration_summary(rebuild=True)['conditional_g']
        by_version = {g['observation_rule_version']: g for g in groups}
        self.assertEqual(set(by_version), {LEGACY_FAITHFUL_G_OBSERVATION_VERSION, FAITHFUL_G_OBSERVATION_VERSION})
        self.assertEqual(by_version[LEGACY_FAITHFUL_G_OBSERVATION_VERSION]['n_execution_units'], 2)
        self.assertEqual(by_version[FAITHFUL_G_OBSERVATION_VERSION]['failed'], 1)
        # Global G metrics only score the current convention. Old pairs stay
        # readable with their own convention, never relabelled as /2.
        stats = global_g_statistics(build_paired_feedback(self.h)['pairs'])
        self.assertEqual((stats['n_pairs'], stats['n_scored']), (1, 1))
        self.assertAlmostEqual(stats['brier'], .49)
        self.predict_g(episode='next')
        request = self.provider.requests[-1]
        self.assertEqual(request['benefit_convention']['observation_rule_version'], FAITHFUL_G_OBSERVATION_VERSION)
        self.assertTrue(all(row['observation_rule_version'] == FAITHFUL_G_OBSERVATION_VERSION
            for row in request['prediction_context']['strategy_outcome_calibration']['conditional_g']))

    def test_late_g_rederivation_preserves_other_signals_and_frozen_forecast(self):
        p = self.predict_g()
        r = self.run_g(p)
        self.check_g(r)
        closed = self.close_g()
        stored = get_evaluation(self.h.store, closed['evaluations'][0]['evaluation_id'])
        forecast = self.h.get_strategy_outcome_prediction(p.prediction_id).to_dict()
        self.check_g(r, coverage='partial', evidence_status='agent_insufficient')
        derived, correction = _live_evaluation(self.h, stored)
        self.assertEqual((stored.benefit['observed'], derived.benefit['observed']), (1, 0))
        self.assertTrue(correction)
        self.assertEqual(derived.cost, stored.cost)
        self.assertEqual(derived.risk, stored.risk)
        self.assertEqual(self.h.get_strategy_outcome_prediction(p.prediction_id).to_dict(), forecast)
        self.assertEqual(len(self.provider.requests), 1)
        self.assertEqual(len(self.h.strategy_prediction_evaluations()), 1)
