"""Deterministic publication representation; original facts and selection stay fixed."""
import copy
import json
import time
import unittest
from unittest.mock import patch

from c_fixtures import CCase
from tests.harness import test_stage_e_history_retrieval as history_tests
from tests.harness.test_history_feedback_budget import _pair
from or_harness.strategy.evidence_views import provider_feedback_pair, provider_paired_feedback
from or_harness.world_model.contracts import (
    FAITHFUL_G_METRIC, FAITHFUL_G_OBSERVATION_VERSION, R_OBSERVATION_VERSION,
    RETRIES_FUTURE_VERSION,
)
from or_harness.world_model.episode_closeout import (
    _filter_pairs_by_model, _selection_meta, build_paired_feedback,
    ensure_selection_meta, paired_feedback_selection_from_block, same_task_comparisons,
)


def material_pair(index=0):
    row = _pair(index, big=True)
    row['model_identity'] = '(unknown)'
    row['benefit'].update(metric=FAITHFUL_G_METRIC,
        observation_rule_version=FAITHFUL_G_OBSERVATION_VERSION, acceptance_id='original-task',
        attribution={'method_fidelity': 'faithful', 'failure_source': 'candidate_method',
                     'assessment': 'Keep the complete recorded cause. ' * 25 + 'FINAL CAUSE.',
                     'source': 'original recorded auditor'})
    row['method_planned']['steps'] = [f'Original modelling step {i}: ' + 'preserve details ' * 20
                                      for i in range(16)]
    row['method_actual']['steps'] = row['method_planned']['steps'][:-1] + ['A different FINAL step.']
    row['method_deviation'] = {'state': 'recorded_difference', 'assessment': 'Recorded explanation.',
        'planned_steps': copy.deepcopy(row['method_planned']['steps']),
        'actual_steps': copy.deepcopy(row['method_actual']['steps'])}
    row['implementation_observation'] = {'event': 'implementation_failure', 'label': 1,
        'observation_rule_version': R_OBSERVATION_VERSION, 'reason': 'Retain the API error cause. ' * 20,
        'semantic_attribution': {'failure_source': 'implementation', 'assessment': 'Wrong API argument.'}}
    row['cost_observed_fact']['dimension_observation_rule_versions'] = {'retries': RETRIES_FUTURE_VERSION}
    row['cost_observed_fact']['conditions'] = {'scope': 'remaining_to_task_end',
        'window_coverage': 'complete', 'family': 'routing', 'structure': {'problem_class': 'milp'},
        'solver_conditions': {'solver': 'highs', 'config': {'time_limit': 20},
                              'constraints': {'timeout': 120}},
        'actual_solving_conditions': [{'solver': 'highs', 'config': {'time_limit': 20}}]}
    row['execution_conditions'] = {'execution_id': f'ex_{index}', 'comparable': True,
        'comparison_key': copy.deepcopy(row['cost_observed_fact']['conditions']['solver_conditions'])}
    row['cost_observed_fact']['work_facts'] = {'calls': ['AUDIT ' * 1000]}
    row['unrecognised_original_note'] = 'Keep unknown fields rather than guessing their importance.'
    return row


class TestPublicationMaterial(unittest.TestCase):
    setUp = history_tests.TestPublishedObservationJoin.setUp
    publish = history_tests.TestPublishedObservationJoin.publish
    read = history_tests.TestPublishedObservationJoin.read
    history = history_tests.TestPublishedObservationJoin.history

    def test_full_methods_causes_conditions_and_unknown_fields_stay_inline(self):
        row = material_pair()
        original = copy.deepcopy(row)
        compact = provider_feedback_pair(row, for_publication=True)
        sent = provider_feedback_pair(compact)
        for key in ('method_planned', 'method_actual', 'benefit', 'implementation_observation'):
            expected = original[key]
            if key == 'benefit':
                expected = copy.deepcopy(expected)
                expected['acceptance_rule'].pop('spec')
                expected['acceptance_rule']['spec_ref'] = '#benefit.acceptance_rule.spec'
            self.assertEqual(sent[key], expected)
        self.assertEqual(sent['unrecognised_original_note'], original['unrecognised_original_note'])
        self.assertEqual(sent['method_deviation']['assessment'], 'Recorded explanation.')
        self.assertEqual(sent['method_deviation']['actual_steps_ref'], 'method_actual.steps')
        self.assertEqual(sent['cost_observed_fact']['conditions']['scope'], 'remaining_to_task_end')
        self.assertEqual(sent['cost_observed_fact']['conditions']['solver_conditions']['config_ref'],
                         'execution_conditions.comparison_key.config')
        self.assertEqual(row, original)
        self.assertEqual(json.dumps(compact).count('orx inspect --bank evaluations --evaluation'), 1)
        self.assertEqual(compact['cost_observed_fact']['window_ref'], '#cost.observed_fact.window')
        # The reference points at the original evaluation, not an EE record.
        original_evaluation = {'cost': {'observed_fact': row['cost_observed_fact']}}
        restored = original_evaluation
        for part in compact['cost_observed_fact']['window_ref'][1:].split('.'):
            restored = restored[part]
        self.assertEqual(restored, row['cost_observed_fact']['window'])

    def test_no_repeated_compaction_loss_or_new_method_judgement(self):
        row = material_pair()
        compact = provider_feedback_pair(row, for_publication=True)
        once = provider_feedback_pair(compact)
        self.assertEqual(provider_feedback_pair(once), once)
        self.assertEqual(provider_feedback_pair(compact, for_publication=True), compact)
        self.assertEqual(once['method_deviation']['state'], 'recorded_difference')
        self.assertNotIn('minor_textual_difference', json.dumps(once))
        # The existing numeric observations survive a second pass.
        self.assertEqual(once['cost_measured'], row['cost_measured'])
        self.assertEqual(once['cost_observed_fact']['values_not_scored']['latency_s'], .02)

    def test_shared_risk_versions_retain_current_rule_and_unknown_labels(self):
        row = material_pair()
        row.pop('implementation_observation')
        for event in row['risk_actual']:
            if event['event'] == 'implementation_failure':
                event['observation_rule_version'] = R_OBSERVATION_VERSION
        compact = provider_feedback_pair(row, for_publication=True)
        sent = provider_feedback_pair(compact)
        self.assertEqual(sent['risk_predicted'], compact['risk_predicted'])
        self.assertEqual(provider_feedback_pair(sent), sent)
        self.assertEqual(sent['risk_observation_rule_versions'], compact['risk_observation_rule_versions'])

    def test_only_declared_audit_fields_are_removed_from_cost_conditions(self):
        row = material_pair()
        row['cost_observed_fact']['conditions']['solver_conditions']['constraints'] = {
            'executor': {'timeout_seconds': 120, 'limits': {'file_bytes': 123},
                         'inputs': 'Harness usage contract', 'unknown_limit': 'KEEP'}}
        bound = row['cost_observed_fact']['lower_bounds']['remaining_latency_s']
        bound['reason'] = 'Only part of the window was measured.'
        compact = provider_feedback_pair(row, for_publication=True)
        executor = compact['cost_observed_fact']['conditions']['solver_conditions']['constraints']['executor']
        self.assertEqual(executor['timeout_seconds'], 120)
        self.assertEqual(executor['limits'], {'file_bytes': 123})
        self.assertEqual(executor['unknown_limit'], 'KEEP')
        self.assertNotIn('inputs', executor)
        kept = compact['cost_observed_fact']['lower_bounds']['remaining_latency_s']
        self.assertEqual(kept['value'], bound['value'])
        self.assertEqual(kept['reason'], bound['reason'])
        self.assertIn('source_refs_ref', kept)

    def test_unknown_bounds_conflicting_values_and_missing_actual_are_not_inferred(self):
        row = material_pair()
        row.pop('method_actual')
        row.pop('method_deviation')
        row['benefit']['observed'] = None
        row['benefit']['label_eligibility'] = 'unknown'
        row['cost_observed_fact']['lower_bounds'] = {'tool_calls': {'value': 4, 'scope': 'partial'}}
        row['cost_observed_fact']['values']['tool_calls'] = 777
        row['cost_measured']['tool_calls'] = {'predicted': 8, 'actual': 4}
        row['risk_actual'][0]['label'] = None
        value = provider_feedback_pair(row, for_publication=True)
        self.assertNotIn('method_actual', value)
        self.assertIsNone(value['benefit']['observed'])
        self.assertIsNone(value['risk_actual'][0]['label'])
        self.assertEqual(value['cost_observed_fact']['lower_bounds'], row['cost_observed_fact']['lower_bounds'])
        self.assertEqual(value['cost_observed_fact']['values']['tool_calls'], 777)
        self.assertNotEqual(value['cost_measured']['tool_calls']['actual'], 777)

    def test_model_filter_preserves_original_directory_and_relevance_order(self):
        self.rows = [material_pair(i) for i in range(4)]
        for i, row in enumerate(self.rows):
            row['model_identity'] = 'a' if i % 2 else 'b'
        raw_meta = {m['pair_ref']: m for m in _selection_meta(list(enumerate(self.rows)))}
        published = self.publish(budget=10**7)
        self.assertIsNone(published['model_identity'])
        for model in ('a', 'b'):
            filtered = _filter_pairs_by_model(published, model)
            self.assertEqual([m['pair_ref'] for m in filtered['selection_meta']],
                             [r['evaluation_id'] for r in filtered['pairs']])
            for meta in filtered['selection_meta']:
                self.assertEqual(meta['plan_method_words'], raw_meta[meta['pair_ref']]['plan_method_words'])
            originals = [r for r in self.rows if r['model_identity'] == model]
            # Compare the SAME ordered set, independent of increased capacity.
            by_id = {r['evaluation_id']: r for r in originals}
            original_block = {'pairs': [by_id[r['evaluation_id']] for r in filtered['pairs']],
                              'selection_meta': filtered['selection_meta']}
            query = {'candidates': [{'method': {'name': 'Original', 'steps': ['modelling step 15']}}]}
            a = paired_feedback_selection_from_block(original_block, **query)
            b = paired_feedback_selection_from_block(ensure_selection_meta(filtered), **query)
            self.assertEqual([(r['evaluation_id'], r['relevance']) for r in a['pairs']],
                             [(r['evaluation_id'], r['relevance']) for r in b['pairs']])
        # Legacy single-model blocks without row identities remain readable.
        legacy = {'model_identity': 'old', 'pairs': [copy.deepcopy(self.rows[0])]}
        legacy['pairs'][0].pop('model_identity')
        self.assertEqual(len(_filter_pairs_by_model(legacy, 'old')['pairs']), 1)

    def test_comparisons_use_original_facts_and_cannot_reference_unpublished_pairs(self):
        self.rows = [material_pair(i) for i in range(2)]
        for row in self.rows:
            row.update(task_id='same-task', problem_identity_digest='same-input')
            row['benefit']['acceptance_rule']['fingerprint'] = 'same-check'
            row['method_actual']['steps'][-1] = row['evaluation_id']
        expected = same_task_comparisons(self.rows)
        self.assertEqual(len(expected), 1)
        block = self.publish(budget=10**7)
        self.assertEqual(block['same_task_comparisons'], expected)
        self.assertEqual(_filter_pairs_by_model(block, '(unknown)')['same_task_comparisons'], expected)
        self.rows[1]['execution_conditions']['comparison_key']['config']['time_limit'] = 99
        self.assertEqual(self.publish(budget=10**7)['same_task_comparisons'], [])
        self.rows[1]['execution_conditions']['comparison_key']['config']['time_limit'] = 20
        first_size = len(json.dumps(provider_feedback_pair(self.rows[0], for_publication=True), ensure_ascii=False))
        limited = self.publish(budget=first_size)
        self.assertEqual(limited['n_pairs_included'], 1)
        self.assertEqual(limited['same_task_comparisons'], [])

    def test_budget_reduction_omits_cases_instead_of_cutting_core_content(self):
        pair = provider_feedback_pair(material_pair(), for_publication=True)
        full = provider_paired_feedback({'pairs': [pair], 'n_pairs_included': 1}, budget_chars=100000)
        small = provider_paired_feedback({'pairs': [pair], 'n_pairs_included': 1}, budget_chars=3000)
        self.assertEqual(full['pairs'][0]['method_planned']['steps'], pair['method_planned']['steps'])
        self.assertEqual(small['pairs'], [])
        self.assertTrue(small['dropped_pair_refs'])
        self.assertLessEqual(len(json.dumps(small, ensure_ascii=False)), 3000)

    def test_final_request_uses_existing_block_and_80k_limits_without_core_clipping(self):
        from tests.harness.test_history_feedback_budget import TestFullRequestBudget
        from or_harness.strategy.evidence_views import chars
        from or_harness.world_model.request_budget import apply_provider_block_budgets, enforce_full_request_budget
        from or_harness.world_model.strategy_prediction import STRATEGY_OUTCOME_SYSTEM_PROMPT
        pairs = [provider_feedback_pair(material_pair(i), for_publication=True) for i in range(4)]
        feedback = provider_paired_feedback({'pairs': pairs, 'n_pairs_included': 4})
        self.assertLessEqual(chars(feedback), 30000)
        request = TestFullRequestBudget()._request(0)
        task = copy.deepcopy(request['prediction_context']['joint_problem'])
        request['prediction_context']['prediction_execution_pairs'] = feedback
        with patch.dict('os.environ', {'OR_HARNESS_INPUT_BUDGET_TOKENS': '80000',
                                       'OR_HARNESS_HISTORY_POOL': '60'}):
            from or_harness.world_model.history_retrieval import history_config
            self.assertEqual(history_config()['pool_size'], 60)
            apply_provider_block_budgets(request['prediction_context'])
            size = enforce_full_request_budget(request, STRATEGY_OUTCOME_SYSTEM_PROMPT)
        self.assertTrue(size['sendable'])
        self.assertLessEqual(size['tokens'], size['bound_tokens'])
        self.assertLessEqual(size['bound_tokens'], 80000)
        self.assertEqual(request['prediction_context']['joint_problem'], task)
        for row in request['prediction_context']['prediction_execution_pairs']['pairs']:
            self.assertEqual(len(row['method_planned']['steps']), 16)
            self.assertTrue(row['benefit']['attribution']['assessment'].endswith('FINAL CAUSE.'))


class TestMaterialFrozenLifecycle(CCase):
    def test_records_observations_frozen_inputs_and_call_count_are_preserved(self):
        from or_harness.world_model.hplus import configure
        from or_harness.world_model.history_retrieval import _pairs_of, global_c_statistics, global_g_statistics
        configure(self.h, 'frozen')
        old = self.h.build_prediction_context(self.task, 'before')
        frozen_input = copy.deepcopy(old.to_dict())
        prediction = self.predict_c(cost={'output_tokens': 500})
        execution = self.run_g(prediction, record=False)
        self.check_g(execution)
        self.finish_c(prediction)
        saved_prediction = self.h.strategy_predictions.get(prediction.prediction_id).to_dict()
        saved_evaluations = self.h.strategy_prediction_evaluations()
        saved_calls = len(self.provider.requests)
        with patch('or_harness.world_model.episode_closeout.time.time', return_value=time.time()):
            # Counterfactual original publication: bypass ONLY the new P1 representation.
            with patch('or_harness.strategy.evidence_views.provider_feedback_pair',
                       side_effect=lambda row, **kw: copy.deepcopy(row)):
                original = build_paired_feedback(self.h, budget_chars=10**7)
            compact = build_paired_feedback(self.h, budget_chars=10**7)
            empty = build_paired_feedback(self.h, budget_chars=0)
        self.assertEqual(original['observation_facts'], compact['observation_facts'])
        self.assertEqual(compact['observation_facts'], empty['observation_facts'])
        for statistic in (global_g_statistics, global_c_statistics):
            self.assertEqual(statistic(_pairs_of(original)), statistic(_pairs_of(compact)))
        pool = [history_tests._pool_row(execution.execution_id)]
        before, after = copy.deepcopy(pool), copy.deepcopy(pool)
        history_tests._attach_evaluation_facts(before, _pairs_of(original))
        history_tests._attach_evaluation_facts(after, _pairs_of(compact))
        self.assertEqual(before, after)
        self.assertEqual(self.h.strategy_predictions.get(prediction.prediction_id).to_dict(), saved_prediction)
        self.assertEqual(self.h.strategy_prediction_evaluations(), saved_evaluations)
        self.assertEqual(old.to_dict(), frozen_input)
        self.assertEqual(self.h.bank.count(), 0)
        self.assertEqual(len(self.provider.requests), saved_calls)
        self.assertTrue(compact['pairs'][0]['method_planned']['steps'])
        self.assertTrue(compact['pairs'][0]['detail_ref'].startswith('orx inspect --bank evaluations'))


if __name__ == '__main__':
    unittest.main()
