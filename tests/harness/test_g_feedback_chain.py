"""Synthetic fixtures for the G submission, persistence and feedback chain.

No build6 records, remote services or fitted labels are used. Executions run
tiny deterministic scripts; provider responses are scripted by GCase.
"""
import copy
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from g_fixtures import COUPLED, SPLIT, GCase
from or_harness.api import ORHarness
from or_harness.cli import main
from or_harness.strategy.embedding_index import LocalHashEmbeddingBackend
from or_harness.world_model.conditions import get_value
from or_harness.world_model.contracts import FAITHFUL_G_CONVENTION
from or_harness.world_model.episode_closeout import _live_evaluation, get_evaluation
from or_harness.world_model.history_retrieval import _evaluation_fact, global_g_statistics


class TestGFeedbackChain(GCase):
    def declaration(self, record, **changes):
        value = {
            'method_fidelity': 'faithful', 'failure_source': 'unknown',
            'acceptance_scope': 'original_task', 'acceptance_id': 'factory-coupling/v1',
            'source': 'synthetic auditor of the captured code and result',
            'assessment': 'Compare the code and BE/CFC assignments with the task coupling.',
            'evidence_refs': ['artifacts.submitted_script', 'artifacts.workspace_result'],
        }
        value.update(changes)
        return value

    def submit(self, record, attribution=None, **kwargs):
        check = {'semantic_probe': {
            'path': 'execution_features.solution_variables.CFC', 'equals': 1},
            'g_attribution': attribution if attribution is not None else self.declaration(record)}
        return self.h.check_task_result(record.execution_id, check, **kwargs)

    def next_view(self, episode='next'):
        self.predict_g(episode=episode)
        return self.provider.requests[-1]['prediction_context']

    def assert_feedback(self, view, prediction_id, execution_id, label):
        pairs = [row for row in view['prediction_execution_pairs']['pairs']
                 if row['prediction_id'] == prediction_id]
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]['benefit'].get('observed'), label)
        cases = [row for row in view['g_candidate_evidence']['cases']
                 if row['execution_id'] == execution_id]
        self.assertEqual(len(cases), 1, view['g_candidate_evidence'])
        self.assertEqual(cases[0]['g']['observed'], label)
        return pairs[0], cases[0]

    def cli(self, *args):
        output = io.StringIO()
        with patch.dict('os.environ', {'OR_EMBEDDING_BACKEND': 'local-hashing'}), redirect_stdout(output):
            code = main(['--home', self.home, *args])
        return code, json.loads(output.getvalue())

    def test_positive_and_candidate_failure_reach_both_feedback_channels(self):
        for episode, method, split, source, label in (
                ('pass', COUPLED, False, 'unknown', 1),
                ('fail', SPLIT, True, 'candidate_method', 0)):
            with self.subTest(label=label):
                p = self.predict_g(method, episode)
                r = self.run_g(p, split=split)
                checked = self.submit(r, self.declaration(r, failure_source=source))
                saved = get_value(self.h.store, checked['g_assessment_ref'])
                self.assertEqual(saved['execution_id'], r.execution_id)
                self.assertEqual(saved['prediction_id'], p.prediction_id)
                self.assertEqual(saved['execution_action_id'], r.action_id)
                self.assertEqual(saved['check_action_id'], checked['action_id'])
                action = self.h.actions.get(checked['action_id'])
                self.assertEqual(action.outcome['g_assessment_ref'], checked['g_assessment_ref'])
                self.assertNotIn('g_attribution', checked['report']['acceptance_rule']['spec'])
                closed = self.close_g(episode)
                self.assertEqual(closed['evaluations'][0]['benefit']['observed'], label)
                pair, case = self.assert_feedback(self.next_view('next-' + episode),
                    p.prediction_id, r.execution_id, label)
                self.assertEqual(pair['benefit']['attribution_ref'], checked['g_assessment_ref'])
                self.assertEqual(case['recorded_cause']['failure_source'], source)
        groups = self.h.calibration_summary()['conditional_g']
        self.assertEqual(sum(row['n_execution_units'] for row in groups), 2)
        self.assertEqual(sum(row['passed'] for row in groups), 1)
        self.assertEqual(sum(row['failed'] for row in groups), 1)

    def test_technical_crash_has_legal_unknown_audit_and_preserves_r(self):
        p = self.predict_g()
        r = self.run_g(p, crash=True)
        checked = self.submit(r, self.declaration(r, method_fidelity='unknown',
            acceptance_scope='unknown', failure_source='implementation',
            evidence_refs=['artifacts.submitted_script']))
        self.assertTrue(checked['g_assessment_ref'])
        benefit = self.close_g()['evaluations'][0]['benefit']
        self.assertNotIn('observed', benefit)
        self.assertIn('crash/timeout', benefit['reason'])
        pair, case = self.assert_feedback(self.next_view(), p.prediction_id, r.execution_id, None)
        self.assertEqual(pair['implementation_observation']['label'], 1)
        self.assertEqual(case['recorded_cause']['failure_source'], 'implementation')
        group = self.h.calibration_summary()['conditional_g'][0]
        self.assertEqual((group['unknown'], group['n_scored']), (1, 0))
        self.assertIsNone(group['brier'])

    def test_passing_checker_does_not_replace_fidelity_or_acceptance_coverage(self):
        for episode, changes in (
                ('fidelity', {'method_fidelity': 'unknown'}),
                ('coverage', {'acceptance_scope': 'partial'}),
                ('no-evidence', {'method_fidelity': 'unknown',
                                 'acceptance_scope': 'unknown', 'evidence_refs': []})):
            with self.subTest(episode=episode):
                # Distinct synthetic tasks respect the existing two-attempt
                # diversity cap instead of changing retrieval to fit tests.
                self.task['task_id'] = 'factory-' + episode
                p = self.predict_g(episode=episode)
                r = self.run_g(p)
                self.assertEqual(self.submit(r, self.declaration(r, **changes))['state'], 'passed')
                b = self.close_g(episode)['evaluations'][0]['benefit']
                self.assertEqual(b['label_eligibility'], 'unknown')
                self.assertNotIn('observed', b)
                self.assertTrue(b['reason'])
                self.assert_feedback(self.next_view('next-' + episode), p.prediction_id, r.execution_id, None)
        groups = self.h.calibration_summary()['conditional_g']
        self.assertEqual(sum(g['n_scored'] for g in groups), 0)
        self.assertEqual(sum(g['unknown'] for g in groups), 3)

    def test_reference_conflict_stays_unknown_until_supported_resolution(self):
        p = self.predict_g()
        r = self.run_g(p)
        conflict = self.declaration(r, failure_source='reference_conflict',
            assessment='Synthetic reference says 999; captured result is 100; the discrepancy is unresolved.')
        checked = self.h.check_task_result(r.execution_id, {
            'reference_objective': 999, 'reference_source': 'bench_declared',
            'g_attribution': conflict})
        closed = self.close_g()
        evaluation_id = closed['evaluations'][0]['evaluation_id']
        self.assertNotIn('observed', closed['evaluations'][0]['benefit'])
        self.assertIn('reference_conflict', closed['evaluations'][0]['benefit']['reason'])
        pair, case = self.assert_feedback(self.next_view(), p.prediction_id, r.execution_id, None)
        self.assertEqual(case['recorded_cause']['failure_source'], 'reference_conflict')
        self.assertEqual(pair['benefit']['attribution_ref'], checked['g_assessment_ref'])
        self.assertEqual(self.h.calibration_summary()['conditional_g'][0]['n_scored'], 0)
        # A separate synthetic trusted reference now resolves the dispute;
        # simply claiming that the benchmark is wrong would not supply G=1.
        self.h.check_task_result(r.execution_id, {
            'reference_objective': 100, 'reference_source': 'independent',
            'reference_version': 'synthetic-independent-check/1',
            'g_attribution': self.declaration(r)})
        self.assert_feedback(self.next_view('resolved'), p.prediction_id, r.execution_id, 1)
        self.assertNotIn('observed', self.h.get_strategy_evaluation(evaluation_id)['benefit'])
        group = self.h.calibration_summary()['conditional_g'][0]
        self.assertEqual((group['passed'], group['n_execution_units']), (1, 1))

    def test_invalid_attribution_is_rejected_without_writes(self):
        p = self.predict_g()
        r = self.run_g(p)
        invalid = [None, [], 'faithful', self.declaration(r, method_fidelity='yes'),
                   self.declaration(r, failure_source='solver_optimal'),
                   self.declaration(r, source=''),
                   self.declaration(r, evidence_refs=['artifacts.missing']),
                   self.declaration(r, resolved_refs={'artifacts.submitted_script': {'sha256': 'wrong'}})]
        before = list(self.h.store.conn.iterdump())
        for value in invalid:
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, 'invalid g_attribution'):
                    self.h.check_task_result(r.execution_id, {'g_attribution': value})
                self.assertEqual(list(self.h.store.conn.iterdump()), before)

    def test_wrong_ownership_and_method_claims_are_not_overwritten(self):
        p = self.predict_g()
        r = self.run_g(p)
        other = self.predict_g(SPLIT, 'other')
        for changes in (
                {'execution_id': 'other-execution'}, {'task_id': 'other-task'},
                {'episode_id': 'other'}, {'prediction_id': other.prediction_id},
                {'prediction_id': ['not-an-id']},
                {'execution_action_id': 'other-action'}, {'strategy_id': 'other-strategy'},
                {'method_planned': SPLIT}, {'method_actual_fingerprint': 'other-method'}):
            with self.subTest(changes=changes):
                with self.assertRaisesRegex(ValueError, 'invalid g_attribution'):
                    self.submit(r, self.declaration(r, **changes))
        with self.assertRaisesRegex(ValueError, 'episode_id does not match'):
            self.submit(r, episode_id='other')
        self.assertIsNone(self.h.bank.get(r.execution_id).execution_features.get('task_check'))

    def test_misplaced_attribution_cannot_succeed_through_check_or_action(self):
        p = self.predict_g()
        r = self.run_g(p)
        declaration = self.declaration(r)
        for check in ({'acceptance_rule': {'spec': {'g_attribution': declaration}}},
                      {'spec': {'g_attribution': declaration}}):
            with self.assertRaisesRegex(ValueError, 'top-level key'):
                self.h.check_task_result(r.execution_id, check)
        for key in ('params', 'outcome'):
            with self.assertRaisesRegex(ValueError, 'check-task --check'):
                self.h.report_action('verify', self.task,
                    **{key: {'g_attribution': declaration}})
        code, output = self.cli('check-task', r.execution_id, '--check',
                                json.dumps({'g_attribution': None}))
        self.assertEqual(code, 2)
        self.assertIn('invalid g_attribution', output['result']['error'])
        code, output = self.cli('action', '--report', 'verify', '--task', json.dumps(self.task),
                                '--outcome', json.dumps({'g_attribution': declaration}))
        self.assertEqual(code, 2)
        self.assertIn('check-task --check', output['result']['error'])
        self.assertIsNone(self.h.bank.get(r.execution_id).execution_features.get('task_check'))

    def test_cli_action_json_errors_are_actionable_and_json_output_is_parseable(self):
        for args, expected in (
                (['--task', self.task['task_id']], 'task JSON object or JSON file'),
                (['--task', '[]'], 'with task_id'),
                (['--task', json.dumps(self.task), '--outcome', '{broken'], '--outcome'),
                (['--task', json.dumps(self.task), '--params', '[]'], '--params')):
            with self.subTest(args=args):
                code, output = self.cli('action', '--report', 'verify', *args)
                self.assertEqual(code, 2)
                self.assertIn(expected, output['result']['error'])
                self.assertNotIn('JSONDecodeError', output['result']['error'])
        code, output = self.cli('action', '--report', 'verify', '--task', json.dumps(self.task),
                                '--outcome', '{"state":"agent_reported"}')
        self.assertEqual(code, 0, output)
        self.assertEqual(output['result']['source'], 'agent_reported')

    def test_cli_submission_survives_reopen_and_repeated_closeout(self):
        p = self.predict_g()
        r = self.run_g(p, record=False)
        check = {'semantic_probe': {'path': 'execution_features.solution_variables.CFC', 'equals': 1},
                 'g_attribution': self.declaration(r)}
        path = Path(self.home) / 'synthetic-check.json'
        path.write_text(json.dumps(check))
        code, output = self.cli('check-task', r.execution_id, '--check', str(path))
        self.assertEqual(code, 0, output)
        checked = output['result']
        code, output = self.cli('record', '--from-staged', r.execution_id)
        self.assertEqual(code, 0, output)
        closed = self.close_g()
        evaluation_id = closed['evaluations'][0]['evaluation_id']
        self.h.close()
        self.h = ORHarness(home=self.home, world_model=self.provider,
                           embedding=LocalHashEmbeddingBackend())
        self.addCleanup(self.h.close)
        self.assertEqual(get_value(self.h.store, checked['g_assessment_ref'])['execution_id'], r.execution_id)
        self.assertEqual(self.h.get_strategy_evaluation(evaluation_id)['benefit']['observed'], 1)
        for _ in range(2):
            self.close_g()
            self.h.calibration_summary(rebuild=True)
        self.assertEqual(self.h.calibration_summary()['conditional_g'][0]['n_execution_units'], 1)
        self.submit(self.h.bank.get(r.execution_id), self.declaration(r))
        self.assertEqual(self.h.calibration_summary()['conditional_g'][0]['n_execution_units'], 1)
        self.assert_feedback(self.next_view(), p.prediction_id, r.execution_id, 1)

    def test_live_rederivation_preserves_original_and_reaches_next_local_request_once(self):
        p = self.predict_g()
        r = self.run_g(p)
        self.h.check_task_result(r.execution_id, {'semantic_probe': {
            'path': 'execution_features.solution_variables.CFC', 'equals': 1}})
        closed = self.close_g()
        evaluation_id = closed['evaluations'][0]['evaluation_id']
        original = copy.deepcopy(self.h.get_strategy_evaluation(evaluation_id))
        self.assertNotIn('observed', original['benefit'])
        checked = self.submit(r)
        current, correction = _live_evaluation(self.h, get_evaluation(self.h.store, evaluation_id))
        self.assertEqual(current.benefit['observed'], 1)
        self.assertEqual(correction['evaluation_id'], evaluation_id)
        self.assertEqual(current.cost, original['cost'])
        self.assertEqual(current.risk, original['risk'])
        self.assertEqual(self.h.get_strategy_evaluation(evaluation_id), original)
        for _ in range(2):
            summary = self.h.calibration_summary(rebuild=True)
            self.assertEqual(summary['conditional_g'][0]['n_execution_units'], 1)
            self.assertEqual(summary['conditional_g'][0]['passed'], 1)
        pair, case = self.assert_feedback(self.next_view(), p.prediction_id, r.execution_id, 1)
        self.assertEqual(pair['evaluation_id'], evaluation_id)
        self.assertEqual(case['g']['attribution_ref'], checked['g_assessment_ref'])

    def test_legacy_error_currency_cannot_mask_a_separate_faithful_g_fact(self):
        pair = {'benefit': {'metric': 'normalized_objective_gap', 'observed': 1,
                            'predicted': 0.8, 'signed_error': -0.2},
                'g_observed_fact': {**FAITHFUL_G_CONVENTION, 'observed': 0,
                                    'label_eligibility': 'attributable'}}
        original = copy.deepcopy(pair)
        local = _evaluation_fact(pair)['g']
        self.assertEqual(local['metric'], FAITHFUL_G_CONVENTION['metric'])
        self.assertEqual(local['observed'], 0)
        self.assertNotIn('predicted', local)
        self.assertNotIn('signed_error', local)
        global_ = global_g_statistics([pair])
        self.assertEqual(global_['n_observed'], 1)
        self.assertEqual(global_['n_scored'], 0)
        self.assertEqual(pair, original)

    def test_unknown_eligibility_cannot_fall_back_to_a_numeric_legacy_label(self):
        pair = {'benefit': {'metric': 'normalized_objective_gap', 'observed': 1,
                            'predicted': 0.8},
                'g_observed_fact': {**FAITHFUL_G_CONVENTION, 'observed': 1,
                                    'label_eligibility': 'unknown',
                                    'reason': 'unresolved reference_conflict'}}
        self.assertNotIn('observed', _evaluation_fact(pair)['g'])
        global_ = global_g_statistics([pair])
        self.assertEqual(global_['n_observed'], 0)
        self.assertEqual(global_['n_scored'], 0)
        self.assertEqual(global_['unknown']['n'], 1)

    def test_final_check_precedes_record_and_closeout_and_original_g_needs_no_recovery(self):
        for episode, fidelity, label in (('normal-known', 'faithful', 1),
                                         ('normal-unknown', 'unknown', None)):
            with self.subTest(episode=episode):
                p = self.predict_g(episode=episode)
                original_at_live_read = []

                def inspect_original(harness, stored):
                    if stored.prediction_id == p.prediction_id:
                        original_at_live_read.append(stored.benefit.get('observed'))
                    return _live_evaluation(harness, stored)

                with patch.object(self.h, 'check_task_result', wraps=self.h.check_task_result) as check, \
                        patch.object(self.h, 'close_episode', wraps=self.h.close_episode) as close:
                    record = self.run_g(p, record=False)
                    self.assertEqual((check.call_count, close.call_count), (0, 0))
                    checked = self.submit(record, self.declaration(record, method_fidelity=fidelity))
                    self.h.record(record)
                    self.assertEqual((check.call_count, close.call_count), (1, 0))
                    with patch('or_harness.world_model.episode_closeout._live_evaluation',
                               side_effect=inspect_original):
                        closed = self.close_g(episode)
                    self.assertEqual((check.call_count, close.call_count), (1, 1))
                self.assertTrue(closed['closeout'])
                self.assertTrue(original_at_live_read)
                self.assertEqual(set(original_at_live_read), {label})
                saved = get_evaluation(self.h.store, closed['evaluations'][0]['evaluation_id'])
                self.assertEqual(saved.benefit.get('observed'), label)
                self.assertEqual(saved.benefit['attribution_ref'], checked['g_assessment_ref'])
                self.close_g(episode)
                self.assert_feedback(self.next_view('next-' + episode), p.prediction_id,
                                     record.execution_id, label)
        groups = self.h.calibration_summary()['conditional_g']
        self.assertEqual(sum(g['n_execution_units'] for g in groups), 2)
        self.assertEqual(sum(g['passed'] for g in groups), 1)
        self.assertEqual(sum(g['unknown'] for g in groups), 1)

    def test_plain_recheck_creates_a_new_revision_without_inheriting_old_attribution(self):
        p = self.predict_g()
        record = self.run_g(p, record=False)
        first = self.submit(record)
        last = self.h.check_task_result(record.execution_id, {'semantic_probe': {
            'path': 'execution_features.solution_variables.CFC', 'equals': 1}})
        self.assertNotEqual(first['action_id'], last['action_id'])
        self.assertNotIn('g_assessment_ref', last)
        self.assertEqual(get_value(self.h.store, first['g_assessment_ref'])['check_action_id'],
                         first['action_id'])
        self.h.record(record)
        closed = self.close_g()
        benefit = closed['evaluations'][0]['benefit']
        self.assertNotIn('observed', benefit)
        self.assertEqual(benefit['task_check']['action_id'], last['action_id'])
        self.assertIn('missing or invalid sourced G attribution', benefit['reason'])
