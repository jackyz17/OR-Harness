"""Real planner/action/executor regressions; scripted forecasts test plumbing only."""
import copy
import json
import time
from types import SimpleNamespace

from g_fixtures import GCase
from or_harness.world_model.contracts import (
    R_OBSERVATION_VERSION, R_FORMAL_OBSERVATION_VERSION, R_SCORING_VERSION,
    R_FORMAL_SCORING_VERSION, RETRIES_FUTURE_VERSION, RETRIES_FORMAL_VERSION,
)
from or_harness.world_model.implementation_facts import (
    implementation_window, corrective_rounds, collect_facts,
    execution_implementation_evidence, formal_event_id,
)
from or_harness.world_model.episode_closeout import summarize_real_outcome, evaluate_strategy_prediction
from or_harness.world_model.cost_signal import retry_error_statistics
from or_harness.world_model.hplus import jobs
from or_harness.strategy.evidence_views import provider_feedback_pair
from or_harness.world_model.history_retrieval import global_c_statistics, global_r_statistics


class TestBuild64Implementation(GCase):
    def predict(self, *, hplus=0):
        self.provider.payload['cost'] = {'retries': .4}
        self.provider.payload['risk'] = {'events': [
            {'event': name, 'probability': .2} for name in
            ('implementation_failure', 'timeout', 'environment_failure')]}
        self.provider.payload['capability_gain'] = {
            'decision': hplus, 'basis': ['A typed expression adapter may change future API decisions.']}
        if hplus:
            self.provider.payload['capability_gain'].update(
                claim='Under mixed numeric domains, normalize native scalar types before constructing solver expressions.',
                verification_conditions=[{'condition': 'Check repaired expression construction and final result.',
                                          'check_basis': 'captured failing test, diff and result'}])
        prediction = self.predict_g()
        choice = self.h.choose_next(prediction.trace.model_info['decision_action_id'], prediction_id=prediction.prediction_id)
        return prediction, self.h.actions.get(choice['action_id'])

    def report(self, choice, *, event_id='bug-1', classification='implementation',
               round_id='repair-1', excluded=False, confirmed=True, execution_id=None, at=None):
        at = time.time() if at is None else at
        fact = {'selection_action_id': choice.action_id, 'event_id': event_id,
            'occurred_at': at, 'phase': 'pre_execute', 'classification': classification,
            'confirmed': confirmed, 'requires_correction': True, 'excluded': excluded,
            'observer': 'test-runner', 'summary': 'Native scalar rejected by expression constructor.',
            'evidence_refs': ['host-log:test/failing-api', 'artifact:failure.py']}
        if execution_id:
            fact.update(execution_id=execution_id, phase='execution')
        if round_id:
            fact['correction'] = {'round_id': round_id, 'started_at': at,
                'method': 'Normalize native scalar before constructing the expression.',
                'result': 'targeted check passed', 'evidence_refs': ['artifact:repair.diff', 'artifact:fixed.py'],
                'verification_refs': ['host-log:test/validation']}
        self.h.report_action('model', self.task, 'ep1', outcome={'implementation_facts': [fact]})
        return fact

    def covered(self, choice, *, implemented=True):
        stop = time.time()
        self.h.report_action('verify', self.task, 'ep1', ended_at=stop, outcome={
            'implementation_coverage': {'selection_action_id': choice.action_id,
                'from': choice.started_at, 'through': stop, 'complete': True,
                'implemented': implemented, 'candidate_complete': True,
                'observer': 'test-runner', 'evidence_refs': ['host-log:complete-implementation-span']}})

    def finish(self, prediction, *, coverage=True):
        stop = time.time()
        decision = self.h.actions.get(prediction.trace.model_info['decision_action_id'])
        outcome = {'finished': True}
        if coverage:
            outcome['correction_coverage'] = {'from': decision.started_at, 'through': stop,
                'complete': True, 'observer': 'test-runner', 'evidence_refs': ['host-log:complete-corrective-span']}
        self.h.report_action('finish_task', self.task, 'ep1', outcome=outcome, ended_at=stop)

    def summary(self, prediction):
        return summarize_real_outcome(self.h, self.h.strategy_predictions.get(prediction.prediction_id))

    def test_a_initial_success_complete_coverage(self):
        p, c = self.predict()
        self.run_g(p)
        self.covered(c)
        self.finish(p)
        s = self.summary(p)
        self.assertEqual(s.implementation_observation['label'], 0)
        self.assertEqual(s.cost['retries']['comparable']['value'], 0)
        e = evaluate_strategy_prediction(p, s)
        scopes = e.risk['r_events_observation']['events']
        self.assertEqual(scopes['implementation_failure']['scope'], 'selected_candidate_implementation_window')
        self.assertEqual(scopes['timeout']['scope'], 'bound_formal_execution')
        self.assertEqual(e.cost['per_dim']['retries']['actual'], 0)
        self.assertIsNone(e.cost['per_dim']['retries']['log_error'])

    def test_b_repaired_local_defect_survives_formal_success(self):
        p, c = self.predict()
        self.report(c)
        record = self.run_g(p)
        self.covered(c)
        self.finish(p)
        s = self.summary(p)
        self.assertEqual(record.execution_features['technical_outcome']['label'], 0)
        self.assertEqual(s.implementation_observation['label'], 1)
        self.assertEqual(s.cost['retries']['comparable']['value'], 1)

    def test_c_local_rounds_and_formal_resubmission_count_once(self):
        p, c = self.predict()
        self.report(c)
        self.report(c, event_id='bug-2', round_id='repair-2')
        failed = self.run_g(p, crash=True)
        repair = self.report(c, execution_id=failed.execution_id, round_id='repair-3')
        q, d = self.predict()
        retry = self.run_g(q)
        repair['correction']['retry_action_id'] = retry.action_id
        self.h.report_action('model', self.task, 'ep1', outcome={'implementation_facts': [repair]})
        self.covered(d)
        self.finish(p)
        self.assertEqual(self.summary(p).cost['retries']['comparable']['value'], 3)
        self.assertEqual(self.summary(q).cost['retries']['comparable']['value'], 0)
        self.assertEqual(self.summary(p).implementation_observation['label'], 1)

    def test_explicit_repair_and_formal_resubmission_are_one_round(self):
        p, c = self.predict()
        failed = self.run_g(p, crash=True)
        repair = self.report(c, execution_id=failed.execution_id)
        q, d = self.predict()
        retry = self.run_g(q)
        repair['correction']['retry_action_id'] = retry.action_id
        for _ in range(2):
            self.h.report_action('model', self.task, 'ep1', outcome={'implementation_facts': [repair]})
        self.covered(d)
        self.finish(p)
        self.assertEqual(self.summary(p).cost['retries']['comparable']['value'], 1)
        self.assertEqual(self.summary(q).cost['retries']['comparable']['value'], 0)

    def test_unlinked_overlapping_repair_keeps_only_trustworthy_lower_bound(self):
        p, c = self.predict()
        self.run_g(p, crash=True)
        self.report(c, event_id='old-method-defect', round_id='old-method-repair')
        q, d = self.predict()
        self.run_g(q)
        self.covered(d)
        self.finish(p)
        for prediction, lower in ((p, 1), (q, 0)):
            with self.subTest(prediction=prediction.prediction_id):
                real = self.summary(prediction).cost['retries']
                self.assertFalse(real['complete'])
                self.assertIsNone(real['comparable']['value'])
                self.assertEqual(real['lower_bound'], lower)
                self.assertTrue(any('association' in gap for gap in real['observed_from']['gaps']))

    def test_abandoned_repair_then_started_reimplementation_are_distinct_rounds(self):
        p, c = self.predict()
        self.run_g(p, crash=True)
        self.report(c, event_id='old-method-defect', round_id='abandoned-repair')
        q = self.predict_g(method={'name': 'replacement method', 'steps': ['Reimplement the failed strategy.']})
        d = self.h.actions.get(self.h.choose_next(q.trace.model_info['decision_action_id'],
                                                 prediction_id=q.prediction_id)['action_id'])
        rework = self.report(d, event_id='replacement-rework', classification='mathematical', round_id='replacement-round')
        retry = self.run_g(q)
        rework['correction']['retry_action_id'] = retry.action_id
        self.h.report_action('model', self.task, 'ep1', outcome={'implementation_facts': [rework]})
        self.covered(d)
        self.finish(p)
        self.assertEqual(self.summary(p).cost['retries']['comparable']['value'], 2)
        self.assertEqual(self.summary(q).cost['retries']['comparable']['value'], 1)

    def test_round_association_requires_a_real_retry_action(self):
        _, c = self.predict()
        repair = self.report(c)
        repair['correction']['retry_action_id'] = 'missing-retry-action'
        before = len(self.h.actions.query(task_id=self.task['task_id']))
        with self.assertRaisesRegex(ValueError, 'actual resubmission'):
            self.h.report_action('model', self.task, 'ep1', outcome={'implementation_facts': [repair]})
        self.assertEqual(len(self.h.actions.query(task_id=self.task['task_id'])), before)

    def test_repeated_choice_reuses_window_before_and_after_execution(self):
        from pathlib import Path
        p, c = self.predict()
        decision_id = p.trace.model_info['decision_action_id']
        for _ in range(2):
            self.assertEqual(self.h.choose_next(decision_id, prediction_id=p.prediction_id)['action_id'], c.action_id)
        self.run_g(p)
        self.assertEqual(self.h.choose_next(decision_id, prediction_id=p.prediction_id)['action_id'], c.action_id)
        selections = [a for a in self.h.actions.query(task_id=self.task['task_id'])
                      if a.parent_action_id == decision_id and a.outcome.get('kind') == 'plan_next_choice']
        self.assertEqual([a.action_id for a in selections], [c.action_id])
        self.assertEqual(implementation_window(p, self.h.actions.query(task_id=self.task['task_id']))['start'], c.started_at)
        work = Path(self.home)/p.prediction_id
        with self.assertRaisesRegex(ValueError, 'consumed|bound|claimed'):
            self.h.execute(self.task, None, str(work/'solve.py'), str(work),
                           prediction_id=p.prediction_id, episode_id='ep1')

    def test_cli_main_flow_selects_before_implementation_and_returns_binding_id(self):
        import io
        from contextlib import redirect_stdout
        from unittest.mock import patch
        from or_harness.cli import main
        self.provider.payload['capability_gain'] = {'decision': 0, 'basis': ['No new mechanism.']}
        candidate = {'strategy_id': 'S04', 'solver': 'highs',
                     'method': {'name': 'selected method', 'steps': ['Implement the selected model.']}}

        def cli(*args):
            output = io.StringIO()
            with patch('or_harness.cli._harness', return_value=self.h), patch.object(self.h, 'close'), redirect_stdout(output):
                self.assertEqual(main(list(args)), 0)
            return json.loads(output.getvalue())

        plan = cli('plan-next', '--task', json.dumps(self.task), '--episode', 'ep1',
                   '--candidates', json.dumps([candidate]))
        decision = plan['result']['decision_action_id']
        prediction = plan['result']['selected_prediction_id']
        self.assertNotIn('choose-next', plan['summary'])
        self.assertIn('implementation observation anchor', plan['summary'])
        self.assertIn(f'orx execute --task ... --prediction {prediction}', plan['summary'])
        chosen = cli('choose-next', '--decision', decision, '--prediction', prediction)
        self.assertIn(chosen['result']['action_id'], chosen['summary'])
        self.assertIn('observation window', chosen['summary'])
        self.assertNotIn('Compatibility choice', chosen['summary'])
        repeated = cli('choose-next', '--decision', decision, '--prediction', prediction)
        self.assertEqual(repeated['result']['action_id'], chosen['result']['action_id'])
        self.assertEqual(len(self.provider.requests), 1)

    def test_d_duplicate_logs_and_formal_host_identity(self):
        p, c = self.predict()
        fact = self.report(c)
        self.h.report_action('verify', self.task, 'ep1', outcome={'implementation_facts': [fact]})
        record = self.run_g(p, crash=True)
        self.report(c, execution_id=record.execution_id, round_id=None)
        actions = self.h.actions.query(task_id=self.task['task_id'], episode_id='ep1')
        facts = collect_facts(actions, [record])
        self.assertEqual(len(facts), 2)
        formal = next(f for f in facts if f['event_id'] == formal_event_id(record.execution_id))
        self.assertEqual(len(formal['provenance']), 2)
        self.assertEqual(len(corrective_rounds(actions, [record], anchor=c.started_at, end=time.time())['rounds']), 1)

    def test_e_probes_planned_steps_and_no_started_correction(self):
        p, c = self.predict()
        self.report(c, excluded=True, round_id=None)
        self.h.report_action('model', self.task, 'ep1', outcome={'steps': ['LP relaxation', 'MILP solve'], 'help': 'API signature'})
        self.run_g(p)
        self.covered(c)
        self.finish(p)
        s = self.summary(p)
        self.assertEqual(s.implementation_observation['label'], 0)
        self.assertEqual(s.cost['retries']['comparable']['value'], 0)

    def test_f_environment_rework_is_cost_not_implementation(self):
        p, c = self.predict()
        self.report(c, classification='environment')
        self.run_g(p)
        self.covered(c)
        self.finish(p)
        s = self.summary(p)
        self.assertEqual(s.implementation_observation['label'], 0)
        self.assertEqual(s.cost['retries']['comparable']['value'], 1)

    def test_g_selection_windows_and_remaining_anchors(self):
        p, c = self.predict()
        self.report(c)
        q, d = self.predict()
        self.report(d, event_id='later-bug', round_id='later-repair')
        actions = self.h.actions.query(task_id=self.task['task_id'], episode_id='ep1')
        self.assertEqual([f['event_id'] for f in implementation_window(p, actions)['facts']], ['bug-1'])
        self.assertEqual([f['event_id'] for f in implementation_window(q, actions)['facts']], ['later-bug'])
        a = self.h.actions.get(p.trace.model_info['decision_action_id']).started_at
        b = self.h.actions.get(q.trace.model_info['decision_action_id']).started_at
        self.assertEqual(len(corrective_rounds(actions, [], anchor=a, end=time.time())['rounds']), 2)
        self.assertEqual(len(corrective_rounds(actions, [], anchor=b, end=time.time())['rounds']), 1)

    def test_h_success_without_host_coverage_stays_unknown(self):
        p, _ = self.predict()
        self.run_g(p)
        self.finish(p, coverage=False)
        s = self.summary(p)
        self.assertIsNone(s.implementation_observation['label'])
        self.assertFalse(s.cost['retries']['complete'])
        self.assertIsNone(s.cost['retries']['comparable']['value'])
        self.assertEqual(s.cost['retries']['lower_bound'], 0)

    def test_i_abandoned_before_execute_is_visible_without_fake_execution(self):
        p, c = self.predict()
        self.report(c, round_id=None)
        result = self.h.close_episode(self.task['task_id'], 'ep1', terminal_state='aborted')
        self.assertEqual(self.h.bank.count(), 0)
        self.assertEqual(len(self.h.bank.pending(task_id=self.task['task_id'])), 0)
        self.assertEqual(result['evaluations'], [])
        gaps = result['calibration_summary']['technical_r']['selected_without_execution']
        self.assertEqual(gaps[0]['prediction_id'], p.prediction_id)
        self.assertEqual(gaps[0]['label'], 1)

    def test_j_material_and_real_next_provider_request_keep_repairs(self):
        p, c = self.predict(hplus=1)
        self.report(c)
        record = self.run_g(p)
        self.covered(c)
        evidence = execution_implementation_evidence(self.h, record)
        self.assertEqual(evidence['label'], 1)
        self.h.close_episode(self.task['task_id'], 'ep1')
        job = jobs(self.h)[0]
        material = self.h.induction_material(induction_id=job['induction_id'])
        serialized = json.dumps(material)
        for needle in ('artifact:failure.py', 'artifact:repair.diff', 'host-log:test/validation'):
            self.assertIn(needle, serialized)
        self.task = {**self.task, 'task_id': 'future-factory'}
        self.predict()
        request = json.dumps(self.provider.requests[-1])
        self.assertIn('artifact:repair.diff', request)
        self.assertIn('host_report', request)

    def test_j_zero_gate_keeps_facts_without_induction(self):
        p, c = self.predict()
        self.report(c)
        record = self.run_g(p)
        self.assertEqual(execution_implementation_evidence(self.h, record)['label'], 1)
        self.assertFalse(jobs(self.h))

    def test_k_versions_are_separate_and_old_records_not_rewritten(self):
        p, c = self.predict()
        record = self.run_g(p)
        self.assertEqual(record.execution_features['technical_outcome']['observation_rule_version'], R_FORMAL_OBSERVATION_VERSION)
        self.assertEqual(p.trace.model_info['risk_convention']['observation_rule_version'], R_OBSERVATION_VERSION)
        self.assertEqual(p.trace.model_info['risk_convention']['scoring_rule_version'], R_SCORING_VERSION)
        self.covered(c)
        self.finish(p)
        self.assertEqual(self.summary(p).implementation_observation['scoring_rule_version'], R_SCORING_VERSION)
        self.assertEqual(self.summary(p).implementation_observation['unit'], 'selected_candidate')
        old = self.h.strategy_predictions.get(p.prediction_id)
        old.trace.model_info['risk_convention'].update(observation_rule_version=R_FORMAL_OBSERVATION_VERSION,
                                                      scoring_rule_version=R_FORMAL_SCORING_VERSION)
        old.trace.model_info['cost_observation_rule_versions']['retries'] = RETRIES_FORMAL_VERSION
        before = copy.deepcopy(old.to_dict())
        s = summarize_real_outcome(self.h, old)
        self.assertEqual(s.implementation_observation['observation_rule_version'], R_FORMAL_OBSERVATION_VERSION)
        self.assertEqual(s.cost['retries']['observation_rule_version'], RETRIES_FORMAL_VERSION)
        self.assertEqual(old.to_dict(), before)

    def test_l_descriptive_retry_statistics_include_zeros_without_epsilon(self):
        stats = retry_error_statistics([(0, 0), (.1, 1), (2, 3)])
        self.assertEqual(stats['n'], 3)
        self.assertEqual(stats['prediction_by_actual']['no_retry']['n'], 1)
        self.assertEqual(stats['prediction_by_actual']['with_retry']['n'], 2)
        self.assertEqual(stats['log_error']['n'], 2)
        self.assertAlmostEqual(stats['mae'], 1.9/3)
        self.assertGreater(stats['prediction_variance'], 0)

    def test_recovered_internal_defect_is_not_hidden_by_success(self):
        p, c = self.predict()
        record = self.run_g(p)
        self.report(c, execution_id=record.execution_id)
        self.covered(c)
        self.finish(p)
        self.assertEqual(self.summary(p).implementation_observation['label'], 1)
        self.assertEqual(self.summary(p).cost['retries']['comparable']['value'], 1)

    def test_distinct_internal_defects_keep_distinct_identities(self):
        p, c = self.predict()
        record = self.run_g(p)
        self.report(c, execution_id=record.execution_id, event_id='runtime-environment',
                    classification='environment', round_id='environment-rework')
        self.report(c, execution_id=record.execution_id, event_id='runtime-api',
                    round_id='api-repair')
        self.covered(c)
        self.finish(p)
        summary = self.summary(p)
        self.assertEqual(summary.implementation_observation['label'], 1)
        self.assertEqual(len(summary.implementation_observation['facts']), 2)
        self.assertEqual(summary.cost['retries']['comparable']['value'], 2)

    def test_unconfirmed_defect_and_incomplete_retries_do_not_become_zeros(self):
        p, c = self.predict()
        self.report(c, confirmed=False)
        self.run_g(p)
        self.covered(c)
        self.finish(p, coverage=True)
        s = self.summary(p)
        self.assertIsNone(s.implementation_observation['label'])
        self.assertFalse(s.cost['retries']['complete'])
        self.assertNotIn('retries', evaluate_strategy_prediction(p, s).cost['per_dim'])

    def test_new_prediction_after_formal_failure_counts_its_started_retry(self):
        p, _ = self.predict()
        self.run_g(p, crash=True)
        q, d = self.predict()
        self.run_g(q)
        self.covered(d)
        self.finish(p)
        self.assertEqual(self.summary(p).cost['retries']['comparable']['value'], 1)
        self.assertEqual(self.summary(q).cost['retries']['comparable']['value'], 1)

    def test_acceptance_failure_counts_only_when_known_before_retry(self):
        p, _ = self.predict()
        first = self.run_g(p, split=True)
        self.check_g(first)
        q, c = self.predict()
        self.run_g(q)
        self.covered(c)
        self.finish(p)
        self.assertEqual(self.summary(p).cost['retries']['comparable']['value'], 1)

    def test_late_acceptance_failure_does_not_retroactively_invent_retry(self):
        p, _ = self.predict()
        first = self.run_g(p, split=True)
        q, c = self.predict()
        self.run_g(q)
        self.check_g(first)
        self.covered(c)
        self.finish(p)
        self.assertEqual(self.summary(p).cost['retries']['comparable']['value'], 0)

    def test_out_of_window_and_unselected_reports_are_refused(self):
        p, c = self.predict()
        self.predict()
        with self.assertRaisesRegex(ValueError, 'outside.*window'):
            self.report(c)
        with self.assertRaisesRegex(ValueError, 'selected choice'):
            self.report(SimpleNamespace(action_id=p.trace.model_info['decision_action_id']))

    def test_end_action_implementation_report_replays_idempotently(self):
        p, c = self.predict()
        action = self.h.begin_action('model', self.task, 'ep1')
        at = time.time()
        outcome = {'implementation_facts': [{
            'selection_action_id': c.action_id, 'event_id': 'ended-bug',
            'occurred_at': at, 'phase': 'implementation', 'classification': 'implementation',
            'confirmed': True, 'requires_correction': True, 'observer': 'runner',
            'evidence_refs': ['host-log:failure']} ]}
        first = self.h.end_action(action['action_id'], outcome=outcome, task=self.task, episode_id='ep1')
        second = self.h.end_action(action['action_id'], outcome=outcome, task=self.task, episode_id='ep1')
        self.assertEqual(first, second)

    def test_no_correction_started_does_not_count_a_retry(self):
        p, c = self.predict()
        self.report(c, round_id=None)
        self.run_g(p)
        self.covered(c)
        self.finish(p)
        self.assertEqual(self.summary(p).implementation_observation['label'], 1)
        self.assertEqual(self.summary(p).cost['retries']['comparable']['value'], 0)

    def test_abandoned_implementation_is_context_for_later_real_evidence(self):
        p, c = self.predict()
        self.report(c, event_id='abandoned-api-bug', round_id=None)
        q, d = self.predict()
        record = self.run_g(q)
        self.covered(d)
        evidence = execution_implementation_evidence(self.h, record)
        self.assertEqual(evidence['label'], 0)
        self.assertIn('abandoned-api-bug', json.dumps(evidence['prior_selected_implementations']))

    def test_late_complete_corrective_report_refreshes_without_rewriting(self):
        p, c = self.predict()
        self.report(c)
        self.run_g(p)
        self.covered(c)
        self.finish(p, coverage=False)
        closed = self.h.close_episode(self.task['task_id'], 'ep1')
        self.assertNotIn('retries', closed['evaluations'][0]['cost']['per_dim'])
        snapshot = copy.deepcopy(self.h.strategy_predictions.get(p.prediction_id).to_dict())
        anchor = self.h.actions.get(p.trace.model_info['decision_action_id']).started_at
        stop = next(a.ended_at for a in self.h.actions.query(task_id=self.task['task_id'])
                    if a.action_type == 'finish_task')
        self.h.report_action('verify', self.task, 'ep1', outcome={'correction_coverage': {
            'from': anchor, 'through': stop, 'complete': True, 'observer': 'late-runner',
            'evidence_refs': ['host-log:complete-span']}})
        self.assertEqual(self.h.calibration_summary()['conditional_cost']['retries']['n'], 1)
        self.assertEqual(self.h.strategy_predictions.get(p.prediction_id).to_dict(), snapshot)
        stored = self.h.strategy_prediction_evaluations(task_id=self.task['task_id'])[0]
        self.assertNotIn('retries', stored['cost']['per_dim'])

    def test_legacy_numeric_feedback_is_not_current_calibration(self):
        row = {'evaluation_id': 'legacy-e', 'prediction_id': 'legacy-p',
            'implementation_observation': {'observation_rule_version': R_FORMAL_OBSERVATION_VERSION, 'label': 0},
            'risk_predicted': [{'event': 'implementation_failure', 'probability': .3}],
            'risk_actual': [{'event': 'implementation_failure', 'label': 'not_occurred',
                             'brier': .09, 'observation_rule_version': R_FORMAL_OBSERVATION_VERSION}],
            'cost_measured': {'retries': {'predicted': .1, 'actual': 2}},
            'cost_log_ratio': {'retries': 3},
            'cost_observed_fact': {'values': {'retries': 2},
                'conditions': {'retry_observation_rule_version': RETRIES_FORMAL_VERSION}}}
        before = copy.deepcopy(row)
        projection = provider_feedback_pair(row)
        self.assertNotIn('retries', projection.get('cost_measured', {}))
        self.assertNotIn('retries', projection.get('cost_log_ratio', {}))
        self.assertNotIn('risk_predicted', projection)
        self.assertFalse(any('brier' in e for e in projection.get('risk_actual', [])))
        self.assertEqual(row, before)
        self.assertEqual(global_c_statistics([row])['dimensions']['retries']['n'], 0)
        self.assertEqual(global_c_statistics([row])['dimensions']['retries']['n_predicted_actual'], 0)
        self.assertFalse(global_r_statistics({'technical_r': {'observation_rule_version': R_FORMAL_OBSERVATION_VERSION}})['events'])

    def test_actual_retry_distribution_does_not_require_a_retry_forecast(self):
        row = {'evaluation_id': 'current-e', 'task_id': 't', 'episode_id': 'e',
               'cost_observed_fact': {'values': {'retries': 2}, 'conditions': {
                   'retry_observation_rule_version': RETRIES_FUTURE_VERSION}}}
        entry = global_c_statistics([row])['dimensions']['retries']
        self.assertEqual(entry['n'], 1)
        self.assertEqual(entry['p90'], 2)
        self.assertEqual(entry['corrective_round_error']['n'], 0)

    def test_timezone_runtime_data_and_environment_classification(self):
        from pathlib import Path
        from or_harness.execution.executor import SafePythonExecutor, ExecutionOutcome
        from or_harness.world_model.r_signal import execution_receipt
        work = Path(self.home)/'tz-runtime'
        work.mkdir()
        script = work/'solve.py'
        script.write_text("import json\nfrom zoneinfo import ZoneInfo\n"
                          "zone = ZoneInfo('Asia/Shanghai')\n"
                          "json.dump({'status':'optimal','objective_value':0},open('result.json','w'))\n")
        outcome = SafePythonExecutor().run(script, work, 'python')
        self.assertEqual(outcome.status, 'optimal', outcome.stderr)
        blocked = ExecutionOutcome(status='error', solver='python', executed=True,
            stderr='SandboxDependencyError: timezone runtime data unavailable')
        fact = execution_receipt(blocked, {})
        self.assertIsNone(fact['label'])
        self.assertEqual(fact['failure_cause'], 'environment_unavailable')
