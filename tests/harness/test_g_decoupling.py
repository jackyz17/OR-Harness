"""G assessment belongs to world-model evaluation, not memory evidence."""
import copy
import io
import json
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

from g_fixtures import GCase, COUPLED, SPLIT
from or_harness.cli import main
from or_harness.strategy.evidence_access import retain_record, resolve_execution
from or_harness.strategy.verification import verify_task_result
from or_harness.world_model.conditions import get_value, put_value
from or_harness.world_model.g_signal import observe_faithful_g, read_g_assessment, method_key
from or_harness.world_model.hplus import get_job


class TestGDecoupling(GCase):
    def value_count(self):
        return self.h.store.conn.execute(
            "SELECT count(*) FROM condition_values WHERE kind='g_assessment'").fetchone()[0]

    def assert_memory_is_generic(self, record):
        self.assertNotIn('g_attribution', record.execution_features['task_check'])
        raw=self.h.store.loads(self.h.store.conn.execute(
            'SELECT payload FROM executions WHERE execution_id=?',(record.execution_id,)).fetchone()[0])
        reference=raw['execution_features']['task_check']['report_ref']
        self.assertNotIn('g_attribution',get_value(self.h.store,reference))
        self.assertNotIn('g_assessment_ref',record.execution_features)

    def test_staged_execution_check_record_close_and_next_request_keep_labels(self):
        expected={}
        for episode,method,split,fidelity,failure,label in (
            ('passed',COUPLED,False,'faithful','unknown',1),
            ('omission',SPLIT,True,'faithful','candidate_method',0),
            ('implementation',COUPLED,True,'deviated','implementation',None)):
            p=self.predict_g(method,episode)
            expected[p.prediction_id]=label
            r=self.run_g(p,split=split,record=False)
            checked=self.check_g(r,fidelity=fidelity,failure=failure)
            self.assertNotIn('g_attribution',checked['report'])
            self.assertTrue(checked['staged'])
            self.h.record(r)
            decoded=self.h.bank.get(r.execution_id)
            self.assert_memory_is_generic(decoded)
            self.assertEqual(decoded.cost.measured,r.cost.measured)
            self.assertEqual(decoded.execution_features['cost_provenance'],r.execution_features['cost_provenance'])
            closed=self.close_g(episode)
            benefit=closed['evaluations'][0]['benefit']
            self.assertEqual(benefit.get('observed'),label)
            self.assertEqual(benefit['attribution_ref'],checked['g_assessment_ref'])
            evaluation=self.h.get_strategy_evaluation(closed['evaluations'][0]['evaluation_id'])
            self.assertEqual(evaluation['benefit']['attribution']['method_fidelity'],fidelity)
            self.assertEqual(evaluation['risk']['implementation_observation']['label'],0)
            saved=self.h.store.loads(self.h.store.conn.execute(
                'SELECT value FROM meta WHERE key=?',
                ('strategy_evaluation|'+evaluation['evaluation_id'],)).fetchone()[0])
            self.assertNotIn('attribution',saved['benefit'])
            self.assertEqual(saved['benefit']['attribution_ref'],checked['g_assessment_ref'])
            before=self.value_count()
            self.close_g(episode)
            self.assertEqual(self.value_count(),before)
        self.predict_g(episode='next')
        block=self.provider.requests[-1]['prediction_context']['prediction_execution_pairs']
        self.assertTrue(block['pairs'])
        for row in block['pairs']:
            self.assertEqual(row['benefit'].get('observed'),expected[row['prediction_id']])
        compared={case[side]['observed'] for case in block['same_task_comparisons'] for side in ('left','right')}
        self.assertEqual(compared,{0,1})
        self.assertEqual(self.value_count(),3)

    def test_plain_task_check_needs_no_g_report_and_hplus_review_still_runs(self):
        self.provider.payload['capability_gain']={'decision':1,
            'claim':'Check the factory coupling boundary.',
            'verification_conditions':[{'condition':'Check the recorded same-factory condition.'}]}
        p=self.predict_g()
        r=self.run_g(p,record=False)
        result=self.check_g(r,evidence=False)
        self.assertEqual(result['state'],'passed')
        self.assertNotIn('g_assessment_ref',result)
        self.h.record(r)
        closed=self.close_g()
        self.assertNotIn('observed',closed['evaluations'][0]['benefit'])
        self.assertEqual(self.value_count(),0)
        self.assertEqual(len(self.provider.requests),1)
        job=closed['induction_schedule']['pending_ids'][0]
        self.h.induction_material(induction_id=job)
        completed=self.h.induce(induction_id=job,review={'outcome':'insufficient_material',
            'conclusion':'The original check is valid but does not establish a reusable increment.'})
        self.assertEqual(get_job(self.h,job)['state'],'completed')
        self.assertEqual(completed['induction_receipt']['outcome'],'insufficient_material')

    def test_late_recheck_does_not_borrow_old_assessment_or_change_frozen_input(self):
        p=self.predict_g()
        r=self.run_g(p)
        first=self.check_g(r)
        closed=self.close_g()
        following=self.predict_g(episode='next')
        input_ref=following.trace.model_info['hplus']['input_ref']
        frozen=copy.deepcopy(get_value(self.h.store,input_ref))
        ordinary=self.check_g(r,evidence=False)
        from or_harness.world_model.episode_closeout import build_paired_feedback
        def current_pair():
            return next(row for row in build_paired_feedback(self.h)['pairs'] if row['prediction_id']==p.prediction_id)
        current=current_pair()
        self.assertEqual(ordinary['state'],'passed')
        self.assertEqual(current['benefit']['label_eligibility'],'unknown')
        self.assertNotIn('observed',current['benefit'])
        self.assertEqual(get_value(self.h.store,input_ref),frozen)
        second=self.check_g(r)
        self.assertNotEqual(second['g_assessment_ref'],first['g_assessment_ref'])
        current=current_pair()
        self.assertEqual(current['benefit']['observed'],1)
        self.assertEqual(current['task_check']['action_id'],second['action_id'])
        self.assertEqual(self.h.get_strategy_evaluation(closed['evaluations'][0]['evaluation_id'])[
            'benefit']['attribution_ref'],first['g_assessment_ref'])
        self.assert_memory_is_generic(self.h.bank.get(r.execution_id))

    def test_changed_check_or_missing_world_model_evidence_keeps_g_unknown(self):
        p=self.predict_g()
        r=self.run_g(p)
        checked=self.check_g(r)
        record=self.h.bank.get(r.execution_id)
        report=copy.deepcopy(record.execution_features['task_check'])
        report['scope']['basis']=[]
        self.h.bank.set_task_check(r.execution_id,report)
        summary=SimpleNamespace(benefit={},eligibility={})
        observe_faithful_g(summary,[self.h.bank.get(r.execution_id)],p,store=self.h.store)
        self.assertNotIn('observed',summary.benefit)
        self.h.bank.set_task_check(r.execution_id,record.execution_features['task_check'])
        future=copy.deepcopy(get_value(self.h.store,checked['g_assessment_ref']))
        future['evaluation_version']='wm-g-assessment/999'
        with self.h.store.transaction() as conn:
            future_ref=put_value(conn,self.h.store,'g_assessment',future)
            conn.execute("UPDATE meta SET value=? WHERE key LIKE 'g_assessment|%'",
                (self.h.store.dumps({'value_ref':future_ref}),))
        summary=SimpleNamespace(benefit={},eligibility={})
        observe_faithful_g(summary,[self.h.bank.get(r.execution_id)],p,store=self.h.store)
        self.assertNotIn('observed',summary.benefit)
        with self.h.store.transaction() as conn:
            conn.execute("UPDATE meta SET value=? WHERE key LIKE 'g_assessment|%'",
                (self.h.store.dumps({'value_ref':checked['g_assessment_ref']}),))
        closed=self.close_g()
        self.assertEqual(closed['evaluations'][0]['benefit']['observed'],1)
        with self.h.store.transaction() as conn:
            conn.execute('DELETE FROM condition_values WHERE value_id=?',(checked['g_assessment_ref'],))
        evaluation=self.h.get_strategy_evaluation(closed['evaluations'][0]['evaluation_id'])
        self.assertEqual(evaluation['risk']['implementation_observation']['label'],0)
        self.assertTrue(evaluation['benefit']['attribution']['issues'])
        ordinary=self.check_g(r,evidence=False)
        self.assertIn('calibration_republished',ordinary)
        from or_harness.world_model.episode_closeout import build_paired_feedback
        current=next(row for row in build_paired_feedback(self.h)['pairs'] if row['prediction_id']==p.prediction_id)
        self.assertNotIn('observed',current['benefit'])
        self.assertEqual(current['implementation_observation']['label'],0)

    def test_historical_inline_read_and_normal_rewrite_preserve_g_once(self):
        p=self.predict_g()
        r=self.run_g(p)
        checked=self.check_g(r)
        record=self.h.bank.get(r.execution_id)
        audit=copy.deepcopy(get_value(self.h.store,checked['g_assessment_ref'])['attribution'])
        audit['method_actual']=method_key(record.method_actual)
        audit['method_planned']=method_key(record.method_planned)
        audit.pop('method_actual_fingerprint');audit.pop('method_planned_fingerprint')
        old=record.to_dict();old['evidence_encoding']='execution-evidence/2'
        old['execution_features']['task_check']['g_attribution']=audit
        with self.h.store.transaction() as conn:
            conn.execute("DELETE FROM meta WHERE key LIKE 'g_assessment|%'")
            conn.execute("DELETE FROM condition_values WHERE kind='g_assessment'")
            conn.execute('UPDATE executions SET payload=? WHERE execution_id=?',
                (self.h.store.dumps(old),r.execution_id))
        before=list(self.h.store.conn.iterdump())
        observed,reference=read_g_assessment(self.h.store,self.h.bank.get(r.execution_id))
        self.assertEqual(observed['method_fidelity'],'faithful')
        self.assertIsNone(reference)
        self.assertEqual(list(self.h.store.conn.iterdump()),before)
        self.h.bank.annotate_features(r.execution_id,{'fixture_amendment':'normal rewrite'})
        record=self.h.bank.get(r.execution_id)
        self.assert_memory_is_generic(record)
        self.assertEqual(self.value_count(),1)
        self.assertEqual(self.close_g()['evaluations'][0]['benefit']['observed'],1)
        with self.h.store.transaction() as conn:
            fact_ref=retain_record(self.h,conn,record)
            conn.execute('DELETE FROM executions WHERE execution_id=?',(r.execution_id,))
        self.assertNotIn('g_attribution',get_value(self.h.store,fact_ref)['checker'])
        retained=resolve_execution(self.h,r.execution_id)
        summary=SimpleNamespace(benefit={},eligibility={})
        observe_faithful_g(summary,[retained],p,store=self.h.store)
        self.assertEqual(summary.benefit['observed'],1)
        self.assertEqual(self.value_count(),1)

    def test_g_write_rolls_back_with_rejected_check_annotation(self):
        p=self.predict_g()
        r=self.run_g(p)
        with patch.object(self.h.bank,'set_task_check',side_effect=RuntimeError('fixture write failure')):
            with self.assertRaisesRegex(RuntimeError,'fixture write failure'):
                self.check_g(r)
        self.assertEqual(self.value_count(),0)
        self.assertIsNone(self.h.bank.get(r.execution_id).execution_features.get('task_check'))
        self.assertEqual(self.h.store.conn.execute(
            "SELECT count(*) FROM meta WHERE key LIKE 'g_assessment|%'").fetchone()[0],0)

    def test_cli_compatibility_routes_optional_g_to_world_model_only(self):
        p=self.predict_g()
        r=self.run_g(p,record=False)
        initial=self.check_g(r)
        check={'semantic_probe':[{'path':'execution_features.solution_variables.CFC','equals':1}],
            'g_attribution':get_value(self.h.store,initial['g_assessment_ref'])['attribution']}
        self.assertNotIn('g_attribution',verify_task_result(r,check))
        def cli(args):
            output=io.StringIO()
            with patch.dict('os.environ',{'OR_EMBEDDING_BACKEND':'local-hashing'}),redirect_stdout(output):
                code=main(['--home',self.home]+args)
            self.assertEqual(code,0,output.getvalue())
            return json.loads(output.getvalue())['result']
        result=cli(['check-task',r.execution_id,'--check',json.dumps(check)])
        self.assertNotIn('g_attribution',result['report'])
        self.assertIn('g_assessment_ref',result)
        cli(['record','--from-staged',r.execution_id])
        self.assertEqual(self.close_g()['evaluations'][0]['benefit']['observed'],1)
        self.assert_memory_is_generic(self.h.bank.get(r.execution_id))
        self.assertEqual(len(self.provider.requests),1)
