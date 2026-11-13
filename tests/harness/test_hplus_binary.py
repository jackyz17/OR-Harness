"""Binary H+ through prediction, real execution, close, induction and feedback."""
import copy
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from test_rule_versions import RuleCase, SHORT_RULE, METHOD
from or_harness.api import ORHarness
from or_harness.cli import main
from or_harness.strategy.embedding_index import LocalHashEmbeddingBackend
from or_harness.world_model.conditions import get_value
from or_harness.world_model.hplus import jobs, receipt_observation, signal_from_payload
from or_harness.world_model.provider import WorldModelProvider


class BinaryProvider(WorldModelProvider):
    name = 'binary-hplus-fixture'

    def __init__(self):
        self.gain = {'decision':1,'claim':'A signed endpoint choice can extend upper-only knowledge to minimization.',
            'basis':['candidate steps and current rule short views'],
            'applies_to':['integer intervals with objective monotone in X'],
            'verification_conditions':[{'condition':'Check the selected lower endpoint and the independently computed optimum.',
                                        'check_basis':'captured enumeration script, variables and task checker'}]}
        self.requests = []

    def predict(self, request, timeout_s=None):
        self.requests.append(copy.deepcopy(request))
        payload={'benefit':{**{k:v for k,v in request['benefit_convention'].items() if k in ('kind','metric','unit')},'value':0.9},
                 'cost':{'tool_calls':3},'risk':{'events':[{'event':'implementation_failure','probability':0.1}]}}
        if self.gain is not None:
            payload['capability_gain']=copy.deepcopy(self.gain)
        return {'payload':payload,'usage':{'prompt_tokens':10,'completion_tokens':5},'latency_s':0.001}


class BinaryCase(RuleCase):
    def setUp(self):
        super().setUp()
        self.provider=BinaryProvider()
        self.h.world_model=self.provider
        self.h.strategy_predictions.provider=self.provider

    def predict(self,name,decision=1,*,parameters=None,boundary=False):
        self.provider.gain['decision']=decision
        n,c=parameters or ((12,0) if boundary else (641,130))
        task={'task_id':name,'family':'assignment','base_task_id':name,
              'description':f'Nonnegative integer X,Y. X+Y <= {n}; X >= 2Y+{c}. '+
                  ('Minimize (X-3)^2+Y; objective is not monotone.' if boundary else 'Minimize 2X-5Y.')}
        candidate={'strategy_id':'endpoint','solver':'python','method':METHOD}
        return self.h.predict_strategy_outcome(task,candidate,name),task

    def closed(self,name='hplus-training',decision=1):
        p,task=self.predict(name,decision)
        _,record=self.solve(name,prediction_id=p.prediction_id)
        close=self.h.close_episode(name,name)
        return p,record,close

    def empty(self,job_id,**extra):
        self.h.induction_material(induction_id=job_id)
        return self.h.induce(induction_id=job_id,notes=['Captured evidence reviewed; no supported new rule.'],
                             review={'outcome':'insufficient_material','conclusion':'No independent check yet.',**extra})


class TestBinaryHplus(BinaryCase):
    def test_positive_closes_to_one_job_and_outer_runner_settles_it(self):
        p,record,closed=self.closed()
        self.assertTrue(closed['induction_schedule']['pending_ids'])
        job_id=closed['induction_schedule']['pending_ids'][0]
        self.assertEqual(closed['induction_gate']['source'],'frozen_hplus')
        from or_harness.world_model.hplus import configure
        with self.assertRaises(ValueError):configure(self.h,'frozen')
        before=self.h.bind_capability_maintenance(p.prediction_id)
        self.assertEqual(before['state'],'not_adopted')
        calls=[]
        def review(material,job):
            calls.append(job['induction_id'])
            self.assertEqual(material['induction_gate']['induction_id'],job_id)
            return {'notes':['Facts reviewed; effect is not independently checked.'],
                    'review':{'outcome':'insufficient_material','hypothesis_outcome':'unknown'}}
        result=self.h.process_induction_jobs(review)
        self.assertTrue(result['completed'])
        self.assertEqual(calls,[job_id])
        job=jobs(self.h)[0]
        action=self.h.actions.get(job['action_id'])
        self.assertEqual(action.action_type,'induce')
        self.assertNotEqual(action.action_id,job['solve_action_id'])
        self.assertIn(record.execution_id,action.outcome['execution_ids'])
        self.assertEqual(self.h.capability_maintenance_binding(p.prediction_id)['adoption_action_id'],action.action_id)
        self.assertTrue(self.h.close_episode(record.task_id,record.task_id)['induction_schedule']['completed'])
        self.assertEqual(self.h.process_induction_jobs(review)['results'],[])
        self.assertEqual(len(self.h.actions.query(action_type='induce')),1)
        receipt=get_value(self.h.store,job['receipt_ref'])
        self.assertEqual(receipt['verification'][0]['state'],'unknown')

    def test_zero_missing_illegal_and_unexecuted_are_distinct_and_cannot_bypass(self):
        for n,value,state in [('zero',0,'explicit'),('missing',None,'missing'),('illegal','1','invalid')]:
            if state=='missing':
                self.provider.gain.pop('decision',None)
                p,task=self.predict(n,0)
                # Omit the decision in the actual next prediction, not by editing a saved one.
                self.provider.gain.pop('decision')
                p=self.h.predict_strategy_outcome(task,{'strategy_id':'endpoint','solver':'python','method':METHOD},n)
            else:
                p,task=self.predict(n,value)
            _,record=self.solve(n,prediction_id=p.prediction_id)
            closed=self.h.close_episode(n,n,induction_decision='allow')
            j=next(j for j in closed['induction_schedule']['jobs'] if j['prediction_id']==p.prediction_id)
            self.assertEqual(j['signal']['decision_status'],state)
            self.assertEqual(j['state'],'skipped')
            with patch('or_harness.world_model.induction_material.build_compact_material',side_effect=AssertionError('expanded')):
                self.assertFalse(self.h.induction_material(task_id=n)['material_assembled'])
            with self.assertRaises(ValueError):
                self.h.induce(induction_id=j['induction_id'],relations=[{**SHORT_RULE,'evidence':[record.execution_id]}])
        p,_=self.predict('not-executed',1)
        result=self.h.close_episode('not-executed','not-executed')
        self.assertIn(p.prediction_id,result['induction_schedule']['unobserved_prediction_ids'])
        self.assertFalse(jobs(self.h,task_id='not-executed'))

    def test_failed_positive_and_incomplete_hypothesis_still_require_review(self):
        self.provider.gain={'decision':1}
        p,task=self.predict('failed',1)
        work=Path(self.home)/'crash';work.mkdir()
        (work/'solve.py').write_text("raise ValueError('broken code')\n")
        r=self.h.execute(task,None,str(work/'solve.py'),str(work),prediction_id=p.prediction_id,episode_id='failed')
        self.h.record(r)
        close=self.h.close_episode('failed','failed',terminal_state='failed')
        job=close['induction_schedule']['jobs'][0]
        self.assertEqual(job['state'],'pending')
        self.assertTrue(job['signal']['gaps'])
        result=self.empty(job['induction_id'])
        self.assertEqual(result['induction_receipt']['outcome'],'insufficient_material')
        self.assertIsNone(receipt_observation(self.h,jobs(self.h)[0])['label'])

    def test_retry_uses_same_material_action_and_receipt_after_reopen(self):
        p,r,close=self.closed('retry')
        job_id=close['induction_schedule']['pending_ids'][0]
        material=self.h.induction_material(induction_id=job_id)
        with patch.object(self.h,'_end_induce_action',side_effect=RuntimeError('interrupted before commit')):
            with self.assertRaises(RuntimeError):
                self.h.induce(induction_id=job_id,relations=[{**SHORT_RULE,'evidence':[r.execution_id]}])
        self.assertEqual(self.h.sbank.count(),0)
        action_id=jobs(self.h)[0]['action_id']
        other=ORHarness(home=self.home,embedding=LocalHashEmbeddingBackend())
        try:
            self.assertEqual(other.induction_material(induction_id=job_id),material)
            args={'induction_id':job_id,'relations':[{**SHORT_RULE,'evidence':[r.execution_id]}]}
            result=other.induce(**args)
            repeated=other.induce(**args)
            self.assertTrue(repeated['already_completed'])
            self.assertEqual(result['action']['action_id'],action_id)
            self.assertEqual(len(other.actions.query(action_type='induce')),1)
            self.assertEqual(other.sbank.count(),1)
        finally:other.close()

    def test_zero_execution_can_be_a_positive_jobs_cross_task_control(self):
        _,zero,_=self.closed('negative-control',0)
        _,positive,closed=self.closed('positive')
        job_id=closed['induction_schedule']['pending_ids'][0]
        material=self.h.induction_material(induction_id=job_id)
        shown=material['material']+material['related_history']['executions']
        self.assertIn(zero.execution_id,[r['execution_id'] for r in shown])
        result=self.h.induce(induction_id=job_id,relations=[{**SHORT_RULE,'evidence':[positive.execution_id,zero.execution_id]}])
        self.assertEqual(result['saved'],1)

    def test_independent_checks_confirm_increment_and_next_provider_sees_only_prior_relevant_receipt(self):
        p,record,close=self.closed('derivation')
        job_id=close['induction_schedule']['pending_ids'][0]
        self.h.induction_material(induction_id=job_id)
        result=self.h.induce(induction_id=job_id,relations=[{**SHORT_RULE,'evidence':[record.execution_id]}],
                             review={'hypothesis_outcome':'refuted','deviation_reason':'The initial upper-only idea needs a sign-aware boundary.'})
        entry_id=result['relations'][0]['saved']
        job=jobs(self.h)[0]
        self.assertIsNone(receipt_observation(self.h,job)['label'])
        self.solve('independent',use=entry_id,parameters=(812,200))
        self.h.close_episode('independent','independent')
        self.solve('boundary',use=entry_id,boundary=True)
        self.h.close_episode('boundary','boundary')
        observation=receipt_observation(self.h,job)
        self.assertEqual(observation['label'],1,observation)
        self.assertEqual(observation['hypothesis_outcome'],'refuted')
        self.predict('next',0)
        next_id=self.h.strategy_predictions.query(task_id='next')[0].prediction_id
        feedback=self.provider.requests[-1]['prediction_context']['hplus_feedback']
        self.assertEqual(feedback['settled_examples'][0]['label'],1)
        self.assertEqual(feedback['settled_examples'][0]['rule_versions'][0]['version'],1)
        self.assertNotIn(next_id,[j['prediction_id'] for j in feedback['settled_examples']])
        from or_harness.world_model.hplus import feedback as feedback_view
        self.assertEqual(feedback_view(self.h,task={'family':'routing'})['settled_examples'],[])

    def test_strict_signal_numbers_and_legacy_reading(self):
        for value in (True,False,'1',2,-1,10**400,float('nan'),float('inf'),[],{}):
            self.assertEqual(signal_from_payload({'decision':value})['decision_status'],'invalid')
        for value in (0,1,0.0,1.0):
            self.assertEqual(signal_from_payload({'decision':value})['decision'],int(value))
        forged=signal_from_payload({'decision':1,'verification_conditions':[
            {'condition':'inspect actual code','fact_bound':True,'effect_verified':True}]})
        self.assertFalse(forged['verification_conditions'][0]['fact_bound'])
        self.assertFalse(forged['verification_conditions'][0]['effect_verified'])
        from or_harness.world_model.contracts import CapabilityGain
        old=CapabilityGain.from_dict({'assessment':'expected','claim':'old stance'})
        self.assertIsNone(old.decision)
        self.assertNotIn('decision',old.to_dict())

    def test_normal_cli_material_submit_receipt_and_restart_status(self):
        p,r,closed=self.closed('cli')
        job_id=closed['induction_schedule']['pending_ids'][0]
        def cli(args):
            out=io.StringIO()
            with patch.dict('os.environ',{'OR_EMBEDDING_BACKEND':'local-hashing'}),redirect_stdout(out):
                code=main(['--home',self.home]+args)
            self.assertEqual(code,0,out.getvalue())
            return json.loads(out.getvalue()),out.getvalue().strip()
        material,text=cli(['induction-material','--induction-id',job_id])
        self.assertEqual(material['result']['budget']['chars_used'],len(text))
        self.assertLessEqual(len(text),32000)
        result,_=cli(['induce','--induction-id',job_id,'--review',json.dumps({
            'outcome':'hypothesis_refuted','hypothesis_outcome':'refuted',
            'conclusion':'Captured result disproved the initial expectation; no reusable boundary established.'}),
            '--note','Reviewed actual script and original check.'])
        self.assertEqual(result['result']['induction_receipt']['prediction_id'],p.prediction_id)
        status,_=cli(['induction-status','--task','cli'])
        self.assertTrue(status['result']['completed'])
        self.assertEqual(status['result']['jobs'][0]['state'],'completed')

    def test_material_identity_mismatch_and_different_completed_submission_are_refused(self):
        _,a,ca=self.closed('source-a');_,b,cb=self.closed('source-b')
        ja=ca['induction_schedule']['pending_ids'][0];jb=cb['induction_schedule']['pending_ids'][0]
        ma=self.h.induction_material(induction_id=ja)
        with self.assertRaises(ValueError):
            self.h.induce(induction_id=jb,relations=[{**SHORT_RULE,'material_id':ma['material_id'],'evidence':[b.execution_id]}])
        self.empty(ja)
        with self.assertRaises(ValueError):
            self.h.induce(induction_id=ja,notes=['different review'])

    def test_covered_negative_needs_frozen_baseline_and_actual_fact_references(self):
        entry_id,_,_=self.seed()
        self.activate(entry_id)
        _,r,c=self.closed('covered-positive')
        job_id=c['induction_schedule']['pending_ids'][0]
        j=jobs(self.h)[0]
        self.empty(job_id,outcome='already_covered',comparison={'sufficient':True,
            'source':'fixture reviewer inspected the frozen input and actual result',
            'assessment':'No additional executable rule relative to this snapshot.',
            'evidence_refs':[j['source_fact_ref'],j['signal']['source_snapshot_ref']]})
        self.assertEqual(receipt_observation(self.h,jobs(self.h)[0])['label'],0)
        _,r,c=self.closed('unsupported-negative')
        self.empty(c['induction_schedule']['pending_ids'][0],outcome='no_increment',comparison={'sufficient':True})
        self.assertIsNone(receipt_observation(self.h,jobs(self.h)[-1])['label'])

    def test_strategy_switch_owns_each_signal_and_late_verification_is_in_receipt(self):
        p1,task=self.predict('switch',1)
        _,r1=self.solve('switch',prediction_id=p1.prediction_id)
        p2,_=self.predict('switch',0)
        _,r2=self.solve('switch',prediction_id=p2.prediction_id)
        closed=self.h.close_episode('switch','switch')
        items=closed['induction_schedule']['jobs']
        positive=next(j for j in items if j['prediction_id']==p1.prediction_id)
        self.assertEqual(positive['execution_ids'],[r1.execution_id])
        self.assertEqual(next(j for j in items if j['prediction_id']==p2.prediction_id)['execution_ids'],[r2.execution_id])
        old=(r1.execution_features['task_check']['acceptance_rule'])['spec']
        check={**old,'hplus_verification':[{'prediction_id':p1.prediction_id,'condition_index':0,'state':'checked',
            'source':'fixture auditor inspected the selected endpoint and independent integer checker',
            'evidence_refs':['artifacts.submitted_script','task_check.checks']}]}
        self.h.check_task_result(r1.execution_id,check)
        result=self.empty(positive['induction_id'])
        self.assertEqual(result['induction_receipt']['verification'][0]['state'],'checked')

    def test_nohplus_zero_review_uses_pre_prediction_library_and_frozen_mode_blocks_learning(self):
        from or_harness.world_model.hplus import configure
        configure(self.h,'nohplus')
        _,record,closed=self.closed('control-zero',0)
        job_id=closed['induction_schedule']['pending_ids'][0]
        # A later unrelated library addition must not enter this control's baseline.
        from or_harness.core.schema import StrategicEntry
        self.h.sbank.add(StrategicEntry(entry_id='',strategy_id='later',pattern={'predicates':{}},
            claim={'text':'future knowledge','method':{'steps':['future operation']}},
            verification={'state':'verified'}))
        material=self.h.induction_material(induction_id=job_id)
        self.assertEqual(material['existing_knowledge'],[])
        self.assertTrue(jobs(self.h)[0]['control_snapshot_ref'])
        self.empty(job_id)
        configure(self.h,'frozen')
        with self.assertRaises(ValueError):
            self.h.induce(notes=['cannot learn during fixed-memory testing'])

    def test_positive_revision_maximum_only_to_lower_endpoint_then_independent_90(self):
        _,training=self.solve()
        self.h.close_episode('training','training')
        upper={**SHORT_RULE,'when':'Maximize an objective increasing in X over a computable integer X interval.',
               'do':'Enumerate Y, take the upper X endpoint for maximization, and compare objectives. Disable on a nonmonotone objective.'}
        entry_id=self.h.induce(relations=[{**upper,'evidence':[training.execution_id]}])['relations'][0]['saved']
        upper_method={'name':'upper endpoint for maximization','steps':[upper['do']]}
        _,old_transfer=self.solve('old-max-transfer',parameters=(941,190),maximize=True,use=entry_id,performed_method=upper_method)
        self.h.close_episode('old-max-transfer','old-max-transfer')
        _,old_boundary=self.solve('old-boundary',boundary=True,use=entry_id)
        self.h.close_episode('old-boundary','old-boundary')
        self.assertEqual(self.h.sbank.get(entry_id).rule['active_version'],1)
        p,_=self.predict('adapt-min',1,parameters=(1000,300))
        _,r=self.solve('adapt-min',parameters=(1000,300),prediction_id=p.prediction_id)
        self.assertEqual(r.quality['objective'],367)
        closed=self.h.close_episode('adapt-min','adapt-min')
        job_id=closed['induction_schedule']['pending_ids'][0]
        material=self.h.induction_material(induction_id=job_id)
        regressions=[]
        for old in (training,old_transfer,old_boundary):
            boundary=old.execution_id==old_boundary.execution_id
            assessment={'entry_id':entry_id,'version':2,'applicability':'not_applicable' if boundary else 'applies',
                'response':'disable' if boundary else None,'method_fidelity':'faithful','effect_coverage':'complete',
                'source':'fixture auditor rechecked captured old scripts against sign-aware revised rule',
                'assessment':'Old maximum uses upper endpoint; nonmonotone fallback enumerates the full set. Prior integer checks cover the effect.',
                'evidence_refs':['artifacts.submitted_script','artifacts.workspace_result']}
            regressions.append({'kind':'regression','execution_id':old.execution_id,'assessment':assessment})
        result=self.h.induce(induction_id=job_id,relations=[{**SHORT_RULE,'entry_id':entry_id,'operation':'revise',
            'evidence':[r.execution_id],'material_id':material['material_id'],'validation':regressions}],
            review={'comparison':{'sufficient':True,'substantive':True,'changed_fields':['when','do'],
                'source':'fixture reviewer compared frozen upper-only rule with sign-aware revision',
                'assessment':'Fixed-Y minimization now selects the lower endpoint; the prior rule only covers maximization.',
                'evidence_refs':[next(j for j in jobs(self.h) if j['induction_id']==job_id)['source_fact_ref'],
                                 next(j for j in jobs(self.h) if j['induction_id']==job_id)['signal']['source_snapshot_ref']]}})
        self.assertEqual(result['induction_receipt']['rule_updates'][0]['version'],2)
        self.assertEqual(self.h.sbank.get(entry_id).rule['active_version'],1)
        # Freeze a prediction before the later independent checks exist.
        frozen,_=self.predict('before-validation',0)
        context=self.h.get_prediction_context(frozen.trace.model_info['prediction_context_id'])
        old_feedback=copy.deepcopy(context.hplus_feedback)
        p,_=self.predict('independent',0)
        _,independent=self.solve('independent',use=entry_id,version=2,prediction_id=p.prediction_id)
        self.h.close_episode('independent','independent')
        self.assertEqual(independent.quality['objective'],90)
        p,_=self.predict('new-boundary',0,boundary=True)
        self.solve('new-boundary',boundary=True,use=entry_id,version=2,prediction_id=p.prediction_id)
        self.h.close_episode('new-boundary','new-boundary')
        self.assertEqual(self.h.sbank.get(entry_id).rule['active_version'],2)
        _,wrong=self.solve('independent-upper',lower=False,performed_method=upper_method)
        self.assertEqual(wrong.quality['objective'],92)
        self.assertEqual(wrong.execution_features['task_check']['state'],'failed')
        self.assertEqual(context.provider_view()['hplus_feedback'],old_feedback)
        self.predict('covered-next',0)
        feedback=self.provider.requests[-1]['prediction_context']['hplus_feedback']
        row=next(e for e in feedback['settled_examples'] if e['prediction_id']==result['induction_receipt']['prediction_id'])
        self.assertEqual(row['label'],1)
        self.assertEqual(row['rule_versions'][0]['version'],2)
        binding=self.h.capability_maintenance_binding(p.prediction_id)
        self.assertIsNone(binding) # zero-signal independent trial did not invent an induction action

    def test_no_implicit_negative_write_when_an_unrelated_positive_is_pending(self):
        _,zero,_=self.closed('zero-with-pending',0)
        _,positive,closed=self.closed('real-positive')
        self.h.induction_material(induction_id=closed['induction_schedule']['pending_ids'][0],related_top_k=0)
        with self.assertRaises(ValueError):
            self.h.induce(relations=[{**SHORT_RULE,'evidence':[zero.execution_id]}])
        from or_harness.world_model.hplus import configure
        with self.assertRaises(ValueError):configure(self.h,'manual')
