"""Updated agent-led policy: normal integer solves, review, management and feedback."""

from decision_fixtures import planned_prediction
import copy
import io
import json
import time
from contextlib import redirect_stdout
from unittest.mock import patch

from review_fixtures import ReviewCase,task,RULE,assessment
from or_harness.api import ORHarness
from or_harness.cli import main
from or_harness.strategy.embedding_index import LocalHashEmbeddingBackend
from or_harness.strategy.rule_versions import inspect_versions,version_row,check_records
from or_harness.strategy.evidence_access import held_records,resolve_execution
from or_harness.world_model.conditions import get_value,put_value
from or_harness.world_model.hplus import jobs,get_job,receipt_observation,feedback
from or_harness.world_model.episode_closeout import enforce_evidence_window,CalibrationPolicy,EvidenceWindowPolicy


class AgentCase(ReviewCase):
    def choose(self,entry,version=1,state='published',**extra):
        return self.h.induce(relations=[{'entry_id':entry,'version':version,
            'publication':{'state':state,'reason':'Agent checked the stated evidence and chooses this offered scope.',**extra}}])

    def manage(self,entry,record,action,verdict='helped',**extra):
        return self.h.induce(effect_attributions=[{'entry_id':entry,'execution_id':record.execution_id,
            'lifecycle_action':action,'verdict':verdict,'note':'Agent inspected this actual adopted use.',**extra}])

    def evict(self):
        return enforce_evidence_window(self.h,policy=EvidenceWindowPolicy(window_episodes=0),
            calibration_policy=CalibrationPolicy(window=1,late_check_grace_days=0))

    def cli(self,args):
        out=io.StringIO()
        with patch.dict('os.environ',{'OR_EMBEDDING_BACKEND':'local-hashing'}),redirect_stdout(out):
            code=main(['--home',self.home]+args)
        self.assertEqual(code,0,out.getvalue())
        return json.loads(out.getvalue())['result']

    def historical_execution(self,name='unpredicted'):
        from pathlib import Path
        ws=Path(self.home)/name;ws.mkdir()
        script=ws/'solve.py'
        script.write_text("import json\njson.dump({'status':'optimal','objective_value':90,'variables':{'X':470,'Y':170}},open('result.json','w'))\n")
        from decision_fixtures import import_legacy_execution
        r=import_legacy_execution(self.h,task(name),'plain-python',str(script),str(ws),episode_id=name,solver='python')
        self.h.record(r)
        self.h.check_task_result(r.execution_id,{'reference_objective':90,'reference_source':'independent'})
        self.h.close_episode(name,name)
        return self.h.bank.get(r.execution_id)



class TestAgentPublicationAndVersions(AgentCase):
    def test_A_B_derivation_and_constructed_boundary_publish_before_real_C(self):
        _,a,ca=self.run_task(task('A',1000,300,maximum=True),decision=1)
        _,b,_=self.run_task(task('B'))
        _,boundary,_=self.run_task(task('constructed-boundary',12,0,boundary=True))
        job=ca['induction_schedule']['pending_ids'][0]
        self.h.induction_material(induction_id=job,execution_ids=[b.execution_id,boundary.execution_id])
        result=self.h.induce(induction_id=job,relations=[{**RULE,'evidence':[a.execution_id,b.execution_id,boundary.execution_id],
            'publication':{'state':'published','reason':'Fix Y; a monotone objective attains its interval optimum at the selected endpoint. Enumerations and nonmonotone fallback agree.',
                'scope':['derived monotone integer interval mechanism','listed enumerations','constructed nonmonotone disable test']}}],
            review={'outcome':'new_rule','conclusion':'Direction-aware endpoint choice and its disable boundary are checked.',
                    'comparison':self.comparison(job,substantive=True,basis='derivation')})
        entry=result['relations'][0]['saved']
        self.assertEqual([a.quality['objective'],b.quality['objective'],boundary.quality['objective']],[2699,90,0])
        self.assertTrue(self.h.sbank.get(entry).is_published)
        self.assertEqual(receipt_observation(self.h,get_job(self.h,job))['label'],1)
        self.assertEqual(result['relations'][0]['publication']['required_checks'],[])
        self.assertEqual(len(self.h.sbank.get(entry).verification['scope']['seed_checks']),3)
        self.assertTrue(any(r.get('evidence')=='strategic_entry' for r in self.h.recall(task('recall'))['recommendations']))
        _,c,closed=self.run_task(task('real-C',812,200),entry_id=entry,
            check_extra={'rule_assessments':[assessment(entry,1,purpose='transfer')]})
        self.assertEqual(c.quality['objective'],196)
        self.assertEqual(closed['rule_validation']['checks'][0]['state'],'passed')
        self.assertEqual(check_records(self.h.store,entry)[-1]['kind'],'transfer')

    def test_one_unreviewed_observation_stays_candidate_even_after_more_checks(self):
        _,a,c=self.run_task(task('one-observation'),decision=1,checker=False)
        job=c['induction_schedule']['pending_ids'][0]
        self.h.induction_material(induction_id=job)
        entry=self.h.induce(induction_id=job,relations=[{**RULE,'evidence':[a.execution_id]}])['relations'][0]['saved']
        self.assertIsNone(receipt_observation(self.h,get_job(self.h,job))['label'])
        self.run_task(task('C',812,200),entry_id=entry)
        self.run_task(task('boundary',12,0,boundary=True),entry_id=entry)
        self.assertFalse(self.h.sbank.get(entry).is_published)
        self.assertIsNone(receipt_observation(self.h,get_job(self.h,job))['label'])

    def test_renamed_same_input_and_one_sided_parent_are_advisory_not_a_gate(self):
        entry,a,b,job=self.derive(publish=True)
        for tsk in (task('renamed-B'),task('reworded-B',wording='Read the unchanged original data and model.'),
                    {**task('branch',700,140),'parent_task_id':'B'}):
            _,r,c=self.run_task(tsk,entry_id=entry,check_extra={'rule_assessments':[assessment(entry,1,purpose='transfer')]})
            check=check_records(self.h.store,entry)[-1]
            self.assertEqual(check['state'],'passed')
            self.assertTrue(any(row['relationship']=='same_source' for row in check['known_source_relations']))
        self.assertTrue(self.h.sbank.get(entry).is_published)

    def test_no_numeric_text_heuristic_decides_instance_or_transfer(self):
        from or_harness.strategy.evidence_views import declared_task_identity,instance_relation
        a=declared_task_identity({'task_id':'one','description':'Minimize 2 X - 5 Y, X+Y<=641, X>=2Y+130.'})
        b=declared_task_identity({'task_id':'two','description':'For X+Y<=641 and X>=2Y+130 minimize 2 X - 5 Y.'})
        self.assertNotIn('text_duplicate_hint',a)
        self.assertNotEqual(instance_relation(a,b),'possible_duplicate')

    def test_reference_only_review_preserves_scope_and_versions_need_explicit_activation(self):
        _,a,_=self.run_task(task('A',1000,300,maximum=True),decision=1,checker=False,close=False)
        checked=self.h.check_task_result(a.execution_id,{'reference_objective':2699})
        self.assertEqual(checked['state'],'passed')
        closed=self.h.close_episode('A','A');job=closed['induction_schedule']['pending_ids'][0]
        self.h.induction_material(induction_id=job)
        upper={**RULE,'when':'Maximization with positive X coefficient and a computable integer interval.',
                     'do':'Enumerate Y, choose the upper feasible X endpoint and compare objectives.'}
        entry=self.h.induce(induction_id=job,relations=[{**upper,'evidence':[a.execution_id],
            'publication':{'state':'published','reason':'Reference objective agrees; scope is this comparison only.'}}])['relations'][0]['saved']
        _,b,cb=self.run_task(task('B'),decision=1)
        jb=cb['induction_schedule']['pending_ids'][0];self.h.induction_material(induction_id=jb)
        revised=self.h.induce(induction_id=jb,relations=[{**RULE,'entry_id':entry,'operation':'revise','evidence':[b.execution_id],
            'publication':{'state':'candidate','reason':'Review the broader direction-aware method first.'},
            'validation':[{'kind':'regression','execution_id':a.execution_id,'assessment':assessment(entry,2)}]}])
        review=revised['relations'][0]['validation'][0]
        self.assertEqual(review['basis'],'historical_review')
        self.assertFalse(review['actual_version_use'])
        self.assertEqual(review['scope']['acceptance']['basis'],['reference_objective'])
        meta=get_value(self.h.store,version_row(self.h.store,entry,2)['metadata_ref'])
        self.assertEqual(meta['required_regressions'],[])
        self.assertFalse(any(c['state']=='pending' for c in check_records(self.h.store,entry,2)))
        self.assertEqual(self.h.sbank.get(entry).rule['active_version'],1)
        _,trial,_=self.run_task(task('v2-trial',812,200),entry_id=entry,version=2)
        use=[e for e in inspect_versions(self.h,entry)['events'] if e['kind']=='use' and e['execution_id']==trial.execution_id][0]
        self.assertEqual(use['version'],2)
        self.choose(entry,2)
        self.assertEqual(self.h.sbank.get(entry).rule['active_version'],2)
        old=get_value(self.h.store,trial.execution_features['used_entries']['versions'][entry])
        self.assertEqual(old['rule']['view_version'],2)
        weaker=self.h.check_task_result(a.execution_id,{'reference_status':'optimal'})
        self.assertNotEqual(weaker['report']['acceptance_rule']['fingerprint'],checked['report']['acceptance_rule']['fingerprint'])
        self.assertEqual(get_value(self.h.store,weaker['report']['previous_check_ref'])['acceptance_rule'],checked['report']['acceptance_rule'])
        self.h.exclude_execution(a.execution_id,'imported fact was assigned to the wrong task')
        self.assertEqual(self.h.sbank.get(entry).rule['active_version'],2)
        self.assertFalse(any(c['state']=='pending' for c in check_records(self.h.store,entry,2)))

    def test_evidence_only_and_publication_are_light_maintenance_with_no_new_job(self):
        entry,a,b,job=self.derive()
        before=len(self.h.actions.query(action_type='induce'))
        result=self.h.induce(relations=[{'entry_id':entry,'evidence':[b.execution_id]}])
        self.assertTrue(result['maintenance_only'])
        self.assertEqual(result['relations'][0]['version'],1)
        self.choose(entry)
        self.assertTrue(self.h.sbank.get(entry).is_published)
        self.assertEqual(len(self.h.actions.query(action_type='induce')),before)
        self.assertNotIn('induction_receipt',result)
        self.assertTrue(self.h.induction_status()['completed'])

    def test_agent_confirms_merge_without_compulsory_source_regressions(self):
        first,a,b,job=self.derive(publish=True)
        _,r,c=self.run_task(task('source',950,210),decision=1)
        j=c['induction_schedule']['pending_ids'][0];self.h.induction_material(induction_id=j)
        source=self.h.induce(induction_id=j,relations=[{**RULE,'do':RULE['do']+' Record finite endpoints.',
            'evidence':[r.execution_id],'publication':{'state':'published','reason':'Agent checked this equivalent finite-endpoint variant.'}}])['relations'][0]['saved']
        _,r,c=self.run_task(task('merge',960,215),decision=1)
        j=c['induction_schedule']['pending_ids'][0];self.h.induction_material(induction_id=j)
        merged=self.h.induce(induction_id=j,relations=[{'entry_id':first,'operation':'merge','merge_entry_ids':[source],
            'merge_reason':'Same interval mechanism; keep the sign-aware target.',
            'publication':{'state':'published','reason':'Agent compared the mechanisms and confirms this target.'}}])
        self.assertEqual(self.h.sbank.get(source).rule['merged_into'],first)
        self.assertEqual(merged['relations'][0]['publication']['required_checks'],[])


class TestAgentUtilityAndCorrections(AgentCase):
    def test_H0_management_promote_demote_keep_and_retire_share_version_lifecycle(self):
        entry,a,b,job=self.derive(publish=True,review=True)
        _,r,closed=self.run_task(task('use',812,200),entry_id=entry)
        self.assertEqual(closed['induction_schedule']['jobs'][0]['state'],'skipped')
        n=len(self.h.actions.query(action_type='induce'))
        self.manage(entry,r,'demote')
        self.assertFalse(self.h.sbank.get(entry).is_published)
        self.manage(entry,r,'keep',verdict='neutral')
        self.assertEqual(self.h.sbank.get(entry).rule['state'],'paused')
        result=self.cli(['induce','--attribute-effect',json.dumps({'entry_id':entry,'execution_id':r.execution_id,
            'verdict':'helped','lifecycle_action':'promote','note':'Agent confirmed useful actual version-one use.'})])
        self.assertTrue(result['maintenance_only'])
        self.assertTrue(self.h.sbank.get(entry).is_published)
        self.assertEqual(len(self.h.actions.query(action_type='induce')),n)
        effects=[e for e in inspect_versions(self.h,entry)['events'] if e['kind']=='effect']
        self.assertEqual([e['version'] for e in effects],[1,1,1])
        self.h.retire(entry,'Agent retires this method after reviewing its accumulated use.')
        self.assertFalse(any(r.get('evidence')=='strategic_entry' for r in self.h.recall(task('after-retire'))['recommendations']))
        self.assertEqual(receipt_observation(self.h,get_job(self.h,job))['label'],1)

    def test_management_requires_actual_adoption_and_resolves_exact_version(self):
        entry,a,b,j=self.derive(publish=True)
        with self.assertRaisesRegex(ValueError,'adoption'):
            self.manage(entry,b,'demote')
        _,r,_=self.run_task(task('used',812,200),entry_id=entry)
        with self.assertRaisesRegex(ValueError,'version'):
            self.manage(entry,r,'demote',version=2)
        result=self.manage(entry,r,'keep')
        self.assertEqual(result['effect_attributions'][0]['version'],1)

    def test_management_of_old_use_does_not_demote_a_later_active_version(self):
        entry,a,b,j=self.derive(publish=True)
        _,old,_=self.run_task(task('v1-used',812,200),entry_id=entry)
        _,r,c=self.run_task(task('revision',930,210),decision=1);job=c['induction_schedule']['pending_ids'][0]
        self.h.induction_material(induction_id=job)
        self.h.induce(induction_id=job,relations=[{**RULE,'do':RULE['do']+' Record finite interval endpoints.',
            'entry_id':entry,'operation':'revise','evidence':[r.execution_id],
            'publication':{'state':'published','reason':'Agent confirms the added finite-endpoint check.'}}])
        result=self.manage(entry,old,'demote',verdict='refuting')
        self.assertEqual(result['effect_attributions'][0]['version'],1)
        self.assertEqual(self.h.sbank.get(entry).rule['active_version'],2)
        self.assertTrue(self.h.sbank.get(entry).is_published)
        self.assertEqual(version_row(self.h.store,entry,1)['state'],'suspended')

    def test_no_automatic_cost_baseline_and_explicit_contrast_reports_both_dimensions(self):
        entry,a,b,j=self.derive(rule={**RULE,'effect_scope':'cost','expect':'Reduce tool_calls while preserving original acceptance.'},publish=True)
        _,base,_=self.run_task(task('cost',812,200),exhaustive=True,measured={'tool_calls':9,'llm_tokens':100})
        _,r,c=self.run_task(task('cost',812,200),entry_id=entry,episode='new',measured={'tool_calls':10,'llm_tokens':25},
            check_extra={'rule_assessments':[assessment(entry,1,verdict='unknown')]})
        checks=check_records(self.h.store,entry)
        self.assertNotIn('cost_comparison',checks[-1])
        self.recheck(r,rule_assessments=[assessment(entry,1,verdict='unknown',comparator_execution_id=base.execution_id,
                                                 cost_dimensions=['tool_calls','llm_tokens'])])
        comparison=check_records(self.h.store,entry)[-1]['cost_comparison']
        self.assertEqual(comparison['dimensions']['tool_calls']['change'],1)
        self.assertEqual(comparison['dimensions']['llm_tokens']['change'],-75)
        self.assertIsNone(comparison['verdict'])
        self.assertEqual(comparison['actual']['method_actual'],r.method_actual)
        self.assertTrue(self.h.sbank.get(entry).is_published)

    def test_estimates_are_visible_but_never_converted_to_measured_cost_evidence(self):
        entry,a,b,j=self.derive(publish=True)
        _,base,_=self.run_task(task('estimated',812,200),exhaustive=True,measured={'llm_tokens':100},provenance='agent_estimate')
        _,r,_=self.run_task(task('estimated',812,200),entry_id=entry,episode='new',measured={'llm_tokens':1},provenance='agent_estimate',
            check_extra={'rule_assessments':[assessment(entry,1,verdict='unknown',comparator_execution_id=base.execution_id,cost_dimensions=['llm_tokens'])]})
        v=check_records(self.h.store,entry)[-1]['cost_comparison']['dimensions']['llm_tokens']
        self.assertEqual(v['actual_eligibility'],'estimate')
        self.assertEqual(v['baseline_eligibility'],'estimate')
        self.assertIsNone(v['change'])
        self.assertEqual(check_records(self.h.store,entry)[-1]['agent_verdict'],'unknown')

    def test_check_and_cost_corrections_do_not_automatically_manage_publication(self):
        entry,a,b,j=self.derive(publish=True,review=True)
        _,r,_=self.run_task(task('use',812,200),entry_id=entry,measured={'tool_calls':3})
        self.recheck(r,rule_assessments=[])
        self.assertTrue(self.h.sbank.get(entry).is_published)
        self.cli(['amend-cost',r.execution_id,'--override','tool_calls=100','--source','agent_observed','--force'])
        self.assertTrue(self.h.sbank.get(entry).is_published)
        self.h.exclude_execution(r.execution_id,'imported onto the wrong task')
        self.assertTrue(self.h.sbank.get(entry).is_published)
        self.h.restore_execution(r.execution_id,'restored correct source')
        self.assertTrue(self.h.sbank.get(entry).is_published)
        self.manage(entry,r,'demote',verdict='refuting')
        self.assertFalse(self.h.sbank.get(entry).is_published)

    def test_ordinary_recheck_omission_preserves_assessment_and_no_full_row_copy(self):
        _,plain,_=self.run_task(task('unreferenced'))
        self.recheck(plain)
        rows=self.h.store.conn.execute("SELECT payload FROM condition_values WHERE kind IN ('rule_execution','induction_execution')").fetchall()
        self.assertEqual(rows,[])
        entry,a,b,j=self.derive(publish=True)
        _,r,_=self.run_task(task('use',812,200),entry_id=entry)
        previous=copy.deepcopy(r.execution_features['task_check']['rule_assessments'])
        checked=self.recheck(r)
        self.assertEqual(checked['report']['rule_assessments'],previous)
        self.assertTrue(self.h.sbank.get(entry).is_published)


class TestLiveMaterialsAndStableJobs(AgentCase):
    def test_positive_anchor_live_queries_and_pagination_include_newer_zero_attempt(self):
        _,a,ca=self.run_task(task('same'),decision=1,episode='A')
        j=ca['induction_schedule']['pending_ids'][0]
        first=self.h.induction_material(induction_id=j,limit=1,related_top_k=0)
        _,b,_=self.run_task(task('same'),episode='B')
        _,c,_=self.run_task(task('same'),episode='C')
        live=self.h.induction_material(induction_id=j,limit=1,related_top_k=0)
        self.assertNotEqual(first['material_id'],live['material_id'])
        self.assertEqual(live['material'][0]['execution_id'],a.execution_id)
        seen={r['execution_id'] for r in live['material']}
        cursor=live['budget']['next_cursor']
        while cursor:
            page=self.h.induction_material(induction_id=j,limit=1,cursor=cursor,related_top_k=0)
            self.assertEqual(page['material'][0]['execution_id'],a.execution_id)
            seen.update(r['execution_id'] for r in page['material'])
            cursor=page['budget']['next_cursor']
        self.assertTrue({a.execution_id,b.execution_id,c.execution_id}<=seen)
        changed=self.h.induction_material(induction_id=j,purpose='implementation',query='actual repair',execution_ids=[b.execution_id])
        self.assertEqual(changed['purpose'],'implementation')
        self.assertNotIn('material_requests',get_job(self.h,j))
        self.assertEqual(self.h.store.conn.execute("SELECT COUNT(*) FROM condition_values WHERE kind='hplus_material'").fetchone()[0],0)

    def test_pending_selected_evidence_survives_window_restart_then_releases_full_holds(self):
        _,a,c=self.run_task(task('trigger'),decision=1)
        _,b,_=self.run_task(task('selected',700,140))
        job=c['induction_schedule']['pending_ids'][0]
        self.h.induction_material(induction_id=job,execution_ids=[b.execution_id],related_top_k=0)
        self.run_task(task('later',812,200));self.evict()
        self.assertIsNone(self.h.bank.get(a.execution_id));self.assertIsNone(self.h.bank.get(b.execution_id))
        self.h.close();self.h=ORHarness(home=self.home,embedding=LocalHashEmbeddingBackend());self.addCleanup(self.h.close)
        m=self.h.induction_material(induction_id=job,execution_ids=[b.execution_id])
        self.assertEqual(m['material'][0]['execution_id'],a.execution_id)
        self.assertTrue(held_records(self.h,job))
        result=self.h.induce(induction_id=job,relations=[{**RULE,'evidence':[a.execution_id,b.execution_id]}])
        self.assertTrue(result['saved'])
        self.assertEqual(held_records(self.h,job),{})
        self.assertEqual(self.h.store.conn.execute("SELECT COUNT(*) FROM condition_values WHERE kind='induction_execution'").fetchone()[0],0)
        self.assertTrue(resolve_execution(self.h,a.execution_id).execution_features['audit_fact_only'])
        self.assertEqual(resolve_execution(self.h,a.execution_id).execution_features['retained_task_payload']['data'],task('trigger')['data'])
        self.assertIsNone(self.h.bank.get(a.execution_id))

    def test_shared_selected_evidence_stays_held_until_both_pending_reviews_finish(self):
        _,shared,_=self.run_task(task('shared',700,140))
        _,a,c=self.run_task(task('first'),decision=1);j1=c['induction_schedule']['pending_ids'][0]
        _,b,c=self.run_task(task('second',720,150),decision=1);j2=c['induction_schedule']['pending_ids'][0]
        for job in (j1,j2):self.h.induction_material(induction_id=job,execution_ids=[shared.execution_id])
        self.empty(j1)
        self.assertIn(shared.execution_id,held_records(self.h,j2))
        self.run_task(task('last',812,200));self.evict()
        self.assertIsNone(self.h.bank.get(shared.execution_id))
        material=self.h.induction_material(induction_id=j2,execution_ids=[shared.execution_id])
        shown=material['material']+material['related_history']['executions']
        self.assertIn(shared.execution_id,[r['execution_id'] for r in shown])
        self.empty(j2)
        self.assertEqual(self.h.store.conn.execute("SELECT COUNT(*) FROM condition_values WHERE kind='induction_execution'").fetchone()[0],0)

    def test_duplicate_close_omission_uses_one_job_and_two_real_rounds_are_distinct(self):
        _,a,c=self.run_task(task('one'),decision=1)
        retry=self.h.close_episode('one')
        self.assertEqual(retry['induction_schedule']['pending_ids'],c['induction_schedule']['pending_ids'])
        self.assertEqual(len(jobs(self.h,task_id='one')),1)
        _,b,other=self.run_task(task('one'),decision=1,episode='other')
        self.assertNotEqual(c['induction_schedule']['pending_ids'],other['induction_schedule']['pending_ids'])
        with self.assertRaisesRegex(ValueError,'ambiguous'):self.h.close_episode('one')

    def test_old_shared_evidence_submission_retries_original_receipt_not_new_job(self):
        _,b,_=self.run_task(task('shared',700,140))
        _,a,c=self.run_task(task('round-one'),decision=1);j1=c['induction_schedule']['pending_ids'][0]
        self.h.induction_material(induction_id=j1,execution_ids=[b.execution_id])
        arguments={'relations':[{**RULE,'evidence':[b.execution_id]}]}
        first=self.h.induce(**arguments)
        _,a,c=self.run_task(task('round-two',720,150),decision=1);j2=c['induction_schedule']['pending_ids'][0]
        self.h.induction_material(induction_id=j2,execution_ids=[b.execution_id])
        retry=self.h.induce(**arguments)
        self.assertTrue(retry['already_completed'])
        self.assertEqual(retry['induction_receipt']['induction_id'],j1)
        self.assertEqual(get_job(self.h,j2)['state'],'pending')
        self.empty(j2)

    def test_two_connections_late_material_never_resets_completed_review(self):
        _,a,c=self.run_task(task('interleaved'),decision=1);j=c['induction_schedule']['pending_ids'][0]
        self.h.induction_material(induction_id=j)
        other=ORHarness(home=self.home,embedding=LocalHashEmbeddingBackend());self.addCleanup(other.close)
        from or_harness.world_model.induction_material import build_compact_material
        def interleave(*args,**kwargs):
            material=build_compact_material(*args,**kwargs)
            other.induce(induction_id=j,notes=['Agent reviewed the available facts.'])
            return material
        with patch('or_harness.world_model.induction_material.build_compact_material',side_effect=interleave):
            self.h.induction_material(induction_id=j,purpose='implementation')
        self.assertEqual(get_job(self.h,j)['state'],'completed')
        self.assertEqual(len(self.h.actions.query(action_type='induce')),1)
        self.assertEqual(held_records(self.h,j),{})
        retry=self.h.induce(induction_id=j,notes=['Agent reviewed the available facts.'])
        self.assertTrue(retry['already_completed'])

    def test_withdrawn_pending_source_can_finish_empty_review_without_becoming_rule_evidence(self):
        _,a,c=self.run_task(task('withdraw'),decision=1);j=c['induction_schedule']['pending_ids'][0]
        self.h.induction_material(induction_id=j)
        self.h.exclude_execution(a.execution_id,'wrong imported source')
        with self.assertRaisesRegex(ValueError,'unknown execution'):
            self.h.induce(induction_id=j,relations=[{**RULE,'evidence':[a.execution_id]}])
        self.empty(j)
        self.assertIsNone(receipt_observation(self.h,get_job(self.h,j))['label'])


class TestAgentLearningFeedback(AgentCase):
    def test_supported_increment_is_independent_of_publication_and_retirement(self):
        entry,a,b,j=self.derive(review=True)
        self.assertFalse(self.h.sbank.get(entry).is_published)
        self.assertEqual(receipt_observation(self.h,get_job(self.h,j))['label'],1)
        self.h.retire(entry,'Agent retires the method for later utility reasons.')
        self.assertEqual(receipt_observation(self.h,get_job(self.h,j))['label'],1)

    def test_empty_recall_and_synonyms_never_supply_a_positive_by_themselves(self):
        entry,a,b,j=self.derive(publish=True)
        self.assertIsNone(receipt_observation(self.h,get_job(self.h,j))['label'])
        _,r,c=self.run_task(task('synonym',930,210),decision=1);job=c['induction_schedule']['pending_ids'][0]
        self.h.induction_material(induction_id=job)
        self.h.induce(induction_id=job,relations=[{**RULE,'do':RULE['do'].replace('Enumerate','Iterate over'),
            'entry_id':entry,'operation':'revise','evidence':[r.execution_id],
            'publication':{'state':'published','reason':'Agent retains the same checked mechanism with clearer wording.'}}])
        self.assertIsNone(receipt_observation(self.h,get_job(self.h,job))['label'])

    def test_safe_formatting_dedup_and_evidence_only_supported_negative(self):
        entry,a,b,j=self.derive(publish=True)
        _,r,c=self.run_task(task('covered',930,210),decision=1);job=c['induction_schedule']['pending_ids'][0]
        self.h.induction_material(induction_id=job)
        out=self.h.induce(induction_id=job,relations=[{**{k:v.replace(' ','  ') for k,v in RULE.items()},
            'entry_id':entry,'operation':'revise','evidence':[r.execution_id]}],
            review={'outcome':'already_covered','comparison':self.comparison(job,substantive=False)})
        self.assertEqual(out['relations'][0]['version'],1)
        self.assertEqual(receipt_observation(self.h,get_job(self.h,job))['label'],0)
        _,r,c=self.run_task(task('more-evidence',940,220),decision=1);job=c['induction_schedule']['pending_ids'][0]
        self.h.induction_material(induction_id=job)
        self.h.induce(induction_id=job,relations=[{'entry_id':entry,'evidence':[r.execution_id]}],
            review={'outcome':'already_covered','comparison':self.comparison(job,substantive=False)})
        self.assertEqual(receipt_observation(self.h,get_job(self.h,job))['label'],0)

    def test_exclusion_invalidates_only_future_learning_support_not_publication_or_old_input(self):
        entry,a,b,j=self.derive(publish=True,review=True)
        before=planned_prediction(self.h,task('before'),{'strategy_id':'endpoint','solver':'python'})
        context=self.h.get_prediction_context(before.trace.model_info['prediction_context_id'])
        frozen=copy.deepcopy(context.hplus_feedback);cutoff=time.time()
        self.h.exclude_execution(a.execution_id,'import assigned to wrong original task')
        self.assertIsNone(receipt_observation(self.h,get_job(self.h,j))['label'])
        self.assertTrue(self.h.sbank.get(entry).is_published)
        self.assertFalse(self.h.sbank.get(entry).verification['scope']['evidence_validity'][a.execution_id]['valid'])
        self.assertEqual(receipt_observation(self.h,get_job(self.h,j),as_of=cutoff)['label'],1)
        self.assertEqual(context.provider_view()['hplus_feedback'],frozen)
        self.h.restore_execution(a.execution_id,'verified the original source')
        self.assertEqual(receipt_observation(self.h,get_job(self.h,j))['label'],1)
        self.assertTrue(self.h.sbank.get(entry).is_published)

    def test_late_checker_enters_only_later_negative_observations(self):
        entry,a,b,j=self.derive(publish=True)
        _,r,c=self.run_task(task('late',930,210),decision=1,checker=False);job=c['induction_schedule']['pending_ids'][0]
        self.h.induction_material(induction_id=job)
        self.h.induce(induction_id=job,notes=['Agent inspected the covered workflow; result check is pending.'],
            review={'outcome':'already_covered','comparison':self.comparison(job,substantive=False)})
        cutoff=time.time();self.assertIsNone(receipt_observation(self.h,get_job(self.h,job))['label'])
        self.h.check_task_result(r.execution_id,{'reference_objective':180})
        self.assertEqual(receipt_observation(self.h,get_job(self.h,job))['label'],0)
        self.assertIsNone(receipt_observation(self.h,get_job(self.h,job),as_of=cutoff)['label'])

    def test_invalid_learning_reference_is_rejected_before_a_receipt_or_rule_write(self):
        _,a,c=self.run_task(task('bad-reference'),decision=1);j=c['induction_schedule']['pending_ids'][0]
        self.h.induction_material(induction_id=j)
        with self.assertRaisesRegex(ValueError,'learning evidence'):
            self.h.induce(induction_id=j,relations=[{**RULE,'evidence':[a.execution_id]}],
                review={'outcome':'new_rule','comparison':{'sufficient':True,'substantive':True,'source':'agent',
                    'assessment':'Reviewed actual code','evidence_refs':['rule_fact:missing']}})
        self.assertEqual(self.h.sbank.count(),0)
        self.assertEqual(get_job(self.h,j)['state'],'pending')

    def test_long_review_is_compact_and_short_conclusion_reaches_actual_next_provider(self):
        entry,a,b,j=self.derive(review=True)
        _,r,c=self.run_task(task('long-review',930,210),decision=1);jl=c['induction_schedule']['pending_ids'][0]
        self.h.induction_material(induction_id=jl)
        self.h.induce(induction_id=jl,notes=['Reviewed the mechanism.'],review={'outcome':'no_increment',
            'conclusion':'Long review '+('same checked fact; '*2000),
            'comparison':self.comparison(jl,substantive=False,assessment='Full comparison '+('recorded premise; '*3000))})
        _,r,c=self.run_task(task('short-review',940,220),decision=1);js=c['induction_schedule']['pending_ids'][0]
        self.h.induction_material(induction_id=js)
        self.h.induce(induction_id=js,notes=['Reviewed existing coverage.'],review={'outcome':'already_covered',
            'conclusion':'The current endpoint mechanism already covers this case.',
            'comparison':self.comparison(js,substantive=False)})
        self.run_task(task('next-provider',950,230))
        sent=self.provider.requests[-1]['prediction_context']['hplus_feedback']
        self.assertIn(get_job(self.h,js)['prediction_id'],[row['prediction_id'] for row in sent['settled_examples']])
        self.assertNotIn('Full comparison',json.dumps(sent))
        self.assertNotIn('comparison',json.dumps(sent.get('settled_examples',[])))

    def test_oversized_first_feedback_does_not_hide_a_later_fitting_receipt(self):
        _,a,c=self.run_task(task('short'),decision=1);js=c['induction_schedule']['pending_ids'][0]
        self.h.induction_material(induction_id=js)
        self.h.induce(induction_id=js,notes=['Reviewed.'],review={'outcome':'no_increment','comparison':self.comparison(js,substantive=False)})
        _,a,c=self.run_task(task('large'),decision=1);jl=c['induction_schedule']['pending_ids'][0]
        self.h.induction_material(induction_id=jl)
        with self.h.store.transaction() as conn:
            extra=[put_value(conn,self.h.store,'analysis_reference',{'case':i}) for i in range(50)]
        comp=self.comparison(jl,substantive=False);comp['evidence_refs']+=extra
        self.h.induce(induction_id=jl,notes=['Reviewed.'],review={'outcome':'no_increment','comparison':comp})
        view=feedback(self.h,task=task('next'),budget_chars=2400)
        self.assertIn(get_job(self.h,js)['prediction_id'],[r['prediction_id'] for r in view['settled_examples']])


class TestReviewNineRetentionAndSupport(AgentCase):
    """Receipt-only evidence, concrete references and upgraded checker versions."""

    def restart(self):
        self.h.close()
        self.h=ORHarness(home=self.home,world_model=self.provider,embedding=LocalHashEmbeddingBackend())
        self.addCleanup(self.h.close)

    def test_P1_1_receipt_only_historical_execution_survives_eviction_and_restart(self):
        b=self.historical_execution()
        review_only=self.historical_execution('review-only')
        self.assertEqual(self.h.strategy_predictions.query(task_id=b.task_id),[])
        self.assertEqual(jobs(self.h,task_id=b.task_id),[])
        _,a,c=self.run_task(task('trigger'),decision=1);j=c['induction_schedule']['pending_ids'][0]
        m=self.h.induction_material(induction_id=j,related_top_k=0)
        self.assertNotIn(b.execution_id,get_value(self.h.store,m['material_id'])['shown_execution_ids'])
        comp=self.comparison(j,substantive=True,evidence_refs=[b.execution_id])
        self.h.induce(induction_id=j,relations=[{**RULE,'evidence':[a.execution_id],'mechanism_comparison':comp}],
                      review={'outcome':'new_rule','comparison':{**comp,'evidence_refs':[review_only.execution_id]}})
        self.assertEqual(receipt_observation(self.h,get_job(self.h,j))['label'],1)
        self.run_task(task('last',812,200));self.evict();self.restart()
        self.assertIsNone(self.h.bank.get(b.execution_id))
        retained=resolve_execution(self.h,b.execution_id)
        self.assertIsNotNone(retained)
        self.assertIsNotNone(resolve_execution(self.h,review_only.execution_id))
        self.assertEqual(retained.execution_features['task_check']['state'],'passed')
        self.assertEqual(receipt_observation(self.h,get_job(self.h,j))['label'],1)
        self.assertGreaterEqual(self.h.induction_status()['observation_counts']['positive'],1)
        self.run_task(task('next'))
        rows=self.provider.requests[-1]['prediction_context']['hplus_feedback']['settled_examples']
        self.assertIn(get_job(self.h,j)['prediction_id'],[r['prediction_id'] for r in rows])
        self.h.exclude_execution(b.execution_id,'reviewer withdrew the imported reference')
        self.assertIsNone(receipt_observation(self.h,get_job(self.h,j))['label'])

    def test_P1_1_unrecoverable_old_receipt_is_unknown_without_breaking_other_feedback(self):
        _,a,c=self.run_task(task('old'),decision=1);j=c['induction_schedule']['pending_ids'][0]
        result=self.empty(j,outcome='no_increment',comparison=self.comparison(j,substantive=False))
        old=copy.deepcopy(result['induction_receipt']);old.pop('receipt_ref')
        old['comparison']['evidence_refs']=['execution:lost-old-execution']
        with self.h.store.transaction() as conn:
            oldref=put_value(conn,self.h.store,'hplus_receipt',old)
            from or_harness.world_model.induction_gate import meta_write
            job=get_job(self.h,j);job['receipt_ref']=oldref;meta_write(self.h.store,'hplus_job:'+j,job)
        _,b,c=self.run_task(task('valid'),decision=1);valid=c['induction_schedule']['pending_ids'][0]
        self.empty(valid,outcome='no_increment',comparison=self.comparison(valid,substantive=False))
        observed=receipt_observation(self.h,get_job(self.h,j))
        self.assertIsNone(observed['label']);self.assertIn('unavailable',str(observed))
        self.assertTrue(self.cli(['induction-status'])['completed'])
        self.run_task(task('after-old'))
        rows=self.provider.requests[-1]['prediction_context']['hplus_feedback']['settled_examples']
        self.assertIn(get_job(self.h,valid)['prediction_id'],[r['prediction_id'] for r in rows])

    def test_P1_3_missing_artifact_or_check_item_cannot_borrow_other_passes(self):
        _,a,c=self.run_task(task('missing-path'),decision=1);j=c['induction_schedule']['pending_ids'][0]
        self.h.induction_material(induction_id=j)
        for ref in ('artifacts.not_captured','task_check.checks.99','task_check.no_such_check'):
            with self.subTest(ref=ref),self.assertRaisesRegex(ValueError,'learning evidence'):
                self.h.induce(induction_id=j,relations=[{**RULE,'evidence':[a.execution_id]}],
                    review={'outcome':'new_rule','comparison':self.comparison(j,substantive=True,evidence_refs=[ref])})
            self.assertEqual(self.h.sbank.count(),0)
            self.assertEqual(get_job(self.h,j)['state'],'pending')

    def test_P1_3_checker_correction_withdraws_support_but_preserves_old_input(self):
        _,a,c=self.run_task(task('checker-correction'),decision=1);j=c['induction_schedule']['pending_ids'][0]
        self.h.induction_material(induction_id=j)
        self.h.induce(induction_id=j,relations=[{**RULE,'evidence':[a.execution_id]}],
            review={'outcome':'new_rule','comparison':self.comparison(j,substantive=True,evidence_refs=['task_check.checks'])})
        prediction=planned_prediction(self.h,task('before-correction'),{'strategy_id':'test','solver':'python'})
        context=self.h.get_prediction_context(prediction.trace.model_info['prediction_context_id'])
        frozen=copy.deepcopy(context.hplus_feedback);cutoff=time.time()
        self.recheck(a,reference_objective=-999)
        observed=receipt_observation(self.h,get_job(self.h,j))
        self.assertIsNone(observed['label'])
        self.assertIn('support',str(observed))
        self.assertEqual(receipt_observation(self.h,get_job(self.h,j),as_of=cutoff)['label'],1)
        self.assertEqual(context.provider_view()['hplus_feedback'],frozen)
        self.assertIn(1,[r['label'] for r in frozen['settled_examples']])

    def test_P1_3_removed_scope_invalidates_support_identical_recheck_does_not(self):
        _,a,c=self.run_task(task('scope-correction'),decision=1);j=c['induction_schedule']['pending_ids'][0]
        self.empty(j,outcome='no_increment',comparison=self.comparison(j,substantive=False))
        self.recheck(a)
        self.assertEqual(receipt_observation(self.h,get_job(self.h,j))['label'],0)
        self.h.check_task_result(a.execution_id,{'reference_status':'optimal'})
        self.assertIsNone(receipt_observation(self.h,get_job(self.h,j))['label'])

    def test_P1_3_real_failure_can_support_an_agent_boundary_conclusion(self):
        _,a,c=self.run_task(task('boundary-fact'),decision=1,check_extra={'reference_objective':-999})
        j=c['induction_schedule']['pending_ids'][0]
        self.assertEqual(a.execution_features['task_check']['state'],'failed')
        self.h.induction_material(induction_id=j)
        self.h.induce(induction_id=j,relations=[{**RULE,'evidence':[a.execution_id]}],review={'outcome':'new_rule',
            'hypothesis_outcome':'refuted','comparison':self.comparison(j,substantive=True,
                assessment='The failed declared reference demonstrates the stated boundary.',evidence_refs=['task_check.checks'])})
        self.assertEqual(receipt_observation(self.h,get_job(self.h,j))['label'],1)

    def test_P1_3_late_check_then_correction_and_explicit_stale_checker_reference(self):
        _,a,c=self.run_task(task('late-correction'),decision=1,checker=False);j=c['induction_schedule']['pending_ids'][0]
        self.empty(j,outcome='no_increment',comparison=self.comparison(j,substantive=False))
        self.h.check_task_result(a.execution_id,{'reference_objective':90})
        cutoff=time.time()
        self.assertEqual(receipt_observation(self.h,get_job(self.h,j))['label'],0)
        bad=self.h.check_task_result(a.execution_id,{'reference_objective':-999})
        self.assertIsNone(receipt_observation(self.h,get_job(self.h,j))['label'])
        self.assertEqual(receipt_observation(self.h,get_job(self.h,j),as_of=cutoff)['label'],0)
        _,b,c=self.run_task(task('stale-reference'),decision=1);new=c['induction_schedule']['pending_ids'][0]
        self.empty(new,outcome='no_increment',comparison=self.comparison(new,substantive=False,
            evidence_refs=[bad['report']['previous_check_ref']]))
        self.assertIsNone(receipt_observation(self.h,get_job(self.h,new))['label'])

    def test_P1_3_named_check_item_depends_only_on_that_executed_item(self):
        _,a,c=self.run_task(task('one-item'),decision=1);j=c['induction_schedule']['pending_ids'][0]
        self.empty(j,outcome='no_increment',comparison=self.comparison(j,substantive=False,
            evidence_refs=['task_check.checks.0']))
        self.assertEqual(receipt_observation(self.h,get_job(self.h,j))['label'],0)
        self.recheck(a,semantic_probe=[{'path':'execution_features.solution_variables.X','equals':-999}])
        # The referenced objective check still passes; a different probe fails.
        self.assertEqual(receipt_observation(self.h,get_job(self.h,j))['label'],0)

    def test_P2_3_legacy_full_copy_never_shadows_a_later_checker_after_eviction(self):
        _,a,c=self.run_task(task('upgrade'),decision=1,checker=False);j=c['induction_schedule']['pending_ids'][0]
        from or_harness.world_model.induction_gate import meta_write
        with self.h.store.transaction() as conn:
            old=put_value(conn,self.h.store,'rule_evidence',{'record':a.to_dict()})
            meta_write(self.h.store,'rule_evidence_latest:'+a.execution_id,old)
        before=time.time()
        self.h.check_task_result(a.execution_id,{'reference_objective':90})
        self.empty(j,outcome='already_covered',comparison=self.comparison(j,substantive=False))
        self.assertEqual(receipt_observation(self.h,get_job(self.h,j))['label'],0)
        self.run_task(task('latest',812,200));self.evict();self.restart()
        record=resolve_execution(self.h,a.execution_id)
        self.assertEqual(record.execution_features['task_check']['state'],'passed')
        self.assertTrue(record.execution_features['task_check']['action_id'])
        self.assertEqual(receipt_observation(self.h,get_job(self.h,j))['label'],0)
        from or_harness.world_model.hplus_support import _report_at
        self.assertEqual(_report_at(self.h,record,before),{})
        self.assertIsNotNone(get_value(self.h.store,old)['record'])


class TestReviewNineTransactionsAndRouting(AgentCase):
    def revision(self,entry,*,publish=True,review=None):
        _,r,c=self.run_task(task('revision',930,210),decision=1);j=c['induction_schedule']['pending_ids'][0]
        self.h.induction_material(induction_id=j)
        args={'induction_id':j,'relations':[{**RULE,'do':RULE['do']+' Record finite interval endpoints.',
              'entry_id':entry,'operation':'revise','evidence':[r.execution_id],
              'publication':{'state':'published' if publish else 'candidate','reason':'Agent reviewed the revision.'}}],
              'review':review}
        return args,j

    def test_P1_2_missing_optional_cost_baseline_does_not_block_close_or_Hplus(self):
        entry,a,b,j=self.derive(publish=True)
        for decision in (1,0):
            with self.subTest(decision=decision):
                base=self.historical_execution('base-'+str(decision))
                # Ensure the baseline exists when the comparison is declared.
                _,r,_=self.run_task(task('current-'+str(decision)),decision=decision,entry_id=entry,close=False,
                    check_extra={'rule_assessments':[assessment(entry,1,comparator_execution_id=base.execution_id,
                                                               cost_dimensions=['tool_calls'])]})
                self.run_task(task('protect-latest-'+str(decision),812,200));self.evict()
                self.assertIsNone(self.h.bank.get(base.execution_id))
                closed=self.h.close_episode(r.task_id,r.task_id)
                checked=check_records(self.h.store,entry)[-1]
                self.assertEqual(closed['rule_validation']['checks'][0]['state'],'passed')
                missing=checked['cost_comparison']
                self.assertEqual(missing['state'],'unavailable')
                self.assertEqual(missing['execution_id'],base.execution_id)
                self.assertTrue(missing['reason'])
                self.assertEqual(self.h.close_episode(r.task_id,r.task_id)['induction_schedule']['pending_ids'],
                                 closed['induction_schedule']['pending_ids'])
                pending=closed['induction_schedule']['pending_ids']
                self.assertEqual(len(pending),decision)
                if decision:self.empty(pending[0])

    def test_P1_4_candidate_merge_waits_for_its_actual_version_then_cannot_resurrect_source(self):
        entry,a,b,j=self.derive(publish=True)
        _,r,c=self.run_task(task('source',950,210),decision=1);js=c['induction_schedule']['pending_ids'][0]
        self.h.induction_material(induction_id=js)
        source=self.h.induce(induction_id=js,relations=[{**RULE,'do':RULE['do']+' Source variant.',
            'evidence':[r.execution_id],'publication':{'state':'published','reason':'Agent reviewed source.'}}])['relations'][0]['saved']
        _,used,_=self.run_task(task('source-use',812,200),entry_id=source)
        args,j=self.revision(entry,publish=False)
        args['relations'][0].update(operation='merge',merge_entry_ids=[source])
        self.h.induce(**args)
        self.assertEqual(self.h.sbank.get(entry).rule['active_version'],1)
        self.assertTrue(self.h.sbank.get(source).is_published)
        self.choose(entry,2)
        self.assertEqual(self.h.sbank.get(source).rule['merged_into'],entry)
        self.assertFalse(self.h.sbank.get(source).is_published)
        self.assertNotIn(source,[r['id'] for r in self.h.embedding_index.items('strategic_knowledge')])
        self.assertEqual(self.h.index_health()['layers']['strategic_knowledge']['missing'],0)
        n=len(inspect_versions(self.h,source)['events'])
        with self.assertRaisesRegex(ValueError,'merge.*target|merged into'):
            self.manage(source,used,'promote')
        self.assertEqual(len(inspect_versions(self.h,source)['events']),n)
        self.assertFalse(self.h.sbank.get(source).is_published)

    def test_P1_4_primary_adoption_cannot_be_overridden_by_auxiliary_version(self):
        entry,a,b,j=self.derive(publish=True)
        _,v1,_=self.run_task(task('v1-use',812,200),entry_id=entry)
        args,j=self.revision(entry);self.h.induce(**args)
        _,v2,_=self.run_task(task('v2-use',813,201),entry_id=entry,version=2)
        with self.assertRaisesRegex(ValueError,'primary|main|execution.*version'):
            self.manage(entry,v1,'demote',version=2,evidence_execution_ids=[v2.execution_id])
        self.assertTrue(self.h.sbank.get(entry).is_published)
        self.assertEqual(version_row(self.h.store,entry,2)['state'],'active')
        self.assertFalse(any(e['kind']=='effect' for e in inspect_versions(self.h,entry)['events']))

    def test_P1_5_frozen_light_maintenance_rejects_before_any_write_API_and_CLI(self):
        entry,a,b,j=self.derive(publish=True)
        _,used,_=self.run_task(task('used',812,200),entry_id=entry)
        args,j=self.revision(entry,publish=False);self.h.induce(**args)
        from or_harness.world_model.hplus import configure
        configure(self.h,'frozen')
        before=inspect_versions(self.h,entry);vectors=copy.deepcopy(self.h.embedding_index.items('strategic_knowledge'))
        with self.assertRaisesRegex(ValueError,'frozen'):
            self.choose(entry,2)
        out=io.StringIO()
        payload={'entry_id':entry,'execution_id':used.execution_id,'lifecycle_action':'demote',
                 'verdict':'refuting','note':'A real adopted use, but frozen libraries cannot be managed.'}
        with patch.dict('os.environ',{'OR_EMBEDDING_BACKEND':'local-hashing'}),redirect_stdout(out):
            code=main(['--home',self.home,'induce','--attribute-effect',json.dumps(payload)])
        self.assertEqual(code,2,out.getvalue())
        self.assertIn('frozen',out.getvalue())
        self.assertEqual(inspect_versions(self.h,entry),before)
        self.assertEqual(self.h.embedding_index.items('strategic_knowledge'),vectors)

    def test_P1_2_excluded_or_missing_optional_baselines_and_explicit_validation_errors(self):
        entry,a,b,j=self.derive(publish=True)
        base=self.historical_execution('excluded-baseline')
        self.h.exclude_execution(base.execution_id,'wrong import source')
        from or_harness.strategy.rule_versions import validate_case
        from or_harness.world_model.induction_gate import meta_write
        with self.h.store.transaction():
            meta_write(self.h.store,'rule_evidence_latest:unreadable-baseline','rule_execution:missing-old-value')
        for baseline in (base.execution_id,'unknown-baseline','unreadable-baseline'):
            _,r,closed=self.run_task(task('optional-'+baseline),entry_id=entry,check_extra={
                'rule_assessments':[assessment(entry,1,comparator_execution_id=baseline,cost_dimensions=['tool_calls'])]})
            comparison=closed['rule_validation']['checks'][0]['cost_comparison']
            self.assertEqual(comparison['state'],'unavailable')
            self.assertEqual(comparison['execution_id'],baseline)
            with self.assertRaisesRegex(ValueError,'baseline'):
                validate_case(self.h,entry,1,{'kind':'transfer','execution_id':r.execution_id,
                    'assessment':assessment(entry,1),'comparator_execution_id':baseline})

    def test_P1_6_receipt_rejection_rolls_back_rule_and_keeps_vector_v1_then_retries_sync(self):
        from or_harness.strategy.embedding_index import document_digest,document_entry
        from or_harness.strategy.vector_recall import recall_vectors
        entry,a,b,j=self.derive(publish=True)
        old=copy.deepcopy(self.h.embedding_index.items('strategic_knowledge'))
        args,j=self.revision(entry,review={'outcome':'revised','comparison':{
            'sufficient':True,'substantive':True,'source':'agent','assessment':'Reviewed revision.',
            'evidence_refs':['execution:does-not-exist']}})
        with self.assertRaisesRegex(ValueError,'learning evidence'):
            self.h.induce(**args)
        self.assertEqual(self.h.sbank.get(entry).rule['active_version'],1)
        self.assertEqual(len(inspect_versions(self.h,entry)['versions']),1)
        self.h.close();self.h=ORHarness(home=self.home,world_model=self.provider,embedding=LocalHashEmbeddingBackend());self.addCleanup(self.h.close)
        self.assertEqual(self.h.embedding_index.items('strategic_knowledge'),old)
        self.assertIn(entry,[r['entry_id'] for r in recall_vectors(self.h,RULE['do'],top_k=5)['strategic_knowledge']])
        self.assertEqual(get_job(self.h,j)['state'],'pending')
        args['review']['comparison']['evidence_refs']=[get_job(self.h,j)['source_fact_ref']]
        called_in_transaction=[]
        def external_failure(ids):
            called_in_transaction.append(self.h.store.conn.in_transaction)
            raise OSError('one simulated external failure')
        with patch.object(self.h.index_sync,'sync_entries',side_effect=external_failure) as sync:
            committed=self.h.induce(**args)
            self.assertFalse(self.h.store.conn.in_transaction)
            self.assertEqual(sync.call_count,1)
        self.assertEqual(called_in_transaction,[False])
        self.assertEqual(committed['index_sync']['state'],'deferred')
        from or_harness.world_model.induction_gate import meta_read
        self.assertIn(entry,meta_read(self.h.store,'rule_index_pending'))
        self.assertEqual(self.h.sbank.get(entry).rule['active_version'],2)
        again=self.h.induce(**args)
        self.assertTrue(again['already_completed'])
        self.assertEqual(again['index_sync']['state'],'synced')
        self.assertEqual(again['induction_receipt']['receipt_ref'],committed['induction_receipt']['receipt_ref'])
        self.assertEqual(len(self.h.actions.query(action_type='induce')),2)
        current=next(x for x in self.h.embedding_index.items('strategic_knowledge') if x['id']==entry)
        self.assertEqual(current['doc_digest'],document_digest(document_entry(self.h.sbank.get(entry))))
        self.assertNotEqual(current['doc_digest'],old[0]['doc_digest'])
        self.assertIn(entry,[r['entry_id'] for r in recall_vectors(self.h,RULE['do'],top_k=5)['strategic_knowledge']])

    def test_P2_1_relation_material_reference_routes_evidence_addition_to_original_receipt(self):
        entry,a,b,j=self.derive()
        _,r,c=self.run_task(task('evidence-round'),decision=1);j=c['induction_schedule']['pending_ids'][0]
        m=self.h.induction_material(induction_id=j)
        payload={'entry_id':entry,'operation':'add_evidence','evidence':[r.execution_id],'material_id':m['material_id']}
        result=self.cli(['induce','--relation',json.dumps(payload)])
        self.assertEqual(get_job(self.h,j)['state'],'completed')
        self.assertEqual(result['induction_receipt']['induction_id'],j)
        retry=self.h.induce(relations=[payload])
        self.assertTrue(retry['already_completed'])
        self.assertEqual(retry['induction_receipt']['receipt_ref'],result['induction_receipt']['receipt_ref'])
        _,r,c=self.run_task(task('another-round',930,210),decision=1);other=c['induction_schedule']['pending_ids'][0]
        self.h.induction_material(induction_id=other)
        with self.assertRaisesRegex(ValueError,'conflict|different'):
            self.h.induce(relations=[{**payload,'induction_id':other}])
        self.assertEqual(get_job(self.h,other)['state'],'pending')
        self.assertTrue(self.h.induce(relations=[{'entry_id':entry,'evidence':[r.execution_id]}])['maintenance_only'])
        self.empty(other)

    def test_P2_2_long_recalled_rules_cannot_starve_control_pagination(self):
        entry,a,b,j=self.derive(publish=True)
        from or_harness.core.schema import StrategicEntry
        for i in range(7):
            self.h.sbank.add(StrategicEntry(entry_id='',strategy_id='endpoints',pattern={'predicates':{'family':'assignment'}},
                claim={'text':('preserve integer optimum '+str(i)+' ')*65,
                       'conditions':{'note':('monotone integer interval '+str(i)+' ')*65},
                       'method':{'steps':[('derive signed endpoint '+str(i)+' ')*65]}},verification={'state':'verified'}))
        _,a,c=self.run_task(task('pages'),decision=1);j=c['induction_schedule']['pending_ids'][0]
        _,b,_=self.run_task(task('pages'),episode='newer')
        _,older,_=self.run_task(task('pages'),episode='newest')
        from or_harness.strategy.evidence_views import chars
        for budget in (32000,2400,1500,1024):
            with self.subTest(budget=budget),patch.dict('os.environ',{'OR_HARNESS_INDUCTION_MATERIAL_CHARS':str(budget)}):
                cursor=None;seen=set();cursors=set()
                for _ in range(5):
                    material=self.cli(['induction-material','--induction-id',j,'--limit','1','--related-top-k','0']+
                                      (['--cursor',cursor] if cursor else []))
                    self.assertEqual(material['material'][0]['execution_id'],a.execution_id)
                    seen.update(r['execution_id'] for r in material['material'])
                    for row in material['material']:
                        if row.get('view')=='whole_record_by_reference':self.assertTrue(row['inspect_hint'])
                    self.assertLessEqual(chars(material),budget)
                    summary=(f"Read {material['count']} task attempt(s) with cross-task controls and short rules. "
                       "Submit when/do/expect plus evidence IDs with `induce --relation`; select the existing entry for revisions.")
                    self.assertEqual(material['budget']['chars_used'],chars({'result':material,'summary':summary}))
                    self.assertLessEqual(material['budget']['chars_used'],budget)
                    cursor=material['budget']['next_cursor']
                    if not cursor:break
                    self.assertNotIn(cursor,cursors);cursors.add(cursor)
                self.assertTrue({b.execution_id,older.execution_id}<=seen)
                self.assertIsNone(cursor)
