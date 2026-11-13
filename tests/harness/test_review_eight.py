"""P1-1..P1-8 regressions through ordinary task and induction entry points."""
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
from or_harness.world_model.conditions import get_value
from or_harness.world_model.hplus import jobs,get_job,receipt_observation,feedback


class TestReviewedReproductions(ReviewCase):
    def test_same_instance_with_reworded_text_is_not_independent_transfer(self):
        entry,a,b,job=self.derive()
        _,again,closed=self.run_task(task('B-reworded',wording='Use the supplied raw input and mathematical specification to find the minimum.'),entry_id=entry)
        self.assertEqual(closed['rule_validation']['checks'][0]['state'],'insufficient_evidence')
        self.assertIn('same',closed['rule_validation']['checks'][0]['reason'])

    def test_omitted_close_episode_does_not_mint_a_second_job(self):
        _,r,closed=self.run_task(task('one-episode'),decision=1)
        repeated=self.h.close_episode('one-episode')
        self.assertEqual(len(jobs(self.h,task_id='one-episode')),1)
        self.assertEqual(repeated['induction_schedule']['jobs'][0]['induction_id'],closed['induction_schedule']['jobs'][0]['induction_id'])

    def test_material_query_can_change_without_reusing_the_first_pack(self):
        _,r,closed=self.run_task(task('query'),decision=1)
        job=closed['induction_schedule']['pending_ids'][0]
        first=self.h.induction_material(induction_id=job,purpose='modelling',related_top_k=1)
        second=self.h.induction_material(induction_id=job,purpose='implementation',related_top_k=3)
        self.assertEqual(second['purpose'],'implementation')
        self.assertNotEqual(first['material_id'],second['material_id'])

    def test_ordinary_A_B_rule_independent_C_196_and_nonmonotone_boundary_0(self):
        entry,a,b,job=self.derive()
        self.assertEqual(a.quality['objective'],2699)
        self.assertEqual(b.quality['objective'],90)
        self.assertNotIn('base_task_id',a.execution_features['task_identity'])
        transfer,boundary=self.activate(entry)
        observation=receipt_observation(self.h,get_job(self.h,job))
        self.assertEqual(observation['label'],1)
        self.assertEqual(observation['validated_rules'][0]['version'],1)

    def test_renaming_and_one_sided_parent_do_not_supply_transfer(self):
        entry,a,b,job=self.derive()
        for name,parent in [('renamed-B',None),('parent-only','B')]:
            t=task(name)
            if parent:t['parent_task_id']=parent
            _,r,c=self.run_task(t,entry_id=entry)
            self.assertEqual(c['rule_validation']['checks'][0]['state'],'insufficient_evidence')

    def test_strong_new_inputs_override_identical_generic_prose(self):
        entry,a,b,job=self.derive()
        # Same structure and generic text is discovery evidence, not identity.
        t=task('C-generic',812,200,wording=task('B')['description'])
        _,r,c=self.run_task(t,entry_id=entry)
        self.assertEqual(c['rule_validation']['checks'][0]['state'],'passed',c['rule_validation'])

    def test_text_only_duplicate_hint_is_pending_while_material_remains_readable(self):
        from or_harness.strategy.evidence_views import declared_task_identity,instance_relation,base_partitions
        a=declared_task_identity({'task_id':'text-a','description':'Minimize 2 X - 5 Y; X+Y<=641, X>=2Y+130.'})
        b=declared_task_identity({'task_id':'text-b','description':'For X+Y<=641 and X>=2Y+130, find the minimum of 2 X - 5 Y.'})
        self.assertEqual(instance_relation(a,b),'possible_duplicate')
        root=declared_task_identity({'task_id':'root','description':'A retained original problem'})
        branch=declared_task_identity({'task_id':'branch','parent_task_id':'root','description':'A rephrased branch'})
        self.assertEqual(len(set(base_partitions([root,branch]))),1)

    def test_exact_trigger_and_all_material_sessions_are_owned_by_the_round(self):
        _,first,c=self.run_task(task('same-task'),decision=1,episode='positive')
        job=c['induction_schedule']['pending_ids'][0]
        _,latest,_=self.run_task(task('same-task'),decision=0,episode='zero')
        _,b,_=self.run_task(task('historical-B',700,140))
        m1=self.h.induction_material(induction_id=job,related_top_k=0)
        self.assertEqual(m1['material'][0]['execution_id'],first.execution_id)
        m2=self.h.induction_material(induction_id=job,purpose='implementation',query='endpoint direction',
                                     related_top_k=2,execution_ids=[b.execution_id])
        self.assertNotEqual(m1['material_id'],m2['material_id'])
        self.assertEqual(self.h.induction_material(induction_id=job,related_top_k=0)['material_id'],m1['material_id'])
        # Only the historical B is cited, without a material/job ID: the
        # unique delivered session still owns A and every displayed case.
        result=self.h.induce(relations=[{**RULE,'evidence':[b.execution_id]}])
        entry=result['relations'][0]['saved']
        meta=get_value(self.h.store,version_row(self.h.store,entry,1)['metadata_ref'])
        self.assertIn(first.execution_id,meta['derivation_execution_ids'])
        self.assertIn(b.execution_id,meta['derivation_execution_ids'])
        self.assertEqual(set(meta['material_read_refs']),{m1['material_id'],m2['material_id']})
        self.assertEqual(get_job(self.h,job)['state'],'completed')
        _,again,c=self.run_task(task('A-again'),entry_id=entry)
        self.assertEqual(c['rule_validation']['checks'][0]['state'],'insufficient_evidence')

    def test_material_and_job_conflicts_are_rejected_before_any_write(self):
        _,a,ca=self.run_task(task('pa'),decision=1)
        _,b,cb=self.run_task(task('pb',700,140),decision=1)
        ja,jb=ca['induction_schedule']['pending_ids'][0],cb['induction_schedule']['pending_ids'][0]
        ma=self.h.induction_material(induction_id=ja)
        self.h.induction_material(induction_id=jb)
        with self.assertRaisesRegex(ValueError,'conflict|different'):
            self.h.induce(induction_id=jb,relations=[{**RULE,'induction_id':ja,'material_id':ma['material_id'],'evidence':[b.execution_id]}])
        self.assertEqual(self.h.sbank.count(),0)

    def test_another_rounds_material_does_not_leak_into_the_current_scope(self):
        _,a,ca=self.run_task(task('rounds'),decision=1,episode='first')
        _,b,_=self.run_task(task('only-first-round',710,140))
        ja=ca['induction_schedule']['pending_ids'][0]
        self.h.induction_material(induction_id=ja,related_top_k=0,execution_ids=[b.execution_id])
        self.empty(ja)
        _,c,cc=self.run_task(task('rounds'),decision=1,episode='second')
        jc=cc['induction_schedule']['pending_ids'][0]
        self.h.induction_material(induction_id=jc,related_top_k=0)
        result=self.h.induce(induction_id=jc,relations=[{**RULE,'evidence':[c.execution_id]}])
        entry=result['relations'][0]['saved']
        meta=get_value(self.h.store,version_row(self.h.store,entry,1)['metadata_ref'])
        self.assertNotIn(b.execution_id,meta['derivation_execution_ids'])
        self.assertEqual(get_job(self.h,ja)['state'],'completed')

    def test_manual_allow_round_can_use_skip_contrast_and_old_close_retry_keeps_latest(self):
        from or_harness.world_model.hplus import configure
        from or_harness.world_model.induction_gate import start_decision
        configure(self.h,'manual')
        _,a,_=self.run_task(task('manual'),decision=0,episode='old')
        self.h.close_episode('manual','old',induction_decision='allow')
        _,b,_=self.run_task(task('manual',700,140),decision=0,episode='new')
        self.h.close_episode('manual','new',induction_decision='skip')
        self.h.close_episode('manual','old')
        self.assertEqual(start_decision(self.h,'manual')['state'],'skip')
        material=self.h.induction_material(task_id='manual',episode_id='old',execution_ids=[b.execution_id])
        result=self.h.induce(relations=[{**RULE,'material_id':material['material_id'],'evidence':[a.execution_id,b.execution_id]}])
        self.assertTrue(result['saved'])

    def test_two_connections_late_material_does_not_reopen_completed_job(self):
        _,a,c=self.run_task(task('interleaved'),decision=1)
        job=c['induction_schedule']['pending_ids'][0]
        self.h.induction_material(induction_id=job,related_top_k=0)
        other=ORHarness(home=self.home,embedding=LocalHashEmbeddingBackend())
        self.addCleanup(other.close)
        from or_harness.world_model.induction_material import build_compact_material
        results=[]
        def interleave(*args,**kw):
            material=build_compact_material(*args,**kw)
            results.append(other.induce(induction_id=job,notes=['Compared actual facts; no reusable increment.']))
            return material
        with patch('or_harness.world_model.induction_material.build_compact_material',side_effect=interleave):
            self.h.induction_material(induction_id=job,purpose='implementation',related_top_k=0)
        self.assertEqual(get_job(self.h,job)['state'],'completed')
        repeated=self.h.induce(induction_id=job,notes=['Compared actual facts; no reusable increment.'])
        self.assertEqual(repeated['induction_receipt']['induction_action_id'],results[0]['induction_receipt']['induction_action_id'])
        self.assertEqual(len(self.h.actions.query(action_type='induce')),1)
        self.assertEqual(self.h.close_episode('interleaved')['induction_schedule']['jobs'][0]['induction_id'],job)
        with self.assertRaisesRegex(ValueError,'different|conflict'):
            self.h.induce(induction_id=job,notes=['Different conclusion.'])

    def test_two_real_positive_rounds_remain_distinct_and_omission_is_ambiguous(self):
        _,a,ca=self.run_task(task('two-rounds'),decision=1,episode='r1')
        _,b,cb=self.run_task(task('two-rounds'),decision=1,episode='r2')
        with self.assertRaisesRegex(ValueError,'ambiguous episode.*method'):
            self.h.close_episode('two-rounds')
        ja,jb=ca['induction_schedule']['pending_ids'][0],cb['induction_schedule']['pending_ids'][0]
        self.assertNotEqual(ja,jb)
        self.empty(ja);self.empty(jb)
        self.assertEqual(len(self.h.actions.query(action_type='induce')),2)

    def test_explicit_execution_resolves_unnamed_round_without_querying_all_episodes(self):
        _,a,ca=self.run_task(task('unnamed-and-named'),decision=1,unnamed=True)
        _,b,cb=self.run_task(task('unnamed-and-named'),decision=1,episode='named')
        with self.assertRaisesRegex(ValueError,'ambiguous episode'):
            self.h.close_episode('unnamed-and-named')
        repeated=self.h.close_episode('unnamed-and-named',execution_id=a.execution_id)
        self.assertEqual(repeated['induction_schedule']['jobs'][0]['execution_ids'],[a.execution_id])
        self.assertEqual(len(repeated['induction_schedule']['jobs']),1)
        self.assertEqual(repeated['induction_schedule']['jobs'][0]['induction_id'],ca['induction_schedule']['jobs'][0]['induction_id'])
        with self.assertRaisesRegex(ValueError,'conflicts'):
            self.h.close_episode('unnamed-and-named','named',execution_id=a.execution_id)

    def test_omitted_optional_review_preserves_support_with_two_withdraw_restore_cycles(self):
        entry,a,b,job=self.derive();transfer,boundary=self.activate(entry)
        self.recheck(transfer)  # acceptance spec omits optional assessments
        self.assertTrue(self.h.sbank.get(entry).is_published)
        cutoff=time.time()
        frozen=self.h.predict_strategy_outcome(task('before-withdrawal'),{'strategy_id':'endpoint','solver':'python'})
        context=self.h.get_prediction_context(frozen.trace.model_info['prediction_context_id'])
        prior=copy.deepcopy(context.hplus_feedback)
        for i in range(2):
            self.h.exclude_execution(transfer.execution_id,f'withdrawal {i}')
            self.assertFalse(self.h.sbank.get(entry).is_published)
            self.assertIsNone(receipt_observation(self.h,get_job(self.h,job))['label'])
            self.h.restore_execution(transfer.execution_id,f'restored {i}')
            self.assertTrue(self.h.sbank.get(entry).is_published)
            self.assertEqual(receipt_observation(self.h,get_job(self.h,job))['label'],1)
        events=[e for e in inspect_versions(self.h,entry)['events'] if e['kind']=='publication']
        self.assertEqual([e['to'] for e in events],['active','paused','active','paused','active'])
        self.assertEqual(len({e['event_id'] for e in events}),5)
        self.assertEqual(receipt_observation(self.h,get_job(self.h,job),as_of=cutoff)['label'],1)
        self.assertEqual(context.provider_view()['hplus_feedback'],prior)

    def test_explicit_assessment_withdrawal_and_reinstatement_are_real_changes(self):
        entry,a,b,job=self.derive();transfer,boundary=self.activate(entry)
        self.recheck(transfer,rule_assessments=[])
        self.assertFalse(self.h.sbank.get(entry).is_published)
        self.recheck(transfer,rule_assessments=[assessment(entry,1)])
        self.assertTrue(self.h.sbank.get(entry).is_published)

    def test_one_of_two_transfers_can_be_withdrawn_without_losing_valid_support(self):
        entry,a,b,job=self.derive();transfer,boundary=self.activate(entry)
        self.run_task(task('C2',920,220),entry_id=entry)
        self.h.exclude_execution(transfer.execution_id,'one source is wrong')
        self.assertTrue(self.h.sbank.get(entry).is_published)
        self.assertEqual(receipt_observation(self.h,get_job(self.h,job))['label'],1)

    def test_evicted_pending_trigger_and_selected_contrast_survive_restart_and_cli_read(self):
        from or_harness.world_model.episode_closeout import enforce_evidence_window,CalibrationPolicy,EvidenceWindowPolicy
        _,a,c=self.run_task(task('evicted-trigger'),decision=1)
        job=c['induction_schedule']['pending_ids'][0]
        _,b,_=self.run_task(task('evicted-contrast',700,140))
        material=self.h.induction_material(induction_id=job,execution_ids=[b.execution_id])
        for i in range(3):self.run_task(task('later-'+str(i),820+i,180+i))
        result=enforce_evidence_window(self.h,policy=EvidenceWindowPolicy(window_episodes=1),
                                      calibration_policy=CalibrationPolicy(window=1,late_check_grace_days=0))
        self.assertIsNone(self.h.bank.get(a.execution_id))
        self.assertIsNone(self.h.bank.get(b.execution_id))
        self.h.close()
        self.h=ORHarness(home=self.home,embedding=LocalHashEmbeddingBackend())
        self.addCleanup(self.h.close)
        out=io.StringIO()
        with patch.dict('os.environ',{'OR_EMBEDDING_BACKEND':'local-hashing'}),redirect_stdout(out):
            code=main(['--home',self.home,'inspect','--bank','experience','--execution',a.execution_id])
        self.assertEqual(code,0,out.getvalue())
        record=json.loads(out.getvalue())['result']['records'][0]
        self.assertEqual(record['execution_features']['task_check']['state'],'passed')
        self.assertTrue(record['method_actual'])
        again=self.h.induction_material(induction_id=job,query='actual endpoint boundary',execution_ids=[b.execution_id])
        self.assertEqual(again['material'][0]['execution_id'],a.execution_id)
        submitted=self.h.induce(induction_id=job,relations=[{**RULE,'evidence':[a.execution_id,b.execution_id]}])
        self.assertEqual(submitted['saved'],1)
        self.assertIsNone(self.h.bank.get(a.execution_id))
        self.assertLessEqual(self.h.bank.count(),2)

    def test_withdrawn_frozen_source_is_not_resurrected_for_submission(self):
        from or_harness.world_model.episode_closeout import enforce_evidence_window,CalibrationPolicy,EvidenceWindowPolicy
        _,a,c=self.run_task(task('withdrawn-trigger'),decision=1)
        job=c['induction_schedule']['pending_ids'][0]
        self.h.induction_material(induction_id=job)
        self.run_task(task('newer',812,200))
        enforce_evidence_window(self.h,policy=EvidenceWindowPolicy(window_episodes=0),
                                calibration_policy=CalibrationPolicy(window=1,late_check_grace_days=0))
        self.assertIsNone(self.h.bank.get(a.execution_id))
        self.h.exclude_execution(a.execution_id,'invalid actual result')
        with self.assertRaisesRegex(ValueError,'unknown execution'):
            self.h.induce(induction_id=job,relations=[{**RULE,'evidence':[a.execution_id]}])
        self.assertEqual(get_job(self.h,job)['state'],'pending')
        self.empty(job,outcome='insufficient_material')
        self.h.restore_execution(a.execution_id,'audit verified the actual artifact')
        self.assertIsNone(self.h.bank.get(a.execution_id))


class TestReviewedRegressionAndCosts(ReviewCase):
    def revision(self):
        upper={**RULE,'when':'X,Y nonnegative integers, linear objective increasing in X, and maximization.',
               'do':'Enumerate Y and use the upper X endpoint for maximization; disable for a nonmonotone objective.'}
        _,a,c=self.run_task(task('old-A',1000,300,maximum=True),decision=1)
        j=c['induction_schedule']['pending_ids'][0]
        self.h.induction_material(induction_id=j,related_top_k=0)
        entry=self.h.induce(induction_id=j,relations=[{**upper,'evidence':[a.execution_id]}])['relations'][0]['saved']
        _,t,_=self.run_task(task('old-transfer',1100,360,maximum=True),entry_id=entry)
        _,boundary,_=self.run_task(task('old-boundary',12,0,boundary=True),entry_id=entry)
        self.assertTrue(self.h.sbank.get(entry).is_published)
        # Duplicate old use should be one representative regression.
        _,repeat,_=self.run_task(task('old-A-repeat',1000,300,maximum=True),entry_id=entry)
        _,b,c=self.run_task(task('new-minimum'),decision=1)
        job=c['induction_schedule']['pending_ids'][0]
        self.h.induction_material(induction_id=job,related_top_k=0)
        result=self.h.induce(induction_id=job,relations=[{**RULE,'entry_id':entry,'operation':'revise','evidence':[b.execution_id]}],
            review={'comparison':self.comparison(job,substantive=True,changed_fields=['when','do'])})
        meta=get_value(self.h.store,version_row(self.h.store,entry,2)['metadata_ref'])
        self.assertEqual(len(meta['regression_cases']),3)
        return entry,job,a,t,boundary

    def test_v2_real_reruns_resolve_old_cases_and_publish_after_independent_transfer(self):
        entry,job,a,t,boundary=self.revision()
        old_ids={a.execution_id,t.execution_id,boundary.execution_id}
        new_ids=set()
        for tsk in [task('v2-A',1000,300,maximum=True),task('v2-transfer',1100,360,maximum=True),
                    task('v2-boundary',12,0,boundary=True)]:
            _,r,c=self.run_task(tsk,entry_id=entry,version=2)
            self.assertEqual(c['rule_validation']['checks'][0]['state'],'passed',c['rule_validation'])
            new_ids.add(r.execution_id)
        checks=[c for c in check_records(self.h.store,entry,2) if c['kind']=='regression' and c['state']=='passed']
        self.assertEqual({c['execution_id'] for c in checks},new_ids)
        self.assertTrue(old_ids<=set(eid for c in checks for m in c['regression_matches'] for eid in m['previous_execution_ids']))
        self.assertTrue(all(m['basis']=='actual_version_use' for c in checks for m in c['regression_matches']))
        _,r,c=self.run_task(task('v2-independent',812,200),entry_id=entry,version=2)
        self.assertEqual(self.h.sbank.get(entry).rule['active_version'],2)
        self.assertEqual(receipt_observation(self.h,get_job(self.h,job))['label'],1)

    def test_changed_data_weaker_checker_and_v1_adoption_leave_v2_regression_open(self):
        entry,job,a,t,boundary=self.revision()
        _,r,c=self.run_task(task('old-A',1001,300,maximum=True),entry_id=entry,version=2)
        self.assertEqual(c['rule_validation']['checks'][0]['state'],'insufficient_evidence')
        self.assertIn('frozen input',c['rule_validation']['checks'][0]['reason'])
        _,r,c=self.run_task(task('old-A',1000,300,maximum=True),entry_id=entry,version=2,
                            check_extra={'semantic_probe':[],'integer':None})
        self.assertEqual(c['rule_validation']['checks'][0]['state'],'insufficient_evidence')
        self.assertIn('acceptance rule',c['rule_validation']['checks'][0]['reason'])
        _,r,c=self.run_task(task('v1-A',1000,300,maximum=True),entry_id=entry,version=1)
        self.assertEqual(self.h.sbank.get(entry).rule['active_version'],1)
        self.assertFalse(any(c['state']=='passed' and c.get('regression_matches') for c in check_records(self.h.store,entry,2)))

    def test_merge_into_existing_candidate_also_requires_source_cases_on_new_execution_ids(self):
        entry,job,a,t,boundary=self.revision()
        _,source_case,c=self.run_task(task('source-max',1200,390,maximum=True),decision=1)
        sj=c['induction_schedule']['pending_ids'][0]
        self.h.induction_material(induction_id=sj,related_top_k=0)
        source_rule={**RULE,'when':'Nonnegative integer interval with maximization and positive X coefficient.',
                     'do':'For each Y derive the upper endpoint from capacity minus Y. For a nonmonotone objective disable endpoint selection.'}
        source=self.h.induce(induction_id=sj,relations=[{**source_rule,'evidence':[source_case.execution_id]}])['relations'][0]['saved']
        self.run_task(task('source-transfer',1300,400,maximum=True),entry_id=source)
        self.run_task(task('source-boundary',14,0,boundary=True),entry_id=source)
        self.assertTrue(self.h.sbank.get(source).is_published)
        _,merge_case,c=self.run_task(task('merge-trigger',950,210),decision=1)
        mj=c['induction_schedule']['pending_ids'][0]
        self.h.induction_material(induction_id=mj,related_top_k=0)
        merged=self.h.induce(induction_id=mj,relations=[{'entry_id':entry,'version':2,'operation':'merge',
            'merge_entry_ids':[source],'evidence':[merge_case.execution_id]}])
        self.assertEqual(merged['relations'][0]['version'],2)
        for tsk in [task('v2-A',1000,300,maximum=True),task('v2-transfer',1100,360,maximum=True),
                    task('v2-boundary',12,0,boundary=True),task('v2-independent',812,200)]:
            self.run_task(tsk,entry_id=entry,version=2)
        self.assertEqual(self.h.sbank.get(entry).rule['active_version'],1)
        self.assertTrue(self.h.sbank.get(source).is_published)
        for tsk in [task('rerun-source',1200,390,maximum=True),task('rerun-source-transfer',1300,400,maximum=True),
                    task('rerun-source-boundary',14,0,boundary=True)]:
            _,r,c=self.run_task(tsk,entry_id=entry,version=2)
            self.assertEqual(c['rule_validation']['checks'][0]['state'],'passed',c['rule_validation'])
        self.assertEqual(self.h.sbank.get(entry).rule['active_version'],2)
        self.assertEqual(self.h.sbank.get(source).rule['merged_into'],entry)

    def test_H0_cost_trial_automatically_finds_measured_nine_to_three_baseline(self):
        rule={**RULE,'effect_scope':'cost','expect':'Reduce tool_calls for the same accepted original problem.'}
        entry,a,b,job=self.derive(rule=rule)
        self.assertEqual(get_job(self.h,job)['state'],'completed')
        _,baseline,_=self.run_task(task('cost-C',812,200),exhaustive=True,measured={'tool_calls':9})
        _,treatment,c=self.run_task(task('cost-C',812,200),episode='new-method',entry_id=entry,
                                    measured={'tool_calls':3},decision=0)
        self.assertEqual(c['induction_schedule']['jobs'][0]['state'],'skipped')
        checked=[v for v in check_records(self.h.store,entry,1) if v.get('execution_id')==treatment.execution_id][-1]
        self.assertEqual(checked['state'],'passed',checked)
        self.assertEqual(checked['cost_comparison']['execution_id'],baseline.execution_id)
        self.assertEqual(checked['cost_comparison']['dimensions']['tool_calls'],{'actual':3,'baseline':9})
        self.assertIn(baseline.execution_id,checked['cost_comparison']['measurement_sources'])
        self.assertEqual(get_job(self.h,job)['state'],'completed')

    def test_estimates_never_validate_a_cost_effect_despite_measured_mask(self):
        rule={**RULE,'effect_scope':'cost','expect':'Reduce llm_tokens while preserving the original acceptance.'}
        entry,a,b,job=self.derive(rule=rule)
        self.run_task(task('estimated-C',812,200),exhaustive=True,measured={'llm_tokens':100},provenance='agent_estimate')
        _,r,c=self.run_task(task('estimated-C',812,200),episode='estimated-new',entry_id=entry,
                            measured={'llm_tokens':1},provenance='agent_estimate')
        checked=[v for v in check_records(self.h.store,entry,1) if v.get('execution_id')==r.execution_id][-1]
        self.assertEqual(checked['state'],'insufficient_evidence')
        self.assertIn('trusted measured',checked['reason'])
        self.assertNotIn('cost_comparison',checked)

    def test_cli_cost_fact_correction_pauses_and_restores_dependent_rule_and_hplus(self):
        rule={**RULE,'effect_scope':'cost','expect':'Reduce tool_calls while preserving acceptance.'}
        entry,a,b,job=self.derive(rule=rule)
        _,baseline,_=self.run_task(task('corrected-cost',812,200),exhaustive=True,measured={'tool_calls':9})
        self.run_task(task('corrected-cost',812,200),episode='new',entry_id=entry,measured={'tool_calls':3})
        self.run_task(task('cost-boundary',12,0,boundary=True),entry_id=entry)
        self.assertTrue(self.h.sbank.get(entry).is_published)
        for value,published,label in [(1,False,None),(9,True,1)]:
            out=io.StringIO()
            with patch.dict('os.environ',{'OR_EMBEDDING_BACKEND':'local-hashing'}),redirect_stdout(out):
                code=main(['--home',self.home,'amend-cost',baseline.execution_id,'--override',f'tool_calls={value}',
                           '--source','agent_observed','--force'])
            self.assertEqual(code,0,out.getvalue())
            self.assertTrue(json.loads(out.getvalue())['result']['rule_validation']['affected_versions'])
            self.assertEqual(self.h.sbank.get(entry).is_published,published)
            self.assertEqual(receipt_observation(self.h,get_job(self.h,job))['label'],label)

    def test_ambiguous_cost_baselines_offer_existing_choices_and_explicit_selection(self):
        rule={**RULE,'effect_scope':'cost','expect':'Reduce tool_calls while preserving acceptance.'}
        entry,a,b,job=self.derive(rule=rule)
        _,base1,_=self.run_task(task('ambiguous-C',812,200),exhaustive=True,measured={'tool_calls':9})
        self.run_task(task('ambiguous-C',812,200),episode='other-baseline',exhaustive=True,measured={'tool_calls':7})
        _,r,c=self.run_task(task('ambiguous-C',812,200),episode='new',entry_id=entry,measured={'tool_calls':3})
        checked=[v for v in check_records(self.h.store,entry,1) if v.get('execution_id')==r.execution_id][-1]
        self.assertEqual(checked['state'],'insufficient_evidence')
        self.assertEqual(len(checked['comparison_candidates']),2)
        self.recheck(r,rule_assessments=[assessment(entry,1,comparator_execution_id=base1.execution_id,cost_dimensions=['tool_calls'])])
        checked=[v for v in check_records(self.h.store,entry,1) if v.get('execution_id')==r.execution_id][-1]
        self.assertEqual(checked['state'],'passed',checked)
        self.assertEqual(checked['cost_comparison']['execution_id'],base1.execution_id)


class TestReviewedKnowledgeFeedback(ReviewCase):
    def covered_job(self,*,checker=True):
        entry,a,b,original=self.derive();self.activate(entry)
        _,r,c=self.run_task(task('covered',930,210),decision=1,checker=checker)
        job=c['induction_schedule']['pending_ids'][0]
        self.h.induction_material(induction_id=job,related_top_k=0)
        return entry,job,r

    def test_format_equivalent_revision_reuses_version_and_never_is_positive(self):
        entry,job,r=self.covered_job()
        result=self.h.induce(induction_id=job,relations=[{**{k:v.replace(' ','  ') for k,v in RULE.items()},
            'entry_id':entry,'operation':'revise','evidence':[r.execution_id]}])
        update=result['induction_receipt']['rule_updates'][0]
        self.assertEqual(update['version'],1)
        self.assertEqual(update['increment']['status'],'equivalent')
        self.assertIsNone(receipt_observation(self.h,get_job(self.h,job))['label'])

    def test_synonym_revision_without_sourced_mechanism_change_stays_unknown(self):
        entry,job,r=self.covered_job()
        result=self.h.induce(induction_id=job,relations=[{**RULE,'do':RULE['do'].replace('Enumerate','Iterate over'),
            'entry_id':entry,'operation':'revise','evidence':[r.execution_id]}])
        update=result['induction_receipt']['rule_updates'][0]
        self.assertEqual(update['version'],2)
        self.assertIsNone(update['substantive'])
        self.assertEqual(update['increment']['status'],'unknown')
        for tsk in [task('synonym-A',1000,300,maximum=True),task('synonym-B'),task('synonym-C',812,200),
                    task('synonym-boundary',12,0,boundary=True),task('synonym-new',1110,330)]:
            self.run_task(tsk,entry_id=entry,version=2)
        self.assertEqual(self.h.sbank.get(entry).rule['active_version'],2)
        self.assertIsNone(receipt_observation(self.h,get_job(self.h,job))['label'])

    def test_covered_evidence_addition_is_negative_and_survives_unknown_receipt_flood(self):
        entry,job,r=self.covered_job()
        comparison=self.comparison(job)
        result=self.h.induce(induction_id=job,relations=[{'entry_id':entry,'operation':'add_evidence','evidence':[r.execution_id]}],
            review={'outcome':'already_covered','comparison':comparison,'conclusion':'Existing endpoint rule covers this case.'})
        self.assertEqual(receipt_observation(self.h,get_job(self.h,job))['label'],0)
        for i in range(6):
            _,u,c=self.run_task(task('unknown-'+str(i),940+i,220+i),decision=1)
            self.empty(c['induction_schedule']['pending_ids'][0])
        self.run_task(task('next-request',956,230),decision=0)
        rows=self.provider.requests[-1]['prediction_context']['hplus_feedback']['settled_examples']
        self.assertIn(job,[get_value(self.h.store,row['receipt_ref'])['induction_id'] for row in rows])
        self.assertEqual(next(row for row in rows if row['prediction_id']==get_job(self.h,job)['prediction_id'])['label'],0)

    def test_late_checker_changes_only_future_negative_feedback_and_explicit_withdrawal_reverses_it(self):
        entry,job,r=self.covered_job(checker=False)
        comparison=self.comparison(job)
        self.h.induce(induction_id=job,relations=[{'entry_id':entry,'operation':'add_evidence','evidence':[r.execution_id]}],
            review={'outcome':'already_covered','comparison':comparison})
        cutoff=time.time()
        self.assertIsNone(receipt_observation(self.h,get_job(self.h,job))['label'])
        frozen=self.h.predict_strategy_outcome(task('before-late-check'),{'strategy_id':'endpoint','solver':'python'})
        context=self.h.get_prediction_context(frozen.trace.model_info['prediction_context_id'])
        prior=copy.deepcopy(context.hplus_feedback)
        y=r.execution_features['solution_variables']['Y']
        check={'reference_objective':2*210-((930-210)//3),'reference_source':'bench_declared','reference_version':'independent integer calculation',
               'integer':{'variables':['X','Y']},'semantic_probe':[{'path':'execution_features.solution_variables.X','min':2*y+210,'max':930-y}]}
        self.h.check_task_result(r.execution_id,check)
        self.assertEqual(receipt_observation(self.h,get_job(self.h,job))['label'],0)
        self.assertIsNone(receipt_observation(self.h,get_job(self.h,job),as_of=cutoff)['label'])
        self.assertEqual(context.provider_view()['hplus_feedback'],prior)
        self.h.exclude_execution(r.execution_id,'correction invalidates this comparison')
        self.assertIsNone(receipt_observation(self.h,get_job(self.h,job))['label'])
        self.h.restore_execution(r.execution_id,'restored checked original fact')
        self.assertEqual(receipt_observation(self.h,get_job(self.h,job))['label'],0)
