"""Persistent deduplication with readable, attributable normal-path facts."""
import copy
import io
import json
from contextlib import redirect_stdout
from unittest.mock import patch

from g_fixtures import GCase, COUPLED, SPLIT
from review_fixtures import ReviewCase, RULE, task
from or_harness.cli import main
from or_harness.core.schema import ExecutionRecord
from or_harness.strategy.evidence_access import retain_record, resolve_execution
from or_harness.strategy.rule_versions import version_row, check_records
from or_harness.world_model.conditions import get_value, put_value
from or_harness.world_model.hplus import get_job
from or_harness.world_model.r_signal import technical_attempt_fact


class TestCompactExecution(GCase):
    def raw_record(self, execution_id):
        return self.h.store.loads(self.h.store.conn.execute(
            'SELECT payload FROM executions WHERE execution_id=?', (execution_id,)).fetchone()[0])

    def test_normal_checks_share_rules_and_methods_without_losing_verdicts(self):
        method={**COUPLED, 'extra':{'solution':{'BE':1,'CFC':1},'explanation':'Keep both products at one factory.'}}
        for episode in ('first', 'second'):
            p=self.predict_g(method=method,episode=episode)
            r=self.run_g(p)
            checked=self.check_g(r)
            closed=self.close_g(episode)
            self.assertEqual(closed['evaluations'][0]['benefit']['observed'],1)
            raw=self.raw_record(r.execution_id)
            self.assertEqual(raw['evidence_encoding'],'execution-evidence/3')
            self.assertEqual(raw['method_actual']['$shared'],raw['method_planned']['$shared'])
            self.assertEqual(raw['execution_features']['method_receipt']['method_ref'],'method_actual')
            report=get_value(self.h.store,raw['execution_features']['task_check']['report_ref'])
            for key in ('state','checks','scope','diffs'):
                self.assertEqual(report.get(key),checked['report'].get(key))
            encoded=self.h.store.loads(self.h.store.conn.execute(
                'SELECT payload FROM condition_values WHERE value_id=?',
                (raw['execution_features']['task_check']['report_ref'],)).fetchone()[0])
            self.assertNotIn('spec',encoded['acceptance_rule'])
            self.assertNotIn('path',encoded['checks'][0])
            self.assertNotIn('g_attribution',encoded)
            decoded=self.h.bank.get(r.execution_id)
            self.assertEqual(decoded.method_actual['extra'],method['extra'])
            self.assertEqual(decoded.cost.measured,r.cost.measured)
            self.assertEqual(decoded.execution_features['cost_provenance'],r.execution_features['cost_provenance'])
            evaluation=self.h.get_strategy_evaluation(closed['evaluations'][0]['evaluation_id'])
            self.assertEqual(evaluation['method_actual'],decoded.method_actual)
            self.assertEqual(evaluation['execution_status'],decoded.quality['status'])
            self.assertEqual(technical_attempt_fact(decoded)['label'],0)
            self.assertNotIn('semantic_attribution',technical_attempt_fact(decoded))
        self.assertEqual(self.h.store.conn.execute(
            "SELECT count(*) FROM condition_values WHERE kind='task_check_rule'").fetchone()[0],1)

    def test_failure_differences_and_unknown_evidence_keep_their_g_meaning(self):
        for episode,method,options,fidelity,failure,expected in (
            ('omitted',SPLIT,{'split':True},'faithful','candidate_method',0),
            ('code-bug',COUPLED,{'split':True},'deviated','implementation',None),
            ('crash',COUPLED,{'crash':True},'unknown','implementation',None)):
            p=self.predict_g(method=method,episode=episode)
            r=self.run_g(p,**options)
            self.check_g(r,fidelity=fidelity,failure=failure)
            closed=self.close_g(episode)
            observed=closed['evaluations'][0]['benefit'].get('observed')
            self.assertEqual(observed,expected)
            decoded=self.h.bank.get(r.execution_id)
            report=decoded.execution_features['task_check']
            if options.get('split'):
                check=next(c for c in report['checks'] if c.get('check')=='semantic_probe')
                self.assertEqual((check['expected'],check['observed'],check['ok']),(1,2,False))
            if options.get('crash'):
                self.assertTrue(decoded.failures)
                self.assertEqual(technical_attempt_fact(decoded)['label'],1)

    def test_retention_recovers_artifact_results_and_unique_legacy_facts(self):
        p=self.predict_g()
        r=self.run_g(p)
        self.check_g(r)
        r=self.h.bank.get(r.execution_id)
        # A hand-added fact has no result-file counterpart and must survive.
        r.execution_features['unique_observation']={'repair':'Verified the solver API signature.'}
        r.execution_features['solution_variables']['unrecorded_in_result']={'a':[1,2,3]}
        r.method_actual['extra']={'solution':copy.deepcopy(r.execution_features['solution_variables'])}
        r.execution_features['method_receipt']['method']=copy.deepcopy(r.method_actual)
        with self.h.store.transaction() as conn:
            reference=retain_record(self.h,conn,r)
            conn.execute('DELETE FROM executions WHERE execution_id=?',(r.execution_id,))
        raw=self.h.store.loads(self.h.store.conn.execute(
            'SELECT payload FROM condition_values WHERE value_id=?',(reference,)).fetchone()[0])
        self.assertNotIn('note',raw['execution_config'])
        self.assertNotIn('method',raw['execution_features']['method_receipt'])
        restored=resolve_execution(self.h,r.execution_id)
        self.assertEqual(restored.method_actual,r.method_actual)
        self.assertEqual(restored.execution_features['solution_variables'],r.execution_features['solution_variables'])
        self.assertEqual(restored.execution_features['unique_observation'],r.execution_features['unique_observation'])
        self.assertEqual(restored.execution_features['task_check'],r.execution_features['task_check'])
        self.assertEqual(restored.cost.measured,r.cost.measured)
        self.assertEqual(restored.execution_features['retained_task_payload']['task_id'],self.task['task_id'])
        with self.h.store.transaction() as conn:
            again=retain_record(self.h,conn,restored)
        saved=self.h.store.loads(self.h.store.conn.execute(
            'SELECT payload FROM condition_values WHERE value_id=?',(again,)).fetchone()[0])
        self.assertNotIn('retained_task_payload',saved['execution_features'])
        self.assertNotIn('retained_task_text',saved['execution_features'])

    def test_old_rows_read_without_migration_and_shared_inputs_roundtrip(self):
        p=self.predict_g()
        r=self.run_g(p)
        legacy=r.to_dict()
        legacy['evidence_encoding']='execution-evidence/2'
        before=copy.deepcopy(legacy)
        self.assertEqual(ExecutionRecord.from_dict(legacy).method_actual,r.method_actual)
        self.assertEqual(legacy,before)
        original={'method':r.method_actual,'profile_snapshot':r.profile_snapshot.to_dict()}
        with self.h.store.transaction() as conn:
            ref=put_value(conn,self.h.store,'storage_roundtrip',original)
        self.assertEqual(get_value(self.h.store,ref),original)
        checked=self.check_g(r)
        report=checked['report']
        report['g_attribution']=get_value(self.h.store,checked['g_assessment_ref'])['attribution']
        report['g_attribution']['method_actual']=copy.deepcopy(r.method_actual)
        report['g_attribution']['method_planned']=copy.deepcopy(r.method_planned)
        self.h.bank.set_task_check(r.execution_id,report)
        raw=self.raw_record(r.execution_id)
        saved=get_value(self.h.store,raw['execution_features']['task_check']['report_ref'])
        self.assertNotIn('g_attribution',saved)
        self.assertEqual(self.close_g()['evaluations'][0]['benefit']['observed'],1)
        context=self.h.build_prediction_context({**self.task,'task_id':'next'},'next',persist=False)
        old={'pairs':[{'evaluation_id':'historical-evaluation','prediction_id':p.prediction_id,
            'benefit':{'acceptance_rule':report['acceptance_rule']},
            'implementation_observation':{'label':0,'reason':'Protocol completed normally.',
                'semantic_attribution':report['g_attribution']}}]}
        original=copy.deepcopy(old)
        context.paired_feedback=old
        sent=context.provider_view()['prediction_execution_pairs']['pairs'][0]
        self.assertEqual(sent['implementation_observation']['label'],0)
        self.assertNotIn('method_actual',sent['implementation_observation']['semantic_attribution'])
        self.assertNotIn('spec',sent['benefit']['acceptance_rule'])
        self.assertEqual(context.paired_feedback,original)


class TestCompactRules(ReviewCase):
    def test_only_cited_evidence_is_in_rule_metadata_and_body_is_not_repeated(self):
        _,a,closed=self.run_task(task('A',1000,300,maximum=True),decision=1)
        _,b,_=self.run_task(task('B'))
        for i,name in enumerate(('seen-C','seen-D','seen-E')):
            self.run_task(task(name,812+37*i,200+9*i))
        job=closed['induction_schedule']['pending_ids'][0]
        self.h.induction_material(induction_id=job)
        result=self.h.induce(induction_id=job,relations=[{**RULE,'evidence':[a.execution_id,b.execution_id],
            'publication':{'state':'published','reason':'Both actual enumeration cases and the endpoint derivation support the stated scope.'}}],
            review={'outcome':'new_rule','comparison':self.comparison(job,substantive=True,basis='derivation')})
        eid=result['relations'][0]['saved']
        entry=self.h.sbank.get(eid)
        row=self.h.store.loads(self.h.store.conn.execute(
            'SELECT payload FROM strategic_entries WHERE entry_id=?',(eid,)).fetchone()[0])
        self.assertNotIn('expected',row)
        self.assertNotIn('prediction_track',row)
        self.assertNotIn('claim',entry.to_dict())
        from or_harness.world_model.state import KnowledgeRef
        frozen=KnowledgeRef.from_entry(entry).to_dict()
        self.assertNotIn('expected_quality_hat',frozen)
        self.assertNotIn('expected_cost_hat',frozen)
        self.assertEqual(KnowledgeRef.from_dict(frozen).to_dict(),frozen)
        from or_harness.core.schema import CostVector
        from or_harness.cli import _summarize_recall
        declared=copy.deepcopy(entry)
        declared.quality_estimated=True;declared.expected_quality_hat=0
        declared.expected_cost_hat=CostVector(tool_calls=0,measured={'tool_calls'})
        recommendation=self.h.selector._from_entry(entry.strategy_id,self.h.profile(task('estimates')),declared).to_dict()
        self.assertEqual(recommendation['expected'],{'quality':0,'cost':{'tool_calls':0}})
        self.assertIn('E[Q]=0',_summarize_recall({'recommendations':[recommendation]}))
        declared.quality_estimated=False
        recommendation=self.h.selector._from_entry(entry.strategy_id,self.h.profile(task('estimates')),declared).to_dict()
        self.assertEqual(recommendation['expected'],{'cost':{'tool_calls':0}})
        from or_harness.strategy.vector_recall import _knowledge_entry
        from or_harness.strategy.embedding_index import document_entry,document_digest
        with patch.object(self.h.sbank,'get',return_value=declared):
            hit=_knowledge_entry(self.h,{'id':eid,'doc_digest':document_digest(document_entry(declared))},
                1.0,self.h.profile(task('estimates')),False)
        self.assertEqual(hit['expected_cost_hat'],{'tool_calls':0})
        self.assertNotIn('expected_quality_hat',hit)
        self.assertNotIn('failure_prob',hit)
        metadata=get_value(self.h.store,version_row(self.h.store,eid,1)['metadata_ref'])
        self.assertEqual(set(metadata),{'seed_evidence','seed_fact_refs','material_ref'})
        self.assertEqual(set(metadata['seed_fact_refs']),{a.execution_id,b.execution_id})
        shown=get_value(self.h.store,metadata['material_ref'])['identities']
        self.assertEqual(len(shown),5)
        self.assertEqual(len(entry.rule['evidence']),2)
        _,trial,closed=self.run_task(task('adopted',743,159),entry_id=eid)
        check=next(c for c in check_records(self.h.store,eid,1) if c.get('execution_id')==trial.execution_id)
        self.assertEqual({r['execution_id'] for r in check['known_source_relations']},set(shown))
        self.assertEqual(trial.execution_features['used_entries']['rule_versions'][eid],1)
        self.h.plan_next(task('next',743,159),'next',candidates=[{
            'solver':'python','method':{'name':'full enumeration','steps':['Enumerate all feasible X,Y.']}}])
        request=self.provider.requests[-1]
        retrieval=request['prediction_context']['retrieval_evidence']
        rules=[]
        def visit(value):
            if isinstance(value,dict):
                if value.get('body')==RULE:rules.append(value)
                for v in value.values():visit(v)
            elif isinstance(value,list):
                for v in value:visit(v)
        visit(retrieval)
        self.assertEqual(len(rules),1,retrieval)
        self.assertNotIn('claim_text',json.dumps(rules[0]))
        self.assertEqual(get_job(self.h,job)['state'],'completed')

    def test_completed_material_cli_returns_receipt_without_reopening_job(self):
        _,_,closed=self.run_task(task('complete'),decision=1)
        job=closed['induction_schedule']['pending_ids'][0]
        submitted=self.empty(job,conclusion='The captured record does not establish a reusable increment.')
        before=list(self.h.store.conn.iterdump())
        output=io.StringIO()
        with patch.dict('os.environ',{'OR_EMBEDDING_BACKEND':'local-hashing'}),redirect_stdout(output):
            code=main(['--home',self.home,'induction-material','--induction-id',job])
        self.assertEqual(code,0,output.getvalue())
        result=json.loads(output.getvalue())['result']
        self.assertEqual(result['state'],'completed')
        self.assertEqual(result['receipt_ref'],submitted['induction_receipt']['receipt_ref'])
        self.assertEqual(result['receipt']['induction_id'],job)
        self.assertFalse(result['material_assembled'])
        self.assertEqual(list(self.h.store.conn.iterdump()),before)
        self.assertEqual(get_job(self.h,job)['state'],'completed')
