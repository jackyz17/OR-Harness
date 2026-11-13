"""Normal execution/check/close paths for short rules and held-out evidence."""
import copy
import io
import json
import os
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from tests.harness.helpers import HarnessTestCase
from or_harness.api import ORHarness
from or_harness.cli import main
from or_harness.core.schema import StrategicEntry
from or_harness.strategy.embedding_index import LocalHashEmbeddingBackend
from or_harness.strategy.rule_versions import check_records, inspect_versions, version_row
from or_harness.strategy.selector import is_publishable
from or_harness.world_model.conditions import get_value

SHORT_RULE = {
    "when": "X,Y are nonnegative integers; for fixed Y, feasible X is an interval and the objective is monotone in X.",
    "do": "Enumerate feasible Y, derive both X endpoints, choose the endpoint from the objective direction and coefficient sign, then compare the resulting objectives. If monotonicity fails, disable this reduction and enumerate the full feasible set.",
    "expect": "Preserve the original integer optimum and all stated constraints.",
}
METHOD = {"name": "signed endpoint enumeration", "steps": [SHORT_RULE["do"]]}


class RuleCase(HarnessTestCase):
    def setUp(self):
        super().setUp()
        self.h = ORHarness(home=self.home, embedding=LocalHashEmbeddingBackend())
        self.addCleanup(self.h.close)

    def solve(self, name="training", *, use=None, version=1, lower=True, boundary=False, base=None,
              assessment=True, coverage="complete", failure_source=None, do_check=True, parameters=None, maximize=None,
              performed_method=None, prediction_id=None):
        independent = name != "training"
        n, c, coefficient = (641, 130, -5) if independent else (1000, 300, 5)
        if parameters:
            n,c=parameters
        maximum=(not independent) if maximize is None else maximize
        coefficient=5 if maximum else -5
        if boundary:
            n, c = 12, 0
        task = {"task_id": name, "family": "assignment", "base_task_id": base or name,
                "description": f"Nonnegative integer X,Y. X+Y <= {n}; X >= 2Y+{c}. " +
                    ("Minimize (X-3)^2+Y; objective is not monotone." if boundary else
                     ("Maximize 2X+5Y." if maximum else "Minimize 2X-5Y."))}
        work = Path(self.home) / (name + '-' + str(len(self.h.actions.query())))
        work.mkdir()
        code = "import json,os\nvalues=[]\n"
        if boundary:
            code += "for y in range(13):\n for x in range(2*y,13-y):\n  values.append(((x-3)**2+y,x,y))\nanswer=min(values)\n"
        else:
            code += f"for y in range({n}+1):\n lo=2*y+{c}\n hi={n}-y\n if lo<=hi:\n  x=" + ("lo" if not maximum and lower else "hi") + f"\n  values.append((2*x+({coefficient})*y,x,y))\n"
            code += "answer=" + ("max" if maximum else "min") + "(values)\n"
        code += "result={'status':'optimal','objective_value':answer[0],'objective_bound':answer[0], 'variables':{'X':answer[1],'Y':answer[2],'reduction_used':" + str(not boundary) + "},'method_performed':" + repr(performed_method or METHOD) + "}\n"
        code += "result['method_performed']['action_id']=os.environ['OR_ACTION_ID']\njson.dump(result,open('result.json','w'))\n"
        (work / 'solve.py').write_text(code)
        r = self.h.execute(task, "endpoint", str(work / 'solve.py'), str(work), solver="python",
                           episode_id=name, method=METHOD,
                           prediction_id=prediction_id,
                           used_entry_ids=[f"{use}@{version}"] if use else None)
        self.h.record(r)
        if do_check:
            x, y = r.execution_features["solution_variables"]["X"], r.execution_features["solution_variables"]["Y"]
            check = {"reference_objective": 0 if boundary else 2*n+3*((n-c)//3) if maximum else 2*c-((n-c)//3),
                "reference_source": "bench_declared", "reference_version": "independent integer optimum",
                "integer": {"variables": ["X", "Y"]},
                "semantic_probe": [{"path":"execution_features.solution_variables.X", "min":2*y+c,"max":n-y}]}
            if boundary:
                check['semantic_probe'].append({'path':'execution_features.solution_variables.reduction_used','equals':False})
            if use and assessment:
                check["rule_assessments"] = [{"entry_id": str(use), "version": version,"verdict":"supported","purpose":"boundary" if boundary else "transfer",
                    "applicability": "not_applicable" if boundary else "applies",
                    "method_fidelity": "faithful", "effect_coverage": coverage,
                    "response": "disable" if boundary else None, "failure_source": failure_source,
                    "source": "fixture auditor inspected captured script and variables against rule premises",
                    "assessment": "Endpoint choice follows coefficient/direction; nonmonotone case disables reduction. Check covers the stated integer optimum and both constraints.",
                    "evidence_refs": ["artifacts.submitted_script", "artifacts.workspace_result"]}]
            report = self.h.check_task_result(r.execution_id, check)
        return task, self.h.bank.get(r.execution_id)

    def seed(self):
        task, r = self.solve()
        self.h.close_episode(task['task_id'], task['task_id'])
        material = self.h.induction_material(task_id=task['task_id'])
        raw = {**SHORT_RULE, "evidence": [r.execution_id], "material_id": material['material_id']}
        response = self.h.induce(relations=[raw])
        self.assertEqual(response['saved'], 1, response)
        return response['relations'][0]['saved'], r, raw

    def activate(self, entry_id):
        task, transfer = self.solve("independent", use=entry_id)
        first = self.h.close_episode(task['task_id'], task['task_id'])
        self.assertEqual(first['rule_validation']['checks'][0]['state'], 'passed', first)
        self.assertFalse(is_publishable(self.h.sbank.get(entry_id)))
        task, boundary = self.solve("nonmonotone", use=entry_id, boundary=True)
        result = self.h.close_episode(task['task_id'], task['task_id'])
        self.assertEqual(result['rule_validation']['checks'][0]['state'], 'passed', result)
        self.h.induce(relations=[{'entry_id':entry_id,'publication':{'state':'published','reason':'Agent inspected the performed endpoint cases and nonmonotone fallback.'}}])
        self.assertTrue(is_publishable(self.h.sbank.get(entry_id)))
        return transfer, boundary


class TestRuleLifecycle(RuleCase):
    def test_normal_closed_task_short_submission_then_transfer_and_boundary(self):
        entry_id, training, _ = self.seed()
        self.assertEqual(training.quality['objective'], 2699)
        entry = self.h.sbank.get(entry_id)
        self.assertFalse(is_publishable(entry))
        self.assertEqual(entry.rule['candidate_version'], 1)
        self.assertFalse(any(r['evidence']=='strategic_entry' for r in self.h.recall({'task_id':'next','family':'assignment'})['recommendations']))
        transfer, boundary = self.activate(entry_id)
        self.assertEqual(transfer.quality['objective'], 90)
        checks = check_records(self.h.store, entry_id)
        self.assertEqual(sum(c['kind']=='transfer' and c['state']=='passed' for c in checks), 1)
        self.assertEqual(sum(c['kind']=='boundary' and c['state']=='passed' for c in checks), 1)
        self.assertEqual(sum(r['evidence']=='strategic_entry' for r in self.h.recall({'task_id':'next','family':'assignment'})['recommendations']), 1)

    def test_known_source_is_reported_without_deciding_transfer_for_the_agent(self):
        entry_id,training,_=self.seed()
        t,r=self.solve('seed-42',use=entry_id,base='training')
        result=self.h.close_episode(t['task_id'],t['task_id'])
        check=check_records(self.h.store,entry_id)[-1]
        self.assertEqual(check['state'],'passed')
        self.assertTrue(any(v['relationship']=='same_source' for v in check['known_source_relations']))
        self.assertFalse(is_publishable(self.h.sbank.get(entry_id)))


    def test_repeat_submission_is_idempotent_and_evidence_does_not_rewrite_body(self):
        entry_id, training, raw = self.seed()
        row = dict(version_row(self.h.store, entry_id, 1))
        repeated = self.h.induce(relations=[raw])
        self.assertEqual(repeated['relations'][0]['saved'], entry_id)
        self.assertTrue(repeated['relations'][0]['idempotent'])
        self.assertEqual(repeated['knowledge_delta']['entries_created'], [])
        self.assertEqual(self.h.sbank.count(), 1)
        task, case = self.solve('new-evidence')
        result = self.h.induce(relations=[{'entry_id':entry_id,'operation':'add_evidence','evidence':[case.execution_id]}])
        self.assertEqual(result['relations'][0]['version'], 1)
        self.assertEqual(dict(version_row(self.h.store,entry_id,1)), row)
        self.assertEqual(result['index_sync']['items'], 0)

    def test_explicit_body_revert_is_a_new_version_but_exact_repeat_is_not(self):
        entry_id,training,raw=self.seed()
        revised={**raw,'entry_id':entry_id,'operation':'revise',
                 'do':SHORT_RULE['do']+' Record finite bounds.'}
        self.assertEqual(self.h.induce(relations=[revised])['relations'][0]['version'],2)
        revert={**raw,'entry_id':entry_id,'operation':'revise'}
        third=self.h.induce(relations=[revert])['relations'][0]
        self.assertEqual(third['version'],3)
        self.assertEqual(third['body_ref'],version_row(self.h.store,entry_id,1)['body_ref'])
        self.assertTrue(self.h.induce(relations=[revert])['relations'][0]['idempotent'])

    def test_missing_adoption_and_partial_scope_remain_visible_to_the_agent(self):
        entry_id,_,_=self.seed()
        t,r=self.solve('no-adoption')
        self.assertEqual(self.h.close_episode(t['task_id'],t['task_id'])['rule_validation']['checks'],[])
        t,r=self.solve('partial',use=entry_id,coverage='partial')
        self.h.close_episode(t['task_id'],t['task_id'])
        c=check_records(self.h.store,entry_id)[-1]
        self.assertEqual(get_value(self.h.store,c['assessment_ref'])['effect_coverage'],'partial')
        self.assertFalse(is_publishable(self.h.sbank.get(entry_id)))


    def test_upper_endpoint_is_wrong_for_independent_minimization(self):
        entry_id, _, _ = self.seed()
        bad={**SHORT_RULE,'do':'Enumerate Y and always choose the upper X endpoint; disable if the objective is nonmonotone.',
             'entry_id':entry_id,'operation':'revise'}
        self.h.induce(relations=[bad])
        task, case = self.solve('wrong-upper',use=entry_id,version=2,lower=False,failure_source='rule',
                               performed_method={'name':'upper endpoint','steps':[bad['do']]})
        self.assertEqual(case.quality['objective'],92)
        result=self.h.close_episode(task['task_id'],task['task_id'])
        self.assertEqual(result['rule_validation']['checks'][0]['state'],'refuted')
        self.assertFalse(is_publishable(self.h.sbank.get(entry_id)))

    def test_active_rule_revision_keeps_old_use_without_compulsory_regressions(self):
        entry_id, _, _=self.seed()
        transfer,boundary=self.activate(entry_id)
        original=get_value(self.h.store,transfer.execution_features['used_entries']['versions'][entry_id])
        new={**SHORT_RULE,'do':SHORT_RULE['do']+' Check finite endpoint bounds explicitly.',
             'entry_id':entry_id,'operation':'revise','evidence':[transfer.execution_id]}
        result=self.h.induce(relations=[new])
        self.assertEqual(result['relations'][0]['version'],2)
        entry=self.h.sbank.get(entry_id)
        self.assertEqual(entry.rule['active_version'],1)
        self.assertEqual(entry.rule['candidate_version'],2)
        self.assertEqual(original['rule']['view_version'],1)
        self.assertFalse(any(c['state']=='pending' for c in check_records(self.h.store,entry_id,2)))
        self.h.close_episode(transfer.task_id,transfer.task_id)
        self.assertEqual(self.h.sbank.get(entry_id).rule['active_version'],1)

    def test_gate_blocks_expansion_and_transition_source_is_explicit(self):
        task,record=self.solve()
        self.h.close_episode(task['task_id'],task['task_id'],induction_decision={'state':'skip','reason':'no reusable mechanism','source':'closeout_hplus_fixture'})
        with patch('or_harness.world_model.induction_material.build_compact_material',side_effect=AssertionError('expensive material assembled')):
            result=self.h.induction_material(task_id=task['task_id'])
        self.assertFalse(result['material_assembled'])
        self.assertEqual(result['induction_gate']['state'],'skip')
        self.h.close_episode(task['task_id'],task['task_id'],induction_decision={'state':'pending','reason':'missing execution evidence'})
        self.assertEqual(self.h.induction_material(task_id=task['task_id'])['induction_gate']['state'],'pending')

    def test_full_default_budget_includes_long_old_rules_and_details_remain_readable(self):
        task,record=self.solve()
        self.h.close_episode(task['task_id'],task['task_id'])
        for index in range(29):
            entry=StrategicEntry(entry_id='',strategy_id='endpoint',pattern={'predicates':{'family':'assignment'}},
                claim={'text':('long historical explanation '+str(index)+' ')*1200,
                       'method':METHOD,'evidence':[{'execution_id':record.execution_id,'role':'source'}]},
                verification={'state':'unverified','checks':['long audit '*1000]})
            self.h.sbank.add(entry)
        result=self.h.induction_material(task_id=task['task_id'])
        serialized=json.dumps(result,ensure_ascii=False,separators=(',',':'))
        self.assertEqual(len(serialized),result['budget']['chars_used'])
        self.assertLessEqual(len(serialized),32000)
        self.assertTrue(result['material'])
        self.assertTrue(result['existing_knowledge'])
        self.assertGreater(result['budget']['omitted_knowledge_count'],0)
        detail=self.h.inspect(bank='strategic',entry_id='1')
        self.assertGreater(len(detail['entries'][0]['claim']['text']),20000)

    def test_prior_read_and_identical_alias_are_advisory_facts(self):
        entry_id,training,_=self.seed()
        t,r=self.solve('renamed-training',use=entry_id,parameters=(1000,300),maximize=True)
        self.h.close_episode(t['task_id'],t['task_id'])
        c=check_records(self.h.store,entry_id)[-1]
        self.assertTrue(any(v['relationship']=='same_source' for v in c['known_source_relations']))
        self.assertEqual(c['state'],'passed')
        self.assertFalse(is_publishable(self.h.sbank.get(entry_id)))


    def test_counterexample_is_evidence_and_agent_explicitly_manages_the_entry(self):
        entry_id,training,_=self.seed();self.activate(entry_id)
        t,bad=self.solve('bad-use',use=entry_id,lower=False,failure_source='rule',
                         performed_method={'name':'upper endpoint','steps':['Choose the upper endpoint.']})
        self.h.close_episode(t['task_id'],t['task_id'])
        self.assertEqual(check_records(self.h.store,entry_id)[-1]['state'],'refuted')
        self.assertTrue(is_publishable(self.h.sbank.get(entry_id)))
        result=self.h.induce(effect_attributions=[{'entry_id':entry_id,'execution_id':bad.execution_id,
            'verdict':'refuting','lifecycle_action':'demote','note':'Agent attributes the observed in-domain failure to this rule.'}])
        self.assertTrue(result['maintenance_only'])
        self.assertEqual(self.h.sbank.get(entry_id).rule['state'],'paused')


    def test_revised_candidate_switches_on_agent_choice_after_optional_reviews(self):
        entry_id,training,_=self.seed()
        transfer,boundary=self.activate(entry_id)
        changed={**SHORT_RULE,'when':SHORT_RULE['when']+' The finite endpoint bounds must be recorded.',
                 'entry_id':entry_id,'operation':'revise','evidence':[training.execution_id]}
        result=self.h.induce(relations=[changed])
        self.assertEqual(result['relations'][0]['version'],2)
        task,newcase=self.solve('new-independent',use=entry_id,version=2,parameters=(508,101))
        self.h.close_episode(task['task_id'],task['task_id'])
        task,newboundary=self.solve('new-boundary',use=entry_id,version=2,boundary=True)
        self.h.close_episode(task['task_id'],task['task_id'])
        self.assertEqual(self.h.sbank.get(entry_id).rule['active_version'],1)
        validations=[]
        for case in (training,transfer,boundary):
            assessment=copy.deepcopy(transfer.execution_features['task_check']['rule_assessments'][0])
            assessment.update(version=2,applicability='applies',effect_coverage='complete',method_fidelity='faithful')
            # Reviewing the previous boundary also checks its fallback under
            # the revised rule, rather than claiming it was an in-domain use.
            if case.execution_id==boundary.execution_id:
                assessment.update(applicability='not_applicable',response='disable')
            validations.append({'kind':'regression','execution_id':case.execution_id,'assessment':assessment})
        result=self.h.induce(relations=[{'entry_id':entry_id,'operation':'add_evidence','version':2,'validation':validations}])
        self.assertTrue(all(v['state']=='passed' for v in result['relations'][0]['validation']),result)
        self.assertTrue(all(v['basis']=='historical_review' for v in result['relations'][0]['validation']))
        self.assertEqual(self.h.sbank.get(entry_id).rule['active_version'],1)
        self.h.induce(relations=[{'entry_id':entry_id,'version':2,'publication':{'state':'published','reason':'Agent concludes the reviewed evidence covers this revision.'}}])
        self.assertEqual(self.h.sbank.get(entry_id).rule['active_version'],2)
        self.assertEqual(self.h.sbank.get(entry_id).rule['candidate_version'],None)
        self.h.close_episode(transfer.task_id,transfer.task_id)
        self.assertEqual(self.h.sbank.get(entry_id).rule['active_version'],2)

    def test_use_and_effect_events_do_not_extend_the_rule_row(self):
        entry_id,_,_=self.seed()
        transfer,_=self.activate(entry_id)
        before=self.h.store.conn.execute('SELECT payload FROM strategic_entries WHERE entry_id=?',(entry_id,)).fetchone()[0]
        effect={'entry_id':entry_id,'execution_id':transfer.execution_id,'verdict':'neutral','note':'Observed use; no comparative benefit claimed.'}
        self.h.induce(effect_attributions=[effect])
        self.h.induce(effect_attributions=[effect])
        events=inspect_versions(self.h,entry_id)['events']
        self.assertEqual(sum(e['kind']=='effect' for e in events),1)
        self.assertEqual(next(e for e in events if e['kind']=='effect')['version'],1)
        after=self.h.store.conn.execute('SELECT payload FROM strategic_entries WHERE entry_id=?',(entry_id,)).fetchone()[0]
        self.assertEqual(before,after)

    def test_index_failure_retries_only_the_affected_unchanged_submission(self):
        task,training=self.solve()
        self.h.close_episode(task['task_id'],task['task_id'])
        raw={**SHORT_RULE,'evidence':[training.execution_id]}
        with patch.object(self.h.index_sync,'sync_entries',return_value={'state':'deferred','reason':'fixture outage'}) as sync:
            first=self.h.induce(relations=[raw])
        entry_id=first['relations'][0]['saved']
        self.assertEqual(sync.call_args.args[0],[entry_id])
        actual=self.h.index_sync.sync_entries
        with patch.object(self.h.index_sync,'sync_entries',wraps=actual) as sync:
            repeated=self.h.induce(relations=[raw])
        self.assertTrue(repeated['relations'][0]['idempotent'])
        self.assertEqual(sync.call_args.args[0],[entry_id])
        with patch.object(self.h.index_sync,'sync_entries',wraps=actual) as sync:
            self.h.induce(relations=[raw])
        self.assertEqual(sync.call_args.args[0],[])
        self.assertEqual(self.h.sbank.count(),1)

    def test_explicit_merge_keeps_source_until_target_passes_and_is_idempotent(self):
        entry_id,training,_=self.seed()
        source=StrategicEntry(entry_id='',strategy_id='old-endpoint',pattern={'predicates':{}},
            claim={'text':SHORT_RULE['expect'],'conditions':{'note':SHORT_RULE['when']},
                   'method':METHOD,'evidence':[]},verification={'state':'verified'})
        self.h.sbank.add(source)
        raw={'entry_id':entry_id,'operation':'merge','merge_entry_ids':[source.entry_id],
             'merge_reason':'Author compared the monotone interval premises, signed endpoint choice and optimum-preservation effect.'}
        merged=self.h.induce(relations=[raw])
        self.assertEqual(merged['relations'][0]['version'],1)
        self.assertEqual(self.h.sbank.get(source.entry_id).rule['merge_pending_into'],entry_id)
        self.assertTrue(is_publishable(self.h.sbank.get(source.entry_id)))
        self.assertTrue(self.h.induce(relations=[raw])['relations'][0]['idempotent'])
        self.activate(entry_id)
        source=self.h.sbank.get(source.entry_id)
        self.assertEqual(source.rule['merged_into'],entry_id)
        self.assertFalse(is_publishable(source))
        self.assertTrue(self.h.induce(relations=[raw])['relations'][0]['idempotent'])
        self.assertEqual(sum(e['kind']=='merge' for e in inspect_versions(self.h,entry_id)['events']),1)

    def test_skip_new_induction_still_registers_existing_targeted_use(self):
        entry_id,_,_=self.seed()
        task,record=self.solve('holdout',use=entry_id)
        result=self.h.close_episode(task['task_id'],task['task_id'],induction_decision={
            'state':'skip','source':'test closeout','reason':'no new mechanism'})
        self.assertEqual(result['rule_validation']['checks'][0]['state'],'passed')
        with patch('or_harness.world_model.induction_material.build_compact_material',side_effect=AssertionError('expansion')):
            material=self.h.induction_material(task_id=task['task_id'])
        self.assertFalse(material['material_assembled'])

    def test_technical_rule_uses_protocol_facts_without_business_acceptance(self):
        task,training=self.solve()
        raw={**SHORT_RULE,'effect_scope':'technical',
             'expect':'Deliver a readable protocol-valid result from this implementation.',
             'evidence':[training.execution_id]}
        entry_id=self.h.induce(relations=[raw])['relations'][0]['saved']
        task,case=self.solve('technical-holdout',do_check=False)
        assessment={'entry_id':entry_id,'version':1,'applicability':'applies',
            'method_fidelity':'faithful','effect_coverage':'complete',
            'source':'fixture auditor inspected captured script and protocol receipt',
            'assessment':'The recorded result supplies all required fields for optimal status.',
            'evidence_refs':['artifacts.submitted_script','artifacts.workspace_result']}
        result=self.h.induce(relations=[{'entry_id':entry_id,'operation':'add_evidence','validation':[
            {'execution_id':case.execution_id,'kind':'transfer','assessment':assessment}]}])
        check=result['relations'][0]['validation'][0]
        self.assertEqual(check['state'],'passed',check)
        self.assertEqual(check['scope']['acceptance']['observation_rule_version'],'wm-r-technical/1')
        self.assertEqual(check['technical_fact']['label'],0)

    def test_cost_comparison_is_selected_by_agent_and_never_certifies_an_advantage(self):
        t,training=self.solve()
        entry=self.h.induce(relations=[{**SHORT_RULE,'effect_scope':'cost','expect':'Use fewer tool_calls.',
            'evidence':[training.execution_id]}])['relations'][0]['saved']
        t,r=self.solve('cost-holdout',use=entry)
        self.h.close_episode(t['task_id'],t['task_id'])
        self.assertNotIn('cost_comparison',check_records(self.h.store,entry)[-1])
        t,base=self.solve('other-input',parameters=(743,159))
        result=self.h.induce(relations=[{'entry_id':entry,'validation':[{'kind':'transfer','execution_id':r.execution_id,
            'comparator_execution_id':base.execution_id,'cost_dimensions':['tool_calls']}]}])
        c=result['relations'][0]['validation'][0]['cost_comparison']
        self.assertFalse(c['same_frozen_input'])
        self.assertIsNone(c['verdict'])
        self.assertIn('execution_conditions',c['baseline'])


    def test_new_storage_codes_roundtrip_and_expired_facts_remain_readable(self):
        entry_id,training,_=self.seed()
        transfer,boundary=self.activate(entry_id)
        raw=self.h.store.loads(self.h.store.conn.execute('SELECT payload FROM executions WHERE execution_id=?',(transfer.execution_id,)).fetchone()[0])
        self.assertEqual(raw['evidence_encoding'],'execution-evidence/2')
        self.assertNotIn('note',raw['execution_features']['execution_config'])
        self.assertEqual(raw['execution_features']['method_receipt']['method_ref'],'method_actual')
        from or_harness.core.schema import ExecutionRecord
        decoded=ExecutionRecord.from_dict(raw)
        self.assertEqual(decoded.cost.measured,transfer.cost.measured)
        self.assertEqual(decoded.method_actual,transfer.method_actual)
        self.assertIsNone(decoded.prediction_snapshot)
        check=next(c for c in check_records(self.h.store,entry_id) if c['kind']=='transfer' and c['state']=='passed')
        self.h.store.conn.execute('DELETE FROM executions WHERE execution_id=?',(transfer.execution_id,))
        fact=self.h.inspect(bank='values',value_ref=check['fact_ref'])['value']
        self.assertEqual(fact['checker']['state'],'passed')
        self.assertEqual(fact['outcome']['objective'],90)
        self.assertTrue(is_publishable(self.h.sbank.get(entry_id)))

    def test_closeout_freezes_rule_check_before_immediate_evidence_eviction(self):
        entry_id,_,_=self.seed()
        task,case=self.solve('immediately-evicted-holdout',use=entry_id)
        with patch.dict(os.environ,{'OR_EVIDENCE_WINDOW_EPISODES':'0','OR_CALIBRATION_WINDOW':'0',
                                    'OR_CALIBRATION_LATE_CHECK_GRACE_DAYS':'0'}):
            result=self.h.close_episode(task['task_id'],task['task_id'])
        self.assertEqual(result['rule_validation']['checks'][0]['state'],'passed')
        self.assertIsNone(self.h.bank.get(case.execution_id))
        check=next(c for c in check_records(self.h.store,entry_id) if c.get('execution_id')==case.execution_id)
        fact=self.h.inspect(bank='values',value_ref=check['fact_ref'])['value']
        self.assertEqual(fact['checker']['state'],'passed')
        self.assertIsInstance(fact['task_text'],str)
        count=len(check_records(self.h.store,entry_id))
        self.h.close_episode(task['task_id'],task['task_id'])
        self.assertEqual(len(check_records(self.h.store,entry_id)),count)
        self.assertEqual(fact['outcome']['objective'],90)
        self.assertFalse(is_publishable(self.h.sbank.get(entry_id)))

    def test_normal_cli_close_read_and_one_short_submission(self):
        work=Path(self.home)/'cli';work.mkdir()
        task={'task_id':'cli-task','family':'assignment','description':'Maximize 2X+5Y over nonnegative integers: X+Y<=1000, X>=2Y+300.'}
        (work/'task.json').write_text(json.dumps(task))
        (work/'solve.py').write_text("import json,os\nrows=[(2*(1000-y)+5*y,1000-y,y) for y in range(234)]\na=max(rows)\nr={'status':'optimal','objective_value':a[0],'variables':{'X':a[1],'Y':a[2]},'method_performed':"+repr(METHOD)+"}\nr['method_performed']['action_id']=os.environ['OR_ACTION_ID']\njson.dump(r,open('result.json','w'))\n")
        def cli(args):
            out=io.StringIO()
            with patch.dict(os.environ,{'OR_EMBEDDING_BACKEND':'local-hashing'}),redirect_stdout(out):
                status=main(['--home',self.home]+args)
            self.assertEqual(status,0,out.getvalue())
            return json.loads(out.getvalue()),out.getvalue().strip()
        result,_=cli(['execute','--task',str(work/'task.json'),'--strategy','endpoint','--solver','python','--code',str(work/'solve.py'),'--workspace',str(work),'--episode','cli-ep'])
        execution_id=result['result']['execution']['execution_id']
        cli(['check-task',execution_id,'--check',json.dumps({'reference_objective':2699,'reference_source':'bench_declared'})])
        cli(['record','--from-staged',execution_id])
        closed,_=cli(['close-episode','--task','cli-task','--episode','cli-ep'])
        self.assertEqual(closed['result']['induction_gate']['state'],'unavailable')
        material,serialized=cli(['induction-material','--task','cli-task'])
        self.assertEqual(material['result']['induction_gate']['source'],'explicit_induction_transition')
        self.assertEqual(len(serialized),material['result']['budget']['chars_used'])
        submitted,_=cli(['induce','--relation',json.dumps({**SHORT_RULE,'evidence':[execution_id]})])
        relation=submitted['result']['relations'][0]
        self.assertEqual(relation['version'],1)
        self.assertFalse(relation['publication']['published'])
        self.assertEqual(submitted['result']['action']['knowledge_delta']['rule_updates'][0]['body_ref'],relation['body_ref'])
