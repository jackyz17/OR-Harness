"""Build separate H+/NoH+/equal-budget random libraries, then freeze/test.

The callback runner is usable with real agents. The command-line integer
fixture exercises the protocol with a scripted provider/reviewer, not an
estimate of real H+ discrimination or industrial transfer performance.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import time
from contextlib import closing
from pathlib import Path

from or_harness.api import ORHarness
from or_harness.world_model.conditions import digest
from or_harness.world_model.hplus import configure, jobs, meta_write, receipt_observation
from or_harness.world_model.snapshot_maintenance import fork_memory


def library_signature(harness):
    """Methods/versions/publication, excluding appended usage diagnostics."""
    return digest([{'entry_id':e.entry_id,'rule':{k:v for k,v in (e.rule or {}).items()
            if k in ('body_ref','active_version','candidate_version','state')},
        'claim':e.claim if not e.rule else None,'predicates':e.predicates}
        for e in harness.sbank.list(include_dormant=True)])


def run_comparison(*, initial_home, training_tasks, test_tasks, output,
                   provider_factory, propose, attempt, review,
                   embedding_factory=None, seed=7):
    """Callbacks are shared across arms; the framework alone changes scheduling.

    propose(h, task) returns the usual candidate. attempt(h, task, prediction)
    uses execute/check-task and returns its actual ExecutionRecord. review
    (h, material, job) returns normal induce arguments. It can arrange sourced
    historical regression checks, never invent future labels. Each question
    has one predicted attempt in this minimal runner. General runners can
    call process_induction_jobs after their existing retry loop instead.
    """
    from or_harness.core.schema import task_check_state
    from or_harness.world_model.hplus import get_value
    out=Path(output)
    out.mkdir(parents=True,exist_ok=False)
    ids=[str(t['task_id']) for t in training_tasks]
    if len(ids)!=len(set(ids)):
        raise ValueError('training stream needs distinct task identities')
    if set(ids)&{str(t['task_id']) for t in test_tasks}:
        raise ValueError('frozen tests must be independent of training source tasks')
    summary={'protocol':'hplus-build-freeze/1','seed':seed,'training_task_ids':ids,
             'test_task_ids':[t['task_id'] for t in test_tasks], 'arms':{},
             'random_budget_basis':'realized H+ executed triggers; random selection uses seed, not outcomes',
             'scope':'callback protocol; fixture runs do not establish predictor quality'}
    for branch in ('hplus','nohplus','random'):
        home=out/('train_'+branch)
        fork_memory(initial_home,home)
        with closing(ORHarness(home=str(home),world_model=provider_factory(),
                              embedding=embedding_factory() if embedding_factory else None)) as h:
            if any(j['state']=='pending' for j in jobs(h)):
                raise ValueError('initial library has unfinished induction work')
            with h.store.transaction() as conn:
                conn.execute("DELETE FROM meta WHERE key IN ('induction_mode','hplus_enabled','hplus_random_allocation')")
            initial=library_signature(h)
            configure(h,branch)
            if branch=='random':
                budget=summary['arms']['hplus']['triggers']
                if budget>len(ids):
                    raise ValueError('one-attempt random runner cannot match a larger trigger budget')
                allocation=random.Random(seed).sample(ids,budget)
                with h.store.transaction():meta_write(h.store,'hplus_random_allocation',allocation)
            else:allocation=None
            round_usage=[];training_results=[]
            for task in training_tasks:
                task=copy.deepcopy(task);task_id=str(task['task_id'])
                prediction=h.predict_strategy_outcome(task,propose(h,task),task_id)
                record=attempt(h,task,prediction)
                bound=h.strategy_predictions.get(prediction.prediction_id).trace.model_info.get('bound_action_id')
                action=h.actions.get(bound) if bound else None
                if not action or action.linked_execution_id!=record.execution_id:
                    raise ValueError('attempt callback must bind the actual execution to its prediction')
                if h.bank.get(record.execution_id) is None:h.record(record)
                training_results.append({'task_id':task_id,'prediction_id':prediction.prediction_id,
                    'execution_id':record.execution_id,'objective':record.quality.get('objective'),
                    'task_check':task_check_state(h.bank.get(record.execution_id))})
                h.close_episode(task_id,task_id,terminal_state='completed' if task_check_state(h.bank.get(record.execution_id))=='passed' else 'failed')
                if h.induction_status(task_id=task_id)['pending_ids']:
                    start=time.perf_counter()
                    settled=h.process_induction_jobs(lambda material,job:review(h,material,job),task_id=task_id)
                    elapsed=time.perf_counter()-start
                    if not settled['completed']:
                        raise ValueError('outer builder returned with required induction receipts missing')
                    for result in settled['results']:
                        job_id=result['induction_receipt']['induction_id']
                        measurement={'induction_id':job_id,'action_id':result['action']['action_id'],
                            'wall_s':elapsed,'source':'perf_counter','scope':'material_to_receipt',
                            'tokens':None,'tool_calls':None}
                        with h.store.transaction():meta_write(h.store,'hplus_round_usage:'+job_id,measurement)
                        round_usage.append(measurement)
            items=[j for j in jobs(h) if j['task_id'] in ids]
            receipts=[get_value(h.store,j['receipt_ref']) for j in items if j.get('receipt_ref')]
            observations=[receipt_observation(h,j) for j in items]
            validated={(r['entry_id'],r['version']) for o in observations for r in o.get('validated_rules') or []}
            metrics={'initial_library_digest':initial,'triggers':sum(j['selected'] for j in items),
                'receipts':len(receipts),'empty_reviews':sum(not r['rule_updates'] for r in receipts),
                'induction_wall_s':sum(r['wall_s'] for r in round_usage),'round_usage':round_usage,
                'induction_tokens':None,'induction_tool_calls':None,
                'independently_validated_increments':len(validated),
                'zero_signal_reviews':sum(j['signal']['decision']==0 and j['selected'] for j in items),
                'confirmed_zero_missed_opportunities':sum(j['signal']['decision']==0 and o['label']==1 for j,o in zip(items,observations)),
                'random_allocation':allocation,
                'training_results':training_results,'receipt_details':receipts,
                'observations':[{'prediction_id':j['prediction_id'],'induction_id':j['induction_id'],
                    'decision':j['signal']['decision'],'label':o['label'],'state':o['verification_state'],
                    'input_ref':j['signal'].get('input_ref'),'receipt_ref':j.get('receipt_ref')} for j,o in zip(items,observations)]}
            if not h.induction_status()['completed']:
                raise ValueError('cannot freeze a library with required pending induction')
            configure(h,'frozen')
            frozen=library_signature(h)
            metrics['frozen_library_digest']=frozen
        # Each test question starts from the same quiescent frozen library.
        tests=[]
        for index,task in enumerate(test_tasks):
            test_home=out/(f'test_{branch}_{index}')
            fork_memory(home,test_home)
            with closing(ORHarness(home=str(test_home),world_model=provider_factory(),
                                  embedding=embedding_factory() if embedding_factory else None)) as h:
                prediction=h.predict_strategy_outcome(task,propose(h,task),str(task['task_id']))
                record=attempt(h,task,prediction)
                if h.bank.get(record.execution_id) is None:h.record(record)
                h.close_episode(str(task['task_id']),str(task['task_id']))
                unchanged=library_signature(h)==frozen
                if not unchanged:raise ValueError('test modified the frozen strategy content')
                tests.append({'task_id':task['task_id'],'execution_id':record.execution_id,
                    'task_check':task_check_state(h.bank.get(record.execution_id)),
                    'objective':record.quality.get('objective'),'library_unchanged':unchanged})
        metrics['frozen_tests']=tests
        metrics['frozen_passes']=sum(t['task_check']=='passed' for t in tests)
        summary['arms'][branch]=metrics
    assert len({m['initial_library_digest'] for m in summary['arms'].values()})==1
    assert summary['arms']['random']['triggers']==summary['arms']['hplus']['triggers']
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    return summary


GENERAL_RULE={
    'when':'X,Y are nonnegative integers, feasible X at fixed Y is an interval and the objective is monotone in X.',
    'do':'Enumerate Y, derive both X endpoints, choose from coefficient sign and objective direction, and compare objectives. For a nonmonotone objective disable the reduction and enumerate the full feasible set.',
    'expect':'Preserve the original integer optimum and the stated constraints.',
}


def integer_fixture(output,seed=7,positive_min=False):
    """Small real solves, intentionally scripted H+ false positive/negative."""
    from or_harness.core.schema import StrategicEntry
    from or_harness.strategy.embedding_index import LocalHashEmbeddingBackend
    from or_harness.world_model.provider import WorldModelProvider

    def task(name,n,c,minimum=True,boundary=False):
        return {'task_id':name,'base_task_id':name,'family':'assignment','fixture':{'n':n,'c':c,'min':minimum,'boundary':boundary},
            'text':f'Nonnegative integer X,Y: X+Y<={n}, X>=2Y+{c}. '+
                   ('Minimize (X-3)^2+Y.' if boundary else 'Minimize 2X-5Y.' if minimum else 'Maximize 2X+5Y.')}

    class Provider(WorldModelProvider):
        name='scripted-hplus-protocol-fixture'
        def predict(self,request,timeout_s=None):
            decision=1 if request['candidate']['task_id'] in ('review-max','learn-min') else 0
            return {'payload':{'cost':{'tool_calls':2},
                'benefit':{k:v for k,v in request['benefit_convention'].items() if k in ('kind','metric','unit')}|{'value':0.8},
                'capability_gain':{'decision':decision,
                    'claim':'This run may add a new signed-endpoint boundary.' if decision else '',
                    'basis':['Scripted experimental stance; deliberately includes a covered positive and a missed minimization opportunity.'],
                    'applies_to':['integer interval elimination'],
                    'verification_conditions':[{'condition':'Check endpoint direction and the independent integer optimum.'}] if decision else []}},
                'usage':None,'latency_s':0.001}

    def propose(h,t):
        return {'strategy_id':'endpoint','solver':'python','method':{'name':'integer interval enumeration','steps':[GENERAL_RULE['do']]}}

    def attempt(h,t,prediction=None):
        f=t['fixture'];n,c=f['n'],f['c'];minimum=f['min'];boundary=f['boundary']
        ws=Path(h.store.home)/('work_'+t['task_id']);ws.mkdir()
        code='import json,os\nvalues=[]\n'
        if boundary:
            code+=f'for y in range({n}+1):\n for x in range(2*y+{c},{n}-y+1):\n  values.append(((x-3)**2+y,x,y))\nanswer=min(values)\n'
        else:
            code+=f'for y in range({n}+1):\n lo=2*y+{c}\n hi={n}-y\n if lo<=hi:\n  x='+('lo' if minimum else 'hi')+f'\n  values.append((2*x+({-5 if minimum else 5})*y,x,y))\nanswer='+('min' if minimum else 'max')+'(values)\n'
        code+="r={'status':'optimal','objective_value':answer[0],'objective_bound':answer[0],'variables':{'X':answer[1],'Y':answer[2],'reduction_used':"+str(not boundary)+"},'method_performed':"+repr(propose(h,t)['method'])+"}\nr['method_performed']['action_id']=os.environ['OR_ACTION_ID']\njson.dump(r,open('result.json','w'))\n"
        (ws/'solve.py').write_text(code)
        entry=h.sbank.get('1')
        v=(entry.rule or {}).get('candidate_version') or (entry.rule or {}).get('active_version') if entry else None
        use=['1@'+str(v)] if v and v>=2 else None
        r=h.execute(t,None if prediction else 'endpoint',str(ws/'solve.py'),str(ws),solver=None if prediction else 'python',
                    episode_id=t['task_id'],prediction_id=prediction.prediction_id if prediction else None,
                    method=propose(h,t)['method'],used_entry_ids=use)
        y=r.execution_features['solution_variables']['Y']
        expected=0 if boundary else 2*c-(n-c)//3 if minimum else 2*n+3*((n-c)//3)
        check={'reference_objective':expected,'reference_source':'bench_declared',
            'reference_version':'independently derived integer optimum', 'integer':{'variables':['X','Y']},
            'semantic_probe':[{'path':'execution_features.solution_variables.X','min':2*y+c,'max':n-y}]}
        if boundary:check['semantic_probe'].append({'path':'execution_features.solution_variables.reduction_used','equals':False})
        if use:check['rule_assessments']=[assessment(v,boundary)]
        h.check_task_result(r.execution_id,check)
        return h.bank.get_pending(r.execution_id) or h.bank.get(r.execution_id)

    def assessment(version,boundary=False):
        return {'entry_id':'1','version':version,'applicability':'not_applicable' if boundary else 'applies',
            'response':'disable' if boundary else None,'method_fidelity':'faithful','effect_coverage':'complete',
            'source':'fixture reviewer inspected actual enumeration script, variables and original integer checker',
            'assessment':'Endpoint follows objective direction; nonmonotone input disables elimination. Independent optimum and both constraints checked.',
            'evidence_refs':['artifacts.submitted_script','artifacts.workspace_result']}

    def review(h,material,job):
        record=h.bank.get(job['execution_ids'][0]); t=h._task_payload_for(record,record.task_id)
        original=h.sbank.get('1')
        # Use the captured task text, not a utility/prediction score, to review the mechanism.
        minimum='Minimize 2X-5Y' in str(t.get('text') or t.get('description'))
        if minimum and not original.rule:
            regression=[{'kind':'regression','execution_id':old.execution_id,'assessment':assessment(2)}
                for old in h.bank.all() if old.task_id=='initial-max']
            return {'relations':[{**GENERAL_RULE,'entry_id':'1','operation':'revise','effect_scope':'task_check',
                'evidence':job['execution_ids'],'material_id':material['material_id'],'validation':regression}],
                'review':{'outcome':'revised','hypothesis_outcome':'supported' if job['signal']['decision'] else 'unknown',
                          'comparison':{'sufficient':True,'substantive':True,'changed_fields':['when','do'],
                              'source':'fixture reviewer compared frozen legacy maximum-only method with the captured minimizing execution',
                              'assessment':'The revised premises admit minimization; the actual steps choose the lower endpoint and retain a nonmonotone fallback.',
                              'evidence_refs':[job['source_fact_ref'],job['signal']['source_snapshot_ref']]},
                          'conclusion':'Extend the maximum-only upper endpoint rule to sign-aware minimization and fallback.'}}
        return {'notes':['Compared actual code/checker with the frozen library.'],
            'review':{'outcome':'already_covered' if not minimum else 'no_increment','hypothesis_outcome':'refuted' if job['signal']['decision'] else 'unknown',
                'conclusion':'This review adds no substantive rule; keep trial evidence and independent validation separate.',
                'comparison':{'sufficient':True,'source':'fixture reviewer inspected frozen library and actual script/result',
                    'assessment':'The performed mechanism is covered by the recorded rule or its already frozen candidate.',
                    'evidence_refs':[job['source_fact_ref'],job['signal']['source_snapshot_ref']]}}}

    output=Path(output)
    initial=output.parent/(output.name+'_initial')
    initial.mkdir(parents=True,exist_ok=False)
    with closing(ORHarness(home=str(initial),embedding=LocalHashEmbeddingBackend(),induction_mode='manual')) as h:
        seed_task=task('initial-max',1000,300,minimum=False)
        r=attempt(h,seed_task);h.record(r);h.close_episode('initial-max','initial-max')
        h.sbank.add(StrategicEntry(entry_id='',strategy_id='upper-only',pattern={'predicates':{'family':'assignment'}},
            status='validated',claim={'text':'For maximization with positive X coefficient choose the upper X endpoint.',
                'conditions':{'note':'Maximization, positive X coefficient, computable integer interval.'},
                'method':{'name':'upper endpoint','steps':['Enumerate Y, choose the upper feasible X, compare objectives.']},
                'evidence':[{'execution_id':r.execution_id,'role':'source'}]},verification={'state':'verified'}))
        h.index_sync.sync_entries(['1'])
    training=[task('review-max',1100,340,minimum=False),
              task('learn-min',1000,300) if positive_min else task('missed-min',641,130),
              task('transfer-min',641,130) if positive_min else task('transfer-min',812,200),
              task('nonmonotone',12,0,boundary=True),task('covered-min',1001,302)]
    report=run_comparison(initial_home=str(initial),training_tasks=training,test_tasks=[task('frozen-independent',901,250)],
        output=str(output),provider_factory=Provider,propose=propose,attempt=attempt,review=review,
        embedding_factory=LocalHashEmbeddingBackend,seed=seed)
    report['fixture_source']='real captured Python enumeration/checks; scripted predictor/reviewer; legacy maximum-only initial rule'
    (output/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,help='new output directory; no existing libraries are overwritten')
    parser.add_argument('--seed',type=int,default=7)
    parser.add_argument('--positive-min',action='store_true',help='also test an H+=1 minimum-endpoint revision with independent optimum 90')
    args=parser.parse_args()
    report=integer_fixture(args.output,args.seed,args.positive_min)
    print(json.dumps({mode:{k:v for k,v in data.items() if k in ('triggers','receipts','empty_reviews',
        'independently_validated_increments','zero_signal_reviews','confirmed_zero_missed_opportunities','frozen_passes')}
        for mode,data in report['arms'].items()},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
