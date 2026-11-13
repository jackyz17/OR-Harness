"""Run A/B -> candidate -> held-out C/boundary -> next provider input.

PYTHONPATH=src python docs/validation/run_review_eight_demo.py [--home EMPTY_HOME]
Uses a substitute predictor, actual Python integer enumeration and normal APIs.
"""
import argparse
import copy
import json
import tempfile
from pathlib import Path

from or_harness.api import ORHarness
from or_harness.strategy.embedding_index import LocalHashEmbeddingBackend
from or_harness.world_model.hplus import get_job,receipt_observation
from or_harness.world_model.provider import WorldModelProvider

RULE={
    'when':'Nonnegative integer X,Y; at fixed Y the feasible X set is an interval and the objective is monotone in X.',
    'do':'Enumerate Y, derive both X endpoints, choose by coefficient sign and objective direction, and compare objectives. Disable endpoint reduction for nonmonotone objectives and enumerate all feasible X,Y.',
    'expect':'Preserve the original integer optimum and all stated constraints.'}
METHOD={'name':'sign-aware integer endpoints','steps':[RULE['do']]}


class SubstituteProvider(WorldModelProvider):
    name='review-eight-demo'
    decision=0
    requests=None

    def predict(self,request,timeout_s=None):
        self.requests=self.requests or []
        self.requests.append(copy.deepcopy(request))
        return {'payload':{
            'benefit':{**{k:v for k,v in request['benefit_convention'].items() if k in ('kind','metric','unit')},'value':0.8},
            'cost':{'tool_calls':3},'capability_gain':{
                'decision':self.decision,'claim':'Endpoint choice depends on objective direction; nonmonotonicity disables reduction.',
                'basis':['candidate and frozen rules'],
                'verification_conditions':[{'condition':'Inspect the endpoint choice and independently check the integer optimum.'}] if self.decision else []}},
            'usage':None,'latency_s':0.001}


def problem(name,n,c,*,maximum=False,boundary=False):
    objective='(X-3)**2+Y' if boundary else '2*X+5*Y' if maximum else '2*X-5*Y'
    return {'task_id':name,'family':'assignment',
        'description':f'Nonnegative integer X,Y, X+Y <= {n}, X >= 2*Y+{c}; '+('maximize ' if maximum else 'minimize ')+objective,
        'data':{'capacity':n,'offset':c},
        'model':{'domains':{'X':'nonnegative integer','Y':'nonnegative integer'},
                 'constraints':[f'X+Y <= {n}',f'X >= 2*Y+{c}'],
                 'objective':{'sense':'max' if maximum else 'min','expression':objective}}}


def execute_checked(h,provider,home,name,n,c,expected,*,decision=0,entry=None,maximum=False,boundary=False):
    task=problem(name,n,c,maximum=maximum,boundary=boundary)
    provider.decision=decision
    prediction=h.predict_strategy_outcome(task,{'strategy_id':'endpoints','solver':'python','method':METHOD},name)
    work=Path(home)/('work-'+name);work.mkdir()
    code='import json,os\nvalues=[]\n'
    if boundary:
        code+=f'for y in range({n}+1):\n for x in range(2*y+{c},{n}-y+1):\n  values.append(((x-3)**2+y,x,y))\nanswer=min(values)\n'
    else:
        code+=f'for y in range({n}+1):\n lo=2*y+{c}\n hi={n}-y\n if lo<=hi:\n  x='+('hi' if maximum else 'lo')+f'\n  values.append((2*x+({5 if maximum else -5})*y,x,y))\nanswer='+('max' if maximum else 'min')+'(values)\n'
    code+="r={'status':'optimal','objective_value':answer[0],'objective_bound':answer[0],'variables':{'X':answer[1],'Y':answer[2],'reduction_used':"+str(not boundary)+"},'method_performed':"+repr(METHOD)+"}\nr['method_performed']['action_id']=os.environ['OR_ACTION_ID']\nr['config']={'time_limit':60,'action_id':os.environ['OR_ACTION_ID']}\njson.dump(r,open('result.json','w'))\n"
    (work/'solve.py').write_text(code)
    record=h.execute(task,None,str(work/'solve.py'),str(work),episode_id=name,prediction_id=prediction.prediction_id,
                     used_entry_ids=[entry+'@1'] if entry else None)
    h.record(record)
    y=record.execution_features['solution_variables']['Y']
    check={'reference_objective':expected,'reference_source':'bench_declared','reference_version':'review integer instances/1',
        'integer':{'variables':['X','Y']},'semantic_probe':[
            {'path':'execution_features.solution_variables.X','min':2*y+c,'max':n-y},
            {'path':'execution_features.solution_variables.Y','min':0}]}
    if boundary:check['semantic_probe'].append({'path':'execution_features.solution_variables.reduction_used','equals':False})
    if entry:
        check['rule_assessments']=[{'entry_id':entry,'version':1,
            'applicability':'not_applicable' if boundary else 'applies','response':'disable' if boundary else None,
            'method_fidelity':'faithful','effect_coverage':'complete',
            'source':'demo fixture reviewer inspected this generated script and the original integer checker',
            'assessment':'Signed endpoint choice is implemented; the nonmonotone case enumerates the full feasible set. Original constraints and optimum are checked.',
            'evidence_refs':['artifacts.submitted_script','artifacts.workspace_result']}]
    checked=h.check_task_result(record.execution_id,check)
    closed=h.close_episode(name,name)
    assert checked['state']=='passed'
    assert record.quality['objective']==expected
    return prediction,record,closed


def run(home):
    provider=SubstituteProvider()
    h=ORHarness(home=home,world_model=provider,embedding=LocalHashEmbeddingBackend())
    try:
        if h.bank.count() or h.sbank.count():raise ValueError('demo needs an empty isolated memory home')
        pa,a,closed=execute_checked(h,provider,home,'A',1000,300,2699,decision=1,maximum=True)
        pb,b,_=execute_checked(h,provider,home,'B',641,130,90)
        job=closed['induction_schedule']['pending_ids'][0]
        material=h.induction_material(induction_id=job,execution_ids=[b.execution_id],related_top_k=1)
        submitted=h.induce(induction_id=job,relations=[{**RULE,'evidence':[a.execution_id,b.execution_id]}])
        entry=submitted['relations'][0]['saved']
        pc,c,cc=execute_checked(h,provider,home,'C',812,200,196,entry=entry)
        pd,boundary,bc=execute_checked(h,provider,home,'boundary',12,0,0,entry=entry,boundary=True)
        observed=receipt_observation(h,get_job(h,job))
        assert h.sbank.get(entry).is_published and observed['label']==1
        provider.decision=0
        h.predict_strategy_outcome(problem('next',920,220),{'strategy_id':'endpoints','solver':'python','method':METHOD},'next')
        return {'provider':'substitute; validates the chain, not predictive discrimination',
            'objectives':{'A':a.quality['objective'],'B':b.quality['objective'],'C':c.quality['objective'],'boundary':boundary.quality['objective']},
            'induction_id':job,'prediction_id':pa.prediction_id,'entry_id':entry,'active_version':h.sbank.get(entry).rule['active_version'],
            'source_execution_ids':[a.execution_id,b.execution_id],
            'independent_execution_ids':[c.execution_id,boundary.execution_id],
            'material_id':material['material_id'],'induction_action_id':submitted['induction_receipt']['induction_action_id'],
            'receipt_ref':get_job(h,job)['receipt_ref'],'hplus_label':observed['label'],
            'next_provider_request_fragment':{'prediction_context':{'hplus_feedback':provider.requests[-1]['prediction_context']['hplus_feedback']}}}
    finally:h.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--home',help='empty isolated memory home; otherwise use a temporary directory')
    args=parser.parse_args()
    if args.home:
        result=run(args.home)
    else:
        with tempfile.TemporaryDirectory(prefix='or-review-eight-') as home:result=run(home)
    print(json.dumps(result,ensure_ascii=False,indent=2))
