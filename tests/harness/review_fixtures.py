"""Ordinary tasks with raw inputs, real integer scripts and normal API entry points."""
import copy
from pathlib import Path

from helpers import HarnessTestCase
from or_harness.api import ORHarness
from or_harness.core.schema import CostVector
from or_harness.strategy.embedding_index import LocalHashEmbeddingBackend
from or_harness.world_model.provider import WorldModelProvider

RULE={
 'when':'X,Y are nonnegative integers; feasible X at fixed Y is an interval and the objective is monotone in X.',
 'do':'Enumerate feasible Y, derive both X endpoints, select using coefficient sign and objective direction, and compare objectives. If monotonicity fails, disable this reduction and enumerate the full feasible set.',
 'expect':'Preserve the original integer optimum and all stated constraints.'}
METHOD={'name':'signed integer endpoints','steps':[RULE['do']]}


class Provider(WorldModelProvider):
    name='eight-fixes-fixture'
    decision=0
    def __init__(self):self.requests=[]
    def predict(self,request,timeout_s=None):
        self.requests.append(copy.deepcopy(request))
        gain={'decision':self.decision,'claim':'Compare endpoint directions and their applicability boundary.' if self.decision else '',
              'basis':['recorded candidate steps and frozen library'],
              'verification_conditions':[{'condition':'Inspect endpoint direction and check the original integer optimum.'}] if self.decision else []}
        return {'payload':{'benefit':{k:v for k,v in request['benefit_convention'].items() if k in ('kind','metric','unit')}|{'value':0.8},
                           'cost':{'tool_calls':3},'capability_gain':gain},'usage':None,'latency_s':0.001}


def task(name,n=641,c=130,*,maximum=False,boundary=False,wording=None):
    return {'task_id':name,'family':'assignment',
      'description':wording or f'Nonnegative integer X,Y: X+Y<={n}; X>=2Y+{c}. '+
                   ('Minimize (X-3)^2+Y.' if boundary else 'Maximize 2X+5Y.' if maximum else 'Minimize 2X-5Y.'),
      'data':{'capacity':n,'offset':c,'objective_coefficients':[2,5 if maximum else -5],'nonmonotone':boundary},
      'model':{'domains':{'X':'nonnegative_integer','Y':'nonnegative_integer'},
               'constraints':[{'coefficients':{'X':1,'Y':1},'op':'<=','rhs':n},
                              {'coefficients':{'X':1,'Y':-2},'op':'>=','rhs':c}],
               'objective':{'sense':'max' if maximum else 'min','coefficients':{'X':2,'Y':5 if maximum else -5},'nonmonotone':boundary}}}


def assessment(entry_id,version,*,boundary=False,**extra):
    return {'entry_id':str(entry_id),'version':version,'verdict':'supported','applicability':'not_applicable' if boundary else 'applies',
        'response':'disable' if boundary else None,'method_fidelity':'faithful','effect_coverage':'complete',
        'source':'fixture reviewer inspected the captured enumeration against the rule and original inputs',
        'assessment':'Endpoint sign and direction match; nonmonotone input disables reduction. Integer optimum and constraints checked.',
        'evidence_refs':['artifacts.submitted_script','artifacts.workspace_result'],**extra}


class ReviewCase(HarnessTestCase):
    def setUp(self):
        super().setUp()
        self.provider=Provider()
        self.h=ORHarness(home=self.home,world_model=self.provider,embedding=LocalHashEmbeddingBackend())
        self.addCleanup(self.h.close)

    def run_task(self,t,*,decision=0,entry_id=None,version=1,episode=None,close=True,
                 checker=True,check_extra=None,measured=None,provenance='agent_observed',exhaustive=False,unnamed=False):
        t=copy.deepcopy(t);episode=None if unnamed else episode if episode is not None else t['task_id']
        performed=({'name':'full integer enumeration','steps':['Enumerate every feasible integer X,Y and compare all objectives.']} if exhaustive else METHOD)
        self.provider.decision=decision
        p=self.h.predict_strategy_outcome(t,{'strategy_id':'endpoints','solver':'python','method':performed},episode)
        n,c=t['data']['capacity'],t['data']['offset'];boundary=t['data']['nonmonotone'];maximum=t['model']['objective']['sense']=='max'
        ws=Path(self.home)/('case_'+str(len(self.h.actions.query())));ws.mkdir()
        code='import json,os\nvalues=[]\n'
        if boundary or exhaustive:
            expression='(x-3)**2+y' if boundary else '2*x+'+str(5 if maximum else -5)+'*y'
            code+=f'for y in range({n}+1):\n for x in range(2*y+{c},{n}-y+1):\n  values.append(({expression},x,y))\nanswer='+('max' if maximum else 'min')+'(values)\n'
        else:
            code+=f'for y in range({n}+1):\n lo=2*y+{c}\n hi={n}-y\n if lo<=hi:\n  x='+('hi' if maximum else 'lo')+f'\n  values.append((2*x+({5 if maximum else -5})*y,x,y))\nanswer='+('max' if maximum else 'min')+'(values)\n'
        code+="r={'status':'optimal','objective_value':answer[0],'objective_bound':answer[0],'variables':{'X':answer[1],'Y':answer[2],'reduction_used':"+str(not (boundary or exhaustive))+"},'method_performed':"+repr(performed)+"}\nr['method_performed']['action_id']=os.environ['OR_ACTION_ID']\nr['config']={'time_limit':60,'action_id':os.environ['OR_ACTION_ID']}\njson.dump(r,open('result.json','w'))\n"
        (ws/'solve.py').write_text(code)
        r=self.h.execute(t,None,str(ws/'solve.py'),str(ws),episode_id=episode,prediction_id=p.prediction_id,
                         used_entry_ids=[f'{entry_id}@{version}'] if entry_id else None)
        if measured:
            r.cost=CostVector.from_dict(measured)
            r.cost.measured=set(measured)
            r.execution_features['cost_provenance']={d:{'source':provenance,'report_ref':'fixture host counter'} for d in measured}
        self.h.record(r)
        if checker:
            y=r.execution_features['solution_variables']['Y']
            check={'reference_objective':0 if boundary else 2*n+3*((n-c)//3) if maximum else 2*c-((n-c)//3),
                'reference_source':'bench_declared','reference_version':'independent integer calculation',
                'integer':{'variables':['X','Y']},
                'semantic_probe':[{'path':'execution_features.solution_variables.X','min':2*y+c,'max':n-y}]}
            if boundary:check['semantic_probe'].append({'path':'execution_features.solution_variables.reduction_used','equals':False})
            if entry_id:check['rule_assessments']=[assessment(entry_id,version,boundary=boundary)]
            check.update(check_extra or {})
            self.h.check_task_result(r.execution_id,check)
        result=self.h.close_episode(t['task_id'],episode) if close else None
        return p,self.h.bank.get(r.execution_id),result

    def derive(self,*,rule=None,publish=False,review=None):
        _,a,closed=self.run_task(task('A',1000,300,maximum=True),decision=1)
        _,b,_=self.run_task(task('B'))
        job=closed['induction_schedule']['pending_ids'][0]
        material=self.h.induction_material(induction_id=job)
        relation={**(rule or RULE),'evidence':[a.execution_id,b.execution_id]}
        if publish:relation['publication']={'state':'published','reason':'Agent derived endpoint choice and checked the stated integer cases.',
                                           'scope':['checked integer instances and the stated monotonicity premise']}
        if review is True:review={'outcome':'new_rule','comparison':self.comparison(job,substantive=True,basis='derivation')}
        result=self.h.induce(induction_id=job,relations=[relation],review=review)
        return result['relations'][0]['saved'],a,b,job

    def activate(self,entry,version=1):
        _,transfer,closed=self.run_task(task('C',812,200),entry_id=entry,version=version)
        self.assertEqual(transfer.quality['objective'],196)
        self.assertEqual(closed['rule_validation']['checks'][0]['state'],'passed',closed['rule_validation'])
        _,boundary,closed=self.run_task(task('boundary',12,0,boundary=True),entry_id=entry,version=version)
        self.assertEqual(boundary.quality['objective'],0)
        self.h.induce(relations=[{'entry_id':entry,'version':version,'publication':{'state':'published','reason':'Agent reviewed the captured trial and boundary facts.'}}])
        self.assertTrue(self.h.sbank.get(entry).is_published,closed['rule_validation'])
        return transfer,boundary

    def recheck(self,record,**changes):
        check=copy.deepcopy(record.execution_features['task_check']['acceptance_rule']['spec'])
        check.update(changes)
        return self.h.check_task_result(record.execution_id,check)

    def comparison(self,job,**changes):
        from or_harness.world_model.hplus import get_job
        job=get_job(self.h,job)
        return {'sufficient':True,'source':'fixture reviewer compared frozen premises, actions and checked facts',
                'assessment':'The lower endpoint handles minimization; the frozen upper-only method covers maximization.',
                'evidence_refs':[job['source_fact_ref'],job['signal']['source_snapshot_ref']],**changes}

    def empty(self,job,**review):
        self.h.induction_material(induction_id=job)
        return self.h.induce(induction_id=job,notes=['Reviewed captured original evidence.'],
                             review={'outcome':'insufficient_material',**review})
