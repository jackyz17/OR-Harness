"""Small real integer solves: derive, freeze, transfer, test a boundary.

PYTHONPATH=src python references/examples/versioned_induction.py
Optional --report FILE writes auditable references. --mass adds 29 long
historical rules; --baseline-root CHECKOUT measures that checkout's material
builder and CLI envelope against exactly the same isolated fixture.
Hash embeddings test the plumbing, not semantic retrieval quality.
"""
from __future__ import annotations

import argparse
import importlib.util
import io
import json
import os
import sys
import tempfile
from contextlib import closing, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

from or_harness.api import ORHarness
from or_harness.cli import main as cli_main
from or_harness.core.schema import StrategicEntry
from or_harness.strategy.embedding_index import LocalHashEmbeddingBackend
from or_harness.strategy.rule_versions import inspect_versions
from or_harness.world_model.conditions import get_value

RULE = {
    'when':'X,Y are nonnegative integers; feasible X at fixed Y is an interval and the objective is monotone in X.',
    'do':'Enumerate feasible Y, derive both X endpoints, choose using coefficient sign and objective direction, and compare objectives. Disable this reduction for a nonmonotone objective and enumerate the full feasible set.',
    'expect':'Preserve the original integer optimum and all stated constraints.',
}


def solve(h, name, *, use=None, minimum=False, boundary=False, wrong=False, rule=None):
    n,c = (12,0) if boundary else (641,130) if minimum else (1000,300)
    description = (f'Nonnegative integer X,Y. X+Y<={n}; X>=2Y+{c}. ' +
                   ('Minimize (X-3)^2+Y.' if boundary else 'Minimize 2X-5Y.' if minimum else 'Maximize 2X+5Y.'))
    task = {'task_id':name,'base_task_id':name,'family':'assignment','text':description}
    workspace = Path(h.store.home) / name
    workspace.mkdir()
    body = rule or RULE
    method = {'name':'integer endpoints','steps':[body['do']]}
    code = 'import json,os\nvalues=[]\n'
    if boundary:
        code += 'for y in range(13):\n for x in range(2*y,13-y):\n  values.append(((x-3)**2+y,x,y))\nanswer=min(values)\n'
    else:
        code += f'for y in range({n}+1):\n lo=2*y+{c}\n hi={n}-y\n if lo<=hi:\n  x=' + ('lo' if minimum and not wrong else 'hi')
        code += f'\n  values.append((2*x+({-5 if minimum else 5})*y,x,y))\nanswer=' + ('min' if minimum else 'max') + '(values)\n'
    code += "result={'status':'optimal','objective_value':answer[0],'objective_bound':answer[0],'variables':{'X':answer[1],'Y':answer[2],'reduction_used':" + str(not boundary) + "},'method_performed':" + repr(method) + '}\n'
    code += "result['method_performed']['action_id']=os.environ['OR_ACTION_ID']\njson.dump(result,open('result.json','w'))\n"
    script = workspace / 'solve.py'
    script.write_text(code)
    record = h.execute(task,'endpoint',str(script),str(workspace),solver='python',episode_id=name,
                       method=method,used_entry_ids=[str(use)+'@1'] if use else None)
    x,y = record.execution_features['solution_variables']['X'],record.execution_features['solution_variables']['Y']
    check = {'reference_objective':0 if boundary else 90 if minimum else 2699,
             'reference_source':'bench_declared','reference_version':'independent signed-endpoint arithmetic',
             'integer':{'variables':['X','Y']},
             'semantic_probe':[{'path':'execution_features.solution_variables.X','min':2*y+c,'max':n-y}]}
    if boundary:
        check['semantic_probe'].append({'path':'execution_features.solution_variables.reduction_used','equals':False})
    if use:
        check['rule_assessments']=[{'entry_id':str(use),'version':1,
            'applicability':'not_applicable' if boundary else 'applies','response':'disable' if boundary else None,
            'method_fidelity':'faithful','effect_coverage':'complete','failure_source':'rule' if wrong else None,
            'source':'example auditor inspected the captured enumeration script and variables',
            'assessment':('The upper-only rule was implemented faithfully; the independent objective check exposes its wrong endpoint.' if wrong else
                          'The objective sign selects the endpoint; nonmonotone input disables reduction. Independent optimum and both constraints are checked.'),
            'evidence_refs':['artifacts.submitted_script','artifacts.workspace_result']}]
    h.check_task_result(record.execution_id,check)
    h.record(record)
    closed=h.close_episode(name,name)
    return h.bank.get(record.execution_id),closed


def load_module(path, name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def measure_material(h, training, baseline_root=None):
    for index in range(29):
        prose=(f'historical endpoint mechanism {index}: compare integer interval bounds and objective direction. ' * 350)
        h.sbank.add(StrategicEntry(entry_id='',strategy_id='endpoint',pattern={'predicates':{'family':'assignment'}},
            claim={'text':prose,'method':{'name':'endpoint','steps':[RULE['do']]},
                   'evidence':[{'execution_id':training.execution_id,'role':'source'}]},
            verification={'state':'unverified','checks':['historical audit '*250]}))
    out=io.StringIO()
    with patch.dict(os.environ,{'OR_EMBEDDING_BACKEND':'local-hashing'}),redirect_stdout(out):
        assert cli_main(['--home',str(h.store.home),'induction-material','--task',training.task_id])==0
    after_text=out.getvalue().strip()
    after=json.loads(after_text)
    assert len(after_text)==after['result']['budget']['chars_used']<=32000
    assert after['result']['material'] and after['result']['related_history']['executions'] and after['result']['existing_knowledge']
    assert any(row['entry_id']=='1' for row in after['result']['existing_knowledge']), 'the relevant active rule must survive long same-family neighbors'
    result={'fixture':'29 constructed long historical same-family entries; not mdsc_sc003 data',
            'after':{'chars':len(after_text),'utf8_bytes':len(after_text.encode()),'reported_chars_used':after['result']['budget']['chars_used'],
                     'n_current':len(after['result']['material']),'n_related':len(after['result']['related_history']['executions']),
                     'n_rules':len(after['result']['existing_knowledge']),'omitted_rules':after['result']['budget']['omitted_knowledge_count']},
            'default_fragment':{'induction_gate':after['result']['induction_gate'],'comparison_table':after['result']['comparison_table'],
                                'existing_knowledge':after['result']['existing_knowledge'][:1]}}
    if baseline_root:
        root=Path(baseline_root)/'src/or_harness'
        old_material=load_module(root/'world_model/maintenance.py','_or_material_baseline')
        old_cli=load_module(root/'cli.py','_or_cli_baseline')
        proxy=SimpleNamespace(induction_material=lambda **kw:old_material.build_induction_material(h,**kw),close=lambda:None)
        args=old_cli.build_parser().parse_args(['--home',str(h.store.home),'induction-material','--task',training.task_id])
        out=io.StringIO()
        with patch.object(old_cli,'_harness',return_value=proxy),redirect_stdout(out):
            assert old_cli.cmd_induction_material(args)==0
        before_text=out.getvalue().strip()
        before=json.loads(before_text)
        result['before']={'chars':len(before_text),'utf8_bytes':len(before_text.encode()),
                          'reported_chars_used':before['result']['budget']['chars_used'],
                          'knowledge_chars':len(json.dumps(before['result']['existing_knowledge'],ensure_ascii=False,separators=(',',':'))),
                          'source':'baseline maintenance builder and CLI summary, same isolated fixture, current compatible bank readers'}
    return result


def run(home, *, mass=False, baseline_root=None):
    with closing(ORHarness(home=str(home),embedding=LocalHashEmbeddingBackend())) as h:
        training,_=solve(h,'training')
        material=h.induction_material(task_id='training')
        saved=h.induce(relations=[{**RULE,'evidence':[training.execution_id],'material_id':material['material_id']}])
        entry_id=saved['relations'][0]['saved']
        assert not saved['relations'][0]['publication']['published']
        transfer,transfer_close=solve(h,'independent',use=entry_id,minimum=True)
        assert not h.sbank.get(entry_id).is_published
        boundary,boundary_close=solve(h,'nonmonotone',use=entry_id,boundary=True)
        assert h.sbank.get(entry_id).is_published
        upper={**RULE,'do':'Enumerate Y and always choose the upper X endpoint; disable on a nonmonotone objective.'}
        wrong_id=h.induce(relations=[{**upper,'evidence':[training.execution_id]}])['relations'][0]['saved']
        wrong,wrong_close=solve(h,'wrong-upper',use=wrong_id,minimum=True,wrong=True,rule=upper)
        assert [r.quality['objective'] for r in (training,transfer,wrong)]==[2699,90,92]
        assert wrong_close['rule_validation']['checks'][0]['state']=='refuted'
        audit=inspect_versions(h,entry_id)
        passed=next(c for c in audit['checks'] if c['kind']=='transfer' and c['state']=='passed')
        report={'source':'isolated real Python integer enumerations plus deterministic hash retrieval; no live model prediction',
                'home':str(home),'rule':{'entry_id':entry_id,'version':1,**RULE},
                'objectives':{'training':2699,'independent_min':90,'wrong_upper_min':92,'nonmonotone':0},
                'actual_execution_ids':[r.execution_id for r in (training,transfer,boundary,wrong)],
                'transfer_check':passed,'transfer_fact':get_value(h.store,passed['fact_ref']),
                'validation':{'transfer':transfer_close['rule_validation'],'boundary':boundary_close['rule_validation'],
                              'wrong_upper':wrong_close['rule_validation']},
                'publication':h.sbank.get(entry_id).rule}
        if mass:
            report['material_size']=measure_material(h,training,baseline_root)
        return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--home',default=None)
    parser.add_argument('--report')
    parser.add_argument('--mass',action='store_true')
    parser.add_argument('--baseline-root')
    args=parser.parse_args()
    home=Path(args.home) if args.home else Path(tempfile.mkdtemp(prefix='or_induction_demo_'))
    if home.exists() and any(home.iterdir()):
        parser.error('--home must be a new/empty isolated directory')
    home.mkdir(parents=True,exist_ok=True)
    report=run(home,mass=args.mass,baseline_root=args.baseline_root)
    if args.report:
        Path(args.report).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'home':str(home),'rule':report['rule'],'objectives':report['objectives'],
                      'material_size':report.get('material_size')},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
