"""Explicit migration helpers for tests outside the U-decision boundary.

Current executions use the real planner/claim/executor. A separate named legacy
import builds historical records for retention-reader regressions only.
"""
from pathlib import Path

from or_harness.world_model.provider import WorldModelProvider


def planned_prediction(h,task,candidate,episode_id=None,*,benefit_convention=None,limits=None,**kwargs):
    options=dict(limits or {})
    if benefit_convention:
        options.update({'benefit_'+key:benefit_convention[key] for key in ('kind','metric','unit') if key in benefit_convention})
    if kwargs:raise ValueError('test decision needs explicit context migration: '+str(sorted(kwargs)))
    plan=h.plan_next(task,episode_id,candidates=[candidate],limits=options)
    pid=plan.get('selected_prediction_id') or next((r['prediction_id'] for r in plan.get('candidates') or []),None)
    if pid is None:raise ValueError('fixture could not obtain a prediction: '+str(plan))
    return h.strategy_predictions.get(pid)


class ExecutionFixtureProvider(WorldModelProvider):
    name='scored-execution-test-fixture'
    def predict(self,request,timeout_s=None):
        return {'payload':{'benefit':{**{k:request['benefit_convention'][k] for k in ('kind','metric','unit')},
            'value':.8,'baseline':{'kind':'unknown','note':'Fixture tests execution behavior, not forecast discrimination.'}},
            'capability_gain':{'decision':0,'basis':['No new induction in this execution fixture.']}},
            'usage':None,'latency_s':0}


def execute_scored(h,task,strategy_id=None,code_path='',workspace='',**kwargs):
    """Explicitly plan an execution fixture; preserve the test's own provider."""
    if kwargs.get('prediction_id') or not strategy_id or not kwargs.get('solver'):
        return h.execute(task,strategy_id,code_path,workspace,**kwargs)
    from or_harness.world_model.hplus import mode,configure
    if mode(h)=='auto':configure(h,'manual')  # legacy offline fixtures keep their induction protocol
    original=(h.world_model,h.strategy_predictions.provider)
    provider=ExecutionFixtureProvider()
    h.world_model=h.strategy_predictions.provider=provider
    try:
        p=planned_prediction(h,task,{'action_type':'execute_strategy','strategy_id':strategy_id,
            'solver':kwargs['solver'],'method':kwargs.get('method') or {}},kwargs.get('episode_id'))
    finally:h.world_model,h.strategy_predictions.provider=original
    return h.execute(task,strategy_id,code_path,workspace,**{**kwargs,'prediction_id':p.prediction_id})


def import_legacy_execution(h,task,strategy_id,code_path,workspace,*,solver,episode_id=None,prediction_id=None,method_planned=None):
    """Import a pre-decision-protocol execution fixture with original facts.

    Only historical-reader tests use this path; no production entry exposes it.
    """
    from or_harness.core.artifacts import capture
    pre=h.snapshot(task,episode_id,include_memory=False)
    action=h.actions.begin_action('execute_strategy',str(task['task_id']),episode_id,pre_snapshot=pre,
        params={'strategy_id':strategy_id,'solver':solver,'fixture_protocol':'legacy-import','method':method_planned or {}})
    record=h._run_executor(Path(code_path),Path(workspace),solver=solver,task_id=str(task['task_id']),
        strategy_id=strategy_id,profile=h.profile(task),action_id=action.action_id,method_planned=method_planned)
    record.action_id=action.action_id
    record.task_text_digest=h.capture_task_text(task)
    record.execution_features['artifacts']={'submitted_script':capture(h.home,Path(code_path).read_bytes(),kind='script',source='legacy fixture import')}
    result=Path(workspace)/'result.json'
    if result.exists():record.execution_features['artifacts']['workspace_result']=capture(h.home,result.read_bytes(),kind='result',source='legacy fixture import')
    h.bank.stage_pending(record)
    h.actions.end_action(action.action_id,status='completed',linked_execution_id=record.execution_id,rollup='reference')
    if prediction_id:bind_legacy(h,prediction_id,action.action_id)
    return record


def bind_legacy(h,prediction_id,action_id):
    """Replay an imported old binding in calibration/attribution reader tests.

    Current scored executions keep their normal guard. Historical mismatched
    forecasts are marked as old protocol fixtures, never execution permission.
    """
    p=h.strategy_predictions.get(prediction_id);a=h.actions.get(action_id)
    if not a or (a.params or {}).get('prediction_id')!=prediction_id or not (a.params or {}).get('scored_adoption'):
        p.trace.model_info['execution_decision_required']=False
        p.trace.model_info['fixture_protocol']='legacy-binding-import'
        h.strategy_predictions._save(p)
    return h.bind_strategy_outcome(prediction_id,action_id)


def scored_cli_argv(home, argv):
    """Prepare a real scored P for old offline CLI setup, then test CLI execution.

    Call explicitly only from seed-data tests; negative authorization tests keep
    their original arguments. The normal CLI end-to-end demo uses its provider.
    """
    import json
    from or_harness.api import ORHarness
    from or_harness.world_model.hplus import configure
    args=list(argv)
    if not args or args[0]!='execute' or '--prediction' in args:return args
    def value(flag,default=None):
        return args[args.index(flag)+1] if flag in args else default
    raw=value('--task'); path=Path(raw)
    task=json.loads(path.read_text() if not raw.lstrip().startswith('{') else raw)
    harness=ORHarness(home=home,world_model=ExecutionFixtureProvider())
    try:
        configure(harness,'manual')  # offline fixture, no binary H+ recommendation
        plan=harness.plan_next(task,value('--episode'),candidates=[{'strategy_id':value('--strategy'),
            'solver':value('--solver'),'config':json.loads(value('--config','{}'))}])
        pid=plan['selected_prediction_id']
        if not pid:raise ValueError('offline setup has no scored decision')
    finally:harness.close()
    return args+['--prediction',pid]


def prediction_anchor(h,prediction):
    decision_id=prediction.trace.model_info.get('decision_action_id')
    return h.actions.get(decision_id).started_at if decision_id else prediction.trace.created_at
