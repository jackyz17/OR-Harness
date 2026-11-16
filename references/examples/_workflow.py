"""Explicit substitute forecasts for runnable examples and legacy audits.

These helpers are example code, not a production authorization bypass. Every
current execution calls the public planner and executor. Normal agents configure
their own provider. Legacy binding samples below model imported historical data.
"""
from or_harness.world_model.provider import WorldModelProvider

LEGACY_CONVENTION={'kind':'solution_quality','metric':'normalized_objective_gap','unit':'1-gap'}


class ExampleProvider(WorldModelProvider):
    name='explicit-example-substitute'
    def predict(self,request,timeout_s=None):
        return {'payload':{'benefit':{**{k:request['benefit_convention'][k] for k in ('kind','metric','unit')},
            'value':.8,'baseline':{'kind':'unknown','note':'A scripted example forecast.'}},
            'capability_gain':{'decision':0,'basis':['This fixture is not a learning recommendation.']}},
            'usage':None,'latency_s':0}


def planned_prediction(h,task,candidate,episode_id=None,*,benefit_convention=None,**kwargs):
    if kwargs:return h.predict_strategy_outcome(task,candidate,episode_id,benefit_convention=benefit_convention,**kwargs)
    convention=benefit_convention or LEGACY_CONVENTION
    limits={'benefit_'+k:convention[k] for k in ('kind','metric','unit') if k in convention}
    plan=h.plan_next(task,episode_id,candidates=[candidate],limits=limits)
    pid=plan.get('selected_prediction_id') or next((c['prediction_id'] for c in plan.get('candidates') or []),None)
    if not pid:raise ValueError('No executable fixture decision: '+str(plan))
    return h.strategy_predictions.get(pid)


def execute_scored(h,task,strategy_id=None,code_path='',workspace='',**kwargs):
    if kwargs.get('prediction_id'):return h.execute(task,strategy_id,code_path,workspace,**kwargs)
    from or_harness.world_model.hplus import mode,configure
    if mode(h)=='auto':configure(h,'manual')  # historical/offline example setup
    original=(h.world_model,h.strategy_predictions.provider)
    h.world_model=h.strategy_predictions.provider=ExampleProvider()
    try:
        plan=h.plan_next(task,kwargs.get('episode_id'),candidates=[{'strategy_id':strategy_id,'solver':kwargs.get('solver'),
            'method':kwargs.get('method') or {}}])
        pid=plan.get('selected_prediction_id')
        if not pid:raise ValueError('No executable fixture decision: '+str(plan))
    finally:h.world_model,h.strategy_predictions.provider=original
    return h.execute(task,strategy_id,code_path,workspace,**{**kwargs,'prediction_id':pid})


def bind_legacy(h,prediction_id,action_id):
    """Read a synthetic imported historical pairing; never authorize solving."""
    prediction=h.strategy_predictions.get(prediction_id);action=h.actions.get(action_id)
    if (action.params or {}).get('prediction_id')!=prediction_id:
        prediction.trace.model_info['execution_decision_required']=False
        prediction.trace.model_info['example_protocol']='imported-legacy-pair'
        h.strategy_predictions._save(prediction)
    return h.bind_strategy_outcome(prediction_id,action_id)


def import_legacy_execution(h,task,strategy_id,code_path,workspace,*,solver,episode_id=None):
    """Construct an explicitly imported old attempt for historical audit examples.

    This is not the normal solving entry. See unified_solving.py for that path.
    """
    from pathlib import Path
    from or_harness.core.artifacts import capture
    pre=h.snapshot(task,episode_id,include_memory=False)
    action=h.actions.begin_action('execute_strategy',str(task['task_id']),episode_id,pre_snapshot=pre,
        params={'strategy_id':strategy_id,'solver':solver,'example_protocol':'historical-import'})
    record=h._run_executor(Path(code_path),Path(workspace),solver=solver,task_id=str(task['task_id']),
        strategy_id=strategy_id,profile=h.profile(task),action_id=action.action_id,method_planned=None)
    record.action_id=action.action_id
    record.task_text_digest=h.capture_task_text(task)
    record.execution_features['artifacts']={'submitted_script':capture(h.home,Path(code_path).read_bytes(),kind='script',source='historical example import')}
    result=Path(workspace)/'result.json'
    if result.exists():record.execution_features['artifacts']['workspace_result']=capture(h.home,result.read_bytes(),kind='result',source='historical example import')
    h.bank.stage_pending(record)
    h.actions.end_action(action.action_id,status='completed',linked_execution_id=record.execution_id,rollup='reference')
    return record
