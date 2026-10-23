"""Isolated storage/request benchmark; emits JSON, calls no model or solver.

Run with PYTHONPATH pointing to either the current src or the pre-change src.
A canonical provider comparison excludes IDs/times, the declared context
schema version and the explicitly changed calibration diagnostic projection.
"""
import copy
import json
import os
import tempfile
from unittest.mock import patch
from or_harness.api import ORHarness
from or_harness.core.schema import CostVector, ExecutionRecord, StrategicEntry
from or_harness.world_model.prediction import ActionSpec
from or_harness.world_model.contracts import CandidateRef
from or_harness.world_model.provider import NotConfiguredProvider
from or_harness.world_model.state import task_text_digest
from or_harness.world_model.strategy_prediction import build_strategy_outcome_request

task={"task_id":"fixture", "family":"routing", "description":"Minimize cost with capacity 10.",
      "spec":{"n_vars":10,"n_constraints":5,"n_int_vars":10},
      "annotations":{"coupling":{"resource_coupling":0.3,"temporal_coupling":0.1,"route_complexity":0.2}}}
def canonical(value):
    if isinstance(value,dict):
        return {k:canonical(v) for k,v in value.items() if k not in {
            "context_id","snapshot_id","context_version","created_at","snapshot_at","as_of","last_consulted_at",
            "validity_corrections","validity_correction_summary","request_size"}}
    if isinstance(value,list): return [canonical(v) for v in value]
    if isinstance(value,str) and (value.startswith('bs_') or value.startswith('ctx_')): return '<id>'
    return value

result=[]
for count in (1,10,100):
    with tempfile.TemporaryDirectory() as home, patch.dict(os.environ,{"OR_EMBEDDING_BACKEND":"none"}), patch('or_harness.api.available_families',return_value={}):
        h=ORHarness(home=home,world_model=NotConfiguredProvider())
        profile=h.profile(task)
        for i in range(count):
            h.bank.append(ExecutionRecord(execution_id=f'ex_fixed_{i:03d}',task_id=f'old_{i:03d}',
                strategy_id='candidate' if i==0 else f'unrelated_{i:03d}', profile_snapshot=profile,
                quality={'status':'optimal','gap':0,'feasible':True},created_at=1,
                cost=CostVector(llm_tokens=10,measured={'llm_tokens'})))
        h.sbank.add(StrategicEntry(entry_id='',strategy_id='candidate',
            pattern={'predicates':{'family':'routing'}},expected_quality_hat=0.8,
            quality_interval=(0.6,1.0),quality_estimated=True,support_n=2,
            expected_cost_hat=CostVector(llm_tokens=10,measured={'llm_tokens'}),
            created_at=1,verification={'state':'verified','claim':'fixture'}))
        recall=h.recall(task,candidates=['candidate'],top=1)
        spec=ActionSpec('execute_strategy','fixture',strategy_id='candidate')
        snap=h.snapshot(task,'e')
        ctx=h.build_prediction_context(task,'e',snapshot=snap,historical=False,candidates=[spec],recall_result=recall)
        request=build_strategy_outcome_request(ctx,CandidateRef(action_type='execute_strategy',strategy_id='candidate',task_id='fixture',episode_id='e'))
        def sizes():
            out={}
            for row in h.store.conn.execute("SELECT name FROM sqlite_master WHERE type='table'"):
                name=row[0]
                if name not in ('belief_snapshots','prediction_contexts','frozen_conditions','runtime_inputs','condition_values','knowledge_availability','executions','action_records'):continue
                data=h.store.conn.execute(f'SELECT count(*),coalesce(sum(length(cast(payload as blob))),0) FROM {name}').fetchone()
                out[name]={'rows':data[0],'payload_bytes':data[1]}
            row=h.store.conn.execute("SELECT count(*),coalesce(sum(length(cast(value as blob))),0) FROM meta WHERE key LIKE 'knowledge_version|%'").fetchone()
            out['knowledge_versions']={'rows':row[0],'payload_bytes':row[1]}
            return out
        before=sizes()
        for i in range(5): h.report_action('model',task,'e',outcome={'step':i})
        after=sizes()
        try:
            from or_harness.world_model.conditions import release_context
        except ImportError:
            pass  # baseline has no consumable runtime input
        else:
            release_context(h.store,ctx.context_id)
        consumed=sizes()
        result.append({'strategies':count,'before':before,'after_5_actions':after,'after_consumption':consumed,
            'request_chars':request['request_size']['chars'],
            'context_payload_bytes':len(h.store.dumps(ctx.to_dict()).encode()),
            'candidate_statistics':canonical(snap.coverage['cell_statistics']['candidate']),
            'cell_evidence':{k:{kk:vv for kk,vv in v.items() if kk!='task_ids'} for k,v in ctx.cell_evidence.items()},
            'knowledge_targets':canonical(ctx.knowledge_targets),
            'provider_semantics':canonical(ctx.provider_view())})
        h.close()
print(json.dumps(result,sort_keys=True,indent=2))
