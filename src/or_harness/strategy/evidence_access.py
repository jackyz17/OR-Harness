"""Resolve live or explicitly retained facts without resurrecting bank rows.

Window expiry is not withdrawal. Explicit validity changes apply to frozen
facts too; immutable values keep their original audit content.
"""
import copy
import time

from or_harness.core.schema import ExecutionRecord
from or_harness.world_model.conditions import get_value,put_value
from or_harness.world_model.induction_gate import meta_read,meta_write


def validity(harness,execution_id):
    record=harness.bank.get(execution_id) or harness.bank.get_pending(execution_id)
    value=meta_read(harness.store,'rule_evidence_validity:'+str(execution_id))
    if value is not None:
        return {**value,'valid':False} if record and record.source=='excluded' else value
    return {'valid':record is None or record.source!='excluded','revision':0}


def retain_record(harness,conn,record,*,make_latest=True):
    """Compatibility name: persist a small audit fact, never another full row."""
    from or_harness.strategy.rule_versions import _facts_ref
    return _facts_ref(harness,conn,record)


def hold_execution(harness,owner,execution_id,*,record=None):
    """Retain full working evidence only while its owning review is pending."""
    record=record or resolve_execution(harness,execution_id)
    if record is None:return None
    action=harness.actions.by_execution(record.execution_id)
    snapshot=harness.get_snapshot(action.pre_snapshot_id) if action and action.pre_snapshot_id else None
    problem=snapshot.problem_state if snapshot else {}
    payload=problem.get('task_input') or problem.get('task_payload') or record.execution_features.get('retained_task_payload')
    text=harness.store.get_task_text(record.task_id,record.task_text_digest) if record.task_text_digest else None
    with harness.store.transaction() as conn:
        ref=put_value(conn,harness.store,'induction_execution',{'record':record.to_dict(),
            'task_payload':copy.deepcopy(payload),'task_text':text or record.execution_features.get('retained_task_text')})
        held=meta_read(harness.store,'induction_hold:'+str(owner)) or {}
        held[record.execution_id]=ref
        meta_write(harness.store,'induction_hold:'+str(owner),held)
    return ref


def held_records(harness,owner):
    return meta_read(harness.store,'induction_hold:'+str(owner)) or {}


def refresh_retained(harness,record):
    """Update only facts already referenced by a review/rule; ordinary checks stay in the bank."""
    owners=[r['key'].split(':',1)[1] for r in harness.store.conn.execute(
        "SELECT key FROM meta WHERE key LIKE 'induction_hold:%'")
        if record.execution_id in (meta_read(harness.store,r['key']) or {})]
    for owner in owners:hold_execution(harness,owner,record.execution_id,record=record)
    if meta_read(harness.store,'rule_fact_latest:'+record.execution_id):
        from or_harness.strategy.rule_versions import _facts_ref
        with harness.store.transaction() as conn:_facts_ref(harness,conn,record)
    # Old full references remain readable but are not recreated by new checks.
    return {'retained_by':owners}


def release_hold(harness,owner):
    with harness.store.transaction() as conn:
        conn.execute('DELETE FROM meta WHERE key=?',('induction_hold:'+str(owner),))
        live=set()
        for row in conn.execute("SELECT value FROM meta WHERE key LIKE 'induction_hold:%'"):
            live.update(harness.store.loads(row['value']).values())
        for row in conn.execute("SELECT value_id FROM condition_values WHERE kind='induction_execution'"):
            if row['value_id'] not in live:conn.execute('DELETE FROM condition_values WHERE value_id=?',(row['value_id'],))


def _from_fact(harness,fact):
    record=ExecutionRecord.from_dict({'execution_id':fact['execution_id'],'task_id':fact['task_id'],
        'strategy_id':fact['strategy_id'],'source':fact.get('source','executed'),'created_at':fact.get('created_at'),
        'profile_snapshot':{'problem_id':fact['task_id'],'family':fact.get('family','unknown')},
        'quality':fact.get('outcome') or {},'cost':fact.get('cost') or {},
        'cost_measured':fact.get('cost_measured') or [],'measurement_scope':fact.get('measurement_scope','attempt'),
        'solver':fact.get('solver') or {},'execution_features':{
            'task_check':fact.get('checker') or {},'artifacts':fact.get('artifacts') or {},
            'task_identity':fact.get('identity'),'retained_task_text':fact.get('task_text'),
            'used_entries':fact.get('used_entries') or {},'execution_config':fact.get('execution_config') or {},
            'cost_provenance':fact.get('cost_provenance') or {d:{'source':'unknown_retained_provenance'} for d in fact.get('cost_measured') or []},
            'audit_fact_only':True}})
    record.method_actual=(fact.get('method') or {}).get('actual')
    if fact.get('task_input_ref'):
        problem=get_value(harness.store,fact['task_input_ref'])
        record.execution_features['retained_task_payload']=copy.deepcopy(problem.get('task_input') or problem.get('task_payload'))
        record.execution_features['retained_task_input_ref']=fact['task_input_ref']
    return record


def _decode(harness,reference):
    raw=get_value(harness.store,reference)
    record=ExecutionRecord.from_dict(raw.get('record') or raw)
    if raw.get('task_payload'):record.execution_features['retained_task_payload']=copy.deepcopy(raw['task_payload'])
    if raw.get('task_text'):record.execution_features['retained_task_text']=raw['task_text']
    return record


def resolve_execution(harness,execution_id,*,include_excluded=False):
    execution_id=str(execution_id)
    if not include_excluded and not validity(harness,execution_id)['valid']:return None
    record=harness.bank.get(execution_id) or harness.bank.get_pending(execution_id)
    if record is None:
        for row in harness.store.conn.execute("SELECT value FROM meta WHERE key LIKE 'induction_hold:%'"):
            reference=harness.store.loads(row['value']).get(execution_id)
            if reference:
                record=_decode(harness,reference);break
    if record is None:
        ref=meta_read(harness.store,'rule_evidence_latest:'+execution_id)
        if ref:record=_decode(harness,ref)  # previous-release full snapshots
    if record is None:
        ref=meta_read(harness.store,'rule_fact_latest:'+execution_id)
        if ref:record=_from_fact(harness,get_value(harness.store,ref))
    if record is None:
        # Compatibility with a pending job frozen by the previous release.
        from or_harness.world_model.hplus import jobs
        for job in jobs(harness):
            if execution_id not in job['execution_ids']:continue
            fact=get_value(harness.store,job['source_fact_ref'])
            if fact.get('execution_ref'):
                record=_decode(harness,fact['execution_ref'])
            elif fact.get('execution_id')==execution_id:
                record=ExecutionRecord.from_dict({'execution_id':execution_id,'task_id':fact['task_id'],
                    'strategy_id':fact['strategy_id'],'source':'executed','created_at':fact.get('created_at'),
                    'profile_snapshot':{'problem_id':fact['task_id'],'family':fact.get('family','unknown')},
                    'quality':fact.get('outcome') or {},'cost':fact.get('cost') or {},
                    'cost_measured':fact.get('cost_measured') or [],'measurement_scope':fact.get('measurement_scope','attempt'),
                    'execution_features':{'task_check':fact.get('checker'),'artifacts':fact.get('artifacts'),
                        'task_identity':fact.get('identity'),'retained_task_text':fact.get('task_text'),
                        'cost_provenance':{d:{'source':'unknown_retained_provenance'} for d in fact.get('cost_measured') or []}}})
                record.method_actual=(fact.get('method') or {}).get('actual')
            break
    if record and not include_excluded and record.source=='excluded':return None
    return record


def set_validity(harness,record,valid,reason):
    with harness.store.transaction() as conn:
        previous=meta_read(harness.store,'rule_evidence_validity:'+record.execution_id) or {'valid':not valid,'revision':0}
        if previous.get('valid')==valid and previous.get('reason')==reason:return previous
        event={'valid':valid,'revision':previous.get('revision',0)+1,'reason':str(reason),
               'at':time.time(),'previous_ref':put_value(conn,harness.store,'rule_validity',previous)}
        meta_write(harness.store,'rule_evidence_validity:'+record.execution_id,event)
        refresh_retained(harness,record)
    return event


def validity_at(harness,execution_id,at):
    value=meta_read(harness.store,'rule_evidence_validity:'+str(execution_id))
    if value is None:
        record=harness.bank.get(execution_id) or harness.bank.get_pending(execution_id)
        correction=record.execution_features.get('correction') or {} if record else {}
        if correction.get('excluded_at',float('inf'))<=at and correction.get('restored_at',float('inf'))>at:
            return {'valid':False,'revision':0,'basis':'legacy_explicit_correction'}
        return {'valid':not(record and record.source=='excluded' and not correction.get('excluded_at')),'revision':0}
    while value.get('at',0)>at and value.get('previous_ref'):
        value=get_value(harness.store,value['previous_ref'])
    return value
