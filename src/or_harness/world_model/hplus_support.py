"""Source-backed knowledge deltas and time-scoped receipt observations."""
import copy

from or_harness.world_model.conditions import get_value
from or_harness.strategy.evidence_access import resolve_execution,validity_at
from or_harness.strategy.rule_support import body_signature,normalized_text


def baseline_entries(harness,job):
    snapshot=get_value(harness.store,job['signal']['source_snapshot_ref'])
    layers=(snapshot.get('coverage') or {}).get('knowledge_layers') or {}
    return [entry for layer in layers.values() for entry in layer if isinstance(entry,dict)]


def increment(harness,job,update,comparison):
    """Hash differences identify versions, never semantic novelty."""
    body=get_value(harness.store,update['body_ref'])
    entries=baseline_entries(harness,job)
    prior=[]
    for entry in entries:
        rule=entry.get('rule') or {}
        if rule.get('body_ref'):
            prior.append((entry['entry_id'],rule['body_ref'],get_value(harness.store,rule['body_ref'])))
        elif entry.get('claim'):
            # A legacy baseline is still a concrete frozen method. Compare
            # its retained premises/actions/effect; reading it never migrates it.
            claim=entry['claim']
            legacy={'when':(claim.get('conditions') or {}).get('note') or str(entry.get('predicates') or {}),
                'do':'\n'.join((claim.get('method') or {}).get('steps') or entry.get('actions') or []),
                'expect':claim.get('text') or '',
                'predicates':entry.get('predicates') or {},'effect_scope':'legacy'}
            prior.append((entry['entry_id'],job['signal']['source_snapshot_ref'],legacy))
    equivalent=next((ref for _,ref,b in prior if body_signature(body)==body_signature(b)),None)
    if equivalent:
        return {'status':'equivalent','substantive':False,'reason':'format-equivalent conditions, steps and effect in the frozen library',
                'baseline_body_refs':[equivalent]}
    if not update.get('new_version'):
        return {'status':'evidence_only','substantive':False,'reason':'no new rule body version'}
    if not entries:
        return {'status':'new_method','substantive':True,'reason':'first executable rule in an empty frozen library',
                'baseline_body_refs':[]}
    comparison=comparison or {}
    if not isinstance(comparison,dict):raise ValueError('mechanism comparison must be a sourced JSON object')
    refs=set(comparison.get('evidence_refs') or [])
    required={job['source_fact_ref'],job['signal']['source_snapshot_ref']}
    same_entry=[item for item in prior if str(item[0])==str(update['entry_id'])]
    selected=same_entry or prior
    changed=[field for field in ('when','do','expect') if selected and
             all(normalized_text(body.get(field,''))!=normalized_text(b.get(field,'')) for _,_,b in selected)]
    claimed=comparison.get('changed_fields') or []
    sourced=(comparison.get('sufficient') is True and isinstance(comparison.get('source'),str) and comparison['source'].strip()
             and isinstance(comparison.get('assessment'),str) and comparison['assessment'].strip()
             and required<=refs)
    if sourced and comparison.get('substantive') is False:
        return {'status':'covered_by_comparison','substantive':False,'comparison':copy.deepcopy(comparison),
                'baseline_body_refs':[ref for _,ref,_ in selected]}
    if sourced and comparison.get('substantive') is True and isinstance(claimed,list) and claimed and set(claimed)<=set(changed):
        return {'status':'sourced_mechanism_change','substantive':True,'changed_fields':claimed,
                'comparison':copy.deepcopy(comparison),'baseline_body_refs':[ref for _,ref,_ in selected]}
    return {'status':'unknown','substantive':None,'reason':'semantic change needs a sourced mechanism comparison against the frozen library',
            'candidate_changed_fields':changed,'baseline_body_refs':[ref for _,ref,_ in selected]}


def source_at(harness,job,cutoff):
    """Late checks affect future observations; original receipts stay immutable."""
    source_id=job['execution_ids'][0]
    if not validity_at(harness,source_id,cutoff).get('valid'):return None,None
    record=resolve_execution(harness,source_id,include_excluded=True)
    if record is None:return None,None
    report=copy.deepcopy(record.execution_features.get('task_check') or {})
    while report.get('checked_at',0)>cutoff:
        ref=report.get('previous_check_ref')
        if not ref:
            report={};break
        report=copy.deepcopy(get_value(harness.store,ref))
    record.execution_features['task_check']=report
    return record,report


def observation(harness,job,cutoff):
    if job.get('state')!='completed':
        return {'label':None,'verification_state':'unobserved' if job['state']=='skipped' else 'pending'}
    receipt=get_value(harness.store,job['receipt_ref'])
    if receipt['created_at']>cutoff:return {'label':None,'verification_state':'not_yet_recorded'}
    from or_harness.strategy.rule_versions import event_records,check_records
    record,report=source_at(harness,job,cutoff)
    validated=[]
    for update in receipt['rule_updates']:
        # Old receipts' hash-only assertion is readable, but cannot establish
        # a semantic increment for a new observation.
        delta=update.get('increment') or increment(harness,job,update,receipt.get('comparison'))
        if delta.get('substantive') is not True:continue
        events=[e for e in event_records(harness.store,update['entry_id'],update['version'])
                if e['kind']=='publication' and e['at']<=cutoff]
        supported=bool(events and events[-1].get('to')=='active')
        if supported:
            # Explicit withdrawals apply even when an old receipt predates
            # the validity registry; natural eviction never revokes a fact.
            checks=[c for c in check_records(harness.store,update['entry_id'],update['version']) if c['at']<=cutoff]
            supported=any(c['kind']=='transfer' and c['state']=='passed' and
                validity_at(harness,c.get('execution_id'),cutoff).get('valid') for c in checks)
        if record is not None and supported:validated.append({**update,'increment':delta})
    label,state=(1,'independently_validated_increment') if validated else (None,'unvalidated')
    comparison=receipt.get('comparison') or {}
    if not validated and receipt['outcome'] in ('already_covered','no_increment'):
        refs=set(comparison.get('evidence_refs') or [])
        frozen_sources={receipt['source_fact_ref'],receipt.get('baseline_snapshot_ref')}
        from or_harness.world_model.g_signal import _acceptance_support
        checked=bool(record and record.method_actual and _acceptance_support(report or {}))
        covered=receipt['outcome']!='already_covered' or bool(baseline_entries(harness,job))
        no_change=all((u.get('increment') or increment(harness,job,u,comparison)).get('substantive') is False for u in receipt['rule_updates'])
        if (comparison.get('sufficient') is True and isinstance(comparison.get('source'),str) and comparison['source'].strip()
                and isinstance(comparison.get('assessment'),str) and comparison['assessment'].strip()
                and checked and covered and no_change and None not in frozen_sources and frozen_sources<=refs):
            label,state=0,'sourced_comparison_no_increment'
    if record is None:label,state=None,'source_withdrawn_or_unavailable'
    from or_harness.world_model.hplus import _verification
    return {'label':label,'verification_state':state,'validated_rules':validated,
        'hypothesis_outcome':receipt['hypothesis_outcome'],'receipt':receipt,
        'observation_basis':{'source_execution_id':job['execution_ids'][0],
            'checker_action_id':(report or {}).get('action_id'),'checked_at':(report or {}).get('checked_at'),
            'validity':validity_at(harness,job['execution_ids'][0],cutoff)},
        'verification':_verification(record,job['signal']) if record else []}
