"""Agent-authored learning conclusions with valid, time-scoped evidence.

Publication, recall coverage and body hashes are not learning labels.
"""
import copy
import time

from or_harness.world_model.conditions import get_value,put_value
from or_harness.strategy.evidence_access import resolve_execution,validity_at

OBSERVATION_VERSION='wm-hplus-review/2'


def increment(harness,job,update,comparison):
    comparison=copy.deepcopy(comparison or {})
    if not isinstance(comparison,dict):raise ValueError('learning comparison must be a JSON object')
    reference=put_value(harness.store.conn,harness.store,'hplus_comparison',comparison)
    declared=comparison.get('substantive')
    return {'substantive':declared if type(declared) is bool else None,
            'status':'agent_review','comparison_ref':reference,
            'body_changed':update.get('new_version') is True}


def _report_at(harness,record,cutoff):
    report=copy.deepcopy(record.execution_features.get('task_check') or {})
    while report.get('checked_at',0)>cutoff:
        previous=report.get('previous_check_ref')
        if not previous:return {}
        report=copy.deepcopy(get_value(harness.store,previous))
    return report


def source_at(harness,job,cutoff):
    eid=job['execution_ids'][0]
    if not validity_at(harness,eid,cutoff).get('valid'):return None,None
    record=resolve_execution(harness,eid,include_excluded=True)
    return (record,_report_at(harness,record,cutoff)) if record else (None,None)


def _reference(harness,job,ref,cutoff):
    """Resolve locatable support; context references alone prove no result."""
    if not isinstance(ref,str) or not ref:raise ValueError('learning evidence needs nonempty references')
    raw={};eid=None;artifact=None
    if ref.startswith('artifacts.') or ref.startswith('task_check.'):
        eid=job['execution_ids'][0];artifact=ref
    elif ':' not in ref or ref.startswith('execution:'):
        eid=ref.split(':',1)[1] if ref.startswith('execution:') else ref
    else:
        try:raw=get_value(harness.store,ref)
        except Exception as exc:raise ValueError('unknown learning evidence reference '+ref) from exc
        eid=raw.get('execution_id')
        if eid is None and raw.get('record'):eid=raw['record'].get('execution_id')
    if eid:
        record=resolve_execution(harness,eid,include_excluded=True)
        if record is None:raise ValueError('learning evidence execution is unavailable: '+str(eid))
        report=_report_at(harness,record,cutoff)
        valid=record.created_at<=cutoff and validity_at(harness,eid,cutoff).get('valid',False)
        has_artifact=bool((record.execution_features.get('artifacts') or {}).get(artifact.split('.',1)[1],{}).get('sha256')) if artifact and artifact.startswith('artifacts.') else bool(record.execution_features.get('artifacts'))
        return {'reference':ref,'execution_id':eid,'valid':valid,'actual_execution':True,
                'checked':bool(report.get('checks')) and report.get('state') in ('passed','failed'),
                'state':report.get('state'),
                'artifact':has_artifact,'checker_action_id':report.get('action_id'),
                'checked_at':report.get('checked_at'),'scope':copy.deepcopy(report.get('scope')),
                'acceptance_rule':((report.get('acceptance_rule') or {}).get('fingerprint'))}
    return {'reference':ref,'valid':True,'actual_execution':False,'checked':False,'context_only':True}


def supported_comparison(harness,job,comparison,cutoff):
    if not isinstance(comparison,dict):return False,{'reason':'no agent learning assessment'}
    refs=comparison.get('evidence_refs') or []
    if not isinstance(refs,list):raise ValueError('comparison.evidence_refs must be a list')
    resolved=[_reference(harness,job,r,cutoff) for r in refs]
    named=bool(isinstance(comparison.get('source'),str) and comparison['source'].strip() and
               isinstance(comparison.get('assessment'),str) and comparison['assessment'].strip())
    basis=comparison.get('basis','recorded_checks')
    actual=[r for r in resolved if r.get('actual_execution')]
    checked=any(r['checked'] for r in actual)
    if basis=='derivation':checked=bool(actual)  # the agent's retained derivation names the actual input facts
    if basis=='artifact_inspection':checked=any(r.get('artifact') for r in actual)
    valid=bool(named and comparison.get('sufficient') is True and resolved and actual and
               all(r['valid'] for r in resolved) and checked)
    return valid,{'basis':basis,'references':resolved,'reason':None if valid else 'insufficient, not yet checked, or excluded supporting facts'}


def observation(harness,job,cutoff):
    if job.get('state')!='completed':
        return {'label':None,'verification_state':'unobserved' if job['state']=='skipped' else 'pending'}
    receipt=get_value(harness.store,job['receipt_ref'])
    if receipt['created_at']>cutoff:return {'label':None,'verification_state':'not_yet_recorded'}
    validated=[];basis=[]
    for update in receipt['rule_updates']:
        review=update.get('increment') or {}
        comparison=(get_value(harness.store,review['comparison_ref']) if review.get('comparison_ref') else
                    review.get('comparison') or receipt.get('comparison') or {})
        supported,proof=supported_comparison(harness,job,comparison,cutoff)
        basis.append(proof)
        if update.get('new_version') is True and comparison.get('substantive') is True and supported:
            validated.append({k:copy.deepcopy(update.get(k)) for k in ('entry_id','version','body_ref','operation')})
    label,state=(1,'agent_supported_increment') if validated else (None,'insufficient_learning_support')
    if receipt.get('outcome') in ('already_covered','no_increment','evidence_added'):
        supported,proof=supported_comparison(harness,job,receipt.get('comparison'),cutoff)
        basis.append(proof)
        if supported and (receipt.get('comparison') or {}).get('substantive') is not True:
            label,state=0,'agent_supported_no_increment'
    record,report=source_at(harness,job,cutoff)
    from or_harness.world_model.hplus import _verification
    return {'label':label,'verification_state':state,'validated_rules':validated,
        'hypothesis_outcome':receipt.get('hypothesis_outcome','unknown'),'receipt':receipt,
        'observation_rule_version':OBSERVATION_VERSION,
        'observation_basis':{'support_refs':list(dict.fromkeys(r['reference'] for b in basis for r in b.get('references') or [])),
            'checks':list({r['execution_id']:{k:r.get(k) for k in ('execution_id','checker_action_id','checked_at','state','valid')}
                           for b in basis for r in b.get('references') or [] if r.get('actual_execution')}.values())[:3],
            'reason':next((b['reason'] for b in basis if b.get('reason')),None)},
        'verification':_verification(record,job['signal']) if record else []}
