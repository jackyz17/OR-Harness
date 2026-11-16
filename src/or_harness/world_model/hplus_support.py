"""Agent-authored learning conclusions with valid, time-scoped evidence.

Publication, recall coverage and body hashes are not learning labels.
"""
import copy
import time

from or_harness.core.storage import StorageError
from or_harness.world_model.conditions import get_value,put_value,digest
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
        try:report=copy.deepcopy(get_value(harness.store,previous))
        except StorageError:return {}
    return report


def _checker_fact(report,path=None):
    """Only the named check item determines an item reference's dependency."""
    fact={k:copy.deepcopy(report.get(k)) for k in
          ('state','checks','scope','reference_source','action_id','checked_at')}
    fact['checked']=bool(report.get('checks')) and report.get('state') in ('passed','failed')
    parts=(path or '').split('.')
    if len(parts)>=3 and parts[:2]==['task_check','checks'] and parts[2].isdecimal():
        index=int(parts[2]);checks=report.get('checks') or []
        if index<len(checks):
            item=checks[index];fact['checks']=[copy.deepcopy(item)]
            fact['checked']=type(item.get('ok')) is bool
            fact['state']='passed' if item.get('ok') is True else 'failed' if item.get('ok') is False else None
            fact['scope']={'basis':[item.get('check')]}
            if not str(item.get('check')).startswith('reference_'):fact['reference_source']=None
    return fact


def _first_late_check(harness,record,after,cutoff,path):
    report=_report_at(harness,record,cutoff);first=None
    while report and report.get('checked_at',0)>=after:
        fact=_checker_fact(report,path)
        if fact['checked']:first=fact
        previous=report.get('previous_check_ref')
        if not previous:break
        try:report=get_value(harness.store,previous)
        except StorageError:break
    return first


def source_at(harness,job,cutoff):
    eid=job['execution_ids'][0]
    try:
        if not validity_at(harness,eid,cutoff).get('valid'):return None,None
        record=resolve_execution(harness,eid,include_excluded=True)
        return (record,_report_at(harness,record,cutoff)) if record else (None,None)
    except (StorageError,ValueError,KeyError):
        return None,None  # one unrecoverable legacy source cannot break other receipts


def _reference(harness,job,ref,cutoff):
    """Resolve locatable support; context references alone prove no result."""
    if not isinstance(ref,str) or not ref:raise ValueError('learning evidence needs nonempty references')
    raw={};eid=None;path=None
    if ref.startswith('artifacts.') or ref.startswith('task_check.'):
        eid=job['execution_ids'][0];path=ref
    elif ':' not in ref or ref.startswith('execution:'):
        eid=ref.split(':',1)[1] if ref.startswith('execution:') else ref
    else:
        try:raw=get_value(harness.store,ref)
        except Exception as exc:raise ValueError('unknown learning evidence reference '+ref) from exc
        if not isinstance(raw,dict):raise ValueError('learning evidence must reference a saved fact: '+ref)
        eid=raw.get('execution_id')
        if eid is None and raw.get('record'):eid=raw['record'].get('execution_id')
    if eid:
        record=resolve_execution(harness,eid,include_excluded=True)
        if record is None:raise ValueError('learning evidence execution is unavailable: '+str(eid))
        if raw.get('task_id') and str(raw['task_id'])!=str(record.task_id):
            raise ValueError('learning evidence belongs to another task: '+ref)
        report=_report_at(harness,record,cutoff)
        valid=record.created_at<=cutoff and validity_at(harness,eid,cutoff).get('valid',False)
        artifacts=record.execution_features.get('artifacts') or {}
        has_artifact=False
        if path:
            value={'artifacts':artifacts,'task_check':report}
            for part in path.split('.'):
                if isinstance(value,dict) and part in value:value=value[part]
                elif isinstance(value,list) and part.isdecimal() and int(part)<len(value):value=value[int(part)]
                else:raise ValueError('learning evidence path does not exist on execution '+str(eid)+': '+ref)
            if path.startswith('artifacts.'):
                artifact=artifacts[path.split('.')[1]]
                from or_harness.core.artifacts import read
                try:read(harness.home,artifact)
                except (OSError,ValueError,KeyError,TypeError,StorageError) as exc:
                    raise ValueError('learning evidence artifact is unavailable or invalid: '+ref) from exc
                has_artifact=True
        else:
            # An execution/fact reference may support artifact inspection only
            # when at least one of its captured artifacts is still readable.
            from or_harness.core.artifacts import read
            for artifact in artifacts.values():
                try:read(harness.home,artifact);has_artifact=True;break
                except (OSError,ValueError,KeyError,TypeError,StorageError):continue
        checker=_checker_fact(report,path)
        declared_report=(raw.get('checker') or ((raw.get('record') or {}).get('execution_features') or {}).get('task_check') or
                         (raw if ref.startswith('task_check:') else {}))
        return {'reference':ref,'execution_id':eid,'valid':valid,'actual_execution':True,
                **{k:checker.get(k) for k in ('checked','state','checks','scope','reference_source')},
                'artifact':has_artifact,'checker_action_id':report.get('action_id'),
                'checked_at':report.get('checked_at'),
                'declared_checker':_checker_fact(declared_report,path) if declared_report else None,
                'path':path,
                'acceptance_rule':((report.get('acceptance_rule') or {}).get('fingerprint'))}
    return {'reference':ref,'valid':True,'actual_execution':False,'checked':False,'context_only':True}


def _support_still_present(original,current):
    """Preserve the observed result and scope, independent of action IDs/time."""
    if not original.get('checked'):return True  # a later first check can supply evidence
    if original.get('state')!=current.get('state'):return False
    if not set((original.get('scope') or {}).get('basis') or [])<=set((current.get('scope') or {}).get('basis') or []):return False
    remaining=[digest(c) for c in current.get('checks') or []]
    for check in original.get('checks') or []:
        signature=digest(check)
        if signature not in remaining:return False
        remaining.remove(signature)
    return original.get('reference_source')==current.get('reference_source')


def supported_comparison(harness,job,comparison,cutoff,*,strict=False):
    if not isinstance(comparison,dict):return False,{'reason':'no agent learning assessment'}
    refs=comparison.get('evidence_refs') or []
    if not isinstance(refs,list):
        if strict:raise ValueError('comparison.evidence_refs must be a list')
        return False,{'reason':'comparison.evidence_refs must be a list'}
    resolved=[];issues=[]
    for ref in refs:
        try:resolved.append(_reference(harness,job,ref,cutoff))
        except (ValueError,StorageError,OSError,KeyError,TypeError) as exc:
            if strict:raise ValueError('invalid learning evidence '+str(ref)+': '+str(exc)) from exc
            issues.append(str(exc))
            resolved.append({'reference':ref,'valid':False,'reason':str(exc)})
    named=bool(isinstance(comparison.get('source'),str) and comparison['source'].strip() and
               isinstance(comparison.get('assessment'),str) and comparison['assessment'].strip())
    basis=comparison.get('basis','recorded_checks')
    for current in resolved:
        original=(comparison.get('support_snapshot') or {}).get(current['reference'])
        if basis=='recorded_checks' or str(current.get('path') or '').startswith('task_check.'):
            dependencies=[c for c in (original,current.get('declared_checker')) if c]
            if original and not original.get('checked') and current.get('actual_execution'):
                record=resolve_execution(harness,current['execution_id'],include_excluded=True)
                first=_first_late_check(harness,record,original.get('checked_at') or 0,cutoff,current.get('path'))
                if first:dependencies.append(first)
            if any(not _support_still_present(dependency,current) for dependency in dependencies):
                current['valid']=False
                current['reason']='supporting checker result or scope was corrected or withdrawn'
                issues.append(current['reason'])
    actual=[r for r in resolved if r.get('actual_execution')]
    checked=any(r.get('checked') for r in actual)
    if basis=='derivation':checked=bool(actual)  # the agent's retained derivation names the actual input facts
    if basis=='artifact_inspection':checked=any(r.get('artifact') for r in actual)
    valid=bool(named and comparison.get('sufficient') is True and resolved and actual and
               all(r['valid'] for r in resolved) and checked)
    return valid,{'basis':basis,'references':resolved,
        'reason':None if valid else '; '.join(issues) or 'insufficient, not yet checked, or excluded supporting facts'}


def freeze_comparison(harness,job,comparison,cutoff):
    """Validate and retain every direct support before releasing working evidence."""
    if comparison is None:return None
    if not isinstance(comparison,dict):raise ValueError('learning comparison must be a JSON object')
    frozen=copy.deepcopy(comparison)
    frozen.pop('support_snapshot',None)  # dependency snapshots are framework facts
    _,proof=supported_comparison(harness,job,frozen,cutoff,strict=True)
    from or_harness.strategy.rule_versions import _facts_ref
    support={}
    for fact in proof.get('references') or []:
        if fact.get('actual_execution'):
            record=resolve_execution(harness,fact['execution_id'],include_excluded=True)
            fact['fact_ref']=_facts_ref(harness,harness.store.conn,record)
        support[fact['reference']]=fact
    frozen['support_snapshot']=support
    return frozen


def observation(harness,job,cutoff):
    if job.get('state')!='completed':
        return {'label':None,'verification_state':'unobserved' if job['state']=='skipped' else 'pending'}
    try:receipt=get_value(harness.store,job['receipt_ref'])
    except (StorageError,KeyError) as exc:
        return {'label':None,'verification_state':'unavailable_receipt','reason':str(exc)}
    if receipt['created_at']>cutoff:return {'label':None,'verification_state':'not_yet_recorded'}
    validated=[];basis=[]
    for update in receipt.get('rule_updates') or []:
        review=update.get('increment') or {}
        try:
            comparison=(get_value(harness.store,review['comparison_ref']) if review.get('comparison_ref') else
                        review.get('comparison') or receipt.get('comparison') or {})
        except (StorageError,KeyError) as exc:
            basis.append({'reason':str(exc)});continue
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
