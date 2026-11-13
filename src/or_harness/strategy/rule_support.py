"""Comparable rule evidence: fact versions, representative regressions and costs."""
import copy
import math
import re

from or_harness.strategy.evidence_views import task_identity,same_frozen_input,instance_relation
from or_harness.strategy.evidence_access import resolve_execution,validity
from or_harness.world_model.conditions import digest,get_value


def fact_basis(record,report=None):
    report=report if report is not None else record.execution_features.get('task_check') or {}
    return digest({'acceptance':(report.get('acceptance_rule') or {}).get('fingerprint'),
        'state':report.get('state'),'checks':report.get('checks'),'scope':report.get('scope'),'intent':report.get('intent'),
        'method_actual':record.method_actual,'code_hash':(record.solver or {}).get('code_hash'),
        'artifacts':{k:v.get('sha256') for k,v in (record.execution_features.get('artifacts') or {}).items() if isinstance(v,dict)}})


def inherit_assessments(record,report,declaration):
    previous=record.execution_features.get('task_check') or {}
    same=(previous.get('rule_assessment_basis') or fact_basis(record,previous))==fact_basis(record,report)
    report['rule_assessment_basis']=fact_basis(record,report)
    for field in ('rule_assessments','hplus_verification'):
        if field not in declaration:
            if same and field in previous:report[field]=copy.deepcopy(previous[field])
        elif not isinstance(declaration[field],list) or not all(isinstance(v,dict) for v in declaration[field]):
            raise ValueError(field+' must be a list of sourced assessments')
        elif field=='rule_assessments' and same and declaration[field]:
            supplied={(str(v.get('entry_id')),v.get('version')) for v in declaration[field]}
            report[field]=[copy.deepcopy(v) for v in previous.get(field) or [] if (str(v.get('entry_id')),v.get('version')) not in supplied]+copy.deepcopy(declaration[field])
        else:report[field]=copy.deepcopy(declaration[field])
    report['rule_assessments_withdrawn']=('rule_assessments' in declaration and declaration['rule_assessments']==[])
    report['assessment_reuse']='unchanged_facts' if same else 'changed_basis_requires_review'


def regression_cases(harness,execution_ids):
    grouped={}
    for eid in execution_ids:
        record=resolve_execution(harness,eid)
        if record is None:
            grouped['missing:'+eid]={'case_id':'missing:'+eid,'execution_ids':[eid],
                'representative_execution_id':eid,'identity':{},'acceptance_fingerprint':None}
            continue
        report=record.execution_features.get('task_check') or {}
        identity=task_identity(harness,record)
        source=identity.get('input_digest') or identity.get('content_digest') or identity['base_id']
        checker=(report.get('acceptance_rule') or {}).get('fingerprint')
        key='regression:'+digest([source,checker,report.get('scope')])
        value=grouped.setdefault(key,{'case_id':key,'execution_ids':[],
            'representative_execution_id':eid,'identity':identity,'acceptance_fingerprint':checker,
            'acceptance_scope':copy.deepcopy(report.get('scope'))})
        value['execution_ids'].append(eid)
    return list(grouped.values())


def required_cases(harness,metadata):
    return metadata.get('regression_cases') or regression_cases(harness,metadata.get('required_regressions') or [])


def with_merge_requirements(harness,entry_id,version,metadata):
    """Merge obligations use the same frozen cases as ordinary revisions."""
    out=copy.deepcopy(metadata)
    cases={case['case_id']:case for case in required_cases(harness,metadata)}
    for row in harness.store.conn.execute("SELECT payload FROM rule_events WHERE entry_id=? AND version=? AND kind='merge'",(str(entry_id),int(version))):
        event=harness.store.loads(row['payload'])
        for case in event.get('regression_cases') or regression_cases(harness,event.get('required_regressions') or []):
            if case['case_id'] in cases:
                cases[case['case_id']]['execution_ids']=list(dict.fromkeys(cases[case['case_id']]['execution_ids']+case['execution_ids']))
            else:cases[case['case_id']]=case
    out['regression_cases']=list(cases.values())
    return out


def regression_matches(harness,record,metadata):
    identity=task_identity(harness,record)
    fingerprint=((record.execution_features.get('task_check') or {}).get('acceptance_rule') or {}).get('fingerprint')
    return [c for c in required_cases(harness,metadata) if c.get('acceptance_fingerprint') and
            c['acceptance_fingerprint']==fingerprint and same_frozen_input(identity,c.get('identity') or {})]


def effective_checks(harness,checks):
    selected={}
    for c in checks:
        if c.get('non_observation'):continue
        ids=[c.get('execution_id'),(c.get('cost_comparison') or {}).get('execution_id')]
        if any(eid and not validity(harness,eid)['valid'] for eid in ids):continue
        if c.get('fact_basis') and c.get('execution_id'):
            record=resolve_execution(harness,c['execution_id'])
            if record is not None and fact_basis(record)!=c['fact_basis']:continue
        if c.get('cost_comparison'):
            current=resolve_execution(harness,c['execution_id'])
            original=c['cost_comparison']
            if current is not None:
                compared,issue=cost_comparison(harness,current,{'comparator_execution_id':original['execution_id'],
                    'cost_dimensions':list(original['dimensions'])})
                if (issue or compared['dimensions']!=original['dimensions'] or
                    original.get('measurement_sources') is not None and compared['measurement_sources']!=original['measurement_sources']):continue
        key=('derivation',c['check_id']) if c['kind']=='derivation' else ('case',c.get('execution_id'))
        old=selected.get(key)
        ordering=lambda x:(-1 if x['state']=='pending' else x.get('observed_at',x['at']),x['at'])
        if old is None or ordering(c)>=ordering(old):selected[key]=c
    return list(selected.values())


def claimed_dimensions(body,assessment):
    from or_harness.core.schema import COST_DIMENSIONS
    declared=assessment.get('cost_dimensions')
    if declared:return list(declared)
    text=str(body.get('expect') or '').lower()
    dims=[d for d in COST_DIMENSIONS if d in text]
    if not dims and ('tokens' in text or 'token' in text):dims=['llm_tokens']
    if not dims and ('工具调用' in text or 'tool call' in text):dims=['tool_calls']
    return dims


def cost_comparison(harness,record,check):
    from or_harness.world_model.cost_eligibility import cost_eligibility,MEASURED
    from or_harness.world_model.g_signal import actual_solving_conditions
    from or_harness.core.schema import COST_DIMENSIONS,task_check_state
    dims=check.get('cost_dimensions') or []
    if not dims or any(d not in COST_DIMENSIONS for d in dims):
        return None,'cost advantage needs explicit claimed resource dimensions'
    actual=actual_solving_conditions(record)
    report=record.execution_features.get('task_check') or {}
    fingerprint=(report.get('acceptance_rule') or {}).get('fingerprint')
    def inspect(other):
        if other.execution_id==record.execution_id:return None,'comparator is the same execution'
        if not same_frozen_input(task_identity(harness,record),task_identity(harness,other)):
            return None,'cost comparator is not the same frozen original task'
        from or_harness.world_model.g_signal import method_key
        technique=lambda m:{k:v for k,v in method_key(m).items() if k in ('name','steps','why','fallback')}
        if not record.method_actual or not other.method_actual or technique(record.method_actual)==technique(other.method_actual):
            return None,'cost comparator needs observed different actual methods'
        conditions=actual_solving_conditions(other)
        if not actual['comparable'] or not conditions['comparable'] or actual['comparison_key']!=conditions['comparison_key']:
            return None,'cost comparator has unknown or different actual solving conditions'
        other_report=other.execution_features.get('task_check') or {}
        if not fingerprint or fingerprint!=(other_report.get('acceptance_rule') or {}).get('fingerprint') or task_check_state(other)!='passed':
            return None,'cost comparator has insufficient or different acceptance rules'
        if record.measurement_scope!=other.measurement_scope:
            return None,'cost dimensions are not measured on the same scope'
        if any(cost_eligibility(r,d)!=MEASURED for r in (record,other) for d in dims):
            return None,'cost comparator includes estimated, partial or unmeasured dimensions'
        values={d:{'actual':getattr(record.cost,d),'baseline':getattr(other.cost,d)} for d in dims}
        if any(not math.isfinite(float(v)) or v<0 for pair in values.values() for v in pair.values()):
            return None,'cost comparator contains invalid measured values'
        return {'execution_id':other.execution_id,'dimensions':values,'scope':record.measurement_scope,
            'actual_conditions':actual,'relation':'strictly_less',
            'measurement_sources':{r.execution_id:{d:copy.deepcopy((r.execution_features.get('cost_provenance') or {}).get(d) or {'source':'executor_or_legacy_observation'}) for d in dims} for r in (record,other)}},None
    if check.get('comparator_execution_id'):
        other=resolve_execution(harness,str(check['comparator_execution_id']))
        return inspect(other) if other is not None else (None,'cost advantage needs a distinct recorded comparator')
    candidates=[]
    for other in harness.bank.all():
        if other.source!='executed':continue
        value,issue=inspect(other)
        if not issue:candidates.append(value)
    if not candidates:return None,'cost advantage needs a distinct recorded comparator with trusted measured dimensions'
    # Equal repeated receipts do not create a second comparison convention.
    groups={digest([v['dimensions'],v['scope'],v['actual_conditions']['comparison_key']]) for v in candidates}
    if len(groups)>1:
        check['comparison_candidates']=[{'execution_id':v['execution_id'],'dimensions':v['dimensions'],'scope':v['scope']} for v in candidates]
        return None,'multiple comparable baseline measurements; choose an existing comparator_execution_id'
    return sorted(candidates,key=lambda c:c['execution_id'])[0],None


def normalized_text(text):
    # Quoted/code fragments keep their whitespace; prose formatting does not.
    parts=re.split(r'(```[\s\S]*?```|`[^`]*`|"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\')',str(text))
    return ''.join(part if i%2 else re.sub(r'\s+',' ',part) for i,part in enumerate(parts)).strip()


def body_signature(body):
    return digest({**{k:normalized_text(body.get(k,'')) for k in ('when','do','expect')},
                   'predicates':body.get('predicates') or {},'effect_scope':body.get('effect_scope','task_check')})
