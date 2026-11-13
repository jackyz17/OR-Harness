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


def cost_comparison(harness,record,check):
    """Return an agent-selected factual contrast, with no automatic winner."""
    from or_harness.core.schema import COST_DIMENSIONS
    from or_harness.world_model.cost_eligibility import cost_eligibility,MEASURED
    from or_harness.world_model.g_signal import actual_solving_conditions
    eid=check.get('comparator_execution_id')
    if not eid:return None,'select an existing comparator_execution_id to compare costs'
    other=resolve_execution(harness,str(eid))
    if other is None or other.execution_id==record.execution_id:
        return None,'select a distinct valid recorded execution as comparator'
    dims=check.get('cost_dimensions')
    if dims is None:dims=[d for d in COST_DIMENSIONS if d in record.cost.measured_dims() or d in other.cost.measured_dims()]
    if not isinstance(dims,list) or any(d not in COST_DIMENSIONS for d in dims):
        return None,'cost_dimensions must name existing resource dimensions'
    values={}
    for dim in dims:
        actual=getattr(record.cost,dim) if dim in record.cost.measured_dims() else None
        baseline=getattr(other.cost,dim) if dim in other.cost.measured_dims() else None
        ae,be=cost_eligibility(record,dim),cost_eligibility(other,dim)
        values[dim]={'actual':actual,'baseline':baseline,'actual_eligibility':ae,'baseline_eligibility':be,
                    'change':actual-baseline if ae==be==MEASURED and actual is not None and baseline is not None else None}
    def facts(r):
        report=r.execution_features.get('task_check') or {}
        return {'execution_id':r.execution_id,'method_actual':copy.deepcopy(r.method_actual),
            'measurement_scope':r.measurement_scope,'execution_conditions':actual_solving_conditions(r),
            'acceptance':copy.deepcopy(report),'cost_provenance':copy.deepcopy(r.execution_features.get('cost_provenance') or {})}
    return {'execution_id':other.execution_id,'dimensions':values,'actual':facts(record),'baseline':facts(other),
        'same_frozen_input':same_frozen_input(task_identity(harness,record),task_identity(harness,other)),
        'acceptance_rule_equal':((record.execution_features.get('task_check') or {}).get('acceptance_rule') or {}).get('fingerprint')==
                                ((other.execution_features.get('task_check') or {}).get('acceptance_rule') or {}).get('fingerprint'),
        'verdict':None,'authority':'agent'},None


def normalized_text(text):
    # Quoted/code fragments keep their whitespace; prose formatting does not.
    parts=re.split(r'(```[\s\S]*?```|`[^`]*`|"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\')',str(text))
    return ''.join(part if i%2 else re.sub(r'\s+',' ',part) for i,part in enumerate(parts)).strip()


def body_signature(body):
    return digest({**{k:normalized_text(body.get(k,'')) for k in ('when','do','expect')},
                   'predicates':body.get('predicates') or {},'effect_scope':body.get('effect_scope','task_check')})
