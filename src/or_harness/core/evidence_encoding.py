"""Shared encoding for execution facts and frozen prediction inputs.

Custom explanations and unique observations survive. Fixed prose and exact
copies use codes/references; readers resolve them without changing labels,
measurement masks or scopes. Reconstructible method trajectory copies are
omitted, while unique run events remain on their execution.
"""
from __future__ import annotations

import copy
import json


def method_fingerprint(method):
    from or_harness.core.schema import normalize_method
    from or_harness.world_model.conditions import digest
    normalized=normalize_method(method) or {}
    return digest({k:v for k,v in normalized.items() if k!='source'})


def share_values(store, value):
    """Intern exactly equal methods/check definitions, never semantic matches."""
    from or_harness.world_model.conditions import put_value
    if isinstance(value,list):return [share_values(store,v) for v in value]
    if not isinstance(value,dict):return value
    if '$shared' in value:return copy.deepcopy(value)
    if isinstance(value.get('steps'),list) and isinstance(value.get('name'),str):
        body={k:copy.deepcopy(v) for k,v in value.items() if k!='source'}
        out={'$shared':put_value(store.conn,store,'method',body)}
        if 'source' in value:out['source']=value['source']
        return out
    out={}
    for k,v in value.items():
        if k=='acceptance_rule' and isinstance(v,dict) and v.get('spec') and v.get('checker'):
            from or_harness.world_model.g_signal import canonical_evidence
            body=canonical_evidence({key:item for key,item in v.items() if key!='fingerprint'})
            out[k]={'$shared':put_value(store.conn,store,'task_check_rule',body),
                    'fingerprint':v.get('fingerprint')}
            continue
        if k=='pairs' and isinstance(v,list):
            out[k]=[{'$shared':put_value(store.conn,store,'feedback_pair',row)}
                    if isinstance(row,dict) and row.get('evaluation_id') and row.get('prediction_id')
                    else share_values(store,row) for row in v]
            continue
        if k=='body' and value.get('body_ref') and isinstance(v,dict):
            from or_harness.world_model.conditions import get_value
            original=get_value(store,value['body_ref'])
            if all(key in original and original[key]==item for key,item in v.items()):
                out[k]={'$shared':value['body_ref'],'$fields':list(v)}
                continue
        kind={'profile_snapshot':'profile','profile':'profile','cir_snapshot':'cir','cir':'cir',
              'task_identity':'task_identity','identity':'task_identity'}.get(k)
        if kind and isinstance(v,dict) and v and set(v)!={'value_ref'}:
            out[k]={'$shared':put_value(store.conn,store,kind,v)}
        else:out[k]=share_values(store,v)
    return out


def resolve_values(store,value):
    from or_harness.world_model.conditions import get_value
    if isinstance(value,list):return [resolve_values(store,v) for v in value]
    if not isinstance(value,dict):return value
    if '$shared' in value:
        out=copy.deepcopy(get_value(store,value['$shared']))
        if '$fields' in value:out={k:out[k] for k in value['$fields']}
        out.update({k:copy.deepcopy(v) for k,v in value.items() if k not in ('$shared','$fields')})
        return out
    return {k:resolve_values(store,v) for k,v in value.items()}


def _at(value,path):
    for key in path.split('.'):
        if not isinstance(value,dict) or key not in value:return None
        value=value[key]
    return value


def _method_extras(value):
    """Replace exact result copies inside method extras by local references."""
    value=copy.deepcopy(value)
    for key in ('method_actual','method_planned'):
        extra=(value.get(key) or {}).get('extra') or {}
        variables=(value.get('execution_features') or {}).get('solution_variables') or {}
        for name,item in list(extra.items()):
            paths=['execution_features.solution_variables']+[
                'execution_features.solution_variables.'+k for k in variables]
            path=next((p for p in paths if item is not None and json.dumps(_at(value,p),sort_keys=True)==json.dumps(item,sort_keys=True)),None)
            if path and not (isinstance(item,dict) and '$record_path' in item):
                # A scalar reference is useful only if it also identifies the
                # duplicate result; the owning result remains the sole value.
                extra[name]={'$record_path':path}
    return value


def _resolve_local(value,root):
    if isinstance(value,list):return [_resolve_local(v,root) for v in value]
    if not isinstance(value,dict):return value
    if set(value)=={'$record_path'}:return copy.deepcopy(_at(root,value['$record_path']))
    return {k:_resolve_local(v,root) for k,v in value.items()}


def compact_check(report,store=None,methods=None):
    """Keep each verdict/difference; share its definition and source methods."""
    report=copy.deepcopy(report)
    if 'g_attribution' in report:
        legacy=report.pop('g_attribution')
        if store is not None:
            # Only old rows use this adapter. New checks save G evidence
            # directly in the world-model log before writing bank facts.
            from or_harness.world_model.g_signal import save_g_assessment
            save_g_assessment(store,methods,report,legacy)
    rule=report.get('acceptance_rule') or {}
    spec=rule.get('spec') or {}
    if store is not None and spec and 'checker' in rule:
        from or_harness.world_model.conditions import put_value
        from or_harness.world_model.g_signal import canonical_evidence
        body=canonical_evidence({k:v for k,v in rule.items() if k!='fingerprint'})
        report['acceptance_rule']={'definition_ref':put_value(store.conn,store,'task_check_rule',body),
                                   'fingerprint':rule.get('fingerprint')}
    probes=spec.get('semantic_probe') or []
    probes=[probes] if isinstance(probes,dict) else probes
    index=0
    for check in report.get('checks') or []:
        if check.get('check')!='semantic_probe':continue
        item=probes[index] if index<len(probes) else None
        if (item and check.get('ok') is True and check.get('source')=='framework' and
                check.get('path')==item.get('path')):
            check['rule_index']=index
            if 'equals' in item and json.dumps(check.get('observed'),sort_keys=True)==json.dumps(item['equals'],sort_keys=True):
                check.pop('observed',None);check['observed_from_rule']=True
            for field in ('source','path','expected','min','max','mode','found'):
                check.pop(field,None)
        index+=1
    return report


def expand_check(report,store=None):
    report=copy.deepcopy(report)
    rule=report.get('acceptance_rule') or {}
    if rule.get('definition_ref') and store is not None:
        from or_harness.world_model.conditions import get_value
        report['acceptance_rule']={**get_value(store,rule['definition_ref']),'fingerprint':rule.get('fingerprint')}
    probes=(report.get('acceptance_rule') or {}).get('spec',{}).get('semantic_probe') or []
    probes=[probes] if isinstance(probes,dict) else probes
    for check in report.get('checks') or []:
        if 'rule_index' not in check:continue
        index=check.pop('rule_index')
        if index>=len(probes):raise ValueError('check definition is unavailable')
        item=probes[index]
        check.update(source='framework',path=item.get('path'),found=True)
        if 'equals' in item:check.update(mode='equals',expected=copy.deepcopy(item['equals']))
        elif 'in' in item:check.update(mode='in',expected=copy.deepcopy(item.get('in') or []))
        else:check.update(mode='range',min=item.get('min'),max=item.get('max'))
        if check.pop('observed_from_rule',False):check['observed']=copy.deepcopy(item['equals'])
    return report

NOTES = {
    "config_sources": "the configuration that actually took effect, split by observer: 'executor' keys are the ones the sandbox set (a script may not overwrite them), 'script_reported' keys are the values the solve script read back from the solver. A key absent here is UNKNOWN — never a copy of the predicted config",
    "configured_instruction": "these values were HANDED to the script by the executor (through the environment). The framework cannot observe whether the solver honoured them, so they are not evidence that the solver used them: only a value the script READ BACK appears under 'script_reported'",
    "config_conflict": "the script reported a value for a key the executor controls; the executor's own value was kept",
    "config_missing": "the solve script reported no config (or its receipt carried no matching action_id stamp): script-internal parameters stay unknown",
    "performed_receipt": "the method the solve script reports it ACTUALLY performed, stamped with this attempt's action id. It is an observation, kept separate from the planned method — a plan is not a performed method",
}
PATHS = {"config_sources": ("execution_config", "note"),
         "configured_instruction": ("execution_config", "executor_configured_note"),
         "config_conflict": ("execution_config", "conflicts_note"),
         "config_missing": ("execution_config", "receipt"),
         "performed_receipt": ("method_receipt", "note")}


def compact_payload(raw,store=None):
    value = _method_extras(raw)
    features = value.get("execution_features") or {}
    codes = []
    for code, (block, field) in PATHS.items():
        row = features.get(block)
        if isinstance(row, dict) and row.get(field) == NOTES[code]:
            row.pop(field); codes.append(code)
    receipt = features.get("method_receipt")
    if isinstance(receipt, dict) and receipt.get("method") == raw.get("method_actual") and receipt.get("method") is not None:
        receipt.pop("method"); receipt["method_ref"] = "method_actual"
    if codes:
        value["explanation_codes"] = codes
    # Script-reported method steps are not separate runtime events. Keep
    # every other trajectory event, including failures/repairs and timings.
    steps=(raw.get('method_actual') or {}).get('steps') or []
    value['trajectory']=[t for t in value.get('trajectory') or [] if not (
        t.get('action','').startswith('method:') and t['action'][7:] in steps and not t.get('duration_s')
        and t.get('outcome')==(raw.get('quality') or {}).get('status')
        and not set(t)-{'action','outcome','duration_s'})]
    if isinstance(features.get('task_check'),dict):
        report=compact_check(features['task_check'],store,raw)
        if store is not None:
            from or_harness.world_model.conditions import put_value
            features['task_check']={'report_ref':put_value(store.conn,store,'task_check',report)}
        else:features['task_check']=report
    for key in ('prediction_snapshot','cir_snapshot','method_actual','method_planned'):
        if value.get(key) is None:value.pop(key,None)
    if not value.get('failures'):value.pop('failures',None)
    if value.get('cost_measured') is not None:
        value['cost']={k:v for k,v in value.get('cost',{}).items() if k in value['cost_measured']}
    value["evidence_encoding"] = "execution-evidence/3"
    if store is not None:value=share_values(store,value)
    return value


def expand_payload(raw,store=None):
    value = resolve_values(store,raw) if store is not None else copy.deepcopy(raw)
    features = value.get("execution_features") or {}
    report=features.get('task_check')
    if isinstance(report,dict):
        if report.get('report_ref') and store is not None:
            from or_harness.world_model.conditions import get_value
            report=get_value(store,report['report_ref'])
        features['task_check']=expand_check(report,store)
    value=_resolve_local(value,value)
    features=value.get('execution_features') or {}
    for code in value.get("explanation_codes") or []:
        if code in PATHS:
            block, field = PATHS[code]
            features.setdefault(block, {}).setdefault(field, NOTES[code])
    receipt = features.get("method_receipt")
    if isinstance(receipt, dict) and receipt.get("method_ref") == "method_actual":
        receipt.setdefault("method", copy.deepcopy(value.get("method_actual")))
        receipt.pop("method_ref", None)
    return value
