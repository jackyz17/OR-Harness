"""Small reading views; complete facts remain in their original stores.

An oversized method is replaced by a reference as a unit, never by a
misleading prefix. No view turns a plan, a match or an adoption into use.
"""
from __future__ import annotations

import copy
import json
import re

from or_harness.core.schema import task_check_state


def chars(value):
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")))


def atomic(value, reference, limit=2400):
    if chars(value) <= limit:
        return copy.deepcopy(value)
    return {"detail_ref": reference, "chars": chars(value), "state": "whole_value_by_reference"}


def declared_task_identity(task):
    """Instance evidence from original inputs; structure alone is not identity."""
    from or_harness.world_model.conditions import digest
    origin_keys=('base_task_id','source_task_id','parent_task_id','original_task_id','same_instance_as')
    origins={str(task[k]) for k in origin_keys if task.get(k)}
    for container in ('lineage','origin','annotations','data_ref'):
        value=task.get(container)
        if isinstance(value,dict):origins.update(str(value[k]) for k in origin_keys if value.get(k))
    original={k:copy.deepcopy(task[k]) for k in ('raw_data','original_data','instance_data','data','input_data','data_ref') if task.get(k) is not None}
    model=task.get('model')
    if isinstance(model,dict) and any(model.get(k) for k in ('objective','constraints','matrix','variables','domains')):
        original['model']=copy.deepcopy(model)
    spec=task.get('spec')
    if isinstance(spec,dict) and any(spec.get(k) for k in ('objective','constraints','coefficients','matrix','parameters','data')):
        original['spec']=copy.deepcopy(spec)
    for key in ('objective','constraints','requirements','business_rules'):
        if isinstance(task.get(key),(dict,list)):original[key]=copy.deepcopy(task[key])
    instance=next((str(task[k]) for k in ('instance_id','source_instance_id','instance_uri') if task.get(k)),None)
    namespace=str(task.get('dataset_id') or task.get('dataset') or '')
    immutable_instance=digest([namespace,instance,task.get('instance_version')]) if instance else None
    input_digest=digest(original) if original else None
    semantics={k:copy.deepcopy(task[k]) for k in ('description','text','objective','constraints','spec','requirements','business_rules','data_ref','coupling') if task.get(k) is not None}
    content=digest(semantics) if semantics else None
    text=str(task.get('text') or task.get('description') or '')
    aliases={'task:'+str(task['task_id'])}|{'task:'+s for s in origins}
    if input_digest:aliases.add('input:'+input_digest)
    if immutable_instance:aliases.add('instance:'+immutable_instance)
    if content and not input_digest:aliases.add('content:'+content)
    return {'identity_version':'task-instance/2','task_id':str(task['task_id']),
        'base_id':sorted(origins)[0] if origins else input_digest or content or str(task['task_id']),
        'basis':'declared_origin' if origins else 'original_inputs' if original else 'retained_text' if content else 'task_id_only',
        'content_digest':content,'input_digest':input_digest,'instance_digest':immutable_instance,
        'aliases':sorted(aliases),'origin_task_ids':sorted(origins),'advisory_only':True,
        'relationship':'variant' if origins else 'lineage_not_declared','hypothetical':False}


def task_identity(harness, record):
    saved = (record.execution_features or {}).get("task_identity")
    if isinstance(saved, dict) and saved.get('identity_version')=='task-instance/2':
        return copy.deepcopy(saved)
    action = harness.actions.by_execution(record.execution_id)
    snapshot = harness.get_snapshot(action.pre_snapshot_id) if action and action.pre_snapshot_id else None
    problem = snapshot.problem_state if snapshot else {}
    payload = ((record.execution_features or {}).get('retained_task_payload') or
               problem.get('task_input') or problem.get("task_payload") or {})
    if payload.get('task_id'):
        result=declared_task_identity(payload)
        result['hypothetical']=bool(snapshot.hypothetical) if snapshot else record.source!='executed'
        return result
    if isinstance(saved,dict):
        result=copy.deepcopy(saved)
        result['aliases']=sorted(set(identity_aliases(result)))
        return result
    origin = payload.get("base_task_id") or payload.get("source_task_id") or payload.get("parent_task_id")
    identity = problem.get("problem_identity_digest")
    # Equal retained problem/text identities prove duplicate source content;
    # different task names by themselves do not prove independence.
    return {"base_id": str(origin) if origin else record.task_text_digest or identity or str(record.task_id),
            "basis": "declared_origin" if origin else "frozen_problem" if identity else
                     "retained_text" if record.task_text_digest else "task_id_only",
            "task_id": record.task_id, "relationship": "variant" if origin else "unresolved",
            "hypothetical": bool(snapshot.hypothetical) if snapshot else record.source != "executed"}


def identity_aliases(identity):
    aliases=set(identity.get('aliases') or [])
    if identity.get('task_id'):aliases.add('task:'+str(identity['task_id']))
    if identity.get('basis')=='declared_origin':aliases.add('task:'+str(identity['base_id']))
    for key in ('input_digest','instance_digest','content_digest'):
        if key=='content_digest' and identity.get('input_digest'):continue
        if identity.get(key):aliases.add(key.split('_')[0]+':'+identity[key])
    if not aliases and identity.get('base_id'):aliases.add('base:'+str(identity['base_id']))
    return aliases


def instance_relation(left,right):
    if identity_aliases(left)&identity_aliases(right):
        return 'same_source'
    if left.get('basis')=='task_id_only' or right.get('basis')=='task_id_only':return 'unknown'
    return 'independent'


def same_frozen_input(left,right):
    if left.get('input_digest') or right.get('input_digest'):
        return bool(left.get('input_digest') and left['input_digest']==right.get('input_digest'))
    for key in ('content_digest','instance_digest'):
        if left.get(key) and right.get(key) and left[key]==right[key]:return True
    return False


def base_partitions(identities):
    """Union declared origins and identical retained content; never parse IDs."""
    parents = {}
    def root(key):
        parents.setdefault(key,key)
        if parents[key] != key:
            parents[key] = root(parents[key])
        return parents[key]
    keys = []
    for identity in identities:
        row=sorted(identity_aliases(identity)) or ['base:'+str(identity['base_id'])]
        keys.append(row)
        for key in row[1:]:
            parents[root(key)] = root(row[0])
    return [root(row[0]) for row in keys]


def compact_execution(harness, record, *, field_limit=2400):
    f = record.execution_features or {}
    ref = f"orx inspect --bank experience --execution {record.execution_id}"
    actual, planned = record.method_actual, record.method_planned
    check = f.get("task_check") or {}
    profile = record.profile_snapshot
    used = f.get("used_entries")
    adoption = {"state": "undeclared"} if not isinstance(used, dict) else {
        "state": "declared_none" if not used.get("entry_ids") else "declared",
        **{k: copy.deepcopy(used[k]) for k in ("entry_ids", "known_entry_ids", "unknown_entry_ids", "versions", "rule_versions") if k in used}}
    task_text = harness.store.get_task_text(record.task_id, record.task_text_digest) if record.task_text_digest else None
    task_text=task_text or f.get('retained_task_text')
    return {
        "execution_id": record.execution_id, "task_id": record.task_id,
        "strategy_id": record.strategy_id, "created_at": record.created_at,
        "identity": task_identity(harness, record), "family": profile.family,
        "problem": {"profile": {"family": profile.family, "problem_class": profile.problem_class},
                    "cir": atomic(record.cir_snapshot, ref + " #cir_snapshot", field_limit)},
        "task_text": atomic(task_text, ref + " #task_text", field_limit) if task_text else {"unknown": "task text not retained"},
        "method": {"basis": "performed" if actual else "planned_only" if planned else "none",
                   "actual": atomic(actual, ref + " #method_actual", field_limit),
                   "planned": atomic(planned, ref + " #method_planned", field_limit)},
        "method_source": (actual or {}).get("source") if actual else "unknown",
        "outcome": {k: (record.quality or {}).get(k) for k in ("status", "feasible", "objective", "gap")},
        "status": (record.quality or {}).get("status"),
        "task_check_state": task_check_state(record) or "never_checked",
        "task_check": {"state": task_check_state(record) or "never_checked", "scope": atomic(check.get("scope"), ref + " #task_check.scope", field_limit),
                       "diffs": atomic(check.get("diffs") or [], ref + " #task_check.diffs", field_limit),
                       "report_ref": ref + " #task_check"},
        "cost": {d: getattr(record.cost, d) for d in sorted(record.cost.measured_dims())},
        "cost_measured": sorted(record.cost.measured_dims()), "measurement_scope": record.measurement_scope,
        "failures": [{"error_class": x.error_class,
                      "error": atomic(x.error, ref + " #failures", field_limit)} for x in (record.failures or [])[:2]],
        "adoption": adoption, "inspect_hint": ref,
    }


def short_rule(entry, *, field_limit=2400):
    ref = f"orx inspect --bank strategic --entry {entry.entry_id}"
    claim = entry.claim or {}
    rule = getattr(entry, "rule", None) or {}
    body = rule.get("body") or {
        "when": (claim.get("conditions") or {}).get("note") or entry.predicates,
        "do": claim.get("method") or entry.actions or {"unknown": "no executable method recorded"},
        "expect": claim.get("text") or {"quality_estimated": entry.quality_estimated},
    }
    return {"entry_id": entry.entry_id, "strategy_id": entry.strategy_id,
            "schema_version": rule.get("schema_version", "legacy"),
            "version": rule.get("view_version"), "active_version": rule.get("active_version"),
            "candidate_version": rule.get("candidate_version"),
            "status": rule.get("state", entry.status), "verification_state": entry.verification_state,
            "verification_scope":atomic((entry.verification or {}).get('scope'),ref+' #verification.scope',field_limit),
            "published": entry.is_published, "support_n": entry.support_n,
            "body": {k: atomic(v, ref + " #" + k, field_limit) for k, v in body.items()},
            "predicates": copy.deepcopy(entry.predicates), "detail_ref": ref,
            "body_ref": rule.get("body_ref"), "historical_state": not bool(rule) or bool(rule.get("legacy_version")),
            "merged_into":rule.get('merged_into'), "current_task_premises":"requires_assessment" if rule else "historical_scope",
            "merge_pending_into":rule.get('merge_pending_into'),
            "candidate_ref": (ref + " --rule-version candidate" if rule.get("candidate_version") is not None else None)}
