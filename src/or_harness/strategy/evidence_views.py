"""Small reading views; complete facts remain in their original stores.

An oversized method is replaced by a reference as a unit, never by a
misleading prefix. No view turns a plan, a match or an adoption into use.
"""
from __future__ import annotations

import copy
import json
import os
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
    # An oversized method is carried WHOLE or by reference, never as a
    # misleading prefix: the reading copy keeps a pointer to the exact field
    # instead of clipping the steps the method needs to be understood.
    method = method_view(record)
    from or_harness.world_model.implementation_facts import execution_implementation_evidence, compact_evidence
    implementation = compact_evidence(execution_implementation_evidence(harness, record))
    for document in ("actual", "planned"):
        value = method.get(document)
        if isinstance(value, dict) and chars(value) > field_limit:
            method[document] = atomic(
                value, ref + f" #method_{document}", field_limit)
    return {
        "execution_id": record.execution_id, "task_id": record.task_id,
        "strategy_id": record.strategy_id, "created_at": record.created_at,
        "identity": task_identity(harness, record), "family": profile.family,
        "problem": {"profile": {"family": profile.family, "problem_class": profile.problem_class},
                    "cir": atomic(record.cir_snapshot, ref + " #cir_snapshot", field_limit)},
        "task_text": atomic(task_text, ref + " #task_text", field_limit) if task_text else {"unknown": "task text not retained"},
        "method": method,
        "implementation_evidence": implementation,
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


def dedupe_methods(planned,actual):
    """One full performed method; exact repeated plan steps use a local link.

    This is content sharing only, never evidence of fidelity or correctness.
    Distinct steps, settings, sources, explanations and fallback stay visible.
    """
    planned,actual=copy.deepcopy(planned),copy.deepcopy(actual)
    if planned and actual and planned.get('steps')==actual.get('steps') and planned.get('steps'):
        planned.pop('steps')
        planned['steps_ref']='method.actual.steps'
    return {'planned':planned,'actual':actual,
            'basis':'performed' if actual else 'planned_only' if planned else 'none'}


def method_view(record):
    from or_harness.core.evidence_encoding import _method_extras
    raw=_method_extras({'method_actual':record.method_actual,'method_planned':record.method_planned,
        'execution_features':{'solution_variables':record.execution_features.get('solution_variables') or {}}})
    return dedupe_methods(raw['method_planned'],raw['method_actual'])


def _pair_ref(row, field):
    evaluation_id = row.get("evaluation_id")
    if not evaluation_id:
        return None
    return (f"orx inspect --bank evaluations --evaluation {evaluation_id}"
            f" #{field}")


def _compact_pair_methods(row):
    """Keep the performed method whole; point the plan at it when they agree.

    The original plan and the performed method are BOTH predictions the
    world model needs, so neither is dropped. When the two describe the same
    method with at most two textual step differences, the performed steps are
    carried once and the plan keeps its name plus the exact differing steps.
    A genuinely different method keeps both full lists.
    """
    planned = row.get("method_planned")
    actual = row.get("method_actual")
    if not (isinstance(planned, dict) and isinstance(actual, dict)):
        return
    if planned.get("name") != actual.get("name"):
        return
    planned_steps = planned.get("steps")
    actual_steps = actual.get("steps")
    if not (isinstance(planned_steps, list) and isinstance(actual_steps, list)
            and planned_steps and len(planned_steps) == len(actual_steps)):
        return
    differences = [
        {"step": index + 1, "planned": planned_step, "actual": actual_step}
        for index, (planned_step, actual_step)
        in enumerate(zip(planned_steps, actual_steps))
        if planned_step != actual_step]
    if len(differences) > 2:
        return
    planned.pop("steps", None)
    planned["steps_ref"] = "method_actual.steps"
    if differences:
        planned["step_differences"] = differences
        row.setdefault("method_deviation", {
            "state": "minor_textual_difference",
            "differences": differences,
            "note": ("the performed steps are carried once under "
                     "method_actual.steps; only the differing plan steps are "
                     "restated here")})


def _compact_pair_cost_fact(row):
    """Cost EVIDENCE without the audit scaffolding.

    Kept: the measured values, each dimension's scope, any lower bound, the
    exclusions that explain a missing dimension, the executions and the
    observation rule. Dropped: window anchors, work-fact call lists, source
    refs and the solver condition block duplicated from the pair's own
    ``execution_conditions``; the exact record is named by ``detail_ref``.
    """
    cost = row.get("cost_observed_fact")
    if not isinstance(cost, dict):
        return
    ref = _pair_ref(row, "cost_observed_fact")
    compact = {}
    for key in ("values", "dimension_scopes", "lower_bounds",
                "boundary_exclusions", "execution_ids",
                "observation_rule_version", "dimension_observation_rule_versions"):
        if cost.get(key) not in (None, {}, [], ""):
            compact[key] = copy.deepcopy(cost[key])
    conditions = cost.get("conditions")
    if isinstance(conditions, dict):
        kept_conditions = {}
        for key in ("scope", "family", "phase", "window_coverage",
                    "structure", "retry_observation_rule_version",
                    "actual_solving_conditions"):
            if conditions.get(key) not in (None, {}, [], ""):
                kept_conditions[key] = copy.deepcopy(conditions[key])
        # Reusable artifact CONTENT stays in the audit record; the
        # prediction only needs the FACT that a reusable artifact existed.
        if conditions.get("reusable_artifacts"):
            kept_conditions["reusable_artifacts"] = True
        solver = conditions.get("solver_conditions")
        if isinstance(solver, dict):
            kept_conditions["solver_conditions"] = {
                key: copy.deepcopy(solver[key]) for key in
                ("solver", "config", "constraints")
                if solver.get(key) not in (None, {}, [], "")}
        if kept_conditions:
            compact["conditions"] = kept_conditions
    if ref:
        compact["detail_ref"] = ref
    row["cost_observed_fact"] = compact


def _compact_pair_implementation(row):
    """Technical outcome + the cause it already carries, not its scaffolding."""
    observation = row.get("implementation_observation")
    if not isinstance(observation, dict):
        return
    ref = _pair_ref(row, "implementation_observation")
    kept = {key: copy.deepcopy(observation[key]) for key in
            ("event", "comparable", "executed", "exit_code", "label",
             "observation_rule_version", "reason", "result_received", "scope",
             "task_id", "solver_interface", "selection_action_id", "complete",
             "semantic_attribution")
            if observation.get(key) not in (None, {}, [], "")}
    if observation.get("reusable_script") is not None and ref:
        kept["reusable_script_ref"] = ref + ".reusable_script"
    if observation.get("predicted_probability") is not None:
        kept["predicted_probability_ref"] = (
            f"risk_predicted[{observation.get('event') or 'implementation_failure'}].probability")
    row["implementation_observation"] = kept


def _compact_pair_task_check(row):
    """Keep the verdict and the action; the check scope is a spec, not a result."""
    check = row.get("task_check")
    if not isinstance(check, dict):
        return
    compact = {key: copy.deepcopy(check[key]) for key in
               ("state", "action_id") if check.get(key) is not None}
    if isinstance(check.get("scope"), dict):
        compact["scope_ref"] = _pair_ref(row, "task_check.scope")
        compact["check_report_ref"] = _pair_ref(row, "task_check")
    row["task_check"] = compact


def _compact_pair_execution_conditions(row):
    """The comparison key and its unknowns; drop the duplicated flat copies."""
    conditions = row.get("execution_conditions")
    if not isinstance(conditions, dict):
        return
    kept = {key: copy.deepcopy(conditions[key]) for key in
            ("comparable", "comparison_key", "execution_id", "unknown")
            if conditions.get(key) not in (None, {}, [], "")}
    row["execution_conditions"] = kept


def _compact_pair_cost_values(row):
    """Do not state the measured actuals twice (``cost_measured`` has them)."""
    fact = row.get("cost_observed_fact")
    if not isinstance(fact, dict) or "values" not in fact:
        return
    measured = row.get("cost_measured")
    if not isinstance(measured, dict) or not measured:
        return
    extra = {key: value for key, value in (fact.get("values") or {}).items()
             if key not in measured}
    fact.pop("values", None)
    if extra:
        fact["values_not_scored"] = extra
    ref = _pair_ref(row, "cost_observed_fact.values")
    if ref:
        fact["values_ref"] = ref


def _compact_pair_risk(row):
    """Drop UNPREDICTED 'did not happen' events from the reading copy.

    They contribute nothing a prediction can learn from (the calibration
    already counts them), while a predicted event, an observed occurrence and
    an UNKNOWN label are kept exactly. The omitted count and the reference
    that restores them travel with the row.
    """
    risk = row.get("risk_actual")
    if not isinstance(risk, list):
        return
    kept = [event for event in risk if isinstance(event, dict) and (
        event.get("predicted") or event.get("label") in ("occurred", None))]
    omitted = len(risk) - len(kept)
    if not omitted:
        return
    row["risk_actual"] = kept
    row["risk_actual_omitted"] = {
        "n": omitted,
        "reason": ("unpredicted events labelled 'not_occurred': counted in "
                   "the calibration population, not repeated per case"),
        "detail_ref": _pair_ref(row, "risk_actual"),
    }


def compact_feedback_pair(value):
    """Project current or cached pairs without changing labels or coverage."""
    row=copy.deepcopy(value)
    for key in ('benefit','g_observed_fact'):
        rule=(row.get(key) or {}).get('acceptance_rule')
        if isinstance(rule,dict) and rule.get('fingerprint') and row.get('evaluation_id'):
            row[key]['acceptance_rule']={k:rule[k] for k in ('checker','fingerprint') if k in rule}
            row[key]['acceptance_rule']['report_ref']=(
                f"orx inspect --bank evaluations --evaluation {row['evaluation_id']} #benefit.acceptance_rule")
    observation=row.get('implementation_observation') or {}
    audit=observation.get('semantic_attribution')
    if isinstance(audit,dict):
        observation['semantic_attribution']={k:copy.deepcopy(audit[k]) for k in
            ('method_fidelity','failure_source','evidence_refs','check_action_id') if k in audit}
    actual=(row.get('cost_observed_fact') or {}).get('method_actual')
    if isinstance(actual,dict):
        for key in ('method_actual','method_planned'):
            steps=(row.get(key) or {}).get('steps')
            if isinstance(steps,list) and actual.get('steps')==steps:
                actual.pop('steps');actual['steps_ref']=key+'.steps'
                break
    _compact_pair_methods(row)
    _compact_pair_risk(row)
    _compact_pair_implementation(row)
    _compact_pair_cost_fact(row)
    _compact_pair_task_check(row)
    _compact_pair_execution_conditions(row)
    _compact_pair_cost_values(row)
    return row


# ---------------------------------------------------------------------------
# provider-only compact history feedback
# ---------------------------------------------------------------------------
#
# The STORED published feedback block is the calibration material: it keeps
# the full per-episode facts and is what ``orx calibration`` and the
# evaluation store expose. What a prediction REQUEST needs is smaller: the
# relevant part of each pair (conditions, the recorded plan/actual
# difference, the original prediction, the real observation, the error, the
# recorded attribution and a reference that restores the rest). This
# projection is DERIVED on every read and never written back, so the raw
# evidence keeps its data semantics.
#
# Nothing here makes a new semantic judgement. Every value is copied from a
# structure the close-out already recorded; when a field is missing it stays
# missing (an unknown risk label never becomes "did not happen" and a crash
# never becomes G=0).

#: Version of the PROVIDER-ONLY feedback projection.
PROVIDER_FEEDBACK_VERSION = "wm-pairs-provider/1"

#: Default character budgets of the three INDEPENDENT provider-only views.
#: They are ceilings, not targets: a block is only as large as its material.
#: The full history stays in the stored block and the evaluation store.
DEFAULT_HISTORY_FEEDBACK_CHARS = 30000
#: Stage E/1: the G/C/R blocks now carry up to 10 LOCAL cases each plus the
#: small GLOBAL statistics. The old 16,000 ceiling assumed a handful of
#: condition groups; a compact case is ~300-450 chars, so the three blocks
#: need more room. This is still a CEILING (per-block trimming keeps the
#: grades/global scalars and records every omission), not a target, and the
#: WHOLE request bound remains the authority.
DEFAULT_GCR_EVIDENCE_CHARS = 24000
DEFAULT_NONCORE_CHARS = 20000


def _env_int(name, default):
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        value = int(float(str(raw).strip()))
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def history_feedback_budget_chars():
    return _env_int("OR_HARNESS_HISTORY_FEEDBACK_CHARS",
                    DEFAULT_HISTORY_FEEDBACK_CHARS)


def gcr_evidence_budget_chars():
    return _env_int("OR_HARNESS_GCR_EVIDENCE_CHARS",
                    DEFAULT_GCR_EVIDENCE_CHARS)


def noncore_budget_chars():
    return _env_int("OR_HARNESS_NONCORE_CHARS", DEFAULT_NONCORE_CHARS)


#: Short codes for the selection reasons. The reason SENTENCES stay in the
#: stored selection; the provider gets the codes plus one block-level legend.
_SELECTION_REASON_CODES = (
    ("a recalled hit names", "recall_execution"),
    ("the candidate's declared method shares", "candidate_method_tokens"),
    ("same task as the current problem", "same_task"),
    ("same structural cell", "same_structural_cell"),
    ("same family as the current problem", "same_family"),
    ("same strategy-id tokens", "same_strategy_tokens"),
    ("background material", "background"),
)

SELECTION_REASON_LEGEND = {
    "recall_execution": ("a recalled hit names one of this pair's "
                         "executions"),
    "candidate_method_tokens": ("the current candidates' declared methods "
                                "share method tokens with this pair's plan"),
    "same_task": "same task as the current problem",
    "same_structural_cell": ("same structural cell (coupling bins) as the "
                             "current problem"),
    "same_family": "same family as the current problem (weaker than a cell)",
    "same_strategy_tokens": ("same strategy-id tokens (a weak hint, never "
                             "proof of method identity)"),
    "background": ("background material: no direct relevance signal to this "
                   "decision, carried only to avoid an empty history"),
}


def _clip(value, limit):
    """A clipped scalar WITH the fact that it was clipped.

    Returns ``(text, clipped)``. A clipped value is never presented as the
    whole value: the caller attaches a ``*_ref`` that restores it.
    """
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text, False
    return text[:limit], True


def _evaluation_ref(row):
    evaluation_id = row.get("evaluation_id")
    if not evaluation_id:
        return None
    return (f"orx inspect --bank evaluations --evaluation {evaluation_id}")


def _field_ref(ref, field):
    """A SHORT pointer at one field of a pair's detail reference."""
    return f"{ref} #{field}" if ref else f"orx inspect #{field}"


def _provider_selection_reasons(reasons):
    codes = []
    for reason in reasons or []:
        text = str(reason)
        code = next((code for prefix, code in _SELECTION_REASON_CODES
                     if text.startswith(prefix)), None)
        if code is None:
            clipped, _ = _clip(text, 100)
            code = f"other: {clipped}"
        if code not in codes:
            codes.append(code)
    return codes


def _provider_steps(value, ref):
    """The steps to SEND for one method: short lists whole, long lists by ref.

    A plan and a performed method are different facts; this never collapses
    one into the other. When the steps were already de-duplicated (an
    identical performed method is pointed at, not repeated), the pointer is
    kept as recorded.
    """
    out = {}
    steps = value.get("steps")
    if not isinstance(steps, (list, str)) and value.get("steps_ref"):
        # The plan points at the performed steps (identical method content
        # is carried once); keep that pointer as recorded.
        out["steps_ref"] = value["steps_ref"]
    if isinstance(steps, list):
        text_steps = [str(step) for step in steps if step is not None]
        out["n_steps"] = value.get("n_steps") or len(text_steps)
        if len(text_steps) <= 4 and chars(text_steps) <= 360:
            out["steps"] = text_steps
        elif text_steps:
            out["steps_ref"] = ref
            out["first_step"], _ = _clip(text_steps[0], 160)
    elif isinstance(steps, str) and steps:
        out["steps"] = steps
    if isinstance(value.get("step_differences"), list):
        differences = []
        for diff in value["step_differences"][:3]:
            if not isinstance(diff, dict):
                continue
            planned, _ = _clip(diff.get("planned"), 120)
            actual, _ = _clip(diff.get("actual"), 120)
            entry = {"step": diff.get("step"), "planned": planned,
                     "actual": actual}
            differences.append(entry)
        if differences:
            out["step_differences"] = differences
            if len(value["step_differences"]) > 3:
                out["step_differences_ref"] = ref
    return out


def _provider_method(row, ref):
    """The plan, the actual method and their RECORDED difference, compactly."""
    out = {}
    planned = row.get("method_planned")
    if isinstance(planned, dict) and planned:
        block = {}
        for key in ("name", "why", "fallback"):
            if planned.get(key) not in (None, ""):
                text, clipped = _clip(planned[key], 160)
                block[key] = text
                if clipped:
                    block[key + "_ref"] = f"{ref}.method_planned.{key}"
        block.update(_provider_steps(planned, ref + ".method_planned.steps"))
        if block:
            out["method_planned"] = block
    actual = row.get("method_actual")
    if isinstance(actual, dict) and actual:
        block = {}
        if actual.get("name") not in (None, ""):
            block["name"] = copy.deepcopy(actual["name"])
        block.update(_provider_steps(actual, ref + ".method_actual.steps"))
        if block:
            out["method_actual"] = block
    deviation = row.get("method_deviation")
    if isinstance(deviation, dict) and deviation:
        compact = {}
        for key in ("state", "actual_name", "planned_name"):
            if deviation.get(key) not in (None, ""):
                compact[key] = copy.deepcopy(deviation[key])
        differences = []
        for diff in (deviation.get("differences") or [])[:3]:
            if not isinstance(diff, dict):
                continue
            planned, _ = _clip(diff.get("planned"), 120)
            actual, _ = _clip(diff.get("actual"), 120)
            differences.append({"step": diff.get("step"),
                                "planned": planned, "actual": actual})
        if differences:
            compact["differences"] = differences
        if deviation.get("note"):
            compact["note"], clipped = _clip(deviation["note"], 160)
            if clipped:
                compact["note_ref"] = f"{ref}.method_deviation"
        if compact:
            out["method_deviation"] = compact
    return out


def _provider_g_attribution(attribution, ref):
    if not isinstance(attribution, dict):
        return {}
    out = {}
    for key in ("method_fidelity", "failure_source", "evidence_status", "acceptance_scope", "source"):
        if attribution.get(key) not in (None, ""):
            out[key], _ = _clip(attribution[key], 120)
    if attribution.get("assessment") not in (None, ""):
        out["assessment"], clipped = _clip(attribution["assessment"], 280)
        if clipped:
            out["assessment_ref"] = ref + ".assessment"
    return out


def _provider_benefit(row, ref):
    """The original prediction, the real observation, the error and the
    attribution the close-out RECORDED (never a cause inferred here)."""
    benefit = row.get("benefit")
    if not isinstance(benefit, dict) or not benefit:
        return {}
    out = {}
    for key in ("metric", "observation_rule_version", "label_eligibility",
                "predicted", "observed", "signed_error", "abs_error",
                "interval_kind", "interval_coverage", "eligibility", "attribution_ref"):
        if benefit.get(key) is not None:
            out[key] = copy.deepcopy(benefit[key])
    if benefit.get("reason") is not None:
        text, clipped = _clip(benefit["reason"], 200)
        out["reason"] = text
        if clipped:
            out["reason_ref"] = f"{ref}.benefit.reason"
    rule = benefit.get("acceptance_rule")
    if isinstance(rule, dict) and rule:
        compact_rule = {key: copy.deepcopy(rule[key]) for key in
                        ("checker", "fingerprint", "report_ref")
                        if key in rule}
        if rule.get("spec") is not None:
            compact_rule["spec_ref"] = (f"{ref}.benefit."
                                        "acceptance_rule.spec")
        if compact_rule:
            out["acceptance_rule"] = compact_rule
    refs = benefit.get("evidence_refs")
    if isinstance(refs, list) and refs:
        out["evidence_refs"] = [_clip(str(item), 120)[0]
                                for item in refs[:4]]
        if len(refs) > 4:
            out["evidence_refs_ref"] = f"{ref}.benefit.evidence_refs"
    notes = benefit.get("notes")
    if isinstance(notes, list) and notes:
        out["notes"] = [_clip(str(item), 160)[0] for item in notes[:2]]
        if len(notes) > 2:
            out["notes_ref"] = f"{ref}.benefit.notes"
    attribution = _provider_g_attribution(benefit.get('attribution'), ref + '.benefit.attribution')
    if attribution:
        out['attribution'] = attribution
    return out


def _provider_observed_fact(row, ref):
    fact = row.get("g_observed_fact")
    if not isinstance(fact, dict) or not fact:
        return {}
    out = {}
    for key in ("metric", "observation_rule_version", "observed",
                "label_eligibility", "acceptance_id", "note", "attribution_ref"):
        if fact.get(key) is not None:
            out[key] = copy.deepcopy(fact[key])
    attribution = _provider_g_attribution(fact.get('attribution'), ref + '.g_observed_fact.attribution')
    if attribution:
        out['attribution'] = attribution
    rule = fact.get("acceptance_rule")
    if isinstance(rule, dict) and rule:
        out["acceptance_rule"] = {
            key: copy.deepcopy(rule[key]) for key in
            ("checker", "fingerprint", "report_ref") if key in rule}
    refs = fact.get("evidence_refs")
    if isinstance(refs, list) and refs:
        out["evidence_refs"] = [_clip(str(item), 120)[0]
                                for item in refs[:4]]
        if len(refs) > 4:
            out["evidence_refs_ref"] = f"{ref}.g_observed_fact.evidence_refs"
    return out


def _provider_cost(row, ref):
    out = {}
    for key in ("cost_log_ratio", "cost_measured",
                "cost_observed_unpredicted"):
        if row.get(key) not in (None, {}, []):
            out[key] = copy.deepcopy(row[key])
    fact = row.get("cost_observed_fact")
    if isinstance(fact, dict) and fact:
        compact = {key: copy.deepcopy(fact[key]) for key in
                   ("dimension_scopes", "boundary_exclusions",
                    "observation_rule_version", "values_not_scored",
                    "values_ref", "detail_ref")
                   if fact.get(key) not in (None, {}, [], "")}
        lower_bounds = fact.get("lower_bounds")
        if isinstance(lower_bounds, dict) and lower_bounds:
            compact["lower_bounds"] = {}
            for dim, entry in lower_bounds.items():
                if not isinstance(entry, dict):
                    compact["lower_bounds"][dim] = copy.deepcopy(entry)
                    continue
                kept = {key: copy.deepcopy(entry[key]) for key in
                        ("value", "scope", "boundary_state")
                        if entry.get(key) is not None}
                compact["lower_bounds"][dim] = kept or copy.deepcopy(entry)
            compact["lower_bounds_ref"] = _field_ref(
                ref, "cost_observed_fact.lower_bounds")
        conditions = fact.get("conditions")
        if isinstance(conditions, dict) and conditions:
            kept = {"family": conditions.get("family")}
            solver = conditions.get("solver_conditions")
            if isinstance(solver, dict) and solver.get("solver"):
                kept["solver_conditions"] = {"solver": solver["solver"]}
            kept = {key: value for key, value in kept.items()
                    if value not in (None, {}, [], "")}
            if kept:
                compact["conditions"] = kept
            compact["conditions_ref"] = _field_ref(
                ref, "cost_observed_fact.conditions")
        if compact:
            out["cost_observed_fact"] = compact
    return out


def _provider_risk(row, ref):
    """What was predicted, what happened, and an UNKNOWN kept unknown.

    The predicted probability is carried ONCE (``risk_predicted``); an
    observed entry points at it and never restates it. Unpredicted
    "not_occurred" events are already removed by ``compact_feedback_pair``;
    the omission count and its reference travel with the row.
    """
    out = {}
    predicted = row.get("risk_predicted")
    if isinstance(predicted, list) and predicted:
        entries = []
        for event in predicted[:5]:
            if not isinstance(event, dict):
                continue
            entry = {"event": event.get("event")}
            if event.get("probability") is not None:
                entry["probability"] = copy.deepcopy(event["probability"])
            if event.get("probability_ref"):
                entry["probability_ref"] = event["probability_ref"]
            entries.append(entry)
        if entries:
            out["risk_predicted"] = entries
    observed = row.get("risk_actual")
    if isinstance(observed, list) and observed:
        rules = []
        for event in observed:
            if not isinstance(event, dict):
                continue
            version = event.get("observation_rule_version")
            if version and str(version) not in rules:
                rules.append(str(version))
        rule_codes = {version: chr(ord("a") + index)
                      for index, version in enumerate(rules)}
        entries = []
        for event in observed[:6]:
            if not isinstance(event, dict):
                continue
            entry = {key: copy.deepcopy(event[key]) for key in
                     ("event", "label", "label_basis", "predicted",
                      "probability_ref", "brier") if key in event}
            version = event.get("observation_rule_version")
            if version and len(rules) > 1:
                entry["rule"] = rule_codes[str(version)]
            entries.append(entry)
        if entries:
            out["risk_actual"] = entries
            if len(rules) == 1:
                # ONE rule covers every entry: state it once at the row level
                # instead of repeating it per event.
                out["risk_observation_rule_version"] = rules[0]
            elif rules:
                out["risk_observation_rule_versions"] = rule_codes
    omitted = row.get("risk_actual_omitted")
    if isinstance(omitted, dict) and omitted:
        compact_omitted = {
            "n": omitted.get("n"),
            "reason": ("unpredicted not_occurred: counted by the "
                       "calibration, not repeated per case"),
            "detail_ref": omitted.get("detail_ref")}
        out["risk_actual_omitted"] = {
            key: value for key, value in compact_omitted.items()
            if value is not None}
    return out


def _provider_execution(row, ref):
    out = {}
    for key in ("execution_status", "failure_classes"):
        if row.get(key) not in (None, {}, []):
            out[key] = copy.deepcopy(row[key])
    check = row.get("task_check")
    if isinstance(check, dict) and check:
        out["task_check"] = {key: copy.deepcopy(check[key]) for key in
                             ("state", "action_id", "scope_ref",
                              "check_report_ref") if key in check}
    conditions = row.get("execution_conditions")
    if isinstance(conditions, dict) and conditions:
        compact = {key: copy.deepcopy(conditions[key]) for key in
                   ("comparable", "unknown", "execution_id")
                   if conditions.get(key) not in (None, {}, [], "")}
        key = conditions.get("comparison_key")
        if key not in (None, {}, []):
            if chars(key) <= 300:
                compact["comparison_key"] = copy.deepcopy(key)
            else:
                compact["comparison_key_ref"] = (
                    f"{ref}.execution_conditions.comparison_key")
        if compact:
            out["execution_conditions"] = compact
    observation = row.get("implementation_observation")
    if isinstance(observation, dict) and observation:
        kept = {}
        for key in ("event", "comparable", "executed", "exit_code",
                    "label", "observation_rule_version",
                    "result_received", "scope", "task_id",
                    "solver_interface", "reusable_script_ref",
                    "predicted_probability_ref", "selection_action_id", "complete"):
            if observation.get(key) not in (None, "", [], {}):
                kept[key] = copy.deepcopy(observation[key])
        if observation.get("reason") is not None:
            text, clipped = _clip(observation["reason"], 160)
            kept["reason"] = text
            if clipped:
                kept["reason_ref"] = _field_ref(
                    ref, "implementation_observation.reason")
        audit = observation.get("semantic_attribution")
        if isinstance(audit, dict) and audit:
            kept["semantic_attribution"] = {
                key: copy.deepcopy(audit[key]) for key in
                ("method_fidelity", "failure_source", "evidence_refs",
                 "check_action_id") if key in audit}
        if kept:
            out["implementation_observation"] = kept
    blocked = row.get("blocked_dimensions")
    if isinstance(blocked, dict) and blocked:
        out["blocked_dimensions"] = {
            str(dim): (len(entries) if isinstance(entries, (list, tuple, set))
                       else 1)
            for dim, entries in blocked.items()}
        out["blocked_dimensions_ref"] = f"{ref}.blocked_dimensions"
    for key in ("exclusion_reasons", "note"):
        if row.get(key) not in (None, "", [], {}):
            value = row[key]
            if key == "note":
                out[key], _ = _clip(value, 200)
            else:
                out[key] = copy.deepcopy(value)
    return out


def provider_feedback_pair(row):
    """ONE pair in the provider's compact history view (never stored).

    Kept: the conditions the decision was made under, the recorded
    plan/actual method difference, the ORIGINAL prediction, the REAL
    observation, the error, the recorded attribution, the G/C/R measurement
    conventions and eligibility, and a reference that restores every omitted
    detail. A crash keeps no observed G (never 0); an unknown label stays
    unknown.
    """
    if not isinstance(row, dict):
        return {}
    row = compact_feedback_pair(row)
    # Read old pairs as historical material, not numeric feedback for a
    # differently defined current target. The stored pair is untouched.
    from or_harness.world_model.contracts import R_OBSERVATION_VERSION, RETRIES_FUTURE_VERSION
    impl = row.get("implementation_observation") or {}
    impl_rule = impl.get("observation_rule_version") or next((e.get("observation_rule_version")
        for e in row.get("risk_actual") or [] if e.get("event") == "implementation_failure"), None)
    if impl_rule != R_OBSERVATION_VERSION:
        impl.pop("predicted_probability_ref", None)
        impl.pop("predicted_probability", None)
        impl["comparable"] = False
        row["risk_predicted"] = [e for e in row.get("risk_predicted") or []
                                 if e.get("event") != "implementation_failure"]
        for event in row.get("risk_actual") or []:
            if event.get("event") == "implementation_failure":
                for key in ("brier", "predicted", "predicted_probability", "probability_ref"):
                    event.pop(key, None)
                event["label_basis"] = "historical_formal"
    fact = row.get("cost_observed_fact") or {}
    rule = (fact.get("dimension_observation_rule_versions") or {}).get("retries") or (fact.get("conditions") or {}).get("retry_observation_rule_version")
    if rule != RETRIES_FUTURE_VERSION:
        for key in ("cost_measured", "cost_predicted", "cost_actual", "cost_log_ratio", "cost_observed_unpredicted"):
            if isinstance(row.get(key), dict):
                row[key].pop("retries", None)
        for key in ("values", "values_not_scored", "lower_bounds"):
            if isinstance(fact.get(key), dict):
                fact[key].pop("retries", None)
    ref = _evaluation_ref(row)
    out = {}
    for key in ("evaluation_id", "prediction_id", "episode_id", "task_id",
                "scope", "relevance", "background"):
        if row.get(key) is not None:
            out[key] = copy.deepcopy(row[key])
    if ref:
        out["detail_ref"] = ref
    if row.get("conditions"):
        out["conditions"] = copy.deepcopy(row["conditions"])
    execution_ids = row.get("execution_ids")
    if isinstance(execution_ids, list) and execution_ids:
        out["execution_ids"] = list(execution_ids[:4])
        if len(execution_ids) > 4:
            out["execution_ids_ref"] = f"{ref}.execution_ids"
    reasons = _provider_selection_reasons(row.get("selected_reason"))
    if reasons:
        out["selected_reason_codes"] = reasons
    method = _provider_method(row, ref or "orx inspect --bank evaluations")
    out.update(method)
    benefit = _provider_benefit(row, ref or "orx inspect")
    if benefit:
        out["benefit"] = benefit
    observed = _provider_observed_fact(row, ref or "orx inspect")
    if observed:
        out["g_observed_fact"] = observed
    out.update(_provider_cost(row, ref or "orx inspect"))
    out.update(_provider_risk(row, ref or "orx inspect"))
    out.update(_provider_execution(row, ref or "orx inspect"))
    return out


#: Optional parts of a provider pair, dropped first when the history block is
#: over its own budget. They are REDUNDANT with a reference or with another
#: carried field, never the prediction/observation/error themselves.
_PROVIDER_OPTIONAL_KEYS = (
    "cost_observed_unpredicted",
    "blocked_dimensions",
    "selected_reason_codes",
)


def _lean_provider_pair(row):
    """The same pair with optional detail dropped (a deterministic reduction).

    Used only when a history block exceeds its provider budget: the facts a
    prediction needs (original prediction, real observation, error, G/C/R
    conventions, eligibility, the RECORDED failure attribution and the
    restore reference) stay. Free-text prose, duplicated audit fields and
    measured values that the row-level ``detail_ref`` restores are dropped
    first, so the number of CASES is preserved before any case is removed.
    The attribution assessment is the LAST thing to go — after prose and
    scaffolding, not before.
    """
    row = copy.deepcopy(row)
    dropped = []
    for key in _PROVIDER_OPTIONAL_KEYS:
        if key in row:
            row.pop(key)
            dropped.append(key)
    method = row.get("method_planned")
    if isinstance(method, dict):
        for key in ("why", "fallback"):
            if key in method:
                method.pop(key)
                dropped.append(f"method_planned.{key}")
    benefit = row.get("benefit")
    if isinstance(benefit, dict):
        if "notes" in benefit:
            benefit.pop("notes", None)
            dropped.append("benefit.notes")
        # ``benefit.attribution`` is the RECORDED cause of the observation
        # (method fidelity / failure source / the auditor's short
        # assessment). It is the highest-value part of a failed case: "the
        # task required integer quantities but the method used continuous
        # variables" is the lesson, while the audit scaffolding around it is
        # not. It is therefore NEVER dropped here — only after every pair has
        # been reduced and before a whole case is removed for budget.
    fact = row.get("cost_observed_fact")
    if isinstance(fact, dict):
        for key in ("lower_bounds", "conditions", "boundary_exclusions",
                    "values_not_scored", "values_ref", "conditions_ref",
                    "lower_bounds_ref"):
            if key in fact:
                fact.pop(key, None)
                dropped.append(f"cost_observed_fact.{key}")
        fact["detail_state"] = "reduced"
    observation = row.get("implementation_observation")
    if isinstance(observation, dict):
        for key in ("reason", "executed", "exit_code", "result_received",
                    "scope", "task_id", "solver_interface"):
            if key in observation:
                observation.pop(key, None)
                dropped.append(f"implementation_observation.{key}")
    for key in ("execution_ids", "execution_ids_ref"):
        if key in row:
            row.pop(key, None)
            dropped.append(key)
    omitted = row.get("risk_actual_omitted")
    if isinstance(omitted, dict) and "reason" in omitted:
        omitted.pop("reason", None)
        dropped.append("risk_actual_omitted.reason")
    if dropped:
        row["reduced"] = True
    return row, dropped


def _provider_comparison(comparison):
    """A same-task side-by-side fact, without repeating whole method lists."""
    if not isinstance(comparison, dict):
        return {}
    out = {key: copy.deepcopy(comparison[key]) for key in
           ("task_id", "acceptance_id", "scope", "metric",
            "observation_rule_version", "acceptance_rule_fingerprint",
            "acceptance_comparison", "problem_identity_digest")
           if comparison.get(key) is not None}
    if comparison.get("execution_conditions") not in (None, {}, []):
        out["execution_conditions"] = copy.deepcopy(
            comparison["execution_conditions"])
    for side in ("left", "right"):
        entry = comparison.get(side)
        if not isinstance(entry, dict):
            continue
        compact = {"evaluation_id": entry.get("evaluation_id")}
        if entry.get("observed") is not None:
            compact["observed"] = copy.deepcopy(entry["observed"])
        refs = entry.get("evidence_refs")
        if isinstance(refs, list) and refs:
            compact["evidence_refs"] = [_clip(str(item), 120)[0]
                                        for item in refs[:4]]
            if len(refs) > 4:
                compact["evidence_refs_ref"] = (
                    f"orx inspect --bank evaluations --evaluation "
                    f"{entry.get('evaluation_id')} #evidence_refs")
        elif isinstance(refs, dict) and refs:
            # Artifact references are keyed by artifact name: keep the
            # RECORDED references (bounded), never the resolved payloads.
            compact["evidence_refs"] = {
                str(key): copy.deepcopy(value)
                for key, value in list(refs.items())[:4]}
            if len(refs) > 4:
                compact["evidence_refs_ref"] = (
                    f"orx inspect --bank evaluations --evaluation "
                    f"{entry.get('evaluation_id')} #evidence_refs")
        method = entry.get("method_actual")
        if isinstance(method, dict) and method:
            name, clipped = _clip(method.get("name") or "", 120)
            compact["method_name"] = name
            if clipped:
                compact["method_name_ref"] = (
                    f"orx inspect --bank evaluations --evaluation "
                    f"{entry.get('evaluation_id')} #method_actual")
            steps = method.get("steps")
            if isinstance(steps, list) and steps:
                compact["n_steps"] = len(steps)
        out[side] = compact
    return out


def provider_paired_feedback(block, *, budget_chars=None):
    """Project a selected feedback block into the provider's compact view.

    The block's own selection (relevance, labels-first ordering, same-task
    comparisons, counts) is preserved: this only shrinks the DELIVERED rows
    and, when the provider-only history budget is exceeded, drops the
    LOWEST-VALUE rows FIRST (the selection already ordered them) and reports
    exactly what was dropped with a reference that restores it.
    """
    block = block or {}
    risk_legends = {**(block.get("risk_reason_legend") or {}),
                    "historical_formal": "historical formal-attempt diagnostic; not a current implementation-window label"}
    budget = (budget_chars if budget_chars is not None
              else history_feedback_budget_chars())
    before_chars = chars(block)
    source_pairs = [pair for pair in (block.get("pairs") or [])
                    if isinstance(pair, dict)]
    projected = [provider_feedback_pair(pair) for pair in source_pairs]
    comparisons = [_provider_comparison(comparison)
                   for comparison in (block.get("same_task_comparisons") or [])
                   if isinstance(comparison, dict)]
    # Only the risk reason codes actually used are explained, once.
    used_risk_codes = set()
    used_selection_codes = set()
    for row in projected:
        for event in row.get("risk_actual") or []:
            if isinstance(event, dict) and event.get("label_basis"):
                used_risk_codes.add(event["label_basis"])
        used_selection_codes.update(row.get("selected_reason_codes") or [])
    base = {key: copy.deepcopy(value) for key, value in block.items()
            if key not in ("pairs", "same_task_comparisons",
                           "risk_reason_legend")}
    base["provider_view_version"] = PROVIDER_FEEDBACK_VERSION
    base["provider_budget_chars"] = budget
    legend = {code: text for code, text in
              risk_legends.items()
              if code in used_risk_codes}
    if legend:
        base["risk_reason_legend"] = legend
    reason_legend = {code: text for code, text in
                     SELECTION_REASON_LEGEND.items()
                     if code in used_selection_codes}
    if reason_legend:
        base["selection_reason_legend"] = reason_legend
    available = block.get("n_pairs_available")
    base_omitted = int(block.get("n_pairs_omitted") or 0)
    selected = int(block.get("n_pairs_included") or len(projected))
    displayed = list(projected)
    reduced = False
    reduced_fields = []

    def assemble():
        out = copy.deepcopy(base)
        out["pairs"] = displayed
        out["same_task_comparisons"] = comparisons
        # The legends explain only codes the DISPLAYED rows still carry.
        used_risk = {event.get("label_basis")
                     for row in displayed
                     for event in row.get("risk_actual") or []
                     if isinstance(event, dict) and event.get("label_basis")}
        used_selection = {code for row in displayed
                          for code in row.get("selected_reason_codes") or []}
        risk_legend = {code: text for code, text in
                       risk_legends.items()
                       if code in used_risk}
        if risk_legend:
            out["risk_reason_legend"] = risk_legend
        else:
            out.pop("risk_reason_legend", None)
        selection_legend = {code: text for code, text in
                            SELECTION_REASON_LEGEND.items()
                            if code in used_selection}
        if selection_legend:
            out["selection_reason_legend"] = selection_legend
        else:
            out.pop("selection_reason_legend", None)
        n_dropped = len(projected) - len(displayed)
        if available is not None:
            omitted = max(0, int(available) - len(displayed))
        else:
            omitted = base_omitted + n_dropped
        out["n_pairs_included"] = len(displayed)
        out["n_pairs_omitted"] = omitted
        out["n_pairs_provider_dropped"] = n_dropped
        out["n_pairs_selected"] = selected
        out["provider_compaction"] = {
            "source": ("the stored selected block; this view is derived and "
                       "never written back"),
            "before_chars": before_chars,
            "after_chars": None,
            "n_pairs_before": len(projected),
            "n_pairs_displayed": len(displayed),
            "n_pairs_dropped_for_size": n_dropped,
            "rows_reduced": reduced,
            "fields_reduced": list(reduced_fields),
            "full_detail": ("every dropped or reduced field is recoverable "
                            "through the row's `detail_ref` and "
                            "`orx calibration`"),
        }
        if n_dropped:
            refs = []
            for row in projected[len(displayed):][:20]:
                ref = row.get("detail_ref")
                if ref:
                    refs.append(ref)
            out["dropped_pair_refs"] = refs
            if n_dropped > len(refs):
                out["dropped_pair_refs_note"] = (
                    f"{n_dropped - len(refs)} further dropped pair(s) are "
                    "listed by `orx calibration`")
            out["omission_note"] = (
                f"{n_dropped} of {len(projected)} selected pair(s) were "
                "dropped to stay within the provider-only history budget "
                f"({budget} characters); the full records remain in the "
                "stored block, `orx calibration` and the evaluation store")
        if reduced:
            out["reduction_note"] = (
                "free-text/prose fields were replaced by their references to "
                "fit the provider-only history budget; predictions, "
                "observations, errors, conventions and eligibility are "
                "unchanged")
        # ``after_chars`` includes its own serialized width: solve the tiny
        # fixed point so the reported size equals the returned block's size.
        compaction = out["provider_compaction"]
        compaction.pop("after_chars", None)
        size_without_field = chars(out)
        digits = len(str(size_without_field + 15))
        for _ in range(4):
            actual = size_without_field + 15 + digits
            if len(str(actual)) == digits:
                break
            digits = len(str(actual))
        compaction["after_chars"] = size_without_field + 15 + digits
        return out

    out = assemble()
    if chars(out) > budget:
        # Stage 1: reduce optional prose on EVERY row before dropping any row
        # — a per-row reduction is preferred over deleting whole cases.
        leaned = [_lean_provider_pair(row) for row in displayed]
        displayed = [row for row, _dropped in leaned]
        reduced_fields = sorted({field for _row, dropped in leaned
                                 for field in dropped})
        reduced = True
        out = assemble()
    while chars(out) > budget and displayed:
        # Stage 2: drop the LOWEST-VALUE rows first. The selection order is
        # the value order (labeled cases first, execution errors last, then
        # relevance and recency), so the tail is the right end to shed.
        displayed.pop()
        out = assemble()
    if chars(out) > budget and out.get("same_task_comparisons"):
        out["same_task_comparisons"] = []
        out["n_same_task_comparisons_omitted"] = len(comparisons)
        out["same_task_comparisons_note"] = (
            "side-by-side comparisons were omitted to fit the provider-only "
            "history budget; the pair rows and `orx calibration` retain the "
            "facts")
        out["provider_compaction"]["after_chars"] = chars(out)
    return out


def provider_feedback_context_block(block):
    """Backward-compatible name for the provider-only history projection."""
    return provider_paired_feedback(block)


def paired_feedback_view(block):
    """The provider-only view (kept name; no longer the full body copy)."""
    return provider_paired_feedback(block)


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
            "body": (copy.deepcopy(body) if rule else {
                k:copy.deepcopy(v) if k=='do' else atomic(v,ref+' #'+k,field_limit)
                for k,v in body.items()}),
            "evidence": copy.deepcopy(rule.get('evidence') or claim.get('evidence') or []),
            "predicates": copy.deepcopy(entry.predicates), "detail_ref": ref,
            "body_ref": rule.get("body_ref"), "historical_state": not bool(rule) or bool(rule.get("legacy_version")),
            "merged_into":rule.get('merged_into'), "current_task_premises":"requires_assessment" if rule else "historical_scope",
            "merge_pending_into":rule.get('merge_pending_into'),
            "candidate_ref": (ref + " --rule-version candidate" if rule.get("candidate_version") is not None else None)}
