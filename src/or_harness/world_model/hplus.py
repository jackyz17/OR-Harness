"""Frozen binary induction scheduling, durable work and factual receipts.

The solving agent executes; the outer building agent consumes the returned
work. A job never calls a model or writes a method on the agent's behalf.
"""
from __future__ import annotations

import copy
import math
import re
import time

from or_harness.world_model.conditions import digest, get_value, put_value
from or_harness.world_model.induction_gate import meta_read, meta_write

HPLUS_VERSION = "wm-hplus-binary/1"
JOB_PREFIX = "hplus_job:"
MODES = ("auto", "hplus", "nohplus", "random", "frozen", "manual")


def signal_from_payload(raw):
    block = raw if isinstance(raw, dict) else {}
    value = block.get("decision")
    valid = type(value) in (int, float) and value in (0, 1) and math.isfinite(value)
    state = "explicit" if valid else "missing" if "decision" not in block else "invalid"
    conditions = block.get("verification_conditions")
    conditions = conditions if isinstance(conditions, list) else []
    conditions = [{'condition':str(c['condition']),'check_basis':c.get('check_basis'),
                   'prediction_made':True,'fact_bound':False,'effect_verified':False}
                  for c in conditions if isinstance(c, dict) and c.get("condition")]
    claim = str(block.get("claim") or "")
    basis = block.get("basis") or []
    basis = [basis] if isinstance(basis, str) else basis if isinstance(basis, list) else []
    gaps = []
    if state != "explicit":
        gaps.append("decision " + state + "; schedule as 0, not an explicit negative")
    if valid and value == 1:
        if not claim.strip():
            gaps.append("learning hypothesis missing; positive still requires review")
        if not conditions:
            gaps.append("concrete verification missing; positive still requires review")
    elif not basis:
        gaps.append("reason missing")
    raw_value=repr(value) if type(value) is float and not math.isfinite(value) else copy.deepcopy(value)
    return {"version": HPLUS_VERSION, "decision": int(value) if valid else 0,
            "decision_status": state, "raw_decision": raw_value,
            "claim": claim, "basis": [str(x) for x in basis],
            "verification_conditions": conditions, "gaps": gaps,
            "legacy_assessment": block.get("assessment")}


def mode(harness):
    configured = meta_read(harness.store, "induction_mode")
    return configured or "auto"


def configure(harness, value):
    if value not in MODES:
        raise ValueError("induction mode must be one of " + str(MODES))
    prior = meta_read(harness.store, "induction_mode")
    if value=='auto' and prior=='hplus':
        return
    if prior and prior != value and value != 'frozen':
        raise ValueError("induction mode is fixed per memory home; fork a separate control home")
    if value=='frozen' and any(j['state']=='pending' for j in jobs(harness)):
        raise ValueError('cannot freeze a library with pending required induction receipts')
    with harness.store.transaction():
        meta_write(harness.store, "induction_mode", value)


def formal(harness):
    return mode(harness) not in ("manual",) and (mode(harness) != "auto" or bool(
        meta_read(harness.store, "hplus_enabled")))


def freeze_signal(harness, prediction, context, request):
    """Only the first save defines the switch and its pre-result inputs."""
    if prediction.trace.model_info.get('call_state') == 'in_flight':
        return
    if mode(harness)=='auto':
        meta_write(harness.store,'induction_mode','hplus')
    signal = prediction.trace.model_info.setdefault("hplus", signal_from_payload(None))
    signal.setdefault("prediction_id", prediction.prediction_id)
    signal.setdefault("frozen_at", prediction.trace.created_at)
    signal.setdefault("snapshot_id", context.snapshot_id)
    signal.setdefault("context_id", context.context_id)
    signal.setdefault("family", (context.joint.profile or {}).get("family"))
    profile=context.joint.profile or {}
    signal.setdefault("structure", {k:copy.deepcopy(profile[k]) for k in ('problem_class','coupling') if k in profile})
    signal.setdefault("input_ref", put_value(harness.store.conn, harness.store, "hplus_input",
        request.get("prediction_context") or {}))
    signal.setdefault('source_snapshot_ref',put_value(harness.store.conn,harness.store,'hplus_snapshot',context.snapshot))
    signal.setdefault("mode", mode(harness))
    if mode(harness) != "manual":
        meta_write(harness.store, "hplus_enabled", True)


def jobs(harness, *, task_id=None, episode_id=None):
    rows = harness.store.conn.execute("SELECT value FROM meta WHERE key LIKE ? ORDER BY key", (JOB_PREFIX + "%",))
    out = [harness.store.loads(r["value"]) for r in rows]
    return sorted([j for j in out if (task_id is None or j["task_id"] == str(task_id)) and
                   (episode_id is None or j["episode_id"] == episode_id)], key=lambda j: (j["created_at"], j["induction_id"]))


def get_job(harness, induction_id):
    return meta_read(harness.store, JOB_PREFIX + str(induction_id))


def resolve_episode(harness,task_id,episode_id,*,execution_id=None,finish_action_id=None):
    chosen=None
    if execution_id:
        chosen=next((a for a in harness.actions.query(task_id=task_id,action_type='execute_strategy') if a.linked_execution_id==execution_id),None)
        if chosen is None:raise ValueError('execution does not identify a real solving round of this task')
    if finish_action_id:
        finish=harness.actions.get(finish_action_id)
        if not finish or finish.action_type!='finish_task' or str(finish.task_id)!=str(task_id):
            raise ValueError('finish action does not belong to this task')
        if chosen and chosen.episode_id!=finish.episode_id:raise ValueError('execution and finish action select different episodes')
        chosen=finish
    if chosen:
        if episode_id is not None and episode_id!=chosen.episode_id:raise ValueError('episode conflicts with the selected real execution')
        return chosen.episode_id
    if episode_id is not None:return episode_id
    actions=[a for a in harness.actions.query(task_id=task_id,action_type='execute_strategy') if a.linked_execution_id]
    episodes={a.episode_id for a in actions}
    if not episodes:
        episodes={p.candidate.episode_id for p in harness.strategy_predictions.query(task_id=task_id)}
    if len(episodes)==1:return next(iter(episodes))
    if len(episodes)>1:
        choices=[{'episode':a.episode_id,'action_id':a.action_id,'execution_id':a.linked_execution_id,
                  'method':(a.params or {}).get('method') or (a.params or {}).get('strategy_id'),
                  'solver':(a.params or {}).get('solver')} for a in actions]
        raise ValueError('ambiguous episode: choose an existing solving round '+str(choices or sorted(str(e) for e in episodes)))
    return None


def _save_job(harness, job):
    previous=get_job(harness,job['induction_id'])
    if previous and previous['state']=='completed':
        if job.get('receipt_ref') not in (None,previous['receipt_ref']):
            raise ValueError('completed induction has a different immutable receipt')
        job.clear();job.update(previous)
        return
    meta_write(harness.store, JOB_PREFIX + job["induction_id"], job)


def _verification(record, signal):
    report = (record.execution_features or {}).get("task_check") or {}
    declared = report.get("hplus_verification") or []
    if isinstance(declared, dict):
        declared = [declared]
    result = []
    for index, condition in enumerate(signal.get("verification_conditions") or []):
        evidence = next((c for c in declared if isinstance(c, dict) and
            c.get("prediction_id") == signal.get("prediction_id") and c.get("condition_index") == index), {})
        refs = evidence.get("evidence_refs") or []
        def exists(ref):
            if not isinstance(ref,str):return False
            if ref.startswith('artifacts.'):
                artifact=((record.execution_features or {}).get('artifacts') or {}).get(ref.split('.',1)[1])
                return isinstance(artifact,dict) and bool(artifact.get('sha256'))
            if ref in ('task_check.checks','task_check.diffs','task_check.scope'):
                return bool(report.get(ref.split('.',1)[1]))
            return False
        supported=bool(isinstance(evidence.get('source'),str) and evidence['source'].strip() and
                       isinstance(refs,list) and refs and all(exists(ref) for ref in refs))
        state=evidence.get('state','unknown')
        if state not in ('checked','not_checked','unavailable','unknown'):state='unknown'
        result.append({"condition_index": index, "condition": condition["condition"],
            "state": state if supported else "unknown",
            'finding':evidence.get('finding'),
            "source": evidence.get("source"), "evidence_refs": refs,
            "reason": None if supported else "no sourced completion of the frozen verification"})
    return result


def schedule_episode(harness, task_id, episode_id):
    """Called before retention; idempotent by episode and bound prediction."""
    scheduled, unobserved = [], []
    predictions = harness.strategy_predictions.query(task_id=task_id, episode_id=episode_id,exact_episode=True)
    for prediction in predictions:
        signal = (prediction.trace.model_info or {}).get("hplus")
        if not signal or signal.get("version") != HPLUS_VERSION:
            continue                         # old records keep their old protocol
        action_id = (prediction.trace.model_info or {}).get("bound_action_id")
        action = harness.actions.get(action_id) if action_id else None
        if not action or not action.linked_execution_id:
            unobserved.append(prediction.prediction_id)
            continue
        execution_id = str(action.linked_execution_id)
        actual_episode=action.episode_id
        identity = "ij_" + digest([str(task_id), actual_episode, prediction.prediction_id])[:24]
        existing = get_job(harness, identity)
        if existing:
            scheduled.append(existing)
            continue
        record = harness.bank.get(execution_id) or harness.bank.get_pending(execution_id)
        if record is None or record.source != "executed":
            unobserved.append(prediction.prediction_id)
            continue
        from or_harness.strategy.rule_versions import _facts_ref
        selected = signal["decision"] == 1
        branch = signal.get("mode", mode(harness))
        if branch == "nohplus":
            selected = True
        elif branch == "random":
            allocation = meta_read(harness.store, "hplus_random_allocation") or []
            selected = str(task_id) in allocation
        elif branch == "frozen":
            selected = False
        elif branch == "manual":
            continue
        with harness.store.transaction() as conn:
            job = {"induction_id": identity, "version": HPLUS_VERSION,
                "task_id": str(task_id), "episode_id": actual_episode,
                "prediction_id": prediction.prediction_id, "signal": copy.deepcopy(signal),
                "solve_action_id": action.action_id, "execution_ids": [execution_id],
                "source_fact_ref": _facts_ref(harness, conn, record),
                "verification": _verification(record, signal), "mode": branch,
                "selected": selected, "state": "pending" if selected else "skipped",
                "created_at": time.time(), "action_id": None, "receipt_ref": None,
                "control_snapshot_ref": signal.get("input_ref") if branch == "nohplus" and signal["decision"] == 0 else None}
            previous=get_job(harness,identity)
            if previous:job=previous
            else:
                if selected:
                    from or_harness.strategy.evidence_access import hold_execution
                    hold_execution(harness,identity,execution_id,record=record)
                _save_job(harness,job)
        scheduled.append(job)
    return schedule_view(scheduled, unobserved)


def schedule_view(items, unobserved=()):
    pending = [j for j in items if j["state"] == "pending"]
    return {"version": HPLUS_VERSION, "jobs": items,
            "pending_ids": [j["induction_id"] for j in pending],
            "completed": not pending, "unobserved_prediction_ids": list(unobserved),
            "next_actions": [{"induction_id": j["induction_id"],
                "material": "orx induction-material --induction-id " + j["induction_id"],
                "complete": "orx induce --induction-id " + j["induction_id"] + " --relation JSON (or --review JSON --note REASON)",
                "required": "one short submission or a sourced empty review receipt"} for j in pending],
            "handoff": "outer building agent must settle pending_ids before declaring library building complete"}


def require_job(harness, induction_id=None, *, task_id=None, material_id=None, allow_completed=False):
    if not formal(harness):
        return None
    if material_id and not induction_id:
        session = get_value(harness.store, material_id)
        induction_id = session.get("induction_id")
    elif material_id and induction_id and get_value(harness.store,material_id).get('induction_id') != induction_id:
        raise ValueError('material session belongs to a different induction job')
    active = getattr(harness, "_active_induction_job", None)
    induction_id = induction_id or (active or {}).get("induction_id")
    candidates = [get_job(harness, induction_id)] if induction_id else [
        j for j in jobs(harness, task_id=task_id) if j["state"] == "pending" and j["selected"]]
    if len(candidates) != 1 or not candidates[0]:
        raise ValueError("H+ building requires one positive executed induction job; pass --induction-id (inspect induction-status)")
    job = candidates[0]
    if task_id is not None and str(task_id) != job["task_id"]:
        raise ValueError("induction task does not match the triggering job")
    if not job["selected"] or (job["mode"] not in ("nohplus", "random") and job["signal"]["decision"] != 1):
        raise ValueError("H+=0 cannot authorize induction material or writes in the H+ branch")
    if job["state"] != "pending" and not (allow_completed and job["state"] == "completed"):
        raise ValueError("induction job is not pending")
    if mode(harness) == "frozen":
        raise ValueError("the strategy library is frozen for testing")
    return job


def submission_job(harness,induction_id,relations,submission_digest=None):
    if not formal(harness):return None
    declared={str(r['induction_id']) for r in relations or [] if r.get('induction_id')}
    if induction_id:declared.add(str(induction_id))
    material_ids={r['material_id'] for r in relations or [] if r.get('material_id')}
    for mid in material_ids:
        session=get_value(harness.store,mid)
        if not session.get('induction_id'):raise ValueError('material is not attached to a positive induction job')
        declared.add(session['induction_id'])
    if len(declared)>1:raise ValueError('conflicting induction IDs in submission/material: '+str(sorted(declared)))
    resolved=next(iter(declared),None)
    if resolved is None and submission_digest:
        previous=[j for j in jobs(harness) if j['state']=='completed' and j.get('submission_digest')==submission_digest]
        if len(previous)>1:raise ValueError('submission matches several completed reviews; choose its induction-id')
        if previous:resolved=previous[0]['induction_id']
    if resolved is None and relations:
        evidence={str(e if isinstance(e,str) else e.get('execution_id')) for r in relations for e in r.get('evidence') or []}
        candidates=[]
        for job in jobs(harness):
            if job['state']!='pending' or not job['selected']:continue
            shown=set(job['execution_ids'])
            for ref in material_sessions(harness,job):shown.update(get_value(harness.store,ref).get('shown_execution_ids') or [])
            if shown&evidence:candidates.append(job)
        if len(candidates)!=1:
            raise ValueError('submission evidence does not identify one positive triggering job; choose its induction-id')
        resolved=candidates[0]['induction_id']
    job=require_job(harness,resolved,allow_completed=True)
    for mid in material_ids:
        if mid not in material_sessions(harness,job):raise ValueError('material session was not delivered for this job: '+mid)
    return job


def material_sessions(harness,job):
    refs=list(job.get('material_sessions') or [])
    if not refs and job.get('material_ref'):
        saved=get_value(harness.store,job['material_ref'])
        if saved.get('material_id'):refs.append(saved['material_id'])
    return refs


def aggregate_material(harness,job):
    refs=material_sessions(harness,job)
    identities={};shown=list(job['execution_ids'])
    for ref in refs:
        session=get_value(harness.store,ref)
        if session.get('induction_id') not in (None,job['induction_id']):
            raise ValueError('material session belongs to another induction job')
        shown.extend(session.get('shown_execution_ids') or [])
        identities.update(session.get('identities') or {})
    from or_harness.strategy.evidence_access import resolve_execution
    from or_harness.strategy.evidence_views import task_identity
    for eid in shown:
        record=resolve_execution(harness,eid)
        if record:identities[eid]=task_identity(harness,record)
    value={'induction_id':job['induction_id'],'task_id':job['task_id'],'episode_id':job['episode_id'],
        'shown_execution_ids':list(dict.fromkeys(shown)),'identities':identities,'read_refs':refs}
    with harness.store.transaction() as conn:
        reference=put_value(conn,harness.store,'induction_material',value)
    return reference,value


def material_for_job(harness,job,builder,**kwargs):
    """Query current evidence every time; retain only delivered references."""
    job=get_job(harness,job['induction_id'])
    if job['state']=='completed':
        if job.get('material_ref') and not job.get('material_policy'):
            return {**copy.deepcopy(get_value(harness.store,job['material_ref'])),'historical':True}
        return {'material_assembled':False,'induction_id':job['induction_id'],'state':'completed',
                'receipt_ref':job['receipt_ref'],'evidence_ids':job.get('selected_execution_ids') or []}
    from or_harness.strategy.evidence_views import atomic
    gate={'state':'allow','source':'frozen_hplus','induction_id':job['induction_id'],
        'prediction_id':job['prediction_id'],'decision':job['signal']['decision'],
        'reason':'positive executed job requires one agent review',
        'hypothesis':atomic(job['signal'].get('claim'),job['signal'].get('input_ref'),450),
        'verification':atomic(job['signal'].get('verification_conditions'),job['signal'].get('input_ref'),450),
        'source_fact_ref':job['source_fact_ref'],'baseline_snapshot_ref':job['signal'].get('source_snapshot_ref')}
    if job.get('control_snapshot_ref'):
        kwargs['frozen_knowledge']=frozen_rules(harness,job)
        gate['knowledge_basis']='prediction-time snapshot'
    from or_harness.strategy.evidence_access import held_records
    result=builder(harness,gate=gate,task_id=job['task_id'],seed_execution_ids=job['execution_ids'],
        retained_execution_ids=list(held_records(harness,job['induction_id'])),induction_id=job['induction_id'],
        episode_id=job['episode_id'],**kwargs)
    with harness.store.transaction():
        current=get_job(harness,job['induction_id'])
        if current['state']=='completed':return result
        session=get_value(harness.store,result['material_id'])
        current['material_policy']='live-query/1'
        current['material_sessions']=list(dict.fromkeys(material_sessions(harness,current)+[result['material_id']]))
        current['selected_execution_ids']=list(dict.fromkeys((current.get('selected_execution_ids') or [])+session['shown_execution_ids']))
        _save_job(harness,current)
    return result


def reserve_action(harness, job):
    with harness.store.transaction():
        current=get_job(harness,job['induction_id'])
        if current['state']=='completed':job.clear();job.update(current);return
        job.clear();job.update(current)
        if current.get('action_id'):
            return
        maintenance = harness._begin_induce_action(None, False)
        job["action_id"] = maintenance["action_id"]
        job["knowledge_before"] = maintenance["knowledge_before"]
        job["maintenance_episode_id"] = maintenance["episode_id"]
        harness.actions.amend_action_params(job["action_id"], induction_id=job["induction_id"],
            capability_prediction_id=job["prediction_id"], source_execution_ids=job["execution_ids"])
        _save_job(harness, job)


def put_submission(harness,arguments,review):
    return put_value(harness.store.conn,harness.store,'induction_submission',{'arguments':arguments,'review':review})


def submission_conflict(harness,job,arguments,review):
    original=get_value(harness.store,job['submission_ref']) if job.get('submission_ref') else {}
    old=original.get('arguments') or {}
    fields=['arguments.'+key for key in sorted(set(old)|set(arguments)) if old.get(key)!=arguments.get(key)]
    if original.get('review')!=review:fields.append('review')
    return ValueError('completed induction submission conflicts in '+', '.join(fields or ['payload'])+
        '; reuse the original payload or inspect receipt '+str(job.get('receipt_ref')))


def complete(harness, job, result, review=None):
    if review is not None and not isinstance(review,dict):
        raise ValueError('review must be a JSON object')
    review = copy.deepcopy(review or {})
    if review.get('hypothesis_outcome','unknown') not in ('supported','refuted','unknown'):
        raise ValueError('hypothesis_outcome must be supported, refuted or unknown')
    comparison=review.get('comparison')
    if comparison is not None and (not isinstance(comparison,dict) or
            not isinstance(comparison.get('evidence_refs',[]),list) or
            not all(isinstance(ref,str) for ref in comparison.get('evidence_refs',[]))):
        raise ValueError('comparison must carry a list of sourced evidence references')
    action = harness.actions.get(job["action_id"])
    if not action or action.action_type != "induce" or not action.ended_at:
        raise ValueError("completion requires the actual finished induction action")
    source=set(job['execution_ids'])|set((action.outcome or {}).get('execution_ids') or [])
    material_id,material=aggregate_material(harness,job)
    source.update((job.get('submitted_execution_ids') or []))
    from or_harness.world_model.hplus_support import increment
    updates=[]
    submission=get_value(harness.store,job['submission_ref']) if job.get('submission_ref') else {}
    submitted=(submission.get('arguments') or {}).get('relations') or []
    for index,row in enumerate(result.get('relations') or []):
        if not row.get('saved'):continue
        update={k:row.get(k) for k in ('version','body_ref','operation','new_version')}
        update['entry_id']=row['saved']
        comparison=(submitted[index].get('mechanism_comparison') if index<len(submitted) else None) or review.get('comparison')
        update['increment']=increment(harness,job,update,comparison)
        update['substantive']=update['increment']['substantive']
        updates.append(update)
    from or_harness.strategy.evidence_access import resolve_execution
    trigger=resolve_execution(harness,job['execution_ids'][0])
    if trigger:
        job['verification'] = _verification(trigger,job['signal'])
    conclusion = review.get("conclusion") or ("short rules submitted for existing validation" if updates else "review completed without a new rule")
    fact = {"induction_id": job["induction_id"], "prediction_id": job["prediction_id"],
        "solve_action_id": job["solve_action_id"], "induction_action_id": job["action_id"],
        "mode": job["mode"], "decision": job["signal"]["decision"],
        "input_ref": job["signal"].get("input_ref"), "material_id": material_id, "material_read_refs":material.get("read_refs") or [],
        'baseline_snapshot_ref':job['signal'].get('source_snapshot_ref'),
        "source_execution_ids": sorted(source), "source_fact_ref": job["source_fact_ref"],
        "verification": job["verification"], "rule_updates": updates,
        "outcome": review.get("outcome") or ("knowledge_submitted" if updates else "insufficient_material"),
        "conclusion": str(conclusion), "hypothesis_outcome": review.get("hypothesis_outcome", "unknown"),
        "comparison": review.get("comparison"), "deviation_reason": review.get("deviation_reason"),
        "created_at": time.time(), "observation_rule_version":"wm-hplus-review/2"}
    if fact["outcome"] not in ("knowledge_submitted", "new_rule", "revised", "evidence_added", "already_covered",
                               "no_increment", "hypothesis_refuted", "insufficient_material"):
        raise ValueError("unknown induction review outcome")
    from or_harness.world_model.hplus_support import supported_comparison
    if fact.get('comparison'):supported_comparison(harness,job,fact['comparison'],time.time())
    for update in updates:
        comp=get_value(harness.store,update['increment']['comparison_ref'])
        if comp:supported_comparison(harness,job,comp,time.time())
    job["receipt_ref"] = put_value(harness.store.conn, harness.store, "hplus_receipt", fact)
    job["state"] = "completed"
    result["induction_receipt"] = {**fact, "receipt_ref": job["receipt_ref"]}
    job["result_ref"] = put_value(harness.store.conn, harness.store, "hplus_result", result)
    _save_job(harness, job)
    from or_harness.strategy.evidence_access import release_hold
    release_hold(harness,job['induction_id'])
    return fact


def receipt_observation(harness, job, *, as_of=None):
    from or_harness.world_model.hplus_support import observation
    return observation(harness,job,time.time() if as_of is None else as_of)


def frozen_rules(harness, job):
    """Control reviewers see the library at prediction, not later additions."""
    from or_harness.core.schema import StrategicEntry
    snapshot=get_value(harness.store,job['signal']['source_snapshot_ref'])
    layers=(snapshot.get('coverage') or {}).get('knowledge_layers') or {}
    out=[]
    for layer in ('verified','legacy_unknown','unverified'):
        for frozen in layers.get(layer) or []:
            data=copy.deepcopy(frozen)
            data['pattern']={'predicates':data.get('predicates') or {}}
            data['verification']={'state':data.get('verification_state') or 'unverified'}
            rule=data.get('rule') or {}
            if rule.get('body_ref'):
                rule['body']=get_value(harness.store,rule['body_ref'])
            out.append(StrategicEntry.from_dict(data))
    return out


def feedback(harness, *, task=None, budget_chars=3500, as_of=None):
    """Select relevant settled facts before applying the existing input budget."""
    from or_harness.strategy.evidence_views import chars, atomic
    cutoff = time.time() if as_of is None else as_of
    task = task or {}
    family = task.get("family")
    def words(text):
        english=set(re.findall(r'[a-z0-9_]+',str(text).lower()))
        chinese=re.findall(r'[\u4e00-\u9fff]+',str(text))
        return english|{s[i:i+2] for s in chinese for i in range(len(s)-1)}
    text=words(str(task.get("text") or task.get("description") or ""))
    from or_harness.profiling.profiler import profile_task
    current=profile_task(task).to_dict() if task.get('task_id') and task.get('family') else {}
    candidates = []
    for job in jobs(harness):
        if job["created_at"] >= cutoff or job["state"] != "completed":
            continue
        observation = receipt_observation(harness, job, as_of=cutoff)
        receipt = observation.get("receipt")
        if not receipt:
            continue
        signal = job["signal"]
        if family and signal.get("family") != family:
            continue
        overlap=len(text & words(signal.get('claim','')))
        structure=signal.get('structure') or {}
        class_match=bool(current.get('problem_class') not in (None,'unknown') and
                         current.get('problem_class')==structure.get('problem_class'))
        candidates.append((overlap+3*class_match, receipt["created_at"], job, observation))
    candidates.sort(key=lambda x: (x[0]>0,x[3]['label'] is not None,x[0],x[1]),reverse=True)
    out = {"feedback_version": HPLUS_VERSION, "as_of": cutoff, "n_applicable": len(candidates),
           "settled_examples": [], "support": "agent learning conclusions supported by valid cited facts; publication is a separate decision"}
    for _, _, job, observation in candidates:
        receipt = observation["receipt"]
        row = {"prediction_id": job["prediction_id"], "decision": job["signal"]["decision"],
               "decision_status": job["signal"]["decision_status"],
               "hypothesis": atomic(job["signal"]["claim"], job["signal"].get("input_ref"), 320),
               "facts": {"execution_ids": receipt["source_execution_ids"][:4], "source_fact_ref": receipt["source_fact_ref"],
                         "verification": atomic(observation.get("verification",receipt["verification"])[:1],job['receipt_ref'],400)},
               "conclusion": atomic(receipt["conclusion"], job["receipt_ref"], 320),
               "label": observation["label"], "verification_state": observation["verification_state"],
               "hypothesis_outcome": observation["hypothesis_outcome"],
               "observation_basis":observation.get("observation_basis"),
               "deviation_reason": atomic(receipt.get("deviation_reason"), job["receipt_ref"], 160),
               "rule_versions": [{k:u.get(k) for k in ("entry_id","version","operation","substantive")} for u in receipt["rule_updates"][:2]], "receipt_ref": job["receipt_ref"]}
        out["settled_examples"].append(row)
        if chars(out) + 32 > budget_chars:
            out["settled_examples"].pop()
            continue
        if len(out["settled_examples"])>=4:break
    out["n_included"] = len(out["settled_examples"])
    return out
