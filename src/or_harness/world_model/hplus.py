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


def _save_job(harness, job):
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
    predictions = harness.strategy_predictions.query(task_id=task_id, episode_id=episode_id)
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
        identity = "ij_" + digest([str(task_id), episode_id, prediction.prediction_id])[:24]
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
                "task_id": str(task_id), "episode_id": episode_id,
                "prediction_id": prediction.prediction_id, "signal": copy.deepcopy(signal),
                "solve_action_id": action.action_id, "execution_ids": [execution_id],
                "source_fact_ref": _facts_ref(harness, conn, record),
                "verification": _verification(record, signal), "mode": branch,
                "selected": selected, "state": "pending" if selected else "skipped",
                "created_at": time.time(), "action_id": None, "receipt_ref": None,
                "control_snapshot_ref": signal.get("input_ref") if branch == "nohplus" and signal["decision"] == 0 else None}
            _save_job(harness, job)
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


def material_for_job(harness, job, builder, **kwargs):
    if job.get("material_ref"):
        return copy.deepcopy(get_value(harness.store, job["material_ref"]))
    # This metadata is included in the builder's total budget from the start.
    from or_harness.strategy.evidence_views import atomic
    gate = {"state": "allow", "source": "frozen_hplus", "induction_id": job["induction_id"],
            "prediction_id": job["prediction_id"], "decision": job["signal"]["decision"],
            "reason": "positive executed job requires one review; missing verification does not cancel it",
            'hypothesis':atomic(job['signal'].get('claim'),job['signal'].get('input_ref'),450),
            'verification':atomic(job['signal'].get('verification_conditions'),job['signal'].get('input_ref'),450),
            'source_fact_ref':job['source_fact_ref'],
            'baseline_snapshot_ref':job['signal'].get('source_snapshot_ref')}
    if job['control_snapshot_ref']:
        kwargs['frozen_knowledge'] = frozen_rules(harness,job)
        gate['knowledge_basis'] = 'prediction-time snapshot, including zero-signal missed-opportunity review'
    result = builder(harness, gate=gate, task_id=job["task_id"], **kwargs)
    material_id = result.get("material_id")
    with harness.store.transaction() as conn:
        if material_id:
            session = get_value(harness.store, material_id)
            session["induction_id"] = job["induction_id"]
            # A new immutable session binds exactly what this job displayed.
            result["material_id"] = put_value(conn, harness.store, "induction_material", session)
            meta_write(harness.store, 'induction_material:last:' + job['task_id'], result['material_id'])
        job["material_ref"] = put_value(conn, harness.store, "hplus_material", result)
        _save_job(harness, job)
    return result


def reserve_action(harness, job):
    if job.get("action_id"):
        return
    with harness.store.transaction():
        current=get_job(harness,job['induction_id'])
        if current.get('action_id'):
            job.update(current)
            return
        maintenance = harness._begin_induce_action(None, False)
        job["action_id"] = maintenance["action_id"]
        job["knowledge_before"] = maintenance["knowledge_before"]
        job["maintenance_episode_id"] = maintenance["episode_id"]
        harness.actions.amend_action_params(job["action_id"], induction_id=job["induction_id"],
            capability_prediction_id=job["prediction_id"], source_execution_ids=job["execution_ids"])
        _save_job(harness, job)


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
    source = set(job["execution_ids"]) | set((action.outcome or {}).get("execution_ids") or [])
    material = get_value(harness.store, job["material_ref"]) if job.get("material_ref") else {}
    source.update(row["execution_id"] for row in material.get("material", []) +
                  (material.get("related_history") or {}).get("executions", []))
    before=get_value(harness.store,job['signal']['source_snapshot_ref'])
    baseline_layers=(before.get('coverage') or {}).get('knowledge_layers') or {}
    known_bodies={(e.get('rule') or {}).get('body_ref') for layer in baseline_layers.values() for e in layer}
    updates = [{"entry_id": r["saved"], "version": r.get("version"), "body_ref": r.get("body_ref"),
                "operation": r.get("operation"), "substantive": bool(r.get("new_version") and r.get('body_ref') not in known_bodies)}
               for r in result.get("relations") or [] if r.get("saved") and r.get("changed", True)]
    trigger = harness.bank.get(job['execution_ids'][0]) or harness.bank.get_pending(job['execution_ids'][0])
    if trigger:
        job['verification'] = _verification(trigger,job['signal'])
    conclusion = review.get("conclusion") or ("short rules submitted for existing validation" if updates else "review completed without a new rule")
    fact = {"induction_id": job["induction_id"], "prediction_id": job["prediction_id"],
        "solve_action_id": job["solve_action_id"], "induction_action_id": job["action_id"],
        "mode": job["mode"], "decision": job["signal"]["decision"],
        "input_ref": job["signal"].get("input_ref"), "material_id": material.get("material_id"),
        'baseline_snapshot_ref':job['signal'].get('source_snapshot_ref'),
        "source_execution_ids": sorted(source), "source_fact_ref": job["source_fact_ref"],
        "verification": job["verification"], "rule_updates": updates,
        "outcome": review.get("outcome") or ("knowledge_submitted" if updates else "insufficient_material"),
        "conclusion": str(conclusion), "hypothesis_outcome": review.get("hypothesis_outcome", "unknown"),
        "comparison": review.get("comparison"), "deviation_reason": review.get("deviation_reason"),
        "created_at": time.time()}
    if fact["outcome"] not in ("knowledge_submitted", "new_rule", "revised", "evidence_added", "already_covered",
                               "no_increment", "hypothesis_refuted", "insufficient_material"):
        raise ValueError("unknown induction review outcome")
    job["receipt_ref"] = put_value(harness.store.conn, harness.store, "hplus_receipt", fact)
    job["state"] = "completed"
    result["induction_receipt"] = {**fact, "receipt_ref": job["receipt_ref"]}
    job["result_ref"] = put_value(harness.store.conn, harness.store, "hplus_result", result)
    _save_job(harness, job)
    return fact


def receipt_observation(harness, job, *, as_of=None):
    cutoff = time.time() if as_of is None else as_of
    if job.get("state") != "completed":
        return {"label": None, "verification_state": "unobserved" if job["state"] == "skipped" else "pending"}
    receipt = get_value(harness.store, job["receipt_ref"])
    if receipt["created_at"] > cutoff:
        return {"label": None, "verification_state": "not_yet_recorded"}
    from or_harness.strategy.rule_versions import event_records
    validated = []
    for update in receipt["rule_updates"]:
        if not update["substantive"]:
            continue
        events = [e for e in event_records(harness.store, update["entry_id"], update["version"])
                  if e["kind"] == "publication" and e["at"] <= cutoff]
        if events and events[-1].get("to") == "active":
            validated.append(update)
    label, state = (1, "independently_validated_increment") if validated else (None, "unvalidated")
    comparison = receipt.get("comparison") or {}
    if not validated and not receipt["rule_updates"] and receipt["outcome"] in ("already_covered", "no_increment"):
        references = set(comparison.get('evidence_refs') or [])
        frozen_sources = {receipt['source_fact_ref'],receipt.get('baseline_snapshot_ref')}
        fact=get_value(harness.store,receipt['source_fact_ref'])
        from or_harness.world_model.g_signal import _acceptance_support
        checked=bool((fact.get('method') or {}).get('actual') and _acceptance_support(fact.get('checker') or {}))
        layers=(get_value(harness.store,receipt['baseline_snapshot_ref']).get('coverage') or {}).get('knowledge_layers') or {}
        existing=[e for layer in layers.values() for e in layer if isinstance(e,dict)]
        covered=receipt['outcome']!='already_covered' or bool(existing)
        if (comparison.get("sufficient") is True and comparison.get("source") and comparison.get("assessment")
                and checked and covered and None not in frozen_sources and frozen_sources <= references):
            label, state = 0, "sourced_comparison_no_increment"
    return {"label": label, "verification_state": state, "validated_rules": validated,
            "hypothesis_outcome": receipt["hypothesis_outcome"], "receipt": receipt}


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
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    out = {"feedback_version": HPLUS_VERSION, "as_of": cutoff, "n_applicable": len(candidates),
           "settled_examples": [], "support": "recorded training receipts; labels require independent validation or sourced sufficient comparison"}
    for _, _, job, observation in candidates[:4]:
        receipt = observation["receipt"]
        row = {"prediction_id": job["prediction_id"], "decision": job["signal"]["decision"],
               "decision_status": job["signal"]["decision_status"],
               "hypothesis": atomic(job["signal"]["claim"], job["signal"].get("input_ref"), 320),
               "facts": {"execution_ids": receipt["source_execution_ids"][:4], "source_fact_ref": receipt["source_fact_ref"],
                         "verification": atomic(receipt["verification"][:1],job['receipt_ref'],400)},
               "conclusion": atomic(receipt["conclusion"], job["receipt_ref"], 320),
               "label": observation["label"], "verification_state": observation["verification_state"],
               "hypothesis_outcome": observation["hypothesis_outcome"],
               "deviation_reason": atomic(receipt.get("deviation_reason"), job["receipt_ref"], 160),
               "rule_versions": receipt["rule_updates"][:2], "receipt_ref": job["receipt_ref"]}
        out["settled_examples"].append(row)
        if chars(out) + 32 > budget_chars:
            out["settled_examples"].pop()
            break
    out["n_included"] = len(out["settled_examples"])
    return out
