"""Versioned short rules and attributable transfer/boundary evidence.

Bodies are immutable condition values. Checks and uses append independently;
the strategic row holds only the currently offered and pending pointers.
Natural-language premises/effects require a sourced scope assessment. Neither
string equality, a solver status nor a checker pass supplies that assessment.
"""
from __future__ import annotations

import copy
import json
import time
import math

from or_harness.core.schema import StrategicEntry, CostVector, normalize_claim, task_check_state
from or_harness.strategy.evidence_views import compact_execution, task_identity,instance_relation
from or_harness.world_model.conditions import digest, get_value, put_value
from or_harness.world_model.induction_gate import meta_read, meta_write, start_decision

RULE_VERSION = "or-rule/2"


def sync_rule_entries(harness, changed_ids=(), affected_ids=()):
    pending=set(meta_read(harness.store,'rule_index_pending') or [])
    wanted=sorted(set(changed_ids) | (pending & set(affected_ids)))
    result=harness.index_sync.sync_entries(wanted)
    if result.get('state') in ('deferred','skipped'):
        pending.update(wanted)
    elif result.get('state')=='synced':
        pending.difference_update(wanted)
    with harness.store.transaction():
        meta_write(harness.store,'rule_index_pending',sorted(pending))
    return result


def _verification_ref(store,conn,report,body_ref):
    raw=copy.deepcopy(report)
    claim=raw.pop('claim',None)
    if claim is not None:
        body=get_value(store,body_ref)
        if claim==body.get('expect'):
            raw['claim_ref']={'value_ref':body_ref,'field':'expect'}
        else:
            raw['claim_ref']={'value_ref':put_value(conn,store,'rule_statement',{'value':claim}),'field':'value'}
    return put_value(conn,store,'rule_verification',raw)


def _verification_value(store,reference):
    raw=copy.deepcopy(get_value(store,reference))
    claim_ref=raw.get('claim_ref')
    if 'claim' not in raw and claim_ref:
        raw['claim']=get_value(store,claim_ref['value_ref'])[claim_ref['field']]
    return raw


def declared_task_identity(task):
    from or_harness.strategy.evidence_views import declared_task_identity as identity_of
    return identity_of(task)


def frozen_entry(entry):
    value = entry.to_dict()
    if entry.rule:
        value.pop("claim", None)
        value.pop("last_consulted_at", None)
        value["rule"] = {k: copy.deepcopy(entry.rule.get(k)) for k in
                         ("schema_version", "view_version", "body_ref", "active_version", "state", "version_state", "legacy_version")}
        value["verification"] = {"state": entry.verification_state, "source": "rule_version"}
        value["expected"].pop("effect_attribution", None)
    return value


def maintenance_references(harness):
    """Small pre/post manifests; unchanged prose is interned once."""
    from or_harness.world_model.state import KnowledgeRef
    refs=[]
    with harness.store.transaction() as conn:
        for entry in harness.sbank.list():
            ref=KnowledgeRef.from_entry(entry).to_dict()
            ref.pop('claim',None)
            ref['value_ref']=put_value(conn,harness.store,'knowledge',frozen_entry(entry))
            ref['claim_ref']=(entry.rule or {}).get('body_ref') if entry.rule else (
                put_value(conn,harness.store,'legacy_claim',entry.claim) if entry.claim else None)
            if entry.rule and entry.rule.get('candidate_version'):
                candidate=version_row(harness.store,entry.entry_id,entry.rule['candidate_version'])
                ref['rule']['candidate_body_ref']=candidate['body_ref'] if candidate else None
            refs.append(ref)
    return refs


def version_row(store, entry_id, version):
    return store.conn.execute("SELECT * FROM rule_versions WHERE entry_id=? AND version=?",
                              (str(entry_id), int(version))).fetchone()


def hydrate_entry(store, entry, version=None):
    if not entry.rule:
        return entry
    rule = copy.deepcopy(entry.rule)
    chosen = version if version is not None else rule.get("active_version") or rule.get("candidate_version")
    if chosen == "candidate":
        chosen = rule.get("candidate_version")
    row = version_row(store, entry.entry_id, chosen) if chosen is not None else None
    if row is None:
        raise ValueError(f"unknown rule version {entry.entry_id}@{chosen}")
    body = get_value(store, row["body_ref"])
    metadata = get_value(store, row["metadata_ref"])
    rule.update(body=copy.deepcopy({k: body[k] for k in ("when", "do", "expect")}),
                view_version=row["version"], body_ref=row["body_ref"], version_state=row["state"],
                legacy_version=bool(metadata.get("legacy_version")))
    entry.rule = rule
    entry.pattern = {"predicates": copy.deepcopy(body.get("predicates") or {})}
    legacy = (get_value(store,metadata['legacy_input_ref']) if metadata.get('legacy_input_ref') else
              copy.deepcopy(body.get('legacy_claim')) or copy.deepcopy((metadata.get('legacy_snapshot') or {}).get('claim')))
    entry.claim = normalize_claim(legacy or {
        "text": "When: " + body["when"] + "\nDo: " + body["do"] + "\nExpect: " + body["expect"],
        "kind": "transfer_rule", "subject": entry.strategy_id,
        "conditions": {"predicates": entry.predicates, "note": body["when"]},
        "method": {"name": entry.strategy_id, "steps": [body["do"]]},
        "evidence": metadata.get("seed_evidence") or [],
    })
    if metadata.get('legacy_version') and (metadata.get('legacy_snapshot') or {}).get('claim') is None:
        entry.claim=None
    if legacy and entry.claim is not None:
        entry.claim["evidence"] = copy.deepcopy(metadata.get("seed_evidence") or [])
    if entry.claim is not None:
        entry.claim["tasks"] = sorted({str(x["task_id"]) for x in metadata.get("seed_evidence") or [] if x.get("task_id")})
        entry.claim["strategy_ids"] = sorted({str(x["strategy_id"]) for x in metadata.get("seed_evidence") or [] if x.get("strategy_id")})
        if len(metadata.get('seed_families') or []) == 1:
            entry.claim['family'] = metadata['seed_families'][0]
    checks = check_records(store, entry.entry_id, row["version"])
    seed_checks=[{'execution_id':eid,'state':fact.get('checker',{}).get('state'),
                  'validity':meta_read(store,'rule_evidence_validity:'+eid) or {'valid':True,'revision':0},
                  'scope':fact.get('checker',{}).get('scope'),
                  'acceptance_rule':fact.get('checker',{}).get('acceptance_rule')}
                 for eid,ref in (metadata.get('seed_fact_refs') or {}).items()
                 for fact in [get_value(store,ref)]]
    choices=[e for e in event_records(store,entry.entry_id,row['version']) if e['kind']=='publication' and e.get('decision')]
    entry.verification = copy.deepcopy(metadata.get("declared_verification") or {
        "state": "fact_checked" if any(c['state'] in ('passed','failed') for c in seed_checks) or checks else "unverified",
        "scope": {"rule_version": row["version"], "n_checks": len(checks), "detail_ref": f"rule:{entry.entry_id}@{row['version']}"},
        "not_covered": ["Only recorded, sourced premises and checked effects are supported."],
    })
    if not metadata.get('declared_verification'):
        entry.verification['scope'].update(seed_checks=seed_checks,authority='agent',
            publication=choices[-1]['decision'] if choices else None)
    derivation = [c for c in checks if c['kind']=='derivation' and c.get('report_ref')]
    explicit = [c for c in derivation if c.get('assertions_ref') and
                get_value(store,c['assertions_ref']).get('assertions')]
    declared = (explicit or derivation)[-1] if derivation else None
    if declared:
        entry.verification = _verification_value(store, declared["report_ref"])
    if not metadata.get('legacy_version'):
        entry.verification.setdefault('scope',{})['evidence_validity']={c['execution_id']:c['validity'] for c in seed_checks}
    return entry


def storage_payload(entry):
    raw = entry.to_dict()
    if entry.rule:
        raw.pop("claim", None)
        raw.pop("verification", None)
        raw["rule"] = {k: copy.deepcopy(v) for k, v in entry.rule.items()
                       if k not in ("body", "body_ref", "view_version", "version_state", "legacy_version")}
        raw["expected"].pop("effect_attribution", None)
    return raw


def check_records(store, entry_id, version=None):
    sql, args = "SELECT payload FROM rule_checks WHERE entry_id=?", [str(entry_id)]
    if version is not None:
        sql += " AND version=?";args.append(int(version))
    return [store.loads(r["payload"]) for r in store.conn.execute(sql + " ORDER BY created_at,check_id", args)]


def event_records(store, entry_id, version=None):
    sql, args = "SELECT payload FROM rule_events WHERE entry_id=?", [str(entry_id)]
    if version is not None:
        sql += " AND version=?";args.append(int(version))
    return [store.loads(r["payload"]) for r in store.conn.execute(sql + " ORDER BY created_at,event_id", args)]


def append_event(store, entry_id, version, kind, *, execution_id=None, data=None):
    identity = {"entry_id": str(entry_id), "version": int(version), "kind": kind,
                "execution_id": execution_id, **copy.deepcopy(data or {})}
    key = "re:" + digest({k: v for k, v in identity.items() if k != "fact_ref" or kind != "evidence"})
    row = store.conn.execute("SELECT payload FROM rule_events WHERE event_id=?", (key,)).fetchone()
    if row:
        return store.loads(row["payload"]), False
    value = {**identity, "event_id": key, "at": time.time()}
    store.conn.execute("INSERT INTO rule_events VALUES (?,?,?,?,?,?,?)",
        (key, str(entry_id), int(version), execution_id, kind, store.dumps(value), value["at"]))
    return value, True


def append_check(store, entry_id, version, kind, state, data):
    identity = {"entry_id": str(entry_id), "version": int(version), "kind": kind,
                "state": state, **copy.deepcopy(data)}
    if kind!='derivation' and not data.get('non_observation'):
        previous=[]
        for row in store.conn.execute('SELECT payload FROM rule_checks WHERE entry_id=? AND version=? ORDER BY created_at,check_id',
                                      (str(entry_id),int(version))):
            value=store.loads(row['payload'])
            if value.get('execution_id')==data.get('execution_id') and not value.get('non_observation'):previous.append(value)
        signature=digest({k:v for k,v in identity.items() if k not in ('fact_ref','observed_at','evidence_semantics')})
        if previous and previous[-1].get('assertion_signature')==signature:return previous[-1],False
        identity['assertion_signature']=signature
        identity['previous_check_id']=previous[-1]['check_id'] if previous else None
    key = "rc:" + digest({k: v for k, v in identity.items() if k != "fact_ref" or not data.get("evidence_semantics")})
    row = store.conn.execute("SELECT payload FROM rule_checks WHERE check_id=?", (key,)).fetchone()
    if row:
        return store.loads(row["payload"]), False
    value = {**identity, "check_id": key, "at": time.time()}
    store.conn.execute("INSERT INTO rule_checks VALUES (?,?,?,?,?,?,?)",
        (key, str(entry_id), int(version), kind, state, store.dumps(value), value["at"]))
    return value, True


def _body(raw, prior=None):
    fields = ("when", "do", "expect")
    if not any(k in raw for k in fields) and prior is not None and not any(k in raw for k in ('claim','text','method','conditions')):
        return copy.deepcopy(prior)
    if any(k in raw for k in fields):
        body = {k: raw.get(k) for k in fields}
        if not all(isinstance(body[k], str) and body[k].strip() for k in fields):
            raise ValueError("a short rule needs nonempty when, do and expect paragraphs")
        body = {k: body[k].strip() for k in fields}
        if sum(len(v) for v in body.values())>6000:
            raise ValueError('short rule paragraphs exceed 6000 characters; keep complete supporting detail by reference')
        if any(x in body["do"].lower() for x in ("no new knowledge", "nothing to add")):
            raise ValueError("do must describe executable actions, not an empty review")
        body["predicates"] = copy.deepcopy(raw.get("predicates") or (raw.get("conditions") or {}).get("predicates") or {})
        body["effect_scope"] = raw.get("effect_scope", "task_check")
        return body
    # The old --relation grammar is a compatibility input, not a grant of
    # v2 publication. Retain its exact text/conditions and declared checks.
    claim = normalize_claim({"text": raw.get("claim") or raw.get("text"), "kind": raw.get("kind"),
        "subject": raw.get("subject"), "method": raw.get("method"), "conditions": raw.get("conditions"),
        "evidence": []})
    if not claim.get("text"):
        raise ValueError("rule requires when/do/expect or a legacy claim")
    method = claim.get("method") or {}
    action = "\n".join(method.get("steps") or [])
    if "method" in raw and not action:
        raise ValueError("method.steps must contain executable actions")
    if any(x in claim["text"].lower() for x in ("no new knowledge", "no new entry", "nothing to add")):
        raise ValueError("placeholder claim; submit an empty review instead")
    return {"when": (claim.get("conditions") or {}).get("note") or "Review the stated premises before use.",
            "do": action, "expect": claim["text"], "_legacy_input": claim,
            "predicates": copy.deepcopy((claim.get("conditions") or {}).get("predicates") or {}),
            "effect_scope": raw.get("effect_scope", "task_check")}


def _resolve(engine, evidence):
    if not isinstance(evidence, (list, tuple, type(None))):
        raise ValueError("evidence must contain execution IDs or {execution_id, role} records")
    if any(not isinstance(x, str) and (not isinstance(x, dict) or not x.get("execution_id")) for x in evidence or []):
        raise ValueError("evidence requires execution_id; bundle citations are not supported")
    items = [{"execution_id": str(x), "role": "derivation"} if isinstance(x, str) else
             {**x, "role": x.get("role", "derivation")} for x in (evidence or [])]
    if not items:
        return {"records": [], "evidence": [], "problem": None}
    resolved = engine._resolve_claim_evidence(items)
    if resolved.get("problem"):
        raise ValueError(resolved["problem"])
    return resolved


def _facts_ref(harness, conn, record):
    # Preserve decisive facts after the evidence window evicts the execution.
    fact = compact_execution(harness, record, field_limit=1000000)
    fact["checker"] = copy.deepcopy((record.execution_features or {}).get("task_check") or {})
    fact["artifacts"] = copy.deepcopy((record.execution_features or {}).get("artifacts") or {})
    action=harness.actions.by_execution(record.execution_id)
    if action and action.pre_snapshot_id:
        fact['task_snapshot_id']=action.pre_snapshot_id
        frozen=conn.execute('SELECT payload FROM frozen_conditions WHERE snapshot_id=?',(action.pre_snapshot_id,)).fetchone()
        if frozen:fact['task_input_ref']=harness.store.loads(frozen['payload']).get('problem_ref')
        else:
            snapshot=harness.get_snapshot(action.pre_snapshot_id)
            if snapshot:fact['task_input_ref']=put_value(conn,harness.store,'problem',snapshot.problem_state)
    elif record.execution_features.get('retained_task_input_ref'):
        fact['task_input_ref']=record.execution_features['retained_task_input_ref']
    fact.update(source=record.source,solver=copy.deepcopy(record.solver),
        execution_config=copy.deepcopy(record.execution_features.get('execution_config') or {}),
        used_entries=copy.deepcopy(record.execution_features.get('used_entries') or {}),
        cost_provenance=copy.deepcopy(record.execution_features.get('cost_provenance') or {}))
    ref=put_value(conn,harness.store,'rule_fact',fact)
    meta_write(harness.store,'rule_fact_latest:'+record.execution_id,ref)
    return ref


def _session(harness, raw, records):
    job=getattr(harness,'_active_induction_job',None)
    if job:
        from or_harness.world_model.hplus import aggregate_material
        return aggregate_material(harness,job)
    reference = raw.get("material_id")
    if reference is None:reference=meta_read(harness.store,'induction_material:last_read')
    if reference is None and records:
        reference = meta_read(harness.store, "induction_material:last:" + str(records[0].task_id))
    if reference:
        value = copy.deepcopy(get_value(harness.store, reference))
        if not str(reference).startswith("induction_material:"):
            raise ValueError("material_id must reference an induction material read")
        pending = [value.get("prior_material_ref")]
        if records and not value.get('gate'):
            pending.append(meta_read(harness.store,"induction_material:last:"+str(records[0].task_id)))
        visited = {reference}
        while pending:
            previous = pending.pop()
            if not previous or previous in visited:
                continue
            visited.add(previous)
            older = get_value(harness.store,previous)
            value["shown_execution_ids"] = list(dict.fromkeys(value.get("shown_execution_ids",[])+older.get("shown_execution_ids",[])))
            value.setdefault("identities",{}).update(older.get("identities") or {})
            pending.append(older.get("prior_material_ref"))
        value["read_refs"] = sorted(visited)
        return reference, value
    return None, {"shown_execution_ids": [], "identities": {}}


def _insert_version(harness, entry, body, metadata, version, state="candidate"):
    with harness.store.transaction() as conn:
        body_ref = put_value(conn, harness.store, "rule", body)
        meta_ref = put_value(conn, harness.store, "rule_metadata", metadata)
        conn.execute("INSERT INTO rule_versions VALUES (?,?,?,?,?,?)",
                     (entry.entry_id, version, body_ref, meta_ref, time.time(), state))
    return body_ref


def _identical_entry(store,body_ref,body=None):
    found=store.conn.execute("SELECT rv.entry_id,rv.version FROM rule_versions rv JOIN strategic_entries se ON se.entry_id=rv.entry_id "
        "WHERE rv.body_ref=? AND json_extract(se.payload,'$.rule.schema_version')=? "
        "AND COALESCE(json_extract(se.payload,'$.rule.state'),'')!='merged' ORDER BY rv.created_at LIMIT 1",(body_ref,RULE_VERSION)).fetchone()
    if found or body is None:return found
    from or_harness.strategy.rule_support import body_signature
    signature=body_signature(body)
    for row in store.conn.execute("SELECT rv.* FROM rule_versions rv JOIN strategic_entries se ON se.entry_id=rv.entry_id WHERE json_extract(se.payload,'$.rule.schema_version')=? AND COALESCE(json_extract(se.payload,'$.rule.state'),'')!='merged' ORDER BY rv.created_at",(RULE_VERSION,)):
        if body_signature(get_value(store,row['body_ref']))==signature:return row
    return None


def _merge_ancestors(harness,entry_id):
    seen=set()
    while entry_id and entry_id not in seen:
        seen.add(entry_id)
        entry=harness.sbank.get(entry_id)
        entry_id=(entry.rule.get('merged_into') or entry.rule.get('merge_pending_into')) if entry and entry.rule else None
    return seen


def _convert_legacy(harness, entry):
    if entry.rule:
        return
    original = entry.to_dict()
    claim = entry.claim or {}
    body = {
        "when": (claim.get('conditions') or {}).get('note') or str(entry.predicates),
        "do": "\n".join((claim.get('method') or {}).get('steps') or entry.actions),
        "expect": claim.get("text") or "Historical statistical commitment",
        "predicates": entry.predicates, "effect_scope": "legacy"}
    with harness.store.transaction() as conn:
        facts = {item['execution_id']:_facts_ref(harness,conn,record)
                 for item in claim.get('evidence') or [] if item.get('execution_id')
                 and (record := harness.bank.get(item['execution_id'])) is not None}
    _insert_version(harness, entry, body, {"legacy_version": True, "legacy_snapshot": original,
        "seed_fact_refs":facts,
        "declared_verification": copy.deepcopy(entry.verification), "seed_evidence": claim.get("evidence") or [],
        "derivation_execution_ids": [], "derivation_base_ids": [], "required_regressions": []}, 1, "legacy")
    entry.rule = {"schema_version": RULE_VERSION, "active_version": 1,
                  "candidate_version": None, "state": "active"}
    harness.sbank.update(entry)


def submit_rule(engine, raw, *, dry_run=False, verify=None, notes=None, force=False):
    harness = engine.harness
    resolved = _resolve(engine, raw.get("evidence"))
    records = resolved["records"]
    from or_harness.world_model.hplus import formal, require_job
    binary_build = formal(harness)
    if binary_build and not getattr(harness,'_existing_entry_maintenance',False):
        if not dry_run and not getattr(harness,'_active_induction_job',None):
            raise ValueError('H+ writes must use induce so the authorized job and receipt commit together')
        require_job(harness, raw.get('induction_id'), material_id=raw.get('material_id'))
    if not binary_build and records and not getattr(harness,'_existing_entry_maintenance',False):
        _,session=_session(harness,raw,records)
        gate=session.get('gate') or start_decision(harness,records[0].task_id)
        if gate['state'] in ('skip','pending'):
            return {'saved':None,'skipped':gate['reason'],'induction_gate':gate}
    target = raw.get("entry_id") or raw.get("target_entry") or raw.get("target_entry_id")
    entry = harness.sbank.get(str(target)) if target is not None else None
    if target is not None and entry is None:
        raise ValueError(f"unknown target entry {target!r}")
    prior = None
    if entry and entry.rule:
        v = raw.get("version") or entry.rule.get("candidate_version") or entry.rule.get("active_version")
        row = version_row(harness.store, entry.entry_id, v)
        prior = get_value(harness.store, row["body_ref"])
    elif entry is not None:
        claim=entry.claim or {}
        prior={'when':(claim.get('conditions') or {}).get('note') or str(entry.predicates),
               'do':'\n'.join((claim.get('method') or {}).get('steps') or entry.actions),
               'expect':claim.get('text') or 'Historical statistical commitment',
               'predicates':entry.predicates,'effect_scope':'legacy'}
    body = _body(raw, prior)
    legacy_input=body.pop('_legacy_input',None)
    from or_harness.strategy.rule_support import body_signature
    if prior is not None and body_signature(body)==body_signature(prior):body=copy.deepcopy(prior)
    operation = raw.get("operation", "create" if target is None else "add_evidence")
    if operation not in ("create", "add_evidence", "revise", "merge"):
        raise ValueError("operation must be create, add_evidence, revise or merge")
    if operation in ('revise','merge') and target is None:
        raise ValueError('revisions and merges require an explicit target entry_id')
    if entry is not None and entry.rule and entry.rule.get('merged_into'):
        raise ValueError('entry was merged into '+str(entry.rule['merged_into'])+'; choose that target explicitly')
    body_ref = "rule:" + digest(body)
    if target is None:
        identical = _identical_entry(harness.store,body_ref,body)
        if identical:
            entry = harness.sbank.get(identical["entry_id"])
            body=get_value(harness.store,version_row(harness.store,entry.entry_id,identical['version'])['body_ref'])
            body_ref='rule:'+digest(body)
    if not records and entry is None:
        raise ValueError("a new rule needs real evidence IDs")
    if dry_run:
        return {"saved": None, "would_create": entry is None, "would_update": entry.entry_id if entry else None,
                "body": body, "publication": {"published": False, "authority":"agent", "decision":raw.get("publication")}}
    merged = []
    for entry_id in raw.get("merge_entry_ids") or []:
        other = harness.sbank.get(str(entry_id))
        if other is None:
            raise ValueError(f"unknown merge source {entry_id!r}")
        if entry is not None and other.entry_id == entry.entry_id:
            continue
        if entry is not None and other.entry_id in _merge_ancestors(harness,entry.entry_id):
            raise ValueError('merge would create a cycle in rule identities')
        merged.append(other)
    created, changed, new_version = entry is None, False, False
    session_ref, session = _session(harness, raw, records)
    with harness.store.transaction() as conn:
        if entry is None and target is None:
            same = _identical_entry(harness.store,body_ref)
            entry = harness.sbank.get(same["entry_id"]) if same else None
        elif entry is not None:
            entry = harness.sbank.get(entry.entry_id)
        created = entry is None
        before_body_ref = (entry.rule or {}).get('body_ref') if entry else None
        before_published=entry.is_published if entry else False
        previous_check_ids = {c['check_id'] for c in check_records(harness.store,entry.entry_id)} if entry else set()
        if entry is None:
            entry = StrategicEntry(entry_id="", strategy_id=str(raw.get("subject") or "rule:" + digest(body)[:12]),
                pattern={"predicates": body["predicates"]}, expected_quality_hat=0.0,
                expected_cost_hat=CostVector(measured=set()),
                provenance=[r.execution_id for r in records][:8],
                rule={"schema_version": RULE_VERSION, "active_version": None, "candidate_version": 1, "state": "candidate"})
            veto = engine._claim_veto(entry)
            if veto is not None:
                if not force:
                    return {"saved": None, "vetoed": veto, "skipped": "cold-archive veto; explicit force required"}
                harness.sbank.revive(veto["pattern_hash"], force=True)
            harness.sbank.add(entry)
            version = 1
        else:
            _convert_legacy(harness, entry)
            rows = list(conn.execute("SELECT * FROM rule_versions WHERE entry_id=? ORDER BY version", (entry.entry_id,)))
            selected_version = raw.get('version') or entry.rule.get('candidate_version') or entry.rule.get('active_version')
            selected_row = next((r for r in rows if r['version']==selected_version),None)
            equal = selected_row if selected_row and selected_row['body_ref']==body_ref else next(
                (r for r in reversed(rows) if r['body_ref']==body_ref),None)
            # An explicit revert changes the currently proposed body too.
            # Reuse its immutable value, but freeze a new chronological version.
            if operation in ('revise','merge') and selected_row and selected_row['body_ref']!=body_ref:
                equal = None
            version = equal["version"] if equal else max(r["version"] for r in rows) + 1
            if equal is None:
                if operation not in ("revise", "merge"):
                    raise ValueError("changed rule content requires operation=revise or merge")
                entry.rule["candidate_version"] = version
        existing = version_row(harness.store, entry.entry_id, version)
        if isinstance(raw.get("prediction"), dict):
            engine._apply_declared_prediction(entry, raw["prediction"])
        annotated = list(dict.fromkeys(entry.applicability + [str(n).strip() for n in notes or [] if str(n).strip()]))
        if annotated != entry.applicability:
            entry.applicability = annotated
            changed = True
        if existing is None:
            shown = list(dict.fromkeys(session.get("shown_execution_ids", []) + [r.execution_id for r in records]))
            identities = dict(session.get("identities") or {})
            identities.update({r.execution_id: task_identity(harness, r) for r in records})
            regressions=[];cases=[]  # historical cases are review material, never compulsory work
            metadata = {"seed_evidence": resolved["evidence"], "derivation_execution_ids": shown,
                "seed_families": sorted({r.profile_snapshot.family for r in records}),
                "derivation_base_ids": sorted({x["base_id"] for x in identities.values()}),
                "derivation_content_digests": sorted({x["content_digest"] for x in identities.values() if x.get("content_digest")}),
                "identities": identities, "material_ref": session_ref, "material_read_refs":session.get("read_refs") or [], "required_regressions": regressions,
                'regression_cases':cases,
                "seed_fact_refs": {r.execution_id: _facts_ref(harness, conn, r) for r in records}}
            if legacy_input is not None:
                metadata['legacy_input_ref']=put_value(conn,harness.store,'legacy_rule_input',legacy_input)
            _insert_version(harness, entry, body, metadata, version)
            new_version = changed = True
        else:
            metadata = get_value(harness.store, existing["metadata_ref"])
        for record in records:
            submitted_role=next((x['role'] for x in resolved['evidence'] if x['execution_id']==record.execution_id),'derivation')
            _event, added = append_event(harness.store, entry.entry_id, version, "evidence", execution_id=record.execution_id,
                data={"role":submitted_role if submitted_role!='derivation' else
                      'derivation' if record.execution_id in metadata.get('derivation_execution_ids',[]) else 'additional_evidence',
                      "fact_ref": _facts_ref(harness, conn, record)})
            changed |= added
        # Preserve every actually declared legacy assertion as an audit,
        # without relabelling derivation checks as transfer validation.
        from or_harness.strategy.induction import _resolve_relation_check, relation_material_gate
        from or_harness.strategy.verification import verify_relation
        declared, check_note = _resolve_relation_check(raw, verify)
        verification = None
        if legacy_input is not None:
            _,added=append_event(harness.store,entry.entry_id,version,'submission',data={
                'input_ref':put_value(conn,harness.store,'legacy_rule_input',legacy_input)})
            changed |= added
        if declared or legacy_input is not None:
            assertions = ((declared or {}).get("check") or {}).get("assertions")
            assert_ref = put_value(conn, harness.store, "rule_assertions", {"claim": (declared or {}).get("claim") or body["expect"], "assertions": assertions})
            evidence_fingerprint = digest({"facts": {r.execution_id:_facts_ref(harness,conn,r) for r in records}, "roles":resolved["evidence"]})
            cached = next((c for c in check_records(harness.store,entry.entry_id,version) if c.get("assertions_ref")==assert_ref
                           and c.get("evidence_fingerprint")==evidence_fingerprint),None)
            verification = _verification_value(harness.store,cached["report_ref"]) if cached else verify_relation(
                str((declared or {}).get("claim") or body["expect"]), evidence=(declared or {}).get("executions") or records,
                roles=resolved["evidence"], assertions=assertions)
            append_check(harness.store, entry.entry_id, version, "derivation", verification["state"],
                {"assertions_ref": assert_ref, "report_ref": _verification_ref(harness.store,conn,verification,body_ref),
                 "evidence": resolved["evidence"], "body_ref": body_ref, "evidence_fingerprint": evidence_fingerprint})
        for other in merged:
            _convert_legacy(harness, other)
            previous_pointer=copy.deepcopy(other.rule)
            source_version=other.rule.get('active_version') or other.rule.get('candidate_version')
            source_row=version_row(harness.store,other.entry_id,source_version)
            source_meta=get_value(harness.store,source_row['metadata_ref'])
            if other.rule.get('merged_into')!=entry.entry_id:
                other.rule.update(merge_pending_into=entry.entry_id,merge_pending_version=version)
                if other.rule.get('active_version') is None:
                    other.rule['state']='merge_pending'
            if other.rule!=previous_pointer:
                harness.sbank.update(other);changed=True
            _,added=append_event(harness.store, entry.entry_id, version, "merge", data={"source_entry": other.entry_id,
                "source_version": other.rule.get("active_version") or other.rule.get("candidate_version"),
                'source_body_ref':source_row['body_ref'],
                'required_regressions':[],
                "assessment": raw.get("merge_reason", "author selected the target after comparing mechanism/premises/effect")})
            changed |= added
        harness.sbank.update(entry)
        validations = []
        for check in raw.get("validation") or []:
            validations.append(validate_case(harness, entry.entry_id, version, check))
        publication = update_publication(harness,entry.entry_id,version,raw.get("publication"))
        if merged and harness.sbank.get(entry.entry_id).is_published:
            _apply_confirmed_merges(harness,entry.entry_id,version)
        entry = harness.sbank.get(entry.entry_id)
        changed |= bool({c['check_id'] for c in check_records(harness.store,entry.entry_id)}-previous_check_ids)
    return {"saved": entry.entry_id, **({"created_entry": entry.entry_id} if created else {"updated_entry": entry.entry_id} if changed else {}),
        "changed": changed, "idempotent": not changed, "version": version, "new_version": new_version,
        "operation":operation,
        "body_ref": body_ref, "entry": entry.to_dict(), "claim": entry.claim,
        "verification": verification or entry.verification, "check_note": check_note,
        "material": relation_material_gate(raw, records) if not body.get('do') else None,
        "publication": publication, "validation": validations,
        "execution_ids": sorted({r.execution_id for r in records} | {v["execution_id"] for v in validations if v.get("fact_ref")}),
        "index_entry_ids": ([entry.entry_id] if before_body_ref!=(entry.rule or {}).get('body_ref') or before_published!=entry.is_published else [])+
                           [x.entry_id for x in merged if harness.sbank.get(x.entry_id).rule.get('state')=='merged'],
        "affected_entry_ids": [entry.entry_id] + [x.entry_id for x in merged]}


def _assessment(record, entry_id, version, supplied=None):
    candidates = [supplied] if supplied else ((record.execution_features or {}).get("task_check") or {}).get("rule_assessments") or []
    return next((copy.deepcopy(x) for x in candidates if isinstance(x, dict) and
                 str(x.get("entry_id")) == str(entry_id) and isinstance(x.get('version'),int)
                 and not isinstance(x.get('version'),bool) and x.get("version") == version and not x.get('withdrawn')), {})


def _sourced(record, assessment):
    """Legacy later-effect evidence check; never a new publication prerequisite."""
    refs = assessment.get("evidence_refs") or []
    artifacts = (record.execution_features or {}).get("artifacts") or {}
    return (isinstance(assessment.get("source"), str) and bool(assessment["source"].strip())
            and isinstance(assessment.get("assessment"), str) and bool(assessment["assessment"].strip())
            and isinstance(refs, list) and "artifacts.submitted_script" in refs
            and all(isinstance(ref, str) and ref.startswith("artifacts.") and
                    isinstance(artifacts.get(ref.split(".", 1)[1]), dict) and
                    artifacts[ref.split(".", 1)[1]].get("sha256") for ref in refs))


def _cost_comparison(harness,record,check):
    from or_harness.strategy.rule_support import cost_comparison
    return cost_comparison(harness,record,check)


def adopted_version(harness,record,entry_id):
    reference=((record.execution_features.get('used_entries') or {}).get('versions') or {}).get(str(entry_id))
    return (get_value(harness.store,reference).get('rule') or {}).get('view_version') if reference else None


def validate_case(harness,entry_id,version,check):
    """Record facts and the agent's assessment; never decide publication."""
    from or_harness.strategy.evidence_access import resolve_execution,validity
    from or_harness.strategy.rule_support import fact_basis
    check=copy.deepcopy(check);kind=check.get('kind','use')
    if kind not in ('use','transfer','boundary','regression','counterexample'):
        raise ValueError('unknown rule observation purpose')
    row=version_row(harness.store,entry_id,version)
    if row is None:raise ValueError('unknown rule version')
    eid=str(check.get('execution_id') or '')
    record=resolve_execution(harness,eid)
    if record is None:
        return append_check(harness.store,entry_id,version,kind,'insufficient_evidence',{
            'execution_id':eid,'non_observation':True,'reason':'execution unavailable or explicitly excluded'})[0]
    assessment=_assessment(record,entry_id,version,check.get('assessment'))
    report=record.execution_features.get('task_check') or {}
    observed=report.get('state')
    identity=task_identity(harness,record)
    metadata=get_value(harness.store,row['metadata_ref'])
    known_sources=[{'execution_id':old,'relationship':instance_relation(identity,value)}
                   for old,value in (metadata.get('identities') or {}).items()]
    actual_version=adopted_version(harness,record,entry_id)
    is_use=actual_version==version and not check.get('assessment')
    sourced=bool(assessment.get('source') and assessment.get('assessment') and assessment.get('evidence_refs'))
    state='fact_checked' if observed in ('passed','failed') else 'insufficient_evidence'
    if sourced and observed=='passed' and assessment.get('verdict')=='supported':state='passed'
    if sourced and observed=='failed' and assessment.get('failure_source')=='rule':state='refuted'
    with harness.store.transaction() as conn:
        body=get_value(harness.store,row['body_ref'])
        value={'execution_id':eid,'identity':identity,'known_source_relations':known_sources,
            'reason':'recorded acceptance and agent assessment; applicability and validation purpose are agent decisions',
            'assessment_ref':put_value(conn,harness.store,'rule_assessment',assessment),
            'agent_verdict':assessment.get('verdict','unknown'),
            'fact_ref':_facts_ref(harness,conn,record),'body_ref':row['body_ref'],'fact_basis':fact_basis(record),
            'scope':{'kind':kind,'acceptance':copy.deepcopy(report.get('scope')),
                     'acceptance_rule':copy.deepcopy(report.get('acceptance_rule'))},
            'acceptance_state':observed,'actual_adopted_version':actual_version,'actual_version_use':is_use,
            'basis':'actual_version_use' if is_use else 'historical_review',
            'validity_revision':validity(harness,eid).get('revision',0),
            'observed_at':report.get('checked_at') or record.created_at}
        technical=body.get('effect_scope')=='technical'
        if technical:
            from or_harness.world_model.r_signal import technical_attempt_fact
            fact=technical_attempt_fact(record);value['technical_fact']=fact
            value['scope']['acceptance']={'basis':['executor_protocol'],'observation_rule_version':fact['observation_rule_version']}
            if sourced and fact['label'] is not None:state='passed' if fact['label']==0 else 'refuted'
        assertions=check.get('assertions')
        if assertions:
            from or_harness.strategy.verification import verify_relation
            result=verify_relation(body['expect'],evidence=[record],roles=[{'execution_id':eid,'role':'case'}],assertions=assertions)
            value['assertion_report_ref']=_verification_ref(harness.store,conn,result,row['body_ref'])
            if result['state']=='verified':state='passed'
        comparator=check.get('comparator_execution_id') or assessment.get('comparator_execution_id')
        if comparator:
            comparison,issue=_cost_comparison(harness,record,{'comparator_execution_id':comparator,
                'cost_dimensions':check.get('cost_dimensions') or assessment.get('cost_dimensions')})
            if issue:raise ValueError(issue)
            value['cost_comparison']=comparison
        result,_=append_check(harness.store,entry_id,version,kind,state,value)
    return result


def _apply_confirmed_merges(harness,entry_id,version):
    for event in event_records(harness.store,entry_id,version):
        if event['kind']!='merge':continue
        source=harness.sbank.get(event['source_entry'])
        if source and source.rule.get('merge_pending_into')==str(entry_id) and source.rule.get('merge_pending_version')==version:
            source.rule.update(state='merged',merged_into=str(entry_id))
            source.rule.pop('merge_pending_into',None);source.rule.pop('merge_pending_version',None)
            harness.sbank.update(source)
            append_event(harness.store,entry_id,version,'merge_applied',data={'source_entry':source.entry_id,'authority':'agent'})


def update_publication(harness,entry_id,version,decision=None):
    """Apply an explicit author choice, otherwise report the existing state."""
    entry=harness.sbank.get(str(entry_id));row=version_row(harness.store,entry_id,version)
    if row is None:raise ValueError('unknown rule version')
    if decision is not None:
        choice={'state':decision} if isinstance(decision,str) else copy.deepcopy(decision)
        if not isinstance(choice,dict) or choice.get('state') not in ('published','publish','candidate'):
            raise ValueError('publication needs state published or candidate and a reason')
        if not isinstance(choice.get('reason'),str) or not choice['reason'].strip():
            raise ValueError('publication choice requires a short reason')
        if entry.rule.get('state')=='merged':raise ValueError('manage the selected merge target')
        with harness.store.transaction() as conn:
            if choice['state'] in ('published','publish'):
                entry,_=harness.sbank.apply_lifecycle_action(str(entry_id),'promote',reason=choice['reason'],version=version)
                _apply_confirmed_merges(harness,str(entry_id),version)
            else:
                entry.rule['candidate_version']=version
                if entry.rule.get('active_version')==version:entry.rule['active_version']=None
                if entry.rule.get('active_version') is None:entry.rule['state']='candidate';entry.status='candidate'
                conn.execute("UPDATE rule_versions SET state='candidate' WHERE entry_id=? AND version=?",(str(entry_id),version))
                harness.sbank.update(entry)
            previous=[e for e in event_records(harness.store,entry_id,version) if e['kind']=='publication']
            payload={'decision':choice,'authority':'agent','to':'active' if choice['state'] in ('published','publish') else 'candidate',
                     'previous_event_id':previous[-1]['event_id'] if previous else None}
            if not previous or previous[-1].get('decision')!=choice:
                append_event(harness.store,entry_id,version,'publication',data=payload)
    entry=harness.sbank.get(str(entry_id))
    checks=check_records(harness.store,entry_id,version)
    metadata=get_value(harness.store,row['metadata_ref'])
    choices=[e for e in event_records(harness.store,entry_id,version) if e['kind']=='publication' and e.get('decision')]
    return {'published':entry.is_published and entry.rule.get('active_version')==version,'state':entry.rule['state'],
        'version':version,'authority':'agent','required_checks':[],
        'distinct_tasks':len({e.get('task_id') for e in metadata.get('seed_evidence') or [] if e.get('task_id')}),
        'support_scope':'agent_declared' if choices else 'recorded_facts','transferability':'agent_decision',
        'decision':choices[-1]['decision'] if choices else None,
        'observations':[{'check_id':c['check_id'],'kind':c['kind'],'state':c['state'],'scope':c.get('scope')} for c in checks[-8:]],
        'reason':'publication and transfer are agent judgments over the cited scope'}


def record_uses(harness,records):
    results=[];affected=set()
    for record in records:
        used=record.execution_features.get('used_entries') or {}
        for entry_id in used.get('known_entry_ids') or []:
            version=adopted_version(harness,record,entry_id)
            if version is None or harness.sbank.get(entry_id) is None:continue
            with harness.store.transaction():
                reference=(used.get('versions') or {}).get(entry_id)
                append_event(harness.store,entry_id,version,'use',execution_id=record.execution_id,
                    data={'adopted_value_ref':reference,'actual_version_use':True})
                assessment=_assessment(record,entry_id,version)
                kind=assessment.get('purpose') or ('boundary' if assessment.get('applicability')=='not_applicable' else 'use')
                check=validate_case(harness,entry_id,version,{'kind':kind,'execution_id':record.execution_id})
            results.append({'entry_id':entry_id,'version':version,'check_id':check['check_id'],'state':check['state'],
                'reason':check['reason'],'publication':update_publication(harness,entry_id,version)})
            affected.add(entry_id)
    return {'checks':results,'affected_entry_ids':sorted(affected),'index_entry_ids':[]}


def settle_episode_uses(harness, task_id, episode_id):
    """Freeze attributable checks before retention can evict their facts."""
    actions = harness.actions.query(task_id=task_id,episode_id=episode_id,action_type='execute_strategy')
    ids = {a.linked_execution_id for a in actions if a.linked_execution_id}
    records = [r for r in harness.bank.query(task_id=task_id) if r.execution_id in ids or
               ((r.execution_features or {}).get('task_check') or {}).get('episode_id')==episode_id]
    result = record_uses(harness,records)
    # A retry can still finish deferred indexing after the raw executions
    # have left the window: the versioned use events retain the affected IDs.
    affected = set(result['affected_entry_ids'])
    if ids:
        marks = ','.join('?' for _ in ids)
        affected.update(row[0] for row in harness.store.conn.execute(
            "SELECT DISTINCT entry_id FROM rule_events WHERE kind='use' AND execution_id IN ("+marks+")",sorted(ids)))
    result['affected_entry_ids'] = sorted(affected)
    if affected:
        result['index_sync'] = sync_rule_entries(harness,result['index_entry_ids'],affected)
    return result


def inspect_versions(harness, entry_id):
    rows = harness.store.conn.execute("SELECT * FROM rule_versions WHERE entry_id=? ORDER BY version", (str(entry_id),))
    return {"entry_id": str(entry_id), "versions": [{**dict(r), "body": get_value(harness.store,r["body_ref"]),
        "metadata": get_value(harness.store,r["metadata_ref"])} for r in rows],
        "checks": check_records(harness.store,entry_id), "events": event_records(harness.store,entry_id)}


def refresh_execution_support(harness,execution_id):
    """Compatibility reader: corrections do not rejudge or publish rules."""
    return {'affected_versions':[],'index_entry_ids':[],'authority':'agent',
            'execution_id':execution_id,'reason':'read corrected facts and explicitly manage the entry when warranted'}


def record_effect(harness,entry_id,item):
    """Store the agent's utility judgment against the actually adopted version."""
    from or_harness.strategy.evidence_access import resolve_execution
    record=resolve_execution(harness,str(item.get('execution_id') or ''))
    version=item.get('version') or (adopted_version(harness,record,entry_id) if record else None)
    version=version or 0
    data={k:copy.deepcopy(v) for k,v in item.items() if k not in ('at','execution_id')}
    data['version_basis']='frozen_adoption' if version else 'unknown'
    with harness.store.transaction():
        old=[e for e in event_records(harness.store,entry_id,version) if e['kind']=='effect']
        signature=digest(data)
        if old and old[-1].get('assessment_signature')==signature and old[-1].get('execution_id')==item.get('execution_id'):
            return old[-1]
        data.update(assessment_signature=signature,previous_effect_id=old[-1]['event_id'] if old else None)
        event,_=append_event(harness.store,entry_id,version,'effect',execution_id=item.get('execution_id'),data=data)
        action=item.get('lifecycle_action') or ('demote' if item.get('verdict')=='refuting' else None)
        if action and version:
            entry,transitions=harness.sbank.apply_lifecycle_action(entry_id,action,
                reason=item.get('note') or 'agent attributed a refuting use',version=version)
            append_event(harness.store,entry_id,version,'management',execution_id=item.get('execution_id'),
                data={'effect_event_id':event['event_id'],'action':action,'reason':item.get('note'),
                      'state':entry.rule['state'],'transitions':transitions})
            if action=='promote':_apply_confirmed_merges(harness,entry_id,version)
    return event
