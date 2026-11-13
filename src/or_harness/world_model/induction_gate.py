"""Historical close-out judgements and compatibility metadata primitives.

New building consumes the frozen binary signal in hplus.py. These old
three-state judgements remain readable and never override that switch.
"""
from __future__ import annotations

import copy
import json

from or_harness.world_model.conditions import get_value, put_value


def meta_read(store, key):
    row = store.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return json.loads(row["value"]) if row else None


def meta_write(store, key, value):
    store.conn.execute("INSERT OR REPLACE INTO meta(key,value) VALUES (?,?)", (key, store.dumps(value)))


def record_decision(harness, task_id, episode_id, decision=None):
    from or_harness.world_model.hplus import formal
    if formal(harness):
        return start_decision(harness,task_id)
    key = "induction_gate:" + str(task_id) + ":" + str(episode_id or "")
    previous = meta_read(harness.store, key)
    if decision is None:
        value = previous or {"state": "unavailable", "source": "hplus_not_connected",
                             "reason": "explicit induction remains available during transition"}
    else:
        value = {"state": decision} if isinstance(decision, str) else copy.deepcopy(decision)
        if value.get("state") not in ("allow", "skip", "pending"):
            raise ValueError("induction decision state must be allow, skip or pending")
        value.setdefault("source", "explicit_closeout_judgement")
        value.setdefault("reason", "reported close-out decision; no threshold inferred")
    value.update(task_id=str(task_id), episode_id=episode_id)
    with harness.store.transaction() as conn:
        ref = put_value(conn, harness.store, "induction_gate", value)
        meta_write(harness.store, key, value)
        latest=meta_read(harness.store,'induction_gate:last:'+str(task_id))
        if not previous or not latest or latest.get('episode_id')==episode_id:
            meta_write(harness.store, "induction_gate:last:" + str(task_id), value)
    return {**value, "decision_ref": ref}


def start_decision(harness, task_id,episode_id=None):
    from or_harness.world_model.hplus import formal, require_job
    if formal(harness):
        try:
            job=require_job(harness,task_id=task_id)
            return {'state':'allow','source':'frozen_hplus','induction_id':job['induction_id'],
                    'reason':'positive bound job requires one induction receipt'}
        except ValueError as exc:
            return {'state':'skip','source':'frozen_hplus','reason':str(exc)}
    value = (meta_read(harness.store,'induction_gate:'+str(task_id)+':'+str(episode_id)) if episode_id is not None else
             meta_read(harness.store, "induction_gate:last:" + str(task_id))) if task_id is not None else None
    if value and value.get("state") in ("allow", "skip", "pending"):
        from or_harness.strategy.evidence_views import chars
        from or_harness.world_model.conditions import digest
        if chars(value)>1200:
            reference='induction_gate:'+digest(value)
            return {'state':value['state'],'source':str(value.get('source') or '')[:100],
                    'reason':'Full close-out judgement by immutable reference','decision_ref':reference}
        return value
    return {"state": "allow", "source": "explicit_induction_transition",
            "reason": "H+ close-out judgement unavailable; caller explicitly requested induction"}
