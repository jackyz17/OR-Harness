"""Small frozen facts, immutable referenced values and consumable inputs.

The legacy snapshot table is READ ONLY here. No current bank is consulted to
reconstruct a past condition. Full inputs live only in runtime_inputs, with
an explicit lease, consumption and capacity limit; prediction_contexts keeps
the small audit manifest after consumption.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import time
from contextlib import contextmanager
from collections import Counter
from typing import Any, Dict, Optional

from or_harness.core.storage import StorageError


class ContextExpired(StorageError):
    """The original input has been released; create a new decision."""


def digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def put_value(conn, store, kind: str, value: Any) -> str:
    key = kind + ":" + digest(value)
    conn.execute("INSERT OR IGNORE INTO condition_values VALUES (?,?,?)",
                 (key, kind, store.dumps(value)))
    return key


def get_value(store, key: str) -> Dict[str, Any]:
    row = store.conn.execute(
        "SELECT payload FROM condition_values WHERE value_id=?", (key,)
    ).fetchone()
    if row is None:
        raise StorageError(f"missing immutable condition value {key!r}")
    value = store.loads(row["payload"])
    if digest(value) != key.split(":", 1)[-1]:
        raise StorageError(f"immutable condition value integrity failure: {key!r}")
    return value


def compact_budget(state):
    """Freeze aggregates and unknown reasons, not another cost event history."""
    out = copy.deepcopy(state or {})
    consumption = out.get("consumption")
    if isinstance(consumption, dict):
        omitted = {}
        for key in ("attempts", "action_costs", "prediction_call_costs", "unattributed_prediction_costs"):
            rows = consumption.pop(key, None)
            if rows:
                omitted[key] = len(rows)
        unattributed = consumption.get("unattributed")
        if isinstance(unattributed, dict):
            unattributed.pop("execution_ids", None)
        for dim, reasons in (consumption.get("unknown_dim_reasons") or {}).items():
            counts = Counter()
            for reason in reasons:
                counts[(str(reason.get("kind")), str(reason.get("source")), str(reason.get("reason")))] += int(reason.get("count", 1))
            consumption["unknown_dim_reasons"][dim] = [
                {"kind": kind, "source": source, "reason": reason, "count": count}
                for (kind, source, reason), count in sorted(counts.items())]
        if omitted:
            consumption["detail_omitted"] = omitted
            consumption["detail_source"] = "execution/action/prediction facts; frozen aggregates retained"
    return out


def save_snapshot(store, snapshot, *, availability_revision=None) -> None:
    """Persist only X/B, identities, and references, never coverage."""
    with store.transaction() as conn:
        raw = snapshot.to_dict()
        raw.pop("coverage", None)
        raw["harness_state"].pop("knowledge", None)
        raw["budget_state"] = compact_budget(raw.get("budget_state"))
        raw["condition_version"] = "wm-conditions/1"
        raw["problem_ref"] = put_value(conn, store, "problem",
                                       raw.pop("problem_state"))
        raw["progress_ref"] = put_value(conn, store, "progress",
                                        raw.pop("task_progress"))
        raw["availability_revision"] = availability_revision
        conn.execute(
            "INSERT OR IGNORE INTO frozen_conditions VALUES (?,?,?,?,?,?,?,?)",
            (snapshot.snapshot_id, snapshot.task_id, snapshot.episode_id,
             snapshot.problem_state.get("task_digest"),
             snapshot.problem_state.get("problem_identity_digest"),
             int(snapshot.hypothetical), snapshot.created_at, store.dumps(raw)))


def read_snapshot(store, snapshot_id):
    """One compatibility boundary: compact conditions first, old payload next."""
    from or_harness.world_model.state import BeliefSnapshot
    if not snapshot_id:
        return None
    row = store.conn.execute(
        "SELECT payload FROM frozen_conditions WHERE snapshot_id=?",
        (snapshot_id,)).fetchone()
    if row is not None:
        raw = store.loads(row["payload"])
        raw["problem_state"] = get_value(store, raw["problem_ref"])
        raw["task_progress"] = get_value(store, raw["progress_ref"])
        raw["coverage"] = {
            "conditions_only": True,
            "missing": ["full prediction view was temporary; no coverage saved"],
            "availability_revision": raw.get("availability_revision"),
            "legacy_available_ref": raw.get("legacy_available_ref"),
            "view_lifecycle": raw.get("view_lifecycle", "conditions_only"),
        }
        return BeliefSnapshot.from_dict(raw)
    row = store.conn.execute(
        "SELECT payload FROM belief_snapshots WHERE snapshot_id=?",
        (snapshot_id,)).fetchone()
    return BeliefSnapshot.from_dict(store.loads(row["payload"])) if row else None


def latest_snapshot(store, task_id, episode_id=None, *, task_digest=None,
                    episode_scoped=True, problem_identity=None, at=None):
    """Indexed current-state read, with a lazy legacy fallback."""
    best = None
    for table in ("frozen_conditions", "belief_snapshots"):
        clauses, args = ["task_id=?"], [task_id]
        if at is not None:
            clauses.append("created_at<=?")
            args.append(at)
        if episode_scoped:
            clauses.append("episode_id IS ?")
            args.append(episode_id)
        if table == "frozen_conditions":
            clauses.append("hypothetical=0")
            if problem_identity is not None:
                clauses.append("(problem_identity_digest=? OR problem_identity_digest IS NULL)")
                args.append(problem_identity)
            elif task_digest is not None:
                clauses.append("(task_digest=? OR task_digest IS NULL)")
                args.append(task_digest)
        rows = store.conn.execute(
            f"SELECT snapshot_id, created_at FROM {table} WHERE "
            + " AND ".join(clauses) + " ORDER BY created_at DESC, snapshot_id DESC",
            args)
        for row in rows:
            if best and (row["created_at"], row["snapshot_id"]) <= (
                    best.created_at, best.snapshot_id):
                break
            snap = read_snapshot(store, row["snapshot_id"])
            if snap.hypothetical:
                continue
            recorded = snap.problem_state.get("task_digest")
            recorded_identity = snap.problem_state.get("problem_identity_digest")
            if problem_identity and recorded_identity:
                if problem_identity != recorded_identity:
                    continue
            elif task_digest and recorded and recorded != task_digest:
                continue
            best = snap
            break
    return best


def prediction_conditions(store, prediction):
    """Resolve frozen prediction facts across v4 manifests, v3 and snapshots.

    This never consumes/reopens a runtime input or consults current banks.
    None is an explicit unknown; a recorded no-budget declaration is kept.
    """
    trace = getattr(prediction, "trace", None)
    info = (getattr(trace, "model_info", None) if trace is not None
            else getattr(prediction, "model_info", None)) or {}
    out = {"budget_state": copy.deepcopy(info.get("frozen_budget_state")),
           "hypothetical": info.get("hypothetical"),
           "problem_identity_digest": info.get("problem_identity_digest"),
           "missing": []}
    if (out["budget_state"] is not None and out["hypothetical"] is not None
            and out["problem_identity_digest"]):
        return out
    envelope = {}
    context_id = info.get("prediction_context_id")
    if context_id:
        raw = store.get_prediction_context(context_id)
        if raw:
            envelope = store.loads(raw)
            if envelope.get("context_version") not in ("wm-context/3", "wm-context/4"):
                out["missing"].append("unsupported frozen context version")
                envelope = {}
    frozen = envelope.get("conditions") or envelope.get("snapshot") or {}
    if out["hypothetical"] is None:
        out["hypothetical"] = frozen.get("hypothetical")
    if not out["problem_identity_digest"]:
        out["problem_identity_digest"] = frozen.get("problem_identity_digest")
    if out["budget_state"] is None:
        out["budget_state"] = copy.deepcopy(frozen.get("budget_state") or
            (envelope.get("solving_context") or {}).get("budget_state"))
    if (out["budget_state"] is not None and out["hypothetical"] is not None
            and out["problem_identity_digest"]):
        return out
    reference = (info.get("conditions_id") or frozen.get("snapshot_id")
                 or envelope.get("snapshot_id") or getattr(trace, "input_snapshot_id", None)
                 or getattr(prediction, "input_snapshot_id", None))
    snapshot = read_snapshot(store, reference) if reference else None
    if snapshot is not None:
        if out["hypothetical"] is None:
            out["hypothetical"] = snapshot.hypothetical
        if out["budget_state"] is None:
            out["budget_state"] = copy.deepcopy(snapshot.budget_state)
        if not out["problem_identity_digest"]:
            out["problem_identity_digest"] = snapshot.problem_state.get("problem_identity_digest")
    if out["hypothetical"] is None:
        out["missing"].append("historical real/hypothetical source unavailable")
    if out["budget_state"] is None:
        out["missing"].append("historical budget condition unavailable")
    return out


def record_availability(conn, store, entry_id, predicates, *, present=True):
    """One small applicability revision per actual change, not per freeze."""
    raw = {"predicates": predicates, "present": bool(present)}
    version = digest(raw)
    last = conn.execute(
        "SELECT version FROM knowledge_availability WHERE entry_id=? "
        "ORDER BY revision DESC LIMIT 1", (entry_id,)).fetchone()
    if last is None or last["version"] != version:
        conn.execute(
            "INSERT INTO knowledge_availability(entry_id,version,payload) VALUES (?,?,?)",
            (entry_id, version, store.dumps(raw)))


def freeze_availability(store, knowledge):
    # Also indexes legacy matching entries the first time they are observed;
    # it does not migrate or rewrite their historical payloads.
    with store.transaction() as conn:
        for refs in knowledge.values():
            for ref in refs:
                record_availability(conn, store, ref["entry_id"],
                                    ref.get("predicates") or {})
        row = conn.execute("SELECT MAX(revision) AS n FROM knowledge_availability").fetchone()
        return int(row["n"] or 0)


def available_entry_ids(store, revision, profile, entry_ids=None):
    from or_harness.core.schema import ProblemProfile, profile_matches
    if revision is None:
        return None
    args = [revision]
    restriction = ""
    if entry_ids is not None:
        if not entry_ids:
            return []
        restriction = " AND entry_id IN (" + ",".join("?" for _ in entry_ids) + ")"
        args.extend(sorted(entry_ids))
    rows = store.conn.execute(
        "SELECT entry_id,payload FROM knowledge_availability WHERE revision IN "
        "(SELECT MAX(revision) FROM knowledge_availability WHERE revision<=?"
        + restriction + " GROUP BY entry_id)", args)
    profile = ProblemProfile.from_dict(profile)
    return sorted(row["entry_id"] for row in rows
                  if (lambda raw: raw["present"] and profile_matches(
                      profile, raw["predicates"]))(store.loads(row["payload"])))


def calibration_projection(summary):
    """Keep audit details at their source; send counts/reasons to the model."""
    out = copy.deepcopy(summary or {})
    corrections = out.pop("validity_corrections", [])
    if corrections or "validity_correction_summary" not in out:
        out["validity_correction_summary"] = {
            "count": len(corrections),
            "kinds": dict(Counter(str(c.get("kind") or "unspecified")
                                  for c in corrections)),
            "fields": dict(Counter(str(f) for c in corrections
                                   for f in (c.get("fields") or []))),
            "scope": "published summary population before candidate filtering",
            "detail_source": "stored strategy_evaluation and published summary",
        }
    return out


def release_context(store, context_id, *, reason="consumed"):
    with store.transaction() as conn:
        leased = conn.execute("SELECT in_use,lease_until FROM runtime_inputs WHERE context_id=?", (context_id,)).fetchone()
        if leased and leased["in_use"] and leased["lease_until"] > time.time():
            return False
        row = conn.execute("SELECT payload FROM prediction_contexts WHERE context_id=?",
                           (context_id,)).fetchone()
        if row:
            header = store.loads(row["payload"])
            if header.get("context_version") != "wm-context/4":
                return  # historical artifacts are never automatically rewritten
            header["lifecycle"] = reason
            conn.execute("UPDATE prediction_contexts SET payload=? WHERE context_id=?",
                         (store.dumps(header), context_id))
        conn.execute("DELETE FROM runtime_inputs WHERE context_id=?", (context_id,))


def expire_contexts(store):
    now = time.time()
    rows = store.conn.execute(
        "SELECT context_id FROM runtime_inputs WHERE (expires_at<=? AND in_use=0) "
        "OR (in_use>0 AND lease_until<=?)", (now, now)).fetchall()
    for row in rows:
        release_context(store, row["context_id"], reason="expired")


def persist_context(store, context):
    """One bounded input plus a durable, small manifest of actual references."""
    expire_contexts(store)
    raw = context.to_dict()
    existing = store.get_prediction_context(context.context_id)
    if existing is not None:
        read_context(store, context.context_id)
        if store.loads(existing).get("input_digest") != digest(raw):
            raise StorageError("cannot overwrite an existing frozen context with different input")
        return
    header = {k: copy.deepcopy(raw.get(k)) for k in (
        "context_version", "context_id", "task_id", "task_digest",
        "effective_input_digest", "episode_id", "snapshot_id", "created_at")}
    header["lifecycle"] = "active"
    header["input_digest"] = digest(raw)
    header["conditions"] = {k: copy.deepcopy(v) for k, v in context.snapshot.items()
                            if k not in ("coverage", "harness_state")}
    # Only actually selected evidence gets an immutable historical value.
    with store.transaction() as conn:
        header["evidence_versions"] = [
            {"identity": hit.get("identity"), "value_ref": put_value(
                conn, store, "evidence", hit)} for hit in context.retrieval.hits]
        header["problem_ref"] = put_value(conn, store, "problem_input",
                                          context.joint.to_dict())
        # The large problem is stored once per actual version even while
        # multiple decisions are active. Runtime input is never a full bank.
        raw["joint"] = {"value_ref": header["problem_ref"]}
        header["feedback_versions"] = {
            name: put_value(conn, store, "feedback", value)
            for name, value in (("calibration", context.strategy_calibration),
                                ("paired", context.paired_feedback),
                                ("hplus", context.hplus_feedback)) if value}
        header["target_versions"] = [put_value(conn, store, "knowledge_target", target)
                                     for target in context.knowledge_targets]
        conn.execute("INSERT OR IGNORE INTO prediction_contexts VALUES (?,?,?,?,?)",
                     (context.context_id, context.task_id, context.episode_id,
                      context.created_at, store.dumps(header)))
        conn.execute("INSERT OR IGNORE INTO runtime_inputs(context_id,task_id,episode_id,expires_at,payload) VALUES (?,?,?,?,?)",
                     (context.context_id, context.task_id, context.episode_id,
                      context.created_at + context_ttl(), store.dumps(raw)))
    # Capacity expiry is explicit, never replacement with today's inputs.
    max_count = max(1, int(os.environ.get("OR_HARNESS_ACTIVE_CONTEXTS", "128")))
    rows = store.conn.execute(
        "SELECT context_id,in_use FROM runtime_inputs ORDER BY in_use DESC, expires_at DESC, context_id DESC"
    ).fetchall()
    for row in rows[max_count:]:
        if not row["in_use"]:
            release_context(store, row["context_id"], reason="expired_capacity")
    max_bytes = max(1024, int(os.environ.get("OR_HARNESS_ACTIVE_CONTEXT_BYTES", "16777216")))
    rows = store.conn.execute(
        "SELECT context_id,in_use,length(CAST(payload AS BLOB)) AS n FROM runtime_inputs "
        "ORDER BY in_use DESC,expires_at DESC,context_id DESC").fetchall()
    used = 0
    for row in rows:
        used += row["n"]
        if used > max_bytes and not row["in_use"]:
            release_context(store, row["context_id"], reason="expired_capacity")
    read_context(store, context.context_id)  # refuse a new input that cannot fit


def context_ttl():
    return max(1.0, float(os.environ.get("OR_HARNESS_CONTEXT_TTL_S", "86400")))


def read_context(store, context_id):
    from or_harness.world_model.context import PredictionContext
    raw = store.get_prediction_context(context_id)
    if raw is None:
        return None
    data = store.loads(raw)
    if str(data.get("lifecycle", "")).startswith(("expired", "closed", "consumed")):
        raise ContextExpired(f"context {context_id!r} expired or consumed; create a new decision")
    if data.get("context_version") == "wm-context/4":
        row = store.conn.execute(
            "SELECT payload,expires_at FROM runtime_inputs WHERE context_id=?",
            (context_id,)).fetchone()
        if row is None or row["expires_at"] <= time.time():
            raise ContextExpired(f"context {context_id!r} expired or consumed; "
                                 "create a new decision (audit manifest remains readable)")
        data = store.loads(row["payload"])
        if isinstance(data.get("joint"), dict) and data["joint"].get("value_ref"):
            data["joint"] = get_value(store, data["joint"]["value_ref"])
    return PredictionContext.from_dict(data)


def prepare_context(store, context):
    """Make frozen references durable BEFORE a provider can spend usage."""
    from or_harness.world_model.context import snapshot_from_context
    if context.version != "wm-context/4":
        return  # never change a legacy frozen input
    existing = store.get_prediction_context(context.context_id)
    if existing is not None:
        read_context(store, context.context_id)  # includes consumed/capacity checks
        return
    if context.created_at + context_ttl() <= time.time():
        raise ContextExpired(f"context {context.context_id!r} expired; create a new decision")
    with store.transaction():
        snap = snapshot_from_context(context)
        if snap.snapshot_id:
            save_snapshot(store, snap)
        persist_context(store, context)


@contextmanager
def lease_context(store, context):
    """Protect an in-flight call without holding a write lock over network I/O."""
    if context.version != "wm-context/4":
        yield
        return
    with store.transaction() as conn:
        changed = conn.execute(
            "UPDATE runtime_inputs SET in_use=in_use+1,lease_until=? WHERE context_id=? AND expires_at>?",
            (time.time() + max(1, float(os.environ.get("OR_HARNESS_CONTEXT_CALL_LEASE_S", "3600"))),
             context.context_id, time.time())).rowcount
        if not changed:
            raise ContextExpired(f"context {context.context_id!r} expired before provider call")
    try:
        yield
    finally:
        with store.transaction() as conn:
            conn.execute("UPDATE runtime_inputs SET in_use=MAX(0,in_use-1) WHERE context_id=?",
                         (context.context_id,))
