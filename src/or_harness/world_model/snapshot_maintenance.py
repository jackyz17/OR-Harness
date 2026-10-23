"""Explicit, preview-first maintenance and isolated fixed-memory copies."""
from __future__ import annotations

import json
import shutil
import sqlite3
import time
from collections import Counter
from pathlib import Path
from urllib.parse import quote

from or_harness.core.storage import StorageError, Store


def readonly_connection(home):
    path = Path(home).expanduser().resolve() / "or_harness.db"
    if not path.is_file():
        raise FileNotFoundError(path)
    conn = sqlite3.connect("file:" + quote(str(path)) + "?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    return conn


def backup_database(source, destination):
    path = Path(destination).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation: a retry never overwrites an earlier recovery copy.
    with path.open("xb"):
        pass
    target = sqlite3.connect(str(path))
    try:
        source.backup(target)
        if target.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise StorageError("backup integrity check failed")
    finally:
        target.close()
    return str(path)


def fork_memory(source_home, destination):
    """Fresh writable home per test question; same base for every question.

    Call after building/quiescing the base. SQLite backup includes WAL data;
    the base is never opened through Store (which initializes schemas).
    """
    source_home = Path(source_home).expanduser().resolve()
    destination = Path(destination).expanduser().resolve()
    if destination == source_home or source_home in destination.parents:
        raise ValueError("destination must be outside the fixed base home")
    destination.mkdir(parents=True, exist_ok=False)
    source = readonly_connection(source_home)
    try:
        backup_database(source, destination / "or_harness.db")
    finally:
        source.close()
    # Indexes and archived audit artifacts belong to the same frozen base.
    from or_harness.strategy.embedding_index import INDEX_SUBDIR
    for name in (INDEX_SUBDIR, "archive", "artifacts"):
        if (source_home / name).is_dir():
            shutil.copytree(source_home / name, destination / name)
    return {"source": str(source_home), "home": str(destination),
            "mode": "isolated_question_copy",
            "note": "reuse this home for retries of ONE question; fork the next question from the same base"}


def _rows(conn, table):
    exists = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                          (table,)).fetchone()
    return conn.execute(f"SELECT * FROM {table}").fetchall() if exists else []


def _identity(raw):
    return str(raw.get("task_id") or ""), str(raw.get("episode_id") or "")


def _preview(conn, expire_open_days=None):
    """Conservative dependency inspection; malformed/unknown references pin."""
    from or_harness.world_model.state import BeliefSnapshot
    protected = {}
    unknown = []
    closed = set()
    blocks = {}
    context_rows = _rows(conn, "prediction_contexts")
    meta = _rows(conn, "meta")
    snapshot_rows = _rows(conn, "belief_snapshots")
    snapshot_ids = {row["snapshot_id"] for row in snapshot_rows}
    last_activity = {}
    for table in ("belief_snapshots", "frozen_conditions", "prediction_contexts",
                  "contract_predictions", "pending_executions"):
        for row in _rows(conn, table):
            task_id = str(row["task_id"])
            last_activity[task_id] = max(last_activity.get(task_id, 0), float(row["created_at"]))
    expiration_cutoff = (time.time() - float(expire_open_days) * 86400
                         if expire_open_days is not None else None)
    unresolved = {}
    def find_references(value):
        if isinstance(value, str):
            return {value} if value in snapshot_ids else set()
        if isinstance(value, dict):
            return set().union(*(find_references(v) for v in value.values())) if value else set()
        if isinstance(value, list):
            return set().union(*(find_references(v) for v in value)) if value else set()
        return set()
    for row in meta:
        key = str(row["key"])
        # These consumers are explicitly adapted to lightweight conditions.
        if key.startswith(("episode_closeout|", "strategy_evaluation|",
                           "capability_trace|", "knowledge_version|",
                           "calibration_summary|", "paired_feedback|")):
            continue
        try:
            for snapshot_id in find_references(json.loads(row["value"])):
                unresolved.setdefault(snapshot_id, []).append("unresolved meta reference: " + key)
        except ValueError:
            for snapshot_id in snapshot_ids:
                if snapshot_id in str(row["value"]):
                    unresolved.setdefault(snapshot_id, []).append("unparsed meta reference: " + key)
    closed_at = {}
    for row in _rows(conn, "episode_closeouts"):
        identity = (str(row["task_id"]), str(row["episode_id"] or ""))
        closed.add(identity)
        closed_at[identity] = float(row["closed_at"])
    # Archive registry remains authoritative after detail was archived.
    for row in meta:
        if str(row["key"]).startswith("episode_closeout|"):
            try:
                raw = json.loads(row["value"])
                if not raw.get("unfinished_actions"):
                    closed.add(_identity(raw))
                    closed_at[_identity(raw)] = float(raw.get("created_at") or 0)
            except ValueError:
                unknown.append("unreadable closeout registry")
    parsed_actions = []
    for table in ("action_records", "pending_executions"):
        for row in _rows(conn, table):
            try:
                raw = json.loads(row["payload"])
            except (ValueError, TypeError):
                unknown.append("unreadable " + table)
                continue
            if not isinstance(raw, dict):
                unknown.append("invalid " + table)
                continue
            if table == "action_records":
                parsed_actions.append(raw)
                task_id = str(raw.get("task_id") or "")
                last_activity[task_id] = max(last_activity.get(task_id, 0), float(raw.get("ended_at") or raw.get("started_at") or time.time()))
            if table == "pending_executions" or raw.get("status") == "running":
                blocks.setdefault(_identity(raw), []).append(table)
    for row in _rows(conn, "runtime_inputs"):
        if row["expires_at"] > time.time():
            blocks.setdefault((str(row["task_id"]), str(row["episode_id"] or "")), []).append("active prediction input")
    selected_predictions = {raw.get("outcome", {}).get("prediction_id") for raw in parsed_actions}
    from or_harness.world_model.episode_closeout import CalibrationPolicy
    cutoff = time.time() - CalibrationPolicy.from_env().late_check_grace_days * 86400
    action_by_execution = {raw.get("linked_execution_id"): raw for raw in parsed_actions}
    for row in _rows(conn, "executions"):
        try:
            raw = json.loads(row["payload"])
        except ValueError:
            unknown.append("unreadable execution reference")
            continue
        action = action_by_execution.get(raw.get("execution_id"))
        identity = _identity(action or raw)
        if not (raw.get("execution_features") or {}).get("task_check") and closed_at.get(identity, 0) > cutoff:
            blocks.setdefault(identity, []).append("late task check grace period")
    for row in meta:
        if str(row["key"]).startswith("capability_trace|"):
            try:
                raw = json.loads(row["value"])
                if raw.get("assessment") == "expected" and not raw.get("effect_verified"):
                    blocks.setdefault(_identity(raw), []).append("pending H+ validation")
            except ValueError:
                unknown.append("unreadable H+ reference")
    # Recognize all parsed sources of snapshot references. Unknown schema
    # objects are held rather than declared unreferenced.
    for table in ("contract_predictions", "world_model_predictions", "prediction_contexts"):
        for row in _rows(conn, table):
            try:
                raw = json.loads(row["payload"])
                if table == "prediction_contexts" and raw.get("context_version") not in (
                        "wm-context/3", "wm-context/4"):
                    unknown.append("unsupported context version")
                if table == "contract_predictions":
                    info = (raw.get("trace") or {}).get("model_info") or {}
                    if raw.get("prediction_id") in selected_predictions and not info.get("bound_action_id"):
                        blocks.setdefault(_identity(raw.get("candidate") or {}), []).append("selected prediction awaits binding")
            except ValueError:
                unknown.append("unreadable " + table)
    candidates = []
    for row in snapshot_rows:
        reasons = list(unknown) + unresolved.get(row["snapshot_id"], [])
        identity = (str(row["task_id"]), str(row["episode_id"] or ""))
        expire_open = (identity not in closed and expiration_cutoff is not None
                       and last_activity.get(identity[0], time.time()) < expiration_cutoff)
        if identity not in closed and not expire_open:
            reasons.append("episode is not verifiably closed")
        scoped_blocks = list(blocks.get(identity, []))
        # A staged attempt without an episode protects the whole task.
        if (identity[0], "") != identity:
            scoped_blocks.extend(blocks.get((identity[0], ""), []))
        if expire_open:
            # Explicitly expiring a historical view never ends an action,
            # deletes a pending attempt, or labels an H+ claim verified.
            scoped_blocks = [reason for reason in scoped_blocks if reason not in (
                "action_records", "pending_executions", "selected prediction awaits binding")]
        reasons.extend(scoped_blocks)
        try:
            raw = json.loads(row["payload"])
            snap = BeliefSnapshot.from_dict(raw)
            availability = (snap.coverage or {}).get("knowledge_layers")
            if availability is None:
                availability = (snap.harness_state or {}).get("knowledge")
            if expire_open and isinstance(availability, dict):
                reasons = [reason for reason in reasons if reason != "pending H+ validation"]
            if not snap.problem_state.get("task_digest") or not snap.problem_state.get("profile"):
                reasons.append("missing historical task identity/profile")
        except (ValueError, KeyError, TypeError):
            reasons.append("unreadable historical snapshot")
        if reasons:
            protected[row["snapshot_id"]] = sorted(set(reasons))
        else:
            candidates.append({"snapshot_id": row["snapshot_id"],
                               "expire_open_view": bool(expire_open),
                               "payload_bytes": len(row["payload"].encode("utf-8"))})
    releasable_contexts = []
    candidate_ids = {row["snapshot_id"] for row in candidates}
    for row in context_rows:
        try:
            raw = json.loads(row["payload"])
        except ValueError:
            continue
        if raw.get("context_version") == "wm-context/3" and raw.get("snapshot_id") in candidate_ids:
            releasable_contexts.append(row["context_id"])
    return {"dry_run": True, "candidates": candidates,
            "candidate_rows": len(candidates),
            "candidate_payload_bytes": sum(row["payload_bytes"] for row in candidates),
            "protected": protected,
            "protection_counts": dict(Counter(reason for reasons in protected.values() for reason in reasons)),
            "closed_contexts_to_release": releasable_contexts,
            "space": {"page_count": conn.execute("PRAGMA page_count").fetchone()[0],
                      "freelist_count": conn.execute("PRAGMA freelist_count").fetchone()[0],
                      "page_size": conn.execute("PRAGMA page_size").fetchone()[0]},
            "note": "only closed recoverable snapshots can be compacted; unresolved references are protected"}


def _knowledge_version_gc(conn, home, snapshot_ids=(), context_ids=()):
    versions = {str(row["key"]): str(row["value"]) for row in _rows(conn, "meta")
                if str(row["key"]).startswith("knowledge_version|")}
    # An archive whose references have not been decoded pins its values.
    archive = Path(home) / "archive"
    if archive.exists() and any(path.is_file() for path in archive.rglob("*")):
        return {"candidates": [], "protected": list(versions),
                "reason": "archive references have not been resolved; shared versions retained"}
    live = []
    tables = [row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    for table in tables:
        columns = {row[1] for row in conn.execute(f'PRAGMA table_info("{table}")')}
        if "payload" not in columns:
            continue
        for row in _rows(conn, table):
            if table == "belief_snapshots" and row["snapshot_id"] in snapshot_ids:
                continue
            if table == "prediction_contexts" and row["context_id"] in context_ids:
                continue
            live.append(str(row["payload"]))
    live.extend(str(row["value"]) for row in _rows(conn, "meta") if row["key"] not in versions)
    candidates, protected = [], []
    for key in versions:
        version = key.split("|", 1)[1]
        (protected if any(version in value for value in live) else candidates).append(key)
    return {"candidates": candidates, "protected": protected,
            "candidate_payload_bytes": sum(len(versions[key].encode()) for key in candidates)}


def _payload_bytes(conn):
    total = 0
    for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall():
        table = row[0]
        columns = {col[1] for col in conn.execute(f'PRAGMA table_info("{table}")')}
        field = "payload" if "payload" in columns else "value" if table == "meta" else None
        if field:
            total += conn.execute(f'SELECT coalesce(sum(length(cast({field} AS BLOB))),0) FROM "{table}"').fetchone()[0]
    return total


def maintain_snapshots(home, *, apply=False, backup=None, vacuum=False, expire_open_days=None):
    """No writes in preview. Application backs up before opening Store."""
    home = Path(home).expanduser().resolve()
    if vacuum and not apply:
        raise ValueError("--vacuum requires --apply")
    if apply and not backup:
        raise ValueError("--apply requires a new --backup path")
    if expire_open_days is not None and float(expire_open_days) <= 0:
        raise ValueError("--expire-open-days must be positive")
    conn = readonly_connection(home)
    try:
        result = _preview(conn, expire_open_days)
        result["knowledge_versions"] = _knowledge_version_gc(
            conn, home, {row["snapshot_id"] for row in result["candidates"]},
            set(result["closed_contexts_to_release"]))
        if not apply:
            return result
        if Path(backup).expanduser().resolve() == home / "or_harness.db":
            raise ValueError("backup cannot be the source database")
    finally:
        conn.close()
    from or_harness.world_model.conditions import (
        calibration_projection, compact_budget, digest, put_value, read_snapshot, save_snapshot,
    )
    from or_harness.world_model.state import BeliefSnapshot
    store = Store(home)
    try:
        with store.locked():
            backup_path = backup_database(store.conn, backup)
            with store.transaction() as transaction:
                # Recheck references under the write lock; never apply a stale preview.
                result = _preview(transaction, expire_open_days)
                result["payload_bytes_before"] = _payload_bytes(transaction)
                for candidate in result["candidates"]:
                    key = candidate["snapshot_id"]
                    row = transaction.execute("SELECT payload FROM belief_snapshots WHERE snapshot_id=?", (key,)).fetchone()
                    snap = BeliefSnapshot.from_dict(store.loads(row["payload"]))
                    layers = (snap.coverage or {}).get("knowledge_layers")
                    if layers is None:
                        layers = (snap.harness_state or {}).get("knowledge")
                    # Preserve availability separately from actual adoption. No
                    # knowledge body is required for the availability verdict.
                    available = (sorted({str(ref["entry_id"]) for refs in layers.values()
                                         for ref in refs if ref.get("entry_id")})
                                 if isinstance(layers, dict) else None)
                    save_snapshot(store, snap)
                    raw = store.loads(transaction.execute(
                        "SELECT payload FROM frozen_conditions WHERE snapshot_id=?", (key,)).fetchone()["payload"])
                    raw["view_lifecycle"] = "expired_maintenance"
                    if available is not None:
                        raw["legacy_available_ref"] = put_value(transaction, store, "availability", {"entry_ids": available})
                    transaction.execute("UPDATE frozen_conditions SET payload=? WHERE snapshot_id=?", (store.dumps(raw), key))
                    recovered = read_snapshot(store, key)
                    if (recovered.problem_state != snap.problem_state or recovered.budget_state != compact_budget(snap.budget_state)
                            or recovered.task_progress != snap.task_progress or recovered.hypothetical != snap.hypothetical):
                        raise StorageError(f"condition roundtrip failed for {key}")
                    transaction.execute("DELETE FROM belief_snapshots WHERE snapshot_id=?", (key,))
                for context_id in result["closed_contexts_to_release"]:
                    row = transaction.execute("SELECT payload FROM prediction_contexts WHERE context_id=?", (context_id,)).fetchone()
                    data = store.loads(row["payload"])
                    # Expiry is explicit; no fake complete v3 object is returned.
                    header = {k: data.get(k) for k in ("context_id", "context_version", "task_id", "episode_id", "snapshot_id", "created_at")}
                    header["lifecycle"] = "expired_maintenance"
                    header["input_digest"] = digest(data)
                    header["problem_ref"] = put_value(transaction, store, "problem_input", data.get("joint") or {})
                    header["evidence_versions"] = [
                        {"identity": hit.get("identity"), "value_ref": put_value(transaction, store, "evidence", hit)}
                        for hit in (data.get("retrieval") or {}).get("hits", [])]
                    header["target_versions"] = [put_value(transaction, store, "knowledge_target", target)
                                                  for target in data.get("knowledge_targets", [])]
                    header["feedback_versions"] = {name: put_value(transaction, store, "feedback", value)
                        for name, value in (("calibration", calibration_projection(data.get("strategy_calibration"))),
                                            ("paired", data.get("paired_feedback")),
                                            ("hplus", data.get("hplus_feedback"))) if value}
                    transaction.execute("UPDATE prediction_contexts SET payload=? WHERE context_id=?", (store.dumps(header), context_id))
                gc = _knowledge_version_gc(transaction, home)
                for key in gc["candidates"]:
                    transaction.execute("DELETE FROM meta WHERE key=?", (key,))
                result["knowledge_versions"] = gc
                result["payload_bytes_after"] = _payload_bytes(transaction)
                result.update({"dry_run": False, "backup": backup_path,
                               "compacted_rows": result["candidate_rows"]})
        if vacuum:
            with store.locked():
                store.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                store.conn.execute("VACUUM")
        result["physical_reclamation_requested"] = bool(vacuum)
        result["space_after"] = _preview(store.conn, expire_open_days)["space"]
        result["database_file_bytes"] = (home / "or_harness.db").stat().st_size
        return result
    finally:
        store.close()
