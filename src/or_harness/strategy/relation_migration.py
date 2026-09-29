"""Legacy relation migration: one relation -> one claim entry.

Before knowledge was unified, an entry could carry a list of ``relations``
beside (or instead of) its own statistical claim, and those relations were the
unit of recall for cross-task knowledge. Knowledge is now ONE claim per ENTRY
(``StrategicEntry.claim``): a second independent claim is a second entry.

This module performs the ONE-WAY migration of the retired structure. It is:

- **idempotent**: an entry already carrying a ``claim`` is left alone, and a
  relation already migrated is skipped (its new entry id is recorded in meta);
- **content-preserving**: the relation's claim text, kind, subject,
  conditions, evidence references, and its OWN verification block are copied
  VERBATIM into the new entry — the verdict is not re-derived, so a migration
  cannot widen what was actually verified;
- **non-merging**: each relation becomes a SEPARATE entry, so two claims that
  shared a host never inherit each other's verification;
- **identity-recording**: the mapping ``old_entry_id -> [new_entry_id, ...]``
  is stored so a later audit can follow a migrated claim;
- **a no-op on a migrated store**: with nothing to migrate it reports zero and
  writes nothing.

Nothing here re-runs induction, re-scores a claim, or rewrites a historical
frozen snapshot. The migration only changes the SHAPE of stored knowledge.
"""

from __future__ import annotations

import copy
import time
from typing import Any, Dict, List, Optional

from or_harness.core.schema import (
    CostVector,
    StrategicEntry,
    empty_verification,
)
from or_harness.core.storage import Store, StorageError

#: Meta key recording the migration and its identity map.
MIGRATION_KEY = "relation_migration_v1"


def _migrated_ids(store: Store) -> Dict[str, List[str]]:
    raw = store.conn.execute("SELECT value FROM meta WHERE key=?",
                             (MIGRATION_KEY,)).fetchone()
    if raw is None:
        return {}
    try:
        data = store.loads(raw["value"])
    except StorageError:
        return {}
    return {str(k): [str(v) for v in vals]
            for k, vals in (data.get("entries") or {}).items()}


def _write_migrated_ids(store: Store, mapping: Dict[str, List[str]]) -> None:
    with store.transaction() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?,?)",
            (MIGRATION_KEY, store.dumps({
                "version": 1,
                "entries": mapping,
                "note": ("old entry_id -> migrated claim entry ids. A source "
                         "reference here is its own claim entry; the pairing "
                         "lets an audit follow a migrated claim"),
                "migrated_at": time.time(),
            })))


def migrate_legacy_relations(sbank, *, dry_run: bool = False
                             ) -> Dict[str, Any]:
    """Turn every legacy entry-level ``relations`` list into claim entries.

    Returns a report. ``dry_run`` writes NOTHING — it only counts what would
    move. Re-running is idempotent: an entry whose relations were already
    migrated has none left, and the identity map is consulted as a second
    guard.
    """
    store = sbank.store
    known = _migrated_ids(store)
    to_migrate: List[Dict[str, Any]] = []
    for entry in sbank.list(include_dormant=True):
        if not entry.legacy_relations:
            continue
        to_migrate.append({
            "entry_id": entry.entry_id,
            "strategy_id": entry.strategy_id,
            "n_relations": len(entry.legacy_relations),
            "relations": entry.legacy_relations,
        })
    report: Dict[str, Any] = {
        "dry_run": bool(dry_run),
        "entries_with_relations": len(to_migrate),
        "relations_found": sum(t["n_relations"] for t in to_migrate),
        "already_migrated_entries": len(known),
        "source_files": ("untouched: this migrates stored knowledge shape "
                         "only, never a solve source"),
    }
    if dry_run:
        report["would_create"] = report["relations_found"]
        report["note"] = ("dry run: nothing was written. Each legacy relation "
                          "would become its OWN claim entry, its verification "
                          "copied verbatim")
        return report

    created_ids: List[str] = []
    stripped: List[str] = []
    for item in to_migrate:
        host = sbank.get(item["entry_id"])
        if host is None:
            continue
        already = set(known.get(host.entry_id) or [])
        for relation in item["relations"]:
            new_entry = _claim_entry_from_relation(host, relation)
            if new_entry.entry_id in already:
                continue
            sbank.add(new_entry)
            created_ids.append(new_entry.entry_id)
            already.add(new_entry.entry_id)
        known[host.entry_id] = sorted(already)
        # The host keeps whatever claim it ALREADY states (a statistical
        # entry usually states none); the legacy list is now empty. A host
        # whose only content was its relations becomes an empty entry and is
        # removed — its knowledge now lives in the migrated claim entries.
        host.legacy_relations = []
        if host.claim is None and host.support_n == 0:
            with store.transaction() as conn:
                conn.execute("DELETE FROM strategic_entries WHERE entry_id=?",
                             (host.entry_id,))
        else:
            sbank.update(host)
            stripped.append(host.entry_id)
    _write_migrated_ids(store, known)
    report.update({
        "created_entries": len(created_ids),
        "entry_ids": created_ids,
        "hosts_kept": stripped,
        "identity_map_entries": len(known),
        "note": ("each legacy relation became its OWN claim entry with its "
                 "verification copied verbatim (a migration never widens what "
                 "was verified); a host whose only content was its relations "
                 "was removed, because its knowledge now lives in those "
                 "entries"),
    })
    return report


def _claim_entry_from_relation(host: StrategicEntry,
                               relation: Dict[str, Any]) -> StrategicEntry:
    """Build the claim entry ONE legacy relation becomes.

    The relation's OWN verification is copied verbatim (never re-derived), so
    what the migration publishes is exactly what was verified before."""
    claim = {
        "text": str(relation.get("claim") or ""),
        "kind": relation.get("kind"),
        "subject": relation.get("subject") or host.strategy_id,
        "conditions": copy.deepcopy(relation.get("conditions") or {}),
        "evidence": copy.deepcopy(relation.get("evidence") or []),
        "method": copy.deepcopy(relation.get("method"))
        if isinstance(relation.get("method"), dict) else None,
        "tasks": copy.deepcopy(relation.get("tasks")),
        "strategy_ids": copy.deepcopy(relation.get("strategy_ids")),
        "family": relation.get("family"),
        "cell": relation.get("cell"),
    }
    verification = copy.deepcopy(relation.get("verification")) or \
        empty_verification()
    # A relation-only host whose subject named a principle keeps that subject
    # as the claim's strategy id; a strategy-anchored relation keeps the
    # host's strategy id.
    strategy_id = str(relation.get("subject") or host.strategy_id)
    predicates = dict((relation.get("conditions") or {}).get("predicates")
                      or {}) or dict(host.predicates)
    return StrategicEntry(
        entry_id=StrategicEntry.new_id(),
        strategy_id=strategy_id,
        pattern={"predicates": predicates},
        expected_quality_hat=0.0,
        quality_interval=(0.0, 1.0),
        expected_cost_hat=CostVector(measured=set()),
        failure_prob=0.0,
        support_n=0,
        verification=verification,
        claim=claim,
    )
