"""Explicit pre-versioning payloads for backward-compatibility tests only.

Public induction always writes v2 candidates. These tests exercise reading
the unversioned JSON that already exists in older banks, without changing
the new publication policy to manufacture a historical entry.
"""
import copy

from or_harness.core.schema import StrategicEntry


def as_historical_entry(harness, entry_id):
    payload = copy.deepcopy(harness.sbank.get(entry_id).to_dict())
    payload.pop('rule', None)
    entry = StrategicEntry.from_dict(payload)
    harness.sbank.update(entry)
    # An actual old database has no v2 rows for this entry, either.
    with harness.store.transaction() as conn:
        for table in ('rule_versions','rule_checks','rule_events'):
            conn.execute('DELETE FROM '+table+' WHERE entry_id=?',(entry_id,))
    harness.index_sync.sync_entries([entry_id])
    return entry
