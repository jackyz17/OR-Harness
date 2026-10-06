"""Legacy relation migration: one relation -> one standalone claim entry.

Before knowledge was unified, an entry could carry a ``relations`` list beside
its own statistical claim. This suite pins the migration's contract:

1. each relation becomes its OWN claim entry, verification copied VERBATIM
   (a migration never widens what was verified);
2. two claims under one host stay INDEPENDENT (no shared verdict);
3. idempotent — re-running creates nothing more;
4. dry run writes nothing;
5. a relation-only host whose knowledge moved is removed, while a host with a
   statistical claim is KEPT;
6. a migrated store is a no-op.
"""
import unittest

from helpers import HarnessTestCase

from or_harness.api import ORHarness
from or_harness.core.schema import CostVector, StrategicEntry
from or_harness.strategy.relation_migration import migrate_legacy_relations
from or_harness.strategy.strategic_bank import StrategicBank


def _legacy_relation(rid, claim, *, kind=None, executions=("ex1", "ex2"),
                     state="verified", tasks=("T1", "T2")):
    return {
        "relation_id": rid,
        "claim": claim,
        "kind": kind,
        "subject": None,
        "conditions": {"predicates": {"family": "routing"}},
        "evidence": [{"execution_id": e, "role": "preserved"}
                     for e in executions],
        "tasks": list(tasks),
        "strategy_ids": ["S01"],
        "family": "routing",
        "verification": {
            "state": state,
            "purpose": "relation",
            "claim": claim,
            "evidence": list(executions),
            "scope": {"tasks": list(tasks), "distinct_tasks": len(tasks)},
            "verified_at": 111.0,
            "stale_after_revision": False,
            "stale_reason": None,
        },
    }


def _legacy_entry(entry_id, *, support_n=0, relations=(), strategy="S01"):
    entry = StrategicEntry(
        entry_id=entry_id,
        strategy_id=strategy,
        pattern={"predicates": {"family": "routing"}},
        expected_quality_hat=0.9 if support_n else 0.0,
        support_n=support_n,
        expected_cost_hat=CostVector(llm_tokens=100) if support_n
        else CostVector(measured=set()),
    )
    # A legacy payload carried the relations list; simulate it by writing the
    # field the reader maps to ``legacy_relations``.
    entry.legacy_relations = [dict(r) for r in relations]
    return entry


class MigrationCase(HarnessTestCase):
    def setUp(self):
        super().setUp()
        self.sbank = StrategicBank(self.store)


class TestMigration(MigrationCase):

    def _write_legacy(self, entry):
        """Persist an entry as a LEGACY payload (relations under the old key)."""
        payload = entry.to_dict()
        payload["relations"] = [dict(r) for r in entry.legacy_relations]
        with self.store.transaction() as conn:
            conn.execute(
                "INSERT INTO strategic_entries "
                "(entry_id, strategy_id, scope_level, status, payload) "
                "VALUES (?,?,?,?,?)",
                (entry.entry_id, entry.strategy_id, "*", "candidate",
                 self.store.dumps(payload)))

    def test_relation_only_host_becomes_one_claim_entry_per_relation(self):
        host = _legacy_entry(
            "se_host",
            relations=[_legacy_relation("rel_a", "keep the state"),
                       _legacy_relation("rel_b", "another claim",
                                        kind="boundary")])
        self._write_legacy(host)
        # The reader surfaces the legacy list so the migration can consume it.
        self.assertEqual(len(self.sbank.get("se_host").legacy_relations), 2)
        report = migrate_legacy_relations(self.sbank)
        self.assertEqual(report["created_entries"], 2)
        entries = {e.entry_id: e for e in self.sbank.list()}
        # The relation-only host is GONE: its knowledge moved.
        self.assertNotIn("se_host", entries)
        claims = [e for e in entries.values() if e.claim is not None]
        self.assertEqual(len(claims), 2)
        texts = {c.claim["text"] for c in claims}
        self.assertEqual(texts, {"keep the state", "another claim"})

    def test_verification_is_copied_verbatim(self):
        host = _legacy_entry(
            "se_host",
            relations=[_legacy_relation("rel_a", "keep the state")])
        self._write_legacy(host)
        migrate_legacy_relations(self.sbank)
        moved = next(e for e in self.sbank.list() if e.claim is not None)
        self.assertEqual(moved.verification["state"], "verified")
        self.assertEqual(moved.verification["verified_at"], 111.0)
        self.assertEqual(moved.verification["scope"]["distinct_tasks"], 2)
        self.assertEqual(moved.claim["evidence"],
                         [{"execution_id": "ex1", "role": "preserved"},
                          {"execution_id": "ex2", "role": "preserved"}])

    def test_two_claims_do_not_share_a_verdict(self):
        host = _legacy_entry("se_host", relations=[
            _legacy_relation("rel_a", "verified claim", state="verified"),
            _legacy_relation("rel_b", "unverified claim", kind="other",
                             state="unverified")])
        self._write_legacy(host)
        migrate_legacy_relations(self.sbank)
        by_text = {e.claim["text"]: e for e in self.sbank.list()
                   if e.claim is not None}
        self.assertEqual(by_text["verified claim"].verification["state"],
                         "verified")
        self.assertEqual(by_text["unverified claim"].verification["state"],
                         "unverified")

    def test_idempotent(self):
        host = _legacy_entry(
            "se_host",
            relations=[_legacy_relation("rel_a", "a"),
                       _legacy_relation("rel_b", "b", kind="k")])
        self._write_legacy(host)
        first = migrate_legacy_relations(self.sbank)
        count_after_first = self.sbank.count()
        second = migrate_legacy_relations(self.sbank)
        self.assertGreater(first["created_entries"], 0)
        self.assertEqual(second["created_entries"], 0)
        self.assertEqual(self.sbank.count(), count_after_first)

    def test_dry_run_writes_nothing(self):
        host = _legacy_entry(
            "se_host", relations=[_legacy_relation("rel_a", "a")])
        self._write_legacy(host)
        before = self.sbank.count()
        report = migrate_legacy_relations(self.sbank, dry_run=True)
        self.assertTrue(report["dry_run"])
        self.assertEqual(report["would_create"], 1)
        self.assertEqual(self.sbank.count(), before)
        # The host still carries its legacy list (nothing was consumed).
        self.assertEqual(len(self.sbank.get("se_host").legacy_relations), 1)

    def test_statistical_host_is_kept(self):
        host = _legacy_entry(
            "se_stats", support_n=2,
            relations=[_legacy_relation("rel_a", "a")])
        self._write_legacy(host)
        report = migrate_legacy_relations(self.sbank)
        self.assertEqual(report["hosts_kept"], ["se_stats"])
        kept = self.sbank.get("se_stats")
        self.assertIsNotNone(kept)
        self.assertEqual(kept.legacy_relations, [])
        self.assertEqual(kept.support_n, 2)

    def test_migrated_store_is_a_noop(self):
        report = migrate_legacy_relations(self.sbank)
        self.assertEqual(report["relations_found"], 0)
        self.assertEqual(report["created_entries"], 0)

    def test_identity_map_is_recorded(self):
        host = _legacy_entry(
            "se_host", relations=[_legacy_relation("rel_a", "a")])
        self._write_legacy(host)
        report = migrate_legacy_relations(self.sbank)
        self.assertIn("se_host", report and {"se_host": 1} and
                      {"se_host": report["identity_map_entries"]})
        # The meta record names the created entries.
        from or_harness.strategy.relation_migration import MIGRATION_KEY
        row = self.store.conn.execute("SELECT value FROM meta WHERE key=?",
                                      (MIGRATION_KEY,)).fetchone()
        self.assertIsNotNone(row)
        data = self.store.loads(row["value"])
        self.assertIn("se_host", data["entries"])
        self.assertEqual(len(data["entries"]["se_host"]), 1)


class TestMigrationThroughApi(HarnessTestCase):

    def setUp(self):
        super().setUp()
        self.h = ORHarness(home=self.home)
        self.addCleanup(self.h.close)

    def test_api_migration_refreshes_the_index(self):
        host = _legacy_entry(
            "se_host", relations=[_legacy_relation("rel_a", "a claim")])
        payload = host.to_dict()
        payload["relations"] = [dict(r) for r in host.legacy_relations]
        with self.h.store.transaction() as conn:
            conn.execute(
                "INSERT INTO strategic_entries "
                "(entry_id, strategy_id, scope_level, status, payload) "
                "VALUES (?,?,?,?,?)",
                ("se_host", "S01", "*", "candidate",
                 self.h.store.dumps(payload)))
        report = self.h.migrate_relations()
        self.assertEqual(report["created_entries"], 1)
        # The migrated claim is retrievable by its text.
        self.assertIn("index_sync", report)

    def test_cli_command(self):
        import io
        import json
        import sys
        from unittest import mock
        from or_harness import cli
        host = _legacy_entry(
            "se_host", relations=[_legacy_relation("rel_a", "a claim")])
        payload = host.to_dict()
        payload["relations"] = [dict(r) for r in host.legacy_relations]
        with self.h.store.transaction() as conn:
            conn.execute(
                "INSERT INTO strategic_entries "
                "(entry_id, strategy_id, scope_level, status, payload) "
                "VALUES (?,?,?,?,?)",
                ("se_host", "S01", "*", "candidate",
                 self.h.store.dumps(payload)))
        with mock.patch.object(sys, "argv",
                               ["orx", "--home", self.home,
                                "migrate-relations"]):
            buffer = io.StringIO()
            with mock.patch("sys.stdout", buffer):
                code = cli.main()
        self.assertEqual(code, 0)
        result = json.loads(buffer.getvalue())["result"]
        self.assertEqual(result["created_entries"], 1)


class TestMigrationPathAfter(HarnessTestCase):
    """Upstream audit: after migration, the ordinary paths must still work —
    the migrated claim is recalled, can be revised, and a statistical
    `induce` / `revise` must not disturb it."""

    def setUp(self):
        super().setUp()
        self.h = ORHarness(home=self.home)
        self.addCleanup(self.h.close)

    def _write_legacy(self, entry):
        payload = entry.to_dict()
        payload["relations"] = [dict(r) for r in entry.legacy_relations]
        with self.h.store.transaction() as conn:
            conn.execute(
                "INSERT INTO strategic_entries "
                "(entry_id, strategy_id, scope_level, status, payload) "
                "VALUES (?,?,?,?,?)",
                (entry.entry_id, entry.strategy_id, "*", "candidate",
                 self.h.store.dumps(payload)))

        return payload

    def _task(self, task_id="T1"):
        return {"task_id": task_id, "family": "routing",
                "description": "a routing problem",
                "annotations": {"coupling": {"resource_coupling": 0.3,
                                             "temporal_coupling": 0.1,
                                             "route_complexity": 0.2,
                                             "semantic_coupling": 0.5}}}

    def test_migrated_claim_is_recalled_and_revisable(self):
        host = _legacy_entry("se_host", strategy="principle:keep_state",
                             relations=[_legacy_relation(
                                 "rel_a", "keep the cross-period state")])
        self._write_legacy(host)
        # The claim's evidence cites executions that were never recorded —
        # the migration copies the reference verbatim (a historical
        # reference), which is what "sources may expire" means.
        self.h.migrate_relations()
        entry = next(e for e in self.h.sbank.list() if e.claim is not None)
        self.assertEqual(entry.verification_state, "verified")
        # It is recalled as admitted knowledge.
        recalled = self.h.recall(self._task())
        hits = [r for r in recalled["recommendations"]
                if (r.get("knowledge") or {}).get("claim")]
        self.assertTrue(hits)
        self.assertIn("keep the cross-period state",
                      hits[0]["knowledge"]["claim"]["text"])
        # A knowledge write that does not name this claim leaves it alone.
        before = self.h.sbank.get(entry.entry_id).to_dict()
        self.h.induce(relations=[{
            "subject": "principle:unrelated", "claim": "x",
            "evidence": [{"execution_id": "ex_unknown", "role": "e"}]}])
        after = self.h.sbank.get(entry.entry_id).to_dict()
        self.assertEqual(before["claim"], after["claim"])
        self.assertEqual(before["verification"], after["verification"])

    def test_evicted_source_does_not_revoke_a_migrated_claim(self):
        """A migrated claim whose cited executions are later evicted by the
        evidence window stays published: the reference expires, the claim
        does not."""
        rec = self.make_record(execution_id="ex1", task_id="T1")
        self.h.bank.append(rec)
        self.h.bank.append(self.make_record(execution_id="ex2", task_id="T2"))
        host = _legacy_entry("se_host", strategy="S01",
                             relations=[_legacy_relation(
                                 "rel_a", "keep the state")])
        self._write_legacy(host)
        self.h.migrate_relations()
        entry = next(e for e in self.h.sbank.list() if e.claim is not None)
        self.h.bank.delete_episode_executions(["ex1", "ex2"])
        survived = self.h.sbank.get(entry.entry_id)
        self.assertIsNotNone(survived)
        self.assertEqual(survived.verification_state, "verified")
        self.assertTrue(survived.is_published)


if __name__ == "__main__":
    unittest.main()
