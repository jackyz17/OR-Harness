"""Legacy-database upgrade compatibility, using a REAL b919497-written bank.

The fixture under ``fixtures/legacy_b919497`` was produced by the b919497
checkout itself (its own ``ORHarness``), not by simulating the old format:
4 routing executions at rc 0.30/0.45/0.62/0.80 and one induced entry. Its
``executions.group_l1`` column holds the RETIRED format
(``family=routing|sc[0.75,1.00]|rc[0.25,0.50]|...``), which is the whole
point — the new code's queries must still find those facts.
"""
import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path

from helpers import HarnessTestCase

from or_harness.api import ORHarness
from or_harness.core.schema import group_key

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "legacy_b919497"
#: A bank written by the release that RETIRED the ladder (``121e29a``). Its
#: group_l1 holds the plain ``family=routing`` form, which an earlier
#: compatibility pass mistook for healthy while the statistics saw nothing.
FIXTURE_121 = Path(__file__).resolve().parent / "fixtures" / "legacy_121e29a"


class TestLegacyDatabaseUpgrade(HarnessTestCase):
    def setUp(self):
        super().setUp()
        # Copy the fixture so a test can never mutate the checked-in bank.
        self.home = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.home, True)
        shutil.copytree(FIXTURE, self.home, dirs_exist_ok=True)
        # HarnessTestCase created its own Store on a different directory.
        self.store.close()

    def _legacy_index_rows(self) -> list:
        conn = sqlite3.connect(str(Path(self.home) / "or_harness.db"))
        try:
            return conn.execute("SELECT group_l1 FROM executions").fetchall()
        finally:
            conn.close()

    def test_index_column_really_is_legacy(self):
        rows = self._legacy_index_rows()
        self.assertEqual(len(rows), 4)
        for (value,) in rows:
            self.assertIn("|", value, "fixture must carry the retired format")

    def test_open_does_not_rewrite_the_fixture(self):
        """Read-only compatibility: opening the bank must not write."""
        before = self._legacy_index_rows()
        h = ORHarness(home=self.home)
        try:
            self.assertEqual(h.bank.count(), 4)
            self.assertEqual(self._legacy_index_rows(), before)
            health = h.bank.index_health()
            self.assertEqual(health["stale_group_index"], 4)
        finally:
            h.close()
        self.assertEqual(self._legacy_index_rows(), before)

    def test_statistics_see_the_historical_facts(self):
        h = ORHarness(home=self.home)
        try:
            records = h.bank.all()
            self.assertEqual(len(records), 4)
            # The cell query is what statistics/induction use.
            for rec in records:
                key = group_key(rec.profile_snapshot)
                found = h.bank.query(group_l1=key)
                self.assertIn(rec.execution_id,
                              {r.execution_id for r in found})
            profile = records[0].profile_snapshot
            self.assertEqual(len(h.stats.evidence(profile, "S01")), 2)
            self.assertEqual(h.stats.for_profile(profile)["S01"].n, 2)
        finally:
            h.close()

    def test_induction_reuses_the_legacy_entry(self):
        """The legacy entry's cell (rc[0.25,0.50]) is re-induced: the same
        knowledge object is refreshed, not duplicated."""
        h = ORHarness(home=self.home)
        try:
            before = {e.entry_id for e in h.sbank.list()}
            entry = h.sbank.list()[0]
            profile = next(r.profile_snapshot for r in h.bank.all()
                           if group_key(r.profile_snapshot).endswith("rc[0.25,0.50]|tc[0.00,0.25]|rx[0.75,1.00]"))
            out = h.induce(strategy_id="S01")["results"]
            touched = [r for r in out if r.get("created") or r.get("updated")]
            self.assertTrue(touched)
            self.assertEqual({e.entry_id for e in h.sbank.list()}, before)
            self.assertEqual(h.sbank.get(entry.entry_id).predicates["family"],
                             "routing")
        finally:
            h.close()

    def test_user_fields_survive_the_upgrade(self):
        """No silent loss of what the user/agent put in the entry."""
        h = ORHarness(home=self.home)
        try:
            entry = h.sbank.list()[0]
            self.assertEqual(entry.strategy_id, "S01")
            self.assertEqual(entry.support_n, 2)
            self.assertEqual(len(entry.provenance), 2)
            self.assertTrue(entry.quality_interval[0] <= entry.quality_interval[1])
            self.assertEqual(entry.status, "candidate")
        finally:
            h.close()


class TestLegacy121e29aUpgrade(HarnessTestCase):
    """The PREVIOUS release's bank must not go dark either.

    ``121e29a`` wrote ``group_l1`` as the plain ``family=routing``. A
    compatibility pass that only understood the ladder-era ``|``-joined form
    reported this database as healthy while every group-keyed query returned
    nothing — facts visible to ``inspect`` but invisible to the statistics.
    """

    def setUp(self):
        super().setUp()
        self.home = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.home, True)
        shutil.copytree(FIXTURE_121, self.home, dirs_exist_ok=True)
        self.store.close()

    def test_open_does_not_rewrite_the_fixture(self):
        conn = sqlite3.connect(str(Path(self.home) / "or_harness.db"))
        before = conn.execute("SELECT group_l1 FROM executions").fetchall()
        conn.close()
        h = ORHarness(home=self.home)
        try:
            self.assertEqual(h.bank.count(), 2)
        finally:
            h.close()
        conn = sqlite3.connect(str(Path(self.home) / "or_harness.db"))
        self.assertEqual(conn.execute("SELECT group_l1 FROM executions").fetchall(),
                         before)
        conn.close()

    def test_statistics_see_the_previous_releases_facts(self):
        h = ORHarness(home=self.home)
        try:
            records = h.bank.all()
            self.assertEqual(len(records), 2)
            profile = records[0].profile_snapshot
            key = group_key(profile)
            self.assertEqual(len(h.bank.query(group_l1=key)), 2)
            self.assertEqual(len(h.stats.evidence(profile, "S01")), 2)
            self.assertEqual(h.stats.for_profile(profile)["S01"].n, 2)
            # Induction is no longer blind to this evidence.
            out = h.induce(strategy_id="S01")["results"][0]
            self.assertIsNotNone(out.get("created") or out.get("updated"))
        finally:
            h.close()

    def test_index_health_notices_this_format(self):
        """The old check only looked for the ladder-era form and called this
        database clean, right next to a statistic of zero."""
        h = ORHarness(home=self.home)
        try:
            self.assertEqual(h.bank.index_health()["stale_group_index"], 2)
        finally:
            h.close()


if __name__ == "__main__":
    unittest.main()
