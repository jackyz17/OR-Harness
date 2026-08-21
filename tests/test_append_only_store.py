from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path

from helpers import experience
from or_experience_bank.core.store import AppendOnlyExperienceStore


class AppendOnlyStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.home = Path(self.temp.name)
        self.store = AppendOnlyExperienceStore(self.home)

    def tearDown(self):
        self.temp.cleanup()

    def test_append_preserves_prior_line_and_rejects_exact_duplicate(self):
        first = experience()
        result1 = self.store.append(first)
        path = self.store.layer_path("modeling")
        original_first_line = path.read_bytes().splitlines(keepends=True)[0]
        second = experience(title="Use binary variables for yes/no assignment decisions")
        result2 = self.store.append(second)
        lines = path.read_bytes().splitlines(keepends=True)
        self.assertEqual(result1.status, "appended")
        self.assertEqual(result2.status, "appended")
        self.assertEqual(lines[0], original_first_line)
        duplicate = experience()
        duplicate_result = self.store.append(duplicate)
        self.assertEqual(duplicate_result.status, "duplicate")
        self.assertEqual(len(path.read_bytes().splitlines()), 2)

    def test_no_update_delete_or_merge_api(self):
        for name in ("update", "delete", "merge"):
            self.assertFalse(hasattr(self.store, name))

    def test_concurrent_append_keeps_independent_json_lines(self):
        records = [experience(title="Unique capacity rule {}".format(i)) for i in range(20)]
        threads = [threading.Thread(target=self.store.append, args=(record,)) for record in records]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        lines = self.store.layer_path("modeling").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 20)
        self.assertTrue(all(isinstance(json.loads(line), dict) for line in lines))

