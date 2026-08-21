from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from helpers import SKILL_DIR, experience


CLI = SKILL_DIR / "scripts" / "or_experience_cli.py"


class CLITests(unittest.TestCase):
    def run_cli(self, home, *args):
        env = dict(os.environ)
        env["OR_EXPERIENCE_BANK_HOME"] = str(home)
        return subprocess.run([sys.executable, str(CLI), *args], capture_output=True, text=True, env=env, timeout=20)

    def test_append_retrieve_stats_json_and_no_mutating_commands(self):
        with tempfile.TemporaryDirectory() as temp:
            home = Path(temp) / "bank-home"
            record_path = Path(temp) / "experience.json"
            record_path.write_text(json.dumps(experience().to_dict()), encoding="utf-8")
            appended = self.run_cli(home, "append", "--input", str(record_path), "--json")
            self.assertEqual(appended.returncode, 0, appended.stderr)
            self.assertEqual(json.loads(appended.stdout)["status"], "appended")
            retrieved = self.run_cli(home, "retrieve", "--layer", "modeling", "--query", "assignment capacity", "--json")
            self.assertEqual(retrieved.returncode, 0, retrieved.stderr)
            self.assertEqual(len(json.loads(retrieved.stdout)), 1)
            stats = self.run_cli(home, "stats", "--json")
            self.assertEqual(json.loads(stats.stdout)["total"], 1)
            help_result = self.run_cli(home, "--help")
            self.assertNotIn("update", help_result.stdout)
            self.assertNotIn("delete", help_result.stdout)

    def test_validate_bank_detects_corrupt_line(self):
        with tempfile.TemporaryDirectory() as temp:
            home = Path(temp) / "bank-home"
            self.run_cli(home, "stats", "--json")
            with (home / "bank" / "repair.jsonl").open("a", encoding="utf-8") as handle:
                handle.write("{broken json\n")
            result = self.run_cli(home, "validate-bank", "--json")
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(result.stdout)
            self.assertFalse(payload["valid"])
            self.assertTrue(payload["errors"])

