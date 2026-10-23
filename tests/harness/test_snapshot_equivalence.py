"""Compare actual provider inputs with the captured local v3 baseline."""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


class TestStorageEquivalence(unittest.TestCase):
    def test_fixed_memory_hits_statistics_targets_and_provider_input(self):
        root = Path(__file__).resolve().parents[2]
        fixture = json.loads((Path(__file__).parent / "fixtures/snapshot_input_v3.json").read_text())
        env = dict(os.environ, PYTHONPATH=str(root / "src"), PYTHONDONTWRITEBYTECODE="1")
        actual = json.loads(subprocess.check_output([
            sys.executable, str(root / "docs/world_model/repro_snapshot_lifecycle.py")], env=env, cwd=root))
        for expected, current in zip(fixture["cases"], actual):
            with self.subTest(strategies=expected["strategies"]):
                for key in ("provider_semantics", "candidate_statistics", "cell_evidence", "knowledge_targets"):
                    self.assertEqual(expected[key], current[key], key)
                self.assertTrue(current["provider_semantics"]["retrieval_evidence"]["hits"])
                self.assertTrue(current["knowledge_targets"])
                self.assertEqual(current["after_5_actions"]["belief_snapshots"]["rows"], 0)
                self.assertEqual(current["after_consumption"]["runtime_inputs"]["rows"], 0)
        # An actual hit and one candidate stay fixed while same-cell noise grows.
        sizes = [row["before"]["prediction_contexts"]["payload_bytes"] +
                 row["before"]["runtime_inputs"]["payload_bytes"] for row in actual]
        self.assertLess(max(sizes) - min(sizes), 1000)
