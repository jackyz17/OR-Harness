"""Executor sandbox and solver adapter tests."""
import json
import os
import textwrap
import unittest
from pathlib import Path

from helpers import HarnessTestCase

from or_harness.adapters.solver import (
    available_families,
    default_adapters,
    probe_all,
)
from or_harness.execution.executor import SafePythonExecutor


GOOD_SCRIPT = textwrap.dedent("""
    import json
    result = {"status": "optimal", "objective_value": 42.0,
              "objective_bound": 42.0, "runtime_seconds": 0.01,
              "solver": "mock"}
    with open("result.json", "w") as fh:
        json.dump(result, fh)
""")


class ExecutorCase(HarnessTestCase):
    def setUp(self):
        super().setUp()
        self.executor = SafePythonExecutor(timeout_seconds=30)
        self.work = Path(self.home) / "workspace"
        self.work.mkdir(parents=True, exist_ok=True)

    def write(self, name: str, content: str) -> Path:
        path = self.work / name
        path.write_text(textwrap.dedent(content), encoding="utf-8")
        return path


class TestSandboxPolicy(ExecutorCase):
    def assertBlocked(self, source: str):
        self.assertIsNotNone(SafePythonExecutor.validate_source(source))

    def assertAllowed(self, source: str):
        self.assertIsNone(SafePythonExecutor.validate_source(source))

    def test_good_script_allowed(self):
        self.assertAllowed(GOOD_SCRIPT)

    def test_network_and_shell_blocked(self):
        self.assertBlocked("import subprocess\nsubprocess.run(['ls'])")
        self.assertBlocked("import socket")
        self.assertBlocked("from urllib import request")
        self.assertBlocked("import requests")
        self.assertBlocked("import shutil")

    def test_dangerous_os_blocked(self):
        self.assertBlocked("import os\nos.system('ls')")
        self.assertBlocked("from os import remove")
        self.assertBlocked("import os\nos.listdir('.')")

    def test_pathlib_blocked(self):
        self.assertBlocked("from pathlib import Path")

    def test_open_restricted_to_result_json(self):
        self.assertBlocked("with open('/etc/passwd') as f: pass")
        self.assertBlocked("with open('../escape.json', 'w') as f: pass")
        self.assertBlocked("name = 'result.json'\nwith open(name, 'w') as f: pass")
        self.assertAllowed("with open('result.json', 'w') as f: f.write('{}')")

    def test_syntax_error_reported(self):
        msg = SafePythonExecutor.validate_source("def broken(:\n")
        self.assertIn("SyntaxError", msg)


class TestExecution(ExecutorCase):
    def test_run_success(self):
        path = self.write("solve.py", GOOD_SCRIPT)
        outcome = self.executor.run(path, self.work, solver="mock")
        self.assertEqual(outcome.status, "optimal")
        self.assertEqual(outcome.objective_value, 42.0)
        self.assertTrue((self.work / "result.json").exists())

    def test_run_failure_normalized(self):
        path = self.write("solve.py", "raise ValueError('model is infeasible')\n")
        outcome = self.executor.run(path, self.work, solver="mock")
        self.assertEqual(outcome.status, "error")
        self.assertIn("ValueError", outcome.normalized_error)

    def test_security_rejection_skips_execution(self):
        path = self.write("solve.py", "import socket\n")
        outcome = self.executor.run(path, self.work, solver="mock")
        self.assertEqual(outcome.status, "error")
        self.assertIn("security policy", outcome.normalized_error)
        self.assertFalse((self.work / "result.json").exists())

    def test_execute_builds_record_with_cost(self):
        path = self.write("solve.py", GOOD_SCRIPT)
        record = self.executor.execute(
            path, self.work, solver="mock", task_id="t1",
            strategy_id="S01", profile=self.make_profile())
        self.assertTrue(record.quality["feasible"])
        self.assertEqual(record.quality["objective"], 42.0)
        self.assertEqual(record.quality["gap"], 0.0)
        self.assertEqual(record.cost.tool_calls, 1.0)
        self.assertEqual(record.cost.retries, 0.0)
        self.assertEqual(record.cost.llm_tokens, 0.0)  # backfilled by harness
        self.assertGreaterEqual(record.cost.latency_s, 0.0)
        self.assertEqual(record.solver["name"], "mock")
        self.assertEqual(len(record.solver["code_hash"]), 16)

    def test_first_failure_is_not_a_retry(self):
        """A first failed attempt is retries=0 (an observed zero): the
        executor never infers a retry from failure status. The retry
        relationship is the harness's declaration (override retries=...)."""
        path = self.write("solve.py", "raise RuntimeError('boom')\n")
        record = self.executor.execute(
            path, self.work, solver="mock", task_id="t1",
            strategy_id="S02", profile=self.make_profile())
        self.assertFalse(record.quality["feasible"])
        self.assertEqual(record.cost.retries, 0.0)
        self.assertIn("retries", record.cost.measured_dims())
        self.assertEqual(len(record.failures), 1)
        self.assertIn("RuntimeError", record.failures[0].error)
        # No result.json: wall-clock proxy is used, and it is explicit.
        self.assertEqual(record.solver_runtime_provenance, "wall_proxy")
        self.assertEqual(record.measurement_scope, "attempt")
        self.assertIn("solver_runtime_s", record.cost.measured_dims())

    def test_reported_runtime_marked_as_reported(self):
        path = self.write("solve.py", GOOD_SCRIPT)  # reports runtime_seconds
        record = self.executor.execute(
            path, self.work, solver="mock", task_id="t1",
            strategy_id="S02", profile=self.make_profile())
        self.assertEqual(record.solver_runtime_provenance, "reported")
        self.assertEqual(record.cost.solver_runtime_s, 0.01)

    def test_verify_catches_illegal_status(self):
        path = self.write("solve.py", """
            import json
            with open("result.json", "w") as fh:
                json.dump({"status": "banana", "objective_value": 1.0}, fh)
        """)
        record = self.executor.execute(
            path, self.work, solver="mock", task_id="t1",
            strategy_id="S01", profile=self.make_profile())
        self.assertFalse(record.quality["feasible"])
        self.assertTrue(record.quality["problems"])

    def test_verify_derives_gap_from_bound(self):
        path = self.write("solve.py", """
            import json
            with open("result.json", "w") as fh:
                json.dump({"status": "feasible", "objective_value": 110.0,
                           "objective_bound": 100.0}, fh)
        """)
        record = self.executor.execute(
            path, self.work, solver="mock", task_id="t1",
            strategy_id="S01", profile=self.make_profile())
        self.assertAlmostEqual(record.quality["gap"], 10.0 / 110.0, places=4)


class TestAdapters(HarnessTestCase):
    def test_seven_adapters_with_families(self):
        adapters = default_adapters()
        self.assertEqual(len(adapters), 7)
        families = {a.solver_family for a in adapters}
        self.assertIn("milp", families)
        self.assertIn("cp_sat", families)

    def test_probe_all_reports(self):
        reports = probe_all()
        self.assertEqual(len(reports), 7)
        for report in reports:
            self.assertIsInstance(report.available, bool)
            self.assertTrue(report.message)

    def test_available_families_mapping(self):
        families = available_families()
        # Every reported family maps to at least one concrete solver.
        for fam, names in families.items():
            self.assertTrue(names)


if __name__ == "__main__":
    unittest.main()
