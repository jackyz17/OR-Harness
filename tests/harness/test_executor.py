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
        # tool_calls / retries / llm_tokens are harness DECLARATIONS: the
        # executor only proves a floor for the sandbox invocation it made.
        self.assertEqual(record.cost.tool_calls, 0.0)
        self.assertEqual(record.cost.retries, 0.0)
        self.assertEqual(record.cost.llm_tokens, 0.0)  # backfilled by harness
        self.assertNotIn("tool_calls", record.cost.measured_dims())
        self.assertNotIn("retries", record.cost.measured_dims())
        self.assertNotIn("llm_tokens", record.cost.measured_dims())
        self.assertEqual(record.execution_features["tool_calls_lower_bound"], 1)
        self.assertGreaterEqual(record.cost.latency_s, 0.0)
        self.assertEqual(record.solver["name"], "mock")
        self.assertEqual(len(record.solver["code_hash"]), 16)

    def test_measured_mask_is_only_what_the_executor_observed(self):
        """The mask means "this value is a real observation". A constant is
        not an observation: tool_calls/retries/llm_tokens must stay out."""
        path = self.write("solve.py", GOOD_SCRIPT)
        record = self.executor.execute(
            path, self.work, solver="mock", task_id="t1",
            strategy_id="S01", profile=self.make_profile())
        self.assertEqual(record.cost.measured_dims(),
                         {"latency_s", "solver_runtime_s"})

    def test_first_failure_is_not_a_retry(self):
        """A first failed attempt carries retries=0, but the executor does not
        MEASURE that — whether this attempt is itself a retry is the harness's
        declaration. The executor only records what it can prove: that it made
        exactly one sandbox invocation."""
        path = self.write("solve.py", "raise RuntimeError('boom')\n")
        record = self.executor.execute(
            path, self.work, solver="mock", task_id="t1",
            strategy_id="S02", profile=self.make_profile())
        self.assertFalse(record.quality["feasible"])
        self.assertEqual(record.cost.retries, 0.0)
        self.assertNotIn("retries", record.cost.measured_dims())
        self.assertEqual(record.execution_features["tool_calls_lower_bound"], 1)
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


class TestCostMeasurementHonesty(ExecutorCase):
    """The cost data must be REAL or explicitly unknown.

    Regression cover for the reported defects: fabricated constants in the
    measured mask, a crashing float() on a malformed report, unvalidated
    runtime values, and a fabricated zero for a script that never ran."""

    def test_non_numeric_runtime_is_rejected_not_crashed(self):
        """float('fast') used to raise ValueError straight out of execute()."""
        path = self.write("solve.py", """
            import json
            with open("result.json", "w") as fh:
                json.dump({"status": "optimal", "objective_value": 1.0,
                           "runtime_seconds": "fast"}, fh)
        """)
        record = self.executor.execute(
            path, self.work, solver="mock", task_id="t1",
            strategy_id="S01", profile=self.make_profile())
        self.assertTrue(record.quality["feasible"])
        self.assertEqual(record.solver_runtime_provenance, "wall_proxy")
        self.assertGreater(record.cost.solver_runtime_s, 0.0)
        notes = record.execution_features["cost_notes"]
        self.assertTrue(any("runtime_seconds" in n for n in notes), notes)

    def test_negative_runtime_is_rejected(self):
        path = self.write("solve.py", """
            import json
            with open("result.json", "w") as fh:
                json.dump({"status": "optimal", "objective_value": 1.0,
                           "runtime_seconds": -5.0}, fh)
        """)
        record = self.executor.execute(
            path, self.work, solver="mock", task_id="t1",
            strategy_id="S01", profile=self.make_profile())
        self.assertEqual(record.solver_runtime_provenance, "wall_proxy")
        self.assertGreaterEqual(record.cost.solver_runtime_s, 0.0)

    def test_non_finite_runtime_is_rejected(self):
        path = self.write("solve.py", """
            import json
            with open("result.json", "w") as fh:
                json.dump({"status": "optimal", "objective_value": 1.0,
                           "runtime_seconds": float("nan")}, fh)
        """)
        record = self.executor.execute(
            path, self.work, solver="mock", task_id="t1",
            strategy_id="S01", profile=self.make_profile())
        self.assertEqual(record.solver_runtime_provenance, "wall_proxy")
        self.assertEqual(record.cost.solver_runtime_s,
                         record.trajectory[0].duration_s)

    def test_runtime_longer_than_the_process_is_flagged(self):
        """A script cannot report more solve time than its own process ran."""
        path = self.write("solve.py", """
            import json
            with open("result.json", "w") as fh:
                json.dump({"status": "optimal", "objective_value": 1.0,
                           "runtime_seconds": 1e9}, fh)
        """)
        record = self.executor.execute(
            path, self.work, solver="mock", task_id="t1",
            strategy_id="S01", profile=self.make_profile())
        checks = record.execution_features["runtime_checks"]
        self.assertTrue(any("wall clock" in c for c in checks), checks)
        # A suspicious measurement must NOT make the solve itself infeasible.
        self.assertTrue(record.quality["feasible"])
        self.assertEqual(record.quality["problems"], [])

    def test_a_rejected_script_measures_nothing(self):
        """A policy-rejected script never ran: 0.0 seconds of solving would
        be a fabricated fact, so nothing enters the measured mask."""
        path = self.write("solve.py", "import subprocess\nsubprocess.run(['ls'])\n")
        record = self.executor.execute(
            path, self.work, solver="mock", task_id="t1",
            strategy_id="S01", profile=self.make_profile())
        self.assertEqual(record.cost.measured_dims(), set())
        self.assertIsNone(record.solver_runtime_provenance)
        self.assertNotIn("tool_calls_lower_bound", record.execution_features)
        notes = record.execution_features["cost_notes"]
        self.assertTrue(any("never ran" in n or "rejected" in n for n in notes),
                        notes)

    def test_runtime_checks_never_enter_quality_problems(self):
        """Runtime sanity is a measurement observation; folding it into
        ``problems`` would flip valid records to infeasible."""
        path = self.write("solve.py", """
            import json
            with open("result.json", "w") as fh:
                json.dump({"status": "optimal", "objective_value": 1.0,
                           "objective_bound": 1.0,
                           "runtime_seconds": 1e9}, fh)
        """)
        outcome = self.executor.run(path, self.work, solver="mock")
        check = SafePythonExecutor.verify(outcome)
        self.assertEqual(check["problems"], [])
        self.assertTrue(check["runtime_checks"])
        self.assertTrue(check["feasible"])

    def test_cpu_bound_and_sleep_bound_both_time_out(self):
        """RLIMIT_CPU above the wall timeout: the wall clock gets the first
        move, so a CPU burn and a stall classify identically."""
        fast = SafePythonExecutor(timeout_seconds=2)
        cpu = self.write("burn.py", """
            import json
            x = 0
            for i in range(10**10):
                x += i
            with open("result.json", "w") as fh:
                json.dump({"status": "optimal"}, fh)
        """)
        stall = self.write("stall.py", """
            import json, time
            time.sleep(60)
            with open("result.json", "w") as fh:
                json.dump({"status": "optimal"}, fh)
        """)
        self.assertEqual(fast.run(cpu, self.work, solver="mock").status, "timeout")
        self.assertEqual(fast.run(stall, self.work, solver="mock").status, "timeout")

    def test_signal_kill_is_explained(self):
        """A bare 'missing result.json' says nothing about why."""
        from or_harness.execution.executor import _exit_note
        self.assertIn("SIGXCPU", _exit_note(-24))
        self.assertIn("SIGKILL", _exit_note(-9))
        self.assertIn("exit code 3", _exit_note(3))
        self.assertEqual(_exit_note(None), "missing result.json")


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
