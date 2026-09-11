"""End-to-end CLI tests: the full harness loop across separate process
invocations — profile -> recommend -> execute -> record -> induce -> recommend
again — plus gc and doctor. Each CLI call is a fresh process, exactly how an
outer harness agent uses it."""
import json
import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

from helpers import HarnessTestCase

REPO = Path(__file__).resolve().parents[2]


def run_orx(home, *argv):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO / "src")
    proc = subprocess.run(
        [sys.executable, "-m", "or_harness.cli", "--home", home, *argv],
        capture_output=True, text=True, env=env, cwd=str(REPO))
    return proc


class TestEndToEndCLI(HarnessTestCase):
    def setUp(self):
        super().setUp()
        self.work = Path(self.home) / "ws"
        self.work.mkdir(parents=True, exist_ok=True)
        self.task = {
            "task_id": "t1", "family": "routing",
            "annotations": {"coupling": {"resource_coupling": 0.9,
                                         "temporal_coupling": 0.1,
                                         "route_complexity": 0.85,
                                         "semantic_coupling": 0.8}},
        }
        self.task_path = self.work / "task.json"
        self.task_path.write_text(json.dumps(self.task), encoding="utf-8")
        self.solve_path = self.work / "solve.py"
        self.solve_path.write_text(textwrap.dedent("""
            import json
            with open("result.json", "w") as fh:
                json.dump({"status": "optimal", "objective_value": 100.0,
                           "objective_bound": 100.0, "runtime_seconds": 0.01},
                          fh)
        """), encoding="utf-8")

    def test_full_loop_across_processes(self):
        # 1. profile
        proc = run_orx(self.home, "profile", "--task", str(self.task_path))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        out = json.loads(proc.stdout)
        self.assertEqual(out["result"]["profile"]["source"], "harness_supplied")
        self.assertIn("summary", out)

        # 2. recall (cold start -> no evidence)
        proc = run_orx(self.home, "recall", "--task", str(self.task_path),
                       "--top", "3")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        out = json.loads(proc.stdout)
        recs = out["result"]["recommendations"]
        self.assertTrue(recs)
        self.assertEqual(recs[0]["evidence"], "no_memory")

        # 3. execute
        proc = run_orx(self.home, "execute", "--task", str(self.task_path),
                       "--strategy", "S01", "--code", str(self.solve_path),
                       "--workspace", str(self.work), "--solver", "highs")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        out = json.loads(proc.stdout)
        execution = out["result"]["execution"]
        self.assertTrue(execution["quality"]["feasible"])

        # 4. record with llm_tokens override (harness backfills its own cost)
        exec_path = self.work / "exec.json"
        exec_path.write_text(json.dumps(execution), encoding="utf-8")
        proc = run_orx(self.home, "record", "--execution", str(exec_path),
                       "--override", "llm_tokens=1840")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        out = json.loads(proc.stdout)
        self.assertTrue(out["result"]["recorded"])
        self.assertIn("induction_hints", out["result"])

        # 5. record a second execution in the same group to enable induction
        proc = run_orx(self.home, "execute", "--task", str(self.task_path),
                       "--strategy", "S01", "--code", str(self.solve_path),
                       "--workspace", str(self.work), "--solver", "highs")
        execution2 = json.loads(proc.stdout)["result"]["execution"]
        exec2_path = self.work / "exec2.json"
        exec2_path.write_text(json.dumps(execution2), encoding="utf-8")
        proc = run_orx(self.home, "record", "--execution", str(exec2_path),
                       "--override", "llm_tokens=1820")
        self.assertEqual(proc.returncode, 0, proc.stderr)

        # 6. induce (harness's explicit call)
        proc = run_orx(self.home, "induce", "--strategy", "S01")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        out = json.loads(proc.stdout)
        results = out["result"]["results"]
        self.assertTrue(any(r.get("created") or r.get("updated") for r in results),
                        msg=proc.stdout)

        # 7. recall again — now entry-backed
        proc = run_orx(self.home, "recall", "--task", str(self.task_path),
                       "--top", "3")
        out = json.loads(proc.stdout)
        s01 = next(r for r in out["result"]["recommendations"]
                   if r["strategy_id"] == "S01")
        self.assertEqual(s01["evidence"], "strategic_entry")

        # 8. inspect both layers
        proc = run_orx(self.home, "inspect", "--bank", "experience")
        self.assertEqual(json.loads(proc.stdout)["result"]["count"], 2)
        proc = run_orx(self.home, "inspect", "--bank", "strategic")
        entries = json.loads(proc.stdout)["result"]["entries"]
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["status"], "candidate")

        # 9. gc dry-run (nothing to compact yet, but the plan is empty cleanly)
        proc = run_orx(self.home, "gc", "--dry-run")
        self.assertEqual(proc.returncode, 0, proc.stderr)

        # 10. doctor
        proc = run_orx(self.home, "doctor")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        out = json.loads(proc.stdout)
        self.assertEqual(len(out["result"]["solvers"]), 7)
        self.assertEqual(out["result"]["memory"]["executions"], 2)

    def test_stdout_is_single_json(self):
        proc = run_orx(self.home, "recall", "--task", str(self.task_path))
        lines = [l for l in proc.stdout.splitlines() if l.strip()]
        self.assertEqual(len(lines), 1)
        parsed = json.loads(lines[0])
        self.assertIn("result", parsed)
        self.assertIn("summary", parsed)

    def test_error_exit_code_and_json(self):
        proc = run_orx(self.home, "recall", "--task", "{bad json")
        self.assertEqual(proc.returncode, 2)
        out = json.loads(proc.stdout)
        self.assertIn("error", out["result"])

    def test_prediction_check_demotes_after_misses(self):
        # Seed an entry, then record three executions far outside its interval.
        proc = run_orx(self.home, "execute", "--task", str(self.task_path),
                       "--strategy", "S01", "--code", str(self.solve_path),
                       "--workspace", str(self.work), "--solver", "highs")
        e1 = json.loads(proc.stdout)["result"]["execution"]
        p = self.work / "e1.json"; p.write_text(json.dumps(e1))
        run_orx(self.home, "record", "--execution", str(p))
        proc = run_orx(self.home, "execute", "--task", str(self.task_path),
                       "--strategy", "S01", "--code", str(self.solve_path),
                       "--workspace", str(self.work), "--solver", "highs")
        e2 = json.loads(proc.stdout)["result"]["execution"]
        p2 = self.work / "e2.json"; p2.write_text(json.dumps(e2))
        run_orx(self.home, "record", "--execution", str(p2))
        proc = run_orx(self.home, "induce", "--strategy", "S01")
        created = json.loads(proc.stdout)["result"]["results"][0]["created"]
        self.assertIsNotNone(created)

        # Three terrible executions (objective 3x the bound -> quality ~0.33)
        # fall far below the entry's [0.5, 1.0] interval -> 3 misses -> suspect.
        bad_solve = self.work / "bad_solve.py"
        bad_solve.write_text(textwrap.dedent("""
            import json
            with open("result.json", "w") as fh:
                json.dump({"status": "feasible", "objective_value": 300.0,
                           "objective_bound": 100.0, "runtime_seconds": 0.01},
                          fh)
        """), encoding="utf-8")
        for i in range(3):
            proc = run_orx(self.home, "execute", "--task", str(self.task_path),
                           "--strategy", "S01", "--code", str(bad_solve),
                           "--workspace", str(self.work), "--solver", "highs")
            ex = json.loads(proc.stdout)["result"]["execution"]
            px = self.work / f"bad{i}.json"; px.write_text(json.dumps(ex))
            proc = run_orx(self.home, "record", "--execution", str(px))
        proc = run_orx(self.home, "inspect", "--bank", "strategic")
        entries = json.loads(proc.stdout)["result"]["entries"]
        self.assertEqual(entries[0]["status"], "suspect")
        self.assertEqual(entries[0]["prediction_track"]["consecutive_misses"], 3)


if __name__ == "__main__":
    unittest.main()
