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
        # A second task in the same family: a claim needs independent evidence.
        self.task2 = dict(self.task, task_id="t2")
        self.task2_path = self.work / "task2.json"
        self.task2_path.write_text(json.dumps(self.task2), encoding="utf-8")
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

        # 2b. predict (cold start -> unknown, never a default zero)
        proc = run_orx(self.home, "predict", "--task", str(self.task_path),
                       "--strategy", "S01")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        out = json.loads(proc.stdout)
        prediction = out["result"]["prediction"]
        self.assertEqual(prediction["source"], "unknown")
        self.assertIsNone(prediction["expected_cost"])

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

        # 5. record a second execution on a DIFFERENT task in the same group:
        #    repetition of one task is not independent evidence.
        proc = run_orx(self.home, "execute", "--task", str(self.task2_path),
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
        # The candidate is formed but NOT published: no admission verdict was
        # supplied, so recall must keep answering from the statistics.
        self.assertIn("not published", results[0]["skipped"])
        entry_id = results[0].get("created") or results[0].get("updated")

        # 7. recall again — still statistics: publishing needs a verified claim
        proc = run_orx(self.home, "recall", "--task", str(self.task_path),
                       "--top", "3")
        out = json.loads(proc.stdout)
        s01 = next(r for r in out["result"]["recommendations"]
                   if r["strategy_id"] == "S01")
        self.assertEqual(s01["evidence"], "conditional_stats")

        # 7b. re-induce WITH an admission check -> published knowledge
        verify = json.dumps({
            "purpose": "rule",
            "claim": "S01 reaches the reference objective in this cell",
            "check": {"reference_objective": 100.0},
            "executions": [execution],
            "supporting": [execution2],
        })
        proc = run_orx(self.home, "induce", "--strategy", "S01",
                       "--verify", verify)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        verified = json.loads(proc.stdout)["result"]["results"][0]
        self.assertEqual(verified["verification"]["state"], "verified")
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
        self.assertEqual(entries[0]["entry_id"], entry_id)
        self.assertEqual(entries[0]["verification"]["state"], "verified")

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

    def test_cli_cost_saving_verification(self):
        """The real-usage path: --verify arrives as JSON, where costs are
        plain dicts. The same evidence verified through the Python API must
        verify here too (it used to report "the cost dimension could not be
        read" because only object attribute access was implemented)."""
        self.seed_entry_with_two_tasks()
        verify = json.dumps({
            "purpose": "cost_saving",
            "claim": "S01 reaches the same quality on t1 for far fewer tokens",
            "check": {"dimension": "llm_tokens", "quality_floor": 0.9},
            "executions": [{
                "execution_id": "ex_candidate", "task_id": "t1",
                "strategy_id": "S01", "family": "routing",
                "measurement_scope": "attempt",
                "cost": {"llm_tokens": 100}, "cost_measured": ["llm_tokens"],
                "quality": {"feasible": True, "objective": 100.0,
                            "status": "optimal"},
            }],
            "supporting": [{
                "execution_id": "ex_baseline", "task_id": "t1",
                "strategy_id": "S01", "family": "routing",
                "measurement_scope": "attempt",
                "cost": {"llm_tokens": 1000}, "cost_measured": ["llm_tokens"],
                "quality": {"feasible": True, "objective": 100.0,
                            "status": "optimal"},
            }],
        })
        proc = run_orx(self.home, "induce", "--strategy", "S01",
                       "--verify", verify)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        report = json.loads(proc.stdout)["result"]["results"][0]["verification"]
        self.assertEqual(report["state"], "verified", report["conclusion"])
        self.assertIn("1000", report["conclusion"])

    def test_cli_verify_rejects_evidence_for_another_strategy(self):
        """A payload for a different strategy must not publish this entry."""
        self.seed_entry_with_two_tasks()
        verify = json.dumps({
            "purpose": "rule", "claim": "c",
            "check": {"reference_objective": 100.0},
            "executions": [{"execution_id": "ex_s04", "task_id": "other",
                            "strategy_id": "S04", "family": "scheduling",
                            "quality": {"feasible": True, "objective": 100.0,
                                        "status": "optimal"}}],
        })
        proc = run_orx(self.home, "induce", "--strategy", "S01",
                       "--verify", verify)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        report = json.loads(proc.stdout)["result"]["results"][0]["verification"]
        self.assertNotEqual(report["state"], "verified")
        self.assertIn("does not correspond", report["conclusion"])

    def test_inspect_covers_all_three_layers(self):
        """Every --bank value must answer (the archive branch silently broke
        once: the api echoed a different bank name than the CLI switched on)."""
        self.seed_entry_with_two_tasks()
        for bank, key in (("experience", "records"), ("strategic", "entries"),
                          ("archive", "cards")):
            proc = run_orx(self.home, "inspect", "--bank", bank)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            out = json.loads(proc.stdout)["result"]
            self.assertEqual(out["bank"], bank)
            self.assertIn(key, out)
        # Retire -> the entry leaves the hot store and a card appears.
        entry_id = json.loads(run_orx(self.home, "inspect", "--bank",
                                      "strategic").stdout)["result"]["entries"][0]["entry_id"]
        proc = run_orx(self.home, "retire", "--entry", entry_id,
                       "--reason", "probe: retire path")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        hot = json.loads(run_orx(self.home, "inspect", "--bank",
                                 "strategic").stdout)["result"]
        self.assertEqual(hot["count"], 0)
        cold = json.loads(run_orx(self.home, "inspect", "--bank",
                                  "archive").stdout)["result"]
        self.assertEqual(cold["count"], 1)
        self.assertEqual(cold["cards"][0]["reason"], "probe: retire path")
        self.assertEqual(cold["cards"][0]["strategy_id"], "S01")

    def seed_entry_with_two_tasks(self):
        """Two recorded tasks -> one admissible claim (S01, routing)."""
        for task_path in (self.task_path, self.task2_path):
            proc = run_orx(self.home, "execute", "--task", str(task_path),
                           "--strategy", "S01", "--code", str(self.solve_path),
                           "--workspace", str(self.work), "--solver", "highs")
            execution = json.loads(proc.stdout)["result"]["execution"]
            path = self.work / f"{execution['task_id']}.json"
            path.write_text(json.dumps(execution), encoding="utf-8")
            run_orx(self.home, "record", "--execution", str(path))
        proc = run_orx(self.home, "induce", "--strategy", "S01")
        self.assertTrue(json.loads(proc.stdout)["result"]["results"][0]["created"])

    def test_quality_misses_demote_at_next_induce(self):
        # Seed an entry, then record three executions far outside its interval.
        proc = run_orx(self.home, "execute", "--task", str(self.task_path),
                       "--strategy", "S01", "--code", str(self.solve_path),
                       "--workspace", str(self.work), "--solver", "highs")
        e1 = json.loads(proc.stdout)["result"]["execution"]
        p = self.work / "e1.json"; p.write_text(json.dumps(e1))
        run_orx(self.home, "record", "--execution", str(p))
        proc = run_orx(self.home, "execute", "--task", str(self.task2_path),
                       "--strategy", "S01", "--code", str(self.solve_path),
                       "--workspace", str(self.work), "--solver", "highs")
        e2 = json.loads(proc.stdout)["result"]["execution"]
        p2 = self.work / "e2.json"; p2.write_text(json.dumps(e2))
        run_orx(self.home, "record", "--execution", str(p2))
        proc = run_orx(self.home, "induce", "--strategy", "S01")
        created = json.loads(proc.stdout)["result"]["results"][0]["created"]
        self.assertIsNotNone(created)

        # Three terrible executions (objective 3x the bound -> quality ~0.33)
        # fall far below the entry's [0.5, 1.0] interval. Recording them keeps
        # the entry untouched (evidence only); the NEXT induce demotes it.
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
            checks = json.loads(proc.stdout)["result"]["prediction_checks"]
            self.assertEqual([c["hit"] for c in checks], [False])
        proc = run_orx(self.home, "inspect", "--bank", "strategic")
        entries = json.loads(proc.stdout)["result"]["entries"]
        self.assertEqual(entries[0]["status"], "candidate")  # record never demotes
        proc = run_orx(self.home, "induce", "--strategy", "S01")
        revisions = json.loads(proc.stdout)["result"]["revisions"]
        self.assertIn("demoted:->suspect", revisions[0]["transitions"])
        proc = run_orx(self.home, "inspect", "--bank", "strategic")
        entries = json.loads(proc.stdout)["result"]["entries"]
        self.assertEqual(entries[0]["status"], "suspect")
        self.assertEqual(entries[0]["prediction_track"]["consecutive_misses"], 3)


if __name__ == "__main__":
    unittest.main()
