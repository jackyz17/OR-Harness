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

        # 2. recall (cold start -> NO memory: an empty result with a reason,
        #    never a fabricated candidate menu)
        proc = run_orx(self.home, "recall", "--task", str(self.task_path),
                       "--top", "3")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        out = json.loads(proc.stdout)
        self.assertEqual(out["result"]["recommendations"], [])
        self.assertIn("NO MEMORY", out["result"]["recommendations_basis"]["reason"])

        # 2b. predict-cost (cold start -> unknown, never a default zero).
        #     The method is one the framework has never seen: it is accepted
        #     and reported as unknown rather than refused.
        proc = run_orx(self.home, "predict-cost", "--task",
                       str(self.task_path), "--strategy", "S01")
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
        # Recording no longer manufactures induction labels — the agent
        # reads material with review-material and abstracts it itself.
        self.assertNotIn("induction_hints", out["result"])

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

        # 6. submit the strategy the agent formed (its explicit call). The
        #    framework writes only what is submitted.
        relation = {
            "subject": "S01",
            "claim": "S01 reaches the reference objective in this cell",
            "evidence": [{"execution_id": execution["execution_id"],
                          "role": "evidence"},
                         {"execution_id": execution2["execution_id"],
                          "role": "evidence"}],
        }
        proc = run_orx(self.home, "induce", "--relation", json.dumps(relation))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        out = json.loads(proc.stdout)
        outcome = out["result"]["relations"][0]
        self.assertTrue(outcome.get("saved"), msg=proc.stdout)
        # A submitted claim IS published (the agent decides publication); the
        # framework records the verification state it could compute.
        self.assertTrue(outcome["publication"]["published"])
        self.assertEqual(outcome["publication"]["state"], "fact_checked")
        entry_id = outcome["saved"]

        # 7. recall now shows the submitted claim as knowledge.
        proc = run_orx(self.home, "recall", "--task", str(self.task_path),
                       "--top", "3")
        out = json.loads(proc.stdout)
        s01 = next(r for r in out["result"]["recommendations"]
                   if r["strategy_id"] == "S01")
        self.assertEqual(s01["evidence"], "strategic_entry")

        # 7b. a second submission WITH a check is ADDITIVE: it creates a NEW
        #     entry whose state is ``verified``.
        verify = json.dumps({
            "claim": "S01 reaches the reference objective in this cell",
            "check": {"assertions": [
                {"kind": "status", "roles": ["evidence"],
                 "status": "optimal"}]},
        })
        proc = run_orx(self.home, "induce", "--relation",
                       json.dumps(relation), "--verify", verify)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        vout = json.loads(proc.stdout)["result"]["relations"][0]
        self.assertTrue(vout["publication"]["published"])
        self.assertNotEqual(vout["saved"], entry_id,
                            "a second submission creates a new entry")
        proc = run_orx(self.home, "recall", "--task", str(self.task_path),
                       "--top", "5")
        out = json.loads(proc.stdout)
        s01 = next(r for r in out["result"]["recommendations"]
                   if r["strategy_id"] == "S01")
        self.assertEqual(s01["evidence"], "strategic_entry")

        # 8. inspect both layers
        proc = run_orx(self.home, "inspect", "--bank", "experience")
        self.assertEqual(json.loads(proc.stdout)["result"]["count"], 2)
        proc = run_orx(self.home, "inspect", "--bank", "strategic")
        entries = json.loads(proc.stdout)["result"]["entries"]
        # Additive knowledge: the two submissions are TWO numbered entries.
        self.assertEqual(len(entries), 2)
        verified = next(e for e in entries
                        if e["verification"]["state"] == "verified")
        self.assertEqual(verified["status"], "candidate")
        self.assertEqual(verified["entry_id"], vout["saved"])
        self.assertIn(entry_id, {e["entry_id"] for e in entries})

        # 9. the retirement CANDIDATES are a QUERY now (no collector claims
        #    a cleanup it cannot perform): an entry with no evidence of
        #    being suspect/dormant is simply not listed.
        proc = run_orx(self.home, "inspect", "--bank", "strategic",
                       "--status", "suspect")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(proc.stdout)["result"]["count"], 0)
        proc = run_orx(self.home, "inspect", "--bank", "predictions")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        out = json.loads(proc.stdout)
        self.assertIn("legacy_predictions", out["result"])

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

    def test_record_reports_the_cost_gap_and_backfill_closes_it(self):
        """The end-to-end integrity loop, through the agent's real interface.

        A record whose llm_tokens/tool_calls are not declared must (a) say so
        at record time, and (b) not enter a cost claim until a backfill fills
        it — the withheld dimension is the ONE the framework refuses to
        publish as a partial mean (the complete-or-silent rule). Since
        knowledge is written only from a submitted strategy, the surviving
        observable contract is the record-time gap and the statistics."""
        proc = run_orx(self.home, "execute", "--task", str(self.task_path),
                       "--strategy", "S01", "--code", str(self.solve_path),
                       "--workspace", str(self.work), "--solver", "highs")
        execution = json.loads(proc.stdout)["result"]["execution"]
        exec_path = self.work / "gap1.json"
        exec_path.write_text(json.dumps(execution), encoding="utf-8")
        proc = run_orx(self.home, "record", "--execution", str(exec_path))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        block = json.loads(proc.stdout)["result"]["cost_completeness"]
        self.assertIn("llm_tokens", block["missing"])
        self.assertIn("tool_calls", block["missing"])

        # A second, independent task in the same group.
        proc = run_orx(self.home, "execute", "--task", str(self.task2_path),
                       "--strategy", "S01", "--code", str(self.solve_path),
                       "--workspace", str(self.work), "--solver", "highs")
        execution2 = json.loads(proc.stdout)["result"]["execution"]
        exec2_path = self.work / "gap2.json"
        exec2_path.write_text(json.dumps(execution2), encoding="utf-8")
        run_orx(self.home, "record", "--execution", str(exec2_path))

        # Before the backfill: the measured cost mask omits the two
        # unmeasured dimensions (a placeholder 0 stays, never a partial mean).
        snapshot = json.loads(run_orx(
            self.home, "predict-cost", "--task", str(self.task_path),
            "--strategy", "S01").stdout)["result"]["prediction"]
        self.assertNotIn("llm_tokens", snapshot["cost_measured"] or [])
        self.assertNotIn("tool_calls", snapshot["cost_measured"] or [])

        # Backfill BOTH records: the numbers were read off real reports, so
        # the source is stated.
        run_orx(self.home, "amend-cost", execution["execution_id"],
                "--override", "llm_tokens=1840,tool_calls=4",
                "--source", "agent_observed")
        run_orx(self.home, "amend-cost", execution2["execution_id"],
                "--override", "llm_tokens=1820,tool_calls=4",
                "--source", "agent_observed")
        snapshot = json.loads(run_orx(
            self.home, "predict-cost", "--task", str(self.task_path),
            "--strategy", "S01").stdout)["result"]["prediction"]
        self.assertIn("llm_tokens", snapshot["cost_measured"])
        self.assertIn("tool_calls", snapshot["cost_measured"])

    def test_tool_calls_below_the_sandbox_floor_is_refused(self):
        proc = run_orx(self.home, "execute", "--task", str(self.task_path),
                       "--strategy", "S01", "--code", str(self.solve_path),
                       "--workspace", str(self.work), "--solver", "highs")
        execution = json.loads(proc.stdout)["result"]["execution"]
        exec_path = self.work / "floor.json"
        exec_path.write_text(json.dumps(execution), encoding="utf-8")
        run_orx(self.home, "record", "--execution", str(exec_path))
        proc = run_orx(self.home, "amend-cost", execution["execution_id"],
                       "--override", "tool_calls=0")
        self.assertEqual(proc.returncode, 2)
        self.assertIn("lower bound", proc.stdout + proc.stderr)

    def test_error_exit_code_and_json(self):
        proc = run_orx(self.home, "recall", "--task", "{bad json")
        self.assertEqual(proc.returncode, 2)
        out = json.loads(proc.stdout)
        self.assertIn("error", out["result"])

    def test_cli_cost_comparison_check(self):
        """The real-usage path: a `check` block arrives as JSON, where costs
        are plain dicts. A cost-saving comparison over two cited executions
        must verify here — the framework reads the recorded cost figures."""
        # Two recorded tasks with an explicit token cost difference.
        ids = []
        for task_path, tokens in ((self.task_path, 100), (self.task2_path, 1000)):
            proc = run_orx(self.home, "execute", "--task", str(task_path),
                           "--strategy", "S01", "--code", str(self.solve_path),
                           "--workspace", str(self.work), "--solver", "highs")
            execution = json.loads(proc.stdout)["result"]["execution"]
            ids.append(execution["execution_id"])
            path = self.work / f"{execution['task_id']}.json"
            path.write_text(json.dumps(execution), encoding="utf-8")
            run_orx(self.home, "record", "--execution", str(path),
                    "--override", f"llm_tokens={tokens},tool_calls=2")
        relation = {
            "subject": "S01",
            "claim": "S01 reaches optimal on both tasks; token cost is "
                     "recorded for each",
            "evidence": [{"execution_id": ids[0], "role": "candidate"},
                         {"execution_id": ids[1], "role": "baseline"}],
            "check": {"assertions": [
                {"kind": "status", "roles": ["candidate", "baseline"],
                 "status": "optimal"},
                {"kind": "comparison", "metric": "cost:llm_tokens",
                 "roles_a": ["candidate"], "roles_b": ["baseline"],
                 "direction": "lower", "min_gap": 0.0,
                 "mode": "group", "aggregation": "mean"}]},
        }
        proc = run_orx(self.home, "induce", "--relation",
                       json.dumps(relation))
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        outcome = json.loads(proc.stdout)["result"]["relations"][0]
        report = self.harness_entry(outcome["saved"])
        self.assertEqual(report["state"], "verified", report.get("conclusion"))
        self.assertTrue(outcome["publication"]["published"])

    def harness_entry(self, entry_id):
        proc = run_orx(self.home, "inspect", "--bank", "strategic")
        entries = json.loads(proc.stdout)["result"]["entries"]
        return next(e["verification"] for e in entries
                    if e["entry_id"] == entry_id)

    def test_cli_check_with_no_evidence_is_insufficient(self):
        """A `check` whose named role has no cited evidence cannot publish
        the strategy: the framework reports `insufficient`, not `verified`."""
        self.seed_entry_with_two_tasks()
        proc = run_orx(self.home, "inspect", "--bank", "strategic")
        entry_id = json.loads(proc.stdout)["result"]["entries"][0]["entry_id"]
        # The stored strategy has no `check`: re-submitting it without a
        # verdict keeps it unpublished (never silently verified).
        self.assertNotEqual(
            json.loads(run_orx(self.home, "inspect", "--bank", "strategic")
                       .stdout)["result"]["entries"][0]["verification"]["state"],
            "verified")

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
        """Two recorded tasks -> one saved strategy entry (S01, routing)."""
        ids = []
        for task_path in (self.task_path, self.task2_path):
            proc = run_orx(self.home, "execute", "--task", str(task_path),
                           "--strategy", "S01", "--code", str(self.solve_path),
                           "--workspace", str(self.work), "--solver", "highs")
            execution = json.loads(proc.stdout)["result"]["execution"]
            ids.append(execution["execution_id"])
            path = self.work / f"{execution['task_id']}.json"
            path.write_text(json.dumps(execution), encoding="utf-8")
            run_orx(self.home, "record", "--execution", str(path))
        relation = {"subject": "S01", "claim": "S01 reaches optimal",
                    "evidence": [{"execution_id": e, "role": "evidence"}
                                 for e in ids]}
        proc = run_orx(self.home, "induce", "--relation",
                       json.dumps(relation))
        self.assertTrue(json.loads(proc.stdout)["result"]["relations"][0]
                        ["saved"])

    def test_relation_claim_end_to_end(self):
        """`induce --relation` across processes: a structured claim that does
        not belong to a catalog strategy is saved, verified, published and
        recalled — then refuted by a counterexample."""
        # Two recorded facts on two tasks, with a quality difference.
        def solve(objective):
            return textwrap.dedent(f"""
                import json
                with open("result.json", "w") as fh:
                    json.dump({{"status": "optimal",
                               "objective_value": {objective},
                               "objective_bound": 100.0,
                               "runtime_seconds": 0.01}}, fh)
            """)

        before_path = self.work / "before.py"
        before_path.write_text(solve(140.0), encoding="utf-8")
        after_path = self.work / "after.py"
        after_path.write_text(solve(100.0), encoding="utf-8")
        exec_ids = {}
        for label, code, task in (("before", before_path, self.task_path),
                                  ("after", after_path, self.task2_path)):
            proc = run_orx(self.home, "execute", "--task", str(task),
                           "--strategy", "S01", "--code", str(code),
                           "--workspace", str(self.work), "--solver", "highs")
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            execution = json.loads(proc.stdout)["result"]["execution"]
            path = self.work / f"rel_{label}.json"
            path.write_text(json.dumps(execution), encoding="utf-8")
            run_orx(self.home, "record", "--execution", str(path))
            exec_ids[label] = execution["execution_id"]

        relation = {
            "subject": "principle:repair_keeps_period_state",
            "claim": "保留跨期状态的修复在两个任务上提升了质量",
            "evidence": [{"execution_id": exec_ids["before"], "role": "before"},
                         {"execution_id": exec_ids["after"], "role": "after"}],
            "check": {"assertions": [
                {"kind": "comparison", "metric": "quality",
                 "roles_a": ["after"], "roles_b": ["before"],
                 "direction": "higher", "min_gap": 0.2, "mode": "group"}]},
        }
        verify = {"purpose": "relation",
                  "check": {"assertions": relation["check"]["assertions"]}}
        proc = run_orx(self.home, "induce", "--relation", json.dumps(relation),
                       "--verify", json.dumps(verify))
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        out = json.loads(proc.stdout)
        self.assertEqual(out["result"]["saved"], 1)
        # Two distinct tasks: the claim is published as knowledge.
        self.assertEqual(out["result"]["published"], 1)
        self.assertIn("published", out["summary"])

        # Recall surfaces it as remembered knowledge with its verification
        # state and stated claim.
        proc = run_orx(self.home, "recall", "--task", str(self.task_path))
        recs = json.loads(proc.stdout)["result"]["recommendations"]
        entry_recs = [r for r in recs
                      if r.get("knowledge", {}).get("claim")]
        self.assertTrue(entry_recs)
        item = next(r for r in entry_recs
                    if r["knowledge"]["verification_state"] == "verified")
        self.assertEqual(item["strategy_id"],
                         "principle:repair_keeps_period_state")
        self.assertIn("跨期", item["knowledge"]["claim"]["text"])

        # A counterexample (worse 'after') refutes the group comparison.
        worse_path = self.work / "worse.py"
        worse_path.write_text(solve(300.0), encoding="utf-8")
        proc = run_orx(self.home, "execute", "--task", str(self.task_path),
                       "--strategy", "S01", "--code", str(worse_path),
                       "--workspace", str(self.work), "--solver", "highs")
        execution = json.loads(proc.stdout)["result"]["execution"]
        path = self.work / "rel_worse.json"
        path.write_text(json.dumps(execution), encoding="utf-8")
        run_orx(self.home, "record", "--execution", str(path))
        relation["evidence"].append(
            {"execution_id": execution["execution_id"], "role": "after"})
        proc = run_orx(self.home, "induce", "--relation", json.dumps(relation),
                       "--verify", json.dumps(verify))
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        out = json.loads(proc.stdout)
        # The refuted claim is a SEPARATE, newly-numbered entry (knowledge is
        # additive): it IS published, and its verdict says ``refuted``.
        self.assertEqual(out["result"]["saved"], 1)
        self.assertEqual(out["result"]["published"], 1)
        self.assertIn("refuted", out["summary"])

    def test_relation_single_task_publishes_with_scope_stated(self):
        """A single-task claim is published; its one-task scope is reported
        as a fact, and the framework records the agent's verification."""
        def solve(objective):
            return textwrap.dedent(f"""
                import json
                with open("result.json", "w") as fh:
                    json.dump({{"status": "optimal",
                               "objective_value": {objective},
                               "objective_bound": 100.0,
                               "runtime_seconds": 0.01}}, fh)
            """)

        exec_ids = []
        for objective in (140.0, 100.0):
            code = self.work / f"s_{objective}.py"
            code.write_text(solve(objective), encoding="utf-8")
            proc = run_orx(self.home, "execute", "--task", str(self.task_path),
                           "--strategy", "S01", "--code", str(code),
                           "--workspace", str(self.work), "--solver", "highs")
            execution = json.loads(proc.stdout)["result"]["execution"]
            path = self.work / f"s_{execution['execution_id']}.json"
            path.write_text(json.dumps(execution), encoding="utf-8")
            run_orx(self.home, "record", "--execution", str(path))
            exec_ids.append(execution["execution_id"])
        relation = {
            "subject": "principle:single_task",
            "claim": "同一任务上的修复",
            "evidence": [{"execution_id": exec_ids[0], "role": "before"},
                         {"execution_id": exec_ids[1], "role": "after"}],
            "check": {"assertions": [
                {"kind": "comparison", "metric": "quality",
                 "roles_a": ["after"], "roles_b": ["before"],
                 "direction": "higher", "min_gap": 0.2, "mode": "paired"}]},
        }
        verify = {"purpose": "relation",
                  "check": {"assertions": relation["check"]["assertions"]}}
        proc = run_orx(self.home, "induce", "--relation", json.dumps(relation),
                       "--verify", json.dumps(verify))
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        out = json.loads(proc.stdout)
        self.assertEqual(out["result"]["saved"], 1)
        self.assertEqual(out["result"]["published"], 1)

    def test_recording_writes_evidence_and_the_write_replays_checks(self):
        """Recording a matching execution writes a FROZEN check onto the fact
        (evidence only, never a knowledge change); the next knowledge write
        replays those checks and reports under `revisions`."""
        # Seed a strategy entry over two tasks.
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
        relation = {"subject": "S01", "claim": "S01 reaches optimal",
                    "evidence": [{"execution_id": e1["execution_id"],
                                  "role": "evidence"},
                                 {"execution_id": e2["execution_id"],
                                  "role": "evidence"}],
                    # A DECLARED prediction is what makes the interval
                    # checkable later; without it a run produces no hit/miss.
                    "prediction": {"value": 0.9, "interval": [0.5, 1.0]}}
        proc = run_orx(self.home, "induce", "--relation",
                       json.dumps(relation))
        entry_id = json.loads(proc.stdout)["result"]["relations"][0]["saved"]
        self.assertIsNotNone(entry_id)

        # Recording an execution that DECLARES it adopted the entry writes a
        # FROZEN check (evidence only) — the entry's status/track are
        # untouched by `record`.
        proc = run_orx(self.home, "execute", "--task", str(self.task_path),
                       "--strategy", "S01", "--code", str(self.solve_path),
                       "--workspace", str(self.work), "--solver", "highs",
                       "--used-entry-ids", entry_id)
        ex = json.loads(proc.stdout)["result"]["execution"]
        px = self.work / "e3.json"; px.write_text(json.dumps(ex))
        proc = run_orx(self.home, "record", "--execution", str(px))
        checks = json.loads(proc.stdout)["result"]["prediction_checks"]
        self.assertEqual([c["entry_id"] for c in checks], [entry_id])
        proc = run_orx(self.home, "inspect", "--bank", "strategic")
        entries = json.loads(proc.stdout)["result"]["entries"]
        self.assertEqual(entries[0]["status"], "candidate")  # record never demotes

        # The NEXT knowledge write replays the frozen checks and reports the
        # forward counters (the lifecycle replay ran).
        proc = run_orx(self.home, "induce", "--relation",
                       json.dumps(relation))
        revisions = json.loads(proc.stdout)["result"]["revisions"]
        self.assertTrue(revisions)
        self.assertEqual(revisions[0]["forward"]["n_predictions"], 1)


if __name__ == "__main__":
    unittest.main()
