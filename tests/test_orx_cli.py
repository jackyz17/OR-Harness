"""End-to-end tests for the orx CLI (ReAct-oriented, file-based runs).

Simulates the harness agent's ReAct loop: each test writes the artifact files
an agent would author, then invokes one CLI command per step, in separate
processes (subprocess), verifying:
  - state survives across processes (files, not memory),
  - stamps enforce the chain (skip / stale rejected),
  - retry of a failed step does not restart the chain,
  - the gold gate blocks appends on mismatch,
  - utility attribution closes the loop on gold match.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ORX = [sys.executable, str(REPO / "scripts" / "orx.py")]

PROBLEM = """A farmer raises cows and chickens.
Each cow yields 40 profit, each chicken 10 profit.
The farmer has at most 200 units of feed; a cow needs 8 units, a chicken 2.
At most 20 chickens may be raised.
How many of each should be raised to maximize profit?"""

MODEL = """<think>
Classic resource-allocation LP. Two competing decisions share a feed budget.
[uses E1]
</think>
<model>
SETS:
  i in Animals = {cow, chicken}
PARAMETERS:
  profit[i]
  feed_need[i]
  feed_limit
  max_chickens
VARIABLES:
  x[i] integer >= 0
OBJECTIVE:
  maximize sum(i, profit[i] * x[i])
CONSTRAINTS:
  C1: sum(i, feed_need[i] * x[i]) <= feed_limit
  C2: x[chicken] <= max_chickens
</model>"""

SIGNATURE = {
    "objective": "linear",
    "decision": ["integer_batch"],
    "constraint": ["capacity"],
    "interaction": "shared_resource_coupled",
    "features": {"resource": "feed"},
}

# Pure-stdlib solver stub: writes a valid result.json without any solver package.
SOLVE_CODE = """
import json
result = {{
    "status": "optimal",
    "solver": "{solver}",
    "objective_sense": "maximize",
    "objective_value": {objective},
    "objective_bound": {objective},
    "mip_gap": 0.0,
    "runtime_seconds": 0.01,
    "variables": {{"x[cow]": 20, "x[chicken]": 20}},
    "diagnostics": {{}},
    "message": "stub solve",
}}
with open("result.json", "w") as handle:
    json.dump(result, handle)
"""


def run_orx(args, cwd, bank_home):
    env = dict(os.environ)
    env["OR_EXPERIENCE_BANK_HOME"] = str(bank_home)
    proc = subprocess.run(
        ORX + args, cwd=str(cwd), env=env, capture_output=True, text=True, timeout=120
    )
    try:
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        payload = {"_raw_stdout": proc.stdout, "_raw_stderr": proc.stderr}
    return proc.returncode, payload


class OrxCLITestBase(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.tmp = Path(tempfile.mkdtemp(prefix="orx_test_"))
        self.bank = self.tmp / "bank"
        self.run_dir = self.tmp / "run"
        self.run_dir.mkdir(parents=True)
        # init the bank
        code, out = run_orx(["init", "--bank-home", str(self.bank)], self.tmp, self.bank)
        assert code == 0, out

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)

    def orx(self, *args):
        return run_orx(list(args), self.run_dir, self.bank)


class TestSolveChain(OrxCLITestBase):
    def test_full_chain_happy_path(self):
        # recall
        (self.run_dir / "problem.txt").write_text(PROBLEM, encoding="utf-8")
        code, out = self.orx("recall", "--problem-file", "problem.txt")
        self.assertEqual(code, 0, out)
        self.assertEqual(out["priors_count"], 0)  # empty bank is fine
        self.assertTrue((self.run_dir / "priors.json").exists())

        # validate (first attempt fails L2: references an undeclared symbol)
        (self.run_dir / "model.txt").write_text(
            MODEL.replace("C2: x[chicken] <= max_chickens", "C2: x[chicken] <= chicken_cap"),
            encoding="utf-8",
        )
        code, out = self.orx("validate")
        self.assertEqual(code, 0, out)
        self.assertFalse(out["passed"])
        self.assertTrue(any(i["type"] == "undefined_symbol" for i in out["issues"]), out)

        # fix and retry — no penalty, no restart
        (self.run_dir / "model.txt").write_text(MODEL, encoding="utf-8")
        code, out = self.orx("validate")
        self.assertEqual(code, 0, out)
        self.assertTrue(out["passed"], out)
        self.assertTrue((self.run_dir / "stamps" / "model.json").exists())

        # signature
        (self.run_dir / "signature.json").write_text(json.dumps(SIGNATURE), encoding="utf-8")
        code, out = self.orx("signature")
        self.assertEqual(code, 0, out)
        self.assertTrue(out["passed"], out)

        # hints before codegen
        code, out = self.orx("hints", "--solver", "highs")
        self.assertEqual(code, 0, out)
        self.assertTrue((self.run_dir / "branches" / "highs" / "hints.json").exists())

        # solve branch 1
        br = self.run_dir / "branches" / "highs"
        br.mkdir(parents=True, exist_ok=True)
        (br / "solve.py").write_text(SOLVE_CODE.format(solver="highs", objective=900), encoding="utf-8")
        code, out = self.orx("solve", "--solver", "highs")
        self.assertEqual(code, 0, out)
        self.assertTrue(out["valid"], out)
        self.assertEqual(out["objective_value"], 900)

        # solve branch 2 (different solver, same objective)
        br2 = self.run_dir / "branches" / "pulp"
        br2.mkdir(parents=True, exist_ok=True)
        (br2 / "solve.py").write_text(SOLVE_CODE.format(solver="pulp", objective=900), encoding="utf-8")
        code, out = self.orx("solve", "--solver", "pulp")
        self.assertEqual(code, 0, out)
        self.assertTrue(out["valid"], out)

        # cross-validate
        code, out = self.orx("cross-validate")
        self.assertEqual(code, 0, out)
        self.assertTrue(out["consistent"], out)
        self.assertEqual(out["best_objective"], 900)

        # gold (user-provided, matches)
        code, out = self.orx("gold", "--answer", "900")
        self.assertEqual(code, 0, out)
        self.assertTrue(out["gold_matched"], out)

        # append a modeling experience
        exp = {
            "layer": "modeling",
            "title": "Shared-feed budget: single capacity constraint over both decisions",
            "polarity": "positive",
            "retrieval_text": "When two decisions consume one shared budget, model one capacity constraint summing per-unit usage.",
            "modeling_aspect": "constraint",
            "action": "sum(i, usage[i] * x[i]) <= budget",
            "rationale": "Aggregated capacity is the minimal correct form.",
        }
        exp_file = self.run_dir / "exp_modeling.json"
        exp_file.write_text(json.dumps(exp), encoding="utf-8")
        code, out = self.orx("append", "--file", str(exp_file))
        self.assertEqual(code, 0, out)
        self.assertEqual(out["status"], "appended", out)

        # episode (terminal)
        code, out = self.orx("episode")
        self.assertEqual(code, 0, out)
        self.assertTrue(out["recorded"], out)
        self.assertEqual(out["produced_realizations"], 1)
        self.assertEqual(out["status"], "SOLVE_FLOW_COMPLETE")

        # status reports complete
        code, out = self.orx("status")
        self.assertEqual(out["phase"], "complete")

    def test_skip_prevention_and_stale_stamp(self):
        # no recall -> validate fails (run dir not initialized)
        code, out = self.orx("validate")
        self.assertEqual(code, 2)
        self.assertIn("not a run directory", out["error"])

        # signature before validate -> chain error (run dir exists but no stamp)
        (self.run_dir / "problem.txt").write_text(PROBLEM, encoding="utf-8")
        self.orx("recall", "--problem-file", "problem.txt")
        (self.run_dir / "signature.json").write_text(json.dumps(SIGNATURE), encoding="utf-8")
        code, out = self.orx("signature")
        self.assertEqual(code, 2)
        self.assertIn("stamp", out["error"])

        # solve before signature -> chain error
        code, out = self.orx("solve", "--solver", "highs")
        self.assertEqual(code, 2)

        # full setup, then edit model after validation -> stale stamp
        (self.run_dir / "model.txt").write_text(MODEL, encoding="utf-8")
        code, out = self.orx("validate")
        self.assertTrue(out["passed"])
        (self.run_dir / "model.txt").write_text(MODEL + "\n# edited", encoding="utf-8")
        code, out = self.orx("signature")
        self.assertEqual(code, 2)
        self.assertIn("stale", out["error"])

    def test_failed_branch_retry_without_chain_restart(self):
        (self.run_dir / "problem.txt").write_text(PROBLEM, encoding="utf-8")
        self.orx("recall", "--problem-file", "problem.txt")
        (self.run_dir / "model.txt").write_text(MODEL, encoding="utf-8")
        self.orx("validate")
        (self.run_dir / "signature.json").write_text(json.dumps(SIGNATURE), encoding="utf-8")
        self.orx("signature")

        br = self.run_dir / "branches" / "highs"
        br.mkdir(parents=True, exist_ok=True)
        # failing code (crashes -> status=error, no valid objective)
        (br / "solve.py").write_text("raise RuntimeError('boom')\n", encoding="utf-8")
        code, out = self.orx("solve", "--solver", "highs")
        self.assertEqual(code, 0, out)
        self.assertEqual(out["status"], "error")
        self.assertTrue(out["normalized_error"])
        # the failed branch is not a valid comparison branch
        record = json.loads((br / "result.json").read_text(encoding="utf-8"))
        self.assertFalse(record["valid"] and record["status"] in {"optimal", "feasible"})

        # fix ONLY the branch code; signature stamp still valid (no re-validate needed)
        (br / "solve.py").write_text(SOLVE_CODE.format(solver="highs", objective=900), encoding="utf-8")
        code, out = self.orx("solve", "--solver", "highs")
        self.assertEqual(code, 0, out)
        self.assertTrue(out["valid"], out)

    def test_gold_gate_blocks_append_on_mismatch(self):
        (self.run_dir / "problem.txt").write_text(PROBLEM, encoding="utf-8")
        self.orx("recall", "--problem-file", "problem.txt")
        (self.run_dir / "model.txt").write_text(MODEL, encoding="utf-8")
        self.orx("validate")
        (self.run_dir / "signature.json").write_text(json.dumps(SIGNATURE), encoding="utf-8")
        self.orx("signature")
        for solver in ("highs", "pulp"):
            br = self.run_dir / "branches" / solver
            br.mkdir(parents=True, exist_ok=True)
            (br / "solve.py").write_text(SOLVE_CODE.format(solver=solver, objective=900), encoding="utf-8")
            self.orx("solve", "--solver", solver)
        self.orx("cross-validate")

        # gold mismatch
        code, out = self.orx("gold", "--answer", "123")
        self.assertEqual(code, 0, out)
        self.assertFalse(out["gold_matched"])

        # append is blocked
        exp_file = self.run_dir / "exp.json"
        exp_file.write_text(json.dumps({"layer": "modeling", "title": "t", "retrieval_text": "r"}), encoding="utf-8")
        code, out = self.orx("append", "--file", str(exp_file))
        self.assertEqual(code, 2)
        self.assertIn("gold", out["error"])

        # new-round archives and allows a fresh modeling attempt
        code, out = self.orx("new-round")
        self.assertEqual(code, 0, out)
        self.assertTrue((self.run_dir / "rounds" / "1").is_dir())
        self.assertTrue((self.run_dir / "problem.txt").exists())
        self.assertFalse((self.run_dir / "model.txt").exists())

    def test_cross_validate_inconsistent_allows_more_branches(self):
        """The old token chain consumed signature_token on cross_validate, making
        'add a third branch to triangulate' impossible. Stamps fix this."""
        (self.run_dir / "problem.txt").write_text(PROBLEM, encoding="utf-8")
        self.orx("recall", "--problem-file", "problem.txt")
        (self.run_dir / "model.txt").write_text(MODEL, encoding="utf-8")
        self.orx("validate")
        (self.run_dir / "signature.json").write_text(json.dumps(SIGNATURE), encoding="utf-8")
        self.orx("signature")
        for solver, obj in (("highs", 900), ("pulp", 850)):
            br = self.run_dir / "branches" / solver
            br.mkdir(parents=True, exist_ok=True)
            (br / "solve.py").write_text(SOLVE_CODE.format(solver=solver, objective=obj), encoding="utf-8")
            self.orx("solve", "--solver", solver)

        code, out = self.orx("cross-validate")
        self.assertFalse(out["consistent"])

        # add a third branch WITHOUT re-validating the model — must work
        br = self.run_dir / "branches" / "scip"
        br.mkdir(parents=True, exist_ok=True)
        (br / "solve.py").write_text(SOLVE_CODE.format(solver="scip", objective=900), encoding="utf-8")
        code, out = self.orx("solve", "--solver", "scip")
        self.assertEqual(code, 0, out)
        self.assertTrue(out["valid"], out)


class TestUtilityLoop(OrxCLITestBase):
    def test_cited_prior_gets_credit_on_gold_match(self):
        # Seed the bank with one modeling experience via a full mini-run.
        seed_run = self.tmp / "seed"
        seed_run.mkdir()
        (seed_run / "problem.txt").write_text(PROBLEM, encoding="utf-8")
        run_orx(["recall", "--problem-file", "problem.txt"], seed_run, self.bank)
        (seed_run / "model.txt").write_text(MODEL, encoding="utf-8")
        run_orx(["validate"], seed_run, self.bank)
        (seed_run / "signature.json").write_text(json.dumps(SIGNATURE), encoding="utf-8")
        run_orx(["signature"], seed_run, self.bank)
        for solver in ("highs", "pulp"):
            br = seed_run / "branches" / solver
            br.mkdir(parents=True, exist_ok=True)
            (br / "solve.py").write_text(SOLVE_CODE.format(solver=solver, objective=900), encoding="utf-8")
            run_orx(["solve", "--solver", solver], seed_run, self.bank)
        run_orx(["cross-validate"], seed_run, self.bank)
        run_orx(["gold", "--answer", "900"], seed_run, self.bank)
        exp = {
            "layer": "modeling",
            "title": "Shared budget needs aggregated capacity constraint",
            "retrieval_text": "Aggregate per-unit usage into one capacity constraint over the shared budget.",
            "modeling_aspect": "constraint",
            "action": "sum(i, usage[i]*x[i]) <= budget",
        }
        exp_file = seed_run / "exp.json"
        exp_file.write_text(json.dumps(exp), encoding="utf-8")
        code, out = run_orx(["append", "--file", str(exp_file)], seed_run, self.bank)
        self.assertEqual(out["status"], "appended", out)
        code, out = run_orx(["episode"], seed_run, self.bank)
        self.assertTrue(out["recorded"], out)
        seeded_id = out["episode_id"]

        # Second run: recall must surface the seeded experience as E1.
        run2 = self.tmp / "run2"
        run2.mkdir()
        (run2 / "problem.txt").write_text(PROBLEM, encoding="utf-8")
        code, out = run_orx(["recall", "--problem-file", "problem.txt"], run2, self.bank)
        self.assertEqual(code, 0, out)
        self.assertEqual(out["priors_count"], 1, out)

        # Model cites [uses E1]; gold matches -> utility credited.
        (run2 / "model.txt").write_text(MODEL, encoding="utf-8")
        run_orx(["validate"], run2, self.bank)
        (run2 / "signature.json").write_text(json.dumps(SIGNATURE), encoding="utf-8")
        run_orx(["signature"], run2, self.bank)
        for solver in ("highs", "pulp"):
            br = run2 / "branches" / solver
            br.mkdir(parents=True, exist_ok=True)
            (br / "solve.py").write_text(SOLVE_CODE.format(solver=solver, objective=900), encoding="utf-8")
            run_orx(["solve", "--solver", solver], run2, self.bank)
        run_orx(["cross-validate"], run2, self.bank)
        run_orx(["gold", "--answer", "900"], run2, self.bank)
        code, out = run_orx(["episode"], run2, self.bank)
        self.assertEqual(code, 0, out)
        self.assertEqual(out["cited_priors"], 1, out)
        self.assertEqual(out["utility_credited"], 1, out)
        self.assertNotEqual(out["episode_id"], seeded_id)


class TestDoctorAndStatus(OrxCLITestBase):
    def test_doctor(self):
        code, out = self.orx("doctor")
        self.assertEqual(code, 0, out)
        self.assertIn("checks", out)
        self.assertTrue(out["checks"]["bank"]["ok"])

    def test_status_on_fresh_run(self):
        # status on a not-yet-recalled run dir (created by recall) -> phase "created"
        (self.run_dir / "problem.txt").write_text(PROBLEM, encoding="utf-8")
        code, out = self.orx("status")
        self.assertEqual(code, 0, out)
        self.assertEqual(out["phase"], "created")

    def test_status_requires_run_dir(self):
        empty = self.tmp / "not_a_run"
        empty.mkdir()
        code, out = run_orx(["status"], empty, self.bank)
        self.assertEqual(code, 2)
        self.assertIn("not a run directory", out["error"])


if __name__ == "__main__":
    unittest.main()
