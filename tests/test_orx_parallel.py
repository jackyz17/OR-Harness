"""Parallel multi-branch execution tests for `orx solve --solver a,b,c`.

The original design contract: heterogeneous branches run CONCURRENTLY
(asyncio.gather + Semaphore, mirroring orchestrator.py); repair WITHIN a
branch is the agent's sequential retry. These tests verify the parallel form:
  - multiple branches execute in one command,
  - they actually overlap in time (wall-clock < sum of branch durations),
  - each branch writes its own result.json,
  - a missing branch fails fast with a clear error,
  - the single-solver form still works (repair retry path).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
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
Resource-allocation LP: two decisions share one feed budget.
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

# Each branch sleeps ~1.2s before writing result.json. Three branches in
# parallel should take ~1.2-2s total; serially they would take ~3.6s+.
SLEEP_SOLVE_CODE = """
import json
import time
time.sleep(1.2)
result = {{
    "status": "optimal",
    "solver": "{solver}",
    "objective_sense": "maximize",
    "objective_value": 900,
    "objective_bound": 900,
    "mip_gap": 0.0,
    "runtime_seconds": 1.2,
    "variables": {{"x[cow]": 20, "x[chicken]": 20}},
    "diagnostics": {{}},
    "message": "stub with sleep",
}}
with open("result.json", "w") as handle:
    json.dump(result, handle)
"""


def run_orx(args, cwd, bank_home):
    env = dict(os.environ)
    env["OR_EXPERIENCE_BANK_HOME"] = str(bank_home)
    proc = subprocess.run(
        ORX + args, cwd=str(cwd), env=env, capture_output=True, text=True, timeout=180
    )
    try:
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        payload = {"_raw_stdout": proc.stdout, "_raw_stderr": proc.stderr}
    return proc.returncode, payload


class TestParallelSolve(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.tmp = Path(tempfile.mkdtemp(prefix="orx_par_"))
        self.bank = self.tmp / "bank"
        self.run_dir = self.tmp / "run"
        self.run_dir.mkdir(parents=True)
        run_orx(["init", "--bank-home", str(self.bank)], self.tmp, self.bank)

        # Drive the chain up to the signature stamp.
        (self.run_dir / "problem.txt").write_text(PROBLEM, encoding="utf-8")
        run_orx(["recall", "--problem-file", "problem.txt"], self.run_dir, self.bank)
        (self.run_dir / "model.txt").write_text(MODEL, encoding="utf-8")
        code, out = run_orx(["validate"], self.run_dir, self.bank)
        assert out["passed"], out
        (self.run_dir / "signature.json").write_text(json.dumps(SIGNATURE), encoding="utf-8")
        code, out = run_orx(["signature"], self.run_dir, self.bank)
        assert out["passed"], out

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)

    def orx(self, *args):
        return run_orx(list(args), self.run_dir, self.bank)

    def _write_branches(self, solvers):
        for solver in solvers:
            br = self.run_dir / "branches" / solver
            br.mkdir(parents=True, exist_ok=True)
            (br / "solve.py").write_text(
                SLEEP_SOLVE_CODE.format(solver=solver), encoding="utf-8"
            )

    def test_parallel_branches_overlap_in_time(self):
        solvers = ["highs", "pulp", "scip"]
        self._write_branches(solvers)

        start = time.monotonic()
        code, out = self.orx("solve", "--solver", ",".join(solvers))
        elapsed = time.monotonic() - start

        self.assertEqual(code, 0, out)
        self.assertTrue(out["parallel"], out)
        self.assertEqual(out["branches_total"], 3)
        self.assertEqual(out["branches_valid"], 3)

        # Concurrency proof: 3 branches x 1.2s sleep each. Serial >= 3.6s;
        # parallel should be well under (allow generous overhead margin).
        self.assertLess(
            elapsed, 3.0,
            "branches appear to have run serially: took {:.2f}s for 3x1.2s branches".format(elapsed)
        )

        # Each branch wrote its own result.json.
        for solver in solvers:
            result = json.loads(
                (self.run_dir / "branches" / solver / "result.json").read_text(encoding="utf-8")
            )
            self.assertEqual(result["solver"], solver)
            self.assertTrue(result["valid"])

    def test_parallel_missing_branch_fails_fast(self):
        self._write_branches(["highs"])  # pulp/scip code not written
        code, out = self.orx("solve", "--solver", "highs,pulp,scip")
        self.assertEqual(code, 2)
        self.assertIn("missing solve.py", out["error"])
        self.assertIn("pulp", out["error"])

    def test_parallel_then_cross_validate(self):
        solvers = ["highs", "pulp"]
        self._write_branches(solvers)
        code, out = self.orx("solve", "--solver", ",".join(solvers))
        self.assertEqual(out["branches_valid"], 2, out)

        code, out = self.orx("cross-validate")
        self.assertEqual(code, 0, out)
        self.assertTrue(out["consistent"], out)
        self.assertEqual(out["best_objective"], 900)

    def test_single_solver_form_still_works(self):
        """The repair-retry path: one solver, one command."""
        self._write_branches(["highs"])
        code, out = self.orx("solve", "--solver", "highs")
        self.assertEqual(code, 0, out)
        self.assertNotIn("parallel", out)
        self.assertEqual(out["solver"], "highs")
        self.assertTrue(out["valid"], out)

    def test_parallel_mixed_success_and_failure(self):
        """One branch crashes, others succeed: per-branch isolation."""
        solvers = ["highs", "pulp"]
        self._write_branches(solvers)
        # break one branch after writing it
        (self.run_dir / "branches" / "pulp" / "solve.py").write_text(
            "raise RuntimeError('boom')\n", encoding="utf-8"
        )
        code, out = self.orx("solve", "--solver", ",".join(solvers))
        self.assertEqual(code, 0, out)
        self.assertEqual(out["branches_total"], 2)
        self.assertEqual(out["branches_valid"], 1, out)
        statuses = {b["solver"]: b["status"] for b in out["branches"]}
        self.assertEqual(statuses["highs"], "optimal")
        self.assertEqual(statuses["pulp"], "error")


if __name__ == "__main__":
    unittest.main()
