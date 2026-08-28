"""Default cross-validation gate tests: min_cross_validation_branches = 3.

The gate is configurable (config file / OR_EXPERIENCE_MIN_CV_BRANCHES env /
dataclass default). These tests verify the DEFAULT behavior:
  - 2 valid branches are NOT enough (the old default) -> consistent=false
  - 3 valid branches pass
  - the env override lowers the gate back to 2
  - the config floor is 2 (a configured 1 is clamped)
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

MODEL = """[THINK]
Resource-allocation LP: two decisions share one feed budget.
[/THINK]
[MODEL]
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
[/MODEL]"""

SIGNATURE = {
    "objective": "linear",
    "decision": ["integer_batch"],
    "constraint": ["capacity"],
    "interaction": "shared_resource_coupled",
    "features": {"resource": "feed"},
}

SOLVE_CODE = """
import json
result = {{
    "status": "optimal",
    "solver": "{solver}",
    "objective_sense": "maximize",
    "objective_value": 900,
    "objective_bound": 900,
    "mip_gap": 0.0,
    "runtime_seconds": 0.01,
    "variables": {{"x[cow]": 20, "x[chicken]": 20}},
    "diagnostics": {{}},
    "message": "stub",
}}
with open("result.json", "w") as handle:
    json.dump(result, handle)
"""


def run_orx(args, cwd, bank_home, extra_env=None):
    env = dict(os.environ)
    env["OR_EXPERIENCE_BANK_HOME"] = str(bank_home)
    if extra_env:
        env.update(extra_env)
    proc = subprocess.run(
        ORX + args, cwd=str(cwd), env=env, capture_output=True, text=True, timeout=180
    )
    try:
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        payload = {"_raw_stdout": proc.stdout, "_raw_stderr": proc.stderr}
    return proc.returncode, payload


class TestDefaultCrossValidationGate(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.tmp = Path(tempfile.mkdtemp(prefix="orx_gate_"))
        self.bank = self.tmp / "bank"
        self.run_dir = self.tmp / "run"
        self.run_dir.mkdir(parents=True)
        run_orx(["init", "--bank-home", str(self.bank)], self.tmp, self.bank)
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

    def _add_branches(self, solvers):
        for solver in solvers:
            br = self.run_dir / "branches" / solver
            br.mkdir(parents=True, exist_ok=True)
            (br / "solve.py").write_text(SOLVE_CODE.format(solver=solver), encoding="utf-8")
        run_orx(["solve", "--solver", ",".join(solvers)], self.run_dir, self.bank)

    def test_two_branches_insufficient_under_default_gate(self):
        self._add_branches(["highs", "pulp"])
        code, out = run_orx(["cross-validate"], self.run_dir, self.bank)
        self.assertEqual(code, 0, out)
        self.assertFalse(out["consistent"], out)
        self.assertIn("need >=3", out["reason"], out)
        self.assertIn("min_cross_validation_branches", out["next"], out)

    def test_three_branches_pass_under_default_gate(self):
        self._add_branches(["highs", "pulp", "scip"])
        code, out = run_orx(["cross-validate"], self.run_dir, self.bank)
        self.assertEqual(code, 0, out)
        self.assertTrue(out["consistent"], out)
        self.assertEqual(out["branches_compared"], 3)

    def test_env_override_restores_two_branch_gate(self):
        self._add_branches(["highs", "pulp"])
        code, out = run_orx(
            ["cross-validate"], self.run_dir, self.bank,
            extra_env={"OR_EXPERIENCE_MIN_CV_BRANCHES": "2"},
        )
        self.assertEqual(code, 0, out)
        self.assertTrue(out["consistent"], out)

    def test_config_floor_is_two(self):
        # A configured value below 2 is clamped to 2, never 1.
        self._add_branches(["highs"])
        code, out = run_orx(
            ["cross-validate"], self.run_dir, self.bank,
            extra_env={"OR_EXPERIENCE_MIN_CV_BRANCHES": "1"},
        )
        self.assertEqual(code, 0, out)
        self.assertFalse(out["consistent"], out)
        self.assertIn("need >=2", out["reason"], out)


if __name__ == "__main__":
    unittest.main()
