"""Induction auto-check tests: every `orx episode` response carries the
trigger decision, so the agent never needs an external reminder to induce.

Covers:
  - episode response includes induction_check with an instruction
  - below the watermark: should_induce=false with the reason
  - after enough cross-family realizations accumulate: should_induce=true
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

PROBLEM_A = """A factory assigns 3 jobs to 2 machines.
Each job must go to exactly one machine; machine 1 has capacity 10, machine 2 has 8.
Job sizes: 4, 5, 6. Minimize total load imbalance."""

PROBLEM_B = """A hospital schedules 3 nurses into 2 wards.
Each nurse must staff exactly one ward; ward 1 needs at most 10 patients covered, ward 2 at most 8.
Nurse coverage capacities: 4, 5, 6. Minimize total uncovered patients."""

PROBLEM_C = """A warehouse packs 3 orders onto 2 trucks.
Each order must go on exactly one truck; truck 1 fits 10 units, truck 2 fits 8.
Order sizes: 4, 5, 6. Minimize total unused capacity."""

MODEL_TMPL = """<think>
Assignment structure with shared capacity.
</think>
<model>
SETS:
  j in Jobs = {{j1, j2, j3}}
  m in Machines = {{m1, m2}}
PARAMETERS:
  size[j]
  cap[m]
VARIABLES:
  y[j,m] binary
OBJECTIVE:
  minimize sum(j, sum(m, size[j] * y[j,m]))
CONSTRAINTS:
  C1: sum(m, y[j,m]) = 1
  C2: sum(j, size[j] * y[j,m]) <= cap[m]
</model>"""

SIGNATURE = {
    "objective": "linear",
    "decision": ["binary_assignment"],
    "constraint": ["assignment_exactly_once", "capacity"],
    "interaction": "shared_resource_coupled",
    "features": {"resource": "machine_capacity"},
}

SOLVE_CODE = """
import json
result = {{
    "status": "optimal",
    "solver": "{solver}",
    "objective_sense": "minimize",
    "objective_value": 15,
    "objective_bound": 15,
    "mip_gap": 0.0,
    "runtime_seconds": 0.01,
    "variables": {{}},
    "diagnostics": {{}},
    "message": "stub",
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


def full_solve_run(bank: Path, tmp: Path, name: str, problem: str, family_note: str) -> dict:
    """Drive one complete solve run; return the episode response."""
    run_dir = tmp / name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "problem.txt").write_text(problem, encoding="utf-8")
    run_orx(["recall", "--problem-file", "problem.txt"], run_dir, bank)
    (run_dir / "model.txt").write_text(MODEL_TMPL, encoding="utf-8")
    code, out = run_orx(["validate"], run_dir, bank)
    assert out["passed"], out
    (run_dir / "signature.json").write_text(json.dumps(SIGNATURE), encoding="utf-8")
    code, out = run_orx(["signature"], run_dir, bank)
    assert out["passed"], out
    br = run_dir / "branches" / "highs"
    br.mkdir(parents=True, exist_ok=True)
    (br / "solve.py").write_text(SOLVE_CODE.format(solver="highs"), encoding="utf-8")
    run_orx(["solve", "--solver", "highs"], run_dir, bank)
    run_orx(["gold", "--answer", "15"], run_dir, bank)
    exp = {
        "layer": "modeling",
        "title": family_note,
        "retrieval_text": family_note + ": exactly-once plus per-resource capacity constraints.",
        "modeling_aspect": "constraint",
        "action": "sum(m, y[j,m]) = 1; sum(j, size[j]*y[j,m]) <= cap[m]",
    }
    exp_file = run_dir / "exp.json"
    exp_file.write_text(json.dumps(exp), encoding="utf-8")
    code, out = run_orx(["append", "--file", str(exp_file)], run_dir, bank)
    assert out["status"] == "appended", out
    code, out = run_orx(["episode"], run_dir, bank)
    assert out["recorded"], out
    return out


class TestInductionAutoCheck(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.tmp = Path(tempfile.mkdtemp(prefix="orx_trig_"))
        self.bank = self.tmp / "bank"
        run_orx(["init", "--bank-home", str(self.bank)], self.tmp, self.bank)

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_episode_response_carries_induction_check(self):
        out = full_solve_run(self.bank, self.tmp, "runA", PROBLEM_A, "Job-machine allocation capacity")
        self.assertIn("induction_check", out, "episode response missing induction_check")
        check = out["induction_check"]
        self.assertIn("should_induce", check)
        self.assertIn("instruction", check)
        # One realization from one family: no heterogeneous cluster yet.
        self.assertFalse(check["should_induce"])
        self.assertIn("instruction", check)

    def test_should_induce_flips_true_when_cluster_forms(self):
        # Three runs from three DIFFERENT families, same structural signature.
        # Trigger semantics (v1): the watermark gate is BYPASSED on the very
        # first run (no prior watermark), so the decision flips as soon as a
        # heterogeneous isomorphic cluster exists — 2 realizations from 2
        # different families already form one.
        out_a = full_solve_run(self.bank, self.tmp, "runA", PROBLEM_A, "Job-machine allocation capacity")
        # One realization: no cluster (needs >=2 families).
        self.assertFalse(out_a["induction_check"]["should_induce"])

        out_b = full_solve_run(self.bank, self.tmp, "runB", PROBLEM_B, "Nurse-ward scheduling coverage")
        # Two realizations from two families: heterogeneous cluster exists,
        # first-run watermark bypass applies -> should_induce flips true.
        check = out_b["induction_check"]
        self.assertTrue(check["should_induce"], check)
        self.assertIn("orx clusters", check["instruction"])

        # A third family keeps the cluster fresh (membership changed).
        out_c = full_solve_run(self.bank, self.tmp, "runC", PROBLEM_C, "Order-truck packing capacity")
        self.assertTrue(out_c["induction_check"]["should_induce"])

    def test_trigger_command_still_works_standalone(self):
        code, out = run_orx(["trigger"], self.tmp, self.bank)
        self.assertEqual(code, 0, out)
        self.assertIn("should_induce", out)


if __name__ == "__main__":
    unittest.main()
