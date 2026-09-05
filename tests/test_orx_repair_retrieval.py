"""Verification: a failing branch retrieves repair knowledge from the bank.

Seeds the Repair Bank with an error→fix experience via a full solve run,
then makes a branch fail with the SAME error class and asserts the branch's
result.json carries the seeded repair hint + error-transition-graph guidance.
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

GOOD_CODE = """
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

# Fails at runtime with a NameError (passes the AST security check, then
# crashes when executed) — the realistic repair-retrieval trigger.
FAILING_CODE = """
import json
value = undefined_symbol + 1
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


def drive_to_signature(run_dir: Path, bank: Path) -> None:
    (run_dir / "problem.txt").write_text(PROBLEM, encoding="utf-8")
    assert run_orx(["recall", "--problem-file", "problem.txt"], run_dir, bank)[0] == 0
    (run_dir / "model.txt").write_text(MODEL, encoding="utf-8")
    code, out = run_orx(["validate"], run_dir, bank)
    assert out["passed"], out
    (run_dir / "signature.json").write_text(json.dumps(SIGNATURE), encoding="utf-8")
    code, out = run_orx(["signature"], run_dir, bank)
    assert out["passed"], out


class TestFailingBranchRetrievesBank(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.tmp = Path(tempfile.mkdtemp(prefix="orx_repair_"))
        self.bank = self.tmp / "bank"
        run_orx(["init", "--bank-home", str(self.bank)], self.tmp, self.bank)

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)

    def orx(self, *args):
        return run_orx(list(args), self.run_dir, self.bank)

    def test_failing_branch_carries_repair_hints(self):
        # --- Phase 1: seed the Repair Bank via a successful solve run. ---
        seed = self.tmp / "seed"
        seed.mkdir()
        drive_to_signature(seed, self.bank)
        br = seed / "branches" / "highs"
        br.mkdir(parents=True, exist_ok=True)
        (br / "solve.py").write_text(GOOD_CODE.format(solver="highs"), encoding="utf-8")
        run_orx(["solve", "--solver", "highs"], seed, self.bank)
        run_orx(["gold", "--answer", "900"], seed, self.bank)

        repair_exp = {
            "layer": "repair",
            "title": "NameError on undefined symbol: declare before use",
            "retrieval_text": "NameError undefined symbol name is not defined: "
                              "declare the variable before referencing it in generated code",
            "diagnosis": "NameError: name is not defined when generated code references "
                         "an undeclared symbol",
            "action": "declare the symbol (or import it) before first use",
            "rationale": "generated code must define every name it references",
            "solver": "highs",
        }
        exp_file = seed / "exp_repair.json"
        exp_file.write_text(json.dumps(repair_exp), encoding="utf-8")
        code, out = run_orx(["append", "--file", str(exp_file)], seed, self.bank)
        self.assertEqual(out["status"], "appended", out)
        run_orx(["episode"], seed, self.bank)

        # --- Phase 2: a NEW run whose highs branch fails with the same error class. ---
        self.run_dir = self.tmp / "run2"
        self.run_dir.mkdir()
        drive_to_signature(self.run_dir, self.bank)
        br = self.run_dir / "branches" / "highs"
        br.mkdir(parents=True, exist_ok=True)
        (br / "solve.py").write_text(FAILING_CODE, encoding="utf-8")

        code, out = self.orx("solve", "--solver", "highs")
        self.assertEqual(code, 0, out)
        self.assertEqual(out["status"], "error")
        self.assertTrue(out["normalized_error"], out)

        # The branch's result.json must carry the seeded repair hint.
        record = json.loads((br / "result.json").read_text(encoding="utf-8"))
        self.assertIn("NameError", record["normalized_error"])
        repair_titles = [h.get("title", "") for h in record["repair_hints"]]
        self.assertTrue(
            any("NameError" in t for t in repair_titles),
            "seeded repair experience not retrieved: {}".format(repair_titles)
        )
        # Error-transition-graph guidance is present (possibly empty lists for a
        # single-record graph, but the field must exist and be structured).
        self.assertIn("repair_graph_guidance", record)
        self.assertIsInstance(record["repair_graph_guidance"], dict)

    def test_failing_branch_in_parallel_also_carries_repair_hints(self):
        """Same guarantee on a repair retry after a failure: the branch's own
        result.json carries its repair hints."""
        # Seed the repair bank first (reuse the phase-1 flow).
        seed = self.tmp / "seed"
        seed.mkdir()
        drive_to_signature(seed, self.bank)
        br = seed / "branches" / "highs"
        br.mkdir(parents=True, exist_ok=True)
        (br / "solve.py").write_text(GOOD_CODE.format(solver="highs"), encoding="utf-8")
        run_orx(["solve", "--solver", "highs"], seed, self.bank)
        run_orx(["gold", "--answer", "900"], seed, self.bank)
        repair_exp = {
            "layer": "repair",
            "title": "NameError on undefined symbol: declare before use",
            "retrieval_text": "NameError undefined symbol name is not defined: "
                              "declare the variable before referencing it",
            "diagnosis": "NameError: name is not defined",
            "action": "guard open() with os.path.exists",
            "rationale": "generated code must define every name it references",
            "solver": "highs",
        }
        exp_file = seed / "exp_repair.json"
        exp_file.write_text(json.dumps(repair_exp), encoding="utf-8")
        run_orx(["append", "--file", str(exp_file)], seed, self.bank)
        run_orx(["episode"], seed, self.bank)

        self.run_dir = self.tmp / "run2"
        self.run_dir.mkdir()
        drive_to_signature(self.run_dir, self.bank)
        br = self.run_dir / "branches" / "highs"
        br.mkdir(parents=True, exist_ok=True)
        (br / "solve.py").write_text(FAILING_CODE, encoding="utf-8")

        code, out = self.orx("solve", "--solver", "highs")
        self.assertEqual(code, 0, out)
        self.assertEqual(out["status"], "error")

        # Failing branch: repair hints present.
        fail_record = json.loads(
            (self.run_dir / "branches" / "highs" / "result.json").read_text(encoding="utf-8"))
        repair_titles = [h.get("title", "") for h in fail_record["repair_hints"]]
        self.assertTrue(any("NameError" in t for t in repair_titles),
                        "repair hint missing on failing branch: {}".format(repair_titles))

        # Fix the branch code and retry: the repaired run carries no repair hints.
        (br / "solve.py").write_text(GOOD_CODE.format(solver="highs"), encoding="utf-8")
        code, out = self.orx("solve", "--solver", "highs")
        self.assertEqual(code, 0, out)
        self.assertTrue(out["valid"], out)
        ok_record = json.loads(
            (self.run_dir / "branches" / "highs" / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(ok_record["repair_hints"], [])


if __name__ == "__main__":
    unittest.main()
