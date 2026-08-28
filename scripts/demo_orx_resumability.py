#!/usr/bin/env python3
"""Demo: cross-process resumability of an orx run (the harness-integration pitch).

Simulates TWO separate harness sessions (each command = one independent process,
like a real agent tool call). Session 1 dies mid-chain (after one solver branch);
session 2 picks up in a FRESH process, inspects state, retries the failed branch,
and completes the chain. No server, no connection, no session memory.
"""

import json
import os
import subprocess
import sys
import tempfile
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


def orx(args, cwd, bank):
    env = dict(os.environ)
    env["OR_EXPERIENCE_BANK_HOME"] = str(bank)
    proc = subprocess.run(ORX + args, cwd=str(cwd), env=env, capture_output=True, text=True)
    return json.loads(proc.stdout.strip().splitlines()[-1])


def main():
    tmp = Path(tempfile.mkdtemp(prefix="orx_demo_"))
    bank = tmp / "bank"
    run = tmp / "run"
    run.mkdir(parents=True)

    print("=== setup ===")
    orx(["init", "--bank-home", str(bank)], tmp, bank)
    (run / "problem.txt").write_text(PROBLEM, encoding="utf-8")

    print("\n=== SESSION 1 (dies mid-chain) ===")
    out = orx(["recall", "--problem-file", "problem.txt"], run, bank)
    print("recall:", out["priors_count"], "priors")
    (run / "model.txt").write_text(MODEL, encoding="utf-8")
    out = orx(["validate"], run, bank)
    print("validate:", out["passed"])
    (run / "signature.json").write_text(json.dumps(SIGNATURE), encoding="utf-8")
    out = orx(["signature"], run, bank)
    print("signature:", out["passed"])
    br = run / "branches" / "highs"
    br.mkdir(parents=True)
    (br / "solve.py").write_text(SOLVE_CODE.format(solver="highs"), encoding="utf-8")
    out = orx(["solve", "--solver", "highs"], run, bank)
    print("solve highs:", out["status"], out["objective_value"])
    print("... session 1 ends (process killed / connection dropped) ...")

    print("\n=== SESSION 2 (fresh process, resumes) ===")
    out = orx(["status"], run, bank)
    print("status: phase =", out["phase"])
    print("status: next =", out["next"])
    # agent reads status, sees 1 branch done, adds the second
    br2 = run / "branches" / "pulp"
    br2.mkdir(parents=True)
    (br2 / "solve.py").write_text(SOLVE_CODE.format(solver="pulp"), encoding="utf-8")
    out = orx(["solve", "--solver", "pulp"], run, bank)
    print("solve pulp:", out["status"], out["objective_value"])
    out = orx(["cross-validate"], run, bank)
    print("cross-validate: consistent =", out["consistent"], "best =", out["best_objective"])
    out = orx(["gold", "--answer", "900"], run, bank)
    print("gold: matched =", out["gold_matched"])
    exp = {
        "layer": "modeling",
        "title": "Shared-feed budget needs one aggregated capacity constraint",
        "retrieval_text": "When two decisions consume one shared budget, aggregate per-unit usage into a single capacity constraint.",
        "modeling_aspect": "constraint",
        "action": "sum(i, usage[i] * x[i]) <= budget",
        "rationale": "Aggregated capacity is the minimal correct form.",
    }
    exp_file = run / "exp.json"
    exp_file.write_text(json.dumps(exp), encoding="utf-8")
    out = orx(["append", "--file", str(exp_file)], run, bank)
    print("append:", out["status"], out["experience_id"])
    out = orx(["episode"], run, bank)
    print("episode:", out["recorded"], "| utility_credited =", out["utility_credited"])

    print("\n=== bank after the run ===")
    out = orx(["stats"], run, bank)
    print(json.dumps(out, indent=2))
    print("\nrun directory preserved at:", run)


if __name__ == "__main__":
    main()
