"""
marketing_xy_budget_20261008 -- marketing budget allocation between channels X and Y.

Model (written AFTER the strategy decision; see task.json.model):
    minimize  10*X + 20*Y
    s.t.      C1: X + Y   <= 1000     (total budget for both channels combined <= $1000)
              C2: 2*X + 3*Y >= 2000   (effectiveness score >= 2000 points;
                                       coefficients 2 and 3 are the LITERAL
                                       "2 times X plus 3 times Y" numbers from the
                                       task text, NOT the per-unit costs 10 / 20)
              X, Y integer >= 0         (budgets are indivisible)

Objective coefficients are the per-unit costs: channel X $10, channel Y $20.

Solver: OR-Tools CP-SAT (exact integer program).
Strategy cand_acf02a4b / prediction sp_447af862f222.
"""
import json
import os
import time

from ortools.sat.python import cp_model

ACTION_ID = os.environ.get("OR_ACTION_ID")

COST_X, COST_Y = 10, 20
CAP = 1000          # C1
MIN_EFF = 2000      # C2

t0 = time.time()

model = cp_model.CpModel()
# Finite domains large enough to contain the optimum (upper bounds from C1).
x = model.NewIntVar(0, CAP, "X")
y = model.NewIntVar(0, CAP, "Y")

model.Add(x + y <= CAP)             # C1 budget cap
model.Add(2 * x + 3 * y >= MIN_EFF)  # C2 effectiveness (literal coefficients 2, 3)

model.Minimize(COST_X * x + COST_Y * y)

solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = 60.0
solver.parameters.num_search_workers = 1
status = solver.Solve(model)
runtime = time.time() - t0

status_map = {
    cp_model.OPTIMAL: "optimal",
    cp_model.FEASIBLE: "feasible",
    cp_model.INFEASIBLE: "infeasible",
    cp_model.MODEL_INVALID: "error",
    cp_model.UNKNOWN: "timeout",
}
status_str = status_map.get(status, "error")

result = {
    "status": status_str,
    "objective_value": None,
    "objective_bound": None,
    "mip_gap": None,
    "runtime_seconds": runtime,
    "config": {
        "action_id": ACTION_ID,
        "time_limit": 60,
        "num_search_workers": 1,
    },
    "method_performed": {
        "action_id": ACTION_ID,
        "name": "cp_sat_direct_integer",
        "steps": [
            "declare non-negative integer variables X and Y",
            "encode C1 budget cap: X + Y <= 1000",
            "encode C2 effectiveness requirement: 2*X + 3*Y >= 2000",
            "minimize 10*X + 20*Y with OR-Tools CP-SAT as an exact integer solver",
            "report optimal objective value and the (X, Y) allocation",
        ],
    },
}

if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
    xv = int(solver.Value(x))
    yv = int(solver.Value(y))
    obj = int(round(solver.ObjectiveValue()))
    bound = solver.BestObjectiveBound()
    result["objective_value"] = obj
    result["objective_bound"] = int(round(bound))
    result["mip_gap"] = 0.0 if status == cp_model.OPTIMAL else abs(obj - bound) / max(1.0, abs(obj))
    result["variables"] = {"X": xv, "Y": yv}
    result["solution_cost_recheck"] = COST_X * xv + COST_Y * yv

with open("result.json", "w") as fh:
    json.dump(result, fh, indent=2)

print(json.dumps(result, indent=2))
