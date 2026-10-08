"""
marketing_channels_budget_2026_10_08 -- marketing budget allocation between channels X and Y.

Model (written AFTER the strategy decision; see task.json.model):
    minimize  50*X + 100*Y
    s.t.      C1: X + Y   <= 2000     (total budget for both channels combined <= $2000)
              C2: 3*X - 2*Y >= 500     ("effort from X three times greater than twice the effort
                                        from Y, difference of at least 500 points")
              X, Y integer >= 0         (budgets allocated are integers)

Solver: OR-Tools CP-SAT (exact integer program). Strategy cand_6247b808 / prediction sp_f3c90ad2d857.
"""
import json
import os
import time

from ortools.sat.python import cp_model

ACTION_ID = os.environ.get("OR_ACTION_ID")

COST_X, COST_Y = 50, 100
CAP = 2000          # C1
MIN_EFFORT = 500    # C2

t0 = time.time()

model = cp_model.CpModel()
# Finite domains large enough to contain the optimum (upper bounds from C1 when the
# other variable is 0). Integer by task requirement ("budgets allocated are integers").
x = model.NewIntVar(0, CAP, "X")
y = model.NewIntVar(0, CAP, "Y")

model.Add(x + y <= CAP)              # C1
model.Add(3 * x - 2 * y >= MIN_EFFORT)  # C2

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
        "name": "cp_sat_integer",
        "steps": [
            "declare integer variables X and Y with non-negative domains",
            "encode budget_limit: X + Y <= 2000 and effort_req: 3*X - 2*Y >= 500",
            "minimize the linear cost 50*X + 100*Y with OR-Tools CP-SAT",
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
