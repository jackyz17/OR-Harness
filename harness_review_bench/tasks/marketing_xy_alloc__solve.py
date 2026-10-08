import json
import os
import time

from ortools.sat.python import cp_model

ACTION_ID = os.environ.get("OR_ACTION_ID")

# --- model data (read back from the task text) -------------------------------
COST_X = 50
COST_Y = 30
CAP = 1000        # total resource allocation for both projects <= 1000
MIN_EXCESS = 200  # x - y >= 200
X_MAX = 700
Y_MAX = 500

start = time.time()

model = cp_model.CpModel()
x = model.NewIntVar(0, X_MAX, "x")
y = model.NewIntVar(0, Y_MAX, "y")

# C1: x + y <= cap
model.Add(x + y <= CAP)
# C2: x - y >= min_excess
model.Add(x - y >= MIN_EXCESS)
# C3: x <= x_max  (implicit in NewIntVar upper bound)
# C4: y <= y_max  (implicit in NewIntVar upper bound)

model.Minimize(COST_X * x + COST_Y * y)

solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = 60.0
solver.parameters.num_search_workers = 1
status = solver.Solve(model)

runtime = time.time() - start

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
            "create CP-SAT integer variables x in 0..700 and y in 0..500",
            "add linear constraints x + y <= 1000 and x - y >= 200",
            "minimize the linear objective 50*x + 30*y with OR-Tools CP-SAT",
            "report optimal objective value and the (x, y) values",
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
    result["variables"] = {"x": xv, "y": yv}
    result["solution_cost_recheck"] = COST_X * xv + COST_Y * yv

with open("result.json", "w") as fh:
    json.dump(result, fh)

print(json.dumps(result, indent=2))
