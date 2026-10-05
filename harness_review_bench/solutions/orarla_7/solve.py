"""
ORCLAW orarla_7: Container transportation (minimum cost)
Method: milp_transport_integer
Solver: ortools linear_solver (SCIP/MIP)
Note: supply[w] <= available (not all must be shipped); demand[p] = required exactly
"""
import json
from ortools.linear_solver import pywraplp

# Data
warehouses = ["Verona", "Perugia", "Rome", "Pescara", "Taranto", "Lamezia"]
ports = ["Genoa", "Venice", "Ancona", "Naples", "Bari"]
supply = {"Verona": 10, "Perugia": 12, "Rome": 20, "Pescara": 24, "Taranto": 18, "Lamezia": 40}
demand = {"Genoa": 20, "Venice": 15, "Ancona": 25, "Naples": 33, "Bari": 21}

# Distance table (km)
distances = {
    ("Verona", "Genoa"): 290, ("Verona", "Venice"): 115, ("Verona", "Ancona"): 355, ("Verona", "Naples"): 715, ("Verona", "Bari"): 810,
    ("Perugia", "Genoa"): 380, ("Perugia", "Venice"): 340, ("Perugia", "Ancona"): 165, ("Perugia", "Naples"): 380, ("Perugia", "Bari"): 610,
    ("Rome", "Genoa"): 505, ("Rome", "Venice"): 530, ("Rome", "Ancona"): 285, ("Rome", "Naples"): 220, ("Rome", "Bari"): 450,
    ("Pescara", "Genoa"): 655, ("Pescara", "Venice"): 450, ("Pescara", "Ancona"): 155, ("Pescara", "Naples"): 240, ("Pescara", "Bari"): 315,
    ("Taranto", "Genoa"): 1010, ("Taranto", "Venice"): 840, ("Taranto", "Ancona"): 550, ("Taranto", "Naples"): 305, ("Taranto", "Bari"): 95,
    ("Lamezia", "Genoa"): 1072, ("Lamezia", "Venice"): 1097, ("Lamezia", "Ancona"): 747, ("Lamezia", "Naples"): 372, ("Lamezia", "Bari"): 333,
}

RATE = 30  # EUR per km

# Create solver
solver = pywraplp.Solver.CreateSolver("SCIP")
if not solver:
    result = {"status": "error", "objective_value": None, "objective_bound": None, "runtime_seconds": 0.0, "variables": {}}
    with open("result.json", "w") as f:
        json.dump(result, f)
    raise Exception("Failed to create solver")

# Variables: x[w,p] >= 0 integer (container counts)
x = {}
for w in warehouses:
    for p in ports:
        x[(w, p)] = solver.IntVar(0.0, float("inf"), f"x_{w}_{p}")

# Objective: minimize sum(w,p) x[w,p] * distance * 30
solver.Minimize(sum(x[(w, p)] * distances[(w, p)] * RATE for w in warehouses for p in ports))

# Supply constraints: sum_p x[w,p] <= supply[w] (can leave some at warehouse)
for w in warehouses:
    solver.Add(sum(x[(w, p)] for p in ports) <= float(supply[w]))

# Demand constraints: sum_w x[w,p] = demand[p] (must satisfy all port demand)
for p in ports:
    solver.Add(sum(x[(w, p)] for w in warehouses) == float(demand[p]))

# Solve
solver.SetTimeLimit(60000)  # 60 seconds
status = solver.Solve()

# Get solution
if status == pywraplp.Solver.OPTIMAL:
    status_str = "optimal"
    obj_value = solver.Objective().Value()
elif status == pywraplp.Solver.FEASIBLE:
    status_str = "feasible"
    obj_value = solver.Objective().Value()
elif status == pywraplp.Solver.INFEASIBLE:
    status_str = "infeasible"
    obj_value = None
elif status == pywraplp.Solver.UNBOUNDED:
    status_str = "unbounded"
    obj_value = None
else:
    status_str = "error"
    obj_value = None

var_values = {}
for (w, p) in x:
    var_values[f"x_{w}_{p}"] = x[(w, p)].solution_value()

result = {
    "status": status_str,
    "objective_value": obj_value,
    "objective_bound": obj_value,  # optimal => bound == objective
    "runtime_seconds": solver.wall_time() / 1000.0,
    "variables": var_values
}

with open("result.json", "w") as f:
    json.dump(result, f, indent=2)

print(f"Solved: {status_str}, objective={obj_value}")
