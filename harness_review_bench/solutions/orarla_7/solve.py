#!/usr/bin/env python3
"""
network_flow_transport: min-cost transportation of containers from warehouses to ports.
Uses COPTPY.
"""

import coptpy as cp
import time
import json

# ── Data ──────────────────────────────────────────────────────────────────────
warehouses = ["Verona", "Perugia", "Rome", "Pescara", "Taranto", "Lamezia"]
ports      = ["Genoa", "Venice", "Ancona", "Naples", "Bari"]

supply = {
    "Verona": 10,
    "Perugia": 12,
    "Rome":   20,
    "Pescara": 24,
    "Taranto": 18,
    "Lamezia": 40,
}

demand = {
    "Genoa":  20,
    "Venice": 15,
    "Ancona": 25,
    "Naples": 33,
    "Bari":   21,
}

distance = {
    ("Verona",  "Genoa"):  290, ("Verona",  "Venice"):  115,
    ("Verona",  "Ancona"):  355, ("Verona",  "Naples"):  715,
    ("Verona",  "Bari"):    810,
    ("Perugia", "Genoa"):  380, ("Perugia", "Venice"):  340,
    ("Perugia", "Ancona"):  165, ("Perugia", "Naples"):  380,
    ("Perugia", "Bari"):    610,
    ("Rome",    "Genoa"):  505, ("Rome",    "Venice"):  530,
    ("Rome",    "Ancona"):  285, ("Rome",    "Naples"):  220,
    ("Rome",    "Bari"):    450,
    ("Pescara", "Genoa"):  655, ("Pescara", "Venice"):  450,
    ("Pescara", "Ancona"):  155, ("Pescara", "Naples"):  240,
    ("Pescara", "Bari"):    315,
    ("Taranto", "Genoa"): 1010, ("Taranto", "Venice"):  840,
    ("Taranto", "Ancona"):  550, ("Taranto", "Naples"):  305,
    ("Taranto", "Bari"):     95,
    ("Lamezia", "Genoa"): 1072, ("Lamezia", "Venice"): 1097,
    ("Lamezia", "Ancona"):  747, ("Lamezia", "Naples"):  372,
    ("Lamezia", "Bari"):    333,
}

COST_PER_CONTAINER_PER_KM = 30.0   # euros per km

# ── Model ──────────────────────────────────────────────────────────────────────
env   = cp.Envr()
model = env.createModel("container_transport")

# Decision variables: x[w][p] = containers from warehouse w to port p
x = {}
for w in warehouses:
    for p in ports:
        x[w, p] = model.addVar(
            lb=0, ub=supply[w],
            vtype=cp.COPT.INTEGER,
            name=f"x_{w}_{p}"
        )

# Objective: total transport cost
model.setObjective(
    sum(COST_PER_CONTAINER_PER_KM * distance[w, p] * x[w, p]
        for w in warehouses for p in ports),
    sense=cp.COPT.MINIMIZE
)

# Supply constraints: can't ship more than available at each warehouse
for w in warehouses:
    model.addConstr(
        sum(x[w, p] for p in ports) <= supply[w],
        name=f"supply_{w}"
    )

# Demand constraints: each port receives exactly its demand
for p in ports:
    model.addConstr(
        sum(x[w, p] for w in warehouses) == demand[p],
        name=f"demand_{p}"
    )

# ── Solve ──────────────────────────────────────────────────────────────────────
start   = time.time()
model.solve()
elapsed = time.time() - start

# ── Map status ────────────────────────────────────────────────────────────────
status_map = {
    cp.COPT.OPTIMAL:    "optimal",
    cp.COPT.INFEASIBLE: "infeasible",
    cp.COPT.UNBOUNDED:  "unbounded",
    cp.COPT.TIMEOUT:    "timeout",
    cp.COPT.NUMERICAL:  "numeric_error",
}
status_str = status_map.get(model.status, f"unknown_{model.status}")

# For MIP: best_bound == objval when optimal (gap = 0)
obj_val   = model.objval if model.status == cp.COPT.OPTIMAL else None
obj_bound = model.objval if model.status == cp.COPT.OPTIMAL else None   # bound == obj when optimal

# Extract non-zero flows
solution = {}
for w in warehouses:
    for p in ports:
        val = x[w, p].x
        if val is not None and val > 1e-6:
            solution[f"x_{w}_{p}"] = val

result = {
    "status":           status_str,
    "objective_value":  obj_val,
    "objective_bound":  obj_bound,
    "runtime_seconds":  elapsed,
    "solution":         solution,
}

with open("result.json", "w") as f:
    json.dump(result, f, indent=2)

print(f"Status   : {status_str}")
print(f"Objective: {obj_val}")
print(f"Runtime  : {elapsed:.3f}s")
print("Solution :", json.dumps(solution, indent=2))
