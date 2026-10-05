"""
orarla_10: Minimum Chain Store Location (Set Cover Problem)
Method: milp_set_cover
- Binary variables x_i for each location i (1 = store at i, 0 = no store)
- Objective: minimize sum(x_i) — minimum number of stores
- Constraint: for each area a, sum of x_j for j in within_800m[a] >= 1
"""

import json
import time
import highspy

# Coverage data: area -> set of locations within 800m that can cover it
coverage = {
    "A": ["A", "C", "E", "G", "H", "I"],
    "B": ["B", "H", "I"],
    "C": ["A", "C", "G", "H", "I"],
    "D": ["D", "J"],
    "E": ["A", "E", "G"],
    "F": ["F", "J", "K"],
    "G": ["A", "C", "E", "G"],
    "H": ["A", "B", "C", "H", "I"],
    "I": ["A", "B", "C", "H", "I"],
    "J": ["D", "F", "J", "K", "L"],
    "K": ["F", "J", "K", "L"],
    "L": ["J", "K", "L"],
}

locations = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]

# Create HiGHS model
h = highspy.Highs()
h.changeObjectiveSense(highspy.ObjSense.kMinimize)

# Add binary decision variables x_i for each location
# Each x_i = 1 if store at location i, 0 otherwise
var_index = {}
variables = {}
for loc in locations:
    v = h.addVariable(lb=0.0, ub=1.0, obj=1.0, type=highspy.HighsVarType.kInteger, name=f"x_{loc}")
    var_index[loc] = v.index
    variables[loc] = v

# Add covering constraints: for each area, at least one store in its coverage set
for area, cover_set in coverage.items():
    # Build expression: sum of x_j for j in cover_set
    expr = None
    for loc in cover_set:
        if expr is None:
            expr = variables[loc]
        else:
            expr = expr + variables[loc]
    # Constraint: sum >= 1
    constr = expr >= 1.0
    h.addConstr(constr, name=f"cover_{area}")

# Solve
start_time = time.time()
h.setOptionValue('mip_rel_gap', 0.0)
h.setOptionValue('mip_abs_gap', 0.0)
h.run()
solve_time = time.time() - start_time

# Get solution
solution = h.getSolution()
model_status = h.getModelStatus()
info = h.getInfo()

status_map = {
    highspy.HighsModelStatus.kOptimal: "optimal",
}
status_str = status_map.get(model_status, str(model_status).replace('HighsModelStatus.', ''))

result = {
    "status": status_str,
    "objective_value": info.objective_function_value if info.objective_function_value is not None else None,
    "objective_bound": info.mip_dual_bound if hasattr(info, 'mip_dual_bound') and info.mip_dual_bound is not None else None,
    "runtime_seconds": solve_time,
    "variables": {}
}

# Extract which locations get stores
stores = []
for loc in locations:
    idx = var_index[loc]
    val = solution.col_value[idx]
    result["variables"][f"x_{loc}"] = val
    if val > 0.5:  # binary variable
        stores.append(loc)

result["stores_selected"] = sorted(stores)
result["num_stores"] = len(stores)

# Write result
with open("result.json", "w") as f:
    json.dump(result, f, indent=2)

print(f"Status: {result['status']}")
print(f"Objective: {result['objective_value']}")
print(f"Stores: {result['stores_selected']}")
print(f"Num stores: {result['num_stores']}")
print(f"Runtime: {solve_time:.4f}s")
