import pulp
import json

# Cover sets: each area j must be covered by at least one open store
cover = {
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

areas = list(cover.keys())

# Decision variables: x_i = 1 if store opened at area i
x = pulp.LpVariable.dicts("x", areas, cat="Binary")

# Problem: minimize number of stores
prob = pulp.LpProblem("store_location", pulp.LpMinimize)
prob += pulp.lpSum([x[a] for a in areas])

# Constraints: each area must be covered by at least one open store
for a in areas:
    covered_by = cover[a]
    prob += pulp.lpSum([x[c] for c in covered_by]) >= 1, f"cover_{a}"

prob.solve(pulp.PULP_CBC_CMD(msg=0))

status = pulp.LpStatus[prob.status]
obj = pulp.value(prob.objective)
vals = {a: pulp.value(x[a]) for a in areas}
open_areas = [a for a in areas if vals[a] >= 0.5]

result = {
    "status": status,
    "objective_value": obj,
    "variables": vals,
    "open_areas": open_areas,
    "solution": vals
}

with open("result.json", "w") as f:
    json.dump(result, f, indent=2)

print(f"Status: {status}")
print(f"Objective: {obj}")
print(f"Open areas: {open_areas}")
