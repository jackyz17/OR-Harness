#!/usr/bin/env python3
"""enumerate_assignment for orarla_8: select 4 of 5 workers, assign 4 tasks to minimize total hours."""
import json
import itertools

# Time matrix: workers I,II,III,IV,V x tasks A,B,C,D
workers = ["I", "II", "III", "IV", "V"]
tasks = ["A", "B", "C", "D"]
time = {
    ("I", "A"): 9, ("I", "B"): 4, ("I", "C"): 3, ("I", "D"): 7,
    ("II", "A"): 4, ("II", "B"): 6, ("II", "C"): 5, ("II", "D"): 6,
    ("III", "A"): 5, ("III", "B"): 4, ("III", "C"): 7, ("III", "D"): 5,
    ("IV", "A"): 7, ("IV", "B"): 5, ("IV", "C"): 2, ("IV", "D"): 3,
    ("V", "A"): 10, ("V", "B"): 6, ("V", "C"): 7, ("V", "D"): 4,
}

best_cost = float("inf")
best_assignment = None
best_excluded = None

# For each worker excluded (5 possibilities)
for excluded in workers:
    selected = [w for w in workers if w != excluded]
    # For each permutation of tasks assigned to selected workers
    for perm in itertools.permutations(tasks):
        cost = sum(time[(w, t)] for w, t in zip(selected, perm))
        if cost < best_cost:
            best_cost = cost
            best_assignment = dict(zip(selected, perm))
            best_excluded = excluded

# Build result
result = {
    "status": "optimal",
    "objective_value": best_cost,
    "objective_bound": best_cost,
    "runtime_seconds": 0.0,
    "variables": {
        "best_assignment": best_assignment,
        "excluded_worker": best_excluded,
    },
}

with open("result.json", "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result, indent=2))
