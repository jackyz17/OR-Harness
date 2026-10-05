#!/usr/bin/env python3
"""
enumerate_exclude_one: solve assignment problem (select 4 of 5 workers, assign to 4 tasks)
by enumerating all 5 ways to exclude one worker, then solving a 4x4 assignment via Hungarian.
"""

import json
import time
from scipy.optimize import linear_sum_assignment
import os

# Hours matrix: workers x tasks
# Workers: I, II, III, IV, V
# Tasks: A, B, C, D
HOURS = {
    ("I", "A"): 9,  ("I", "B"): 4,  ("I", "C"): 3,  ("I", "D"): 7,
    ("II", "A"): 4, ("II", "B"): 6, ("II", "C"): 5, ("II", "D"): 6,
    ("III", "A"): 5, ("III", "B"): 4, ("III", "C"): 7, ("III", "D"): 5,
    ("IV", "A"): 7, ("IV", "B"): 5, ("IV", "C"): 2, ("IV", "D"): 3,
    ("V", "A"): 10, ("V", "B"): 6, ("V", "C"): 7, ("V", "D"): 4,
}

WORKERS = ["I", "II", "III", "IV", "V"]
TASKS = ["A", "B", "C", "D"]

def solve_exclude_one(excluded: str) -> tuple:
    """Solve 4x4 assignment with one worker excluded. Returns (cost, assignment_dict)."""
    active_workers = [w for w in WORKERS if w != excluded]
    n = len(active_workers)
    
    # Build 4x4 cost matrix (active_workers x TASKS)
    cost_matrix = []
    for w in active_workers:
        row = [HOURS[(w, t)] for t in TASKS]
        cost_matrix.append(row)
    
    # Hungarian algorithm: row_ind = worker index, col_ind = task index
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    total_cost = sum(cost_matrix[r][c] for r, c in zip(row_ind, col_ind))
    
    assignment = {}
    for r, c in zip(row_ind, col_ind):
        worker = active_workers[r]
        task = TASKS[c]
        assignment[task] = worker
    
    return total_cost, assignment


def main():
    start = time.time()
    
    best_cost = float("inf")
    best_exclusion = None
    best_assignment = None
    
    for excluded in WORKERS:
        cost, assignment = solve_exclude_one(excluded)
        print(f"Exclude {excluded}: cost={cost}, assignment={assignment}")
        if cost < best_cost:
            best_cost = cost
            best_exclusion = excluded
            best_assignment = assignment
    
    elapsed = time.time() - start
    
    print(f"\nBest: exclude {best_exclusion}, total hours={best_cost}")
    print(f"Assignment: {best_assignment}")
    
    # Build result
    # Variables: for each (worker, task) pair, x = 1 if assigned
    variables = {}
    for task, worker in best_assignment.items():
        for t, w in [(task, worker)]:
            pass
        variables[f"x_{worker}_{task}"] = 1
    
    # Add zero assignments for excluded worker
    for task in TASKS:
        variables[f"x_{best_exclusion}_{task}"] = 0
    
    # Also record non-assigned (worker, task) pairs as 0
    assigned_pairs = {(w, t) for t, w in best_assignment.items()}
    for w in WORKERS:
        for t in TASKS:
            key = f"x_{w}_{t}"
            if key not in variables:
                variables[key] = 0
    
    result = {
        "status": "optimal",
        "objective_value": best_cost,
        "objective_bound": best_cost,
        "runtime_seconds": elapsed,
        "variables": variables,
        "assignment": best_assignment,
        "excluded_worker": best_exclusion,
    }
    
    with open("result.json", "w") as f:
        json.dump(result, f)
    
    print(f"\nResult written to result.json")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
