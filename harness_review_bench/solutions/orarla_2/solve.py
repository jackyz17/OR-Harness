#!/usr/bin/env python3
"""
orarla_2: Fighter jet pilot training allocation
Chosen strategy: direct_calculation

This is a direct arithmetic problem:
- Year 1: a1=10 jets, each trains 5 pilots → 10*5 = 50 pilots
- Year 2: a2=15 jets, each trains 5 pilots → 15*5 = 75 pilots
- Total trained pilots by end of year 2 = 50 + 75 = 125
"""

import json
import time

def solve():
    start = time.time()

    # Given parameters
    a1 = 10  # year-1 production
    a2 = 15  # year-2 production
    rate = 5  # pilots per training jet per year

    # Direct calculation (no optimization needed)
    pilots_year1 = a1 * rate
    pilots_year2 = a2 * rate
    total_pilots = pilots_year1 + pilots_year2

    elapsed = time.time() - start

    result = {
        "status": "optimal",
        "objective_value": total_pilots,
        "objective_bound": total_pilots,
        "mip_gap": 0.0,
        "runtime_seconds": elapsed,
        "variables": {
            "pilots_year1": pilots_year1,
            "pilots_year2": pilots_year2,
            "total_pilots": total_pilots
        },
        "method_performed": {
            "name": "direct_calculation",
            "steps": [
                "Recognize this is a direct arithmetic calculation, not an optimization problem",
                "Year 1: pilots = a1 × 5 = 10 × 5 = 50",
                "Year 2: pilots = a2 × 5 = 15 × 5 = 75",
                "Total pilots by end of year 2 = 50 + 75 = 125"
            ]
        }
    }

    with open("result.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"Result: {total_pilots} trained pilots by end of year 2")
    print(f"Year 1: {a1} jets × {rate} pilots/jet = {pilots_year1} pilots")
    print(f"Year 2: {a2} jets × {rate} pilots/jet = {pilots_year2} pilots")
    print(f"Total: {total_pilots} pilots")

if __name__ == "__main__":
    solve()
