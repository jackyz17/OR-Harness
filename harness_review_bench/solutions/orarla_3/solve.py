#!/usr/bin/env python3
"""
OR-Harness solve for orarla_3 (planning)
Method: Pure Python Enumeration
Problem: Marketing campaign optimization - find minimum total spending on 
         advertising and promotion to achieve at least 100 impact points.
"""

import json
import os

def solve():
    # Problem parameters
    # Effectiveness: 2*advertising + 3*promotion >= 100
    # Balance: advertising - promotion <= 50
    # Objective: minimize advertising + promotion
    # advertising, promotion are integers >= 0
    
    best_total = float('inf')
    best_ad = None
    best_promo = None
    
    # Enumerate advertising from 0 to some reasonable upper bound
    for ad in range(0, 201):
        # From effectiveness constraint: 3*promotion >= 100 - 2*ad
        # So promotion >= max(0, ceil((100 - 2*ad)/3))
        required_promo = max(0, (100 - 2 * ad + 2) // 3 if ad < 50 else 0)
        
        for promo in range(required_promo, 201):
            # Check effectiveness constraint: 2*ad + 3*promo >= 100
            effectiveness = 2 * ad + 3 * promo
            if effectiveness < 100:
                continue
            
            # Check balance constraint: ad - promo <= 50
            if ad - promo > 50:
                continue
            
            total = ad + promo
            if total < best_total:
                best_total = total
                best_ad = ad
                best_promo = promo
    
    # If we couldn't find a solution in range, expand search
    if best_ad is None:
        for ad in range(0, 501):
            for promo in range(0, 501):
                if 2 * ad + 3 * promo >= 100 and ad - promo <= 50:
                    total = ad + promo
                    if total < best_total:
                        best_total = total
                        best_ad = ad
                        best_promo = promo
    
    # Compute objective bound (trivial: best_total is optimal)
    result = {
        "status": "optimal" if best_ad is not None else "infeasible",
        "objective_value": best_total,
        "objective_bound": best_total,  # Since enumeration finds optimal
        "mip_gap": 0.0,
        "runtime_seconds": None,  # Not applicable for enumeration
        "variables": {
            "advertising_spend": best_ad,
            "promotion_spend": best_promo
        },
        "method_performed": {
            "strategy_id": "orarla_3_pure_python_enum",
            "solver": "Pure Python Enumeration",
            "action_id": os.environ.get("OR_ACTION_ID", "unknown")
        }
    }
    
    print(json.dumps(result, indent=2))
    
    with open("result.json", "w") as f:
        json.dump(result, f, indent=2)

if __name__ == "__main__":
    solve()
