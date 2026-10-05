"""
solve.py for orarla_2: Fighter jet pilot training allocation.

Problem interpretation:
- a1=10 jets produced in year 1, a2=15 jets produced in year 2
- Each training jet trains 5 pilots per year
- Training takes 2 years (starts year 1, ends year 2)
- Question: how many trained pilots available by end of year 2?

Model:
- t1 = jets allocated to training in year 1 (integer, 0..10)
- t2 = jets allocated to training in year 2 (integer, 0..15)

Key insight: Year 1 combat jets (10-t1) need pilots from year 0, which don't exist.
Therefore t1 must equal 10 (all year 1 jets must train).

Year 1 jets train for 2 years, producing 5 pilots per jet per year.
By end of year 2, the first cohort of 5*t1 = 50 pilots is fully trained.

Year 2 jets start training in year 2; they won't be fully trained until year 3.
So by end of year 2, only the year 1 cohort contributes trained pilots.

Objective: maximize trained pilots by end of year 2 = 5 * t1
Since t1 is fixed at 10 (must train all year 1 jets), optimal objective = 50.
"""

import json
import highspy

# Constants from problem
A1 = 10  # Year 1 jet production
A2 = 15  # Year 2 jet production
PILOTS_PER_JET_YEAR = 5  # Each training jet trains 5 pilots per year
TRAINING_YEARS = 2  # Training duration

# Create HiGHS instance
h = highspy.Highs()
h.setOptionValue("output_flag", False)

# Set maximize
h.changeObjectiveSense(highspy.ObjSense.kMaximize)

# Decision variables:
# t1 = jets allocated to training in year 1 (integer, 0..10) -> obj coeff = 5 (pilots per jet)
# t2 = jets allocated to training in year 2 (integer, 0..15) -> obj coeff = 0 (not ready by year 2)
# addVariable(lb, ub, obj, vtype, name)
t1_var = h.addVariable(
    0,           # lb
    A1,          # ub
    PILOTS_PER_JET_YEAR,  # obj coefficient (5 pilots per jet)
    highspy.HighsVarType.kInteger,  # vtype
    "t1"         # name
)

t2_var = h.addVariable(
    0,           # lb
    A2,          # ub
    0,           # obj coefficient (0 - year 2 cohort not ready by year 2)
    highspy.HighsVarType.kInteger,  # vtype
    "t2"         # name
)

# Constraint: Year 1 jets must all train (t1 >= A1)
# Since year 1 combat jets need pilots and no year 0 pilots exist, all year 1 jets must train
h.addConstr(t1_var >= A1, "year1_must_train")

# Solve
h.setOptionValue("mip_rel_gap", 0)
h.setOptionValue("mip_abs_gap", 0)
h.setOptionValue("time_limit", 60)
h.run()

# Get solution
sol = h.getSolution()
status = h.getModelStatus()
runtime = h.getRunTime()

# Extract values
t1_val = sol.col_value[t1_var.index]
t2_val = sol.col_value[t2_var.index]

# Objective: 5 * t1 (pilots from year 1 cohort ready by end of year 2)
objective_value = PILOTS_PER_JET_YEAR * t1_val

# Year 2 cohort not ready yet (training takes 2 years)
# By end of year 2, only year 1 cohort is fully trained
fully_trained = PILOTS_PER_JET_YEAR * t1_val

# Total pilot-years by end of year 2:
# Year 1 jets: t1 * 5 pilots/year * 2 years
# Year 2 jets: t2 * 5 * 1 year (half-trained)
total_pilot_years = PILOTS_PER_JET_YEAR * (TRAINING_YEARS * t1_val + 1 * t2_val)

# Write result
result = {
    "status": "optimal" if status == highspy.HighsModelStatus.kOptimal else "suboptimal",
    "objective_value": objective_value,
    "objective_bound": objective_value,
    "runtime_seconds": runtime,
    "variables": {
        "t1": t1_val,
        "t2": t2_val,
        "fully_trained_pilots_by_end_year2": fully_trained,
        "pilot_years_by_end_year2": total_pilot_years,
        "note": f"Year 1 jets (t1={t1_val}) train for 2 years, producing {fully_trained:.0f} fully trained pilots by end of year 2. Year 2 jets (t2={t2_val}) started training in year 2, not yet fully trained."
    }
}

with open('result.json', 'w') as f:
    json.dump(result, f, indent=2)

print(f"Status: {result['status']}")
print(f"Objective (fully trained pilots by end of year 2): {objective_value:.0f}")
print(f"t1={t1_val:.0f}, t2={t2_val:.0f}")
print(f"Result written to result.json")
