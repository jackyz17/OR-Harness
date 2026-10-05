import json
import time
import highspy

# --- Parameters ---
products = ['I', 'II', 'III']
quarters = ['Q1', 'Q2', 'Q3', 'Q4']
h = {'I': 2, 'II': 4, 'III': 3}
c_delay = {'I': 20, 'II': 20, 'III': 10}
c_hold = 5
cap = 15000
final_inv_req = 150

d = {
    ('I', 'Q1'): 1500, ('I', 'Q2'): 1000, ('I', 'Q3'): 2000, ('I', 'Q4'): 1200,
    ('II', 'Q1'): 1500, ('II', 'Q2'): 1500, ('II', 'Q3'): 1200, ('II', 'Q4'): 1500,
    ('III', 'Q1'): 1000, ('III', 'Q2'): 2000, ('III', 'Q3'): 1500, ('III', 'Q4'): 2500,
}
delay_weights = {'Q1': 4, 'Q2': 3, 'Q3': 2, 'Q4': 1}

# --- Build model ---
model = highspy.Highs()
model.setOptionValue('time_limit', 60)
model.setOptionValue('mip_rel_gap', 0)
model.setOptionValue('mip_abs_gap', 0)

# Decision variables: set objective at creation time
produce = {}
inventory = {}
backlog = {}
for p in products:
    for q in quarters:
        produce[p, q] = model.addVariable(
            0, float('inf'), 0.0,
            highspy.HighsVarType.kInteger, f'produce_{p}_{q}')
        inventory[p, q] = model.addVariable(
            0, float('inf'), c_hold,
            highspy.HighsVarType.kInteger, f'inventory_{p}_{q}')
        backlog[p, q] = model.addVariable(
            0, float('inf'), float(c_delay[p] * delay_weights[q]),
            highspy.HighsVarType.kInteger, f'backlog_{p}_{q}')

# Capacity constraints per quarter
for q in quarters:
    model.addConstr(h['I'] * produce['I', q] +
                    h['II'] * produce['II', q] +
                    h['III'] * produce['III', q] <= cap,
                    f'C_cap_{q}')

# Product I cannot be produced in Q2
model.addConstr(produce['I', 'Q2'] == 0, 'C_no_I_Q2')

# Inventory balance and backlog constraints
for p in products:
    # Q1: inv = produce - demand + backlog; backlog >= demand - produce
    model.addConstr(inventory[p, 'Q1'] == produce[p, 'Q1'] - d[p, 'Q1'] + backlog[p, 'Q1'],
                    f'C_bal_{p}_Q1')
    model.addConstr(backlog[p, 'Q1'] >= d[p, 'Q1'] - produce[p, 'Q1'], f'C_bl_{p}_Q1')

    # Q2
    model.addConstr(inventory[p, 'Q2'] == inventory[p, 'Q1'] + produce[p, 'Q2']
                    - d[p, 'Q2'] + backlog[p, 'Q1'] - backlog[p, 'Q2'], f'C_bal_{p}_Q2')
    model.addConstr(backlog[p, 'Q2'] >= backlog[p, 'Q1'] + d[p, 'Q1'] - produce[p, 'Q1']
                    - inventory[p, 'Q2'], f'C_bl_{p}_Q2')

    # Q3
    model.addConstr(inventory[p, 'Q3'] == inventory[p, 'Q2'] + produce[p, 'Q3']
                    - d[p, 'Q3'] + backlog[p, 'Q2'] - backlog[p, 'Q3'], f'C_bal_{p}_Q3')
    model.addConstr(backlog[p, 'Q3'] >= backlog[p, 'Q2'] + d[p, 'Q2'] - produce[p, 'Q2']
                    - inventory[p, 'Q3'], f'C_bl_{p}_Q3')

    # Q4
    model.addConstr(inventory[p, 'Q4'] == inventory[p, 'Q3'] + produce[p, 'Q4']
                    - d[p, 'Q4'] + backlog[p, 'Q3'] - backlog[p, 'Q4'], f'C_bal_{p}_Q4')
    model.addConstr(backlog[p, 'Q4'] >= backlog[p, 'Q3'] + d[p, 'Q3'] - produce[p, 'Q3']
                    - inventory[p, 'Q4'], f'C_bl_{p}_Q4')

# Final inventory >= 150
for p in products:
    model.addConstr(inventory[p, 'Q4'] >= final_inv_req, f'C_final_{p}')

model.changeObjectiveSense(highspy.ObjSense.kMinimize)

# Solve
t0 = time.time()
model.run()
t1 = time.time()
runtime = t1 - t0

sol = model.getSolution()
ms = model.getModelStatus()
st = ms.name
info = model.getInfo()

if 'optimal' in st.lower() or st == 'kOptimal':
    objective = model.getObjectiveValue()
    mip_gap = float(info.mip_gap) if hasattr(info, 'mip_gap') else 0.0
    obj_bound = float(info.mip_dual_bound) if hasattr(info, 'mip_dual_bound') else objective

    vars_out = {}
    for p in products:
        for q in quarters:
            vars_out[f'produce_{p}_{q}'] = produce[p, q].index
            vars_out[f'inventory_{p}_{q}'] = inventory[p, q].index
            vars_out[f'backlog_{p}_{q}'] = backlog[p, q].index

    result = {
        'status': 'optimal',
        'objective_value': objective,
        'objective_bound': obj_bound,
        'mip_gap': mip_gap,
        'runtime_seconds': runtime,
        'variables': {name: float(sol.col_value[idx]) for name, idx in vars_out.items()}
    }
else:
    result = {
        'status': st,
        'objective_value': None,
        'objective_bound': None,
        'mip_gap': None,
        'runtime_seconds': runtime,
        'variables': {}
    }

with open('result.json', 'w') as f:
    json.dump(result, f)
