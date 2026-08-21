# Structural Signature

## Contents

1. What it is
2. Controlled vocabularies (core four dimensions)
3. Open feature slots
4. Alignment rules
5. Examples
6. math_type derivation

## What it is

A structural signature is a compact fingerprint of an optimization model's mathematical skeleton. It captures the **structure** (not the domain content) so that heterogeneous problems sharing the same mathematical form can be discovered for offline induction.

A signature has four core dimensions (O/D/C/I) with controlled vocabularies, plus an open `features` slot for problem-specific characteristics. The signature is carried by every Modeling Bank record (realization and pattern) and every Episode.

Design decision (**D9**): the core four dimensions are stable and universal — they are never `none`. Open feature slots allow growth without schema changes.

## Controlled vocabularies (core four dimensions)

### O — Objective Structure (single value)

| Value | Meaning | Example |
|---|---|---|
| `linear` | Linear objective function | `maximize 400*cows + 120*sheep` |
| `convex` | Convex nonlinear objective | `minimize sqrt(x^2 + y^2)` |
| `minmax` | Minimize a maximum (bottleneck) | `minimize max(completion_time[i])` |
| `multi_objective_weighted` | Weighted sum of multiple objectives | `minimize 0.7*cost + 0.3*lateness` |
| `feasibility_only` | No optimization, just satisfy constraints | Find a feasible schedule |

### D — Decision Structure (list, multi-valued)

| Value | Meaning | Example |
|---|---|---|
| `binary_assignment` | Binary 0/1 assignment variables | `x[i,j] ∈ {0,1}` — assign task i to machine j |
| `integer_batch` | Integer quantities/batch sizes | `x[i] ∈ Z+` — number of items to produce |
| `continuous_flow` | Continuous flow/quantity variables | `f[i,j] ≥ 0` — flow on edge (i,j) |
| `multi_index_2d` | 2D indexed variables (any type) | `x[i,t]` — production of item i in period t |
| `multi_index_3d` | 3D+ indexed variables (any type) | `x[i,j,t]` — route of vehicle i to node j at time t |

A model can have multiple decision types (e.g. `["binary_assignment", "integer_batch"]`).

### C — Constraint Structure (list, multi-valued)

| Value | Meaning | Example |
|---|---|---|
| `capacity` | Resource capacity limit | `sum(a, manure_rate[a] * x[a]) <= 800` |
| `flow_conservation` | Flow in = flow out at nodes | `sum(in_edges) f[i,j] = sum(out_edges) f[j,k]` |
| `assignment_exactly_once` | Each item assigned exactly once | `sum(j, x[i,j]) = 1` for all i |
| `covering` | Minimum coverage requirement | `sum(j, covers[i,j] * x[j]) >= 1` for all i |
| `precedence` | Ordering/temporal precedence | `start[i] + duration[i] <= start[j]` |
| `big_m_linking` | Big-M linking constraint | `x[i,j] <= M * y[j]` |

A model typically has multiple constraint types.

### I — Interaction/Coupling (single value)

| Value | Meaning | Example |
|---|---|---|
| `independent` | Decisions don't interact (decomposable) | Separate single-variable problems |
| `shared_resource_coupled` | Decisions coupled via shared scarce resource | Animals share manure capacity |
| `fixed_charge_coupling` | Fixed-charge: binary activates continuous cost | `y=1 → x>0; y=0 → x=0` |
| `nonlinear_interaction` | Nonlinear coupling between decisions | `x*y` products, SOC constraints |

## Open feature slots

The `features` field is a free-form `Dict[str, str]`. Any descriptive key-value pair may be included. Missing keys are not penalized during alignment.

**Recommended keys** (non-binding, periodically adopted from high-frequency usage):

| Key | Example values | When to use |
|---|---|---|
| `temporal` | `multi_period_balance`, `rolling_horizon` | Multi-period problems |
| `network` | `path_on_graph`, `tree_structure` | Network/graph problems |
| `resource` | `shared_scarce`, `multi_resource` | Resource allocation |
| `uncertainty` | `scenario_tree`, `robust_box` | Stochastic/robust optimization |

You may invent new keys (e.g. `"domain": "livestock_farm"`). High-frequency keys get adopted into the recommendation list over time.

## Alignment rules

When the induction pipeline compares signatures for cluster discovery:

1. **Core dims**: O must match exactly; I must match exactly; D and C use subset/intersection matching (a cluster forms if the core dims agree on at least one value per multi-valued dimension).
2. **Features**: match on the **intersection** of keys — if record A has `{"temporal": "..."}` and record B has `{"resource": "..."}`, they match on features (no conflict, no penalty). If both have `"temporal"`, the values must match.
3. **A member with no features** joins any group — missing is not penalized.
4. **Heterogeneity requirement**: a cluster is valid only if its members span ≥2 `problem_family` values (e.g. `assignment` + `scheduling` + `production_planning`).

## Examples

### Example 1: Farmer livestock allocation

```json
{
  "objective": "linear",
  "decision": ["integer_batch"],
  "constraint": ["capacity", "covering"],
  "interaction": "shared_resource_coupled",
  "features": {"resource": "shared_scarce", "domain": "livestock_farm"}
}
```

### Example 2: Vehicle routing (CVRP)

```json
{
  "objective": "linear",
  "decision": ["binary_assignment", "multi_index_2d"],
  "constraint": ["capacity", "flow_conservation", "assignment_exactly_once"],
  "interaction": "shared_resource_coupled",
  "features": {"network": "path_on_graph", "resource": "shared_scarce"}
}
```

### Example 3: Job-shop scheduling

```json
{
  "objective": "minmax",
  "decision": ["binary_assignment", "multi_index_2d"],
  "constraint": ["precedence", "assignment_exactly_once"],
  "interaction": "shared_resource_coupled",
  "features": {"temporal": "machine_sequence"}
}
```

### Example 4: Multi-period production planning

```json
{
  "objective": "linear",
  "decision": ["integer_batch", "continuous_flow"],
  "constraint": ["capacity", "flow_conservation"],
  "interaction": "shared_resource_coupled",
  "features": {"temporal": "multi_period_balance"}
}
```

**Induction discovery**: Examples 1-4 share `objective=linear` (1,4) or `interaction=shared_resource_coupled` (all four), but the cross-family cluster forms because they all share the `capacity` constraint + `shared_resource_coupled` interaction despite spanning 4 different problem families (livestock / routing / scheduling / production).

## math_type derivation

`math_type` is NOT a stored field. It is derived from the signature as a one-line summary:

```
f"{D}+{I} problem, {O} objective"
```

Example: `"integer_batch+shared_resource_coupled problem, linear objective"`

This avoids maintaining a separate type field that could drift out of sync with the signature.
