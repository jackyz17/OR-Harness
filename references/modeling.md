# The model representation

Before writing solver code, write the problem as a GAMS-style model representation and carry it in the task JSON's top-level `model` field. The framework verifies it deterministically (no LLM) and measures structural coupling directly from the declared constraints — the single best coupling source the profiler has. This is a convention, not a requirement: everything works without it, but every later step (profiling, grouping, strategy selection) gets weaker.

## Why before code

The model representation is the cheapest place to catch structural errors. An undeclared symbol or a missing constraint found here costs one revision; found in solver code it costs an execution failure, a retry (retries are a CostVector dimension), and a polluted record. For highly coupled problems — where this framework earns its keep — modeling errors are the most expensive kind.

## Syntax (five blocks)

Headers are case-insensitive, with an optional trailing colon. `#` comment lines are ignored.

```
SETS:
 i in Projects = {p0, p1}
 j in Resources = {r0, r1}
PARAMETERS:
 E[i,j]
 C[i,j]
 alpha
 limit[j]
VARIABLES:
 x[i,j] continuous >= 0
 y[i] binary
OBJECTIVE:
 maximize sum(i, sum(j, (E[i,j] - alpha * C[i,j]) * x[i,j]))
CONSTRAINTS:
 C1: sum(i, x[i,j]) <= limit[j]
 C2: sum(j, C[i,j] * x[i,j]) <= budget
```

- **SETS**: `index in Name = {members}` — members become resolvable literal indices.
- **PARAMETERS**: `name[index-expr]` — declared for symbol cross-reference.
- **VARIABLES**: `name[index-expr] type` — type is `binary`, `integer`, or `continuous` (default).
- **OBJECTIVE**: one line, `maximize`/`minimize` + expression. `sum(index, expr)` is summation notation.
- **CONSTRAINTS**: one per line, `Cn: expression`.

## Constraint label rules (critical)

Labels MUST be `C1`, `C2`, `C3`, ... — anything else is rejected by L2. This is the single most common verification failure.

## Verification layers (deterministic, no LLM)

**L1 — format**: all five blocks present and non-empty.

**L2 — symbol cross-reference**: every symbol referenced in OBJECTIVE/CONSTRAINTS must be declared in SETS/PARAMETERS/VARIABLES (or be a reserved token: `sum`/`prod` notation, math keywords and functions, `Cn` labels, single-letter index variables inside brackets).

Issues come back as `{layer, code, detail}` — e.g. `[L2] undefined_symbol: 'budget' is referenced in C2 but not declared`. Fix them and resubmit the task JSON; the framework never loops or regenerates on its own.

## What the framework derives from it

- **resource_coupling** = fraction of variables appearing in more than one constraint. In the example above, `x` appears in both C1 and C2 → rc = 1.0. Drop C2 and rc = 0.0 (constraints independent).
- **temporal_coupling** = fraction of variables indexed by a temporal set (name matching time/period/stage/day/hour/week/month).
- **route_complexity** = fraction of variables indexed by a network set (arc/edge/road/link/route/leg).
- **semantic_coupling** is NOT derived — business semantics are invisible to structure; supply it via `annotations.coupling.semantic_coupling` if you want it grouped on.

If you also supply coupling values and they contradict the derivation across a bin boundary, `orx profile` returns `coupling_warnings` — the structural measurement wins for grouping; your original values are preserved in annotations.
