# The model representation

Read this page when you are writing your first model or CIR: the five-block syntax, the constraint-label rules, what the framework derives, and what it refuses.

The model representation is the intermediate artifact between **choosing a strategy** and **writing solver code**: once the strategy is decided, write the problem as a GAMS-style model representation and carry it in the task JSON's top-level `model` field, then translate it into solve.py. The framework verifies it deterministically (no LLM) and reports what its declared constraints say about structural coupling as a DIAGNOSTIC (`derivation.model_coupling`). It is NOT a prerequisite for strategy selection: choosing a strategy relies on the task text, the CIR, the profile, and the evidence's expected quality/cost/risk. Because the model is a POST-strategy artifact, it never moves the structural key — the profile you retrieved with stays the profile the execution is filed under.

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
- **semantic_coupling** is NOT derived — business semantics are invisible to structure; supply it via `annotations.coupling.semantic_coupling` to keep it in the profile. It is understanding-level information: it never enters grouping keys or entry predicates (only the derived dimensions `resource_coupling` / `temporal_coupling` / `route_complexity` condition the statistics, so an unverifiable hand-typed number can never split the evidence or block a match).

If you also supply coupling values and they contradict the derivation across a bin boundary, `orx profile` returns `coupling_warnings` — the structural measurement wins for grouping; your original values are preserved in annotations.

> The former mechanism feature layer (four scalar WHY-dimensions) has been
> superseded by the CIR (below), which represents coupling structure
> explicitly rather than compressing it into scalars.

## Coupling-Aware Intermediate Representation (CIR)

The CIR is an explicit, inspectable structured representation of *how* the components of an optimization problem interact. It is produced **before** the canonical model — from the natural-language task description — and provides modeling guidance that improves the correctness of the canonical representation. Scalar coupling scores (the four dimensions above) are derived summaries; the CIR is the primary representation.

### When to use it

Submit the CIR via `orx profile --task t.json` (the single analysis entry) **before choosing a strategy** — it is the understanding artifact that improves both the profile derivation and the model you write later. The CIR lives in the task JSON's optional `coupling` field and never requires a `model` field. After the model is written, `orx profile` reports its structure as a diagnostic (`derivation.model_coupling`) — useful to sanity-check your formulation, but it does not change the key.

### Schema (domain-general — all `kind`/`type` fields are free-form strings)

| Section | Fields | Notes |
|---|---|---|
| `entities` | `name, kind, attrs` | `kind` examples: resource, product, site, period, route, … (never enumerated) |
| `decisions` | `name, kind, indexes, attrs` | `kind` examples: production, allocation, routing, inventory, … |
| `constraints` | `id, kind, expr, attrs` | `kind` examples: capacity, balance, temporal, precedence, logical, global, … |
| `relations` | `source, target, type, evidence, detail` | `type` examples: uses_resource, shares_resource, competes_for, precedes, flows_to, depends_on, constrained_by, … |
| `coupling_groups` | `type, members, resource, implication` | `type` examples: shared_bottleneck, route_convergence, temporal_propagation_chain, global_constraint, cross_stage_coupling, … |
| `issues` | `layer, code, detail` | Validation report; written by the framework, accepted on input for round-trip |

These six keys are the ONLY allowed ones. A `coupling` object carrying anything else is rejected at `orx profile` (exit 2) with a structured `error` naming the offending key — because the alternative is worse than an error: an unrecognised key parses to an EMPTY CIR, and every consumer downstream then treats a malformed problem as an uncoupled one.

### Shape contract (fail-closed)

| Failure | `error.cause` | Fix |
|---|---|---|
| `"coupling": {"cir": {...}}` (nested) | `unknown_keys` | Put the keys directly under `coupling` |
| `"entites": [...]` (typo) | `unknown_keys` | Allowed keys are the six above |
| `"entities": {...}` (dict, not list) | `bad_type` | Use a list |
| `"coupling": {}` (empty) | `empty_cir` | Fill it in, drop the field, or pass `--allow-empty-cir` |
| `"coupling": {"resource_coupling": 0.9}` (scalar) | `unknown_keys` | Scalars belong in `annotations.coupling` |
| `"coupling": "text"` (not an object) | `not_object` | Wrap it in an object |

Every rejection carries a `hint`, and the two mistakes that actually happen — nesting and scalar confusion — get a targeted one. `OR_CIR_STRICT=0` downgrades the policy checks (`unknown_keys`, `empty_cir`) for a legacy task, reporting them in `result.coupling.shape_lints` instead of failing; structural problems (`not_object`, `bad_type`) are never downgraded, because a payload that cannot be deserialized has no lenient reading.

### Health (what the CIR actually contributed)

`result.coupling.health` reports `{present, parsed, entities, decisions, constraints, relations, issues, contributes_scalars, note}`. `contributes_scalars` is false when the CIR carries no decisions — and a CIR with no decisions derives NO scalar dimension, so a profile whose coupling came from the CIR can silently have come from nowhere. Read `health.note` rather than assuming a present CIR means a structural signal.

### Evidence levels (critical)

| Level | Meaning |
|---|---|
| `structural` | Derived from constraint-variable co-occurrence only. **Never** a semantic claim. Produces generic `depends_on` edges. |
| `semantic` | Supported by entity/constraint semantics (e.g. resource-kind entity + capacity constraint → `uses_resource`). The only semantic upgrade performed deterministically. |
| `declared` | Directly asserted by the agent. Subject to L2 referential validation. |

Co-occurrence is structural evidence only. A semantic relation (`shares_resource`, `competes_for`, …) requires additional entity/constraint semantics; otherwise the edge stays as `depends_on`.

### Validation (deterministic, no LLM)

- **L1 — format**: required fields present, no duplicate node names, valid evidence levels.
- **L2 — referential integrity**: every relation source/target and coupling-group member/resource must resolve to a known entity, decision, or constraint.

### What the framework derives

- **ProblemSignature scalars**: when a CIR is present, `resource_coupling` = fraction of decisions that are the source of ≥1 resource relation (`uses_resource`/`shares_resource`/`competes_for`); `temporal_coupling` / `route_complexity` = fraction of decisions with time-like/network-like indexes. Derivation priority: CIR > spec > supplied.
- **Structural relations**: from the `model` field's constraint-variable co-occurrence → `depends_on` edges (evidence=`structural`).
- **Semantic upgrade**: a structural edge upgrades to `uses_resource` ONLY when the target entity is resource-like AND a capacity-ish constraint mentions BOTH the source decision AND the target resource. Co-occurrence alone never produces semantic relations; capacity-ish words are deliberately narrow (`capacity`, `limit`, `cap`, `budget`, `resource`, `available` — not `demand`/`max`).
- **Coupling groups**: `shared_bottleneck` (≥2 decisions using the same resource) is the only pattern derived deterministically. Other group types are the agent's responsibility — declared by the agent, validated and rendered by the framework.
- **Modeling guidance**: each coupling group renders an explicit implication (e.g. "ensure one aggregate capacity constraint covers all relevant decisions").

The `model` field is verified independently (L1/L2 above) and its own structural reading is reported as `derivation.model_coupling` — a DIAGNOSTIC. It is not cross-checked against the CIR: the CIR is the pre-strategy understanding and the model is a post-strategy artifact, so they are reported side by side for you to compare rather than reconciled automatically.
