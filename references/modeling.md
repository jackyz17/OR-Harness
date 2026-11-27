# Task inputs and model representation

Use `task.json` to carry the original requirements. Add `coupling` before planning when an explicit CIR helps describe the relationships that affect the solve. Add an optional GAMS-style `model` after selecting a method when a separate formulation helps you inspect the model before writing solver code. The framework checks the supplied model's syntax and symbols and reports `derivation.model_coupling` as a diagnostic. The original problem input determines the shared structural key.

## Start from the shared original input

The minimal task file is `{"task_id":"T","family":"assignment","text":"Actual original requirements and data"}`. Reuse `--task task.json` in both interactive and batch workflows. A structured `spec` is optional. If later commands must see the same CIR, store it in `task.json.coupling`; `profile --cir cir.json` affects that call, not every subsequent command.

Read `result.profile`, `result.derivation`, `result.coupling.modeling_guidance`, and `result.coupling.health` from `profile`. See [the command reference](commands.md#3-understand-and-retrieve) for profile and recall outputs. After execution, follow [the self-check step](../SKILL.md#3b-self-check-the-answer-then-decide-to-repair-re-solve-or-commit).

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

## Self-check inputs and the CIR's limits

The CIR is understanding-level material for modelling; it is NOT an
independent truth about the task. A relation or attribute derived from it
(including a scalar coupling value) can guide the model and the retrieval key,
but it cannot establish that a constraint is covered, that a variable is
integer, or that an answer is correct. Integer domains come from the TASK text
and the model's own declaration — never from a field name, a CIR node, or an
evidence count.

Before committing, use [the self-check flow](../SKILL.md#3b-self-check-the-answer-then-decide-to-repair-re-solve-or-commit):
read the original input, review the model and code, and choose any applicable
method (hand calculation, code review, an independent verification script,
enumeration, a bound or dual). `check.json` carries only the deterministic
checks the framework can compute (see [check-task](commands.md#5-execute-and-check));
it replaces neither the agent's judgment about which constraints matter nor an
optimality argument the task requires. An agent-run verification script and its
result can be attached to the same execution with `check-task --verification`
and stay a separate, agent-reported source.

## CLI profile input example

Read the pre-strategy structure and modelling guidance. CIR comes from `--cir` or `task.coupling`. Use lists under `entities`, `decisions`, `constraints`, `relations`, `coupling_groups` and `issues`. Name entities and decisions, give constraints IDs, and give coupling groups a `type`. Relation endpoints refer to declared nodes.

```json
{"coupling": {
  "entities": [{"name": "R1", "kind": "resource", "attrs": {}}],
  "decisions": [{"name": "x", "kind": "production", "indexes": ["i", "t"]}],
  "constraints": [{"id": "C1", "kind": "capacity", "expr": "sum(x) <= limit"}],
  "relations": [{"source": "x", "target": "R1", "type": "uses_resource",
                 "evidence": "semantic", "detail": "shared capacity"}],
  "coupling_groups": []
}}
```

Set relation evidence to `structural` for co-occurrence, `semantic` for a supported relationship, or `declared` for an agent assertion. Store scalar annotations in `annotations.coupling`.

**Read** `result.profile`, `result.derivation` and `result.coupling.{health,modeling_guidance}`. Derivation uses CIR, structured `spec`, then annotations. A supplied model's coupling appears in `derivation.model_coupling`.

**Fix input** For a `cir_format` error, read `result.error.hint` and correct the indicated field. Use `--allow-empty-cir` for an intentional empty CIR. See [CIR shape and strictness](modeling.md).

## File input and isolation

**Workspace** Keep the submitted script inside `--workspace`. Execution uses a
fresh isolated `/work`, with declared files at `inputs/<path>`, read-only
`solve.py`, writable `result.json` and `scratch/`. Existing host workspace
files are not exposed. Only this attempt's `result.json` is exported to
`--workspace/result.json`; scratch is temporary (64 MiB total). Use in-process
solver APIs. Network and subprocess launches remain blocked; PuLP/CBC and
Pyomo backends that launch a binary require an in-process alternative.
`pathlib` and dynamic paths are supported. The script has a wall-clock limit
in addition to its solver settings. Use labels `C1`, `C2` and so on for L2 checks.

Declare explicit regular files in the same task used for plan-next and execute:

```json
{
  "task_id":"network-example", "family":"routing",
  "text":"Solve the network specified by the declared nodes and edges.",
  "input_resources":[
    {"source":"data/nodes.csv","path":"network/nodes.csv"},
    {"source":"data/edges.csv","path":"network/edges.csv"},
    {"source":"data/config.json","path":"config.json"}
  ]
}
```

`source` is relative to the CLI task JSON file, or to the caller's cwd in the
Python API. Explicit absolute sources are also supported. `path` is a unique
relative POSIX file name under `inputs/`, without `..`, symlinks or directory
declarations. An optional full `sha256` pins expected content. The framework
streams each file into the existing content-addressed artifact store and
freezes its source, relative path, bytes and hash with the task/prediction.
Execution rechecks content and exposes only the matching files. Changed input
declarations require a new plan; missing/changed bytes at submission are saved
as a failed bound attempt. No file contents enter the model prompt or script.
The trusted Runner must supply the public-input list: no automatic attachment,
workspace or benchmark-directory scan includes reference answers, checkers,
other tasks or memory banks.

```python
from pathlib import Path
import csv, json

filename = "nodes.csv"
path = Path("inputs") / "network" / filename
with path.open() as f:
    nodes = list(csv.DictReader(f))
with open(Path("inputs") / "config.json") as f:
    config = json.load(f)
# Installed pandas.read_csv(path), numpy.loadtxt(path), and native solver
# readers can use the same authorized files. Write results from THIS solve.
```

CSV, JSON, TXT and arbitrary regular instance formats use this same interface;
the corresponding parser must be installed. Text-only tasks omit
`input_resources`. The Python API is unchanged:
`h.execute(task, prediction_id=P, code_path="ws/solve.py", workspace="ws")`.

Linux user/mount/network namespaces, `unshare` and `libseccomp.so.2` are required.
Inputs and runtime files are read-only mounts; the remaining jail is read-only
except `result.json`, bounded `scratch/` and log descriptors. Python auditing
adds path diagnostics, while mounts/chroot and syscall filters also constrain
native readers, aliases and third-party libraries. Runtime library directories,
the current process's maps/executable link and CPU metadata are exposed for
numerical runtimes; host `/proc`, fd directories, credentials and cwd are not.
Network, process launches, namespace changes and cross-process access are
blocked. CPU/address-space/file limits and whole-process-group timeout cleanup
remain active. Missing isolation fails closed; there is no AST-only fallback.
The host kernel and installed runtime are trusted. This is not a VM or metadata
secrecy guarantee, and concurrent host filesystem mutation is outside this
boundary. Run under a dedicated account/container for hostile multi-tenant use.
Keep task files, submission workspaces and Harness memory outside runtime
library directories; detected overlaps are rejected before launch. The Runner
must also keep other tasks' materials and secrets out of trusted distributions.

One selected method can include reading, LP relaxation, feasible-bound
construction, certified reduction and final MILP in the same execute. Keep
real-instance solving and code trials in formal attempts. Before planning,
inspect data/structure and use only tiny artificial API probes. See
[self-check](self_check.md) for post-execution validation and its cost/evidence.
