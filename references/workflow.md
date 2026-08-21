# Workflow

## Contents

1. Runtime layout
2. End-to-end flow
3. Experience layers
4. Branch isolation and repair
5. Extraction and append-only rules
6. Quick start

## Runtime layout

Runtime data is outside the installed Skill. `OR_EXPERIENCE_BANK_HOME` overrides the default `~/.hermes/or-experience-bank`.

```text
bank/                 immutable JSONL facts: one file per flat layer + modeling_bank.jsonl
episodes/             problem-level scene snapshots (append-only, two-phase)
index/                rebuildable embedding vectors and metadata
trajectories/         append-only AttemptRecord evidence
runs/                 isolated generated-code branch workspaces
logs/                 optional operational logs
```

Derived indexes (embedding index, the repair error-transition graph) are rebuildable from facts at any time; facts are the only source of truth.

## End-to-end flow

The named mode is **Heterogeneous Parallel Solver Exploration with Intra-Branch Sequential Repair**, front-loaded by a verified Structured Modeling stage and closed by gold-gated comparative synthesis.

1. Normalize the user problem into family, objective, entities, and constraints.
2. **Structured Modeling (before any branch)**: the agent emits `<think>` + `<model>` in the GAMS-style DSL (SETS/PARAMETERS/VARIABLES/OBJECTIVE/CONSTRAINTS, symbolic indexing, inline set members). `ModelingGate` runs L1 format, L2 structural (declared-vs-referenced symbols, signature consistency), and optional L3 semantic judge. Failure feeds issues back and re-models (≤3 rounds). **No branch is created until the model passes.**
3. Extract a **structural signature** (core dims O/D/C/I + open feature slots) from the verified model.
4. Build a formulation-stage query and retrieve Modeling Bank vectors.
5. Detect requested solver availability. Missing modules and license failures are environment outcomes, not modeling errors.
6. Create one isolated branch workspace per available solver. All branches share the verified model.
7. Retrieve solver-scoped Implementation Bank records and generate complete code that implements the verified model.
8. Execute with a fixed Python executable, argument list, environment allowlist, timeout, output limits, and Unix resource limits.
9. Require the code to write `result.json`. Capture missing/invalid results as execution feedback; record the failure into the per-solve `FailureBuffer`.
10. Validate runtime, schema, solver status, numeric objective, variables, reference objective, and optional semantic validator.
11. On a repairable failure, normalize feedback, consult the **Repair error-transition graph** (rebuilt on demand) plus Repair Bank records, summarize only current state, and generate the next complete code version.
12. Stop on accepted success, explicit terminal status, repeated normalized error, unchanged code, timeout, environment failure, or maximum attempts.
13. After all branches finish, compare validation and only compare objectives when senses and formulations are comparable. Record a base **Episode** (model, signature, branch outcomes, failure count).
14. **Gold evaluation (separate step, Option A)**: once the gold answer arrives, compare the selected branch. On mismatch, append an Episode supplement and drive an **outer reflection** back to Structured Modeling (≤3 rounds). On match, run **comparative synthesis** over success + buffered failures, admit synthesized lessons through the judge, route them to the right bank (modeling → `ModelingStore`, others → flat store), and append the Episode gold supplement.

Branches cannot read sibling branch paths or trajectories through the orchestrator. The local executor does not provide a kernel-level filesystem or network sandbox; run production generated code inside an OS/container sandbox when stronger isolation is required.

## Experience layers

- **Modeling Bank**: solver-independent, general modeling methods and math techniques. `ModelingExperience` schema (all records are peers; `modeling_aspect` classifies target), own append-only store, carries the structural signature.
- **Implementation Bank**: language and solver API mechanics.
- **Repair Bank**: normalized error, diagnosis, successful repair, or ineffective action; feeds the derived error-transition graph.
- **Solving Bank**: timeouts, gaps, bounds, numerics, parameters, scale, and solver choice.
- **Episode Store**: problem-level snapshots (not a retrieval layer); raw material for offline induction and provenance target.

Retrieval uses embedding vectors plus cosine similarity after metadata hard filtering. Solver-specific records cannot cross solvers; solver-family records remain inside a family; solver-agnostic records may cross solvers. Repair additionally uses the error-transition graph with generality-gated migration.

## Branch repair context

Attempt 1 receives the normalized problem, the **verified model**, Modeling hits, Implementation hits, solver context, output contract, and execution limits. Later attempts receive the latest formulation, latest complete code, latest feedback, resolved/unresolved issues, ineffective repairs, Repair graph guidance, and Repair/Solving hits. Full historic conversations and unbounded logs are not repeated.

## Extraction and append-only rules

Trajectory is evidence; experience is a reusable atomic conclusion. **Failure experiences are never appended alone** — they are buffered, then contrasted with the eventual success by a synthesis step; only synthesized, judge-admitted lessons are appended. Positive prescriptive experience requires solver-feasible evidence or error-before/success-after evidence. Negative experience identifies a concrete action to avoid and needs repeated failure, a proven alternative, or explicit solver rejection.

Existing JSONL lines are never rewritten. Corrections are new records linked through `related_experience_ids`, `derived_from_experience_ids`, `contradicts_experience_ids`, or `possible_duplicate_of`. Exact content hashes are rejected as duplicates. Embedding indexes and the repair graph are derived data and may be rebuilt.

## Quick start

```bash
export OR_EXPERIENCE_BANK_HOME=/tmp/or-experience-demo
python3 scripts/or_experience_cli.py solve \
  --problem "Assign three tasks to machines and minimize total cost" \
  --mock-demo --solvers mock-a,mock-b --max-attempts 3 --json
python3 scripts/or_experience_cli.py stats --json
python3 scripts/or_experience_cli.py retrieve \
  --layer repair --solver mock-a \
  --query "invalid linear expression repaired in next attempt" --json
```

The mock demonstration intentionally makes one branch fail then succeed and another succeed immediately. It writes experience that the final retrieve command can see. Mock results are not real optimization results.
