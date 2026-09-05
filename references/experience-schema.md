# Experience Schema

## Contents

1. Source-of-truth contract
2. ExperienceRecord (flat layers: implementation / repair / solving)
3. Scope rules
4. Positive and negative examples
5. Duplicate behavior
6. Modeling Bank: fused schema (separate store)
7. Structural signature
8. Episode schema (problem-level snapshot)

## Source-of-truth contract

Each flat layer is an append-only JSONL file. One complete JSON object occupies one line. An accepted record never changes. Mutable counts, utility, Q-values, use timestamps, and dynamic confidence are forbidden. The Modeling Bank uses the fused `ModelingExperience` schema in its own store (`bank/modeling_bank.jsonl`); the same append-only rule applies.

## ExperienceRecord

```json
{
  "schema_version": "1.0",
  "experience_id": "exp_<uuid>",
  "created_at": "ISO-8601 UTC",
  "layer": "modeling|implementation|repair|solving",
  "polarity": "positive|negative",
  "title": "independent short title",
  "retrieval_text": "self-contained embedding text",
  "problem_context": {
    "problem_family": "cvrp|scheduling|assignment|general_milp|...",
    "objective_type": "minimize|maximize|feasibility|unknown",
    "stage": "formulation|implementation|repair|solving",
    "keywords": []
  },
  "scope": {
    "generality": "solver_agnostic|solver_family|solver_specific",
    "solver_family": "milp|cp_sat|cp|unknown|null",
    "solver": "gurobi|scip|ortools|null",
    "language": "python|null",
    "api": "gurobipy|pyscipopt|ortools.cp_model|null"
  },
  "trigger": {
    "situation": "when it applies",
    "normalized_error": null,
    "solver_status": null,
    "performance_symptom": null
  },
  "policy": {
    "diagnosis": "cause",
    "action": "one atomic action",
    "rationale": "why",
    "example": null,
    "limitations": null
  },
  "evidence": {
    "problem_id": "prob_...",
    "branch_ids": [],
    "attempt_ids": [],
    "solver_feedback_summary": "short evidence",
    "validation_level": "unverified|runtime_only|solver_feasible|semantic_checked",  # cross_solver_consistent is legacy (pre single-solver era), kept for old records
    "causal_confidence": "low|medium|high"
  },
  "related_experience_ids": [],
  "derived_from_experience_ids": [],
  "contradicts_experience_ids": [],
  "possible_duplicate_of": null,
  "content_hash": "sha256"
}
```

## Scope rules

- `solver_specific` requires `scope.solver`; it may name a proprietary API.
- `solver_family` requires `scope.solver_family` and is visible only inside that family.
- `solver_agnostic` must not name a solver API and may be retrieved by every solver.
- License errors remain environment/repair observations and must not be generalized as modeling limitations.

## Examples

Positive solver-specific action: “For OR-Tools CP-SAT, scale decimal coefficients to integers before constructing `LinearExpr`.” This belongs to Implementation Bank with `solver=ortools`, `api=ortools.cp_model`.

Negative solver-agnostic action: “Avoid replacing flow conservation equality with `<=`; it permits routes to terminate at intermediate nodes.” This belongs to Modeling Bank and needs a failing semantic check or cross-attempt correction as evidence.

Rejected vague experience: “Carefully check constraints.” It does not name an atomic action or reusable decision.

## Duplicate behavior

Canonical content excludes only generated identity/time/hash fields and `possible_duplicate_of`. Its SHA-256 is checked while holding the append lock. Exact duplicates return `status=duplicate` and do not modify the prior line. Near duplicates may be appended with `possible_duplicate_of`; they are not merged or reranked.

## Modeling Bank: fused schema (separate store)

The Modeling Bank does NOT use the flat `ExperienceRecord` above. It uses the `ModelingExperience` schema and lives in its own append-only store (`bank/modeling_bank.jsonl`). All records are peers — directly-solved records (`status=null`) and induced records (`status=validated`) coexist without a hierarchy. Each record carries a required `modeling_aspect` (one of: constraint, objective, variable, classification, structure) classifying which part of the model the experience targets.

```json
{
  "layer": "modeling",
  "experience_id": "exp_<uuid>",
  "title": "general modeling method / math technique",
  "polarity": "positive|negative",
  "retrieval_text": "self-contained embedding text",
  "math_scope": {
    "structural_signature": {
      "objective": "linear|convex|minmax|multi_objective_weighted|feasibility_only",
      "decision": ["binary_assignment|integer_batch|continuous_flow|multi_index_2d|multi_index_3d"],
      "constraint": ["capacity|flow_conservation|assignment_exactly_once|covering|precedence|big_m_linking"],
      "interaction": "independent|shared_resource_coupled|fixed_charge_coupling|nonlinear_interaction",
      "features": {"<open_key>": "<value>"}
    },
    "exclusions": ["when this method does NOT apply"]
  },
  "method": {
    "action_template": "parameterizable correct form",
    "wrong_form": "typical wrong form (same record, not a separate negative)",
    "rationale": "why it works",
    "derivation_ref": "textbook / paper / formula"
  },
  "evidence": {
    "source_episodes": ["prob_..."],
    "solver_feedback_summary": "short evidence",
    "validation_level": "solver_feasible|...",  # cross_solver_consistent is legacy
    "causal_confidence": "low|medium|high"
  },
  "derived_from_experience_ids": [],
  "contradicts_experience_ids": [],

  "modeling_aspect": "constraint",
  "created_at": "ISO-8601 UTC",
  "content_hash": "sha256",
  "role_schema": {},
  "role_mappings": [],
  "applicability_conditions": [],
  "counterexamples": [],
  "validation": {"source_consistency": "", "transfer_tests": []},
  "scoring": {"coverage": 0, "transferability": 0, "validation": 0, "novelty": 0, "complexity": 0, "counterexample_penalty": 0, "total": 0},
  "status": null
}
```

All records hold a GENERAL, reusable method (no concrete scenario parameters — those live in the Episode). Induced records (produced by offline induction) additionally populate `role_schema`, `role_mappings`, `applicability_conditions`, `counterexamples`, `validation`, `scoring`, and set `status` to `validated` (or `refuted`, but refuted records are never appended to the bank). Directly-solved records have `status=null` and empty induction fields.

### Structural signature

The signature is core four dims (objective/decision/constraint/interaction) plus an open `features` slot. Core values come from controlled vocabularies; `features` keys are open (e.g. `temporal`, `network`, `resource`, `uncertainty`). Alignment matches on core dims and on the intersection of feature keys; missing keys are not penalized. `math_type` is derived from the signature, not stored.

## Episode schema (problem-level snapshot)

Episodes are problem-level scene snapshots at `episodes/episodes.jsonl`, distinct from attempt-level trajectories. Two-phase append: a `base` record after solving, a `gold_supplement` after the gold verdict. Base record:

```json
{
  "record_kind": "base",
  "episode_id": "ep_<id>",
  "problem_id": "prob_...",
  "problem": "original text",
  "normalized_spec": {"problem_family": "...", "objective": "...", "verified_model": "...", "status": "success|failed", "failure_count": 0},
  "structural_signature": {},
  "branches": [{"solver": "gurobi", "status": "optimal", "attempts": 1, "objective_value": 7.0, "termination_reason": "optimal"}],
  "final_objective": 7.0,
  "gold_answer": null,
  "produced_realization_ids": [],
  "created_at": "ISO-8601 UTC"
}
```

The supplement carries `record_kind: "gold_supplement"`, `problem_id`, `gold_answer`, `matched`, and `produced_realization_ids`. Neither record is ever edited; the supplement links by `problem_id`.

## Induction trigger sidecar (mutable-stats, append-only)

A "sidecar" is a small companion file that lives NEXT TO the fact store, never inside it: the fact store stays append-only (a written line never changes), while mutable counters go in the sidecar (附属统计文件——专记会变的统计数字，不写进主库那一行).

The offline induction trigger keeps its own mutable-stats sidecar at `bank/induction_trigger_log.jsonl` (decision D2: a sidecar, never a modification of the fact layer). Each induction run appends one line recording the watermark and the cluster-membership snapshot used for cooldown:

```json
{"realization_count": 12, "cluster_signatures": ["linear|binary_assignment|capacity|shared_resource_coupled@exp_a,exp_b,exp_c"]}
```

`realization_count` is the accumulation watermark — a "last time we ran, the bank had this many realizations" marker (新增计数基线). The v1 trigger fires only when the current count minus this baseline reaches N new realizations. `cluster_signatures` fingerprints each induced cluster's members so an unchanged cluster is not re-induced. This file is read to decide whether to induce and is never consulted by online retrieval.

## Lifecycle state, utility stats, and the cold archive (Phase 2.3)

The bank evolves: new experiences arrive, induction distills new patterns, and some experiences turn out to be useless or harmful. They are retired WITHOUT editing or physically deleting their content (the audit chain — content-hash dedup, Episode provenance, induction `derived_from` — depends on immutable content). What changes is the lifecycle STATE and the STATS, both in mutable sidecars.

### Lifecycle state (`bank/lifecycle.json`, mutable)

```json
{"exp_abc": {"state": "deprecated", "deprecated_at": "2026-08-19", "reason": "utility 0.02"}}
```

State machine: `active` (default) → `deprecated` (harmful/long-term low-utility; moved OUT of the hot bank into the cold archive). Retrieval, induction candidates, and the rebuilt index all exclude `deprecated`.

### Utility stats (`bank/utility_stats.json`, mutable)

```json
{"exp_abc": {"retrieval_count": 15, "utility_count": 2}}
```

`retrieval_count` = times surfaced by retrieve(); `utility_count` = times it contributed to a successful solve. Soft delete (降权, not deletion): when `retrieval_count >= alpha` (default 5, a grace window protecting new experiences) AND `utility/retrieval < beta` (default 0.1), the record's retrieval score is multiplied by a penalty (default 0.3) so it sinks in ranking. The record is never deleted.

### Cold archive (`archive/deprecated.jsonl`, append-only, COMPRESSED)

A deprecated record is compressed into a provenance card (体积约为原记录 10-20%): bulky `retrieval_text`, full `method` body, and `validation`/`scoring` detail are dropped; small provenance fields are kept, PLUS the `retrieval_text` embedding vector (方案甲 — vector kept, original text dropped) so approximate dedup still works:

```json
{"experience_id": "exp_bad", "content_hash": "<ORIGINAL full-record hash>",
 "layer": "modeling", "polarity": "positive", "title": "...",
 "summary": "one-line human-readable gist", "structural_signature": {...},
 "source_episodes": ["ep_3"], "superseded_by": null,
 "created_at": "...", "deprecated_at": "...", "deprecate_reason": "utility 0.02",
 "retrieval_vector": [0.01, -0.03, ...]}
```

Anti-resurrection dedup (`archive/deprecated_index.json`, mutable) is two-layered, because hash alone misses reworded duplicates: (1) exact `content_hash` match (verbatim resurrection), (2) cosine-similarity on the stored vector >= 0.8 (reworded resurrection). `append()` rejects a candidate matching either layer with status `rejected_deprecated`, so a retired harmful experience cannot re-enter the bank verbatim or reworded.
