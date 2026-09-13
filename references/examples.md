# Worked examples

Four complete walkthroughs. JSON fragments are real CLI output shapes (abridged where marked with `...`).

## Example 0: coupling-aware understanding (the pre-model step)

**T0.** A scheduling task arrives: three production modes (M1, M2, M3) all feed into the same downstream LDA capacity. Before writing any model, you extract the coupling structure and submit it as a CIR:

```bash
$ orx understand --task t.json
{"result": {
  "cir": {
    "entities": [{"name": "LDA", "kind": "resource"}, ...],
    "decisions": [{"name": "x_M1", "kind": "production"}, {"name": "x_M2", ...}, {"name": "x_M3", ...}],
    "constraints": [{"id": "C1", "kind": "capacity", "expr": "sum(x_M1, x_M2, x_M3) <= cap_LDA"}],
    "relations": [
      {"source": "x_M1", "target": "LDA", "type": "uses_resource", "evidence": "semantic", ...},
      {"source": "x_M2", "target": "LDA", "type": "uses_resource", "evidence": "semantic", ...},
      {"source": "x_M3", "target": "LDA", "type": "uses_resource", "evidence": "semantic", ...}
    ],
    "coupling_groups": [{"type": "shared_bottleneck", "members": ["x_M1", "x_M2", "x_M3"],
      "resource": "LDA", "implication": "Multiple decisions (x_M1, x_M2, x_M3) consume the same
      resource (LDA); ensure one aggregate capacity constraint covers all relevant decisions."}],
    "issues": []
  },
  "modeling_guidance": [{"type": "shared_bottleneck", "members": ["x_M1", "x_M2", "x_M3"],
    "resource": "LDA", "implication": "...aggregate capacity constraint..."}],
  "cir_warnings": []
},
 "summary": "CIR validated: 1 entities, 3 decisions, 1 constraints, 3 relations. Modeling guidance (1):
   - [shared_bottleneck] Multiple decisions (x_M1, x_M2, x_M3) consume the same resource (LDA);
     ensure one aggregate capacity constraint covers all relevant decisions."}
```

**Verification**: entity `LDA` covers the shared capacity mentioned in the task. All three modes have `uses_resource` edges to `LDA`. The `shared_bottleneck` group matches the task's coupling pattern. `issues` is empty. You carry this guidance into step 2 (write the canonical model with an aggregate capacity constraint `sum(x_M1, x_M2, x_M3) <= cap_LDA`).

### Do not do this (CIR negative example)

Do not submit a CIR with a `shares_resource` relation based only on variable name similarity:

```json
// BAD: no semantic evidence — "x_M1" and "x_M2" both contain "M" so they "share" something
{"source": "x_M1", "target": "x_M2", "type": "shares_resource", "evidence": "structural"}
```

Co-occurrence in a constraint is structural evidence only — it produces a generic `depends_on` edge, never a semantic `shares_resource`. To claim `shares_resource`, both decisions must link to the same resource entity with `uses_resource` edges, or the agent must declare it with `evidence="declared"` and validate it against the model.

## Example 1: cold start → first induction (and the restraint that isn't a bug)

**T1.** First routing task. No memory, so `recall` returns candidates with `evidence="no_memory"` — the catalog provides the strategy vocabulary (applicability, actions, solver family) but no quality/cost/risk claims:

```bash
$ orx recall --task t.json --top 2
{"result": {"recommendations": [
  {"strategy_id": "S04", "name": "construction-plus-ls", "score": -Infinity,
   "expected": {"quality": 0.0, "cost": {}, "failure_prob": 0.0},
   "evidence": "no_memory", "confidence": 0.0, "cross_family": false,
   "risk_warnings": [], "basis": "no experience for this strategy×profile pair; catalog vocabulary only"},
  ...],
 "available_solver_families": {"milp": ["highs", "pulp"], ...}},
 "summary": "Top candidate: S04 (construction-plus-ls), score -inf, evidence=no_memory, E[Q]=0.0, P(fail)=0.0. ..."}
```

You pick S04 based on structural fit (it's a general-purpose strategy with no applicability constraints) and execute it:

```bash
$ orx execute --task t.json --strategy S04 --code solve.py --workspace ws --solver highs
{"result": {"execution": {"execution_id": "ex_6096390d0949", "task_id": "t1",
  "strategy_id": "S04", "quality": {"feasible": true, "objective": 100.0, "gap": 0.0, ...},
  "cost": {"llm_tokens": 0.0, "tool_calls": 1.0, "solver_runtime_s": 0.5, ...}, ...}},
 "summary": "Execution ex_6096390d0949 finished with status optimal ... Nothing is recorded yet ..."}

$ orx record --execution ex_6096390d0949.json --override llm_tokens=1840
{"result": {"execution_id": "ex_6096390d0949", "recorded": true,
  "prediction_checks": [], "induction_hints": []},
 "summary": "Recorded ex_6096390d0949. No induction hints."}
```

**T2.** S02 runs and scores Q=0.98. **No C2 hint fires** — S02 has n=1, and single observations never count. This restraint is the design working: one lucky run is not a pattern.

**T3–T4.** After S02 repeats ~0.95 on **two further tasks** (t3, t4 — the claim needs distinct tasks, not one task run three times), meanQ ≥ 0.75 over n=3 and `record` returns:

```json
{"induction_hints": [{"criterion": "C2", "strategy_ids": ["S02"],
  "group_key": "family=routing",
  "reason": "S02 performs high: meanQ=0.95 over n=3",
  "evidence": {"observed_mean_quality": 0.95, "direction": "high", "n": 3,
                "execution_ids": ["ex_...", "ex_...", "ex_..."]}}]}
```

You judge it worth committing:

```bash
$ orx induce --strategy S02
{"result": {"results": [{"created": "se_c977f8ac78e8", "entry": {
  "entry_id": "se_c977f8ac78e8", "strategy_id": "S02",
  "pattern": {"predicates": {"family": "routing",
                             "resource_coupling": [0.78, 0.94], ...}},
  "expected": {"quality_hat": 0.95, "quality_interval": [0.5, 1.0], ...},
  "status": "candidate", "prediction_track": {"n_predictions": 0, ...},
  "provenance": ["ex_...", "ex_...", "ex_..."], "support_n": 3}, ...}]},
 "summary": "Created/updated 1 entries: se_c977f8ac78e8"}
```

Note the interval `[0.5, 1.0]`: with n=3 the honest floor is 0.35 width — the entry cannot pretend to more certainty than three runs support. Note also what made it admissible: three executions from three distinct tasks (t2, t3, t4). Had all three been the same task, `induce` would have returned `skipped: "needs independent evidence: all 3 observations come from 1 task [...] — a claim requires >=2 tasks"` and the executions would have stayed in the Evidence Bank as `conditional_stats` only.

## Example 2: quality tied, cost diverges (what only memory can learn)

Routing group, n=2 each on distinct tasks: S01 and S04 both deliver ~0.70 quality, but S01 costs 3× the tokens. `record` fires **C1 on the cost dimension**:

```json
{"induction_hints": [{"criterion": "C1", "strategy_ids": ["S01", "S04"],
  "reason": "cost contrast between strategies: S01 meanQ=0.70 vs S04 meanQ=0.70",
  "evidence": {"kind": "cost", "dimension": "llm_tokens",
    "observed": {"S01": 4500.0, "S04": 1500.0},
    "n": {"S01": 2, "S04": 2}, "execution_ids": {...}}}]}
```

After you induce both entries, `recall --memory-mode cost-aware` ranks S04 first; `--memory-mode strategic` (no cost weighting) still prefers whichever has the higher point quality estimate. That gap between the two modes is the experimental support for the thesis that cost awareness is a necessary component of memory — not an optional extra.

## Example 3: a claim meeting its counterexample (miss → demote at the next induce)

`se_020` was induced from routing executions with rc∈[0.78,0.94], predicting
S04 quality in [0.8, 0.95]. Its applicability is exactly that: the family plus
the demonstrated span — no ladder, no bins.

A routing task with rc=0.80 arrives, inside the span. `recall` matches
`se_020` and you execute; quality comes in at 0.55 — outside the interval.
`record` writes the frozen check onto the fact and changes nothing:

```json
{"prediction_checks": [{"entry_id": "se_020", "hit": false, "predicted": 0.95,
  "interval": [0.8, 0.95], "observed": 0.55, "hit": false}],
 "cost_feedback": {...}}
```

The claim covered that task when the execution ran, so the miss is evidence
against the claim. Two more like it and the next `orx induce` reports:

```json
{"revisions": [{"entry_id": "se_020", "strategy_id": "S04",
  "forward": {"n_predictions": 3, "n_hits": 0, "hit_rate": 0.0,
              "consecutive_misses": 3, "calibration_error": 0.4},
  "misses": ["ex_...", "ex_...", "ex_..."],
  "transitions": ["demoted:->suspect"]}]}
```

`suspect` is downweighted ×0.5 and labelled in `recall`; retirement to the
cold archive remains your explicit, irreversible call via `orx retire`.

Note what a *record outside the span* does instead: it does not match the
claim, so it is neither a check nor a counterexample — and if the strategy
keeps doing well outside the span too, the next `induce` widens the claim's
intervals to follow that evidence. No operation to remember, no level to
choose.
