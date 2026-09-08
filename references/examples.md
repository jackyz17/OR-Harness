# Worked examples

Three complete walkthroughs. JSON fragments are real CLI output shapes (abridged where marked with `...`).

## Example 1: cold start → first induction (and the restraint that isn't a bug)

**T1.** First routing task. No memory, so recommendations come from catalog priors:

```bash
$ orx recommend --task t.json --top 2
{"result": {"recommendations": [
  {"strategy_id": "S04", "name": "construction-plus-ls", "score": -2005.05,
   "expected": {"quality": 0.6, "cost": {"llm_tokens": 2000.0, ...}, "failure_prob": 0.35},
   "evidence": "prior", "confidence": 0.35, "cross_family": false,
   "risk_warnings": [], "basis": "no memory evidence; catalog prior"},
  ...],
 "available_solver_families": {"milp": ["highs", "pulp"], ...}},
 "summary": "Top recommendation: S04 (construction-plus-ls), score -2005.05, evidence=prior, E[Q]=0.6, P(fail)=0.35. ..."}
```

You execute S04, record with your token count:

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

**T2.** S02 runs and scores Q=0.98 (its prior says 0.75); S01 has one prior run at 0.85. **No C1/C2 hint fires** — S02 has n=1, and single observations never count as divergence. This restraint is the design working: one lucky run is not a pattern.

**T3–T4.** After S02's second and third runs confirm ~0.95, `record` returns:

```json
{"induction_hints": [{"criterion": "C2", "strategy_ids": ["S02"],
  "group_key": "family=routing|sc[0.75,1.00]|rc[0.75,1.00]|...",
  "reason": "S02 performs better than its prior: E[gap-quality]=0.95 vs prior 0.75",
  "evidence": {"observed_mean_quality": 0.95, "prior_quality": 0.75, "n": 3,
                "execution_ids": ["ex_...", "ex_...", "ex_..."]}}]}
```

You judge it worth committing:

```bash
$ orx induce --strategy S02
{"result": {"results": [{"created": "se_c977f8ac78e8", "entry": {
  "entry_id": "se_c977f8ac78e8", "strategy_id": "S02",
  "pattern": {"scope_level": "L1", "predicates": {"family": "routing", ...}},
  "expected": {"quality_hat": 0.95, "quality_interval": [0.5, 1.0], ...},
  "status": "candidate", "prediction_track": {"n_predictions": 0, ...},
  "provenance": ["ex_...", "ex_...", "ex_..."], "support_n": 3}, ...}]},
 "summary": "Created/updated 1 entries: se_c977f8ac78e8"}
```

Note the interval `[0.5, 1.0]`: with n=3 the honest floor is 0.35 width — the entry cannot pretend to more certainty than three runs support.

## Example 2: quality tied, cost diverges (what only memory can learn)

Routing group, n=2 each: S01 and S04 both deliver ~0.70 quality, but S01 costs 3× the tokens. Priors said their costs were comparable. `record` fires **C1 on the cost dimension**:

```json
{"induction_hints": [{"criterion": "C1", "strategy_ids": ["S01", "S04"],
  "reason": "cost contrast contradicts prior expectations: S01 meanQ=0.70 vs S04 meanQ=0.70",
  "evidence": {"kind": "cost", "dimension": "llm_tokens",
    "observed": {"S01": 4500.0, "S04": 1500.0},
    "prior": {"S01": 2000.0, "S04": 2000.0},
    "n": {"S01": 2, "S04": 2}, "execution_ids": {...}}}]}
```

After you induce both entries, `recommend --memory-mode cost-aware` ranks S04 first; `--memory-mode strategic` (no cost weighting) still prefers whichever has the higher point quality estimate. That gap between the two modes is the experimental support for the thesis that cost awareness is a necessary component of memory — not an optional extra.

## Example 3: cross-family generalization → miss → tighten (not delete, not demote)

`se_020` is an L2 entry: any family × rc∈[0.75,1.0], predicting S04 quality in [0.8, 0.95], provenance entirely in routing.

A scheduling task with rc=0.8 arrives. `recommend` matches `se_020` but flags it:

```json
{"strategy_id": "S04", "evidence": "strategic_entry", "confidence": 0.42,
 "cross_family": true,
 "risk_warnings": ["cross-family generalization from L2 entry se_020; confidence discounted x0.6"]}
```

You execute anyway; quality comes in at 0.55 — outside the interval. `record`'s automatic chain:

```json
{"prediction_checks": [{"entry_id": "se_020", "hit": false,
  "observed_quality": 0.55, "interval": [0.8, 0.95],
  "scope_tightened": {"tightened": "se_020", "new_scope": "L1",
                      "predicates": {"family": "routing", "resource_coupling": [0.75, 1.0]}}}]}
```

The pattern tightened back to L1: the *range* was wrong, not necessarily the content — so no suspect demotion (that is for content misses, tracked by `consecutive_misses`). If the routing-only prediction later misses three times in a row, *then* the entry demotes to `suspect`, and retirement to the cold archive remains your explicit, irreversible call via `orx retire`.
