# Induction Pipeline (Offline Structural Induction)

How the framework turns accumulated Modeling Bank records into solver-validated,
cross-family optimization patterns. This is the framework's core novelty (Phase 3).

## orx command mapping (harness mode)

In harness mode you drive the pipeline through 7 `orx` commands; the internal
7-stage pipeline maps onto them as follows:

| Internal stage | orx command | Who produces the content |
|---|---|---|
| 1. candidates (cluster discovery) | `orx clusters` | framework (inverted index, no LLM) |
| trigger gating | `orx trigger` | framework (3 gates) |
| 2. encoding (batch signature verify) | (inside `orx clusters`) | framework |
| 3. alignment (role correspondence) | `orx align` | **you** (role bindings, must cite realization_id) |
| 4. inducer (hypothesis generation) | `orx induce` | **you** (1-3 candidates, status=hypothesis) |
| 5. counterexample (solver refutation) | `orx refute` | **you** propose; **framework executes** the refutation code |
| 6. validation (consistency + transfer + scoring) | `orx validate-pattern` + `orx append-pattern` | framework checks consistency; **you supply** transfer evidence (with/without-principle objectives) |
| 7. pipeline (append / archive) | `orx append-pattern` | framework (append validated peers; archive refuted) |

The 5 chained commands form a stamped chain keyed by `cluster_id`:
`align → induce → refute → validate-pattern → append-pattern`.
Each content-producing command writes a template artifact on first call; you
fill it, then re-run the same command to stamp it. Editing a stamped artifact
invalidates the stamp (content-hash check).

## What it is (and is not)

- It IS **structural induction** over **heterogeneous but structurally-isomorphic**
  experiences: different problem families (inventory / production / workforce / ...),
  the same mathematical skeleton. From their structural correspondence it induces a
  general principle that none of the sources states on its own.
- It is NOT **summarization or redundancy compression** (Auto-Dreamer's region
  rewriting). Sources are never merged, rewritten, or deleted; a validated principle is
  a NEW append-only record, and it must prove transfer to an UNSEEN task.

The operational test that separates induction from summary:

```
P not in any single M_i        (novelty: the principle restates no source)
P(M_i) ~= true for all sources (source consistency / coverage)
Transfer(P, unseen) > 0        (it improves a task it was NOT induced from)
```

If unseen transfer shows no improvement, the candidate is refuted — that gate is what
keeps "induction" honest.

## The seven stages (one pass per candidate cluster)

```
ModelingStore.all_records()           (all are peers, carry a signature)
  1. candidates     isomorphic + cross-family cluster discovery (inverted index, no O(N^2))
  2. encoding       batch structural-signature encoding / verification
  3. alignment      map abstract structural roles across the cluster (grounded)
  4. inducer        generate candidate principle(s) as status=hypothesis
  5. counterexample LLM proposes failure conditions; the EXECUTOR verdict decides
  6. validation     source consistency + UNSEEN transfer + scoring
  7. pipeline       validated -> append peer record (status=validated); refuted -> archive in report
```

### 1. candidates.py — cluster discovery

Two-level inverted index over the structural signature: bucket by `signature.core_key()`
(fixed core-4), then sub-split by open-feature agreement via union-find (a member with no
features joins any group — missing is not penalized). A cluster is kept only if it spans
`min_families` (>=2) problem families. Members carry provenance handles
(`evidence.source_episodes`) and a priority score (size / heterogeneity / retrieval hits /
recency) used for trigger ordering.

### 2. encoding.py — batch structural encoding

Reuses `modeling/signature_extractor.SignatureExtractor`. A record with a valid existing
signature is reused verbatim; one without is routed to the agent's LLM with the standard
extraction prompt. A bare `{}` is NOT treated as an existing signature.

### 3. alignment.py — cross-memory role correspondence

The RELATION step. Maps abstract roles (CANONICAL_ROLES: `resource_pool`, `capacity_limit`,
`competing_decisions`, `objective_contribution`, `demand_requirement`, `coupling_constraint`,
`time_period`, `flow_balance`) to each member's concrete entities. Every binding must cite
its `realization_id` (grounding); ungrounded bindings are dropped. Produces the pattern's
`role_schema` + `role_mappings`.

### 4. inducer.py — hypothesis generation

Grounded in the alignment (never a free-text summary). Yields 1-3 candidate principles,
each `status=hypothesis` with a `complexity` estimate (Minimum-Explanation preference feeds
the scoring penalty).

### 5. counterexample.py — solver-backed refutation

The LLM proposes failure conditions (fixed-charge / nonconvex / min-batch / ...) and writes
a small refutation program, but the program's EXECUTED output (`SafePythonExecutor`) is the
only verdict. A crashed or unexecuted refutation is NOT a counterexample (anti self-judgment).
Confirmed counterexamples shrink `applicability_conditions` and are recorded; they never
delete the hypothesis.

### 6. validation.py — verification and scoring

```
Score = alpha*C + beta*T + gamma*V + delta*N - lambda*K - mu*X
  C coverage        fraction of sources the principle is consistent with
  T transferability fraction of unseen tasks improved (with vs without principle)
  V validation      counterexample-survival strength
  N novelty         1 if the principle is contained in no single source
  K complexity      Minimum-Explanation penalty
  X counterexample  penalty per confirmed solver-refuted counterexample
```

`decide` requires: not refuted-flat, `total >= validation_threshold`, AND at least one
`improved` unseen-transfer test. Otherwise `refuted` (kept, not deleted).

### 7. pipeline.py — orchestration

A `validated` principle is assembled into a `ModelingExperience` peer record (`method.action_template` = the principle statement, `role_schema`, `role_mappings`, `validation`, `scoring`, `status=validated`, `derived_from_experience_ids`) and appended to the Modeling Bank as a new peer. Sources stay untouched. A `refuted` candidate is recorded in the run report, NOT the bank.

## Trigger policy (v1, decision D4)

`induction/trigger.py` gates when a pass runs. Three gates, in order:

1. **candidate gate** — at least one heterogeneous isomorphic cluster exists.
2. **cooldown** — a cluster whose membership is unchanged since the last run is skipped.
3. **watermark** — 新增计数基线: the run fires only when at least `min_new_realizations`
   NEW realizations have accumulated since the last run (bypassed on the very first run).

State lives in an append-only "sidecar" (附属统计文件——主库旁边另开的小文件，专记会变的统计数字，不写进主库) at `bank/induction_trigger_log.jsonl` (see
[experience-schema.md](experience-schema.md#induction-trigger-sidecar-mutable-stats-append-only)).

## Harness contract (D18)

The framework owns rules (prompts, controlled vocabularies, parsing, the executor verdict,
scoring); the agent owns the LLM and the solver executor. `InductionPipeline` holds neither:
LLM steps take injected drivers (`LLMBackedAligner/Inducer/CounterexampleSearcher`) and the
transfer comparison takes an injected `transfer_solver(task, principle) -> objective`. In
harness mode the agent supplies these via the `orx` commands; the `LLMBacked*` classes are
for standalone runs/tests.

## CLI

```bash
python3 scripts/orx.py trigger                     # check the 3 gates
python3 scripts/orx.py clusters                    # list candidate clusters
python3 scripts/orx.py align --cluster <id>        # template -> fill -> re-run to stamp
python3 scripts/orx.py induce --cluster <id>
python3 scripts/orx.py refute --cluster <id>
python3 scripts/orx.py validate-pattern --cluster <id>
python3 scripts/orx.py append-pattern --cluster <id>
```
