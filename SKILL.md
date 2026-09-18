---
name: or-harness
description: >
  Formulate, solve, debug, validate, and improve operations research and
  optimization problems — LP, MILP, scheduling, routing, assignment,
  network flow, resource allocation, supply-chain, and other highly coupled
  industrial OR scenarios. Provides coupling-aware problem understanding
  (CIR), a sandboxed executor with seven solver adapters, and a two-layer
  strategy memory (Execution Evidence facts + Strategic Knowledge
  commitments) that learns which solving strategies fit which problem
  structures at what execution cost. Use when the user asks to formulate,
  solve, retry, decompose, validate, or debug an optimization model, or to
  query or manage accumulated OR strategy experience. Do not use for one-off
  optimization questions with no repetition, generic math proofs, or
  non-optimization tasks.
---

# OR-Harness: Strategy Learning for Optimization Agents

You are the orchestrator. This capability layer only advises and executes — you retain full control: you may refuse any recommendation, request alternatives, execute without recording, override recorded costs, and you alone decide when to induce and when to collect garbage.

OR-Harness never runs autonomously and keeps no hidden state: every command is a stateless call against an explicit memory directory (`--home` or `$OR_HARNESS_HOME`). It calls a model ONLY when you explicitly configure a provider (`--world-model URL::MODEL` or `ORHarness(world_model=...)`) and explicitly invoke a prediction command — no command reaches out on its own, and with no provider configured every world-model command returns `not_configured`. A prediction is a shadow hypothesis you may ignore; it never becomes a fact, and an unexecuted candidate's prediction is never real feedback.

## Core workflow (the loop to run for every optimization task)

1. **Profile (the single analysis entry)** — submit your coupling understanding as the task JSON's optional `coupling` field (a CIR — Coupling-Aware Intermediate Representation) and run `orx profile --task t.json`. One call returns: the validated CIR with `modeling_guidance` (explicit, inspectable coupling groups), the problem profile, and the derivation report (each coupling dimension's value and origin). The profile serves two purposes: it helps YOU understand the problem structure, and it is the structural key that retrieves comparable evidence in the next step. **A task without a `model` field is a normal state** — strategy selection needs the task text, the CIR, and the profile, never a finished formulation.
2. **Recall and compare strategies** — `orx recall --task t.json --top 3`. The result carries two independent channels: `recommendations` (structural — the profile decides which evidence is comparable) and `vector_recall` (text similarity — unfiltered by structural cell, so a near-identical problem in a different bucket is still visible). Read the structural verdict for reuse and the text hits for context; neither is a substitute for the other. `--exclude` any candidate you distrust and re-recall.

   **Optionally**: with a world model configured, `orx plan-next` compares candidates by their PREDICTED consequences (same quality/cost/risk yardstick) and suggests a first step.
3. **Choose** — weigh quality vs. cost vs. risk yourself. When quality estimates are tied, prefer the cheaper candidate (that preference is exactly what this memory exists to learn). Pick the concrete solver from `available_solver_families`, heeding advisories. Freeze the pre-execution cost expectation with `orx predict --task t.json --strategy S` and pass the returned snapshot back at record time (`--prediction`) — feedback compares against the prediction actually used, never a post-hoc estimate. Unknown cost is reported as unknown, never as zero. If you planned with `plan-next`, record your explicit choice with `orx choose-next` (accept, deviate, or reject).
4. **Model the problem (intermediate representation, after the strategy is chosen)** — NOW write the GAMS-style model representation (SETS / PARAMETERS / VARIABLES / OBJECTIVE / CONSTRAINTS; see [references/modeling.md](references/modeling.md)) as the task JSON's top-level `model` field. The model is the blueprint for solve.py, written under the chosen strategy: decomposition, rolling horizon, or relaxation strategies may alter the formulation and execution plan. The framework verifies it (L1 format + L2 symbol cross-reference) and derives exact structural coupling from the declared constraints. Re-run `orx profile` to get the CIR ↔ model cross-check (`cir_warnings`, e.g. a CIR decision not declared as a model variable) — this is how you catch "the model missed a coupling the CIR declared" before coding.
5. **Write solve.py** — follow the chosen strategy's `actions` (the framework never generates code). The model from step 4 is your blueprint — the code is a translation, not a re-derivation. The script must write `result.json` with `status, objective_value, objective_bound, mip_gap, runtime_seconds`.
6. **Predict cost (optional world-model shadow)** — with a world model configured (`--world-model`), you may ask for a structured outcome prediction of the candidate execution (`orx predict-outcome`) BEFORE executing; you compare it afterwards (`orx bind-outcome`) to accumulate calibration evidence.
7. **Execute** — `orx execute ...`. Every execution — successes AND failures — is automatically staged in a pending area (never lost, even if you immediately retry). Inspect `result.execution.quality.problems` before recording. If you made a shadow prediction, bind it now (`orx bind-outcome --prediction <id> --action <action_id from the execute output>`).
8. **Verify** (see Verification below) — a checked execution is what you record; the check comes first.
9. **Record** — `orx record --execution <json> --override llm_tokens=<your actual token count> [--override-mode replace|increment] [--prediction <predict-output.json>]`. `--override` default is `replace` (idempotent — the value IS the measurement; re-applying never double-counts); use `increment` only for an additional measured amount within the same attempt. The response lists `unrecorded_staged_executions` for this task — if a failed first attempt is sitting there, backfill it with `orx record --from-staged <id>` (verbatim, no re-typing). Read the returned `induction_hints`, `prediction_checks`, and `cost_feedback` (computed only against the prediction snapshot you passed).
10. **Decide on induction** — hints are evidence, not orders. Induce only when you judge the pattern worth generalizing.
11. **On failure** — follow Recovery below before retrying.

## Tool policy (when to invoke which capability)

| Situation | Action |
|---|---|
| No `--world-model` configured | Run the core loop directly (steps 1–5, 7–9). Every world-model command returns an explicit `not_configured` — no command silently degrades |
| World model configured, any task | `recall` for candidates → `predict-outcome` the chosen candidate BEFORE executing → `bind-outcome` after. Every execution feeds calibration; an unbound prediction is discarded evidence |
| World model configured, high-stakes decision (expensive execution, tied candidates, unfamiliar cell) | `plan-next` to compare candidates by PREDICTED consequences before choosing; bind the selected path's prediction rather than predicting the chosen candidate again |
| Facing an induction decision (hints accumulated, ≥2 tasks of evidence) | `assess-induction` to evaluate the induction's value BEFORE committing; accept or reject explicitly |
| No embedding backend configured | The text channel is off: `recall` returns `degraded` plus the structural channel alone. Enable it with `OR_EMBEDDING_BASE_URL` + `OR_EMBEDDING_MODEL` + `OR_EMBEDDING_API_KEY`, or by injecting a backend |
| Embedding backend configured, no index yet | Run `orx rebuild-index` once. Afterwards `record` / `induce` / `retire` keep the index current incrementally |
| Budget declared and near its limit | Check `orx budget` before each model call; an exceeded budget stops planning automatically |
| Simple problem, single independent constraint, no shared resources | CIR optional — state the skip decision explicitly |

**Cost discipline**: every world-model call spends real tokens, charged to the decision action or the maintenance scope. `plan-next` and `assess-induction` are avoidable planning overhead — skip them when a plain `recall` already answers the question. `predict-outcome` / `bind-outcome` are NOT in that category: they collect calibration data (one pair per executed action) and are skipped only when no world model is configured.

## Trigger situations (recognize these before acting)

| Situation | Action |
|---|---|
| Problem has shared resources, cross-stage dependencies, or temporal propagation | Extract a CIR and submit it as the task's `coupling` field before choosing a strategy |
| `profile` returns non-empty `coupling.cir.issues` or `cir_warnings` | Fix the CIR (or reconcile it with the `model`) and re-run `profile` before writing solve.py |
| `recall` reports `degraded` | Only the structural channel ran. Read the `reason` — an empty or degraded `vector_recall` is not evidence that no similar memory exists |
| A `vector_recall` hit is labelled `different_cell` | Read its text and observed outcome for context; `profile_cell` shows which structure its numbers actually describe |
| A knowledge hit has `reusable: false` | `reason` names the conflict, or reports `unknown` (a value needed to decide is missing) — measure it before applying |
| An execution's `quality.problems` is non-empty | Fix the script and re-execute; the record waits until the check passes |
| Solver returns `infeasible` | Check variable bounds and conflicting constraints; if the task is genuinely infeasible, record the status — it is valuable evidence |
| A solve succeeds but the objective looks wrong (magnitude or direction) | Re-derive the model from the task; never adjust constraints to match a reference value |
| You predicted one strategy/solver and executed another | `bind-outcome` records the mismatch and skips the comparison — bind the action that actually ran |
| A hint rests on repeated runs of one `task_id` | A claim needs ≥2 distinct tasks. Either record a genuinely independent instance under its own `task_id`, or record a second task first |
| `record` reports `index_sync: deferred` | The fact is saved but not yet text-searchable. Recover with `orx rebuild-index` |
| A task's requirements, capacity or objective change | That is a NEW task context: use a new `task_id` (or episode). The old X is not inherited — progress from a differently-versioned task is refused, and a prediction is scored against the version it was made under, not the current one |
| A knowledge class reports `reliability: null` / `insufficient_history` | Too few resolved samples. That is honest unknown, not a failure — the class grants no value until it has history |
| `orx contract` returns `status: "contract_only"` | The contract is implemented but no prediction service is attached — no forecast was made. Do not read it as one; see [references/world_model_contract.md](references/world_model_contract.md) |

## Coupling dimensions (operational definitions)

Structural grouping — the foundation of all memory — keys on these. Supply them accurately or let the framework derive them (priority: CIR structure > model > spec):

| Dimension | Measures | Derivable? |
|---|---|---|
| resource_coupling | fraction of decisions involved in a resource relation (CIR) / fraction of decision variables appearing in MORE THAN ONE constraint (model) | yes — cir > model > spec |
| temporal_coupling | fraction of decisions/variables indexed by a temporal set (time/period/stage/...) | yes — cir > model > spec |
| route_complexity | fraction of decisions/variables indexed by a network set (arc/edge/link/...) | yes — cir > model > spec |
| semantic_coupling | business-semantic relatedness — invisible to structure | NO — always your call |

With a CIR present, `resource_coupling` = fraction of decisions that are the source of ≥1 `uses_resource`/`shares_resource`/`competes_for` relation; `temporal_coupling`/`route_complexity` = fraction of decisions with time-like/network-like indexes. Without a CIR, the model-based definitions apply.

These are measured from constraint structure, not guessed from the problem's *name*: two independent resource constraints mean rc≈0, however resource-flavoured the task sounds.

## Verification (before every record)

### CIR verification (at `orx profile`, before choosing a strategy)

Check `result.coupling.cir` and `result.coupling.modeling_guidance`:
- **Entity coverage**: every resource, product, site, period, or route mentioned in the task description appears as an entity or is reachable via a decision's indexes.
- **Relation plausibility**: every `uses_resource` / `shares_resource` / `competes_for` relation connects a decision to a resource-like entity — not two decisions with no shared resource.
- **Coupling group alignment**: the detected `shared_bottleneck` or agent-declared coupling groups correspond to coupling patterns the task description actually implies (e.g. "all modes use the same downstream capacity" → `shared_bottleneck` on that capacity).
- **No unresolved issues**: `result.coupling.cir.issues` is empty — non-empty means L1/L2 validation found structural problems (dangling references, duplicate names, bad evidence levels).
- **cir_warnings** (when a `model` is also present — i.e. after you wrote the model in step 4): each warning flags a CIR ↔ model inconsistency (e.g. a CIR decision not declared as a model variable).

### Execution verification (after `execute`, before `record`)

Check `result.execution`:
- `quality.status` is one of optimal/feasible/infeasible/unbounded/timeout/error — treat anything else as a broken script, not a solver result.
- `quality.feasible` is true and `quality.objective` is finite when you expect a solution.
- `quality.gap` is recorded (derived from the bound when the solver omits it).
- `quality.problems` is empty — non-empty means verification caught something (illegal status, missing objective, non-finite value).
- The objective value is plausible for the problem (right order of magnitude, correct min/max direction) — the sandbox checks structure, not semantics; only you know what the number should mean.

## Recovery (when execution fails)

- **status=error, normalized_error mentions security policy** — your script used a blocked construct (network/shell/pathlib/dynamic `open()` paths). Rewrite using only stdlib and a literal `open('result.json', 'w')`. Subprocess-based solvers (e.g. PuLP's CBC backend) cannot run in the sandbox — switch to an in-process solver (ortools GLOP/CP-SAT, highspy). The failed execution is staged automatically; record it (`--from-staged`) so the memory learns this too.
- **status=error, traceback in normalized_error** — read the error, fix the model or script, re-execute. Every attempt is its own record: a first failure is `retries=0` (an observed zero), and an attempt that follows earlier ones in one retry loop declares it with `--override retries=N` (the absolute count of NEW retries this attempt adds).
- **status=timeout** — the strategy may be too heavy for this scale. Re-recall with `--exclude <strategy>` and try the next candidate; record the timeout (it is a fact worth remembering).
- **`recall` returns no candidates** — no strategy's applicability matches the profile. Check the profile's coupling dims; if they are extreme, relax your exclusions or reconsider the coupling values. A `vector_recall` hit for a similar problem whose structure differed is a strong hint that the coupling values are off.
- **A `coupling_warnings` entry at profile time** — your supplied value contradicts the structural derivation. The derived value is what grouping will use; trust the measurement and correct your annotation.

## Core concepts (terminology is strict)

- **Execution Evidence Bank** — append-only episodic facts: the strategy actually used, the quality/cost actually observed, failures, artifacts. Never stores generalizations; the single source of truth. Only cost dimensions may be backfilled (`llm_tokens`); nothing else is rewritten.
- **Strategic Knowledge Bank** — induced commitments: expected quality, cost, and failure risk, with prediction intervals, a calibration track, and applicability predicates read off the supporting evidence. Mutation happens at INDUCTION time only. Creating an entry takes ≥2 supporting executions from ≥2 distinct tasks; **publishing** it takes a passed admission check (`induce --verify`).
- **Conditional statistics** — on-the-fly aggregation over the Evidence Bank per (strategy × structural cell). A recount of observations, not a commitment; never persisted.
- **group / evidence set** — one (family, structural cell, strategy) triple: the observations that may be aggregated together. A cell is the measurable coupling dims quantized to `[0.00,0.25] [0.25,0.50] [0.50,0.75] [0.75,1.00]`, with unmeasured dimensions in their own `[unknown]` cell. Structurally different regions of one family stay separate — pooling a region scoring 1.0 with one scoring 0.1 once produced a "0.55 everywhere" claim.
- **CostVector** — five dimensions, stored raw and never folded: `llm_tokens, tool_calls, solver_runtime_s, retries, latency_s`. `retries` counts only extra attempts beyond the first. Unknown ≠ zero: each record carries a measured-dimension mask (`cost_measured`) and a solver-runtime provenance (`reported` vs `wall_proxy`), and unmeasured dimensions are excluded from means, comparisons, and prediction errors. (In Chinese documentation: 代价, not 成本 — the price paid at decision time, not bookkeeping.)
- **Cold archive** — cards for retired entries that veto re-induction of the same failed generalization; `induce --force` lifts that veto when you judge the environment has drifted.

## Commands (quick reference — full specs in [references/commands.md](references/commands.md))

Every command prints one JSON line: `{"result": {...}, "summary": "2-4 sentence agent-readable text"}`. Exit codes: `0` success, `2` usage/precondition error, `1` crash. All accept `--home DIR` (default `$OR_HARNESS_HOME`, else `./or_harness_home`).

| Command | Use it for |
|---|---|
| `profile --task t.json [--code solve.py] [--cir cir.json]` | The analysis entry: CIR validation + modeling guidance + profile + derivation report. Run it first, and again after writing the `model` |
| `recall --task t.json [--top 3] [--exclude S04 S06] [--memory-mode M] [--include-unverified]` | Both retrieval channels: structural `recommendations[]` + text-similarity `vector_recall` |
| `predict --task t.json --strategy S` | Freeze the pre-execution cost expectation; pass the snapshot to `record --prediction` |
| `execute --task t.json --strategy S --code solve.py --workspace DIR --solver NAME` | Sandbox-run solve.py and stage the execution |
| `record --execution <json\|path> \| --from-staged <id> [--override llm_tokens=N] [--prediction <json>]` | Persist the fact: cost backfill → quality checks → cost feedback → C1–C6 hints |
| `induce [--strategy S \| --all] [--rebuild] [--dry-run] [--force] [--note TEXT] [--verify JSON]` | The only place knowledge changes |
| `inspect --bank experience\|strategic\|archive\|actions\|snapshots\|predictions\|texts [--task ID]` | Query one memory layer |
| `snapshot --task t.json [--episode ep1]` | Freeze and persist the current belief state |
| `action --report TYPE --task t.json [--episode ep1]` | Report an action YOU performed |
| `budget --task ID [--episode ep1] [--declare llm_tokens=50000,...]` | Consumption view over all real action costs |
| `predict-outcome` / `bind-outcome` | World-model shadow prediction, then its comparison against the real action. With knowledge targets in play, the prediction also covers the capability-evolution side (`knowledge_changes`) and is judged at two stages: on `record` (evidence landed) and on the next `induce` (a claim formed / moved) |
| `contract` | Build or read a **unified world-model contract** (no model call): `--kind strategy_outcome` / `capability_evolution` builds a versioned, serializable object; `--payload <json>` reads a stored prediction and reports its version (current / legacy / unsupported). With no provider configured the built object says `status="contract_only"` — the contract is implemented, the prediction service is NOT. See [references/world_model_contract.md](references/world_model_contract.md) |
| `plan-next` / `choose-next` | Bounded planning over predicted consequences, then your explicit choice. `--delta W` weights the predicted knowledge term (`U = αQ − βC − γR + δK`); `--prediction-mode` selects what is predicted |
| `assess-induction [--bundle f.json \| --candidates-only]` | Evaluate an induction's value before committing |
| `bind-induction-outcome --assessment ID` | Judge an induction assessment's predictions against the induction that actually ran |
| `gc [--mode compact\|purge] [--dry-run]` / `retire --entry ID --reason "..."` | Derived-layer disposal |
| `rebuild-index [--layer both\|execution\|strategic] [--dry-run]` | First build or repair of the retrieval index |
| `doctor` | Environment and retrieval-index self-check |

### Embedding configuration (the text channel)

The text channel is **off unless a real embedding model is configured**, deliberately: a lexical hash is not semantic retrieval, and presenting one as semantic similarity would make text matching look meaningful when it is not. Set `OR_EMBEDDING_BASE_URL` + `OR_EMBEDDING_MODEL` + `OR_EMBEDDING_API_KEY` (any OpenAI-compatible `/embeddings` endpoint — **not** the `OR_WM_*` chat variables, since the two models may be different endpoints with different dimensions), or inject a backend in Python. `OR_EMBEDDING_BACKEND=local-hashing` selects the deterministic offline backend for hermetic runs and tests.

## Decision guidance

- **induction_hints after record**: hints are evidence, not orders. C1 (significant contrast between ≥2 strategies in the same structural cell, judged separately for quality and cost), C2 (extreme high/low performance with n≥2), C3 (in-group quality drift), C4 (fallback exercised), C5 (same-direction performance reproduced in ≥2 families at the same structure), C6 (all-feasible stable success). Induce when you judge the pattern worth generalizing; you may also induce with no hint at all. Hints never verify knowledge.
- **status vs verification**: `status` tracks how the entry has behaved (`candidate` = plausible, unproven; `validated` = ≥5 frozen checks with ≥70% hits **and a passed admission check**; `suspect` = 3 consecutive content misses, downweighted ×0.5 — treat its estimates as warnings, not facts; `dormant` = not consulted for 10 tasks, excluded from matching, and the next induction wakes it if new matching evidence arrived). `verification.state` tracks whether the CLAIM was checked (`unverified` / `verified` / `insufficient_evidence` / `refuted`). They cannot contradict: `validated` requires `verified`, and a `refuted` claim can never be `validated`. Forward calibration never substitutes for admission. Both are applied by `orx induce`, never by `record`. Cost deviations never demote or re-scope an entry — they stay on the execution as `cost_feedback` evidence.
- **confidence & cross_family**: a family-free pattern (only harness-authored entries are) is discounted and labelled when it is applied outside its provenance families — weigh it accordingly.
- **When to gc**: when `inspect` shows large groups fully covered by validated entries. Run `--dry-run` first and review the plan.

## Output requirements

Report to the user:
- the chosen strategy and the concrete solver, with the evidence that drove the choice (`evidence`, `confidence`, any `risk_warnings` / `solver_advisories` you acted on);
- the objective value and the key decisions in the solution, with the status (`optimal` / `feasible` / ...) stated explicitly;
- the assumptions you made and the data you had to interpret — especially when the task was ambiguous;
- the real cost you paid, including the `llm_tokens` backfill;
- any failed attempt that is part of the story, with its status (a timeout or infeasibility is a result, not a thing to bury).

A plan suggestion is a recommendation, not a decision: say which candidate you chose and whether you deviated from the suggestion.

## Discipline (the few mistakes that matter)

These are the misconceptions that actually cost work — each is a positive rule, not a prohibition:

- **A prediction is a shadow.** Choose on evidence; compare the prediction afterwards with `bind-outcome`. `plan-next` suggests, `choose-next` decides, `execute` acts — pick the one the moment calls for.
- **A contract is not a prediction.** `orx contract` returns `status="contract_only"` when no provider is configured: the schema exists, no forecast was made. Reading it as a forecast is the same error as reading `not_configured` as a result.
- **A knowledge entry appearing is not a capability gain.** Predicting, binding the fact, and verifying the effect are three different things; only the third supports "the harness got stronger". The capability contracts carry no composite H score, and `harness_state` knowledge refs / experience counts / tool config are evidence ABOUT H, not measured H.
- **An attempt is not a strategy window.** One `execute_strategy` call is one solve attempt; modeling / repair / verify are auxiliary overhead, reported separately and never folded into a predicted scope. A window-scope prediction is scored only when `trace.comparable` is true.
- **Predict before acting.** The input snapshot freezes at prediction time; a prediction made after the execution is hindsight, and binding it to a different strategy's action records a mismatch instead of a score.
- **The model comes after the strategy, before the code.** Strategy selection uses the task text, the CIR, the profile, and the evidence; the `model` is the blueprint you then write, and solve.py translates it.
- **Measure coupling from structure.** The problem's name is not evidence; two independent resource constraints mean rc≈0 however resource-flavoured the task sounds.
- **A signal is not the decision.** A hint, a `vector_recall` similarity, a model's self-reported `confidence`, and a plan suggestion are all inputs — the choice and the admission verdict remain yours or the framework's, respectively.
- **Similarity discovers; structure decides reuse.** `different_cell` hits are context to read, not statistics to apply. `applies` may be reused; `conflicts` and `unknown` may not — an unmeasured condition is not a satisfied one.
- **A claim needs two tasks, and a verdict.** Record a genuinely independent instance under its own `task_id`; publish with `induce --verify` on real executions, not on a program's printed verdict or the candidate's own summary.
- **Failures are raw material.** Record every attempt, including the ones that failed or timed out; backfill from the staging area with `--from-staged` rather than re-typing an execution. C4's recovery chain exists only if both facts are in the bank.
- **Cost learning needs the backfill.** `llm_tokens` is invisible to the sandbox — supply it at record time or the cost dimension stays unmeasured.
- **The index is derived data.** Leave `{home}/index/*.embedding.json` alone; `record` / `induce` / `retire` maintain it, `rebuild-index` repairs it.
- **Legacy memories stay reachable.** A memory with no vector (`unindexed`) is still yours to query through `inspect` and the profile channel.
- **Retirement is deliberate.** `retire` is irreversible; `--force` on induction lifts a cold-archive veto — reserve both for genuine drift and genuine dead ends.

## References (read on demand)

- [references/world_model_contract.md](references/world_model_contract.md) — the unified prediction contracts (strategy outcome / capability evolution), what `contract_only` means, the attempt-vs-strategy-window scope rule, the `H = F(M, W_OR, Pi, R, T)` capability sources, and the legacy migration table. Read before building a prediction contract or interpreting a stored one.
- [references/modeling.md](references/modeling.md) — the GAMS-style model representation: syntax, constraint label rules, verification layers, and the Coupling-Aware Intermediate Representation (CIR) schema. Read before writing your first model or CIR.
- [references/concepts.md](references/concepts.md) — why the two-layer memory, CostVector dimensions, and disposal ladder are designed this way. Read when you need the "why" behind a mechanism.
- [references/induction.md](references/induction.md) — C1–C6 semantics (including cross-execution recovery), how applicability is read off evidence, the offline lifecycle. Read before your first `induce`, and whenever a hint's meaning is unclear.
- [references/examples.md](references/examples.md) — three complete walkthroughs (cold-start restraint, cost-only learning, a claim meeting its counterexample). Read when unsure how the pieces fit together in practice.
