---
name: or-harness
description: >
  Formulate, solve, retry, decompose, validate and debug large-scale industrial
  optimization problems (LP, MILP, scheduling, routing, assignment, network
  flow, capacity planning, resource allocation, supply chain) whose decisions
  are highly coupled through shared resources, cross-stage dependencies or
  temporal propagation. Use when the user asks to formulate, solve, retry,
  decompose, validate or debug an optimization model; to compare candidate
  solving strategies by expected quality/cost/risk; to retrieve or manage
  accumulated execution evidence and reusable strategic knowledge; or to run
  offline consolidation over completed tasks. Provides coupling-aware problem
  understanding, a sandboxed executor with seven solver adapters, a two-layer
  strategy memory, and a required world-model consequence predictor. Do not use for
  one-off optimization questions with no repetition, generic math proofs, or
  non-optimization tasks.
---

# OR-Harness: strategy learning for optimization agents

You are the orchestrator; this layer advises, executes and remembers. Every command is a stateless call against an explicit memory directory (`--home` or `$OR_HARNESS_HOME`) — no hidden state, no background work. You may refuse any recommendation, request alternatives, execute without recording, override recorded costs, and you alone decide when to consolidate.

The world model is a REQUIRED component of both the online loop and offline consolidation: configure a provider (`--world-model URL::MODEL` or `ORHarness(world_model=...)`). If none is configured, `not_configured` is a BLOCKER — the loop cannot run to spec, and that is a configuration error, not a legitimate degraded mode. Multi-step planning (`plan-next`) remains optional.

## Invariants (do not reinterpret these)

```
profile / coupling-aware understanding  BEFORE  detailed formulation
recommendation ≠ selection ≠ execution
prediction ≠ fact        hypothetical ≠ observed        unknown ≠ 0
online solving ≠ offline consolidation
no world-model parameter update inside an episode
predict-outcome / bind-outcome / predict-capability  are REQUIRED
plan-next / choose-next                               are OPTIONAL
Execution Evidence ≠ Strategic Knowledge
knowledge growth ≠ demonstrated capability improvement
predicted cost ≠ measured cost
```

## Online solving

```
Problem P
  → profile / coupling-aware understanding        (CIR + profile + guidance)
  → retrieve evidence and strategic knowledge     (recall: structural + text)
  → generate candidate strategies
  → MANDATORY world-model consequence prediction  (predict-outcome)
  → optional multi-step planning                  (plan-next)
  → compare and EXPLICITLY select
  → model the problem                             (intermediate representation)
  → implement / solve / repair / verify
  → record real execution facts
  → update solving context X
  → replan when necessary
  → finish the task
```

1. **Profile** — put your coupling understanding in the task JSON's `coupling` field (a CIR) and run `orx profile --task t.json`. One call returns the validated CIR with `modeling_guidance`, the problem profile, and a derivation report naming each coupling dimension's value and origin. The profile does two jobs: it improves YOUR formulation, and it is the structural key that retrieves comparable evidence. A task with no `model` field is normal — strategy selection needs the task text, the CIR and the profile, never a finished formulation.
2. **Recall and compare** — `orx recall --task t.json --top 3`. Two independent channels: `recommendations` (structural — the profile decides which evidence is comparable) and `vector_recall` (text similarity, unfiltered by structural cell, so a near-identical problem in another cell is still visible). Read the structural verdict for reuse and the text hits for context; neither substitutes for the other. `--exclude` a candidate you distrust and re-recall.
3. **Choose** — weigh quality vs. cost vs. risk yourself. When quality is tied, prefer the cheaper candidate. Pick the solver from `available_solver_families`. Freeze the pre-execution expectation with `orx predict --task t.json --strategy S` and pass that snapshot to `record --prediction`: feedback compares against the prediction actually used, never a post-hoc estimate. If you planned with `plan-next`, record your explicit choice with `orx choose-next`.
4. **Model the problem** (AFTER the strategy is chosen) — write the GAMS-style representation (SETS / PARAMETERS / VARIABLES / OBJECTIVE / CONSTRAINTS; [references/modeling.md](references/modeling.md)) as the task JSON's `model` field. The strategy may change the formulation (decomposition, rolling horizon, relaxation). The framework verifies it (L1 format + L2 symbol cross-reference) and reports what its structure says about coupling as a DIAGNOSTIC (`derivation.model_coupling`). The model is a POST-strategy artifact, so it never moves the structural key: the profile you retrieved with is the profile the execution is filed under. Do NOT re-run `profile` expecting the key to change — it will not.
5. **Write solve.py** — follow the chosen strategy's `actions`; the framework never generates code. The `model` is the blueprint, the code is a translation. The script must write `result.json` with `status`, `objective_value`, `objective_bound`, `mip_gap`, `runtime_seconds`.
6. **Execute** — `orx execute ...`. Every attempt, success or failure, is staged automatically. Inspect `result.execution.quality.problems` before recording.
7. **Verify, then record** — see Verification below; the check comes first. `orx record --execution <json> --override llm_tokens=<actual>`. The response lists `unrecorded_staged_executions` — backfill a failed attempt with `orx record --from-staged <id>` (verbatim, never re-typed).
8. **On failure** — follow Recovery below before retrying.
9. **Decide on induction** — hints are evidence, not orders. Induce only when you judge the pattern worth generalizing.

## Offline consolidation (a separate timescale, never inside an episode)

```
completed episode / trajectory
  → close out the real outcome and prediction feedback   (close-episode)
  → archive Execution Evidence
  → form induction / revision candidates
  → MANDATORY capability-evolution prediction            (predict-capability)
  → accept / reject / defer explicitly                   (accept-/reject-capability)
  → real induction / revision                            (induce)
  → verification                                         (induce --verify)
  → publish eligible Strategic Knowledge
  → evaluate reuse on later real tasks                   (evaluate-capability)
```

Induction never runs automatically after an online action. A knowledge change is observable immediately; a capability improvement requires later real tasks or an independent evaluation — never the operation's own report.

## Tool policy

| Situation | Action |
|---|---|
| No `--world-model` configured | STOP. The world model is mandatory; `not_configured` blocks both the online loop and offline consolidation. Configure a provider before proceeding. |
| Any task | `recall` → `predict-outcome` the chosen candidate BEFORE executing → `bind-outcome` after. MANDATORY: every executed action feeds calibration; an unbound prediction is discarded evidence |
| OPTIONAL — high-stakes decision (expensive run, tied candidates, unfamiliar cell) | `plan-next` compares candidates by PREDICTED consequences before choosing; bind the selected path's prediction rather than re-predicting the chosen candidate. This, and only this, is skippable |
| Facing an induction decision | `assess-induction` to evaluate the value BEFORE committing; then accept or reject explicitly |
| No embedding backend configured | The text channel is off: `recall` returns `degraded` plus the structural channel alone |
| Embedding configured, no index yet | `orx rebuild-index` once; `record` / `induce` / `retire` keep it current afterwards |
| Simple problem, one independent constraint, no shared resources | CIR optional — state the skip decision explicitly |

**Cost discipline**: every world-model call spends real tokens, charged to the decision action or the maintenance scope. `plan-next` and `assess-induction` are avoidable planning overhead — skip them when a plain `recall` already answers the question. `predict-outcome` / `bind-outcome` (one pair per executed action) and `predict-capability` are MANDATORY: they are the only source of calibration evidence and are never skipped.

## Recognition table (what to do when you see this)

| Situation | Action |
|---|---|
| Shared resources, cross-stage dependencies, or temporal propagation | Extract a CIR and submit it as `coupling` before choosing a strategy |
| `profile` exits 2 with `error.kind: cir_format` | The CIR shape is wrong — read `error.hint` (it names the offending key) and resubmit. A malformed CIR is never silently read as an empty one |
| `profile` returns non-empty `coupling.cir.issues` | Fix the CIR and re-run `profile` before writing solve.py |
| `coupling.health.contributes_scalars` is false | The CIR parsed but derives no dimension (no decisions, or empty). The profile's coupling then comes from the spec or stays `null` — `health.note` says which |
| `recall` or `orx context` reports `degraded` | Only the structural channel ran. No backend, no task text, a missing index and a backend failure are four DIFFERENT facts, and all four differ from "ran and matched nothing" — an empty `vector_recall` is never evidence that no similar memory exists |
| A `vector_recall` hit is `different_cell` | Read its text and outcome for context; `profile_cell` shows which structure its numbers describe |
| A knowledge hit has `reusable: false` | `reason` names the conflict, or `unknown` — a needed value is missing; measure it before applying |
| You are about to treat a `similarity` as a quality or cost estimate | It is a DISCOVERY signal. Read the hit's `evidence_class` and applicability label; a `different_cell` hit is context to read, never a statistic to apply |
| `quality.problems` is non-empty | Fix the script and re-execute; the record waits for the check |
| Solver returns `infeasible` | Check bounds and conflicting constraints; if genuinely infeasible, record it — valuable evidence |
| A solve succeeds but the objective looks wrong | Re-derive the model from the task; never adjust constraints to match a reference value |
| You predicted one strategy/solver and executed another | `bind-outcome` records the mismatch and skips the comparison — bind the action that actually ran |
| A hint rests on repeated runs of one `task_id` | A claim needs ≥2 distinct tasks. Record a genuinely independent instance under its own `task_id` |
| `record` reports `index_sync: deferred` | The fact is saved but not yet text-searchable. Recover with `orx rebuild-index` |
| Requirements, capacity or objective changed | That is a NEW task context: a new `task_id` (or episode). Old X is not inherited, and a prediction is scored against the version it was made under |
| A knowledge class reports `reliability: null` | Too few resolved samples. Honest unknown, not a failure — no value is granted until it has history |
| `orx contract` returns `status: "contract_only"` | No forecast was made, even with a provider configured. `provider_configured`, `service_available` and `prediction_made` are three facts; only the last makes it `valid` |
| Several candidates must be compared on the SAME evidence | `orx context --task t.json` once, then one prediction per candidate with `--context CTX`; `plan-next` does this internally |
| An episode is over | `orx close-episode --task ID --episode EP --terminal STATE`; idempotent. [references/episode_closeout.md](references/episode_closeout.md) |
| An execution fact is WRONG and must stop counting | `orx exclude-execution --execution EX --reason "..."` — the bank is append-only, so the row is preserved but `source` becomes `excluded` and it leaves every statistics/induction/retrieval path. Reverse with `orx restore-execution`. [references/commands.md](references/commands.md) |
| A prediction must be bound to a real run | `orx bind-outcome --prediction ID --action <ac_... OR the ex_... id `orx execute` printed>` |
| You want to know what past predictions were worth | `orx calibration` (closed episodes only; below the sample minimum it reports `insufficient_evidence`) and `orx evaluations` |
| Deciding whether an OFFLINE improvement is worth doing | `orx predict-capability` per candidate, then `orx compare-capability`. It recommends the largest NET saving over the declared window (cumulative saving minus the one-time maintenance cost, SAME unit) with no quality degradation; cross-unit savings, non-paying candidates, undeclared windows and unexecutable operation types are reported, not ranked, and `defer` is legitimate |
| You accept a capability recommendation | `orx accept-capability --recommendation <json>` — the only offline entry that changes knowledge, and it runs the operation the prediction was about (a `retire` retires; an unsupported type is refused). `reject-capability` declines and changes nothing |
| A capability operation has run | `orx bind-capability --prediction ID` records the FACT (knowledge delta, real cost, scope consistency). It never sets `effect_verified` |
| You want to know whether the improvement helped | Wait for qualified LATER tasks, then `orx evaluate-capability --prediction ID`. Only closed episodes of tasks whose WORK ran after the operation, inside the frozen target and outside its own scope count, and only once the declared horizon is met — otherwise it stays `pending`. `orx capability-feedback` shows fact vs effect |

## Verification (before every record)

**CIR** (at `orx profile`, before choosing a strategy) — check `result.coupling.cir`, `coupling.health` and `modeling_guidance`:

- `coupling.health.contributes_scalars` is true, or `health.note` explains why not — a CIR that parses to nothing derives no dimension, so the profile's coupling is NOT a structural measurement;
- every resource, product, site, period or route in the task description appears as an entity or is reachable via a decision's indexes;
- every `uses_resource` / `shares_resource` / `competes_for` relation connects a decision to a resource-like entity, not two decisions with no shared resource;
- each detected coupling group matches a coupling pattern the task actually implies (e.g. "all modes use the same downstream capacity" → `shared_bottleneck` on that capacity);
- `coupling.cir.issues` is empty.

**Model** (after writing the `model`, before writing solve.py) — check `result.derivation`:

- `model_verification.issues` is empty (L1 format + L2 symbol cross-reference);
- `model_coupling` is present and plausible — it is a DIAGNOSTIC reading of the model's own structure, so a divergence from the profile's `origin` dimensions is worth understanding, but it never changes the structural key.

**Execution** (after `execute`, before `record`) — check `result.execution`:

- `quality.status` is one of optimal/feasible/infeasible/unbounded/timeout/ error — anything else is a broken script, not a solver result;
- `quality.feasible` is true and `quality.objective` is finite when a solution is expected;
- `quality.gap` is recorded (derived from the bound when the solver omits it);
- `quality.problems` is empty;
- the objective is plausible for the problem (right order of magnitude, right min/max direction) — the sandbox checks structure, not semantics.

## Recovery

- **`status=error`, security policy** — the script used a blocked construct (network/shell/pathlib/dynamic `open()` paths). Use only stdlib and a literal `open('result.json', 'w')`. Subprocess-based solvers (PuLP's CBC) cannot run in the sandbox — switch to an in-process solver (ortools, highspy). Record the failed attempt with `--from-staged` so the memory learns it.
- **`status=error`, traceback** — fix the model or script and re-execute. Each attempt is its own record: a first failure is `retries=0` (an observed zero); a retry declares `--override retries=N` (the absolute count of NEW retries).
- **`status=timeout`** — the strategy may be too heavy for this scale. Re-recall with `--exclude <strategy>`, try the next candidate, and record the timeout.
- **`recall` returns no candidates** — no strategy's applicability matches the profile. Check the coupling dims; a `vector_recall` hit for a structurally different problem is a strong hint that the coupling values are off.
- **A `coupling_warnings` entry at profile time** — your supplied value contradicts the derivation. The derived value wins for grouping; correct your annotation.

## Core concepts

- **Execution Evidence Bank** — append-only episodic facts: the strategy actually used, the quality/cost actually observed, failures, artifacts. Never stores generalizations. Only cost dimensions may be backfilled.
- **Strategic Knowledge Bank** — induced commitments (expected quality, cost, failure risk, prediction intervals, applicability read off evidence). Mutation happens at INDUCTION time only. Creating an entry takes ≥2 executions from ≥2 distinct tasks; publishing it takes a passed admission check (`induce --verify`).
- **Conditional statistics** — on-the-fly aggregation per (strategy × structural cell); a recount, never persisted.
- **CostVector** — five dimensions, stored raw and never folded: `llm_tokens, tool_calls, solver_runtime_s, retries, latency_s`. Each record carries a measured-dimension mask meaning "this value is a real observation": the executor measures only `latency_s` and `solver_runtime_s`, while `tool_calls` (ALL tool invocations — commands, sandbox runs, solver calls), `retries` and `llm_tokens` are YOUR declarations and stay unknown until supplied. Unknown ≠ zero. A strategic entry publishes a cost dimension only when every supporting record measured it.
- **Cold archive** — cards for retired entries that veto re-induction of the same failed generalization; `induce --force` lifts that veto when you judge the environment has drifted.

Detailed semantics — structural grouping, coupling derivation, CIR, memory layers, the disposal ladder and the verification philosophy — are in [references/concepts.md](references/concepts.md).

## Commands

Every command prints one JSON line: `{"result": {...}, "summary": "..."}`. Exit codes: `0` success, `2` usage/precondition error, `1` crash. All accept `--home DIR` (default `$OR_HARNESS_HOME`, else `./or_harness_home`). Exact arguments and output semantics: [references/commands.md](references/commands.md).

| Command | Use it for |
|---|---|
| `profile` | The analysis entry: CIR validation + modeling guidance + profile + derivation. Run it first, and again after writing the `model` |
| `recall` | Both retrieval channels: structural `recommendations[]` + text `vector_recall` |
| `predict` | Freeze the pre-execution cost expectation; pass the snapshot to `record --prediction` |
| `execute` | Sandbox-run solve.py and stage the execution |
| `record` | Persist the fact: cost backfill → quality checks → cost feedback → induction-pattern hints |
| `amend-cost` | Backfill cost dimensions of an ALREADY-RECORDED execution (the repair path for a reported cost gap) |
| `induce` | The only place knowledge changes |
| `inspect` | Query one memory layer |
| `snapshot` / `action` / `budget` | Freeze belief state / report an action YOU performed / consumption view |
| `predict-outcome` / `bind-outcome` | World-model shadow prediction, then its comparison against the real action |
| `predict-strategy` / `bind-strategy` | Strategy-outcome prediction (wm-so/1) over one frozen context + one candidate; then the prediction–execution linkage |
| `plan-next` / `choose-next` | Bounded planning over predicted consequences, then your explicit choice |
| `context` | Build or read the FROZEN prediction input context (no model call, no solver) |
| `contract` | Build or read a unified world-model contract (no model call) |
| `close-episode` / `calibration` / `evaluations` | Episode close-out, experience calibration, per-prediction evaluation records |
| `assess-induction` / `bind-induction-outcome` | Evaluate an induction's value before committing; judge it afterwards |
| `predict-capability` / `compare-capability` | Predict what ONE offline learning operation would change; compare frozen predictions |
| `accept-capability` / `reject-capability` | Your explicit offline choice: accept runs the real operation, reject changes nothing |
| `bind-capability` / `evaluate-capability` / `capability-feedback` | Bind the maintenance FACT, then judge the EFFECT on later real tasks |
| `gc` / `retire` / `rebuild-index` / `doctor` | Derived-layer disposal; index build/repair; environment self-check |

## Decision guidance

- **induction hints after record** are evidence, not orders. Induce when you judge the pattern worth generalizing; you may also induce with no hint. Semantics: [references/induction.md](references/induction.md).
- **status vs verification** — `status` tracks how an entry has behaved (`candidate` / `validated` / `suspect` / `dormant`); `verification.state` tracks whether the CLAIM was checked. They cannot contradict: `validated` requires `verified`, and a `refuted` claim can never be `validated`. Both are applied by `induce`, never by `record`.
- **When to gc** — when `inspect` shows large groups fully covered by validated entries. Run `--dry-run` first.

## Output requirements

Report to the user: the chosen strategy and solver with the evidence that drove the choice; the objective value and key decisions with the status stated explicitly; the assumptions you made and data you had to interpret; the real cost you paid including the `llm_tokens` backfill; and any failed attempt that is part of the story. A plan suggestion is a recommendation — say which candidate you chose and whether you deviated.

## Discipline

- **A signal is not the decision.** A hint, a `similarity`, a model's self-reported `confidence` and a plan suggestion are inputs — the choice and the admission verdict remain yours or the framework's.
- **A prediction is a shadow.** `plan-next` suggests, `choose-next` decides, `execute` acts. A prediction never becomes a fact, and an unexecuted candidate's prediction is never real feedback.
- **Similarity discovers; structure decides reuse.** `applies` may be reused; `conflicts` and `unknown` may not — an unmeasured condition is not a satisfied one. One memory hit by both channels is ONE piece of evidence.
- **Measure coupling from structure.** Two independent resource constraints mean rc≈0, however resource-flavoured the task sounds.
- **The model comes after the strategy, before the code.**
- **Predict before acting.** A prediction made after the execution is hindsight, and binding it to a different strategy's action records a mismatch instead of a score.
- **An attempt is not a strategy window.** One `execute` call is one attempt; modeling/repair/verify is auxiliary overhead. The same strategy chosen again is a different selection round.
- **Failures are raw material.** Record every attempt, including timeouts and infeasibilities; backfill from staging rather than re-typing.
- **Cost learning needs the backfill.** `llm_tokens`, `tool_calls` and `retries` are invisible to the sandbox — the executor measures only `latency_s` and `solver_runtime_s`. Supply them at record time (`--override`) or with `amend-cost`, or the dimension stays unmeasured and no cost claim can be published for it.
- **A knowledge entry appearing is not a capability gain.** Predicting, binding the fact and verifying the effect are three different things; only the third supports "the harness got stronger".
- **The sample is counted in TASK-EPISODES.** Ten predictions bound to one execution are ONE independent truth, and a task counts as later only when its WORK ran after the operation.
- **Retirement is deliberate.** `retire` is irreversible; `--force` on induction lifts a cold-archive veto. Reserve both for genuine drift and genuine dead ends.
- **The index is derived data.** Leave `{home}/index/*.embedding.json` alone; `record` / `induce` / `retire` maintain it, `rebuild-index` repairs it.

## References

Read on demand — one hop, no chains.

- [references/commands.md](references/commands.md) — exact CLI/API arguments and output semantics for every command. Read before calling something whose flags you do not remember.
- [references/concepts.md](references/concepts.md) — the "why": two-layer memory, structural grouping and coupling derivation, CIR, CostVector, disposal ladder, verification philosophy.
- [references/modeling.md](references/modeling.md) — the GAMS-style model syntax, constraint label rules, verification layers, and the CIR schema. Read before writing your first model or CIR.
- [references/induction.md](references/induction.md) — the four induction-worthy patterns, applicability as family + structural cell, the creation gate and admission verification, the offline lifecycle.
- [references/world_model_contract.md](references/world_model_contract.md) — the unified prediction contracts, `contract_only`, attempt vs strategy window, the capability sources of H, the legacy migration table.
- [references/prediction_context.md](references/prediction_context.md) — the frozen prediction input context: joint representation, math-attribute origins, the two retrieval channels and their evidence classes.
- [references/strategy_outcome.md](references/strategy_outcome.md) — the strategy-outcome prediction service (wm-so/1): protocol, comparison yardstick, prediction–choice–execution binding.
- [references/episode_closeout.md](references/episode_closeout.md) — closing an episode, the real-outcome summary, per-field evaluation, the experience calibration channel.
- [references/examples.md](references/examples.md) — complete walkthroughs. Read when unsure how the pieces fit together.
