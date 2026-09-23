---
name: or-harness-wm
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
  → retrieve REAL memory                          (recall: structural + text)
  → YOU propose the candidate methods             (no built-in menu exists)
  → MANDATORY world-model consequence prediction  (predict-outcome)
  → optional multi-step planning                  (plan-next --candidates)
  → compare and EXPLICITLY select
  → model the problem                             (intermediate representation)
  → implement / solve / repair / verify
  → CHECK the answer against the original task    (check-task)
  → record real execution facts
  → update solving context X
  → replan when necessary
  → finish the task
```

1. **Profile** — put your coupling understanding in the task JSON's `coupling` field (a CIR) and run `orx profile --task t.json`. One call returns the validated CIR with `modeling_guidance`, the problem profile, and a derivation report naming each coupling dimension's value and origin. The profile does two jobs: it improves YOUR formulation, and it is the structural key that retrieves comparable evidence. A task with no `model` field is normal — strategy selection needs the task text, the CIR and the profile, never a finished formulation.
2. **Recall** — `orx recall --task t.json --top 3`. This is a report on what the memory really holds, **not a menu**: a strategy appears only if it was really executed (`conditional_stats`) or an admission-verified claim covers it (`strategic_entry`). With nothing in either, `recommendations` is EMPTY and `recommendations_basis.reason` says so. Two independent channels: `recommendations` (structural — the profile decides which evidence is comparable) and `vector_recall` (text similarity, unfiltered by structural cell, so a near-identical problem in another cell is still visible). Read the structural verdict for reuse and the text hits for context; neither substitutes for the other. `--candidate X` restricts the report to methods YOU are considering and lists the memory-less ones under `candidates_without_evidence`. `--exclude` a candidate you distrust and re-recall. An empty structural channel does NOT mean "nothing similar exists" — check whether the text channel is `degraded`.
3. **Propose and choose** — **you name the candidates**; the framework has no directory to enumerate and will not invent one. Weigh quality vs. cost vs. risk yourself. When quality is tied, prefer the cheaper candidate. Pick the solver from `available_solver_families`. Any method id is accepted by `predict`/`execute` — an unknown cost basis comes back as `source="unknown"` with `expected_cost=null`, never a refusal. Freeze the pre-execution expectation with `orx predict --task t.json --strategy S` and pass that snapshot to `record --prediction`: feedback compares against the prediction actually used, never a post-hoc estimate. If you planned with `plan-next`, record your explicit choice with `orx choose-next`.
4. **Model the problem** (AFTER the strategy is chosen) — write the GAMS-style representation (SETS / PARAMETERS / VARIABLES / OBJECTIVE / CONSTRAINTS; [references/modeling.md](references/modeling.md)) as the task JSON's `model` field. The strategy may change the formulation (decomposition, rolling horizon, relaxation). The framework verifies it (L1 format + L2 symbol cross-reference) and reports what its structure says about coupling as a DIAGNOSTIC (`derivation.model_coupling`). The model is a POST-strategy artifact, so it never moves the structural key: the profile you retrieved with is the profile the execution is filed under. Do NOT re-run `profile` expecting the key to change — it will not.
5. **Write solve.py** — follow the method you chose. The framework never generates code and never supplies the method's action list (it is not in any directory — if the memory recorded one for this method, `recall` shows it under `recommendation.knowledge.actions`; otherwise it is absent and you decide). The `model` is the blueprint, the code is a translation. The script must write `result.json` with `status`, `objective_value`, `objective_bound`, `mip_gap`, `runtime_seconds` — and, whenever a task check will need to read the answer, `variables` (variable name → value).
6. **Execute** — `orx execute ...`. Every attempt, success or failure, is staged automatically. Inspect `result.execution.quality.problems` before recording.
7. **Check the ANSWER against the task** — `orx check-task <execution_id> --check '{...}'`. `execute`'s verdict covers the solver's own MODEL; this covers the TASK. Declare the bases that apply (`reference_objective`, `reference_status`, `integer`, `recompute_objective`, `semantic_probe`, `intent`). `failed` means the answer does not satisfy the task — the attempt stays recorded with its real cost but stops counting as a success sample; diagnose the cause yourself (see Recovery) and re-solve in the SAME episode. `insufficient` means the validity is UNKNOWN — not a pass. Never change the task to match a reference value.
8. **Record** — `orx record --execution <json> --override llm_tokens=<actual>`. The response lists `unrecorded_staged_executions` — backfill a failed attempt with `orx record --from-staged <id>` (verbatim, never re-typed).
9. **On failure** — follow Recovery below before retrying.
10. **Decide on induction** — hints are evidence, not orders. Induce only when you judge the pattern worth generalizing.

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

**Induction forms a CLAIM; it does not describe the method.** An entry's `strategy_type` / `actions` / `fallback_strategy_id` are whatever the harness (or a migration) wrote on it — the framework fills in none of them, because there is no directory to copy from and inferring a method's actions from its id would be fabrication. An empty field means the memory does not record it; record it yourself when you know it (`recall` then carries it under `recommendation.knowledge`).

### Structured relation claims

Read the cross-task evidence yourself: identify the **common condition**, compare the differences (comparable? same measurement scope? same difficulty mix?), and state the **claim and its boundary**. Then submit it:

```
orx induce --relation '{"subject": "principle:cross_period_state",
  "claim": "在时间耦合>=0.5 的调度任务上，时间分解必须保留跨期衔接状态",
  "evidence": [{"execution_id": "ex_..", "role": "dropped"},
               {"execution_id": "ex_..", "role": "preserved"}],
  "conditions": {"predicates": {"family": "scheduling",
                                "temporal_coupling": [0.5, 1.0]}},
  "check": {"assertions": [...]}}' \
  --verify '{"purpose": "relation", "check": {"assertions": [...]}}'
```

This is the path for knowledge that is **not** one strategy's statistics — a modeling principle, a necessary condition, a repair pattern. No strategy id is required (`subject` names it), and it does not go through the statistical admission gate. The framework derives the evidence identity (tasks/family/strategy ids) from the recorded facts, so you submit only execution ids and the role each plays. [references/induction.md](references/induction.md) has the assertion semantics and the publication gate.

## Tool policy

| Situation | Action |
|---|---|
| No `--world-model` configured | STOP. The world model is mandatory; `not_configured` blocks both the online loop and offline consolidation. Configure a provider before proceeding. |
| Any task | `recall` → `predict-outcome` the chosen candidate BEFORE executing → `bind-outcome` after. MANDATORY: every executed action feeds the M2 shadow-evaluation channel; an unbound prediction is discarded evidence |
| Any task (the CALIBRATION channel) | `predict-strategy` the candidate BEFORE executing → `bind-strategy` after → the close-out evaluates it. This is the wm-so/1 channel that `orx calibration` aggregates — the M2 `predict-outcome` / `bind-outcome` pair above is a DIFFERENT log and never feeds calibration. Bind at least one per executed action: an unbound strategy prediction is discarded calibration evidence |
| OPTIONAL — high-stakes decision (expensive run, tied candidates, unfamiliar cell) | `plan-next` compares candidates by PREDICTED consequences before choosing; bind the selected path's prediction rather than re-predicting the chosen candidate. This, and only this, is skippable |
| Facing an induction decision | `assess-induction` to evaluate the value BEFORE committing; then accept or reject explicitly |
| No embedding backend configured | The text channel is off: `recall` returns `degraded` plus the structural channel alone |
| Embedding configured, no index yet | `orx rebuild-index` once; `record` / `induce` / `retire` keep it current afterwards |
| Simple problem, one independent constraint, no shared resources | CIR optional — state the skip decision explicitly |
| Empty memory bank (cold start) | `recall` returns `recommendations: []` with a reason — that is the NORMAL cold-start state, not an error. Propose the methods you want to try yourself: the framework ships no directory and generates no menu. They become candidates the moment they really run |
| `recall` returns no recommendations but you expected some | Read `recommendations_basis`: `candidates_with_memory` lists the ids this cell HAS memory for. If your id is not among them, nothing was ever executed or induced for it here — propose it and run it |
| You want to compare a method the framework has never seen | Just name it: `predict --strategy <id>` returns `source: "unknown"` (cost is UNKNOWN, not zero) and `execute --strategy <id>` runs it. No directory membership is checked anywhere |

**Cost discipline**: every world-model call spends real tokens, charged to the decision action or the maintenance scope. `plan-next` and `assess-induction` are avoidable planning overhead — skip them when a plain `recall` already answers the question. `predict-outcome` / `bind-outcome` and `predict-strategy` / `bind-strategy` (one pair each per executed action) and `predict-capability` are MANDATORY: they are the only source of evaluation evidence and are never skipped.

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
| A solve succeeds but the objective looks wrong | Re-derive the model from the task; never adjust constraints to match a reference value |
| `quality.feasible` is true and the objective matches your reference, but the answer may not be a valid one (fractional units for an integer task, an unmet constraint) | `orx check-task <execution_id> --check '{...}'` BEFORE treating it as a success. `execute`'s verdict covers the solver's own MODEL; a relaxed LP is `optimal` with `gap=0` and still wrong. On `failed`: diagnose the cause yourself and re-solve in the SAME episode — the attempt stays recorded with its real cost and becomes contrast evidence, but it can no longer count as a success sample |
| `check-task` reports `insufficient` | The validity is UNKNOWN, not confirmed. Declare the check basis that applies, or make solve.py report `variables` so a domain check can run — an unchecked answer is not evidence that the task was solved |
| A task-result verdict arrives AFTER the episode closed | Record it anyway: `orx check-task` works on recorded executions too. The stored evaluation is never rewritten, but the corrected judgment is used from then on (the sample leaves the calibration means under `exclusions.validity_corrected`) |
| You predicted one strategy/solver and executed another | `bind-outcome` records the mismatch and skips the comparison — bind the action that actually ran |
| A hint rests on repeated runs of one `task_id` | A claim needs ≥2 distinct tasks. Record a genuinely independent instance under its own `task_id` |
| `record` reports `index_sync: deferred` | The fact is saved but not yet text-searchable. Recover with `orx rebuild-index` |
| Requirements, capacity or objective changed | That is a NEW task context: a new `task_id` (or episode). Old X is not inherited, and a prediction is scored against the version it was made under |
| A knowledge class reports `reliability: null` | Too few resolved samples. Honest unknown, not a failure — no value is granted until it has history |
| `orx contract` returns `status: "contract_only"` | No forecast was made, even with a provider configured. `provider_configured`, `service_available` and `prediction_made` are three facts; only the last makes it `valid` |
| Several candidates must be compared on the SAME evidence | `orx context --task t.json` once, then one prediction per candidate with `--context CTX`; `plan-next` does this internally |
| A lesson is NOT one strategy's statistics (a modeling principle, a necessary condition, a repair pattern) | Submit it as a structured relation: `orx induce --relation '{...}' --verify '{"purpose":"relation",...}'`. No strategy id is needed — `subject` names it. Its verdict is its own, and it is published on ≥2 independent tasks. [references/induction.md](references/induction.md) |
| A relation you verified is no longer showing up in `recall` | Check `recall --include-unverified` for its state: `refuted` (an assertion failed on real evidence) or `stale_after_revision` (you changed the claim without re-verifying). Re-submit with `--verify` to re-publish |
| An episode is over | `orx close-episode --task ID --episode EP --terminal STATE`; idempotent. A close-out ends the episode — it does NOT certify the answer (read its `task_checks`). [references/episode_closeout.md](references/episode_closeout.md) |
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

**Task result** (after `execute`, before treating the answer as a success) — run `orx check-task <execution_id> --check '{...}'`:

- the task's own requirements are turned into check bases: a reference value (`reference_objective`, with `tolerance` when your reference is itself approximate), a required status, integer domains (`integer`), an objective recomputation (`recompute_objective`), or explicit value probes (`semantic_probe`). Only what you declare is checked; everything else is listed under `scope.unchecked`;
- a `passed` verdict covers the DECLARED bases only — it is not proof that the model represents the task. If you cannot check something that matters, say so rather than implying it was validated;
- on `failed`: the answer does not satisfy the task. Read `reflection_material` and decide whether the TASK was misread, the MODEL mis-specified, the implementation wrong, or the REFERENCE basis itself wrong. State the basis for whatever you change and re-solve in the SAME episode. Never alter the task to match a reference value, and do not assume every mismatch is a modeling error;
- on `insufficient`: the validity is UNKNOWN. Declare the missing basis or make `solve.py` report `variables`; an unchecked answer must not be cited as evidence that the task was solved;
- a deliberately non-answer (`intent: "relaxation"` / `"intermediate"`) is recorded as such: its solver-side optimum is never presented as the task's answer.

**Relation claim** (before `induce --relation`) — check your own submission:

- every `evidence` entry names a **recorded** execution and the **role** it plays in this claim — the framework refuses unknown ids and missing roles;
- the `check.assertions` cover the parts of your claim you are actually asserting. A claim with two checkable parts needs two assertions; declaring one leaves the other explicitly unchecked in the verification scope;
- `aggregation` matches your quantifier: `"all"` when you mean "on every such task" (one comparable counterexample refutes it), `"mean"` when you mean "on average across this batch";
- `mode` matches your evidence: `"paired"` only compares same-task pairs — if your evidence is not paired, use `"group"` and say so in the claim;
- the `conditions` predicates are the ones you mean; omitted, they are read off the evidence's own cell;
- ≥2 distinct tasks if you intend the claim to be published as knowledge. A single-task relation is a verified fact about that task and will be reported as not published.

## Recovery

- **`check-task` says `failed`** — the answer does not satisfy the task. The framework returns MATERIAL, not a diagnosis: the task text, the artifacts that were produced, the check report and the earlier attempts of the episode. Decide yourself whether the task was MISREAD (a constraint or unit you did not model), the MODEL is mis-specified (wrong variable domain, wrong objective, a missing constraint), the IMPLEMENTATION is wrong (the code does not solve the model you wrote), or the REFERENCE BASIS is wrong (your gold value or tolerance is off). Report your reasoning as a `model` action with `revised_from` naming the failed execution, then re-solve in the SAME episode. Do not rebuild from scratch by default, do not assume the failure is a modeling error, and NEVER change the task to match a reference value.
- **`check-task` says `insufficient`** — nothing was confirmed either way. Declare the check basis that applies, or make `solve.py` report `variables`. Do not treat an unchecked answer as validated.
- **`status=error`, security policy** — the script used a blocked construct (network/shell/pathlib/dynamic `open()` paths). Use only stdlib and a literal `open('result.json', 'w')`. Subprocess-based solvers (PuLP's CBC) cannot run in the sandbox — switch to an in-process solver (ortools, highspy). Record the failed attempt with `--from-staged` so the memory learns it.
- **`status=error`, traceback** — fix the model or script and re-execute. Each attempt is its own record: a first failure is `retries=0` (an observed zero); a retry declares `--override retries=N` (the absolute count of NEW retries).
- **`status=timeout`** — the strategy may be too heavy for this scale. Re-recall with `--exclude <strategy>`, try the next candidate, and record the timeout.
- **`close-episode` refuses with `state: "pending"`** — an action of the episode is still `running`, and a running scope has no final numbers. End it (or let it finish), then close. If a `running` action is left over from an `execute` that raised before producing a record, it was already ended as `failed` by the framework — a genuinely stranded one means the process died mid-call.
- **`recall` returns no candidates** — read `recommendations_basis.reason`. On a cold bank this is the EXPECTED state, not an error: propose the methods you want to try and run them. If you expected memory for a specific id, check `candidates_with_memory` — an id absent from it was never executed or induced in this cell. A `vector_recall` hit for a structurally different problem is a strong hint that the coupling values are off.
- **A `coupling_warnings` entry at profile time** — your supplied value contradicts the derivation. The derived value wins for grouping; correct your annotation.

## Core concepts

- **Execution Evidence Bank** — append-only episodic facts: the strategy actually used, the quality/cost actually observed, failures, artifacts. Never stores generalizations. Only cost dimensions may be backfilled.
- **Strategic Knowledge Bank** — induced commitments (expected quality, cost, failure risk, prediction intervals, applicability read off evidence). Mutation happens at INDUCTION time only. Creating an entry takes ≥2 executions from ≥2 distinct tasks; publishing it takes a passed admission check (`induce --verify`).
- **Conditional statistics** — on-the-fly aggregation per (strategy × structural cell); a recount, never persisted.
- **No strategy directory** — the framework ships no list of method names, descriptions, applicability rules, actions or fallbacks, and nothing loads one. A strategy is a candidate because the MEMORY holds something about it here (a recorded execution, or an admission-verified claim). With neither, `recall` is empty and says so. You propose the methods; `predict`/`execute` accept any id, and an unknown cost basis is reported as `unknown`, never as zero.
- **CostVector** — five dimensions, stored raw and never folded: `llm_tokens, tool_calls, solver_runtime_s, retries, latency_s`. Each record carries a measured-dimension mask meaning "this value is a real observation": the executor measures only `latency_s` and `solver_runtime_s`, while `tool_calls` (ALL tool invocations — commands, sandbox runs, solver calls), `retries` and `llm_tokens` are YOUR declarations and stay unknown until supplied. Unknown ≠ zero. A strategic entry publishes a cost dimension only when every supporting record measured it.
- **Cold archive** — cards for retired entries that veto re-induction of the same failed generalization; `induce --force` lifts that veto when you judge the environment has drifted.

Detailed semantics — structural grouping, coupling derivation, CIR, memory layers, the disposal ladder and the verification philosophy — are in [references/concepts.md](references/concepts.md).

## Commands

Every command prints one JSON line: `{"result": {...}, "summary": "..."}`. Exit codes: `0` success, `2` usage/precondition error, `1` crash. All accept `--home DIR` (default `$OR_HARNESS_HOME`, else `./or_harness_home`). Exact arguments and output semantics: [references/commands.md](references/commands.md).

| Command | Use it for |
|---|---|
| `profile` | The analysis entry: CIR validation + modeling guidance + profile + derivation. Run it first, and again after writing the `model` |
| `recall` | Both retrieval channels: structural `recommendations[]` + text `vector_recall`. A REPORT on real memory, not a menu — an empty list plus `recommendations_basis` is a valid answer |
| `predict` | Freeze the pre-execution cost expectation; pass the snapshot to `record --prediction`. Any strategy id is accepted; no evidence means `source="unknown"` |
| `execute` | Sandbox-run solve.py and stage the execution |
| `check-task` | Check whether an execution's ANSWER satisfies the original task (not merely that the solver solved its own model) |
| `record` | Persist the fact: cost backfill → quality checks → cost feedback → induction-pattern hints |
| `amend-cost` | Backfill cost dimensions of an ALREADY-RECORDED execution (the repair path for a reported cost gap) |
| `induce` | The only place knowledge changes |
| `inspect` | Query one memory layer |
| `snapshot` / `action` / `budget` | Freeze belief state / report an action YOU performed / consumption view |
| `predict-outcome` / `bind-outcome` | World-model shadow prediction, then its comparison against the real action |
| `predict-strategy` / `bind-strategy` | Strategy-outcome prediction (wm-so/1) over one frozen context + one candidate; then the prediction–execution linkage |
| `plan-next` / `choose-next` | Bounded planning over predicted consequences, then your explicit choice. `--candidates` is REQUIRED — the planner generates no menu |
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
- **A solved model is not a solved task.** `execute`'s verdict covers the solver's own model; `check-task` covers the TASK. A relaxed answer that matches your reference is still wrong, and a confirmed-wrong answer stops being a success sample without leaving the evidence set (its cost is real). Task validity and knowledge validity are separate gates.
- **A close-out is not a certification.** `close-episode` ends the episode. It reports its `task_checks` coverage precisely so "the episode closed" is never read as "the answer was right".
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
- [references/episode_closeout.md](references/episode_closeout.md) — closing an episode, the real-outcome summary, per-field evaluation, the experience calibration channel, and how a task-result verdict enters it.
- Runnable examples: [`references/examples/task_check.py`](references/examples/task_check.py) (check → diagnose → repair → record → close-out) and [`references/examples/episode_closeout.py`](references/examples/episode_closeout.py) (predict → execute → bind → close → calibrate). Read them when unsure how the pieces fit together.
