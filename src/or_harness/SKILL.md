---
name: or-harness-wm
description: >
  Formulate, solve, debug, validate, and improve operations research and
  optimization problems — LP, MILP, scheduling, routing, assignment,
  network flow, resource allocation, supply-chain, and other highly coupled
  industrial OR scenarios. Provides coupling-aware problem understanding
  (CIR), a sandboxed executor with seven solver adapters, and a two-layer
  strategy memory (Execution Evidence facts + Strategic Knowledge
  commitments) that learns which solving strategies fit which problem
  structures at what execution cost. Use when the user asks to formulate, solve, retry,
  decompose, validate, or debug an optimization model, or to query/manage
  accumulated OR strategy experience. Do not use for one-off optimization
  questions with no repetition, generic math proofs, or non-optimization tasks.
---

# OR-Harness: Strategy Learning for Optimization Agents

You are the orchestrator. This capability layer only advises and executes — you retain full control: you may refuse any recommendation, request alternatives, execute without recording, override recorded costs, and you alone decide when to induce and when to collect garbage.

OR-Harness never calls an LLM, never runs autonomously, and keeps no hidden state: every command is a stateless call against an explicit memory directory (`--home` or `$OR_HARNESS_HOME`).

## Core workflow (the loop to run for every optimization task)

1. **Profile (the single analysis entry)** — submit your coupling understanding as the task JSON's optional `coupling` field (a CIR — Coupling-Aware Intermediate Representation) and run `orx profile --task t.json`. One call returns: the validated CIR with `modeling_guidance` (explicit, inspectable coupling groups), the problem profile, and the derivation report (each coupling dimension's value and origin). The profile serves two purposes: it helps YOU understand the problem structure, and it is the structural key that retrieves comparable evidence in the next step. **A task without a `model` field is a normal state** — strategy selection needs the task text, the CIR, and the profile, never a finished formulation.
2. **Recall and compare strategies** — `orx recall --task t.json --top 3`. The profile is only the retrieval key: it decides WHICH evidence is comparable to this problem, not which strategy to pick. The choice itself rests on each candidate's expected quality, expected cost, and failure risk (`score = α·Q̂ − β·Ĉ − γ·R̂`), read from `evidence`, `confidence`, and `risk_warnings`; check `solver_advisories` for solvers that failed in this environment before. When there is no experience yet, candidates return with `evidence="no_memory"`, `score=-inf`, `confidence=0` — the catalog is a structural vocabulary (applicability, actions, fallback, solver family), not a source of fabricated priors. You may `--exclude` any candidate and re-recall. **Optionally**: with a world model configured, `orx plan-next` compares candidates by their PREDICTED consequences (same quality/cost/risk yardstick) and suggests a first step.
   *Verify*: every candidate you seriously consider carries either real evidence (`evidence != "no_memory"`) or an explicit structural-fit justification from the catalog — if you cannot articulate why a `no_memory` candidate fits, exclude it and re-recall rather than defaulting to the first row.
3. **Choose** — weigh quality vs. cost vs. risk yourself. When quality estimates are tied, prefer the cheaper candidate (that preference is exactly what this memory exists to learn). Pick the concrete solver from `available_solver_families`, heeding advisories. Freeze the pre-execution cost expectation with `orx predict --task t.json --strategy S` and pass the returned snapshot back at record time (`--prediction`) — feedback compares against the prediction actually used, never a post-hoc estimate. Unknown cost is reported as unknown, never as zero. If you planned with `plan-next`, record your explicit choice with `orx choose-next` (accept, deviate, or reject).
4. **Model the problem (intermediate representation, after the strategy is chosen)** — NOW write the GAMS-style model representation (SETS / PARAMETERS / VARIABLES / OBJECTIVE / CONSTRAINTS; see [references/modeling.md](references/modeling.md)) as the task JSON's top-level `model` field. The model is the blueprint for solve.py, written under the chosen strategy: decomposition, rolling horizon, or relaxation strategies may alter the formulation and execution plan. The framework verifies it (L1 format + L2 symbol cross-reference) and derives exact structural coupling from the declared constraints. Re-run `orx profile` to get the CIR ↔ model cross-check (`cir_warnings`, e.g. a CIR decision not declared as a model variable) — this is how you catch "the model missed a coupling the CIR declared" before coding.
5. **Write solve.py** — follow the chosen strategy's `actions` (the framework never generates code). The model from step 4 is your blueprint — the code is a translation, not a re-derivation. The script must write `result.json` with `status, objective_value, objective_bound, mip_gap, runtime_seconds`.
6. **Predict cost (optional world-model shadow)** — with a world model configured (`--world-model`), you may ask for a structured outcome prediction of the candidate execution (`orx predict-outcome`) BEFORE executing — it never changes your choice, and you compare it afterwards (`orx bind-outcome`) to accumulate calibration evidence.
7. **Execute** — `orx execute ...`. Every execution — successes AND failures — is automatically staged in a pending area (never lost, even if you immediately retry). Inspect `result.execution.quality.problems` before recording. If you made a shadow prediction, bind it now (`orx bind-outcome --prediction <id> --action <action_id from the execute output>`).
8. **Verify** (see Verification below) — do not record an execution you have not checked.
9. **Record** — `orx record --execution <json> --override llm_tokens=<your actual token count> [--override-mode replace|increment] [--prediction <predict-output.json>]`. `--override` default is `replace` (idempotent — the value IS the measurement; re-applying never double-counts); use `increment` only for an additional measured amount within the same attempt. The response lists `unrecorded_staged_executions` for this task — if a failed first attempt is sitting there, backfill it with `orx record --from-staged <id>` (verbatim, no re-typing). Read the returned `induction_hints`, `prediction_checks`, and `cost_feedback` (computed only against the prediction snapshot you passed).
   *Verify*: `recorded: true` in the response; if you passed `--prediction` and both `prediction_checks` and `cost_feedback` are absent, the snapshot did not match this execution (wrong strategy or scope) — re-check which prediction you froze in step 3, never re-predict post-hoc.
10. **Decide on induction** — hints are evidence, not orders. Induce only when you judge the pattern worth generalizing.
11. **On failure** — follow Recovery below before retrying.

## Tool policy (when to invoke which capability)

| Situation | Action |
|---|---|
| No `--world-model` configured | Run the core loop (steps 1–5, 7–9) directly; every world-model command returns an explicit `not_configured` — never a silent skip |
| World model configured + routine task with existing evidence | `recall` is enough; optionally `predict-outcome` one candidate for calibration |
| World model configured + high-stakes decision (expensive execution, tied candidates, unfamiliar cell) | `plan-next` — compare candidates by PREDICTED consequences before choosing |
| Facing an induction decision (hints accumulated, ≥2 tasks of evidence) | `assess-induction` — evaluate the induction's value BEFORE committing; accept/reject explicitly |
| Budget declared and near its limit | Check `orx budget` before each model call; an exceeded budget stops planning automatically |
| Simple problem, single independent constraint, no shared resources | CIR optional — state the skip decision explicitly |

**Cost discipline**: every world-model call spends real tokens (charged to the decision action / maintenance scope). Do not plan or assess when a plain `recall` already answers the question with published knowledge.

## Coupling dimensions (operational definitions)

Structural grouping — the foundation of all memory — keys on these. Supply them accurately or let the framework derive them (priority: CIR structure > model > spec):

| Dimension | Measures | Derivable? |
|---|---|---|
| resource_coupling | fraction of decisions involved in a resource relation (CIR) / fraction of decision variables appearing in MORE THAN ONE constraint (model) | yes — cir > model > spec |
| temporal_coupling | fraction of decisions/variables indexed by a temporal set (time/period/stage/...) | yes — cir > model > spec |
| route_complexity | fraction of decisions/variables indexed by a network set (arc/edge/link/...) | yes — cir > model > spec |
| semantic_coupling | business-semantic relatedness — invisible to structure | NO — always your call |

With a CIR present, `resource_coupling` = fraction of decisions that are the source of ≥1 `uses_resource`/`shares_resource`/`competes_for` relation; `temporal_coupling`/`route_complexity` = fraction of decisions with time-like/network-like indexes. Without a CIR, the model-based definitions apply.

Do not guess these from the problem's *name* ("it's a resource allocation problem, so rc must be high") — measure from the constraint structure. Two independent resource constraints means rc≈0, however resource-flavored the problem sounds.

## Verification (before every record)

### CIR verification (at `orx profile`, before choosing a strategy)

Check `result.coupling.cir` and `result.coupling.modeling_guidance`:
- **Entity coverage**: every resource, product, site, period, or route mentioned in the task description appears as an entity or is reachable via a decision's indexes.
- **Relation plausibility**: every `uses_resource` / `shares_resource` / `competes_for` relation connects a decision to a resource-like entity — not two decisions with no shared resource.
- **Coupling group alignment**: the detected `shared_bottleneck` or agent-declared coupling groups correspond to coupling patterns the task description actually implies (e.g. "all modes use the same downstream capacity" → `shared_bottleneck` on that capacity).
- **No unresolved issues**: `result.coupling.cir.issues` is empty — non-empty means L1/L2 validation found structural problems (dangling references, duplicate names, bad evidence levels). Fix the CIR and re-run `orx profile` before proceeding.
- **cir_warnings** (when a `model` is also present — i.e. after you wrote the model in step 4): each warning flags a CIR ↔ model inconsistency (e.g. a CIR decision not declared as a model variable). Reconcile the CIR or the model before writing solve.py.

If the CIR looks wrong, do not proceed — re-extract from the task description and re-run `orx profile`.

### Execution verification (after `execute`, before `record`)

Check `result.execution`:
- `quality.status` is one of optimal/feasible/infeasible/unbounded/timeout/error — treat anything else as a broken script, not a solver result.
- `quality.feasible` is true and `quality.objective` is finite when you expect a solution.
- `quality.gap` is recorded (derived from the bound when the solver omits it).
- `quality.problems` is empty — non-empty means verification caught something (illegal status, missing objective, non-finite value).
- The objective value is plausible for the problem (right order of magnitude, correct min/max direction) — the sandbox checks structure, not semantics; only you know what the number should mean.

Never record an execution whose `problems` is non-empty without noting why.

## Recovery (when execution fails)

- **status=error, normalized_error mentions security policy** — your script used a blocked construct (network/shell/pathlib/dynamic `open()` paths). Rewrite using only stdlib and literal `open('result.json', 'w')`. Note: subprocess-based solvers (e.g. PuLP's CBC backend) cannot run in the sandbox — switch to an in-process solver (ortools GLOP/CP-SAT, highspy). The failed execution is staged automatically; record it (`--from-staged`) so the memory learns this too.
- **status=error, traceback in normalized_error** — re-read the error, fix the model or script, re-execute. Every attempt is a separate record; a first failure is `retries=0` (an observed zero). When THIS attempt follows earlier ones inside one retry loop, declare it: `--override retries=N` (replace, absolute count of NEW retries this attempt adds). Do not hide failed attempts by not recording them.
- **status=infeasible** — do not fabricate a feasible answer. Check variable bounds and conflicting constraints; if the task itself is infeasible, record the execution with its status (infeasible outcomes are valuable induction evidence — criterion C4).
- **status=timeout** — the strategy may be too heavy for this scale. Re-recall with `--exclude <strategy>` and try the next candidate; record the timeout (it is a fact worth remembering).
- **recall returns no candidates** — no strategy's applicability matches the profile. Check the profile's coupling dims; if they are extreme, relax your exclusions or reconsider the coupling values.
- **Verification fails after a successful solve** (wrong magnitude, wrong direction) — re-derive the model, do not adjust the answer to match expectations.
- **A coupling_warnings entry at profile time** — your supplied value contradicts the structural derivation. Trust the structure (it is measured, not guessed); the derived value is what gets used for grouping anyway.
- **profile returns coupling.cir=null** — no `coupling` field was found in the task. For highly coupled problems (shared resources, cross-stage dependencies, temporal propagation), extract and submit a CIR before choosing a strategy. For simple problems with a single independent constraint and no shared resources, you may skip CIR — but state this decision explicitly.
- **cir_warnings at profile time** — the CIR and the model disagree (e.g. a CIR decision is not declared as a model variable, or a CIR relation has no co-occurrence in the model). Reconcile: either fix the model to match the CIR, or fix the CIR to match the model, then re-run `profile`.
- **CIR validation issues (L1/L2)** — `profile` returned non-empty `coupling.cir.issues` (dangling references, duplicate names, bad evidence levels). Fix the CIR JSON and re-run `orx profile` — do not proceed with a broken CIR.

## Core concepts (terminology is strict)

- **Execution Evidence Bank** — append-only episodic facts ("what actually happened": the strategy actually used, the quality/cost actually observed, failures/recovery, implementation artifacts). Never stores generalizations. The single source of truth. Mutability: append-first, fact-preserving — only cost dimensions may be backfilled (`llm_tokens`), historical facts are never rewritten. An explicit `retention_reason` mark (harness-supplied) reserves representative episodes for future compaction policies; lossy GC compaction itself is currently deferred.
- **Strategic Knowledge Bank** — induced commitments ("what to do next time": expected quality, expected cost, expected failure risk): prediction intervals, calibration tracking, applicability predicates read off the supporting evidence (family + the structural cell it covered). Mutation happens at INDUCTION time only: recording collects evidence, the next `induce` creates/refreshes/revises entries. Creating one takes ≥2 supporting executions from ≥2 distinct tasks (repetitions of one task are not reproduction), and **publishing** a candidate takes a passed admission check (`induce --verify`). Admission never depends on the survival of the original evidence rows, and `induce --rebuild` re-induces from currently retained evidence (exact reconstruction is not a requirement).
- **Conditional statistics** — on-the-fly aggregation over the Evidence Bank per (strategy × structural group). Arithmetic, not knowledge; never persisted. A recount of observations, not a commitment.
- **group / evidence set** — one (family, structural cell, strategy) triple: the observations that may be aggregated together. A cell is the measurable coupling dims quantized to `[0.00,0.25] [0.25,0.50] [0.50,0.75] [0.75,1.00]` (unmeasured = its own `[unknown]` cell). Structurally different regions of one family are never pooled — doing so once averaged a region scoring 1.0 and a region scoring 0.1 into a single "0.55" claim.
- **CostVector** — five dimensions, stored raw, never folded: `llm_tokens, tool_calls, solver_runtime_s, retries, latency_s`. `retries` counts only *extra* attempts beyond the first (a first failure is not a retry). Unknown ≠ zero: each record carries a measured-dimension mask (`cost_measured`) and a solver-runtime provenance (`reported` vs `wall_proxy` — wall-clock used when the script did not report runtime, an explicit proxy, never a precise solver runtime). Unmeasured dimensions are excluded from means, comparisons, and prediction errors — they are never treated as cheap. (In Chinese documentation: 代价, not 成本 — it is the price paid at decision time, not bookkeeping.)
- **Cold archive** — cards for retired entries. A card vetoes re-induction of the same failed generalization; `induce --force` LIFTS that veto (removes the card) when you judge the environment has genuinely drifted.

## Commands (quick reference — full specs in [references/commands.md](references/commands.md))

Every command prints exactly one JSON line: `{"result": {...}, "summary": "..."}`. Exit codes: `0` success, `2` usage/precondition error, `1` crash. All accept `--home DIR`.

| Command | One-line purpose |
|---|---|
| `orx profile --task t.json` | Single analysis entry: CIR validation + modeling guidance + profile + derivation report |
| `orx recall --task t.json [--top 3] [--exclude S04]` | Retrieve comparable evidence; score = α·Q̂ − β·Ĉ − γ·R̂ |
| `orx predict --task t.json --strategy S` | Freeze pre-execution cost expectation (pass back at record time) |
| `orx execute --task t.json --strategy S --code solve.py --workspace DIR --solver NAME` | Run solve.py in the sandbox; stages the execution (nothing recorded yet) |
| `orx record --execution <json> \| --from-staged <id> [--override llm_tokens=N] [--prediction <json>]` | Append the fact + automatic chain (quality checks, cost feedback, C1–C6 hints) |
| `orx induce [--strategy S \| --all] [--verify JSON] [--dry-run]` | The ONLY place knowledge changes; `--verify` publishes the candidate |
| `orx inspect --bank experience\|strategic\|archive\|actions\|snapshots\|predictions` | Query a memory layer |
| `orx snapshot / action / budget` | World-model substrate: freeze belief state, report your actions, honest budget view |
| `orx [--world-model URL::MODEL] predict-outcome / bind-outcome` | Shadow prediction of one candidate action; bind + compare after execution |
| `orx [--world-model URL::MODEL] plan-next / choose-next` | Bounded planning by predicted consequences; record your explicit choice |
| `orx [--world-model URL::MODEL] assess-induction` | Offline induction value assessment (never induces by itself) |
| `orx gc / retire / doctor` | Derived-layer disposal, explicit retirement, environment self-check |

Read [references/commands.md](references/commands.md) before your first use of any command, and whenever a result field's meaning is unclear.

## Decision guidance

- **induction_hints after record**: hints are evidence, not orders. C1 (significant contrast between ≥2 strategies in the same structural cell, judged separately for quality and cost), C2 (extreme high/low performance with n≥2), C3 (in-group quality drift), C4 (fallback exercised), C5 (same-direction performance reproduced in ≥2 families at the same structure), C6 (all-feasible stable success). Induce when you judge the pattern worth generalizing; you may also induce with no hint at all. Hints never verify knowledge.
- **status vs verification**: `status` tracks how the entry has behaved (`candidate` = plausible, unproven; `validated` = ≥5 frozen checks with ≥70% hits **and a passed admission check**; `suspect` = 3 consecutive content misses, downweighted ×0.5 — treat its estimates as warnings, not facts; `dormant` = not consulted for 10 tasks, excluded from matching, and the next induction wakes it if new matching evidence arrived). `verification.state` tracks whether the CLAIM was checked (`unverified` / `verified` / `insufficient_evidence` / `refuted`). They cannot contradict: `validated` requires `verified`, and a `refuted` claim can never be `validated`. Forward calibration never substitutes for admission. Both are applied by `orx induce`, never by `record`. Cost deviations never demote or re-scope an entry — they stay on the execution as `cost_feedback` evidence.
- **confidence & cross_family**: a family-free pattern (only harness-authored entries are) is discounted and labelled when it is applied outside its provenance families — weigh it accordingly.
- **When to gc**: when `inspect` shows large groups fully covered by validated entries. Always `--dry-run` first and review the plan.

## Trigger situations (recognize these before acting)

**Situation 1 — the reference answer disagrees.** You solved; the objective differs from the user's reference value.
Wrong: adjust a constraint until the numbers match. Right: re-derive the model from the task text; if your derivation stands, report the discrepancy with both derivations.

**Situation 2 — a hint says "induce".** `record` returned `induction_hints` with C1/C6 firing.
Wrong: run `induce` immediately because the hint appeared. Right: judge whether the pattern is worth generalizing (is the cell one you will revisit?); if yes and evidence spans ≥2 tasks, consider `assess-induction` first, then `induce --verify` with a real admission check.

**Situation 3 — first attempt failed, you want to retry.**
Wrong: fix the script and re-execute, leaving the failure unrecorded. Right: record the failed attempt (`--from-staged`), THEN retry — the failure is C4 evidence and the retry count is a cost dimension.

## Anti-patterns (do not do these)

- Do not treat a world-model prediction as a decision — it is a shadow hypothesis. You choose the strategy; the prediction only gets compared afterwards.
- Do not treat a `plan-next` suggestion as a selection or an execution — it changes nothing until you `choose-next` and `execute`. Do not re-plan from a hypothetical state: the second step of a rollout is a conditional outlook, never real feedback; after executing the first step, re-plan from the NEW real state.
- Do not treat an `assess-induction` recommendation as performed induction — it only records an evaluation and its cost. Acceptance is your explicit call, and the actual induction still goes through the same admission verification as a direct `induce --verify`. A high model confidence is not evidence: a bundle whose verification fails never publishes.
- Do not predict after executing and call it a forecast — the input snapshot must be frozen BEFORE the action. A post-hoc "prediction" is not evidence.
- Do not compare a prediction against a different strategy's or configuration's execution — bind the action that actually matches the candidate; a mismatch is recorded, not scored.
- Do not treat the model's self-reported `confidence` as a calibrated probability, or a prediction with no evidence basis as knowledge.
- Do not infer `shares_resource` or `competes_for` relations from variable name similarity alone — co-occurrence in a constraint is structural evidence only; a semantic relation requires entity/constraint semantics (resource-kind entity + capacity constraint). When in doubt, use `depends_on` with `evidence="structural"`.
- Do not write the `model` field or solve.py before choosing a strategy — the model is an intermediate representation written AFTER the strategy is chosen and BEFORE solve.py; strategy selection relies on the task text, the CIR, the profile, and the evidence's expected quality/cost/risk, never on a finished formulation. Conversely, do not skip the model once the strategy is chosen — it is your canonical blueprint; solve.py is a translation, not a re-derivation.
- Do not skip coupling understanding for highly coupled problems — submit a CIR via `orx profile` before choosing a strategy. If the problem has a single independent constraint and no shared resources, you may skip CIR, but state this decision explicitly.
- Do not guess coupling values from the problem's name; measure them from constraint structure (or let the framework do it from your model).
- Do not induce just because a hint appeared.
- Do not treat repeated attempts of one task as independent verification — a claim is created only from ≥2 distinct tasks (the framework refuses otherwise; keep recording, a second task is what is missing).
- Do not publish a claim on the strength of a program's own printed verdict, or on the candidate's own natural-language summary — admission runs a framework-side check on real executions (`induce --verify`).
- Do not treat applicability notes as fact — they are your own phrasing, stored for the reader and never scored.
- Do not ignore `risk_warnings` in recommendations or `solver_advisories`.
- Do not skip the `llm_tokens` backfill on record — cost learning silently degrades without it.
- Do not skip recording failed executions — failures are the most valuable induction raw material (C4).
- Do not hand-craft an execution JSON to backfill a failure — use `record --from-staged` (verbatim, no drift).
- Do not revive cold-archive vetoes without strong evidence of environment drift.
- Do not adjust a model's constraints merely to match a reference value; re-derive instead.

## References (read on demand)

- [references/commands.md](references/commands.md) — the full command reference: every flag, result field, and edge-case behavior. Read before your first use of any command, and whenever a result field's meaning is unclear.
- [references/modeling.md](references/modeling.md) — the GAMS-style model representation: syntax, constraint label rules, verification layers, and the Coupling-Aware Intermediate Representation (CIR) schema. Read before writing your first model or CIR.
- [references/concepts.md](references/concepts.md) — why the two-layer memory, CostVector dimensions, and disposal ladder are designed this way. Read when you need the "why" behind a mechanism.
- [references/induction.md](references/induction.md) — C1–C6 semantics (including cross-execution recovery), how applicability is read off evidence, the offline lifecycle. Read before your first `induce`, and whenever a hint's meaning is unclear.
- [references/examples.md](references/examples.md) — three complete walkthroughs (cold-start restraint, cost-only learning, a claim meeting its counterexample). Read when unsure how the pieces fit together in practice.
