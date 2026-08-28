---
name: or-experience-bank
description: Solve, verify, and debug operations research optimization problems (LP, MILP, scheduling, transportation, network flow, resource allocation, inventory) by driving the OR Experience Bank CLI (`orx`) — verified-upfront modeling, multi-solver cross-validation, an append-only experience bank, and offline structural induction. Use when the user asks to formulate, solve, validate, or debug an optimization model, or to query/manage accumulated OR experience. Do not use for generic mathematical proofs, pure data analysis, or non-optimization tasks.
---

# OR Experience Bank (orx)

## Purpose

Solve OR problems through a verified pipeline (model → validate → multi-solver → cross-validate → gold gate), accumulate the lessons into an append-only experience bank, and — offline — induce transferable optimization principles from accumulated experience. You orchestrate; the `orx` CLI validates, executes, and stores.

## When to Use

Use this skill when:
- The user asks to formulate, solve, verify, or debug an optimization model (LP / MILP / scheduling / routing / network flow / resource allocation / inventory).
- The user asks what the experience bank knows about an OR topic, or wants to manage accumulated experiences.
- The harness signals that offline induction should run (accumulated cross-family realizations) — though in practice the `induction_check` field in every `orx episode` response is the authoritative cue.

Do not use this skill when:
- The task is a generic mathematical proof or symbolic manipulation with no optimization model.
- The task is pure data analysis (statistics, plotting) with no decision variables or objective.
- The user only wants conceptual OR theory explained with no problem to solve.

## How This Works

You are the orchestrator. The framework is a set of **stateless CLI commands** (`orx ...`); you call one per step, read its JSON output, think, and decide the next step. All state lives in **files in your working directory** (the run directory) — every command is an independent process, so you can stop, inspect, retry, or resume at any step.

**The chain is enforced by stamps, not trust.** Each gate command (`validate`, `signature`, ...) stamps the artifact it approved with a content hash. The next command refuses to run if the predecessor stamp is missing or the file changed after stamping. You cannot skip a step or silently edit a validated artifact — but you CAN freely retry any step.

**First action in a fresh environment:** run `orx doctor`. If `orx` is not on PATH, the framework is not installed — use `python3 <repo>/scripts/orx.py` as a fallback and tell the user to `pip install -e .` (see docs/deployment.md).

## Run Directory Layout

```
problem.txt                     the problem (written by recall)
priors.json                     recalled experiences + [En] citation labels
model.txt                       YOU write: [THINK]...[/THINK][MODEL]...[/MODEL]
signature.json                  YOU write: structural signature JSON
stamps/model.json               L1+L2 verdict + hash of model.txt
stamps/signature.json           vocabulary verdict + hash of signature.json
branches/<solver>/hints.json    bank hints pulled BEFORE codegen
branches/<solver>/solve.py      YOU write: complete solver script
branches/<solver>/result.json   execution outcome + hints (written by solve)
cross_validation.json           cross-solver comparison verdict
gold.json                       gold verdict (user-provided or consistency-only)
experiences.json                appended experience ids
episode.json                    terminal record + utility credit
rounds/<n>/                     archived artifacts of reflection round n
journal.jsonl                   audit log of every orx command
```

`orx status` at any time tells you where you are and what's next.

## model.txt Format (read before writing model.txt)

model.txt contains EXACTLY two sibling marker blocks, in this order — THINK first, then MODEL, at the top level (NOT nested):

```
[THINK]
Your analysis: objective, decisions, constraints, structural insights.
Cite applied priors here: [uses E1]
[/THINK]
[MODEL]
SETS:
  ...
PARAMETERS:
  ...
VARIABLES:
  ...
OBJECTIVE:
  ...
CONSTRAINTS:
  C1: ...
[/MODEL]
```

Critical rules:
- **Use the square-bracket markers `[THINK]...[/THINK]` and `[MODEL]...[/MODEL]`** (uppercase or lowercase). Angle-bracket tags (`<think>...</think>`) are also accepted by the parser, but many harnesses reserve `<think>` as their own reasoning-channel marker and strip or transform it before it reaches the file — square brackets always survive the trip.
- `[MODEL]` comes AFTER `[/THINK]` (siblings), never inside the THINK block.
- Both closing markers are required; an unclosed `[THINK]` fails L1.
- The five blocks inside `[MODEL]` (SETS/PARAMETERS/VARIABLES/OBJECTIVE/CONSTRAINTS) must all be present and non-empty.

| ❌ Wrong | Why it fails |
|---|---|
| `[THINK]...[MODEL]...[/MODEL][/THINK]` | Nesting — MODEL must be a sibling after `[/THINK]` |
| `[THINK]...` with no `[/THINK]` | L1 failure: the parser needs both markers |
| Inventing other markers (`[REASONING]`, `[RESPONSE]`) | Only THINK and MODEL exist |
| Descriptive constraint labels (`manure:`) | L2 only recognizes `C1:`, `C2:`, ... |

## Core Workflow — Online Solve

```
NL Problem
  │
  ▼
orx recall --problem-file problem.txt     → priors.json (read it: cat priors.json)
  │
  ▼
YOU write model.txt  (cite applied priors as [uses E1] inside [THINK])
  │
orx validate                             → issues? fix model.txt, retry freely
  ▼                                        passed? stamps/model.json
YOU write signature.json
  │
orx signature                            → vocab errors? fix, retry (model stamp intact)
  ▼
orx hints --solver <solver>              → read hints BEFORE writing code
YOU write branches/<solver>/solve.py     (one per solver)
  │
orx solve --solver a,b,c                 → branches run CONCURRENTLY (parallel
  ▼                                        exploration); failed? read that branch's
(repeat for ≥3 different solvers)          result.json hints, fix code, re-run
orx solve --solver <failed>              → single-branch retry (repair is serial)
  │
orx cross-validate                       → consistent? proceed
  │                                        inconsistent? add a third branch, re-run
  ▼
═══ GOLD GATE: gold comes ONLY from the user/problem. NEVER self-derive. ═══
  │  If the user hasn't provided gold: STOP and ask.
  │
orx gold --answer <value>                → matched? proceed to append
  │                                        mismatched? DO NOT append; reflect;
  │                                        orx new-round; re-model (≤3 rounds)
  ▼
YOU write experience files (one per lesson, all layers that had events)
orx append --file exp_<layer>.json       → repeat per lesson
  │
orx episode                              → terminal: episode.json + utility credit
  │                                        + induction_check (should_induce?)
  ▼
induction_check.should_induce == true? ── yes ──► run the Offline Induction
  │                                              Workflow before the next solve
  no
  ▼
done (report to user)
```

### Step-by-step thinking guide

| After you observe... | Think... | Then do |
|---|---|---|
| `priors.json` returned | Which priors apply to this problem's structure? | Compose model.txt citing `[uses En]` for the ones you actually apply |
| `validate` issues (L1) | Markers/blocks malformed | Re-read the model.txt Format section; fix the `[THINK]`/`[MODEL]` markers and five blocks |
| `validate` issues (L2) | Which symbol is undeclared? | Declare it in SETS/PARAMETERS/VARIABLES, or fix the reference |
| `signature` vocab errors | Which core-dim value is out of vocabulary? | Fix only that value in signature.json |
| `hints` output | Which API gotchas apply to this solver? | Write solve.py applying the hints |
| `solve` (parallel) returned | Which branches failed, which agreed? | Fix ONLY the failing branches' solve.py, re-run `orx solve --solver <failed>` |
| `solve` (single) failed | What does normalized_error + repair_hints say? | Fix ONLY branches/<solver>/solve.py, re-run solve |
| `cross-validate` inconsistent | Which branch is the outlier? | Add a third solver branch to triangulate |
| gold matched | What did I learn across ALL layers? | Write one experience file per lesson, append each |
| gold mismatched | Why was the modeling DIRECTION wrong (not the code)? | `orx new-round`, re-model from scratch |
| `episode` returned `induction_check.should_induce: true` | Accumulation crossed the watermark | Run the Offline Induction Workflow before the next solve |

## Solver Selection Strategy

Which solvers to branch on is YOUR decision each run. Do not default to the same pair every time — the value of cross-validation comes from **heterogeneity**, and the bank only grows API knowledge for solvers you actually use.

How to choose:

1. **Check availability first**: `orx doctor` lists importable solvers. Only branch on available ones.
2. **Prefer heterogeneous families**: pairing a `milp` solver with `cp_sat` (OR-Tools) validates the model across different solving paradigms — stronger evidence than two milp solvers agreeing. Pairing a direct-API solver (highs/scip/gurobi/copt) with a modeling-framework branch (pulp/pyomo) validates at the API level.
3. **Rotate across runs**: vary your solver pair from run to run (e.g. highs+ortools, then pulp+scip, then gurobi+pyomo). Rotation (a) spreads API knowledge into the Implementation/Repair banks so future runs benefit, (b) avoids over-fitting your code generation to one API's habits, (c) keeps the bank's solver coverage balanced.
4. **Match the problem**: CP-SAT requires integer coefficients (scale deliberately); commercial solvers (gurobi/copt) need licenses; pyomo needs a backend solver installed.
5. **When the bank has solver-specific hints**: `orx hints --solver <s>` returns accumulated API knowledge for THAT solver — a solver with rich hints is cheaper to write correct code for, but do not let this collapse into always picking the same two.

Minimum requirement stays: ≥3 valid branches from **different** solvers (configurable via `min_cross_validation_branches`, default 3).

## Experience Synthesis — What to Write to Each Bank

After gold match, write one JSON file per lesson and `orx append --file` each:

| If this happened during the solve... | `layer` | Experience file fields |
|---|---|---|
| Structural modeling insight | `modeling` | title, retrieval_text, modeling_aspect (constraint/objective/variable/classification/structure), action, rationale |
| Solver API gotcha | `implementation` | title, retrieval_text, diagnosis, action, rationale, solver |
| Error → fix | `repair` | title, retrieval_text, diagnosis, action, rationale, solver |
| Performance tuning | `solving` | title, retrieval_text, diagnosis, action, rationale, solver |

Checklist before `orx episode`: modeling insight? API gotcha? error→fix? performance tuning? Write only layers that had events — never fabricate.

## Core Workflow — Offline Induction

**You do NOT wait for an external signal.** Every `orx episode` response carries an `induction_check` field (the 3-gate trigger decision evaluated right after your realizations were appended). When it says `should_induce: true`, run the induction chain BEFORE starting the next solve:

```
orx episode -> ... "induction_check": {"should_induce": true, ...}   ← your cue
  │
orx clusters                             → candidate clusters (cross-family isomorphic)
  │
  ▼ (per cluster, under <bank>/induction/<cluster_id>/)
orx align --cluster <id>                 → writes alignment.json template; YOU fill it
orx align --cluster <id>                 → stamps the filled alignment
  │
orx induce --cluster <id>                → writes hypotheses.json template; YOU fill it
orx induce --cluster <id>                → stamps hypotheses (1-3, grounded in roles)
  │
orx refute --cluster <id>                → writes refutations.json template; YOU fill it
orx refute --cluster <id>                → EXECUTES your programs; verdicts decided by execution
  │
orx validate-pattern --cluster <id>      → writes validation.json template; YOU fill it
orx validate-pattern --cluster <id>      → stamps transfer evidence
  │
orx append-pattern --cluster <id>        → scores + appends validated patterns (terminal)
```

Roles come from the canonical set: resource_pool, capacity_limit, competing_decisions, objective_contribution, demand_requirement, coupling_constraint, time_period, flow_balance.

## Tool Usage

| Command | Use it when |
|---|---|
| `orx doctor` | Fresh environment, or `command not found` / weird failures — verify python/bank/solvers/indexes |
| `orx status` | You forgot where you are; after resuming an interrupted run |
| `orx recall --problem-file <f>` | Starting a run (ALWAYS first — priors must influence the model) |
| `orx validate` | model.txt written or edited |
| `orx signature` | signature.json written or edited |
| `orx hints --solver <s>` | BEFORE writing solve.py for solver s (first time AND after a failure) |
| `orx solve --solver <s>` | solve.py written or fixed (single branch / repair retry) |
| `orx solve --solver a,b,c` | All branch codes written — run them CONCURRENTLY (parallel exploration) |
| `orx cross-validate` | ≥3 valid branches exist (configurable: `min_cross_validation_branches`) |
| `orx gold --answer <v>` | User provided gold (or explicitly confirmed none) |
| `orx append --file <f>` | Gold matched, one lesson per file |
| `orx episode` | All layers covered — terminal |
| `orx new-round` | Gold mismatched, archiving the failed round |
| `orx trigger` | Manually checking the induction gates (optional — `orx episode` already carries `induction_check`) |
| `orx query / show / deprecate / stats` | Bank inspection and management, anytime |

Do not reimplement validation logic manually (symbol cross-checks, vocab checks, objective comparison) — the commands already do it deterministically.

## Verification

| Step | Success criterion |
|---|---|
| `recall` | `priors_count >= 0` (empty is fine — first solve) |
| `validate` | `passed: true` |
| `signature` | `passed: true` |
| `solve` | `status` in {optimal, feasible} |
| `cross-validate` | `consistent: true` |
| gold gate | Gold from user/problem ONLY; compare `best_objective` |
| `append` | `status: "appended"` (not duplicate/rejected) |
| `episode` | `recorded: true` AND read `induction_check.should_induce` — if true, induction is due NOW |
| induction chain | each step stamps; final `appended` non-empty |

**Cross-solver consistency does NOT prove correctness** — two solvers can agree on the same wrong relaxation. Gold mismatch + consistent solvers almost always means the MODEL is wrong.

## Failure Recovery

| Failure | Recovery |
|---|---|
| `validate` issues | Fix model.txt, re-run validate (free retry, no penalty) |
| `solve` branch failed | Read `normalized_error` + `repair_hints` in branches/<solver>/result.json; fix solve.py; re-run `orx solve --solver <solver>` (repair within a branch is serial by design) |
| All branches fail | Solver not installed → tell the user. Modeling issue → revise model.txt, re-validate |
| `cross-validate` inconsistent | Add a third branch with a different solver, re-run cross-validate |
| Gold mismatch | DO NOT append. Reflect on the modeling direction; `orx new-round`; re-model (≤3 rounds) |
| `append` says duplicate | Rephrase with new insight or skip |
| Stamp "stale" error | You edited a stamped artifact; re-run the gate command for the new content |
| Gold recorded incorrectly, run already finished | Episodes are append-only facts — start a FRESH run (`orx recall` in a new directory) and re-solve with the correct gold; never delete episode.json by hand |
| Hypothesis refuted | Archive the lesson; do not resubmit without new evidence |

## Output Requirements

When reporting to the user, include:
- The selected objective value and which solvers agreed (cross-validation status)
- The verified model (or its key structure) and any assumptions you stated
- Gold verdict (matched / mismatched / not provided)
- Appended experience ids and what each lesson says
- Cited priors and whether they were credited (`utility_credited`)

On failure: what was tried, the concrete failure reason, and the next step — never a bare "it failed".

When reporting **induction results**: clusters processed, patterns validated/refuted, the scoring breakdown (C/T/V/N/K/X → total) per hypothesis, and which hypotheses were refuted and why.

Do not:
- Present a solver's self-reported status without the result.json evidence
- Hide assumptions you made when the input was ambiguous

## Common Pitfalls

| ❌ Pitfall | ✅ Correct |
|---|---|
| Pre-composing model.txt before `recall` | Run recall FIRST, read priors.json, THEN compose the model |
| Missing `[/THINK]` closing marker | Always close both blocks; the parser needs them literally |
| Writing `<think>` tags in model.txt | Many harnesses strip `<think>` before it reaches the file — use `[THINK]...[/THINK]` square-bracket markers |
| Nesting `[MODEL]` inside `[THINK]` | They are siblings: `[/THINK]` closes first, then `[MODEL]` starts |
| Descriptive constraint labels (`manure:`) | Use `C1:`, `C2:`, ... — L2 only recognizes `C\d+` |
| Undeclared symbols in OBJECTIVE/CONSTRAINTS | Declare every symbol in SETS/PARAMETERS/VARIABLES first |
| `prod`/`exp`/`log` in OBJECTIVE | Use an AUXILIARY block for nonlinear relationships |
| result.json field `"objective"` | Field is `objective_value`; `status` is lowercase |
| Code patch instead of complete script | solve.py must be the COMPLETE script every time |
| Citing `[uses E7]` when only E1-E3 exist | Only cite tags present in priors.json labels |
| Citing a prior you didn't apply | Citation = utility credit; false credits corrupt ranking |
| One `solve` then `cross-validate` | Need ≥3 valid branches from different solvers (default; configurable) |
| Always branching on the same solver pair | Rotate across runs (see Solver Selection Strategy) — the bank only learns APIs you actually use, and cross-family agreement is stronger evidence |
| Running branches one-by-one when you could batch | Write all branch codes first, then `orx solve --solver a,b,c` — branches explore in PARALLEL |
| Ignoring `repair_hints` on failure | Read them before switching solvers — they may contain the exact fix |
| Only writing to the Modeling Bank | Check ALL four layers before `orx episode` |
| Self-deriving gold from solver output | Gold comes ONLY from the user/problem statement |
| Appending after gold mismatch | Never append wrong-model lessons; reflect and re-model |
| Made-up `unseen_tasks` in induction | Use REAL problems from past episodes (`orx query` to find them) |
| Fabricating transfer numbers | You MUST have actually solved with/without the principle |
| No-op refutation code | The program must print `{"principle_failed": true|false, ...}` as its LAST stdout line |
| Editing model.txt after validate | The stamp goes stale; re-run `orx validate` after any edit |
| Arguing with a rejected command | Read the error JSON, fix the artifact, re-run |
| Printing the result instead of writing result.json | Your solve.py must WRITE result.json in its cwd (`open('result.json', 'w')`) — stdout is not parsed for results |
| Hand-writing branches/<s>/result.json yourself | result.json is written by `orx solve` (it validates + enriches your solver's output); hand-written files lack the `valid` field and will not count in cross-validate |
| Constructing the result.json path dynamically (`os.path.dirname(__file__)` + ...) | The sandbox requires a LITERAL path: `open('result.json', 'w')` — the branch cwd is already correct |
| Re-recording gold on a completed run | Episodes are append-only; if gold was recorded wrong, start a FRESH run (`orx recall` in a new directory) |
| Ignoring `induction_check` in the episode response | `should_induce: true` means induction is due NOW — process clusters before the next solve; skipping it starves the reflow loop |

## Examples

Three worked examples with exact command sequences and reasoning live in [references/examples.md](references/examples.md). Read it when handling:
- **A normal solve with gold match** (Example 1) — the happy path, including a mid-chain repair
- **Ambiguous input / missing information** (Example 2) — what to assume, what to ask
- **Gold mismatch caused by a wrong modeling direction** (Example 3) — the reflection loop, and why cross-solver consistency did not catch it

## References

Read these only when needed (progressive disclosure):

| When you need | Read |
|---|---|
| GAMS-style DSL syntax, L1/L2/L3 verification | [references/modeling-contract.md](references/modeling-contract.md) |
| Signature vocabularies and alignment rules | [references/structural-signature.md](references/structural-signature.md) |
| Record schemas (all banks + Episode) | [references/experience-schema.md](references/experience-schema.md) |
| result.json contract, sandbox rules, solver API notes | [references/solver-adapters.md](references/solver-adapters.md) |
| Utility attribution, soft delete, cold archive | [references/bank-lifecycle.md](references/bank-lifecycle.md) |
| Induction pipeline internals | [references/induction-pipeline.md](references/induction-pipeline.md) |
| Worked examples (positive / ambiguous / negative) | [references/examples.md](references/examples.md) |

Implementation: [src/or_experience_bank/cli/](src/or_experience_bank/cli/) · Entry: `python3 scripts/orx.py` (or `orx` when installed)
