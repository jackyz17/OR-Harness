# OR Harness

A Python framework for LLM agents to **solve OR problems with parallel solver exploration**, **accumulate verified solving experience**, and — offline — **induce new optimization principles from past experience**. Pure standard library, zero dependencies.

[中文版](README_zh.md)

---

## Quick start

```bash
# Solve an OR problem (mock demo, no solver/LLM needed)
python3 scripts/or_experience_cli.py solve --mock-demo --problem "Assign tasks to machines to minimize cost" --json

# Full solve walkthrough with a real LLM (you ARE the LLM):
#   the framework sends prompts, you answer them
PYTHONPATH=src python3 scripts/demo_farmer_walkthrough.py

# Offline induction over accumulated experience (mock demo)
python3 scripts/or_experience_cli.py induce --mock-demo --json

# Full induction walkthrough: 3 heterogeneous problems -> 1 validated pattern
PYTHONPATH=src python3 scripts/demo_induction_walkthrough.py

# Run all tests
cd tests && PYTHONPATH=../src python3 -m unittest discover -p "test_*.py"
```

---

## What it does

### 1. Solve OR problems — parallel multi-solver exploration

- **Structured modeling first**: the agent formalizes the problem as `<think>` + `<model>` (GAMS-style DSL); a `ModelingGate` verifies it (format / structure / semantics, ≤3 fix rounds) **before any code runs**.
- **Heterogeneous parallel exploration**: the verified model fans out to up to **7 solver branches** — Gurobi, SCIP, HiGHS, COPT, OR-Tools (CP-SAT), PuLP, Pyomo — each in an isolated sandbox with intra-branch sequential repair.
- **Gold-gated loop**: after solving, the result is compared to a gold answer (supplied by the caller). A mismatch triggers **reflective re-modeling** (≤3 outer rounds) instead of blind retries.

### 2. Accumulate experience — a living experience bank

- Successful solves are contrasted with buffered failures (comparative synthesis) and pass an **admission judge** before entering the bank.
- The **Modeling Bank** stores modeling methods as peers — directly-solved (`status=null`) and induced (`status=validated`) records coexist. Each record carries a `modeling_aspect` (constraint/objective/variable/classification/structure) and a structural signature. Episodes record problem-level snapshots; a derived repair graph powers error→fix guidance.
- The bank **evolves without losing its history**: records are append-only, but a lifecycle state (`active → deprecated`) + utility-tracked soft delete retire bad experiences into a compressed cold archive, with anti-resurrection dedup.

### 3. Induce general principles offline

Periodically (manual or gated by a trigger policy), the framework mines accumulated modeling records for **heterogeneous but structurally-isomorphic clusters** (e.g. inventory / production / workforce scheduling all sharing a *scarce-resource allocation* skeleton), aligns their structural roles, and induces a candidate principle — **which only becomes a `validated` record if it survives solver-backed counterexample search AND improves an unseen task**. Validated records then re-enter the online loop as planning priors.

See [references/induction-pipeline.md](references/induction-pipeline.md) for the full induction loop.

### 4. Pattern reflow into online solving (Phase 4.1)

Validated records feed back into the online loop as **planning priors**. The framework recalls relevant records before modeling, injects them into the prompt (`[E1]...`, `[E2]...`), and the agent cites applied experiences with `[uses En]`. On a gold match, the framework credits utility for cited experiences — closing the loop with soft-delete scoring and induction triggers. This is the final segment: **solve → experience → induction → reflow → better solve**.

---

## How it fits together

```
Natural-language OR problem
      │
      ▼
Structured Modeling (think→model→verify)  ◀─── planning priors ([uses En])
      │
      ▼
7 parallel solver branches ──▶ Cross-solver validation ──▶ matches gold?
      │                                                      ├─ no  → reflective re-modeling ↺
      │                                                      └─ yes → comparative synthesis
      ▼                                                                │ (admission judge)
Experience Bank                                                       ▼
  Modeling Bank: all records are peers (status=null | validated)
  Episodes · Implementation · Repair · Solving
  Lifecycle: active → deprecated → cold archive
      │
      ▼ (triggered offline)
Structural Induction: candidates → encoding → alignment → inducer (hypothesis)
                    → counterexample (solver-refuted) → validation (unseen transfer)
                    → validated → pattern (append-only)
```

---

## Harness integration

This framework is a **tool under a harness agent**. The split of responsibilities:

- **Framework owns rules**: schemas, controlled vocabularies, the `ModelingGate` verifiers, parsing, dedup, append-only stores, retrieval (query construction, embedding search, ranking, filtering), and all prompt templates.
- **Agent owns the LLM**: the agent generates the `<think>`/`<model>`, structural signature, solver code, comparative synthesis, judge verdicts, and (offline) alignment/hypothesis/counterexample outputs. The framework never calls an LLM on its own; every LLM touchpoint takes an injected `llm_client`.

In harness mode, **the agent IS the LLM** — there is no external API, no wrapper script. The framework sends prompts; the agent answers them. See [SKILL.md](SKILL.md) for the full operational contract.

## CLI reference (standalone/demo only)

The CLI is for standalone mode (no harness agent: cron, batch) and demos/tests. In harness mode, use the Python API directly.

```bash
python3 scripts/or_experience_cli.py solve       --problem "..." [--solvers a,b,c] [--mock-demo] [--json]
python3 scripts/or_experience_cli.py solve       --interactive-llm --problem-file problem.txt   # harness: you answer prompts on stdin
python3 scripts/or_experience_cli.py retrieve    --layer modeling --query "..." [--json]
python3 scripts/or_experience_cli.py append      --input experience.json [--json]
python3 scripts/or_experience_cli.py induce      [--auto] [--min-new-realizations 3] [--mock-demo] [--json]
python3 scripts/or_experience_cli.py stats|rebuild-index|validate-bank [--json]
```

- CLI `solve` runs the single-shot flow (solve + auto-extract). The **two-step harness flow** (`solve(defer_extraction=True)` + `evaluate_with_gold(gold)`) is only available via the Python API.
- CLI `induce --interactive-llm` cannot complete transfer validation (raises `RuntimeError` for the transfer solver). Use the Python API to inject a real transfer solver.
- Standalone mode needs an LLM wrapper command (`--llm-command`).
- Runtime data defaults to `~/.hermes/or-experience-bank`; override with `OR_EXPERIENCE_BANK_HOME`.
- `--mock-demo` runs fully offline with fake LLM + mock solvers — for demos/tests only.

## Project layout

```
src/or_experience_bank/
├── core/          # schemas, append-only stores, lifecycle, utility tracker
├── modeling/      # modeling gate (GAMS-style DSL), signature extraction
├── experience/    # comparative extraction, admission judge, failure buffer
├── retrieval/     # embedding index, retriever, repair graph, modeling retriever
├── solving/       # orchestrator, execution sandbox, reflection
├── solvers/       # 7 solver adapters + registry
└── induction/     # candidates, encoding, alignment, inducer, counterexample,
                   # validation, trigger, pipeline
scripts/           # CLI + demos (farmer walkthrough, induction walkthrough)
references/        # DSL, signature, schemas, workflow, prompts (deployed with the harness)
tests/             # 237 unittest cases
```

## Documentation

| Doc | What it is |
|---|---|
| [SKILL.md](SKILL.md) | operational contract for the harness agent (you ARE the LLM) |
| [references/modeling-contract.md](references/modeling-contract.md) | GAMS-style DSL syntax, constraint label rules, three-layer verification |
| [references/structural-signature.md](references/structural-signature.md) | signature schema, controlled vocabularies, alignment rules, examples |
| [references/induction-pipeline.md](references/induction-pipeline.md) | the offline induction loop in detail |
| [references/experience-schema.md](references/experience-schema.md) | record schemas: signature, modeling experience, episode, lifecycle |
| [references/workflow.md](references/workflow.md) | the online solving workflow |
| [references/prompts.md](references/prompts.md) | prompt templates with concrete input→output examples for all 12 agent outputs |
| [references/trajectory-schema.md](references/trajectory-schema.md) | AttemptRecord, BranchResult, SolveResult, termination values |
| [references/solver-adapters.md](references/solver-adapters.md) | 7 solver adapters, result contract, execution controls |
| [docs/project-overview.md](docs/project-overview.md) | full project introduction for group meetings |
