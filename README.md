# OR Experience Bank

A Python framework for LLM agents to **solve OR problems with parallel solver exploration**, **accumulate verified solving experience**, and — offline — **induce new optimization principles from past experience**. Pure standard library, zero dependencies.

[中文版](README_zh.md)

---

## Quick start

```bash
# Environment self-check (first action in a fresh environment)
python3 scripts/orx.py doctor

# Initialize the bank
python3 scripts/orx.py init

# Cross-process resumability demo (two sessions, one run — no server, no state in memory)
python3 scripts/demo_orx_resumability.py

# Run all tests
cd tests && PYTHONPATH=../src python3 -m unittest discover -p "test_*.py"
```

---

## Architecture: LLM-as-orchestrator + framework-as-tools

The harness agent (Claude Code / openclaw / Hermes class) is the orchestrator: it reads [SKILL.md](SKILL.md), thinks, and calls one **stateless `orx` command** per step. All cross-call state lives in **files** (the run directory), never in a server process — every command is an independent process, so runs survive connection drops, session changes, and process restarts.

**The chain is enforced by stamps, not tokens.** Each gate command stamps the artifact it approved with a content hash; the next command refuses to run if the predecessor stamp is missing or the file changed after stamping. Steps are unskippable, yet every step is freely retryable.

```
NL Problem
  │
  ▼
orx recall ──► priors.json ([E1],[E2],... reflow)
  │
  ▼
agent writes model.txt ──► orx validate (L1+L2 gate) ──► stamp
agent writes signature.json ──► orx signature (vocab gate) ──► stamp
  │
orx hints ──► bank hints BEFORE codegen
agent writes branches/<solver>/solve.py ──► orx solve ×≥2 (sandbox exec)
  │
orx cross-validate ──► ≥2 branches agree?
  │
orx gold ──► matches user-provided gold?
  ├── yes ──► orx append ×N (one lesson per bank layer) ──► orx episode (terminal)
  └── no  ──► reflect ──► orx new-round ──► re-model (≤3 rounds)
```

## What it does

### 1. Solve OR problems — parallel multi-solver exploration

- **Structured modeling first**: the agent formalizes the problem as `<think>` + `<model>` (GAMS-style DSL); `orx validate` verifies it (format / structure) **before any code runs**.
- **Heterogeneous parallel exploration**: the verified model fans out to up to **7 solver branches** — Gurobi, SCIP, HiGHS, COPT, OR-Tools (CP-SAT), PuLP, Pyomo — each in an isolated sandbox. A failed branch is fixed by editing only that branch's code and re-running `orx solve` — the chain is never restarted.
- **Gold-gated loop**: after cross-validation, the result is compared to a gold answer (supplied ONLY by the user/problem). A mismatch triggers **reflective re-modeling** (`orx new-round`, ≤3 outer rounds) instead of blind retries.

### 2. Accumulate experience — a living experience bank

- Lessons are synthesized per bank layer (modeling / implementation / repair / solving) and admitted via `orx append` with content-hash dedup and anti-resurrection.
- The **Modeling Bank** stores modeling methods as peers — directly-solved (`status=null`) and induced (`status=validated`) records coexist. Episodes record problem-level snapshots; a derived repair graph powers error→fix guidance.
- The bank **evolves without losing its history**: records are append-only, but a lifecycle state (`active → deprecated`) + utility-tracked soft delete retire bad experiences into a compressed cold archive.

### 3. Induce general principles offline

Periodically (gated by a trigger policy), the framework mines accumulated modeling records for **heterogeneous but structurally-isomorphic clusters** (e.g. inventory / production / workforce scheduling all sharing a *scarce-resource allocation* skeleton), aligns their structural roles, and induces a candidate principle — **which only becomes a `validated` record if it survives solver-backed counterexample search AND improves an unseen task**. Validated records then re-enter the online loop as planning priors.

### 4. Pattern reflow into online solving

Validated records feed back into the online loop as **planning priors**. `orx recall` surfaces relevant records before modeling; the agent cites applied experiences with `[uses En]`. On a gold match, `orx episode` credits utility for cited experiences — closing the loop: **solve → experience → induction → reflow → better solve**.

---

## Harness integration

Deploy as a **skill** in three layers (see [docs/deployment.md](docs/deployment.md) for the full guide):

```bash
# Layer 1: framework (once per machine) — puts `orx` on PATH
pip install -e ".[solvers-free]"

# Layer 2: bank (once per machine/team)
orx init

# Layer 3: skill doc (once per harness)
cp -r <repo> ~/.claude/skills/or-experience-bank/     # Claude Code example
```

The agent then drives everything through its native shell tool. No server process, no connection management, no MCP config. The first agent action in a fresh environment is `orx doctor` (environment self-check); `orx status` re-orients after any interruption.

## orx command reference

```bash
# deployment
orx doctor                        # python / bank / solvers / indexes self-check
orx init                          # initialize the bank directory

# online solve chain (run directory = cwd)
orx recall --problem-file p.txt   # start run + fetch planning priors
orx validate                      # L1+L2 gate on model.txt -> stamp
orx signature                     # vocab gate on signature.json -> stamp
orx hints --solver <s>            # bank hints BEFORE codegen
orx solve --solver <s>            # sandbox-execute one branch (repair retry)
orx solve --solver a,b,c          # branches execute CONCURRENTLY (parallel exploration)
orx cross-validate                # >=2 valid branches agree within tolerance
orx gold [--answer <v>]           # record gold verdict (user-provided / consistency-only)
orx append --file exp.json        # admit one experience (gold gate enforced)
orx episode                       # terminal: episode + utility credit
orx new-round                     # archive artifacts for a reflection round
orx status                        # where am I, what's next

# bank
orx query --layer <L> --query "..."   # search modeling/implementation/repair/solving
orx show --id <id>                    # fetch one full record
orx deprecate --id <id> --reason "..."# retire a record (cold archive)
orx stats                             # bank statistics

# offline induction (per cluster, under <bank>/induction/<cluster_id>/)
orx trigger / clusters
orx align / induce / refute / validate-pattern / append-pattern --cluster <id>
```

Every command prints a single compact JSON object on stdout (the ReAct observation); long content goes to files in the run directory.

## Project layout

```
src/or_experience_bank/
├── core/          # schemas, append-only stores, lifecycle, utility tracker
├── modeling/      # modeling gate (GAMS-style DSL), signature extraction
├── experience/    # comparative extraction, admission judge, failure buffer
├── retrieval/     # embedding index, retriever, repair graph, modeling retriever
├── solving/       # orchestrator, execution sandbox, reflection
├── solvers/       # 7 solver adapters + registry
├── induction/     # candidates, encoding, alignment, inducer, counterexample, validation, trigger, pipeline
└── cli/           # orx: stateless commands over file-based runs (stamped chain)
scripts/           # orx entry + demos
references/        # DSL, signature, schema, workflow, examples, lifecycle, solvers
tests/             # unittest cases
```

## Documentation

| Doc | What it is |
|---|---|
| [SKILL.md](SKILL.md) | the agent's operational contract (ReAct workflow over `orx`) |
| [references/workflow.md](references/workflow.md) | end-to-end flow from the command-calling perspective |
| [references/modeling-contract.md](references/modeling-contract.md) | GAMS-style DSL syntax, AUXILIARY block, L1/L2/L3 verification |
| [references/structural-signature.md](references/structural-signature.md) | signature schema, controlled vocabularies, alignment rules, examples |
| [references/experience-schema.md](references/experience-schema.md) | record schemas: ModelingExperience, ExperienceRecord, Episode, lifecycle |
| [references/bank-lifecycle.md](references/bank-lifecycle.md) | utility attribution, soft delete, cold archive, anti-resurrection |
| [references/induction-pipeline.md](references/induction-pipeline.md) | offline induction loop + orx command mapping |
| [references/examples.md](references/examples.md) | 3 worked examples: positive, ambiguous, negative |
| [references/solver-adapters.md](references/solver-adapters.md) | 7 solver adapters, result.json contract, sandbox rules, solver API notes |
| [docs/deployment.md](docs/deployment.md) | deployment guide: framework / bank / skill layers |
| [docs/project-overview.md](docs/project-overview.md) | full project introduction (Chinese, for group meetings) |
