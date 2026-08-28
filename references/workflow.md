# Workflow

The end-to-end flow from the `orx` command-calling perspective: what you (the
agent) produce, what command you run, and what the framework enforces at each
step.

## Runtime layout

Runtime data lives in two places. The **bank** is persistent and shared across
runs; the **run directory** is your per-problem working directory (typically
your cwd). `OR_EXPERIENCE_BANK_HOME` overrides the default
`~/.hermes/or-experience-bank`.

```text
<bank_home>/
bank/                 immutable JSONL facts: modeling_bank.jsonl + flat layers
episodes/             problem-level scene snapshots (append-only, two-phase)
index/                rebuildable embedding vectors and metadata
archive/              deprecated.jsonl (compressed cards) + deprecated_index.json
induction/            per-cluster induction working directories
logs/                 optional operational logs

<run_dir>/            your working directory for ONE solve
problem.txt  priors.json  model.txt  signature.json
stamps/  branches/  cross_validation.json  gold.json
experiences.json  episode.json  rounds/  journal.jsonl
```

Derived indexes (embedding index, repair error-transition graph) are
rebuildable from facts at any time; facts are the only source of truth.

## Online solve flow (Part 1)

```
NL Problem
    │
    ▼
[T] orx recall ──────────── validated patterns [E1],[E2],... (reflow)
    │
    ▼
[A] [THINK]/[MODEL] ──► [T] orx validate ── L1+L2 gate ──► stamps/model.json
    │                        issues? ⟲ fix model.txt, retry freely
    ▼
[A] signature JSON ───► [T] orx signature ── vocab gate ──► stamps/signature.json
    │
    ▼
[T] orx hints ────────── bank hints BEFORE codegen
[A] solver code ──────► [T] orx solve --solver a,b,c ── concurrent sandbox exec
    │                     (branches run in PARALLEL; repair within a branch
    ▼                      is the agent's serial retry loop)
branches/<s>/result.json
    │
    ▼
[T] orx cross-validate ── ≥3 valid branches agree within tolerance
    │
    ├─ gold match ──► [A] synthesis ──► [T] orx append ×N ─┐
    └─ mismatch ───► [A] reflection ──► orx new-round ↺    │
    │                                                       │
    ▼                                                       │
[T] orx episode ◄── terminal: episode.json + utility credit ┘
```

`[A]` = you produce the content; `[T]` = you run the command, the framework
verifies/executes/stores.

Step-by-step responsibilities:

1. **recall** `[T]` — pull validated patterns before modeling into
   `priors.json`. Cite any you apply with `[uses En]` inside the `[THINK]` block.
2. **validate** `[A] produces, framework gates]` — L1 format + L2 structural.
   Failure returns structured `issues`; fix model.txt and retry (free, no
   penalty). The stamp records the content hash — editing model.txt afterwards
   invalidates it.
3. **signature** `[A] produces, framework validates]` — controlled
   vocabularies; out-of-vocabulary values return errors, fix signature.json
   and retry (the model stamp is unaffected).
4. **hints + solve ×≥3** `[A] produces code, framework executes]` — pull
   hints BEFORE writing each branch's solve.py; write ALL branch codes, then
   run them in ONE command: `orx solve --solver a,b,c` executes the branches
   **concurrently** (asyncio.gather bounded by `max_parallel_branches` — the
   heterogeneous parallel exploration contract). Each branch runs in its own
   sandbox with AST validation; the per-branch result.json carries bank hints
   (implementation always; repair + graph guidance on failure; solving on
   performance symptom). **Repair within a branch is serial by design**: a
   failed branch is fixed by editing ONLY that branch's solve.py and
   re-running `orx solve --solver <failed>`.
5. **cross-validate** `[T]` — ≥3 branches (configurable via `min_cross_validation_branches`,
   default 3) must agree within tolerance. On inconsistency, add another
   branch and re-run (no token was consumed; the signature stamp stays valid).
6. **append ×N** `[A] synthesizes, framework admits]` — synthesize lessons for
   EACH bank layer that had an event during this solve (see the WRITE column
   in the table below). One JSON file per lesson, `orx append --file` each.
   Content-hash dedup + anti-resurrection; the modeling layer gets the run's
   signature.
7. **episode** `[T]` — terminal; credits utility for cited priors on gold
   match; writes episode.json.

**Reflection (gold mismatch)**: you analyze why the modeling direction was
wrong (formulation, not code), run `orx new-round` to archive the failed
round, then re-model from scratch. ≤3 outer rounds. There is no "reflect"
command — reflection is your creative act.

## Experience layers (Part 2)

| Bank | Schema | Store file | READ: Reaches you via | WRITE: When you call `orx append` |
|---|---|---|---|---|
| Modeling | `ModelingExperience` (fused) | `bank/modeling_bank.jsonl` | `orx recall` → `priors.json` `[E1]...` | Gold match: you found a structural insight (constraint type, objective formulation, variable design) |
| Implementation | flat `ExperienceRecord` | `bank/implementation.jsonl` | `orx hints` / `orx solve` → `implementation_hints` | You hit a solver API gotcha (wrong attribute, missing call, format quirk) during any branch |
| Repair | flat `ExperienceRecord` | `bank/repair_bank.jsonl` | `orx solve` (on failure) → `repair_hints` + `repair_graph_guidance` | You hit an error and fixed it (error → fix → outcome during code generation) |
| Solving | flat `ExperienceRecord` | `bank/solving_bank.jsonl` | `orx solve` (on symptom) → `solving_hints` | You hit a performance issue and tuned it (timeout, large gap, numerics) |
| Episode | `EpisodeRecord` | `episodes/episodes.jsonl` | never during solving (induction raw material) | `orx episode` (terminal, automatic) |

Free-form inspection: `orx query` / `orx show` / `orx deprecate`.

## Offline induction flow (Part 3)

Triggered separately from solving (accumulation watermark + heterogeneous
cluster + cooldown). Per cluster, a 5-step stamped chain inside
`<bank>/induction/<cluster_id>/`:

```
[T] orx trigger ──► should_induce?
[T] orx clusters ──► [{cluster_id, families, members}]
    │
    ▼ (per cluster)
[T] orx align ──► alignment.json (template first, you fill, re-run to stamp)
[T] orx induce ──► hypotheses.json (status=hypothesis, never knowledge yet)
[T] orx refute ──► refutations.json (executor verdict decides)
[T] orx validate-pattern ──► validation.json (source consistency + unseen tasks)
[T] orx append-pattern ──► validated peers appended | refuted archived
    │
    ╰──► reflow: validated patterns become [En] priors in future orx recall
```

## Append-only rules

- Existing JSONL lines are never rewritten. Corrections are new records linked
  through `related_experience_ids` / `derived_from_experience_ids` /
  `contradicts_experience_ids`.
- Exact content hashes are rejected as duplicates.
- Failure experiences are never appended alone — they are synthesized into
  contrast lessons at `orx append` time.
- Mutable state (lifecycle, utility counts) lives in sidecars, never in fact lines.
- Deprecated records move to the compressed cold archive; anti-resurrection
  (hash + cosine ≥ 0.8) blocks re-entry.

## Quick start

```bash
# Environment self-check (first action in a fresh environment)
python3 scripts/orx.py doctor

# Initialize the bank
python3 scripts/orx.py init

# Cross-process resumability demo (two sessions, one run)
python3 scripts/demo_orx_resumability.py
```
