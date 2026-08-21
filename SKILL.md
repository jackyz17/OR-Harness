---
name: or-experience-bank
description: Build, solve, debug, and compare Operations Research models with verified-upfront modeling (think->model->verify gate), isolated parallel solver branches (Gurobi / SCIP / HiGHS / COPT / OR-Tools / PuLP / Pyomo), intra-branch sequential repair, outer reflection on gold-answer mismatch, comparative success/failure experience synthesis, an append-only layered experience bank (Modeling / Implementation / Repair / Solving) plus problem-level Episodes, and offline structural induction that distills solver-validated cross-family optimization patterns from accumulated realizations. Use for natural-language OR modeling, solver code generation, infeasible/unbounded/timeout/API-error repair, OR experience retrieval, multi-solver comparison, offline principle induction, or experience-bank inspection.
---

# OR Experience Bank

Use **Heterogeneous Parallel Solver Exploration with Intra-Branch Sequential Repair**, front-loaded by a **verified Structured Modeling stage** and closed by **gold-gated comparative experience synthesis**. An offline **structural induction** pipeline distills solver-validated cross-family patterns from accumulated experience.

### Full flow at a glance

```
[A] = you produce   [F] = framework executes   #n = your output # (see §1)

NL Problem
    │
    ▼
[F] Normalize → recall planning priors [E1],[E2]... ◀─── reflow (validated records)
    │
    ▼
[A] #1  IMDI + <model> GAMS-DSL → [F] ModelingGate (L1+L2+[A]#3 L3, ≤3 rounds ↺) → [A] #2 signature
    │
    ▼
[F] Fan out → 7 solver branches (Gurobi/SCIP/HiGHS/COPT/OR-Tools/PuLP/Pyomo, parallel)
    │   [A] #4 generate code → [F] sandbox+validate result.json → repair ↺ (≤max_attempts)
    ▼
[F] Cross-solver validation → select best → Episode base record (gold pending)
    │
    ════════ solve(defer_extraction=True) → evaluate_with_gold(gold) ════════
    │
    ▼
[F] |objective - gold| ≤ tolerance?
    │
    ├─ yes → [A] #5 synthesis (success vs FailureBuffer) → [A] #6 admission judge
    │              → [F] append to bank → [F] utility credit [uses En] → [F] episode supplement
    │
    └─ no  → [A] #7 reflection ("why was modeling direction wrong?") → back to modeling ↺ (≤3 rounds)

    ──────────── OFFLINE (triggered by accumulation, separate from online loop) ────────────

    Modeling Bank → [F] cluster → [F] encode → [A] #8 align → [A] #9 induce (hypothesis)
    → [A] #10 counterexample (framework executes refutation code) → [F] validate (#11 unseen transfer)
    → validated → append peer (status=validated) | refuted → archive (not in bank) → reflow ──↺
```

**Your 11 outputs**: #1 think+model · #2 signature · #3 semantic judge · #4 solver code · #5 synthesis · #6 admission · #7 reflection · #8 alignment · #9 hypothesis · #10 counterexample · #11 transfer solver

---

## 0. You ARE the LLM — read this first

This framework is a TOOL under a harness agent. The split of responsibilities (**D18**):

- **Framework owns rules**: schemas, controlled vocabularies, the `ModelingGate` verifiers, parsing, dedup, append-only stores, the derived repair graph, **retrieval** (query construction, embedding search, ranking, filtering), and all prompt *templates*.
- **Agent owns the LLM**: the agent generates the `<think>`/`<model>`, the structural signature, the solver code, the comparative synthesis, the judge verdicts, and (offline) the alignment, hypothesis, and counterexample outputs.

**In a harness environment you ARE the LLM.** There is no external API, no wrapper script, no `openai` package. You read the framework's prompts, produce the answer with your own reasoning, and the framework validates/executes/stores it. The framework retrieves relevant experience from the bank and injects it into each prompt **automatically** — you do not select or query the bank yourself.

**Anti-patterns to avoid**: writing an OpenAI/HTTP wrapper script, reading API credentials from the environment, adjusting framework timeouts to reach an API, manually querying the bank before solving.

### How the framework calls you

Every LLM touchpoint goes through an injected `LLMClient` with two methods:

| Method | When | Your response format |
|---|---|---|
| `generate_text(prompt) → str` | Modeling (think+model), solver code, reflection | Free text; code may use ```python fences |
| `generate_object(prompt) → Any` | Signature, semantic issues, synthesis, judge, alignment, hypothesis, counterexample | **Raw JSON only** — no markdown fences, no prefix text |

In CLI `--interactive-llm` mode, the `StdinLLMClient` prints each prompt to stderr (`===== LLM PROMPT (respond text|json, end with '<<<END_LLM>>>') =====`) and reads your answer from stdin until a line containing only `<<<END_LLM>>>`.

---

## 1. Agent Output Contract

The framework will ask you for these outputs. Each has a strict format the framework parses and validates. **Return only what is asked — do not add extra text outside the expected format.**

### Online solve + gold evaluation

| # | When | Method | Format | Framework validates/stores |
|---|---|---|---|---|
| 1 | Modeling stage | `generate_text` | `<think>...\n<model>...</model>` | L1 format, L2 structural, L3 semantic |
| 2 | Signature extraction | `generate_object` | `{"objective":"...","decision":[...],"constraint":[...],"interaction":"...","features":{...}}` | Controlled-vocabulary check |
| 3 | Semantic judge (L3) | `generate_object` | `[{"type":"...","detail":"..."}]` or `[]` | Feeds issues back if non-empty |
| 4 | Code generation (per branch) | `generate_text` | Complete Python | Executes in sandbox, reads `result.json` |
| 5 | Comparative synthesis | `generate_object` | `[{"layer","title","retrieval_text","polarity","diagnosis","action","rationale"}]` | Routes to admission judge |
| 6 | Admission judge | `generate_object` | `{"accept":true/false,"reason":"..."}` | Appends to bank if accepted |

### Outer reflection (gold mismatch)

| # | When | Method | Format | Framework does |
|---|---|---|---|---|
| 7 | Reflection | `generate_text` | Free text — new modeling direction | Feeds back into a fresh modeling round |

### Offline induction (separate from online loop)

| # | When | Method | Format | Framework does |
|---|---|---|---|---|
| 8 | Alignment | `generate_object` | `{"roles":[...],"bindings":[...]}` (each binding must cite a `realization_id`) | Grounded check; drops ungrounded bindings |
| 9 | Hypothesis generation | `generate_object` | `{"statement":"...","rationale":"...","complexity":N}` | Stamped `status=hypothesis` |
| 10 | Counterexample search | `generate_object` | `{"conditions":[...],"refutation_code":"..."}` | **Executes** refutation code; only executor verdict decides |
| 11 | Transfer solver | injected callable | `float` (objective) | With/without principle comparison; no improvement → refuted |

---

## 2. Structured Modeling — what you generate

When the framework sends the modeling prompt, it has **already**:
- Normalized the problem `[framework]`
- Retrieved planning priors from the Modeling Bank (all records as `[E1]...`, `[E2]...`) `[framework]`
- Injected them into the prompt `[framework]`

**Your job `[agent]`**: produce `<think>` + `<model>` in the GAMS-style DSL.

### Output format

```
<think>
Your analysis: identify objective, decision variables, constraints,
key structural insights, and which injected principles (if any) you applied.
If you applied a principle, cite it: [uses E1]
</think>
```

### Concrete example — a valid model

<model>
SETS:
  a in Animals = {cow, sheep, chicken}

PARAMETERS:
  sell_price[a]
  feed_cost[a]
  manure_rate[a]
  manure_limit
  max_chickens
  min_cows
  min_sheep
  max_total

VARIABLES:
  x[a] integer >= 0

OBJECTIVE:
  maximize sum(a, (sell_price[a] - feed_cost[a]) * x[a])

CONSTRAINTS:
  C1: sum(a, manure_rate[a] * x[a]) <= manure_limit
  C2: x[chicken] <= max_chickens
  C3: x[cow] >= min_cows
  C4: x[sheep] >= min_sheep
  C5: sum(a, x[a]) <= max_total
</model>
```

### GAMS-style DSL rules (critical — L2 will reject violations)

- **Five required blocks**: `SETS`, `PARAMETERS`, `VARIABLES`, `OBJECTIVE`, `CONSTRAINTS` (exact spellings, each on its own line, ending with `:`).
- **Constraint labels must be `C1:`, `C2:`, ...** — the L2 structural validator only recognizes `C\d+` as a constraint label. Descriptive names like `manure:` or `chicken_cap:` are treated as **undeclared symbols** and rejected.
- **No inline `#` comments** in PARAMETERS/VARIABLES lines — the parser may truncate the symbol name.
- **Set members go in braces**: `a in Animals = {cow, sheep, chicken}`.
- **Symbolic indexing**: use `x[a]`, `x[i,t]` etc. — declared in SETS, referenced in VARIABLES/PARAMETERS/OBJECTIVE/CONSTRAINTS.
- **Every symbol referenced in OBJECTIVE/CONSTRAINTS must be declared** in SETS/PARAMETERS/VARIABLES. The L2 validator does a cross-reference check.

See [modeling-contract.md](references/modeling-contract.md) for the full DSL specification.

### What happens after you submit the model

1. `[framework]` **L1 format check**: tags present, five blocks non-empty.
2. `[framework]` **L2 structural check**: declared-vs-referenced symbol cross-reference + signature consistency.
3. `[framework]` **L3 semantic check** (if LLM available): you are asked to judge whether the model has missing/spurious constraints vs the original problem — return `[]` if faithful, or `[{"type":"...","detail":"..."}]` for defects.
4. If any layer fails, `[framework]` feeds the issues back into a new modeling prompt and you retry (≤3 rounds).
5. If the model passes, `[framework]` extracts a **structural signature** (asks you for the JSON) and proceeds to solver branches.

### Structural signature — what you generate

When asked to extract a signature, return a JSON object with four core dimensions (controlled vocabulary) plus an open features slot:

```json
{
  "objective": "linear",
  "decision": ["integer_batch"],
  "constraint": ["capacity", "covering"],
  "interaction": "shared_resource_coupled",
  "features": {"resource": "shared_scarce", "domain": "livestock_farm"}
}
```

**Controlled vocabularies** (values outside these are rejected and retried):

| Dim | Cardinality | Allowed values |
|---|---|---|
| `objective` | single | `linear` \| `convex` \| `minmax` \| `multi_objective_weighted` \| `feasibility_only` |
| `decision` | list | `binary_assignment` \| `integer_batch` \| `continuous_flow` \| `multi_index_2d` \| `multi_index_3d` |
| `constraint` | list | `capacity` \| `flow_conservation` \| `assignment_exactly_once` \| `covering` \| `precedence` \| `big_m_linking` |
| `interaction` | single | `independent` \| `shared_resource_coupled` \| `fixed_charge_coupling` \| `nonlinear_interaction` |
| `features` | open | Any descriptive keys. Recommended: `temporal`, `network`, `resource`, `uncertainty` |

See [structural-signature.md](references/structural-signature.md) for alignment rules and examples.

---

## 3. Solve flow — step by step with responsibilities

### Step 1: Problem normalization `[framework]`

The framework normalizes the problem into `{problem_family, objective, entities, constraints}`. You do not participate.

### Step 2: Planning priors retrieval `[framework]`

If `modeling_retriever` is wired, the framework retrieves relevant modeling records from the Modeling Bank and injects them into the modeling prompt as `[E1]...`, `[E2]...`. All records are peers — no distinction between directly-solved and induced. You do not control what gets retrieved.

### Step 3: Structured modeling `[agent]` → `[framework]`

You produce `<think>` + `<model>` (see §2). The framework validates (L1/L2/L3). On failure, it feeds issues back and you retry (≤3 rounds). **No solver branch is created until the model passes.**

### Step 4: Solver availability check `[framework]`

The framework checks which of the 7 solvers are installed (module + license). Missing solvers become warnings, not errors.

### Step 5: Parallel solver branch execution `[agent]` + `[framework]`

For **each available solver**, the framework:
1. `[framework]` Retrieves Implementation Bank records for that solver.
2. `[framework]` Builds the codegen prompt (includes verified model, solver context, implementation/modeling experience hits).
3. `[agent]` You generate complete Python code implementing the verified model for that solver. Write `result.json` in the branch directory.
4. `[framework]` Executes the code in `SafePythonExecutor` (timeout, rlimits, no network, no shell).
5. `[framework]` Validates `result.json` (schema, solver status, objective, variables).

#### result.json schema (mandatory — your code MUST write this file)

```json
{
  "status": "optimal|feasible|infeasible|unbounded|timeout|error|unknown",
  "solver": "highs",
  "objective_sense": "minimize|maximize|feasibility|unknown",
  "objective_value": 1234.5,
  "objective_bound": 1234.0,
  "mip_gap": 0.001,
  "runtime_seconds": 1.23,
  "variables": {"x[0]": 1, "x[1]": 0},
  "diagnostics": {},
  "message": ""
}
```

**Critical**: the field is `objective_value` (NOT `objective`). The `status` value must be **lowercase** (e.g. `"optimal"`, not `"Optimal"`). An exit without this file is a failure even if stdout claims success.

#### Security sandbox (blocked imports)

The `SafePythonExecutor` AST-validates your code before execution. The following top-level modules are **blocked**:

`subprocess`, `socket`, `urllib`, `http`, `requests`, `pathlib`, `shutil`

**`import os` and `import os.path` are ALLOWED** — but dangerous `os.*` calls are blocked: `os.system`, `os.popen`, `os.exec*`, `os.remove`, `os.listdir`, `os.walk`, `os.environ`, etc. Use `os.path.join()` / `os.path.exists()` freely.

If you need file I/O, use the built-in `open()` — but it is restricted to writing `result.json` in the branch directory only. For `json` output, `import json` is allowed. For `sys`, `import sys` is allowed (e.g. `sys.exit()`).

If your code is rejected, the `normalized_error` field will contain `"security policy: blocked import <module>"` or `"blocked call os.<func>()..."`. Check `BranchResult.execution.normalized_error` or `AttemptRecord.normalized_error` to see the rejection reason.

On failure (code error, infeasible, timeout):
6. `[framework]` Records the failure in `FailureBuffer`.
7. `[framework]` Retrieves Repair Bank records + rebuilds the error-transition graph for `(solver, normalized_error)`.
8. `[framework]` Builds a repair prompt (includes latest code, latest feedback, repair guidance, repair hits).
9. `[agent]` You generate a **complete** corrected code version (not a patch).
10. Repeat 4-9 (≤ max_attempts). Stop on: success, repeated error, unchanged code, timeout, max attempts.

### Step 6: Cross-solver validation `[framework]`

After all branches finish, the framework compares valid branches. If ≥2 branches have matching objectives (within tolerance), validation upgrades to `cross_solver_consistent`.

### Step 7: Result selection `[framework]`

The framework selects the best branch by validation level, solver status, then objective/bound/gap/runtime.

### Step 8: Episode base record `[framework]`

The framework records a problem-level Episode (model, signature, branch outcomes, failure count). **No experience is appended yet** (deferred to gold evaluation).

---

## 4. Gold evaluation — two-step harness flow (Option A)

In the harness, gold arrives **after** solving. So `solve()` and evaluation are separate:

### Step 9: Gold comparison `[framework]`

Someone (the user, or you the agent) provides the gold answer. The framework calls `evaluate_with_gold(gold)`:
- **Match** → proceed to comparative synthesis (Step 10).
- **Mismatch** → proceed to reflection (Step 11).

### Step 10 (match): Comparative synthesis + admission `[agent]` → `[framework]`

1. `[framework]` If there are buffered failures, builds a **contrast prompt** (success vs all failures). If no failures, builds a **success-only prompt**.
2. `[agent]` You produce a JSON array of experience candidates — each with `layer`, `title`, `retrieval_text`, `polarity`, `diagnosis`, `action`, `rationale`. Self-classify each lesson by where it applies: `modeling` (formulation), `implementation` (API mechanics), `repair` (error→fix), `solving` (performance/solver-choice).
3. `[framework]` For each candidate, builds a judge prompt.
4. `[agent]` You judge: `{"accept": true/false, "reason": "..."}`.
5. `[framework]` Appends accepted candidates to the right bank: `modeling` → `ModelingStore` (as a peer record with structural signature and `modeling_aspect`); others → flat store. Appends the Episode gold supplement.
6. `[framework]` If `utility_tracker` is wired and planning priors were recalled, credits `utility_count` for cited experiences (precise via `[uses En]`). This closes the loop with soft-delete scoring.

### Step 11 (mismatch): Outer reflection `[agent]` → `[framework]`

1. `[framework]` Builds a reflection prompt (includes problem, gold, selected objective, per-branch outcomes).
2. `[agent]` You analyze why the **modeling direction** was wrong (not the code) and state how the model should change.
3. `[framework]` Feeds your reflection back into a fresh Structured Modeling round (Step 3), creating a new solve cycle (≤3 outer rounds).

### Failure paths

| Situation | What happens | What you should do |
|---|---|---|
| Modeling gate fails 3 rounds | Framework returns `modeling_gate_failed`, no branches run | Re-read the problem; if you cannot model it, tell the user |
| Gold mismatch after 3 reflection rounds | Framework returns mismatched verdict | Report to user that the problem may need a different approach |
| No solver available | Framework raises `NoSolverAvailable` | Tell user which solvers to install |
| All branches fail/infeasible | Framework returns no selected branch | Report branch failures to user; suggest reformulation |

---

## 5. Offline structural induction ("举一反三")

Separate from the online solve loop. The framework mines accumulated Modeling Bank records for **heterogeneous but structurally-isomorphic** clusters and induces cross-family optimization principles.

| Signal | Meaning | How to check |
|---|---|---|
| Accumulation watermark | Enough new realizations since last run | `stats --json` → compare realization count to last induction |
| Heterogeneous clusters exist | ≥2 problem families sharing a structural signature | `induce --auto` checks this automatically |
| Cooldown elapsed | Cluster membership hasn't changed since last run | `induce --auto` skips unchanged clusters |

**Rule of thumb**: trigger after every N=3+ new realizations accumulate AND you've solved problems from ≥2 different families (e.g. assignment + scheduling + inventory). Induction over a single family produces no cross-family insight.

### What you do during induction (7 stages, one pass per cluster)

| Stage | `[agent]` or `[framework]` | What you produce |
|---|---|---|
| 1. Candidates | `[framework]` | Discovers isomorphic + cross-family clusters (inverted index, no O(N²)) |
| 2. Encoding | `[framework]` | Batch-encodes/verifies signatures |
| 3. Alignment | `[agent]` | `generate_object`: role mappings (e.g. `resource_pool` ↔ warehouse/machine/labor). **Each binding must cite its source `realization_id`** — ungrounded bindings are dropped. |
| 4. Inducer | `[agent]` | `generate_object`: candidate principle(s) as `status=hypothesis` (never knowledge yet). Grounded in the alignment, not a free-text summary. |
| 5. Counterexample | `[agent]` + `[framework]` | `generate_object`: failure conditions + refutation code. **The framework executes the code; only the executor verdict decides.** A crashed refutation is NOT a counterexample. |
| 6. Validation | `[framework]` | Source consistency + **unseen transfer** (with vs without principle) + scoring `αC+βT+γV+δN−λK−μX`. **No transfer improvement → refuted.** This is the gate that makes induction ≠ summary. |
| 7. Pipeline | `[framework]` | `validated` → append as a new peer record (sources untouched, append-only, `status=validated`). `refuted` → archived in run report, NOT in the bank. |

### Providing unseen tasks for transfer validation

The transfer solver requires unseen OR tasks that were **not** in the source cluster. You provide them via `--unseen-task "..."` (repeatable) or the Python API `unseen_tasks=[...]`.

**How to choose good unseen tasks**:
- Same structural signature (same O/D/C/I) but a **different problem family** than any cluster member.
- Same mathematical skeleton, different business domain (e.g. if the cluster is inventory+production, use workforce scheduling as the unseen task).
- You must also supply a **transfer solver** — a callable that takes `(task_text, principle_or_None)` and returns an objective value. In harness mode this is the agent's own solve capability.

**Critical gap**: the CLI `induce --interactive-llm` path does **not** inject a real transfer solver (it raises `RuntimeError`). To complete real induction validation, use the **Python API**:

```python
pipeline = InductionPipeline(
    ...,
    transfer_solver=my_transfer_solver,  # must be injected
    unseen_tasks=["..."],
)
```

See [induction-pipeline.md](references/induction-pipeline.md) for the full pipeline.

### Pattern reflow into online solving (Phase 4.1, D5)

Validated records (`status=validated` only — never `refuted`) feed back into the online modeling stage as planning priors. When wired (`modeling_retriever` + `utility_tracker` injected):

1. `[framework]` Recalls modeling records `[E1]...`, `[E2]...` before modeling.
2. `[agent]` You cite applied experiences with `[uses En]` inside `<think>`.
3. `[framework]` Parses citations — only injected `En` tags map to ids (you cannot invent a citation).
4. `[framework]` On gold match, credits `utility_count` for cited experiences (precise attribution). Low-utility records sink in retrieval ranking (soft delete, never physical deletion).

This closes the full loop: **solve → experience → induction → reflow → better solve**.

---

## 6. Experience bank overview

### Layers and stores

| Bank | Schema | Store file | What it holds |
|---|---|---|---|
| **Modeling** | `ModelingExperience` (all records are peers; `modeling_aspect` classifies: constraint/objective/variable/classification/structure) | `bank/modeling_bank.jsonl` | Solver-independent modeling methods. Records from direct solving (`status=null`) and induction (`status=validated`) are peers. Own store, own index. Carries the structural signature. |
| **Implementation** | Flat `ExperienceRecord` | `bank/implementation.jsonl` | Solver/API mechanics. |
| **Repair** | Flat `ExperienceRecord` | `bank/repair_bank.jsonl` | Error→repair→outcome. A derived **error-transition graph** (single graph, `(solver, normalized_error)` composite nodes, generality-gated migration) is rebuilt on demand. |
| **Solving** | Flat `ExperienceRecord` | `bank/solving_bank.jsonl` | Timeouts, gaps, bounds, numerics, parameters, solver choice. |
| **Episode** | `EpisodeRecord` | `episodes/episodes.jsonl` | Problem-level scene snapshots (model, signature, branch outcomes, gold, produced realization ids). Two-phase append-only (base after solve, supplement after gold). Raw material for offline induction. |

### Rules (non-negotiable)

- **Append-only**: record content is never edited or physically deleted. Content-hash dedup, Episode provenance, and induction `derived_from` depend on immutable content.
- **Failure experiences are never appended alone**: they are buffered in `FailureBuffer`, then contrasted with the eventual success by a synthesis step. Only synthesized, judge-admitted lessons are appended.
- **Lifecycle**: `active` → `deprecated` (harmful/low-utility; moved to cold archive). Mutable state/stats live in sidecars (`lifecycle.json`, `utility_stats.json`), never in the fact store.
- **Utility + soft delete**: `retrieval_count ≥ 5` AND `utility/retrieval < 0.1` → score ×0.3. Never deleted.
- **Cold archive**: deprecated records compressed to provenance cards at `archive/deprecated.jsonl` (bulky fields dropped, embedding vector kept). Keeps hot bank small.
- **Anti-resurrection**: `append()` rejects candidates matching a deprecated archive entry — by exact content-hash or by embedding cosine similarity ≥0.8. A retired experience cannot re-enter in either form.
- **Retrieval is automatic**: the framework constructs queries, searches (cosine similarity + metadata filters), ranks (utility-adjusted), and injects into prompts. You never manually query the bank during solving.
- **Only `validated` patterns may surface to online solving** (Phase 4.1). `hypothesis`/`refuted` records are never used as knowledge.

---

## 7. Common mistakes

| ❌ Mistake | ✅ Correct |
|---|---|
| Writing an OpenAI/HTTP wrapper script | You ARE the LLM; answer prompts directly |
| Reading API credentials from the environment | No credentials needed in harness mode |
| Using descriptive constraint labels (`manure:`, `chicken_cap:`) | Use `C1:`, `C2:`, ... — L2 only recognizes `C\d+` |
| Adding inline `#` comments in PARAMETERS lines | Put comments in `<think>` instead |
| Referencing undeclared symbols in OBJECTIVE/CONSTRAINTS | Declare every symbol in SETS/PARAMETERS/VARIABLES first |
| Returning JSON wrapped in ```json fences for `generate_object` | Return raw JSON, no fences, no prefix |
| Generating a code patch instead of complete code | Always return the complete latest code |
| Manually querying the bank before solving | Retrieval is automatic; the framework injects experience into prompts |
| Using `hypothesis` patterns as knowledge | Only `validated` patterns surface to online solving |
| Not providing unseen tasks for induction transfer | Supply `--unseen-task` or `unseen_tasks=[...]`; without it, induction cannot validate |
| Expecting CLI `induce --interactive-llm` to complete transfer validation | CLI raises `RuntimeError` for transfer solver; use Python API |
| Trusting solver self-report instead of `result.json` | Framework requires `result.json`; stdout claims are ignored |
| Writing `"objective"` in result.json | Field name is `"objective_value"` (see schema in Step 5) |
| Writing `"Optimal"` (capitalized) in result.json | Status must be **lowercase**: `"optimal"` |
| `import os` in solver code | `import os` is **allowed** (for `os.path`); but `os.system`, `os.popen`, `os.environ` etc. are blocked. See security sandbox in Step 5 |
| Not knowing why a branch failed | Check `result.branch_errors` (maps `branch_id` to `normalized_error`) or `branch.normalized_error` |

---

## Appendix A: CLI reference (standalone/demo only)

The CLI is for **standalone mode** (no harness agent: cron, batch) and **demos/tests**. In harness mode, use the Python API directly.

```bash
# Solve (mock demo — no solver/LLM needed)
python3 scripts/or_experience_cli.py solve --mock-demo --problem "..." --json

# Solve (standalone with external LLM wrapper)
python3 scripts/or_experience_cli.py solve --llm-command "wrapper-cmd" --problem "..." --json

# Solve (harness interactive — YOU answer prompts on stdin)
python3 scripts/or_experience_cli.py solve --interactive-llm --problem-file problem.txt

# Offline induction (mock demo)
python3 scripts/or_experience_cli.py induce --mock-demo --json

# Offline induction (auto-triggered)
python3 scripts/or_experience_cli.py induce --auto --min-new-realizations 3 --json

# Full induction walkthrough
PYTHONPATH=src python3 scripts/demo_induction_walkthrough.py

# Bank management
python3 scripts/or_experience_cli.py stats --json
python3 scripts/or_experience_cli.py rebuild-index --json
python3 scripts/or_experience_cli.py validate-bank --json
python3 scripts/or_experience_cli.py retrieve --layer modeling --query "..." --json
python3 scripts/or_experience_cli.py append --input experience.json --json
```

**Note**: CLI `solve` currently runs the single-shot flow (solve + auto-extract in one call). The two-step harness flow (`solve(defer_extraction=True)` + `evaluate_with_gold(gold)`) is only available via the Python API. CLI `induce --interactive-llm` cannot complete transfer validation (raises `RuntimeError` for the transfer solver).

Use `--config config.example.yaml` or environment variables. Runtime data defaults to `~/.hermes/or-experience-bank`; override with `OR_EXPERIENCE_BANK_HOME`.

For dependency-free demos, use `--mock-demo --solvers mock-a,mock-b`. Treat mock output as test/demo only, never as a real OR result.

---

## Appendix B: Output contract

When reporting results to the user, include: selected branch, branch termination reasons, validation level, objective comparability, discrepancies, warnings, retrieved experience IDs, appended experience IDs, gold verdict, and episode reference. Never expose secrets, full environment variables, unredacted absolute user paths, or oversized tracebacks.

---

## References

| Doc | What it is |
|---|---|
| [references/modeling-contract.md](references/modeling-contract.md) | GAMS-style DSL syntax, constraint label rules, three-layer verification |
| [references/structural-signature.md](references/structural-signature.md) | Signature schema, controlled vocabularies, alignment rules, examples |
| [references/induction-pipeline.md](references/induction-pipeline.md) | Offline induction loop: 7 stages, trigger policy, scoring formula |
| [references/experience-schema.md](references/experience-schema.md) | Record schemas: ExperienceRecord, ModelingExperience, Episode, lifecycle |
| [references/workflow.md](references/workflow.md) | Online solving workflow (14 steps) |
| [references/trajectory-schema.md](references/trajectory-schema.md) | AttemptRecord, BranchResult, SolveResult, termination values |
| [references/solver-adapters.md](references/solver-adapters.md) | 7 solver adapters, result contract, execution controls |
| [references/prompts.md](references/prompts.md) | Prompt templates for all 11 agent outputs |

Design decisions and roadmap: [docs/redesign-plan.md](docs/redesign-plan.md). Implementation status: [docs/PROGRESS.md](docs/PROGRESS.md).
