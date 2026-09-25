# Command reference

Every command prints exactly one JSON line to stdout: `{"result": {...}, "summary": "agent-readable 2-4 sentences"}`. Exit codes: `0` success, `2` usage/precondition error, `1` crash. All commands accept `--home DIR` (default `$OR_HARNESS_HOME`, else `./or_harness_home`).

## `orx profile --task t.json [--cir cir.json] [--allow-empty-cir]`

The single analysis entry: CIR validation + modeling guidance + problem profile + derivation report, in one call. It serves two purposes: it helps you understand the problem structure, and it produces the structural key that retrieves comparable evidence at `recall` time. A task **without** a `model` field is a normal state — strategy selection needs the task text, the CIR, and the profile. The `model` is written only AFTER the strategy is chosen, so it is verified and reported as a DIAGNOSTIC (`derivation.model_coupling`); it never moves the structural key. There is deliberately NO solve-script parameter: a script is a post-strategy artifact, and no identity-bearing path accepts one.

**CIR side** (`result.coupling`): when the task carries a `coupling` field (or `--cir` is given), the CIR (Coupling-Aware Intermediate Representation — an explicit, inspectable set of entities, decisions, constraints, and relations) is validated and returns `modeling_guidance` with concrete coupling implications for the model you will write. Without a `coupling` field, `coupling.cir=null` with a prompt message (not an error).

**A malformed CIR is a precondition error (exit 2), never a silent empty one.** The shape is checked before anything is parsed, because a misspelled or nested payload would otherwise deserialize to an empty CIR and every consumer downstream would treat a malformed problem as an uncoupled one. The rejection is a structured object: `result.error = {kind: "cir_format", cause, detail, hint, ...}` — `cause` is the failure class and `hint` names the repair:

| `cause` | Meaning | Fix |
|---|---|---|
| `not_object` | The `coupling` value is a string/list/number | Wrap it in an object |
| `unknown_keys` | Unrecognised key(s); `unknown_keys` lists them | Allowed keys are exactly `entities`, `decisions`, `constraints`, `relations`, `coupling_groups`, `issues` |
| `bad_type` | A list key holds a non-list (e.g. `"entities": {...}`) | Use `"entities": [ ... ]` |
| `empty_cir` | A CIR that carries no structure at all | Fill it in, drop the `coupling` field, or pass `--allow-empty-cir` |

Two mistakes get a targeted hint, because they are the ones that actually happen. **Nested**: `{"coupling": {"cir": {...}}}` is rejected with a hint telling you to put the keys directly under `coupling`. **Scalar confusion**: `{"coupling": {"resource_coupling": 0.9}}` is rejected with a hint that scalar coupling belongs in `annotations.coupling`, not `task.coupling`.

`result.coupling.health` is an honest account of what the CIR contributed: `{present, parsed, entities, decisions, constraints, relations, issues, contributes_scalars, note}`. `parsed` is false when the CIR object is present but empty, and `contributes_scalars` is false when it carries no decisions — no scalar dimension can be derived without decisions, so a caller that only checks `cir != null` would believe it had a structural signal it does not have.

`OR_CIR_STRICT=0` downgrades the POLICY checks (`unknown_keys`, `empty_cir`) so a legacy task with extra keys still loads. The downgrade is a LINT, not a silence: the problems appear in `result.coupling.shape_lints`. Structural problems are never downgraded — a payload that cannot be deserialized has no lenient reading.

CIR schema (all `kind`/`type` fields are free-form strings — domain-general, no hard-coded vocabulary):

```json
{
  "coupling": {
    "entities":     [{"name": "R1", "kind": "resource", "attrs": {}}],
    "decisions":    [{"name": "x", "kind": "production", "indexes": ["i", "t"]}],
    "constraints":  [{"id": "C1", "kind": "capacity", "expr": "sum(x) <= limit"}],
    "relations":    [{"source": "x", "target": "R1", "type": "uses_resource",
                       "evidence": "semantic", "detail": "..."}],
    "coupling_groups": []
  }
}
```

Relation `evidence` levels: `structural` (from constraint-variable co-occurrence only — never a semantic claim), `semantic` (supported by entity/constraint semantics), `declared` (agent-asserted). Co-occurrence alone produces generic `depends_on` edges; a semantic relation like `uses_resource` requires additional evidence (resource-kind entity + capacity constraint).

The `modeling_guidance` entries each carry `{type, members, resource, implication}` — e.g. `shared_bottleneck`: "Multiple decisions (x, y) consume the same resource (R1); ensure one aggregate capacity constraint covers all relevant decisions."

**Profile side** (`result.profile` + `result.derivation`): coupling derivation priority — CIR (task JSON `coupling` field, best — derived from its relations/indexes) > structured `spec` fields > your `annotations.coupling` supply. `semantic_coupling` is never derived. **The key is IDENTITY, frozen before the strategy**: the `model` field and a solve script are POST-strategy artifacts, so neither is ever a key source — the model is verified (L1+L2) and its own reading is reported as `derivation.model_coupling` (a diagnostic), so a divergence from the key is VISIBLE without being silently absorbed into it. The `derivation` report carries per-dimension value/origin/notes and `coupling_warnings` when a supplied value contradicts the structural derivation across a bin boundary. The retrieval signature and the modeling guidance come from the SAME structure, not two parallel lines.

## `orx recall --task t.json [--top 3] [--exclude S04 S06] [--candidate ID ...] [--memory-mode M] [--include-unverified]`

Recalls accumulated experience through **two independent channels**. They answer different questions and are never blended into one number — and an empty result on one never erases the other.

**There is no candidate menu.** Nothing ships a list of method names, descriptions or applicability rules. A strategy appears in `recommendations` when the memory really holds something about it in this structural cell — a recorded execution (`conditional_stats`) or an admission-verified claim (`strategic_entry`). With nothing in either, `recommendations` is **empty** and `recommendations_basis.reason` says so; there is no `no_memory` row, no `-inf` score, no zero-quality placeholder.

**Structural channel — "what may I REUSE?"** (`result.recommendations[]`). Score = `α·Q̂ − β·C_scalar − γ·R̂` (weights configurable via `--alpha/--beta/--gamma/--cost-weights`). Evidence precedence per strategy: **published** Strategic Knowledge entry → conditional statistics → *nothing* (the strategy is simply absent). An entry that is not published (an unverified candidate, or one whose admission check was `refuted` / `insufficient_evidence`) is skipped, so recall falls back to the statistics — which is what "we have a candidate but no knowledge yet" should look like. Every recommendation carries `strategy_id` and nothing else about the method's identity: no `name`, no `description`. What the backing entry itself records travels in `recommendation.knowledge` (`entry_id`, `strategy_type`, `actions`, `fallback_strategy_id`, `applicability`, `support_n`, `verification_state`) — empty lists mean the memory does not record it.

`--candidate ID` (repeatable) is **your** proposal set: the methods you are considering. It restricts `recommendations` to those ids and reports the rest under `result.candidates_without_evidence`. It never creates evidence, never scores a memory-less strategy, and never blocks its execution.

`result.recommendations_basis` is always present and names why the list holds what it holds — the three possible empty cases (no memory at all, memory filtered out by your `exclude`/`candidate`/`top`, or `--memory-mode none`) are distinguished in words:

```jsonc
"recommendations_basis": {
  "n_recommendations": 0,
  "evidence_kinds": [],
  "memory_mode": "cost-aware",
  "candidates_with_memory": [],          // ids this cell HAS memory for
  "candidates_proposed": ["custom:x"],   // null when you proposed none
  "reason": "NO MEMORY for this structural cell: ... propose the methods you want to try ..."
}
```

**Relation channel — "what does the memory SAY about this condition?"** (`result.knowledge[]`). Structured relation claims whose conditions match this task, carried **separately** from `recommendations`. The separation is deliberate: `recommendations` is keyed on the strategy ids memory holds and filtered by `is_publishable`, which speaks about the **statistical** claim — so a verified relation whose host statistics were never verified would be filtered out, and a relation-only entry names no strategy id at all. Each item carries `relation_id`, `claim`, `kind`, `conditions`, `evidence`, `tasks`, `verification_state`, `verification_scope`, `published`, and `newer_evidence_since_verification` (matching executions recorded after the verdict — a visibility annotation, not a lifecycle state). Published relations may be used as strategic grounds; `unverified` / `refuted` / stale ones appear only with `--include-unverified`, clearly labelled, and are never dressed up as available knowledge.

**Text channel — "what should I LOOK AT?"** (`result.vector_recall`). The task text is embedded and compared with every indexed memory, **with no structural pre-filter**, so a nearly identical problem from a different structural cell still surfaces. `similarity` is the raw text cosine: it is **never** a quality, cost, or risk estimate and never enters a score.

```jsonc
"vector_recall": {
  "backend": {"source": "remote|local-hashing", "model_id": "...", "dimension": 384},
  "execution_evidence": [{
    "execution_id": "ex_...", "task_id": "t1", "strategy_id": "S04",
    "similarity": 0.87,                 // discovery signal only
    "task_text_excerpt": "…first 200 chars of the memory's own text…",
    "task_text_digest": "…",
    "observed_quality": {...}, "observed_cost": {...},   // what ACTUALLY happened
    "failures": 2, "status": "feasible", "measurement_scope": "attempt",
    "task_check": "passed|failed|insufficient|null",
    "task_check_limitations": ["…present when the answer was NOT validated…"],
    "profile_cell": "family=vrp|rc[0.25,0.50]|...",
    "structural_match": "same_cell|different_cell|unknown"
  }],
  "strategic_knowledge": [{
    "entry_id": "se_...", "strategy_id": "S04", "similarity": 0.81,
    "claim_text": "…applicability notes + strategy description + readable predicates…",
    "expected_quality_hat": 0.9, "expected_cost_hat": {...},
    "cost_measured": [...], "failure_prob": 0.1,
    "status": "candidate", "verification_state": "verified", "support_n": 3,
    "applicability": [...], "risk_conditions": [...], "predicates": {...},
    "structural_match": "applies|conflicts|unknown",
    "reusable": true, "reason": "…present when not reusable…"
  }],
  "stale_indexed": {"execution_evidence": 0, "strategic_knowledge": 0},
  "unindexed": {"execution_evidence": 0, "strategic_knowledge": 0,
                "note": "query these via `orx inspect --bank experience|strategic`"}
}
```

Field discipline: `observed_*` (what happened) and `expected_*` (what the knowledge claims) are separate fields — a promise is never read as a measurement. `task_check` carries the TASK-level verdict when one exists: `failed` means the answer was confirmed NOT to satisfy the task, so its `observed_quality` is a solver-side figure and must not be read as a quality the strategy achieved (`task_check_limitations` says so in words); `null` means **never checked** — which is not a pass. `structural_match` for executions compares `group_key` values; for entries it reports the applicability verdict, where `conflicts` names the contradiction and `unknown` means a value needed to decide is missing. `same_cell` and `applies` are independent judgments with different bases. Dormant, retired and (by default) unpublished entries are filtered out **before** the `top_k` cut, so an ineligible item never takes a slot a usable memory could have filled. `stale_indexed` counts items whose document changed under the index (excluded — the stored vector describes text that no longer exists).

**Degradation is explicit.** When the text channel cannot run, `result.degraded = {"path": "profile_only", "reason": ...}` explains it, and the structural result is returned intact:

| Situation | `degraded.reason` |
|---|---|
| No embedding backend configured (no env, no injection) | `no embedding backend configured` |
| Task JSON has no textual field | `no task text in task JSON` |
| No index file yet | `index missing; run orx rebuild-index` |
| Index built by a different embedding model | `embedding model changed (was X, now Y)` |
| Embedding call failed | `embedding backend error: ...` |

**“Could not look” is not “looked and found nothing”.** A `degraded` block means the channel never ran; a `vector_recall` block with empty hit lists means it ran and nothing matched. The two are reported separately so an empty structural channel plus a degraded text channel is never read as “no similar memory exists”.

The retrieval document is built from `text`, `description`, `objective`, `requirements`, `business_rules`, `constraints` and `spec` (in that order), so a task that states its problem in a plain `text` field is indexed the same way as one using `description`. Only a task with NONE of these keys reports `no task text`.

`vector_recall.unindexed` counts current memories with no vector — legacy records whose text was never captured, or a write whose index sync was deferred — and states where to find them (`orx inspect --bank experience|strategic`, `--bank texts`, and the profile channel).

`--include-unverified` is the offline/inspection view: unpublished candidates appear in BOTH channels (the structural path returns them with a warning; the text path stops filtering by admission state).

When no experience exists for any strategy, `recommendations` is EMPTY and `recommendations_basis` says why. That is the normal cold-start state: propose the methods you want to try and they will be predicted, executed and recorded. `recall` is a memory report, not a menu — `predict`/`execute` accept any id you name.

Result: `result.recommendations[]`, each `{strategy_id, score, expected{quality, cost, failure_prob}, evidence, evidence_refs, confidence, cross_family, risk_warnings, basis, cost_known_dims, cost_basis_dims, knowledge}` plus `result.available_solver_families` (family → usable solver names; pick the concrete solver yourself) and `result.solver_advisories` (solvers with environment-class failures in this memory — e.g. a subprocess-based solver the sandbox rejected before). Note: when `evidence="conditional_stats"`, the `expected` fields report OBSERVED means (a recount from the Evidence Bank), not a knowledge commitment — no interval, no calibration track, no lifecycle.

`--memory-mode`: `none` (memory deliberately NOT consulted: `recommendations` is empty and the basis says it was a choice) | `cases` (statistics, no cost weighting) | `strategic` (entries + statistics, no cost weighting) | `cost-aware` (adds cost scalarization).

**Read-only.** `recall` writes nothing — the query text is embedded in memory only, no text row is created, no index item is touched, no migration runs. Text is persisted on the WRITE paths (`execute` / `record`).

## Embedding configuration (the text channel)

The text channel is **off unless a real embedding model is configured**, deliberately: a lexical hash is not semantic retrieval, and presenting one as semantic similarity would make text matching look meaningful when it is not.

| Setting | Meaning |
|---|---|
| `OR_EMBEDDING_BASE_URL` | OpenAI-compatible endpoint base (e.g. `https://host/v1`) |
| `OR_EMBEDDING_MODEL` | embedding model name |
| `OR_EMBEDDING_API_KEY` | credential (sent as a bearer header, never persisted) |
| `OR_EMBEDDING_BACKEND` | optional: `auto` (default), `local-hashing`, `none` |

These are NOT the `OR_WM_*` chat-model variables: the two models may be different endpoints with different dimensions. `auto` uses the endpoint only when all three of base URL / model / key are present, and otherwise returns no backend at all. It never silently substitutes the local hashing backend — that one is selected explicitly (`OR_EMBEDDING_BACKEND=local-hashing`) for hermetic offline runs and tests, and its index (`model_id=local-hashing-embedding-v1`) is invisible to a real model and vice versa, so two vector spaces never mix. A backend can also be injected directly in Python: `ORHarness(home=..., embedding=MyBackend())`.

The index lives in `{home}/index/execution_evidence.embedding.json` and `{home}/index/strategic_knowledge.embedding.json`. Each item is `{id, doc_digest, vector}` — **no record snapshot**: recall re-reads each hit's CURRENT record by id and compares `doc_digest`, so a stale item can only cause a miss (reported under `vector_recall.stale_indexed`), never a wrong answer. The whole index is derived data: safe to delete, rebuildable at any time.

If the text is English/CJK mixed, both are handled (the local backend tokenizes CJK per character; a real model does its own tokenization).

## `orx execute --task t.json --strategy S04 --code solve.py --workspace DIR --solver NAME [--verification basic]`

You write `solve.py` following the strategy's actions (the framework never generates code). It runs in a sandbox: no network/shell/pathlib, `open()` only for a literal relative `result.json`, POSIX rlimits + wall-clock timeout. Your script must write `result.json` with at least `status` (optimal|feasible|infeasible|unbounded|timeout|error), `objective_value`, `objective_bound`, `mip_gap`, `runtime_seconds`, and — whenever a task-level check will need to read the answer — `variables`, an object of variable name → value (e.g. `{"x1": 2, "x2": 3}`). The solution vector is what makes `orx check-task` possible: integrality and objective recomputation can only be checked against actual values. It is stored on the record as `execution_features.solution_variables` (truncated past 2000 entries, with `solution_variables_truncated` naming what was dropped — a check needing a dropped variable reports it as unchecked rather than passing).

The profile used in `execute` is the same frozen pre-strategy signature from `recall` — derived from `coupling` (CIR) / `spec` / `annotations` / `model` only.

Every execution is automatically staged in a pending area (successes and failures alike) — staging is a safety net, not recording. Result: `result.execution` = a full ExecutionRecord (id, quality check, CostVector, `cost_measured` mask, `solver_runtime_provenance`, `execution_features` with solver diagnostics if the solver reported any, `cir_snapshot` = the CIR that was actually solved when the task carries a `coupling` field, and `task_text_digest` = the task-text VERSION this execution was produced under). **Nothing is recorded yet.**

The executor measures only what it can observe: `latency_s` and `solver_runtime_s`. `tool_calls`, `retries` and `llm_tokens` are **yours to declare** — they stay out of `cost_measured` (a constant is not a measurement). `tool_calls` counts ALL tool invocations in the attempt's scope (shell commands, file reads/writes, sandbox runs, solver calls), so the executor records only its provable floor in `execution_features.tool_calls_lower_bound`. `retries` gets a measured zero only when the framework can PROVE this is the first attempt of that (task, episode, strategy); otherwise it stays unknown. A script rejected by the sandbox policy measures nothing at all (`cost_measured` is empty) — 0.0 seconds of solving would be a fabricated fact. `execution_features.cost_notes` explains any rejected or suspicious value.

`execute` also persists the task text (`task_texts`, keyed by `(task_id, text_digest)`) — that is where the retrieval document comes from. Solving the SAME `task_id` with different content creates separate versions, so each execution stays linked to the text actually in force, and the two can never be confused later.

## `orx check-task <execution_id> [--check JSON] [--episode ep1]`

Answers the question `execute` cannot: **does this answer satisfy the ORIGINAL task?** The executor's own verdict covers the solver's MODEL (a legal status, a finite objective, a gap); a relaxed LP answered with fractional values is `optimal` with `gap=0` and still wrong. Without this call such an answer enters recall, the conditional statistics, the world-model feedback and offline induction as a success sample.

Run it between `execute` and `record`, or later on an already-recorded execution (a late correction is a real event — see the close-out section below).

**`--check JSON`** declares what the framework may verify. Every base is optional; only what you declare is evaluated, and everything else is listed as UNCHECKED in `report.scope.unchecked`:

| Key | Meaning |
|---|---|
| `reference_objective` (+ `tolerance`) | the reported objective must agree with the reference. Default tolerance `1e-6 * max(1, abs(reference))` — the SAME rule admission verification uses |
| `reference_status` | the reported solver status must equal it (e.g. `"optimal"`) |
| `integer` | `{"variables": ["x1","x2"]?, "tolerance": 1e-6?}` — every named variable (or every recorded variable) must be integral. This is the basis that catches an LP relaxation |
| `recompute_objective` | `{"coefficients": {"x1": 7}, "constant": 0?, "tolerance": 1e-6?}` — the objective is recomputed from the recorded solution and compared with the reported one |
| `semantic_probe` | one or more `{"path", equals\|min\|max\|in}` probes over the record payload, e.g. `{"path": "execution_features.solution_variables.x1", "min": 0}` |
| `intent` | `"relaxation"` or `"intermediate"` — marks an execution whose answer is deliberately NOT the task's answer. It is recorded and reported, never treated as a pass for the task |

**Three verdicts, and none of them is a default:**

- **`passed`** — every declared basis held on the recorded values. The report still names what it did not check (the model's fidelity to the task, undeclared constraints), so `passed` is never read as "fully validated".
- **`failed`** — a declared basis ran and did not hold. The execution is **not** demoted or rewritten: its observed quality and cost stand and it stays in the evidence set (the cost is real, the failure is raw material). What changes is that it can no longer count as a success sample: every quality consumer (statistics, world-model feedback, calibration) reads `0.0` for it, while its cost still counts in the task total.
- **`insufficient`** — no basis declared, no solution vector recorded, a needed variable missing, or the execution produced no usable result. **Not a pass, not a failure**, and never a demand that you supply a reference. An unchecked answer's validity is UNKNOWN.

The framework does **not** parse natural-language constraints: a constraint is checked only through a `semantic_probe` or a `recompute_objective` you declare. A matching objective never proves the model correct.

**Recording.** Two channels: a `verify` ACTION (the framework really ran these checks, so it is logged with its verdict — the per-check history is auditable) and `execution_features.task_check` on the execution itself (a narrow annotation that works for staged and recorded executions alike). `task_check.state` is `passed` / `failed` / `insufficient`; a malformed or absent annotation reads as "no check", never as a pass.

**On `failed`, the response carries `reflection_material`** — the task text, the code hash, the recorded solution vector, the check report and the earlier attempts of the episode — plus the `next` instruction. Locating the cause (task interpretation, model, implementation, or the reference basis itself) is YOUR job: the framework does not classify the failure as a modeling mistake, does not rebuild the model, and never relaxes the task to match a reference value.

## `orx predict --task t.json --strategy S`

Pre-execution cost expectation for one (task, strategy), with explicit provenance: `source=entry` (a matching Strategic Knowledge entry), `stats` (conditional statistics over attempt-scope evidence — a recount, with `support_n` and per-dimension `support_per_dim`), or `unknown` (no usable evidence: no data, task-scope data only, or clearly different scale — expected cost is `null`, NEVER a default zero presented as cheap). Pass the printed snapshot to `orx record --prediction` so feedback compares against the prediction actually used.

## `orx record --execution <json|path> | --from-staged <id> [--override llm_tokens=1840,tool_calls=9] [--override-mode replace|increment] [--prediction <json>] [--retain-reason contrast] | --discard-staged <id>`

Appends the fact to the Execution Evidence Bank, then runs the automatic chain: cost backfill (replace by default — idempotent, never double-counts) → quality checks against matching entries, frozen ONTO THE FACT (`execution_features.quality_feedback`: the interval in force, the observed quality, hit/miss — only the strategy that actually ran is checked, attempt scope only) → cost feedback against the record's frozen pre-execution prediction snapshot (same strategy, same scope, both sides measured; written into `execution_features.cost_feedback`) → induction-pattern hint checks (the `intervention_recovery` pattern detects cross-execution recovery chains automatically: a failed attempt under one solver followed by success under another).

**Recording never changes knowledge.** No entry is promoted, demoted, or woken here — the frozen checks are replayed by the next `orx induce`.

`--retain-reason <free text>` explicitly marks this episode as representative evidence, reserved for future compaction policies. When omitted, any mark the record already carries is preserved — no automatic marking is performed.

**Cost completeness.** `result.cost_completeness` appears when any dimension is still unmeasured: `missing` names them, `lower_bounds` carries the provable floor (the sandbox invocation), and `note` gives the exact `orx amend-cost` call that closes the gap. Recording is never blocked — the fact is honest as it stands, and an unmeasured dimension is recorded as UNKNOWN rather than fabricated — but an unmeasured dimension supports no cost claim and no cost prediction until every supporting record measures it. `--execution` accepts the bare record, the `orx execute` envelope, or the legacy one-level form.

**Task text and index sync (best effort).** `record` can be reached WITHOUT `execute`, so it resolves the text link itself: the digest the caller supplied, else the task's most recent real belief snapshot (`hypothetical=False`, whose frozen `task_payload` is rebuilt through the same reader used at capture time). If neither exists, `task_text_digest` stays `None` and the record is simply not vector-indexed — the text is **never invented**, and the memory keeps its full profile-based visibility. After the fact is durable, the execution's index item is refreshed; an embedding failure does **not** roll anything back and is reported as `result.index_sync = {state: "deferred", reason: ...}` (recover later with `orx rebuild-index`). `state` is `synced` / `deferred` / `skipped` (`skipped` = no backend configured, or the text is unavailable).

## `orx amend-cost <execution_id> --override llm_tokens=1840,tool_calls=9 [--mode replace|increment]`

Backfills cost dimensions of an **already-recorded** execution, in place — nothing is re-run and nothing is appended. This is the repair path for a cost gap (`record`'s `cost_completeness.missing`, `induce`'s `cost_claim_withheld`): an unmeasured dimension supports no cost claim until every supporting record measures it, and this is how you close it without re-executing.

`--mode replace` (default) means the value IS the measurement — re-applying the same override is idempotent and never double-counts. `--mode increment` adds an additional measured amount within the record's scope. Cost feedback is recomputed against the frozen prediction snapshot, so the stored summary can never disagree with the stored fact.

A `tool_calls` declaration below the sandbox's provable lower bound is refused (exit 2): the executor demonstrably ran the solve script, so the record itself refutes the number. The response reports `still_missing` — the dimensions that remain unmeasured — so you can see whether the gap is closed.

Result: `result.{execution_id, recorded, prediction_checks[], cost_feedback?, induction_hints[]}` and, when same-task executions are staged but unrecorded, `result.unrecorded_staged_executions[]` — backfill those with `--from-staged` (records the original payload verbatim; never re-type an execution JSON by hand). Always backfill `llm_tokens` here — it is invisible to the sandbox. A task's full cost is the sum of its recorded attempt-scope records (`api.task_cost_summary`): every attempt charged to the strategy that actually ran it; retries = sum of per-attempt NEW retries; end-to-end latency is unknown unless you supply explicit task timing (never inferred by max or sum).

## `orx induce [--strategy S | --all] [--family F] [--cell GROUP_KEY] [--peer-strategy S] [--peer-cell GROUP_KEY] [--relation JSON] [--rebuild] [--dry-run] [--force] [--note TEXT] [--verify JSON]`

Consolidates facts into Strategic Knowledge (your explicit call — hints never auto-induce). This is the ONLY place knowledge changes: it (i) builds or refreshes the claim of each (family, structural cell, strategy) evidence set — its applicability is read off the supporting records (the cell its evidence occupies, never a cross-sample span that could stretch across incomparable regions); and (ii) replays the frozen checks recorded on the facts, reporting what it did under `result.revisions` (promotion at ≥5 checks / ≥70% hits **with a verified claim**, demotion at 3 consecutive misses, dormancy wakeup). New entries are born `candidate` with honest intervals (width floored by sample size; n=2 cannot claim [0.95, 1.0]). One evidence set owns exactly one claim: re-inducing refreshes it rather than restating it (dormant entries included, so a woken claim keeps its id). `--rebuild` re-induces the whole Strategic Knowledge Bank from currently retained evidence (cold archive preserved); the result may differ from the previous bank — exact reconstruction is not a requirement. The framework fills in NONE of an entry's `strategy_type` / `actions` / `fallback_strategy_id` — there is no built-in directory to copy them from, and inferring a method's actions from its id would be fabrication. What the harness writes is what travels; an empty field means the memory does not record it. `--dry-run` never writes — including under `--force` (a rehearsal does not lift a cold-archive veto).

**Independent evidence gate (creation only).** Creating a claim takes at least 2 supporting executions from ≥2 distinct `task_id`s: repeating one task is repetition, not reproduction. Every count comes from the same facts — executed, attempt-scope, in the target's structural cell — so a task-scope total can never stand in for an independent attempt. When only one task is behind the evidence the call creates nothing and returns `verification {tasks, required_tasks}` plus a `skipped` reason — the evidence is NOT discarded (recall still answers from it as `conditional_stats`), and refreshing an entry that already exists is never blocked, no matter how repetitive the new evidence is.

**`--verify JSON` publishes the candidate.** Admission and creation are separate: the gate above decides whether evidence may become a *candidate*; this decides whether the candidate's claim may become *published knowledge*. See [induction.md](induction.md#admission-verification-offline-what-makes-a-claim-published) for the payload and the three checks (`rule` / `repair` / `cost_saving`). The framework computes the verdict from the executions you supply; without it the entry is `unverified` and **is not published** — recall answers from `conditional_stats` and `predict_cost` will not quote it.

**`--family` / `--cell` scope the check to ONE structural unit.** A verification payload is written for a specific claim, so it must apply to a specific target set — one payload applied to every cell of a strategy would cross-contaminate families (an allocation check overwriting a scheduling verdict). `--family F` restricts induction to that family; `--cell GROUP_KEY` restricts it to one structural cell (the full `group_key` token, e.g. `family=routing|rc[..]|tc[..]|rx[..]`, compared against each record's DERIVED key so a stale index column never mis-selects). Either flag implies the `--all` scope — naming a unit is itself an explicit selection — and composes with `--strategy`.

What the verdict requires, concretely: a declared criterion the framework can evaluate itself (`reference_status`, `reference_objective`, `semantic_probe`) or a genuinely independent comparison — **feasibility alone is a precondition, not a check**, and a bare `semantic_ok` boolean is recorded as `agent-declared` but cannot carry a verdict. Every execution you pass is evaluated (a counterexample anywhere in the batch refutes, whatever the order), the evidence must be for THIS strategy and family, and a comparison must be like-for-like (same task, same measurement scope for costs; a repeated execution id is not an independent comparison). Failed executions yield `insufficient_evidence` (not `refuted`): "we could not check it" is not "we checked it and it failed".

Each element of `result.results` reports `created` (entry id) / `updated` (entry id) / `skipped` (human-readable reason: fewer than 2 executions, needs independent evidence, cold-archive veto, restatement, or `recorded as an unverified candidate`), plus `verification` when a check ran — read the reason, it tells you what the memory is still missing.

**Cost claims are complete-or-silent.** A cost dimension enters the entry's claim only when EVERY supporting record measured it: a mean over a subset ("3 of 4 records reported tokens") would be a partial observation published as a full claim, and it feeds strategy selection and world-model prediction as if it were the whole truth. When a dimension falls short, the entry is still created and its complete dimensions are still published, but the incomplete one is withheld and reported under `cost_claim_withheld {dimensions: {dim: {n_measured, n}}, supporting_executions, note}`. The fix is a backfill — `orx amend-cost <execution_id> --override <dim>=<value>` amends the fact in place, nothing is re-run — after which the next `induce` restores the claim.

`--note "TEXT"` (optional, repeatable, you phrase it): free-text applicability notes attached to the entries this call creates or refreshes. They are kept for the reader and shown by `inspect`; they never enter scoring — the framework does not pretend to verify a sentence. Because these notes are part of an entry's retrieval document, the knowledge index items are refreshed after this call (`result.index_sync`, best effort — same `synced`/`deferred`/`skipped` contract as `record`). `--dry-run` writes nothing, index included.

**Induction reads relations, not only one cell's means.** `--peer-strategy S` names another strategy to compare against inside each target's OWN cell (the `strategy_contrast` pattern); `--peer-cell GROUP_KEY` names another structural cell to compare the SAME strategy against (the `advantage_reversal` pattern, i.e. where its advantage weakens or flips). Both are repeatable, and both are **read-only**: each relation is written as one line under the entry's `risk_conditions` naming the observed difference (the peer label, its n, its mean quality, and only the cost dimensions measured on EVERY record on both sides — a partial mean would make the ratio meaningless), and reported under `result.results[].peer_relations`. Peer evidence never enters the target's statistics, never satisfies the admission gate, and never creates an entry on its own: a contrast is a reason to look, not a claim by itself. Without these flags the induction output is unchanged.

**`--relation JSON` submits a STRUCTURED relation claim** (repeatable). This is the path for cross-task knowledge that is **not** one strategy's statistics — a modeling principle, a necessary condition, a repair pattern. Payload:

```json
{"subject": "principle:cross_period_state",
 "claim": "the sentence being asserted",
 "evidence": [{"execution_id": "ex_..", "role": "dropped"},
              {"execution_id": "ex_..", "role": "preserved"}],
 "conditions": {"predicates": {"family": "scheduling",
                               "temporal_coupling": [0.5, 1.0]}},
 "check": {"assertions": [...]},
 "kind": "intervention_recovery"}
```

`claim` and `evidence` are required; every evidence entry needs an `execution_id` for a **recorded** fact and a `role` naming the part it plays in *this* claim (free strings — `dropped`/`preserved`, `before`/`after`, `strategy_a`, …). The framework DERIVES `tasks`, `family`, structural cell and `strategy_ids` from those facts, so a caller never submits a second, contradictory identity. `subject` is optional and matters for knowledge that names no strategy id: with no host entry the claim creates a **relation-only entry** (`support_n = 0`, no statistical claim) whose `strategy_id` is the subject; with a host it is appended to that entry's `relations`. `conditions` are the applicability predicates (omitted → read off the evidence's cell). `kind` is an optional note about what prompted the claim — never a verification template.

Verification reuses `--verify` with `purpose: "relation"`; each assertion carries its own semantics (`probe` / `status` / `comparison`). `comparison` declares `metric`, `roles_a`/`roles_b`, `direction`, `min_gap`, `mode` (`paired` compares only same-task pairs, one per side; `group` compares the sides' means over a metric every referenced record measured), and `aggregation` (`all` = every pair must meet the gap, one comparable counterexample refutes; `mean` = the batch mean must). Pairing is per assertion, never a property of the whole batch. The verdict is `verified` only when every declared assertion held — "no violation found within this scope", never a guarantee about future tasks; `insufficient_evidence` covers an unmeasured metric, a missing counterpart or nothing computable declared (**not** a refutation); `refuted` means an assertion ran on real evidence and failed. Only the declared parts are covered: "quality higher AND tokens lower" needs both assertions.

**Publication is per relation**, on two conditions: its own verdict is `verified` (and not stale), and its verification scope covers ≥2 distinct tasks. A single-task relation is SAVED and verifiable as a fact about that task, but reported as not published (`result.relations[].publication.reasons` says why). Neither condition touches the host entry's statistical claim, and the host's admission never grants the relation anything. Re-submitting the same `subject`+`kind` REVISES that relation (a substantive change without a fresh verdict marks it `stale_after_revision`; a fresh verdict wins; an identical re-submission keeps the verdict). Result: `result.{relations[], saved, published}`. See [induction.md](induction.md#structured-relation-claims-induce---relation).

## `orx inspect --bank experience|strategic|archive|actions|snapshots|predictions|texts [--task ID] [--strategy S] [--status candidate] [--episode EP]`

Queries a memory layer. Strategic entries include `prediction_track {n_predictions, hit_rate, calibration_error, consecutive_misses}`, `status` (candidate|validated|suspect|dormant), `provenance` (execution ids). `actions` / `snapshots` query the world-model substrate (below): the unified action log and the frozen belief snapshots.

`--bank texts` is the **retrieval source documents** — the task-text versions captured on the write paths, keyed by `(task_id, text_digest)`. It is not a knowledge bank: it makes no claim, feeds no statistic, and has no lifecycle; it exists so the embedding index has a source and so a memory with no vector is still findable. Use it to inspect **which text version an execution was produced under**: every execution carries `task_text_digest`, and `--task ID` lists that task's versions. With `--task` omitted, every retained version is listed. This is the documented look-up entry point for the records counted under `vector_recall.unindexed`.

## `orx snapshot --task t.json [--episode ep1]`

Freeze and persist the current **belief state** for a task (the world-model substrate): H (value-copied verified/legacy/unverified knowledge refs + experience counts + tool config), P (profile + task digest + CIR/model digests + the problem-relevant task payload when no `task_ref` exists), X (the episode's accumulated progress — selected plan, model artifact, current solution, verification evidence — each field labelled with `provenance` (observed vs agent_reported) AND `epistemic` (fact/inferred/unknown), two orthogonal dimensions), B (declared budget + consumption), and a coverage view (cell statistics + layered knowledge, no composite capability score). Snapshots are frozen by deep-copy isolation: later bank writes never change what a snapshot meant. Within an episode, each action's post state merges into the accumulated progress, so the next action inherits what earlier actions established.

## `orx action --report TYPE --task t.json [--episode ep1] [--params JSON] [--outcome JSON] [--cost JSON] | --amend-cost ACTION_ID --cost JSON`

Report an action **you** performed (TYPE ∈ model/select_strategy/verify/finish_task) — the library did not execute it; the record is labelled `source=agent_reported` and a missing pre-state is marked `pre_snapshot_missing`, never fabricated. Explicitly supplied cost dimensions (including an explicit zero) are marked measured. `--amend-cost` backfills an action's cost (replace semantics, idempotent).

## `orx budget --task ID [--episode ep1] [--declare llm_tokens=50000,...]`

The budget view for a task/episode: consumption over ALL real action costs — recorded AND staged-but-unrecorded executions (deduplicated by execution_id) plus own-cost actions (a macro's reference cost is never re-counted; its own additional spend is). Status: `exceeded` (a measured dimension over the limit; a declared latency budget is judged per-attempt), `ok` (every declared dimension measured and within), `unconfirmed` (known spend within limits but a declared dimension is unknown — NOT confirmed within budget), `no_budget_declared`. Executions of the task with no episode linkage are reported separately as `unattributed` and never charged to a fresh episode. Declarations persist in the store, so separate CLI invocations see the same budget. Hypothetical actions are excluded from every real consumption view.

`orx execute --episode ep1` and `orx induce` automatically record their actions in the unified log: execute as a macro (`pre` snapshot before execution, `post` after, `rollup=reference`, linked to the execution id); induce in the **maintenance scope** (`__maintenance__` / `maint_<ts>`) with a real pre/post knowledge state, verification results, a knowledge delta, and a business result that separates `created` (verified) from `created_unverified` (a candidate is NOT knowledge growth), `updated` / `revised` / `refused` / `unchanged`. A crashed induction still leaves a `failed` action with its pre state. Dry-run persists nothing.

## `orx context --task t.json [--episode ep1] [--top 3] [--cir cir.json] [--math JSON] [--include-unverified] [--no-persist]`
## `orx context --context-id CTX_ID`

**The frozen prediction input context (world-model phase 2).** One call gathers everything a prediction may condition on, freezes it, and persists it. Full spec: [references/prediction_context.md](prediction_context.md).

Build mode (`--task`) freezes ONE belief snapshot, runs the EXISTING recall path once (both channels), and assembles:

- the **joint problem representation**: the task's own text and payload, the CIR's relations (kept as relations, not three coupling numbers), the math attributes each with an **origin**, the structural profile and its derivation report;
- **X/B** from that same snapshot (accumulated progress + budget state);
- the **retrieval evidence** of both channels, deduplicated by identity;
- the **harness capability evidence** (`H = F(M, W_OR, Pi, R, T)`, evidence statuses only, no composite score) plus a capability VERSION identity;
- the **external execution constraints** (declared budget + consumption status, available solver families, executor limits).

```bash
# a task with scene text and an optional CIR, and NO model field
orx context --task t.json --episode ep1 --top 3

# declare math attributes you actually know (origin recorded as 'declared')
orx context --task t.json --math '{"linearity": "linear", "objective_kind": "min"}'

# read a stored context back — re-runs NOTHING (no embedding, no model)
orx context --context-id ctx_ab12cd34ef56
```

**What it does NOT do**: no prediction-model call, no solver execution, no induction, no capability re-measurement. The only external call is the configured embedding backend on the existing retrieval path — a read.

**A task with no `model` field is a normal input.** So is a task with no CIR, and one with no textual field. Each absent part is listed in `result.missing` with what it means; nothing is guessed from `family`. Read `result.joint.math` — every attribute carries its `origins`, and `result.joint.unknowns` names what nothing established.

**Degradation is per part.** `result.degraded[]` names the part and the reason:

| Situation | `degraded[].reason` |
|---|---|
| No embedding backend configured | `no embedding backend configured` |
| Task JSON has no textual field | `no task text in task JSON` |
| No index file yet | `index missing; run orx rebuild-index` |
| One layer's index unusable | the part is `retrieval.semantic.<layer>` |
| Embedding call failed | `embedding backend error: ...` |
| The channel ran and found nothing | **no `degraded` entry** — `semantic.status` is `ok` |

An empty evidence set with a degraded channel is **not** "nothing comparable exists", and `result.retrieval.notes` says so.

**Evidence classes** (`result.evidence_classes`) keep kinds apart: `execution_fact`, `verified_knowledge`, `unverified_knowledge`, `legacy_knowledge`, `structural_recommendation`. Retrieval never upgrades one into another, unverified candidates are hidden unless `--include-unverified` is passed, and one memory hit by both channels is ONE piece of evidence (`identity = layer:id`) carrying both channel names.

**Reuse is version-verified.** `result.task_digest` is the task VERSION the context was built for. Passing a recall result or context whose version disagrees is refused with a named reason — never silently aligned.

**One CIR for the whole request.** `--cir` is resolved once and drives the joint representation, the snapshot's structural cell and the retrieval; a supplied CIR REPLACES the task's own `coupling` (check `result.joint.sources.cir`: `caller_supplied` / `task_coupling` / `none`). Without this one request could carry two structural judgments and retrieve knowledge from the wrong cell.

**Bounded to the frozen moment.** With a historical `--snapshot`, memories created AFTER it are excluded from the retrieval and reported under `execution_constraints.retrieval_bounding` (`dropped` / `unbounded_kept`).

**Historical reconstruction is not creation-time filtering.** With a `--snapshot`, the knowledge view comes from the snapshot's **frozen** `coverage.knowledge_layers` (so an entry revised afterwards is read at the value it had), the retrieval is only what was saved with the snapshot, and the reliability and cell-evidence blocks — which a snapshot does not save — are reported MISSING rather than read from today's banks. Every historical build says so in `result.missing`.

**Structure consistency is checked after the effective CIR is resolved.** A snapshot taken under a different structure than the effective input (explicit CIR included) is REFUSED with the conflicting dimension named; the same check guards context reuse at `predict-outcome`. An unmeasured dimension is not a mismatch.

**The memory version digests content.** `result.capability_version.knowledge_content_digest` covers the decision-relevant content of the memory actually consulted (entry fields — including `applicability` and `actions` — plus the carried hits). Revising an entry moves it; a re-read does not (read timestamps are listed under `excluded_keys`). Only the digest and its composition are carried, never the digested content.

## `orx [--world-model URL::MODEL] predict-outcome --task t.json --action-spec spec.json [--episode ep1] [--parent-action ACTION_ID] [--context CTX_ID | --no-context]`

**World-model outcome prediction (shadow mode).** Ask a configured world model for a structured prediction of ONE candidate action's consequences, from the frozen pre-action state. The candidate is an `ActionSpec` JSON (`{"action_type": "execute_strategy", "task_id": ..., "strategy_id": "S01", "solver": "highs", ...}`) — a hypothesis, NOT a recorded action. The prediction carries: expected execution status / feasibility / quality / failure risk / per-dimension cost, predicted successor state changes (hypothetical — never written to real state), the model's self-reported confidence (**uncalibrated**), its claimed evidence basis, and fields it explicitly declined to predict.

- **Input context** (phase 2): by default one context is built for this call and sent to the provider under the request's `prediction_context` key; the stored prediction records `model_info.prediction_context_id` so the exact frozen input can be resolved later. `--context CTX_ID` reuses a FROZEN context (its identity is verified against this task/version/episode — the way several candidates of one decision share one input, and the way a stored context replays without reading today's banks). Reusing a context replays its FROZEN conditions: the request's `state` (X/B), the `candidate_knowledge_targets` and the `prediction_reliability` all come from the context, and `model_info.conditions_source` says `frozen_context`. The one exception is the BUDGET: it is an external limit, not a prediction condition, so the pre-call check uses the current ledger and a difference is REPORTED (`model_info.budget_checked_at_call_time`) without rewriting the frozen constraint. `--no-context` sends no context at all, so the request keeps its pre-phase-2 shape exactly. The context changes what the model is GIVEN, not the output protocol — that switch belongs to the next phase.
- **Configuration boundary**: `--world-model BASE_URL::MODEL` (OpenAI-compatible endpoint; API key from `$OR_WM_API_KEY`). Credentials never persist. The world model is MANDATORY: without a provider the command returns an explicit `not_configured` BLOCKER — the online loop and offline consolidation cannot run to spec. No other command invokes a model.
- **Shadow discipline**: the prediction changes NOTHING. `recall`, `predict`, `execute`, `record` behave identically whether or not you predict. You remain the decision-maker.
- **Calibration duty (mandatory)**: one predict–bind pair per executed action is REQUIRED — executed prediction–comparison pairs are the only source of calibration evidence, and there is no legitimate skip. After a `plan-next` selection, bind the selected path's prediction instead of predicting again. (`plan-next` itself is optional.)
- **Call cost**: the model call's own spend (tokens/latency from provider usage) is recorded on the prediction and, with `--parent-action`, charged to that action's own cost — separate from the PREDICTED cost of the target action.
- **Timing**: predict BEFORE executing. The input snapshot is frozen at prediction time; later bank changes never rewrite it.

### What the model is asked to predict (H included)

The prediction covers X and B **and H** — what the action does to accumulated experience and strategic knowledge. Which of these are requested depends on `--prediction-mode` (global flag, or per-`plan-next`):

| Mode | Requested | Knowledge term in scoring |
|---|---|---|
| `x-b-only` | X/B only (no knowledge targets sent) | off |
| `h-x-b` | X/B and H | forced off |
| `h-x-b-value` (default) | X/B and H | live at `--delta` (default **0.0**) |

A `knowledge_changes` item names a **target** — an existing entry (it must truly exist in the frozen knowledge view; a model cannot invent knowledge) or a hypothesis (allowed, but it must declare `expected_observation` and `check_condition`) — plus a **change** (`adds_evidence` / `supports` / `revises` / `refutes` / `candidate_forms` / `narrows`), a **horizon** (`after_execution` or `after_consolidation`), optional **preconditions**, and a **prediction basis**. The framework supplies the candidate targets and all magnitudes; the model contributes a direction, so it cannot raise its own value by asserting confidence.

### How knowledge predictions are judged

Verdicts are **stage-partitioned** so the stages never block each other: the immediate X/B comparison and the knowledge verdicts coexist on one prediction, and a later stage can still be written after an earlier one.

| Stage | Triggered by | Judged against |
|---|---|---|
| `after_execution` | `record` | the evidence that execution produced (did it land in the predicted cell; is the observed quality inside the claim's own interval) |
| `after_consolidation` | the next `induce` | the induction's recorded transition (`entries_created` / `entry_changes`) |

States: `pending` (opportunity not yet arrived, or a precondition unmet — **never counted as a failure**), `fulfilled`, `contradicted`, `missed`, `inconclusive`. A stage verdict is written once; re-running is a no-op.

Honesty boundaries: an entry **forming** is not publishable strategic knowledge (admission verification still decides that), and a cell's mean quality moving does not **prove** a rule — the verdict text says so.

Resolved verdicts aggregate into `prediction_class_reliability` (per change × horizon). That measured reliability — never the model's self-reported confidence — is what later predictions may draw on; a class below the minimum sample count reports `reliability: null` (`insufficient_history`) and grants no value.

## `orx bind-induction-outcome --assessment ASSESSMENT_ID`

Bind an `assess-induction` assessment's predictions to the induction that actually ran, and record the verdict. This is the **maintenance decision's** own slow feedback: "was it right to expect this induction to form a claim?"

It is deliberately **not** the same as the ordinary-action knowledge evaluation above: that one judges what a *solving* action predicted about the knowledge it fed; this one judges the *induction decision's* expectation. Both are needed and neither substitutes for the other.

The verdict is computed from the induction that actually followed — never from the assessment's own opinion of itself. A rejected or deferred recommendation is `inconclusive`, not a miss: the predicted consequence was never given a chance to occur. Idempotent: a second call returns the stored verdict.

## `orx bind-outcome --prediction ID --action ACTION_ID|EXECUTION_ID`

Bind a prediction to the real action that ran, then compare. `--action` accepts **either** the action id (`ac_...`) **or** the `execution_id` (`ex_...`) that the action produced — the latter is what `orx execute` prints, so the natural predict → execute → bind loop needs no separate lookup, and a mismatch between the prediction's episode and the execution's episode does not block binding (it is recorded as a `binding_mismatch` and the comparison covers only the matching parts). Type/strategy/solver are checked: a mismatch (you predicted strategy A, executed B) is recorded and NOT scored — no counterfactual truth is fabricated. The comparison covers only fields both sides define: status category, feasibility, quality (when the execution produced a solution), and cost per dimension (both sides measured — the same both-sides-measured discipline as cost feedback). Missing comparisons are listed with reasons. The feedback is APPENDED to the frozen prediction — the original is never modified, and re-running the comparison is idempotent (the model is never re-invoked). Online comparison records facts and errors only; knowledge updates still go through explicit offline induction with verification.

Query predictions with `orx inspect --bank predictions [--task ID]`.

## `orx [--world-model URL::MODEL] predict-strategy --task t.json --candidate c.json [--episode ep1] [--context CTX_ID] [--cir cir.json]`

**Strategy-outcome prediction under the wm-so/1 protocol.** Predict ONE candidate strategy's benefit / cost / risk / uncertainty from a frozen prediction input context. The candidate is a `CandidateRef` JSON (`{"action_type": "execute_strategy", "strategy_id": "S01", "solver": "highs", "config": {"time_limit": 60}, "scope": "attempt"}`) or a legacy `ActionSpec` (execution params and budget hint preserved verbatim; an unmappable scope such as `"task"` is refused). Same strategy_id with a different solver/config is a DIFFERENT candidate — the config travels with the prediction into the choice and the execution binding.

- **Input**: `--context CTX_ID` reuses a frozen context (identity verified against the task/version/episode AND the effective input version — pass the SAME `--cir` you built the context with); the default builds one fresh context for this call. The provider receives the full context CONTENT (joint problem representation, retrieval evidence, capability evidence, constraints), not ids.
- **Output**: a `StrategyOutcomePrediction` — benefit with metric/unit/baseline (`solution_quality` must be NORMALIZED in [0,1]; a raw objective value is refused, never clamped), cost as a CostVector with a predicted-dimension mask, risk as named events separate from cost, uncertainty with the model's self-report recorded as explicitly UNCALIBRATED. Every failure state (not configured, provider error, empty payload, invalid JSON, NaN, out-of-range probability) is distinguishable and persisted with whatever usage the call consumed. One call, no retries, no defaults.
- **Binding**: after you execute the candidate, `orx bind-strategy --prediction ID --action ACTION_ID` records the prediction–execution linkage (identity checked: task/episode/strategy/solver/config; a mismatch is recorded, never scored). Full spec: [strategy_outcome.md](strategy_outcome.md).

## `orx bind-strategy --prediction ID --action ACTION_ID`

Bind a strategy-outcome prediction to the real action that ran. The binding checks request identity (action type/task/episode/strategy/solver and the candidate's execution config) and sets `trace.comparable` only when the bound action is a completed real scope. **Unknown is not a match**: a field the executed action could not observe (no episode recorded, a config key the action log never carries) is recorded under `binding_unknown` — separately from a known `binding_mismatch` — and the fields that depend on it stay unevaluable at close-out. Idempotent (a re-bind re-evaluates the comparability); no model call; nothing re-billed. The per-field evaluation itself happens at episode close-out (`orx close-episode`); see [episode_closeout.md](episode_closeout.md).

## `orx close-episode --task ID [--episode ep1] [--terminal completed|failed|aborted|budget_exhausted] [--finish-action ACTION_ID] [--min-samples N]`

**Episode close-out.** Closes ONE episode: evaluates its bound strategy-outcome predictions against their real outcomes (field by field: benefit error under the prediction's own declared metric/baseline, per-dimension cost error where both sides measured the same scope, Brier scores for labelled risk events, interval coverage) and publishes the experience calibration summary that later episodes' prediction contexts read. Reads what was recorded — no solver run, no model call, no induction. Unfinished actions are reported (their predictions stay pending, never fabricated into endings); a failed/aborted/budget-exhausted episode closes honestly under its own terminal state. Idempotent: re-closing returns the stored record and counts nothing twice.

**A close-out is the end of the episode, NOT a certification of the answer.** The result carries `task_checks = {n_executions, verdicts, unchecked, note}` reporting how many of the episode's executions carry a task-result verdict (`orx check-task`). An execution confirmed NOT to satisfy the task contributes `0.0` to the benefit observation and the gate is recorded in `evaluations[].benefit.task_check_gated` — so a reader seeing "predicted 0.8, observed 0.0" can tell a confirmed-wrong answer from a genuinely bad solve. Unchecked answers keep their observed value and the note says their validity is UNKNOWN. Full spec: [episode_closeout.md](episode_closeout.md).

**Late corrections change later USE, not history — and they are FIELD-SCOPED.** A task check run on an already-closed episode writes the verdict onto the fact, but the STORED evaluation is never rewritten: it is the honest record of what was known then. The calibration uses a LIVE re-derivation instead, reported in `validity_corrections[]`:
- a check that now FAILS re-derives the benefit observation to `0.0` while **keeping the measured cost** (`kind: "live_rederivation"`, `fields: ["benefit"]`, `counted: true`) — a wrong answer still cost what it cost;
- a check that was WITHDRAWN restores the un-gated observation;
- an execution WITHDRAWN from the evidence set (`exclude-execution`) removes the sample entirely (`exclusions.validity_corrected`);
- the correction is scoped to the executions the evaluation actually compared, so a late verdict on one execution does not disqualify a sibling's evaluation.

When the affected episode is still in the calibration WINDOW the published summary is REPUBLISHED as part of the correction (the result carries `calibration_republished`), so later predictions see the corrected judgment without waiting for a manual rebuild.

## `orx calibration [--min-samples N] [--rebuild]`

Read the published strategy-outcome experience-calibration summary (wm-calib/2). Aggregated from the WINDOW — the newest N closed task-episodes (default 50, `OR_CALIBRATION_WINDOW`) — so this is a single-row read of the last published object, not a history scan. Groups are keyed by `strategy_outcome|<model identity>|<metric>|<unit>|<scope>`: a different model/metric/unit/scope is a different group and never pools. Each group reports directed statistics (`mean_benefit_signed_error`, `mean_cost_log_ratio`), interval coverage AND width, per-event Brier means, and `mean_predicted_probability` alongside `scored_occurrence_rate` (same denominator). The summary also carries `occurrence` — the framework's own event counts by OBSERVATION UNIT (`n_observation_units`, `n_occurred`/`n_not_occurred`/`n_unknown`, `unit_occurrence_rate` over its own denominator), which must never be subtracted from a mean probability computed over a different sample set. Below the sample minimum a group reports `insufficient_evidence` with `reliability: null`; no figure is invented. `applicability: "global_diagnostic"` states plainly that these are global statistics with no per-strategy breakdown. This is the OR strategy-outcome reliability, kept separate from the legacy knowledge-prediction reliability, and a measured record — not a promise that future predictions improve.

`--rebuild` rebuilds and republishes the summary from the current window instead of reading the published one: the explicit migration path for a store that predates this version, and the repair path for an unpublished summary.

## `orx evaluations [--task ID] [--episode ep1] [--evaluation EVALUATION_ID]`

List (or read one) stored post-hoc evaluations of strategy-outcome predictions (read-only). Excluded evaluations are neither hits nor misses; pending ones wait for their scope to end. An excluded evaluation's `risk.observed_units` still feed the occurrence statistics — exclusion removes a prediction from the scoring samples, not the execution from what happened.

## `orx archive-calibration [--dry-run]`

Move OUT-OF-WINDOW episode detail to the archive (retention). Only DETAIL moves — evaluation, strategy-outcome prediction and frozen-context payloads — into `{home}/archive/calibration/calibration-NNNN.jsonl`; the close-out registry tombstone stays ONLINE so a repeated close remains idempotent and the window stays locatable. Episodes with an UNCHECKED execution are held online for the late-check grace period (`OR_CALIBRATION_LATE_CHECK_GRACE_DAYS`, default 30) so a late verdict can still land; an episode whose executions all carry a verdict is not held. The archive is bounded by THREE caps — per file (`OR_CALIBRATION_ARCHIVE_MAX_FILE_BYTES`, 64 MB), total (`OR_CALIBRATION_ARCHIVE_MAX_TOTAL_BYTES`, 1 GB) and age (`OR_CALIBRATION_ARCHIVE_RETENTION_DAYS`, 365) — whichever bites first evicts the oldest file, so the archive cannot grow without bound. Re-running is idempotent. A restored payload never re-enters the calibration automatically. `--dry-run` reports what would move.

## `orx retention`

Report the online/archive retention state and the three separate scopes (window / late-check grace / archive caps) as separate numbers, plus the online counts (window episodes, closed total, predictions, contexts) and the archive files with their sizes.

## `orx [--world-model URL::MODEL] predict-capability --operation JSON [--task t.json] [--bundle bundle.json] [--horizon TEXT] [--horizon-tasks N] [--budget JSON] [--task-id ID] [--episode ep1] [--timeout S]`

**Capability-evolution prediction (`wm-ce/1`).** Predicts what ONE offline learning operation (`--operation` JSON: `operation_type` of `induce`/`revise`/`reverify`/`retire`, plus `strategy_id`/`config`) would change in FUTURE task performance: `expected_changes` (metric, unit, direction, value, baseline), `learning_cost`, `degradation_risk`, `uncertainty` and `verification_conditions`. The FRAMEWORK fixes the operation identity, the experience scope (from the bundle's own executions), the task targeting, the horizon and the per-metric baselines; the model fills prediction content only and may CITE a frozen baseline, never set one. Cost references are frozen **per dimension** (`cost:solver_runtime_s`, `cost:llm_tokens`) because several dimensions share the `resource_cost` metric while their units do not convert — a relative cost change is converted through the reference for ITS OWN unit, and a cost change that names no unit gets none. A failed call is a persisted failure with whatever usage it consumed. `status="contract_only"` with no provider; the service IS implemented, so `prediction_made` is what makes it `valid`.

## `orx compare-capability --predictions ID[,ID...] [--horizon-tasks N] [--allow-quality-loss]`

Compare frozen capability predictions and recommend one, or `defer` (read-only: no operation runs, no knowledge changes). ONE bounded rule: among candidates predicting a quantified resource saving with no quality degradation, recommend the largest **net saving over the declared window** — the CUMULATIVE saving minus that candidate's OWN ONE-TIME predicted maintenance cost **in the same unit**. The cost is paid once while the saving accrues per task, so a candidate that costs more than a single task saves is still recommended when the window pays it back. A candidate whose learning cost is quoted in another currency, whose saving does not pay back within the window, whose window was never declared, whose operation type has no execution path in this build, or whose savings cannot be compared across units is reported as `incomparable` rather than ranked. `defer` / `insufficient_evidence` are legitimate outcomes.

## `orx accept-capability --recommendation JSON [--prediction ID] [--verify JSON] [--note TEXT] [--force]`

**EXPLICITLY** accept a recommendation and run the real offline operation on the prediction's OWN frozen scope (the only offline entry that changes knowledge). The operation TYPE decides what runs: `induce`/`revise` go through the existing induction path, `retire` removes the named target entry, and any other type is REFUSED — a candidate is never executed as a different operation under its name. A scope is never widened by re-reading the current bank.

## `orx reject-capability --recommendation JSON [--prediction ID] [--reason TEXT]`

Explicitly decline or defer a recommendation: no operation runs and NO knowledge changes. The decision is recorded with its reason so the history is auditable.

## `orx bind-capability --prediction ID [--adoption-action ACTION_ID]`

**Stage 1 of the capability feedback.** Bind the REAL maintenance FACT: whether the operation happened, what knowledge actually changed (created / revised / **retired** entries — a retirement IS a knowledge change), the admission verdict of what it produced, the REAL cost, the operation's own end time, and whether the scope used matches the scope predicted. Idempotent by prediction. It CANNOT set `effect_verified` — a verified entry is a knowledge change, not evidence that future performance improved.

## `orx evaluate-capability --prediction ID [--tasks ID,...] [--paired JSON] [--allow-descriptive]`

**Stage 2 of the capability feedback.** Judge the prediction against REAL later-task results, or record a pre-arranged paired comparison. Only CLOSED episodes of tasks whose WORK really ran AFTER the operation — its real execution time, not its close-out timestamp, and under the knowledge entries the operation produced — fall inside the prediction's FROZEN targeting and are NOT part of its own experience scope participate. The sample is counted in **TASK-EPISODES**, not prediction records: ten predictions bound to one execution are ONE independent truth and cannot satisfy a ten-task horizon. The declared horizon must be reached or the result stays `pending`. A paired record is READ AND USED (its treated-minus-reference difference is the observed change) — merely existing is not attribution, and a pair taken on another metric/unit, or citing a task that never ran, is reported as unusable. `observed_improvement` is the only state that sets `effect_verified`; a pending horizon, a descriptive movement, insufficient evidence and a refutation are all first-class outcomes, and only a FINAL verdict short-circuits a repeat call.

## `orx capability-feedback [--prediction ID]`

The two-stage feedback state of every capability prediction (read-only): fact bound vs effect verified, plus whether a paired reference exists. No model call, no re-evaluation, no re-billing.

## `orx [--world-model URL::MODEL] plan-next --task t.json [--episode ep1] [--candidates specs.json] [--horizon 1|2] [--max-calls N] [--delta W] [--prediction-mode M] [--protocol legacy|strategy-outcome]`

**Bounded next-step planning.** Compare a small set of candidate actions by their PREDICTED consequences and get a suggested first step. The decision:

1. freezes ONE root snapshot for the whole comparison (all candidates see the same state);
2. predicts each root candidate's first-step consequences (default ≤3 candidates, ≤6 model calls total);
3. with `--horizon 2`, builds a HYPOTHETICAL successor state from each first prediction's `state_changes` and predicts the continuation FROM that successor (a genuine state-conditioned two-step rollout — never two independent root predictions);
4. scores each path as `U = alpha*Q_terminal − beta*C_path − gamma*R_terminal + delta*K` (terminal quality / incremental predicted cost on the common measured dimensions / terminal failure risk / knowledge value — longer paths never win by accumulating quality terms; step risks are never summed or multiplied);
5. suggests the FIRST step of the best path.

**The knowledge term `delta*K`.** `K` combines a framework-side structural need (how thin the support is, how much room the claim's interval still has above its honest floor, how much *independent* cross-task reuse the cell has) with the model's predicted direction. `--delta` defaults to **0.0**, so an unmodified call scores exactly as it did before the term existed.

**A missing `K` adds NOTHING** (`path.incomparable["knowledge"]` says so). This is the deliberate mirror of unknown risk: an unknown *downside* is charged in full, but an unknown *upside* is paid nothing — paying it would make the system prefer whichever action it understands least. `K = None` means "no justified value", never "worth nothing". `delta_knowledge` and `knowledge_detail` on each path show the term and why it came out that way, labelled `heuristic_uncalibrated` (it is a transparent heuristic, not a calibrated expected value).

- **Candidates**: `--candidates` (your own ActionSpec list) is REQUIRED. The framework does not generate a candidate menu — there is no directory to enumerate. You propose the methods you want compared (with their configuration), and the world model predicts their consequences; `orx recall` shows what memory already holds for this problem.
- **Bounds**: candidate count, horizon (1–2), `--max-calls`, and a wall-clock budget. Exhaustion truncates with an explicit reason — never a silent partial answer. The real planning spend (the model calls) is charged ONCE to the decision action and reported in `planning_cost` — sunk, never part of any path's score.
- **A suggestion is not a selection**: `plan-next` never writes `X.selected_plan` and never executes. Only `choose-next` does.
- **Unknowns**: missing quality/cost/risk predictions are reported per path under `incomparable` (unknown never auto-wins); a second step the first prediction cannot support (no incumbent) is truncated and marked `conditional_unsupported`; with an undeclared or partially-unknown budget, `budget_confirmation` is `unknown`/`unconfirmed` — never claimed "within budget". An already-exceeded real budget stops planning before any model call (`status=fallback`).
- **Requirements**: a configured `--world-model` provider (mandatory). Without one every path is `not_configured` and no suggestion is made. Planning supports `execute_strategy` candidates; other action types are reported as not plannable. `plan-next` is an OPTIONAL planning aid — it does not affect the mandatory predict/bind duty.
- **`--protocol strategy-outcome`**: the same decision loop under the wm-so/1 protocol — ONE frozen context for the whole comparison, one strategy-outcome prediction per candidate (benefit/cost/risk/uncertainty), a conservative comparison (`U = alpha*G - beta*C - gamma*R`; unknown cost charged the peak share, unknown risk the full weight, unknown benefit nothing; the knowledge term is OFF), and a suggestion you accept through the SAME `choose-next`. Horizon is FIXED at 1 for this protocol (one macro comparison, then re-planning from the real observation); `--horizon 2` with it is refused. When no candidate carries a usable prediction the plan reports `no_valid_predictions` and suggests nothing — fall back to `recall`/Selector or choose yourself. Full spec: [strategy_outcome.md](strategy_outcome.md).

## `orx choose-next --decision ACTION_ID [--chosen spec.json | --rejected] [--note "..."]`

Record YOUR explicit choice after a plan: accept the suggestion, pick another candidate (a deviation, recorded with its reason), or reject all. Only this call writes `X.selected_plan`; the choice itself produces no execution quality. Then execute the step with `orx execute`, record the result with `orx record`, and bind the executed step's prediction with `orx bind-outcome`. Re-plan from the new real state afterwards — the old plan stays as the suggestion of its time.

## `orx [--world-model URL::MODEL] assess-induction [--bundle bundle.json | --candidates-only] [--workload forecast.json]`

**Offline maintenance assessment.** Scans the banks for induction candidate bundles (frozen evidence: exact execution IDs, tasks, statistics, trigger reasons — new claims need ≥2 executions from ≥2 tasks; revisions reference the existing entry and its state at bundle time), then asks the world model to predict the INDUCTION action's consequences: candidate formation probability, expected reuse benefit, generalization risk, and the resulting claim's quality/cost. The recommendation (`induce_new` / `revise` / `defer` / `insufficient_evidence`) comes with a value decomposition (`net_value = α·benefit − γ·risk`, workload forecasts scale the benefit term) and the real assessment cost, charged once to a maintenance-scope action (`__maintenance__`) — never to a business task's budget. An assessment NEVER induces: it only records. Accepting it (`accept_induction` API) runs the existing `induce` strictly on the bundle's execution IDs (no silent scope widening) with your admission check; rejecting it records the rejection and touches nothing. The model's confidence never bypasses evidence gates, budgets, or verification. `induction_assessment="shadow"` evaluates and records but withholds advice; `"disabled"` returns explicitly without model calls.

## `orx gc [--mode compact|purge] [--dry-run]`

Disposes only of the derived layer. `compact` is **deferred**: lossy evidence compaction is paused until the summary consumption contract exists (statistics and induction currently ignore `source="compacted"` rows, so summarizing raw facts would bias conditional statistics — e.g. 90 successes + 10 failures would read as 100% failure rate). The command still runs and reports the deferral; raw facts are untouched. `purge` lists retirement candidates (suspect/dormant entries) without retiring them.

## `orx retire --entry ID --reason "..."`

Your explicit, irreversible confirmation: moves an entry to the cold archive. Its vector leaves the retrieval index with it, so a later `recall` can never surface the retired advice.

## `orx exclude-execution --execution EXECUTION_ID --reason "..." [--superseded-by EXECUTION_ID]`

Withdraw a **wrong execution fact** from the evidence set. The Evidence Bank is append-only, so a bad observation is never deleted — but it must stop counting. The row is preserved for audit, its `source` becomes `excluded`, and the reason is recorded on the fact under `execution_features.correction`. Every statistics / induction / trigger / retrieval path requires `source == "executed"`, so the fact drops out of **all** of them at once; its vector is removed from the execution index immediately. `--superseded-by` names the corrected re-run that replaces it (a link, never an inference — the correction is always your explicit statement). Derived layers pick the change up at the next `orx induce`. This is the ONLY way to un-count a fact: `record` never rewrites, and the fact's `execution_id` stays visible in `inspect --bank experience`.

## `orx restore-execution --execution EXECUTION_ID --reason "..."`

Reverse an exclusion — a second explicit statement, because an exclusion can itself be wrong. The fact counts as evidence again (source back to `executed`) and the correction history keeps both decisions. Re-indexing is deliberately NOT automatic: run `orx rebuild-index --layer execution` to re-embed it, then `orx induce` to refresh the derived layers.

## `orx rebuild-index [--layer both|execution|strategic] [--dry-run]`

Explicit retrieval-index maintenance — the only command that embeds in bulk and the only place index vectors are (re)created in volume. It is needed in exactly three situations:

1. **First build.** Until an index exists, `recall` reports `degraded` ("index missing; run `orx rebuild-index`") and uses the structural channel only.
2. **Embedding-model change.** An index built by another `model_id` is refused WHOLESALE — vectors from two models are not comparable, so it is never partially reused.
3. **Settling a deferred sync.** If an embedding call failed during `record` / `induce`, that memory is unindexed (`result.index_sync.state = "deferred"`) until a rebuild picks it up.

Not a routine path: `record` refreshes the execution item, `induce` refreshes the knowledge items, and `retire` drops the retired vector.

`--dry-run` counts what would be indexed and **writes nothing** — no embedding call, no index file, not even the index directory. Content that cannot be indexed is reported as `unindexable` (a record whose text was never captured is not indexed). The index is derived data: rebuilding it never changes a fact or an entry.

Result: `{dry_run, layers: {"execution_evidence"|"strategic_knowledge": {items, model_id, dimension, unindexable}}, backend}`. Without a configured backend and without `--dry-run`, the command fails with exit code 2 and an explicit message rather than silently doing nothing.

## `orx contract [--kind strategy_outcome|capability_evolution] [--payload JSON]`

**Unified world-model contract: build one, or read a stored one. No model call is ever made.**

`--kind` BUILDS a versioned, serializable contract object:

```bash
# a candidate strategy's predicted consequences (metric + baseline required
# whenever a benefit value is given)
orx contract --kind strategy_outcome --task t.json \
  --spec '{"action_type":"execute_strategy","strategy_id":"S01","scope":"attempt"}' \
  --benefit '{"kind":"solution_quality","metric":"normalized_objective_gap",
              "unit":"1-gap","value":0.8,
              "baseline":{"kind":"conditional_stats","value":0.7}}' \
  --cost '{"expected":{"llm_tokens":1200},"expected_measured":["llm_tokens"]}' \
  --risk '{"events":[{"event":"task_failure","probability":0.2}]}'

# a LEGACY ActionSpec is accepted as --spec too: its execution params and
# budget_hint are preserved verbatim, and an unmappable legacy scope such as
# "task" is REFUSED (exit 2) instead of being silently shrunk to one attempt
orx contract --kind strategy_outcome --task t.json \
  --spec '{"action_type":"execute_strategy","strategy_id":"S01",
           "measurement_scope":"attempt",
           "params":{"time_limit":60,"mip_gap":0.01,"seed":42}}'

# a candidate learning operation's predicted capability consequences
orx contract --kind capability_evolution \
  --operation '{"operation_type":"induce","strategy_id":"S01"}' \
  --horizon "next 10 matching routing tasks" --horizon-tasks 10 \
  --verification '{"condition":"interval holds on 5 unseen tasks","evaluable":true}'
```

`--payload` READS a stored prediction payload and reports its contract version:

```bash
orx contract --payload prediction.json
# -> {"contract_version":"wm-contract/1", "legacy":false, "supported":true, "contract":{...}}
# -> {"contract_version":"legacy/unversioned", "legacy":true, "legacy_view":{...}}
# -> exit 2 on an unsupported version: the payload is NEVER guessed at
```

**Read `status` together with three SEPARATE facts.** A built contract is `status="contract_only"` unless a prediction was really produced and validated:

| Field | Meaning |
|---|---|
| `provider_configured` | a provider is attached to this instance |
| `service_available` | this build implements the service for this kind **and** a provider is configured |
| `prediction_made` | a prediction was really produced and validated |

Only `prediction_made` makes `status="valid"`. Building a contract returns `contract_only` **even with a provider configured** — zero model calls with empty benefit/cost/risk is not a forecast. The capability-evolution kind has a contract AND a service (`wm-ce/1`), so a configured provider makes it `service_available`; `prediction_made` still requires the real call. `unsupported`/`invalid` mean nothing was predicted. The capability contract carries no composite H score, and a legacy payload is read through a view that derives **no** capability increment, risk severity or measurement.

Full contract definition, the attempt-vs-strategy-window scope rule, the `H = F(M, W_OR, Pi, R, T)` sources, and the migration table: [world_model_contract.md](world_model_contract.md).

## `orx doctor`

Self-check: solver availability (7 adapters probed), memory sizes, staged-but-unrecorded executions (audit your pending area), home path, and **retrieval-index health** (`result.retrieval_index`).

Index health per layer reports the index item count vs the number of CURRENT documents, plus `stale` (indexed with an out-of-date document digest), `missing` (a current document with no item), and `orphaned` (an item whose record is gone). `configured: false` means no embedding backend is present, so recall will use the structural channel only. **Read-only**: `doctor` builds nothing and never triggers an index write — a missing index is reported with the `orx rebuild-index` hint instead.
