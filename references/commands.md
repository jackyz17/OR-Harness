# Command reference

Every command prints exactly one JSON line to stdout: `{"result": {...}, "summary": "agent-readable 2-4 sentences"}`. Exit codes: `0` success, `2` usage/precondition error, `1` crash. All commands accept `--home DIR` (default `$OR_HARNESS_HOME`, else `./or_harness_home`).

## `orx profile --task t.json [--cir cir.json]`

The single analysis entry: CIR validation + modeling guidance + problem profile + derivation report, in one call. It serves two purposes: it helps you understand the problem structure, and it produces the structural key that retrieves comparable evidence at `recall` time. A task **without** a `model` field is a normal state — strategy selection needs the task text, the CIR, and the profile. The `model` is written only AFTER the strategy is chosen, so it is verified and reported as a DIAGNOSTIC (`derivation.model_coupling`); it never moves the structural key. There is deliberately NO solve-script parameter: a script is a post-strategy artifact, and no identity-bearing path accepts one.

**CIR side** (`result.coupling`): when the task carries a `coupling` field (or `--cir` is given), the CIR (Coupling-Aware Intermediate Representation — an explicit, inspectable set of entities, decisions, constraints, and relations) is validated and returns `modeling_guidance` with concrete coupling implications for the model you will write. Without a `coupling` field, `coupling.cir=null` with a prompt message (not an error).

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

## `orx recall --task t.json [--top 3] [--exclude S04 S06] [--memory-mode M] [--include-unverified]`

Recalls accumulated experience through **two independent channels**. They answer different questions and are never blended into one number.

**Structural channel — "what may I REUSE?"** (`result.recommendations[]`). Score = `α·Q̂ − β·C_scalar − γ·R̂` (weights configurable via `--alpha/--beta/--gamma/--cost-weights`). Evidence precedence per strategy: **published** Strategic Knowledge entry → conditional statistics → **no evidence** (`evidence="no_memory"`, `score=-inf`, `confidence=0`). An entry that is not published (an unverified candidate, or one whose admission check was `refuted` / `insufficient_evidence`) is skipped, so recall falls back to the statistics — which is what "we have a candidate but no knowledge yet" should look like.

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

Field discipline: `observed_*` (what happened) and `expected_*` (what the knowledge claims) are separate fields — a promise is never read as a measurement. `structural_match` for executions compares `group_key` values; for entries it reports the applicability verdict, where `conflicts` names the contradiction and `unknown` means a value needed to decide is missing. `same_cell` and `applies` are independent judgments with different bases. Dormant, retired and (by default) unpublished entries are filtered out **before** the `top_k` cut, so an ineligible item never takes a slot a usable memory could have filled. `stale_indexed` counts items whose document changed under the index (excluded — the stored vector describes text that no longer exists).

**Degradation is explicit.** When the text channel cannot run, `result.degraded = {"path": "profile_only", "reason": ...}` explains it, and the structural result is returned intact:

| Situation | `degraded.reason` |
|---|---|
| No embedding backend configured (no env, no injection) | `no embedding backend configured` |
| Task JSON has no textual field | `no task text in task JSON` |
| No index file yet | `index missing; run orx rebuild-index` |
| Index built by a different embedding model | `embedding model changed (was X, now Y)` |
| Embedding call failed | `embedding backend error: ...` |

The retrieval document is built from `text`, `description`, `objective`, `requirements`, `business_rules`, `constraints` and `spec` (in that order), so a task that states its problem in a plain `text` field is indexed the same way as one using `description`. Only a task with NONE of these keys reports `no task text`.

`vector_recall.unindexed` counts current memories with no vector — legacy records whose text was never captured, or a write whose index sync was deferred — and states where to find them (`orx inspect --bank experience|strategic`, `--bank texts`, and the profile channel).

`--include-unverified` is the offline/inspection view: unpublished candidates appear in BOTH channels (the structural path returns them with a warning; the text path stops filtering by admission state).

When no experience exists for any strategy, all candidates return with `evidence="no_memory"` — the catalog still provides the strategy vocabulary (applicability, actions, fallback, solver family) but makes no quality/cost/risk claims. Pick based on structural fit and your own judgment.

Result: `result.recommendations[]`, each `{strategy_id, name, score, expected{quality, cost, failure_prob}, evidence, evidence_refs, confidence, cross_family, risk_warnings, basis, cost_known_dims, cost_basis_dims}` plus `result.available_solver_families` (family → usable solver names; pick the concrete solver yourself) and `result.solver_advisories` (solvers with environment-class failures in this memory — e.g. a subprocess-based solver the sandbox rejected before). Note: when `evidence="conditional_stats"`, the `expected` fields report OBSERVED means (a recount from the Evidence Bank), not a knowledge commitment — no interval, no calibration track, no lifecycle.

`--memory-mode`: `none` (no memory consulted; all candidates return `no_memory`) | `cases` (statistics, no cost weighting) | `strategic` (entries + statistics, no cost weighting) | `cost-aware` (adds cost scalarization).

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

You write `solve.py` following the strategy's actions (the framework never generates code). It runs in a sandbox: no network/shell/pathlib, `open()` only for a literal relative `result.json`, POSIX rlimits + wall-clock timeout. Your script must write `result.json` with at least `status` (optimal|feasible|infeasible|unbounded|timeout|error), `objective_value`, `objective_bound`, `mip_gap`, `runtime_seconds`.

The profile used in `execute` is the same frozen pre-strategy signature from `recall` — derived from `coupling` (CIR) / `spec` / `annotations` / `model` only.

Every execution is automatically staged in a pending area (successes and failures alike) — staging is a safety net, not recording. Result: `result.execution` = a full ExecutionRecord (id, quality check, CostVector with `llm_tokens=0` UNMEASURED — that dimension is yours to backfill, `cost_measured` mask, `solver_runtime_provenance`, `execution_features` with solver diagnostics if the solver reported any, `cir_snapshot` = the CIR that was actually solved when the task carries a `coupling` field, and `task_text_digest` = the task-text VERSION this execution was produced under). **Nothing is recorded yet.**

`execute` also persists the task text (`task_texts`, keyed by `(task_id, text_digest)`) — that is where the retrieval document comes from. Solving the SAME `task_id` with different content creates separate versions, so each execution stays linked to the text actually in force, and the two can never be confused later.

## `orx predict --task t.json --strategy S`

Pre-execution cost expectation for one (task, strategy), with explicit provenance: `source=entry` (a matching Strategic Knowledge entry), `stats` (conditional statistics over attempt-scope evidence — a recount, with `support_n` and per-dimension `support_per_dim`), or `unknown` (no usable evidence: no data, task-scope data only, or clearly different scale — expected cost is `null`, NEVER a default zero presented as cheap). Pass the printed snapshot to `orx record --prediction` so feedback compares against the prediction actually used.

## `orx record --execution <json|path> | --from-staged <id> [--override llm_tokens=1840,tool_calls=9] [--override-mode replace|increment] [--prediction <json>] [--retain-reason contrast] | --discard-staged <id>`

Appends the fact to the Execution Evidence Bank, then runs the automatic chain: cost backfill (replace by default — idempotent, never double-counts) → quality checks against matching entries, frozen ONTO THE FACT (`execution_features.quality_feedback`: the interval in force, the observed quality, hit/miss — only the strategy that actually ran is checked, attempt scope only) → cost feedback against the record's frozen pre-execution prediction snapshot (same strategy, same scope, both sides measured; written into `execution_features.cost_feedback`) → C1–C6 induction-hint checks (C4 detects cross-execution recovery chains automatically: a failed attempt under one solver followed by success under another).

**Recording never changes knowledge.** No entry is promoted, demoted, or woken here — the frozen checks are replayed by the next `orx induce`.

`--retain-reason <free text>` explicitly marks this episode as representative evidence, reserved for future compaction policies. When omitted, any mark the record already carries is preserved — no automatic marking is performed.

**Task text and index sync (best effort).** `record` can be reached WITHOUT `execute`, so it resolves the text link itself: the digest the caller supplied, else the task's most recent real belief snapshot (`hypothetical=False`, whose frozen `task_payload` is rebuilt through the same reader used at capture time). If neither exists, `task_text_digest` stays `None` and the record is simply not vector-indexed — the text is **never invented**, and the memory keeps its full profile-based visibility. After the fact is durable, the execution's index item is refreshed; an embedding failure does **not** roll anything back and is reported as `result.index_sync = {state: "deferred", reason: ...}` (recover later with `orx rebuild-index`). `state` is `synced` / `deferred` / `skipped` (`skipped` = no backend configured, or the text is unavailable).

Result: `result.{execution_id, recorded, prediction_checks[], cost_feedback?, induction_hints[]}` and, when same-task executions are staged but unrecorded, `result.unrecorded_staged_executions[]` — backfill those with `--from-staged` (records the original payload verbatim; never re-type an execution JSON by hand). Always backfill `llm_tokens` here — it is invisible to the sandbox. A task's full cost is the sum of its recorded attempt-scope records (`api.task_cost_summary`): every attempt charged to the strategy that actually ran it; retries = sum of per-attempt NEW retries; end-to-end latency is unknown unless you supply explicit task timing (never inferred by max or sum).

## `orx induce [--strategy S | --all] [--family F] [--cell GROUP_KEY] [--rebuild] [--dry-run] [--force] [--note TEXT] [--verify JSON]`

Consolidates facts into Strategic Knowledge (your explicit call — hints never auto-induce). This is the ONLY place knowledge changes: it (i) builds or refreshes the claim of each (family, structural cell, strategy) evidence set — its applicability is read off the supporting records (the cell its evidence occupies, never a cross-sample span that could stretch across incomparable regions); and (ii) replays the frozen checks recorded on the facts, reporting what it did under `result.revisions` (promotion at ≥5 checks / ≥70% hits **with a verified claim**, demotion at 3 consecutive misses, dormancy wakeup). New entries are born `candidate` with honest intervals (width floored by sample size; n=2 cannot claim [0.95, 1.0]). One evidence set owns exactly one claim: re-inducing refreshes it rather than restating it (dormant entries included, so a woken claim keeps its id). `--rebuild` re-induces the whole Strategic Knowledge Bank from currently retained evidence (cold archive preserved); the result may differ from the previous bank — exact reconstruction is not a requirement. New entries inherit `strategy_type`, `actions` and `fallback_strategy_id` from the catalog vocabulary (harness-supplied values are never overwritten). `--dry-run` never writes — including under `--force` (a rehearsal does not lift a cold-archive veto).

**Independent evidence gate (creation only).** Creating a claim takes at least 2 supporting executions from ≥2 distinct `task_id`s: repeating one task is repetition, not reproduction. Every count comes from the same facts — executed, attempt-scope, in the target's structural cell — so a task-scope total can never stand in for an independent attempt. When only one task is behind the evidence the call creates nothing and returns `verification {tasks, required_tasks}` plus a `skipped` reason — the evidence is NOT discarded (recall still answers from it as `conditional_stats`), and refreshing an entry that already exists is never blocked, no matter how repetitive the new evidence is.

**`--verify JSON` publishes the candidate.** Admission and creation are separate: the gate above decides whether evidence may become a *candidate*; this decides whether the candidate's claim may become *published knowledge*. See [induction.md](induction.md#admission-verification-offline-what-makes-a-claim-published) for the payload and the three checks (`rule` / `repair` / `cost_saving`). The framework computes the verdict from the executions you supply; without it the entry is `unverified` and **is not published** — recall answers from `conditional_stats` and `predict_cost` will not quote it.

**`--family` / `--cell` scope the check to ONE structural unit.** A verification payload is written for a specific claim, so it must apply to a specific target set — one payload applied to every cell of a strategy would cross-contaminate families (an allocation check overwriting a scheduling verdict). `--family F` restricts induction to that family; `--cell GROUP_KEY` restricts it to one structural cell (the full `group_key` token, e.g. `family=routing|rc[..]|tc[..]|rx[..]`, compared against each record's DERIVED key so a stale index column never mis-selects). Either flag implies the `--all` scope — naming a unit is itself an explicit selection — and composes with `--strategy`.

What the verdict requires, concretely: a declared criterion the framework can evaluate itself (`reference_status`, `reference_objective`, `semantic_probe`) or a genuinely independent comparison — **feasibility alone is a precondition, not a check**, and a bare `semantic_ok` boolean is recorded as `agent-declared` but cannot carry a verdict. Every execution you pass is evaluated (a counterexample anywhere in the batch refutes, whatever the order), the evidence must be for THIS strategy and family, and a comparison must be like-for-like (same task, same measurement scope for costs; a repeated execution id is not an independent comparison). Failed executions yield `insufficient_evidence` (not `refuted`): "we could not check it" is not "we checked it and it failed".

Each element of `result.results` reports `created` (entry id) / `updated` (entry id) / `skipped` (human-readable reason: fewer than 2 executions, needs independent evidence, cold-archive veto, restatement, or `recorded as an unverified candidate`), plus `verification` when a check ran — read the reason, it tells you what the memory is still missing.

`--note "TEXT"` (optional, repeatable, you phrase it): free-text applicability notes attached to the entries this call creates or refreshes. They are kept for the reader and shown by `inspect`; they never enter scoring — the framework does not pretend to verify a sentence. Because these notes are part of an entry's retrieval document, the knowledge index items are refreshed after this call (`result.index_sync`, best effort — same `synced`/`deferred`/`skipped` contract as `record`). `--dry-run` writes nothing, index included.

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
- **Configuration boundary**: `--world-model BASE_URL::MODEL` (OpenAI-compatible endpoint; API key from `$OR_WM_API_KEY`). Credentials never persist. Without the flag, the command returns an explicit `not_configured` error — and NO other command ever invokes a model.
- **Shadow discipline**: the prediction changes NOTHING. `recall`, `predict`, `execute`, `record` behave identically whether or not you predict. You remain the decision-maker.
- **Calibration duty**: with a world model configured, one predict–bind pair per executed action is the required loop — executed prediction–comparison pairs are the only source of calibration evidence, and skipping them is legitimate only when no provider is configured (`not_configured`). After a `plan-next` selection, bind the selected path's prediction instead of predicting again.
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

**Episode close-out.** Closes ONE episode: evaluates its bound strategy-outcome predictions against their real outcomes (field by field: benefit error under the prediction's own declared metric/baseline, per-dimension cost error where both sides measured the same scope, Brier scores for labelled risk events, interval coverage) and publishes the experience calibration summary that later episodes' prediction contexts read. Reads what was recorded — no solver run, no model call, no induction. Unfinished actions are reported (their predictions stay pending, never fabricated into endings); a failed/aborted/budget-exhausted episode closes honestly under its own terminal state. Idempotent: re-closing returns the stored record and counts nothing twice. Full spec: [episode_closeout.md](episode_closeout.md).

## `orx calibration [--min-samples N]`

Read the published strategy-outcome experience-calibration summary (read-only). Closed episodes only — an active episode never reads its own not-yet-closed feedback. Groups below the sample minimum report `insufficient_evidence` with `reliability: null`; no figure is invented. This is the OR strategy-outcome reliability, kept separate from the legacy knowledge-prediction reliability, and a measured record — not a promise that future predictions improve.

## `orx evaluations [--task ID] [--episode ep1] [--evaluation EVALUATION_ID]`

List (or read one) stored post-hoc evaluations of strategy-outcome predictions (read-only). Excluded evaluations are neither hits nor misses; pending ones wait for their scope to end.

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

- **Candidates**: `--candidates` (your own ActionSpec list, recommended when you have domain hypotheses), or the catalog vocabulary filtered by applicability and available solvers. Without memory, candidates carry no fabricated performance claims — consequences come from the world model.
- **Bounds**: candidate count, horizon (1–2), `--max-calls`, and a wall-clock budget. Exhaustion truncates with an explicit reason — never a silent partial answer. The real planning spend (the model calls) is charged ONCE to the decision action and reported in `planning_cost` — sunk, never part of any path's score.
- **A suggestion is not a selection**: `plan-next` never writes `X.selected_plan` and never executes. Only `choose-next` does.
- **Unknowns**: missing quality/cost/risk predictions are reported per path under `incomparable` (unknown never auto-wins); a second step the first prediction cannot support (no incumbent) is truncated and marked `conditional_unsupported`; with an undeclared or partially-unknown budget, `budget_confirmation` is `unknown`/`unconfirmed` — never claimed "within budget". An already-exceeded real budget stops planning before any model call (`status=fallback`).
- **Requirements**: a configured `--world-model` provider. Without one every path is `not_configured` and no suggestion is made. Planning supports `execute_strategy` candidates; other action types are reported as not plannable.
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
