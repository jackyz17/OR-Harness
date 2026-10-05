# Command reference

Read this page when you are about to call a command and need its flags, its return states, or what to do when it fails. The workflow itself is in [../SKILL.md](../SKILL.md); this page only documents the calls.

Every command prints exactly one JSON line to stdout: `{"result": {...}, "summary": "agent-readable 2-4 sentences"}`. Exit codes: `0` success, `2` usage/precondition error, `1` crash. All commands accept `--home DIR` (default `$OR_HARNESS_HOME`, else `./or_harness_home`).

## Index by work stage

**Start with `profile`, then `recall`** — see stages 2 onwards in order. The first stage below is optional setup, not a prerequisite sequence.

| Stage | Commands |
|---|---|
| (optional) Setup and diagnostics | `doctor`, `contract` |
| Understand the task | `profile`, `recall`, `context` (optional), `predict-cost` |
| Predict and choose a strategy | `predict-strategy`, `plan-next`, `choose-next` |
| Execute and verify | `execute`, `check-task`, `bind-strategy` |
| Record facts and cost | `record`, `amend-cost`, `snapshot`, `action`, `budget` |
| Close out and calibrate | `close-episode`, `calibration`, `archive-calibration`, `inspect` |
| Offline maintenance | `review-material`, `induction-candidates`, `induction-material`, `predict-capability`, `compare-capability`, `accept-capability`, `reject-capability`, `bind-capability`, `evaluate-capability`, `induce` |
| Correct the record | `exclude-execution`, `restore-execution`, `retire` |
| Retrieval index | `rebuild-index` |

If you already know the problem, jump straight there:

| Your problem | Read |
|---|---|
| A command failed with exit 2, or a script was rejected | the command's own entry, plus `execute` for sandbox policy |
| `recall` came back empty or `degraded` | the `recall` entry, then [../SKILL.md](../SKILL.md) step 2 |
| A prediction looks wrong for the candidate I chose | [strategy_outcome.md](strategy_outcome.md) for the protocol, [prediction_context.md](prediction_context.md) for what it was conditioned on |
| `check-task` failed or returned `insufficient` | the `check-task` entry |
| The calibration numbers look surprising | [episode_closeout.md](episode_closeout.md) |
| A candidate was skipped at induction | [induction.md](induction.md)

## Passing ids between commands

| You need | Read it from | Hand it to |
|---|---|---|
| `task_id` / `--episode` | **yours to choose.** One logical problem keeps ONE `task_id` however often you solve it; an episode label groups the attempts of one solving effort | every command that takes `--task` / `--episode` |
| Planning decision action id | `plan-next` → `result.decision_action_id` | `choose-next --decision` |
| The chosen candidate's prediction id | `plan-next` → `result.plan.candidates[].prediction_id`, or `predict-strategy` → `result.prediction_id` | `choose-next --prediction`, `execute --prediction`, `bind-strategy --prediction` |
| Execution id | `execute` → `result.execution_id` | `check-task`, `record --from-staged`, `exclude-execution` |
| Action id | `execute` → `result.action_id` | `bind-strategy --action`, `action --amend-cost` |
| Frozen evidence package | `induction-candidates` → `result.bundles[]` | `predict-capability --bundle` |
| Maintenance prediction id | `predict-capability` → `result.prediction_id` | `compare-capability --predictions`, `accept-capability --prediction`, `evaluate-capability --prediction` |
| Recommendation | `compare-capability` → `result.recommendation` | `accept-capability --recommendation`, `reject-capability --recommendation` |

Every command entry below follows the same shape: purpose, minimal call, key inputs, key outputs, what to do next, and the failure modes that change your next step.

## Stage 1 (optional) — Setup and diagnostics

### `orx doctor`

Self-check: solver availability (7 adapters probed), memory sizes, staged-but-unrecorded executions (audit your pending area), home path, and **retrieval-index health** (`result.retrieval_index`).

Index health per layer reports the index item count vs the number of CURRENT documents, plus `stale` (indexed with an out-of-date document digest), `missing` (a current document with no item), and `orphaned` (an item whose record is gone). `configured: false` means no embedding backend is present, so recall will use the structural channel only. **Read-only**: `doctor` builds nothing and never triggers an index write — a missing index is reported with the `orx rebuild-index` hint instead.

**Next.** Run `rebuild-index` if `retrieval_index.configured` is true and items are `missing`/`stale`; backfill anything listed as staged-but-unrecorded with `record --from-staged`.

### `orx contract [--kind strategy_outcome|capability_evolution] [--payload JSON]`

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

**Next.** Read `status` together with `provider_configured`, `service_available` and `prediction_made`: only `prediction_made` makes it a forecast, and a legacy payload is read through `legacy_view`, which derives no capability increment.

## Stage 2 — Understand the task

### `orx profile --task t.json [--cir cir.json] [--allow-empty-cir]`

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

**Next.** Pass the same task to `recall`, then to `predict-strategy` or `plan-next`. Failures: a malformed CIR exits 2 with `result.error.cause`/`hint` and is never read as an uncoupled problem; `coupling_warnings` means your declared value lost to the derived one for grouping.

### `orx recall --task t.json [--top 3] [--exclude S04 S06] [--candidate ID ...] [--memory-mode M] [--include-unverified]`

Recalls accumulated experience through **two independent channels**. They answer different questions and are never blended into one number — and an empty result on one never erases the other.

**There is no candidate menu.** Nothing ships a list of method names, descriptions or applicability rules. A strategy appears in `recommendations` when the memory really holds something about it in this structural cell — a recorded execution (`conditional_stats`) or an admission-verified claim (`strategic_entry`). With nothing in either, `recommendations` is **empty** and `recommendations_basis.reason` says so; there is no `no_memory` row, no `-inf` score, no zero-quality placeholder.

**Structural channel — "what may I REUSE?"** (`result.recommendations[]`). **Recall DISCOVERS; it does not DECIDE.** Rows are in RELEVANCE order (a published entry before a bare recount; then strategy id) — recall computes NO utility score and applies NO utility ranking. Each row REPORTS the backing memory's `expected{quality, cost, failure_prob}` for you to weigh after the world model predicts a candidate's consequences. `--top` is a MATERIAL BUDGET (a READ cap), never a decision: the cap and any omission are reported under `result.recall_budget` (`n_matched`, `returned`, `omitted`). Evidence precedence per strategy: **published** Strategic Knowledge entry → conditional statistics → *nothing* (the strategy is simply absent). An entry that is not published (an unverified candidate, or one whose admission check was `refuted` / `insufficient_evidence`) is skipped, so recall falls back to the statistics — which is what "we have a candidate but no knowledge yet" should look like. Every recommendation carries `strategy_id` and nothing else about the method's identity: no `name`, no `description`. What the backing entry itself records travels in `recommendation.knowledge` (`entry_id`, `strategy_type`, `actions`, `fallback_strategy_id`, `applicability`, `support_n`, `verification_state`) — empty lists mean the memory does not record it.

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

**Claim channel — "what does the memory SAY about this condition?"** A published knowledge claim reaches you through `result.recommendations[]` like any admitted entry: its `knowledge` block carries the stated claim, its verification state and scope, and `newer_evidence_since_verification` (matching executions recorded after the verdict — a visibility annotation, not a lifecycle state).

`result.held_claims[]` carries the claims that would NOT survive that filter — a claim-only entry whose `strategy_id` is a free-form subject, or an entry not yet verified — each with its `verification_state` and `newer_evidence_since_verification`. Published claims may be used as strategic grounds; `unverified` / `refuted` / stale ones appear only here (and only with `--include-unverified`), clearly labelled, and are never dressed up as available knowledge.

**Text channel — "what should I LOOK AT?"** (`result.vector_recall`). The task text is embedded and compared with every indexed memory, **with no structural pre-filter**, so a nearly identical problem from a different structural cell still surfaces. `similarity` is the raw text cosine: it is **never** a quality, cost, or risk estimate and never enters a score.

**An execution hit is usable experience on its own** — it does not have to be promoted to strategic knowledge before you may read and adapt it. The hit therefore carries the METHOD, not just an id and a mean:

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
    "method": {                          // HOW the work was done (never rewritten)
      "planned": {"name": "...", "steps": ["..."]},   // the DECLARED intent
      "actual":  {"name": "...", "steps": ["..."]}|null,  // method_performed receipt
      "basis": "performed|planned_only|none"          // how strong the summary is
    },
    "method_basis": "performed|planned_only|none",     // = method.basis (flat)
    "solution_variables": {"x1": 10, ...}|null,        // what was PRODUCED, if reported
    "failure_summary": [{"error_class": "model",       // one line per failure
                         "error": "AttributeError: ...",
                         "recovery_action": null}],
    "reuse_trace": {"adapted_from": ["ex_..."],        // the agent's own citations
                    "adaptation": "..."}|null,          // (READ, not "reuse worked")
    "inspect_hint": "orx inspect --bank experience --id ex_...",  // fetch the full record
    "profile_cell": "family=vrp|rc[0.25,0.50]|...",
    "structural_match": "same_cell|different_cell|unknown"
  }],
  "suggested_candidates": [{              // only when strategic_knowledge is EMPTY
    "method_name": "...", "steps": ["..."],
    "from_execution_id": "ex_...", "from_task_id": "t1", "strategy_id": "S04",
    "similarity": 0.87, "basis": "performed|planned_only",
    "task_check": "passed|null", "structural_match": "same_cell",
    "note": "…presented for consideration, not ranked or vouched for…"
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

**The two channels are used in order.** `strategic_knowledge` / an admitted-ENTRY `recommendations[]` (`evidence: "strategic_entry"`) is checked FIRST. A `conditional_stats` recommendation is a recount over the evidence bank, not admitted knowledge — when no admitted entry applies and the text channel found comparable executions, the EXECUTION hits are the main source and `recall` fills `result.evidence_candidates[]` (a copy of `suggested_candidates`): the methods those past executions actually recorded, each stamped with its `from_execution_id`, `basis` and `task_check`. They are **suggestions, not a ranked menu and not a voucher**: the caller still predicts, executes and `check-task`s whichever it picks. Reading them:

- `method.planned` is intent; `method.actual` (with `basis: "performed"`) is what the run reports it did. `basis: "planned_only"` is real but unobserved; `basis: "none"` means the record reports no method at all (UNKNOWN — never guessed).
- `task_check: "failed"` / `failures` present → a FAILED/REPAIR case: its cost and failure are real, but it must never be read as a working strategy. `task_check: null` → never checked, NOT a pass.
- A hit is a hint to ADAPT AND PROVE, not to copy: check applicability, adapt, then predict/execute/check it as your own candidate. `inspect_hint` fetches the full record on demand.
- When you adapt a past case, cite it with `execute --adapted-from ex_... [--adaptation "…"]`; the citation is recorded on the fact as `reuse_trace` and means the case was READ — never that reuse succeeded (see the `execute` section).

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

When no experience exists for any strategy, `recommendations` is EMPTY and `recommendations_basis` says why. That is the normal cold-start state: propose the methods you want to try and they will be predicted, executed and recorded. `recall` is a memory report, not a menu — `predict-cost`/`execute` accept any id you name.

Result: `result.recommendations[]`, each `{strategy_id, expected{quality, cost, failure_prob}, evidence, evidence_refs, confidence, cross_family, risk_warnings, basis, cost_known_dims, cost_basis_dims, knowledge}` (NO `score`), plus `result.recall_budget` plus `result.available_solver_families` (family → usable solver names; pick the concrete solver yourself) and `result.solver_advisories` (solvers with environment-class failures in this memory — e.g. a subprocess-based solver the sandbox rejected before). Note: when `evidence="conditional_stats"`, the `expected` fields report OBSERVED means (a recount from the Evidence Bank), not a knowledge commitment — no interval, no calibration track, no lifecycle.

`--memory-mode`: `none` (memory deliberately NOT consulted: `recommendations` is empty and the basis says it was a choice) | `cases` (statistics only) | `strategic` (entries + statistics) | `cost-aware` (like strategic; the mode is retained so the experiments runner can vary cost sensitivity — recall itself makes no cost ranking).

**Read-only.** `recall` writes nothing — the query text is embedded in memory only, no text row is created, no index item is touched, no migration runs. Text is persisted on the WRITE paths (`execute` / `record`).

**Next.** Name the candidates you want and predict them. Failures: an empty `recommendations` with `recommendations_basis.reason` is a valid cold-start answer, not an error; a `degraded` block means the text channel never ran, which is a different fact from an empty hit list.

### Embedding configuration (the text channel)

The text channel is **off unless a real embedding model is configured**, deliberately: a lexical hash is not semantic retrieval, and presenting one as semantic similarity would make text matching look meaningful when it is not.

| Setting | Meaning |
|---|---|
| `OR_EMBEDDING_BASE_URL` | OpenAI-compatible endpoint base (e.g. `https://host/v1`) |
| `OR_EMBEDDING_MODEL` | embedding model name |
| `OR_EMBEDDING_API_KEY` | credential (sent as a bearer header, never persisted) |
| `OR_EMBEDDING_BACKEND` | optional: `auto` (default), `local-hashing`, `none` |

These are NOT the `OR_WM_*` chat-model variables: the two models may be different endpoints with different dimensions. `auto` uses the endpoint only when all three of base URL / model / key are present, and otherwise returns no backend at all. It never silently substitutes the local hashing backend — that one is selected explicitly (`OR_EMBEDDING_BACKEND=local-hashing`) for hermetic offline runs and tests, and its index (`model_id=local-hashing-embedding-v1`) is invisible to a real model and vice versa, so two vector spaces never mix. A backend can also be injected directly in Python: `ORHarness(home=..., embedding=MyBackend())`. Index files, item shape and staleness: see the `rebuild-index` entry below.

If the text is English/CJK mixed, both are handled (the local backend tokenizes CJK per character; a real model does its own tokenization).

### `orx context --task t.json [--episode ep1] [--top 3] [--cir cir.json] [--math JSON] [--include-unverified] [--no-persist]`

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

**Structure consistency is checked after the effective CIR is resolved.** A snapshot taken under a different structure than the effective input (explicit CIR included) is REFUSED with the conflicting dimension named; the same check guards context reuse at `predict-strategy` (`--context CTX`). An unmeasured dimension is not a mismatch.

**The memory version digests content.** `result.capability_version.knowledge_content_digest` covers the decision-relevant content of the memory actually consulted (entry fields — including `applicability` and `actions` — plus the carried hits). Revising an entry moves it; a re-read does not (read timestamps are listed under `excluded_keys`). Only the digest and its composition are carried, never the digested content.

**Next.** Pass `result.context_id` to `predict-strategy --context` with the SAME `--cir` you built it with; a mismatched effective input is refused. Failures: `--no-persist` returns the context without storing it, so a later `--context-id` lookup cannot resolve it.

### `orx predict-cost --task t.json --strategy S`

Pre-execution **cost** expectation for one (task, strategy), with explicit provenance: `source=entry` (a matching Strategic Knowledge entry), `stats` (conditional statistics over attempt-scope evidence — a recount, with `support_n` and per-dimension `support_per_dim`), or `unknown` (no usable evidence: no data, task-scope data only, or clearly different scale — expected cost is `null`, NEVER a default zero presented as cheap). Pass the printed snapshot to `orx record --prediction` so feedback compares against the prediction actually used.

**Cost is not the world model.** This command calls NO provider and forecasts no benefit, risk or uncertainty: it reports what past evidence says an attempt costs. Use `predict-strategy` (or `plan-next`, which calls it once per candidate) for a consequence prediction.

**Next.** Hand the printed snapshot to `record --prediction` so feedback compares against the prediction actually used. This command calls no provider and forecasts no benefit or risk — use `predict-strategy` for that.

## Stage 3 — Predict and choose a strategy

### `orx [--world-model URL::MODEL] predict-strategy --task t.json --candidate c.json [--episode ep1] [--context CTX_ID] [--cir cir.json]`

**Strategy-outcome prediction under the wm-so/1 protocol.** Predict ONE candidate strategy's benefit / cost / risk / uncertainty from a frozen prediction input context. The candidate is a `CandidateRef` JSON (`{"action_type": "execute_strategy", "strategy_id": "S01", "solver": "highs", "config": {"time_limit": 60}, "scope": "attempt"}`) or a legacy `ActionSpec` (execution params and budget hint preserved verbatim; an unmappable scope such as `"task"` is refused). Same strategy_id with a different solver/config is a DIFFERENT candidate — the config travels with the prediction into the choice and the execution binding.

- **Input**: `--context CTX_ID` reuses a frozen context (identity verified against the task/version/episode AND the effective input version — pass the SAME `--cir` you built the context with); the default builds one fresh context for this call. The provider receives the full context CONTENT (joint problem representation, retrieval evidence, capability evidence, constraints), not ids.
- **Output**: a `StrategyOutcomePrediction` — benefit with metric/unit/baseline (`solution_quality` must be NORMALIZED in [0,1]; a raw objective value is refused, never clamped), cost as a CostVector with a predicted-dimension mask, risk as named events separate from cost, uncertainty with the model's self-report recorded as explicitly UNCALIBRATED. Every failure state (not configured, provider error, empty payload, invalid JSON, NaN, out-of-range probability) is distinguishable and persisted with whatever usage the call consumed. One call, no retries, no defaults.
- **Binding**: the ordinary flow establishes the association at `execute --prediction` time, BEFORE the run. `orx bind-strategy --prediction ID --action ACTION_ID` is the MANUAL/recovery path for an action that was already run without it: it applies the SAME identity rules (action type/task/episode/strategy/solver and the candidate's execution config), so a manual bind behaves exactly like an automatic one. Full spec: [strategy_outcome.md](strategy_outcome.md).

**Describe the candidate's METHOD.** A candidate carries a `method` object — `{"name": str, "steps": [str, ...], "why": str?, "fallback": str?}` — and the model predicts from it. A bare `strategy_id` (or a solver name) tells the model nothing about what would happen, so an undescribed method is an unanswerable request. Supply it when you propose the candidate, in the caller's own words; an empty value means YOU did not describe it, never that there is nothing to describe. This matters most at a cold start, where there is no memory to read a description from. The solver and `config` stay their own fields: those are what the execution runs with, not what the approach is.

**Benefit metric: declare ONE meaning, BEFORE the result is seen.** The prediction's `kind`/`metric` decides how it is later compared and observed:

| You are predicting | Declare | Observed from |
|---|---|---|
| how well the solver solved the model | `kind=solution_quality`, `metric=normalized_objective_gap` | the solver's gap (an `optimal` status is a gap of 0) |
| whether the ANSWER satisfies the TASK | `kind=effective_completion`, `metric=task_result_check_passed` | the execution's own `check-task` verdict (1.0 passed, 0.0 confirmed failed, UNKNOWN when unchecked) |

The two are different measurements and are never interchanged. `valid_progress` has no observation channel in this build: it is reported, never scored, and is never quietly replaced by a completion rate.

**Failure is one state with a structured reason.** A call that produced no usable prediction is `status="invalid"` and carries `result.failure`:

| `failure.kind` | What happened | What to do |
|---|---|---|
| `truncated` | the endpoint stopped for length (`finish_reason` is `length`/`max_tokens`) | raise the output budget (`--wm-max-tokens` / `$OR_WM_MAX_TOKENS`) and predict again; if thinking is ON it is spending the same budget, so also consider `OR_WM_ENABLE_THINKING` |
| `wrong_top_level` | valid JSON that is not an object — the field case of `[]` | fix the request or the prompt; a bigger budget will not help |
| `empty_response` | the endpoint returned `""` or `null` | check the endpoint; the answer was never produced |
| `unparsable` | the content is not JSON | inspect the endpoint's output format |
| `timeout` / `network_error` | the call did not complete | retry, or raise `--wm-timeout` / `$OR_WM_TIMEOUT` (and the plan's `--time-budget`) |
| `unusable_payload` | a JSON object that failed contract validation | the payload's own notes name the offending field |

`finish_reason: null` means the endpoint reported nothing: how the answer ended is UNKNOWN, and it is never read as a clean stop. A failed prediction is RECORDED under its own id with its real call cost, and it stays READABLE — a provider outage never blocks reading the log or closing the episode.

**Next.** Keep the chosen candidate's `result.prediction_id` and pass it to `execute --prediction`. Branch A of the workflow: do not also run `plan-next` for the same decision.

### `orx [--world-model URL::MODEL] plan-next --task t.json [--episode ep1] [--candidates specs.json] [--horizon 1] [--max-calls N] [--delta W] [--prediction-mode M]`

**Bounded next-step planning (wm-so/1 — the only planning protocol).** Compare a small set of candidate strategies by their PREDICTED consequences and get a suggested first step:

1. freezes ONE root snapshot for the whole comparison (all candidates see the same state);
2. builds ONE frozen prediction input context and predicts every candidate against it (default ≤3 candidates, ≤6 model calls, plus a wall-clock budget);
3. scores each candidate on ONE conservative yardstick: `U = alpha*G − beta*C − gamma*R`, where an unpredicted cost dimension is charged that dimension's PEAK normalized share (unknown cost is never free), a risk event with no probability is charged the full weight, and a benefit in a non-comparable currency contributes NOTHING (unknown upside is never rewarded). The knowledge term is OFF for this protocol;
4. suggests the candidate with the highest utility.

**Horizon is FIXED at 1.** There is no imagined multi-step rollout: one macro comparison, then you re-plan from the REAL observation of the chosen step. `--horizon` accepts only 1; any other value is refused with the replacement path named rather than silently ignored.

- **Candidates**: `--candidates` (your own ActionSpec list) is REQUIRED. The framework does not generate a candidate menu — there is no directory to enumerate. You propose the methods you want compared (with their configuration), and the world model predicts their consequences; `orx recall` shows what memory already holds for this problem.
- **Reuse the prediction — do not predict again.** Every compared candidate's prediction id is returned under `result.plan.candidates[]`. Accept with `orx choose-next --decision <id> --prediction <the chosen candidate's id>`, then pass the SAME id straight to `orx execute --prediction <id>`: the association to that prediction is established before the run. Calling `predict-strategy` for the same candidate a second time would create a DUPLICATE prediction of one decision.
- **Bounds**: candidate count, `--max-calls`, and a wall-clock budget. Exhaustion truncates with an explicit reason — never a silent partial answer. The real planning spend is charged ONCE to the decision action and reported in `planning_cost` — sunk, never part of any candidate's score.
- **A suggestion is not a selection**: `plan-next` never writes `X.selected_plan` and never executes. Only `choose-next` does.
- **Unknowns are reported, not hidden**: missing cost/benefit/risk predictions appear per candidate under `score.incomparable`; with an undeclared or partially-unknown budget, `budget_confirmation` is `unknown`/`unconfirmed` — never claimed "within budget". An already-exceeded real budget stops planning before any model call (`status=fallback`).
- **No usable prediction at all** → `status=no_valid_predictions` with the real statuses and NO suggestion. Fall back to `recall`/Selector ordering or choose yourself; the calls that happened and their cost are recorded.
- **Requirements**: a configured `--world-model` provider (mandatory). Planning supports `execute_strategy` candidates; other action types are reported as not plannable. `plan-next` is an OPTIONAL planning aid — it does not affect the mandatory predict/bind duty, and it REPLACES a separate `predict-strategy` when you use it. Full spec: [strategy_outcome.md](strategy_outcome.md).

**Next.** Pass `result.decision_action_id` to `choose-next`, then execute the chosen candidate with its `result.plan.candidates[].prediction_id`. **Time budget.** `--time-budget S` (default 120) bounds the WHOLE decision: before each call the remaining budget is passed down as that call's timeout, and after the last call the elapsed time is re-checked. The result reports the values that were in force under `result.effective_parameters`. Failures: `--horizon` other than 1 exits 2 by design; `no_valid_predictions` means no candidate carried a usable prediction — fall back to `recall` ordering or choose yourself; a per-candidate prediction that failed is reported under that candidate with its own `failure.kind` (see `predict-strategy`), and the OTHER candidates still get scored.

### `orx choose-next --decision ACTION_ID (--prediction PREDICTION_ID | --chosen spec.json | --rejected) [--note "..."]`

Record YOUR explicit choice after a plan: accept the suggestion, pick another candidate (a deviation, recorded with its reason), or reject all. Only this call writes `X.selected_plan`; the choice itself produces no execution quality.

**Name the candidate by its prediction id** (the ordinary path): `--prediction` is the id `plan-next` returned for the candidate you chose, under `result.plan.candidates[].prediction_id`. The candidate is read from the DECISION's own recorded comparison, so a prediction that belongs to another decision (or was never compared here) is refused rather than silently chosen — you never have to re-type the candidate JSON, and the tool cannot pick a candidate the decision never evaluated. `--chosen` (the full ActionSpec) remains available; if both are given they must name the same candidate. The deviation test compares the candidate REFERENCE — action type, strategy, solver and configuration — never the whole plan JSON (a cosmetic key is not a deviation) and never `strategy_id + solver` alone (the same pair under a different config is a different candidate).

Then execute the step with `orx execute --prediction <that same prediction_id>`, which establishes the association to the prediction the plan already made — do NOT call `predict-strategy` again. Record the result with `orx record` and re-plan from the new real state afterwards; the old plan stays as the suggestion of its time.

**Next.** `execute --prediction <the chosen candidate's id>`. Failures: an unknown or non-`select_strategy` decision action is refused; a prediction id that is not among the decision's compared candidates is refused; a rejected suggestion is recorded as `last_rejected_suggestion` and never overwrites `selected_plan`.

## Stage 4 — Execute and verify

### `orx execute --task t.json [--prediction PREDICTION_ID | --strategy S04 --solver NAME] --code solve.py --workspace DIR [--episode ep1] [--method JSON] [--adapted-from ex_a,ex_b] [--adaptation TEXT]`

Run this for the strategy you CHOSE. Predicting several candidates is how the choice is made; executing one is what the loop does. A further attempt is your decision after a real failure or an unresolved uncertainty — nothing here starts a second solve on its own, and every attempt that does run is charged its own real cost.

**Two ways in, and they differ in what is established when.**

- **With `--prediction`** the association is established BEFORE the run. The FROZEN candidate supplies the strategy, the solver and the episode (do not re-type them — an explicit argument that contradicts the candidate is refused before anything executes); the prediction's status, the problem identity and the claim on the prediction are all checked up front. A prediction already consumed by another attempt is refused (one attempt per candidate); a prediction made for a DIFFERENT problem is refused (note that writing the `model` field after predicting is NOT a change — the problem identity excludes post-strategy artifacts). The action and its link are persisted before the executor starts, so an interrupted execution still shows which attempt the prediction was testing; after the run only the RESULT and the observed configuration are added. Precondition failures exit 2 with the reason and spend NOTHING.
- **Without `--prediction`** you supply `--strategy` and `--solver` explicitly: a legitimate unpredicted attempt, recorded as such. `bind-strategy` can associate a prediction with it later.

You write `solve.py` following the strategy's actions (the framework never generates code). It runs in a sandbox and its **static policy** is precise, so read it before debugging a rejected script: `subprocess`, `socket`, `urllib`, `http`, `requests` and `shutil` are blocked imports, `pathlib` is blocked, and only `os` / `os.path` are allowed from `os` (calls such as `os.chdir`, `os.walk`, `os.remove` are refused). **Importing a solver library is allowed** — `ortools`, `highspy` and other in-process solvers are how you are expected to solve; only a subprocess-based solver (PuLP's CBC) cannot run here. Every constraint label must be `C1`, `C2`, `C3`, … so the L2 symbol cross-reference can resolve it. POSIX rlimits + a wall-clock timeout apply.

Your script writes its answer with a literal `open('result.json', 'w')`. The script is executed with `--workspace` as its working directory, and it must live inside that workspace, so the relative path resolves to `<workspace>/result.json` and no other path is accepted. **A `result.json` left there by an earlier run is removed before this attempt starts**, and a file older than the run is refused — otherwise a script that wrote nothing would hand back the PREVIOUS run's status and objective as this attempt's result, which is exactly how a stale sample would enter the calibration. Your script must write `result.json` with at least `status` (optimal|feasible|infeasible|unbounded|timeout|error), `objective_value`, `objective_bound`, `mip_gap`, `runtime_seconds`, and — whenever a task-level check will need to read the answer — `variables`, an object of variable name → value (e.g. `{"x1": 2, "x2": 3}`). The solution vector is what makes `orx check-task` possible: integrality and objective recomputation can only be checked against actual values. It is stored on the record as `execution_features.solution_variables` (truncated past 2000 entries, with `solution_variables_truncated` naming what was dropped — a check needing a dropped variable reports it as unchecked rather than passing).

**Report the method that actually ran (optional but recommended).** The configuration receipt tells the framework WHAT PARAMETERS took effect; the method receipt tells it HOW the work was done. Have `solve.py` write the method it really carried out back into `result.json` under `method_performed`, stamped the same way:

```python
import json, os
json.dump({"status": "optimal", "objective_value": 10755,
           "method_performed": {
               "action_id": os.environ.get("OR_ACTION_ID"),
               "name": "rolling-horizon decomposition",
               "steps": ["relax the coupling constraint",
                         "solve the master", "recombine"]}},
          open("result.json", "w"))
```

It is accepted only when its `action_id` matches this attempt (a leftover is refused), read on the failure paths too, and the performed steps are added to the record's trajectory in order. The result lands on the record as `method_actual` (`method_planned` carries the candidate's own method, or the `--method` you passed) and under `execution_features.method_receipt`. **A plan is not a fact**: without a receipt the performed method stays `None`, and the outcome reports `config_observed`-style evidence — never a copy of the plan. `--method JSON` states the method for an UNPREDICTED attempt (`{"name": ..., "steps": [...]}`).

**Cite the past cases you adapted from (optional, `--adapted-from ex_a,ex_b`).** When you reused a method from `vector_recall.execution_evidence` / `evidence_candidates`, name those executions here and add one line with `--adaptation "..."`. This is written to the record as `execution_features.reuse_trace = {adapted_from: [...], adaptation: "...", note: "..."}`. It is a CITATION, never a claim of benefit: it records that you READ those cases, **not** that reusing them worked — and "was recalled" is never counted as "reused successfully". Whether the reuse helped is judged by THIS attempt's own task check and by calibration, never by the citation. Unknown ids are kept verbatim (the framework does not validate existence); the field is a fact about what you looked at, and a later reader (and the offline induction step) can retrace it.

**Report the configuration that really took effect (optional but recommended).** To let the framework tell a predicted configuration from an observed one, have `solve.py` write the parameters it really used back into `result.json` under a `config` object, stamped with the action id the executor passes through `$OR_ACTION_ID`:

```python
import json, os
cfg = {"action_id": os.environ.get("OR_ACTION_ID"),
       "time_limit": 60,          # the limit the SOLVER actually applied
       "seed": 42}
json.dump({"status": "optimal", "objective_value": 10755,
           "objective_bound": 10755, "runtime_seconds": 4.1,
           "config": cfg}, open("result.json", "w"))
```

The receipt is accepted ONLY when its `action_id` matches this attempt's action, so a leftover `result.json` from an earlier run in the same workspace is never read as this run's configuration. It is read on the FAILURE paths too (a timeout after the parameters were set still reports them). The result appears under `execution_features.execution_config` — `values`, `sources`, and where relevant `executor_configured`, `conflicts`:

| Key | Meaning |
|---|---|
| `values` | what really took effect, keyed by parameter |
| `sources[key]` | `executor` (the sandbox's own policy — `solver`, `script_timeout_s` = the wall clock over the WHOLE script), `script_reported` (a value the script READ BACK), or `executor_configured` (an INSTRUCTION the executor handed the script) |
| `executor_configured` | values the executor passed to the script through the environment (e.g. `solver_timeout_s`). The framework cannot observe whether the solver honoured them, so they are **not** evidence of what the solver used — only a read-back value under `script_reported` is |
| `conflicts` | a key the SCRIPT reported that contradicts something the executor really did. The executor's own value stands; the disagreement is recorded. A script cannot overwrite (`script_timeout_s: 1` against a real 120s is kept as 120 and flagged) |

Two timeout notions are kept apart: `script_timeout_s` is the wall clock over the whole script (the sandbox's own limit), while the limit handed to the SOLVER is a different number the script must read back itself. A key absent from `values` is UNKNOWN — the association never copies the predicted config in to manufacture a match.

The profile used in `execute` is the same frozen pre-strategy signature from `recall` — derived from `coupling` (CIR) / `spec` / `annotations` / `model` only.

Every execution is automatically staged in a pending area (successes and failures alike) — staging is a safety net, not recording. Result: `result.execution` = a full ExecutionRecord (id, quality check, CostVector, `cost_measured` mask, `solver_runtime_provenance`, `execution_features` with solver diagnostics if the solver reported any, `cir_snapshot` = the CIR that was actually solved when the task carries a `coupling` field, and `task_text_digest` = the task-text VERSION this execution was produced under). **Nothing is recorded yet.**

The executor measures only what it can observe: `latency_s` and `solver_runtime_s`. `tool_calls`, `retries` and `llm_tokens` are **yours to supply** — they stay out of `cost_measured` until you do (a constant is not a measurement). **For `llm_tokens`, prefer the REAL figure**: `record --usage-file <json>` ingests the outer framework's ATTEMPT-level usage report (its hook/log/storage output — OpenClaw `llm_output`, Hermes `post_api_request`, or the canonical `{prompt_tokens, completion_tokens, reasoning_tokens, cached_tokens, model, calls}` shape), records the full口径 total and marks it a host observation. Only fall back to a hand-typed `--override llm_tokens=` when no report exists — that value defaults to `agent_estimate` (a hand-typed number with no stated source: shown, but **never** used as a calibration actual or a measured cost claim). Pass `--override-source agent_observed` when you READ the number off a real report, or `provider_usage` when the provider itself reported it. A late report also backfills through `amend-cost <id> --usage-file <json>`. `tool_calls` counts ALL tool invocations in the attempt's scope (shell commands, file reads/writes, sandbox runs, solver calls), so the executor records only its provable floor in `execution_features.tool_calls_lower_bound`. `retries` gets a measured zero only when the framework can PROVE this is the first attempt of that (task, episode, strategy) — no earlier `execute_strategy` action of the same triple ever produced an execution — so a FIRST attempt records `retries=0` as an observed fact while a later one stays unknown until you declare it (`execution_features.retries_proof` says which case applied). A script rejected by the sandbox policy measures nothing at all (`cost_measured` is empty) — 0.0 seconds of solving would be a fabricated fact. `execution_features.cost_notes` explains any rejected or suspicious value.

`execute` also persists the task text (`task_texts`, keyed by `(task_id, text_digest)`) — that is where the retrieval document comes from. Solving the SAME `task_id` with different content creates separate versions, so each execution stays linked to the text actually in force, and the two can never be confused later.

**Next.** `check-task <result.execution_id>` before treating the answer as a success, then `record --from-staged <result.execution_id>`. Failures: a sandbox-policy rejection measures nothing and still stages the attempt; with `--prediction`, a precondition failure (unknown prediction, changed problem, contradictory argument, an already-claimed prediction, a script outside the workspace) exits 2 BEFORE the run and spends nothing; `result.prediction_binding` reports `bound`, `config_observed` and `config_unknown`, and a `bound: false` names the reason and can be rebuilt with `bind-strategy`. An executor exception AFTER the attempt started (a filesystem error, an interruption) exits 2 with an actionable line: the attempt is staged as its OWN failure fact (with its measured cost and failure class) and, for a prediction-driven run, the association is released so the SAME prediction can be tested against a real attempt — the failure stays an observation, and the retry is a separate attempt (no proven `retries=0`).

### `orx check-task <execution_id> [--check JSON] [--episode ep1]`

Answers the question `execute` cannot: **does this answer satisfy the ORIGINAL task?** The executor's own verdict covers the solver's MODEL (a legal status, a finite objective, a gap); a relaxed LP answered with fractional values is `optimal` with `gap=0` and still wrong. Without this call such an answer enters recall, the conditional statistics, the world-model feedback and offline induction as a success sample.

Run it between `execute` and `record`, or later on an already-recorded execution (a late correction is a real event — see the close-out section below). Choosing a basis is your responsibility: the framework only verifies what you declare, so a wrong reference is not detected by it, and an undeclared constraint is simply not covered. **Never replace a benchmark reference with your own derived value and call the task passed** — declare `reference_source` so a pass against a self-derived value is not read as an independent confirmation.

**`--check JSON`** declares what the framework may verify. Every base is optional; only what you declare is evaluated, and everything else is listed as UNCHECKED in `report.scope.unchecked`:

| Key | Meaning |
|---|---|
| `reference_objective` (+ `tolerance`) | the reported objective must agree with the reference. Default tolerance `1e-6 * max(1, abs(reference))` — the SAME rule admission verification uses |
| `reference_status` | the reported solver status must equal it (e.g. `"optimal"`) |
| `integer` | `{"variables": ["x1","x2"]?, "tolerance": 1e-6?}` — every named variable (or every recorded variable) must be integral. This is the basis that catches an LP relaxation |
| `recompute_objective` | `{"coefficients": {"x1": 7}, "constant": 0?, "tolerance": 1e-6?}` — the objective is recomputed from the recorded solution and compared with the reported one |
| `semantic_probe` | one or more `{"path", equals\|min\|max\|in}` probes over the record payload, e.g. `{"path": "execution_features.solution_variables.x1", "min": 0}` |
| `semantic_probe` path | a PATH that does not resolve reports `insufficient` (a typo is NOT a refutation) and names the available top-level paths |
| `reference_source` (+ `reference_version`) | `bench_declared` \| `independent` \| `self_derived` — WHERE the reference value came from. A value you derived yourself, or left UNSTATED, is reported as self-derived: a pass is then a consistency check, not evidence the answer satisfies the task |
| `intent` | `"relaxation"` or `"intermediate"` — marks an execution whose answer is deliberately NOT the task's answer. It is recorded and reported, never treated as a pass for the task |

**Three verdicts, and none of them is a default:**

- **`passed`** — every declared basis held on the recorded values. The report still names what it did not check (the model's fidelity to the task, undeclared constraints), so `passed` is never read as "fully validated", and `report.reference_source` states whether the reference was independent.
- **`failed`** — a declared basis ran and did not hold. The execution is **not** demoted or rewritten: its observed quality and cost stand and it stays in the evidence set (the cost is real, the failure is raw material). What changes is that it can no longer count as a success sample: every quality consumer (statistics, world-model feedback, calibration) reads `0.0` for it, while its cost still counts in the task total.
- **`insufficient`** — no basis declared, no solution vector recorded, a needed variable missing, or the execution produced no usable result. **Not a pass, not a failure**, and never a demand that you supply a reference. An unchecked answer's validity is UNKNOWN.

The framework does **not** parse natural-language constraints: a constraint is checked only through a `semantic_probe` or a `recompute_objective` you declare. A matching objective never proves the model correct.

**Recording.** Two channels: a `verify` ACTION (the framework really ran these checks, so it is logged with its verdict — the per-check history is auditable) and `execution_features.task_check` on the execution itself (a narrow annotation that works for staged and recorded executions alike). `task_check.state` is `passed` / `failed` / `insufficient`; a malformed or absent annotation reads as "no check", never as a pass.

**Read the result like this.** Top level: `result.state` (`passed`/`failed`/`insufficient`) and `result.report`; `result.report.conclusion` is the human sentence, `result.report.checks[]` the per-basis outcome, `result.report.scope.unchecked[]` what was NOT checked, `result.reflection_material` (present on `failed`) the material for the repair, and `result.next` the recommended next step (present when there is one). A check written to a STAGED execution is carried onto the recorded fact, so `check-task` before or after `record --from-staged` gives the same close-out verdict — the order does not matter, only that both happen before `close-episode`.

**On `failed`, the response carries `reflection_material`** — the task text, the code hash, the recorded solution vector, the check report and the earlier attempts of the episode — plus the `next` instruction. Locating the cause (task interpretation, model, implementation, or the reference basis itself) is YOUR job: the framework does not classify the failure as a modeling mistake, does not rebuild the model, and never relaxes the task to match a reference value.

**Next.** `record --from-staged` the attempt either way. On `failed`, read `reflection_material`, fix the candidate, and predict it again; on `insufficient`, declare the missing basis or have `solve.py` report `variables` — an unchecked answer's validity is UNKNOWN.

### `orx bind-strategy --prediction ID --action ACTION_ID`

Bind a strategy-outcome prediction to an action that ALREADY ran. This is the manual/recovery path — the ordinary flow associates the prediction at `execute --prediction` time, before the run — and it applies the SAME rules, so a manual bind is indistinguishable from an automatic one. It checks request identity (action type/task/episode/strategy/solver and the candidate's execution config) and sets `trace.comparable` only when the bound action is a completed real scope. **A missing receipt is a caveat, not a discard**: a field the executed action could not observe (no episode recorded, a config key the action log or the script's config receipt never reported) is recorded under `binding_unknown` — separately from a known `binding_mismatch` — and the close-out blocks only the comparison it really invalidates (an unreported `time_limit` blocks nothing; a different solver blocks the outcome dimensions and keeps the cost). The config comparison reads the action's own `execution_config` (the executor's report plus the script's stamped receipt) and reports `config_observed` / `config_unknown`, so the predicted config is never copied in to manufacture a match. Idempotent (a re-bind re-evaluates the comparability); no model call; nothing re-billed. The per-field evaluation itself happens at episode close-out (`orx close-episode`); see [episode_closeout.md](episode_closeout.md).

**Next.** Close the episode when the scope ends, so the associated prediction is evaluated. Failures: a mismatch or unknown identity field is recorded and makes the prediction non-comparable rather than silently matching it.

## Stage 5 — Record facts and cost

### `orx record --execution <json|path> | --from-staged <id> [--usage-file <json|path>] [--usage-host <host>] [--override llm_tokens=1840,tool_calls=9] [--override-mode replace|increment] [--override-source agent_estimate|agent_observed|provider_usage] [--force] [--prediction <json>] [--method JSON] [--method-actual JSON] | --discard-staged <id>`

**What `--usage-file` accepts.** Either an inline JSON object or a path. Two shapes:

1. **The versioned HostUsageReport** (`schema: "or-host-usage/1"`) — the attempt-level report the HOST produces (it owns the LLM connection and the tool invocations, so the real `llm_tokens` and `tool_calls` are its facts). One report fills BOTH dimensions with no hand-typing:

```json
{"schema": "or-host-usage/1",
 "host": "openclaw",
 "model": "paratera/DeepSeek-V4.1-Flash",
 "scope": {"task_id": "...", "episode_id": "...", "attempt_id": "..."},
 "tokens": {"prompt_tokens": 12345, "completion_tokens": 678,
            "reasoning_tokens": 291, "cached_tokens": 0, "calls": 3},
 "tool_calls": 38,
 "tool_calls_lower_bound": 1,
 "measured": ["prompt_tokens", "completion_tokens", "tool_calls"],
 "provenance": {"llm_tokens": "provider_usage",
                "tool_calls": "agent_observed"},
 "notes": "..."}
```

Rules: `llm_tokens` = prompt + completion (the FULL口径; reasoning/cached are sub-facts inside those sides, never added again; a single side alone is a LOWER bound, `basis=completion_only`/`prompt_only`). `tool_calls` = ALL tool invocations in the attempt's scope (shell, file reads/writes, sandbox runs, solver calls) and must be `>= tool_calls_lower_bound` — a report contradicting its own bound is refused. Only dimensions in the `measured` whitelist enter `cost_measured`; everything else stays UNKNOWN, never zero. Each dimension is recorded under its own `provenance` entry (`llm_tokens` → `provider_usage`, `tool_calls` → `agent_observed` by default).

2. **The older permissive shape** — every key optional, kept for backward compatibility:

```json
{"prompt_tokens": 700, "completion_tokens": 300,
 "reasoning_tokens": 120, "cached_tokens": 40,
 "model": "your-model", "source": "your-host",
 "report_id": "optional-dedupe-key",
 "calls": [{"id": "call-1", "prompt_tokens": 400, "completion_tokens": 150}]}
```

A per-call `calls` list is summed only when the totals are absent (a call's own prompt+completion is never added twice). Repeats are collapsed: a `report_id`/`usage_id` seen before, or a call `id` seen before, is not counted twice, and a late report (`"late": true`) for a known call REPLACES its figure instead of adding to it. A report with no token numbers records nothing and says so — unknown is never zero.

**`--usage-host <host>`** locates the report through a NAMED adapter instead of passing the file: `openclaw` looks under `<home>/host_usage/` (per-attempt `openclaw-usage.<execution_id>.json`, then the untagged `openclaw-usage.json`); `generic` reads `host-usage.json` there. When no report is found, `record` still succeeds — the dimensions stay UNKNOWN (never zero) and the summary says the adapter found nothing. An unknown host name exits 2 with the known hosts listed. `$OR_HOST_USAGE_FILE` may name a default report path (checked last, after `--usage-file` and `--usage-host`).

**How the host produces the report** (for the external implementation — not this repo): (1) on each model call, request usage (a streaming endpoint needs `stream_options.include_usage=true`, exposed as `compat.supportsUsageInStreaming:true`); (2) at attempt end, write one `or-host-usage/1` JSON per attempt with the summed tokens, the whole-scope tool-call count and the measured whitelist; (3) place it where the adapter looks (`<home>/host_usage/openclaw-usage.<attempt>.json`) or hand it to the agent for `--usage-file`.

**Conflict rule.** Do not pass `--usage-file` (or `--usage-host`) AND `--override` for the SAME dimension: the host report is a real observation, so the override is refused rather than silently overwriting it. `--override` remains for dimensions the report does not carry.

**What `--candidate` takes on `predict-strategy`.** An inline JSON object or a path, in the contract's `CandidateRef` shape or the legacy `ActionSpec` shape (the same objects `plan-next --candidates` takes, as a LIST):

```json
{"action_type": "execute_strategy", "task_id": "t1", "episode_id": "ep1",
 "strategy_id": "my-method", "solver": "highs",
 "method": {"name": "rolling-horizon", "steps": ["relax coupling", "solve master"]},
 "config": {"time_limit": 60}}
```

`method` is a PLAN in YOUR words (a bare string is read as the name); `config` holds execution parameters only — a `method`-like key placed in `config` is moved to `method` and recorded, and a key that names no execution parameter is refused before an attempt is spent. `recall --candidate` is different: it takes a bare method ID for filtering.

Appends the fact to the Execution Evidence Bank, then runs the automatic chain: cost backfill (replace by default — idempotent, never double-counts) → quality checks against matching entries, frozen ONTO THE FACT (`execution_features.quality_feedback`: the interval in force, the observed quality, hit/miss — only the strategy that actually ran is checked, attempt scope only) → cost feedback against the record's frozen pre-execution prediction snapshot (same strategy, same scope, both sides measured; written into `execution_features.cost_feedback`).

**Recording never changes knowledge AND never interprets it.** No entry is promoted, demoted, or woken here — the frozen checks are replayed by the next `orx induce`. `record` also emits no induction labels: it does not decide what the fact MEANS. Read the material with `orx review-material` and abstract it yourself.

**Method declarations for a record you assembled yourself.** `--method JSON` is the PLAN (`method_planned`); `--method-actual JSON` is what you DECLARE was actually performed. Use the latter only when you really observed the processing. A value the record already carries is never overwritten, so a script-reported performance stands over a later declaration, and the plan is never promoted to fact.

There is no retention mark to set: retention is decided by the evidence WINDOW (`orx enforce-window`), not by a per-record label. To reserve an episode as contrast evidence, set the explicit `contrast` marker in its `execution_features` (the channel knowledge evaluation reads).

**Cost completeness.** `result.cost_completeness` appears when any dimension is still unmeasured: `missing` names them, `lower_bounds` carries the provable floor (the sandbox invocation), and `note` gives the exact `orx amend-cost` call that closes the gap. Recording is never blocked — the fact is honest as it stands, and an unmeasured dimension is recorded as UNKNOWN rather than fabricated — but an unmeasured dimension supports no cost claim and no cost prediction until every supporting record measures it. `--execution` accepts the bare record, the `orx execute` envelope, or the legacy one-level form.

**Task text and index sync (best effort).** `record` can be reached WITHOUT `execute`, so it resolves the text link itself: the digest the caller supplied, else the task's most recent real belief snapshot (`hypothetical=False`, whose frozen `task_payload` is rebuilt through the same reader used at capture time). If neither exists, `task_text_digest` stays `None` and the record is simply not vector-indexed — the text is **never invented**, and the memory keeps its full profile-based visibility. After the fact is durable, the execution's index item is refreshed; an embedding failure does **not** roll anything back and is reported as `result.index_sync = {state: "deferred", reason: ...}` (recover later with `orx rebuild-index`). `state` is `synced` / `deferred` / `skipped` (`skipped` = no backend configured, or the text is unavailable).

Result: `result.{execution_id, recorded, prediction_checks[], cost_completeness?, cost_feedback?}` plus, when same-task executions are staged but unrecorded, `result.unrecorded_staged_executions[]` — backfill those with `--from-staged` (records the original payload verbatim; never re-type an execution JSON by hand).

**Next.** `close-episode`, or go back to prediction if you are retrying. Failures: `cost_completeness.missing` names the dimensions still unmeasured, and `index_sync.state != synced` is a deferred index write, recovered with `rebuild-index` — neither blocks the fact.

### `orx amend-cost <execution_id> [--override llm_tokens=1840,tool_calls=9] [--usage-file JSON|PATH] [--usage-host <host>] [--usage-source host] [--mode replace|increment] [--source agent_observed|agent_estimate|provider_usage] [--force]`

Backfills cost dimensions of an **already-recorded** execution, in place — nothing is re-run and nothing is appended. This is the repair path for a cost gap (`record`'s `cost_completeness.missing`, `induce`'s `cost_claim_withheld`, `close-episode`'s `cost_completeness_warnings`): an unmeasured dimension supports no cost claim until every supporting record measures it, and this is how you close it without re-executing.

`--mode replace` (default) means the value IS the measurement — re-applying the same override is idempotent and never double-counts. `--mode increment` adds an additional measured amount within the record's scope. Cost feedback is recomputed against the frozen prediction snapshot, so the stored summary can never disagree with the stored fact.

**Where the number came from is recorded, and it decides what the number MAY be used as.** `--source` (default `agent_estimate`) tags the amended dimension: `provider_usage` (a provider/host reported it — a real observation), `agent_observed` (you read it off a real report), or `agent_estimate` (the default — a hand-typed number with no stated source). The tag travels in `execution_features.cost_provenance`, so a real measurement is never indistinguishable from an estimate in the stored fact. An `agent_estimate` is kept and shown everywhere, but it is **never** used as a calibration actual, a measured cost claim or a learning evidence mean — see [concepts.md](concepts.md). A single-side token figure (`basis=completion_only`/`prompt_only`) is likewise a LOWER bound and cannot stand in for a complete total. **A framework-measured dimension cannot be silently overwritten**: `latency_s` / `solver_runtime_s` (and any dimension recorded with `provider_usage`) are refused unless `--force` is passed — a real observation is never silently rewritten.

The token口径 is the FULL total (prompt + completion; reasoning/cached are sub-facts, never added again). A declaration is tagged with its basis, and the budget view reports `token_basis_mixed` rather than summing a legacy completion-only figure with a full total as if they were the same unit.

A `tool_calls` declaration below the sandbox's provable lower bound is refused (exit 2): the executor demonstrably ran the solve script, so the record itself refutes the number. The response reports `still_missing` — the dimensions that remain unmeasured — so you can see whether the gap is closed.

Result: `result.{execution_id, mode, cost, cost_measured, still_missing}` — the amended record's five dimensions, the mask that now includes the backfilled one(s), and the dimensions still UNKNOWN (each supports no cost claim until measured). A host report (`--usage-file`/`--usage-host`) adds `result.usage` (the normalized report: host, model, token breakdown with its basis, per-dimension provenance). **A late host report backfills through `--usage-file` or `--usage-host`** (the host's hook/log output, or `@path`): a versioned `or-host-usage/1` report applies BOTH its dimensions at once, each under its own provenance (`llm_tokens` → `provider_usage`, `tool_calls` → `agent_observed`), exactly as `record --usage-file` would have; the older permissive shape backfills `llm_tokens` with `source=provider_usage`. Prefer that (or `--source provider_usage`/`agent_observed` when you read a number off a real report) over a bare hand-typed `llm_tokens`, which defaults to a declaration. A task's full cost is the sum of its recorded attempt-scope records (`api.task_cost_summary`): every attempt charged to the strategy that actually ran it; retries = sum of per-attempt NEW retries; end-to-end latency is unknown unless you supply explicit task timing (never inferred by max or sum).

**Next.** Re-run `induce` if the gap was a `cost_claim_withheld` dimension. Failures: a `tool_calls` value below the sandbox's provable floor is refused (exit 2), since the record itself refutes the number.

### `orx snapshot --task t.json [--episode ep1]`

Freeze and persist the current **belief state** for a task (the world-model substrate): H (value-copied verified/legacy/unverified knowledge refs + experience counts + tool config), P (profile + task digest + CIR/model digests + the problem-relevant task payload when no `task_ref` exists), X (the episode's accumulated progress — selected plan, model artifact, current solution, verification evidence — each field labelled with `provenance` (observed vs agent_reported) AND `epistemic` (fact/inferred/unknown), two orthogonal dimensions), B (declared budget + consumption), and a coverage view (cell statistics + layered knowledge, no composite capability score). Snapshots are frozen by deep-copy isolation: later bank writes never change what a snapshot meant. Within an episode, each action's post state merges into the accumulated progress, so the next action inherits what earlier actions established.

**Next.** Use it as the frozen pre-state for a prediction context, or read it to see what the episode had established. Failures: a task payload without `task_id` and `family` raises; it is never fabricated.

### `orx action --report TYPE --task t.json [--episode ep1] [--params JSON] [--outcome JSON] [--cost JSON] | --amend-cost ACTION_ID --cost JSON`

Report an action **you** performed (TYPE ∈ model/select_strategy/verify/finish_task) — the library did not execute it; the record is labelled `source=agent_reported` and a missing pre-state is marked `pre_snapshot_missing`, never fabricated. Explicitly supplied cost dimensions (including an explicit zero) are marked measured. `--amend-cost` backfills an action's cost (replace semantics, idempotent).

**Next.** Re-read it through `inspect --bank actions`, or amend its cost later with `--amend-cost`. Failures: an action reported without a pre-state is marked `pre_snapshot_missing` rather than filled in.

### `orx budget --task ID [--episode ep1] [--declare llm_tokens=50000,...]`

The budget view for a task/episode: consumption over ALL real action costs — recorded AND staged-but-unrecorded executions (deduplicated by execution_id) plus own-cost actions (a macro's reference cost is never re-counted; its own additional spend is). Status: `exceeded` (a measured dimension over the limit; a declared latency budget is judged per-attempt), `ok` (every declared dimension measured and within), `unconfirmed` (known spend within limits but a declared dimension is unknown — NOT confirmed within budget), `no_budget_declared`. Executions of the task with no episode linkage are reported separately as `unattributed` and never charged to a fresh episode. Declarations persist in the store, so separate CLI invocations see the same budget. Hypothetical actions are excluded from every real consumption view.

`orx execute --episode ep1` and `orx induce` automatically record their actions in the unified log: execute as a macro (`pre` snapshot before execution, `post` after, `rollup=reference`, linked to the execution id); induce in the **maintenance scope** (`__maintenance__` / `maint_<ts>`) with a real pre/post knowledge state, verification results, a knowledge delta, and a business result that separates `created` (verified) from `created_unverified` (a candidate is NOT knowledge growth), `updated` / `revised` / `refused` / `unchanged`. A crashed induction still leaves a `failed` action with its pre state. Dry-run persists nothing.

**Next.** `close-episode` when the episode ends. Failures: `unconfirmed` means spend is within limits but a declared dimension is unknown — it is NOT confirmation within budget; `exceeded` names the measured dimension over the limit.

## Stage 6 — Close out and calibrate

### `orx close-episode --task ID [--episode ep1] [--terminal completed|failed|aborted|budget_exhausted] [--finish-action ACTION_ID] [--min-samples N]`

**Episode close-out.** Closes ONE episode: evaluates its bound strategy-outcome predictions against their real outcomes (field by field: benefit error under the prediction's own declared metric/baseline, per-dimension cost error where both sides measured the same scope, Brier scores for labelled risk events, interval coverage) and publishes the experience calibration summary that later episodes' prediction contexts read. Reads what was recorded — no solver run, no model call, no induction. Unfinished actions are reported (their predictions stay pending, never fabricated into endings); a failed/aborted/budget-exhausted episode closes honestly under its own terminal state. Idempotent: re-closing returns the stored record and counts nothing twice.

**Unknown cost dimensions are warned about, never zeroed.** When any of the task's records still carry an UNKNOWN `llm_tokens` or `tool_calls` (no host report, no backfill), the result carries `cost_completeness_warnings = {unknown_dimensions: {dim: [execution_ids]}, n_records, note}`: calibration already accepts ONLY measured dimensions, so the warning makes the gap visible at the moment it can still be repaired (`amend-cost --usage-file`/`--usage-host`) instead of surfacing later as a silently shrunken sample. Closing is never blocked — the fact is honest as it stands.

**A close-out is the end of the episode, NOT a certification of the answer.** The result carries `task_checks = {n_executions, verdicts, unchecked, note}` reporting how many of the episode's executions carry a task-result verdict (`orx check-task`). Under the observation rules (`wm-obs/2`) the quality observation is ALWAYS the solver's own figure (how well it solved the model), and a confirmed task failure travels as a SEPARATE fact under `evaluations[].benefit.task_check` — the quality number is never rewritten to `0.0`, so "predicted 0.8, observed 1.0" plus `task_check.state="failed"` reads as "the solver solved its model, but the answer failed the task". The completion question is answered by the `effective_completion`/`task_result_check_passed` channel instead. Unchecked answers keep their observed value and the note says their validity is UNKNOWN. Full spec: [episode_closeout.md](episode_closeout.md).

**Late corrections change later USE, not history — and they are FIELD-SCOPED.** A task check run on an already-closed episode writes the verdict onto the fact, but the STORED evaluation is never rewritten: it is the honest record of what was known then. The calibration uses a LIVE re-derivation instead, reported in `validity_corrections[]`:
- a check that now FAILS moves the SEPARATE `task_check` fact while **keeping the measured quality observation AND the measured cost** (`kind: "live_rederivation"`, `fields: ["benefit"]`, `counted: true`) — a wrong answer still cost what it cost, and its solver-side quality is unchanged;
- a check that was WITHDRAWN restores the un-gated state;
- an execution WITHDRAWN from the evidence set (`exclude-execution`) removes the sample entirely (`exclusions.validity_corrected`);
- the correction is scoped to the executions the evaluation actually compared, so a late verdict on one execution does not disqualify a sibling's evaluation.

When the affected episode is still in the calibration WINDOW the published summary is REPUBLISHED as part of the correction (the result carries `calibration_republished`), so later predictions see the corrected judgment without waiting for a manual rebuild.

**Next.** Read `result.task_checks` for coverage, then run the offline operations if you are considering maintenance. Failures: a refusal with `state: "pending"` means an action is still `running` — end it and close again; a re-close after an interrupted publication reports `recovered_publication: true`.

### `orx calibration [--min-samples N] [--rebuild]`

Read the published strategy-outcome experience-calibration summary (wm-calib/2). Aggregated from the WINDOW — the newest N closed task-episodes (default 50, `OR_CALIBRATION_WINDOW`) — so this is a single-row read of the last published object, not a history scan. Groups are keyed by `strategy_outcome|<model identity>|<metric>|<unit>|<scope>`: a different model/metric/unit/scope is a different group and never pools. Each group reports directed statistics (`mean_benefit_signed_error`, `mean_cost_log_ratio`), interval coverage AND width, per-event Brier means, and `mean_predicted_probability` alongside `scored_occurrence_rate` (same denominator). The summary also carries `occurrence` — the framework's own event counts by OBSERVATION UNIT (`n_observation_units`, `n_occurred`/`n_not_occurred`/`n_unknown`, `unit_occurrence_rate` over its own denominator), which must never be subtracted from a mean probability computed over a different sample set. Below the sample minimum a group reports `insufficient_evidence` with `reliability: null`; no figure is invented. `applicability: "global_diagnostic"` states plainly that these are global statistics with no per-strategy breakdown. This is the OR strategy-outcome reliability, kept separate from the legacy knowledge-prediction reliability, and a measured record — not a promise that future predictions improve.

`--rebuild` rebuilds and republishes the summary from the current window instead of reading the published one: the explicit migration path for a store that predates this version, and the repair path for an unpublished summary.

**Next.** The published summary is frozen into NEW prediction contexts automatically; no further call is needed. Failures: a group with too few samples reports `insufficient_evidence` with a null reliability — an honest unknown, not a bad score.

### `orx archive-calibration [--dry-run]`

Move OUT-OF-WINDOW episode detail to the archive (retention). Only DETAIL moves — evaluation, strategy-outcome prediction and frozen-context payloads — into `{home}/archive/calibration/calibration-NNNN.jsonl`; the close-out registry tombstone stays ONLINE so a repeated close remains idempotent and the window stays locatable. Episodes with an UNCHECKED execution are held online for the late-check grace period (`OR_CALIBRATION_LATE_CHECK_GRACE_DAYS`, default 30) so a late verdict can still land; an episode whose executions all carry a verdict is not held. The archive is bounded by THREE caps — per file (`OR_CALIBRATION_ARCHIVE_MAX_FILE_BYTES`, 64 MB), total (`OR_CALIBRATION_ARCHIVE_MAX_TOTAL_BYTES`, 1 GB) and age (`OR_CALIBRATION_ARCHIVE_RETENTION_DAYS`, 365) — whichever bites first evicts the oldest file, so the archive cannot grow without bound. Re-running is idempotent. A restored payload never re-enters the calibration automatically.

**`--dry-run` reports; it never writes.** In particular it does NOT perform the legacy-store registry migration, so previewing an old store cannot silently register every historical close-out. On a pre-registry store the preview therefore reports an empty window plus `pending_migration` naming the explicit step that does perform it (`orx calibration --rebuild`, or one real archive pass). The same discipline applies to `inspect --bank retention`: a read does not migrate.

**Next.** Re-run without `--dry-run` to apply it. Failures: `--dry-run` never writes, so a pre-registry store previews as an empty window plus a `pending_migration` note.

### `orx enforce-window [--window-episodes N] [--open-grace-days D] [--dry-run]`

Bound the Execution Evidence Bank to a recent window of COMPLETE episodes (default 800, `$OR_EVIDENCE_WINDOW_EPISODES`). The eviction UNIT is the whole episode: the oldest episodes leave at once, so a contrast/repair chain is never split. Order is by the close-out registry's `closed_at`, the same ordering the calibration uses.

Three things are NEVER evicted: episodes inside the calibration window (they calibrate), episodes awaiting a task verdict within the late-check grace period (a late check must still land), and young UNCLOSED episodes. An unclosed episode is exempt from the count bound while its outcome is unknown, but it is not exempt forever: `--open-grace-days` (default 30, `$OR_EVIDENCE_OPEN_GRACE_DAYS`) bounds an episode that never closes, and it is evicted as a whole and reported separately (`evicted_unclosed`). A single episode larger than the budget is KEPT whole and reported, because the bound is a count of episodes and nothing promises a byte budget.

The pass also drops the evicted vector items and any task-text version no remaining execution references, so the bank and its index shrink together. It is idempotent, crash-recoverable (re-running finishes an interrupted pass) and NEVER touches your solve sources.

**An eviction is a source reference EXPIRING, never a refutation or a withdrawal.** A knowledge entry keeps its own content, conditions, expected effect and verification scope; the expired reference is simply historical. `exclude-execution` remains the explicit correction channel (a withdrawal), and the two are never conflated.

In `close-episode` the same pass runs automatically, but LAST — after the calibration publish and the archive pass — and is guarded by one indexed COUNT, so an ordinary close walks nothing. Run the command directly to force it (e.g. after importing history).

**`--dry-run` reports; it never writes** (no row, no vector, no text).

**Next.** Nothing to chain; the bank is now bounded. Failures: none — a run that evicts nothing is a valid answer.

### `orx migrate-relations [--dry-run]`

One-way migration for a store written BEFORE knowledge was unified (when an entry could carry a `relations` list beside its own statistical claim). Each relation becomes its OWN claim entry, its verification block copied **verbatim** — a migration never widens what was verified. A host whose only content was its relations is removed (its knowledge now lives in those entries); a host with a statistical claim is kept. The mapping `old_entry_id -> [new_entry_id, ...]` is recorded in meta for audit.

Idempotent: re-running creates nothing more (an already-migrated entry has no relations left, and the identity map is a second guard). **`--dry-run` reports and writes nothing.**

**Next.** Nothing to chain. Failures: none — a no-op on a store with no legacy relations.

### `orx inspect --bank experience|strategic|archive|actions|snapshots|predictions|texts|evaluations|retention|capability [--task ID] [--strategy S] [--status candidate] [--episode EP] [--evaluation ID] [--prediction ID]`

**The ONE read entry point** — every query lives here, so an agent has a single place to look and a single answer shape to parse.

- `experience` / `strategic` / `archive` / `actions` / `snapshots` / `texts` — the stored layers, filtered as named. Strategic entries include `prediction_track {n_predictions, hit_rate, calibration_error, consecutive_misses}`, `status` (candidate|validated|suspect|dormant), `provenance` (execution ids). `--status suspect` / `--status dormant` is the **retirement-candidate view**: nothing is deleted by asking, and retiring is explicit (`orx retire`). `actions` / `snapshots` query the world-model substrate (below).
- `predictions` — listing mode returns EVERY prediction generation under one roof, each under its own key (`legacy_predictions`, `strategy_predictions`, `capability_predictions`). They are separate logs answering separate questions — only the `strategy_predictions` (wm-so/1) channel feeds `orx calibration`.

With `--prediction ID` the command does a **KEY LOOKUP**: it returns that ONE record (plus the generation it resolved to, under `kind`) and nothing else — no log listing is attached. A key lookup costs the key, not the history.
- `evaluations` — the stored post-hoc strategy-outcome evaluations (`--task` / `--episode` filter; `--evaluation ID` reads one). Excluded evaluations are neither hits nor misses; pending ones wait for their scope to end.
- `retention` — the calibration retention scopes (window / late-check grace / archive caps) plus the online and archive counts. The CALIBRATION window is bounded here; the RAW evidence bank is bounded separately by the evidence window (`orx enforce-window`), and the two are reported as different scopes — the evidence bound covers the Evidence Bank, NOT the whole project directory.
- `capability` — the two-stage capability feedback state (fact bound vs effect verified). With `--prediction ID`, one prediction's full state: the prediction, its maintenance binding and its effect evaluation.

**Next.** Depends on the bank you read: `predictions --prediction ID` and `capability --prediction ID` are single-record lookups, while the listing banks answer "what does this memory hold". Failures: a key lookup returns that record only, with no log listing attached.

### `--bank texts`

`--bank texts` is the **retrieval source documents** — the task-text versions captured on the write paths, keyed by `(task_id, text_digest)`. It is not a knowledge bank: it makes no claim, feeds no statistic, and has no lifecycle; it exists so the embedding index has a source and so a memory with no vector is still findable. Use it to inspect **which text version an execution was produced under**: every execution carries `task_text_digest`, and `--task ID` lists that task's versions. With `--task` omitted, every retained version is listed. This is the documented look-up entry point for the records counted under `vector_recall.unindexed`.

## Stage 7 — Offline maintenance

### `orx review-material [--strategy S] [--task T] [--limit N] [--cursor C]`

**The material entry point (no model call, no writes).** Reads a BATCH of completed tasks straight from the evidence bank — no detector candidate and no sample-count gate — so new, failed, cross-cell and cross-method-name material all reach you. Per attempt it returns a COMPACT entry: `task_text` (an excerpt, or an explicit `unknown` marker when the version is not retained), `problem` (a COUNT-based summary of the profile and CIR — the full joint representation stays on the record, re-readable by id), `method` (`planned`/`actual`, steps clipped, with `basis`), `outcome` (status, objective, gap, and the executed `code_hash`), `task_check` (state, basis, and `reference_source` — who supplied the reference; absent reads as `never_checked`, never a pass), `failures`, clipped `trajectory`, `cost`, and `attempts_of_task` / `independent_task` (a same-task retry is marked `false`). When a previous same-task attempt exists, a `changes` block reports the code-hash and planned-method differences as FACTS — the framework does not interpret what caused them.

`material` keeps EVERY attempt (each failure with its own cost), `task_chains` groups their execution ids per task in order, `existing_knowledge` lists the entries that RELATE to the batch (by strategy id, by CITED execution, or by shared family) with `related_by` reasons, and `cross_task_hint` reports how many DISTINCT tasks the batch spans (plus the history count) so you look across tasks for a recurring mechanism instead of restating one. `budget` reports chars used/limit, `omitted_execution_ids`, `truncated_by_limit`, and `next_cursor` — pass `next_cursor` back as `--cursor` to read the next (older) batch, so a long history is walked in distinct batches rather than the same newest records being re-shown. Missing fields are marked `unknown` individually; one unknown never drops the rest of the fact.

`--strategy` / `--task` narrow the batch, `--limit N` keeps the N most recent attempts, and `OR_HARNESS_REVIEW_MATERIAL_CHARS` (default 24000) bounds size. Read the material, form a claim (condition -> how -> consequence -> boundary), then submit it with `induce --relation`.

**Next.** `orx induce --relation '{...}'`; `orx induction-candidates` / `orx induction-material` are an optional cell lead.

### `orx induction-candidates`

**A cell-LEAD view, not the material gate (no model call).** Freezes structural-cell candidates from current evidence: a cell with ≥2 executions / ≥2 distinct tasks is a `new_claim` or `revision`; a THIN cell or a single task's repeated runs is a `cell_observation` carrying an `admission_note`. A thin cell is VISIBLE, never dropped. Every bundle also freezes the METHOD material of its evidence (`methods[]`). Bundle ids are content-addressed, so an id stays valid across calls over unchanged evidence.

This makes NO model call and changes NO knowledge. Pass a bundle to `orx predict-capability --bundle` to price an offline operation against its OWN evidence scope, or read it with `orx induction-material`. An empty result means no cell lead exists — it is NOT "nothing to review": `orx review-material` still reads the batch directly.

**Next.** `orx review-material` (default) or `orx induction-material` (a lead), then `induce --relation` to submit your claim — or `predict-capability --operation op.json --bundle <bundle_id>`.

### `orx induction-material [--bundle BUNDLE_ID] [--strategy S]`

Organizes the READABLE material for a structural-cell LEAD: per candidate, the recorded methods (planned and performed), the outcome, the task-check state, and the verification state of each cited execution. No model call, no writes — it is what you read before writing a claim in your own words.

Each candidate reports `material_report` — a REPORT, not an admission verdict: `basis` names the strongest method content present (`performed` / `planned_only` / `none`), the counts (`n_supporting`, `n_with_method`, `n_planned_only`, `n_checked_pass`) are stated, and `missing` lists the content that is absent (`method_performed`, `task_check`). The framework does not decide whether "there is enough to abstract"; you read the material and decide. It never derives a technique from a number: record how the work was actually done (the solve script's `method_performed` receipt is the observation; `execute --method` is only the plan).

**Next.** Form the claim (condition -> how -> consequence -> boundary) and submit it with `induce --relation`, citing the executions (or the `bundle_id`) you were shown.

### `orx induce [--strategy S | --all] [--family F] [--cell GROUP_KEY] [--relation JSON] [--dry-run] [--force] [--note TEXT] [--verify JSON]`

Consolidates facts into Strategic Knowledge (your explicit call — hints never auto-induce). This is the ONLY place knowledge changes: it (i) builds or refreshes the claim of each (family, structural cell, strategy) evidence set — its applicability is read off the supporting records (the cell its evidence occupies, never a cross-sample span that could stretch across incomparable regions); and (ii) replays the frozen checks recorded on the facts, reporting what it did under `result.revisions` (promotion at ≥5 checks / ≥70% hits **with a verified claim**, demotion at 3 consecutive misses, dormancy wakeup). New entries are born `candidate` with honest intervals (width floored by sample size; n=2 cannot claim [0.95, 1.0]). One evidence set owns exactly one claim: re-inducing refreshes it rather than restating it (dormant entries included, so a woken claim keeps its id). There is NO full-bank rebuild: an existing claim is refreshed IN PLACE from currently retained evidence (`--all` covers every cell), which is how drift is repaired without replacing long-term knowledge with a recent-window restatement. The framework fills in NONE of an entry's `strategy_type` / `actions` / `fallback_strategy_id` — there is no built-in directory to copy them from, and inferring a method's actions from its id would be fabrication. What the harness writes is what travels; an empty field means the memory does not record it. `--dry-run` never writes — including under `--force` (a rehearsal does not lift a cold-archive veto).

**Independent evidence gate (creation only).** Creating a claim takes at least 2 supporting executions from ≥2 distinct `task_id`s: repeating one task is repetition, not reproduction. Every count comes from the same facts — executed, attempt-scope, in the target's structural cell — so a task-scope total can never stand in for an independent attempt. When only one task is behind the evidence the call creates nothing and returns `verification {tasks, required_tasks}` plus a `skipped` reason — the evidence is NOT discarded (recall still answers from it as `conditional_stats`), and refreshing an entry that already exists is never blocked, no matter how repetitive the new evidence is.

**`--verify JSON` publishes the candidate.** Admission and creation are separate: the gate above decides whether evidence may become a *candidate*; this decides whether the candidate's claim may become *published knowledge*. See [induction.md](induction.md#admission-verification-offline-what-makes-a-claim-published) for the payload and the three checks (`rule` / `repair` / `cost_saving`). The framework computes the verdict from the executions you supply; without it the entry is `unverified` and **is not published** — recall answers from `conditional_stats` and `predict_cost` will not quote it.

**`--family` / `--cell` scope the check to ONE structural unit.** A verification payload is written for a specific claim, so it must apply to a specific target set — one payload applied to every cell of a strategy would cross-contaminate families (an allocation check overwriting a scheduling verdict). `--family F` restricts induction to that family; `--cell GROUP_KEY` restricts it to one structural cell (the full `group_key` token, e.g. `family=routing|rc[..]|tc[..]|rx[..]`, compared against each record's DERIVED key so a stale index column never mis-selects). Either flag implies the `--all` scope — naming a unit is itself an explicit selection — and composes with `--strategy`.

What the verdict requires, concretely: a declared criterion the framework can evaluate itself (`reference_status`, `reference_objective`, `semantic_probe`) or a genuinely independent comparison — **feasibility alone is a precondition, not a check**, and a bare `semantic_ok` boolean is recorded as `agent-declared` but cannot carry a verdict. Every execution you pass is evaluated (a counterexample anywhere in the batch refutes, whatever the order), the evidence must be for THIS strategy and family, and a comparison must be like-for-like (same task, same measurement scope for costs; a repeated execution id is not an independent comparison). Failed executions yield `insufficient_evidence` (not `refuted`): "we could not check it" is not "we checked it and it failed".

Each element of `result.results` reports `created` (entry id) / `updated` (entry id) / `skipped` (human-readable reason: fewer than 2 executions, needs independent evidence, cold-archive veto, restatement, or `recorded as an unverified candidate`), plus `verification` when a check ran — read the reason, it tells you what the memory is still missing.

**Cost claims are complete-or-silent.** A cost dimension enters the entry's claim only when EVERY supporting record measured it: a mean over a subset ("3 of 4 records reported tokens") would be a partial observation published as a full claim, and it feeds strategy selection and world-model prediction as if it were the whole truth. When a dimension falls short, the entry is still created and its complete dimensions are still published, but the incomplete one is withheld and reported under `cost_claim_withheld {dimensions: {dim: {n_measured, n}}, supporting_executions, note}`. The fix is a backfill — `orx amend-cost <execution_id> --override <dim>=<value>` amends the fact in place, nothing is re-run — after which the next `induce` restores the claim.

`--note "TEXT"` (optional, repeatable, you phrase it): free-text applicability notes attached to the entries this call creates or refreshes. They are kept for the reader and shown by `inspect`; they never enter scoring — the framework does not pretend to verify a sentence. Because these notes are part of an entry's retrieval document, the knowledge index items are refreshed after this call (`result.index_sync`, best effort — same `synced`/`deferred`/`skipped` contract as `record`). `--dry-run` writes nothing, index included.

**A comparison is submitted as a claim, not written as prose.** Induction no longer takes peer flags: a contrast against another strategy (same cell) or another cell (same strategy) is a STRUCTURED knowledge claim — read the material with `orx induction-material` and submit it with `--relation`, citing the executions on both sides. The statistical path records no free-text contrast: a sentence the framework cannot check is not knowledge. Entries that already carry such text keep it; nothing writes new lines.

**`--relation JSON` submits a STRUCTURED knowledge claim** (repeatable). This is the path for cross-task knowledge that is **not** one strategy's statistics — a modeling principle, a necessary condition, a repair pattern. Payload:

```json
{"subject": "principle:cross_period_state",
 "claim": "the sentence being asserted",
 "evidence": [{"execution_id": "ex_..", "role": "dropped"},
              {"execution_id": "ex_..", "role": "preserved"}],
 "conditions": {"predicates": {"family": "scheduling",
                               "temporal_coupling": [0.5, 1.0]}},
 "check": {"assertions": [...]},
 "kind": "rule"}
```

`claim` and `evidence` are required; every evidence entry needs either an `execution_id` for a **recorded** fact (or a `bundle_id` from `induction-candidates`, expanded into that bundle's frozen evidence set) and a `role` naming the part it plays in *this* claim (free strings — `dropped`/`preserved`, `before`/`after`, `strategy_a`, …). The framework DERIVES `tasks`, `family`, structural cell and `strategy_ids` from those facts, so a caller never submits a second, contradictory identity. `subject` is optional and matters for knowledge that names no strategy id: the claim creates a **claim-only entry** (`support_n = 0`, no statistical claim) whose `strategy_id` is the subject. ONE entry is ONE claim — identity is `strategy_id` + structural cell + `kind`, so a second independent claim (a different cell or kind) is a SEPARATE entry and never inherits the first's verdict. `conditions` are the applicability predicates (omitted → read off the evidence's cell). `kind` is an optional note about what prompted the claim — never a verification template.

A relation submission is a knowledge WRITE like any other: it records a maintenance action with the pre state, a knowledge `{entries_created, entry_changes}` delta, the index result, and the same knowledge feedback — so a relation is as auditable as a statistical induction, and re-submitting the same claim updates rather than duplicating. When the cited evidence reports NO method and the claim declares none, the outcome carries a non-blocking `material` warning: the claim is saved as written, but the framework did not (and will not) derive a technique from the numbers.

Verification reuses `--verify` with `purpose: "relation"`; each assertion carries its own semantics (`probe` / `status` / `comparison`). `comparison` declares `metric`, `roles_a`/`roles_b`, `direction`, `min_gap`, `mode` (`paired` compares only same-task pairs, one per side; `group` compares the sides' means over a metric every referenced record measured), and `aggregation` (`all` = every pair must meet the gap, one comparable counterexample refutes; `mean` = the batch mean must). Pairing is per assertion, never a property of the whole batch. The verdict is `verified` only when every declared assertion held — "no violation found within this scope", never a guarantee about future tasks; `insufficient_evidence` covers an unmeasured metric, a missing counterpart or nothing computable declared (**not** a refutation); `refuted` means an assertion ran on real evidence and failed. Only the declared parts are covered: "quality higher AND tokens lower" needs both assertions.

**The declared checks may live in EITHER place — the embedded `check` block above OR the standalone `--verify` payload — and both are read.** An embedded `check` is never silently ignored. When BOTH are supplied, `--verify` wins and the outcome records `check_note {check_source: "verify_arg", overridden_embedded_keys, note}` so the override is visible; when only the embedded block is present the outcome records `check_note.check_source: "embedded_relation_check"`.

**Publication is per relation**, on two conditions: its own verdict is `verified` (and not stale), and its verification scope covers ≥2 distinct tasks. A single-task relation is SAVED and verifiable as a fact about that task, but reported as not published (`result.relations[].publication.reasons` says why). Neither condition touches the host entry's statistical claim, and the host's admission never grants the relation anything. Re-submitting the same `subject`+`kind` REVISES that relation (a substantive change without a fresh verdict marks it `stale_after_revision`; a fresh verdict wins; an identical re-submission keeps the verdict). Result: `result.{relations[], saved, published}`. See [induction.md](induction.md#structured-relation-claims-induce---relation).

**Next.** `inspect --bank strategic` to read the result, then `predict-capability` if you plan a further change. Failures: a `skipped` reason names what the memory is still missing (fewer than 2 executions, no independent task, cold-archive veto, or an unverified candidate); `--dry-run` writes nothing, index included.

### `orx [--world-model URL::MODEL] predict-capability --operation JSON [--task t.json] [--bundle bundle.json] [--horizon TEXT] [--horizon-tasks N] [--budget JSON] [--task-id ID] [--episode ep1] [--timeout S]`

**Capability-evolution prediction (`wm-ce/1`).** Predicts what ONE offline learning operation (`--operation` JSON: `operation_type` of `induce`/`revise`/`reverify`/`retire`, plus `strategy_id`/`config`) would change in FUTURE task performance: `expected_changes` (metric, unit, direction, value, baseline), `learning_cost`, `degradation_risk`, `uncertainty` and `verification_conditions`. The FRAMEWORK fixes the operation identity, the experience scope (from the bundle's own executions), the task targeting, the horizon and the per-metric baselines; the model fills prediction content only and may CITE a frozen baseline, never set one. Cost references are frozen **per dimension** (`cost:solver_runtime_s`, `cost:llm_tokens`) because several dimensions share the `resource_cost` metric while their units do not convert — a relative cost change is converted through the reference for ITS OWN unit, and a cost change that names no unit gets none. A failed call is a persisted failure with whatever usage it consumed. `status="contract_only"` with no provider; the service IS implemented, so `prediction_made` is what makes it `valid`.

**Next.** `compare-capability --predictions` once two or more predictions exist. Failures: `contract_only` means the provider produced no forecast — the service is available but nothing was predicted.

### `orx compare-capability --predictions ID[,ID...] [--horizon-tasks N] [--allow-quality-loss]`

Compare frozen capability predictions and recommend one, or `defer` (read-only: no operation runs, no knowledge changes). ONE bounded rule: among candidates predicting a quantified resource saving with no quality degradation, recommend the largest **net saving over the declared window** — the CUMULATIVE saving minus that candidate's OWN ONE-TIME predicted maintenance cost **in the same unit**. The cost is paid once while the saving accrues per task, so a candidate that costs more than a single task saves is still recommended when the window pays it back. A candidate whose learning cost is quoted in another currency, whose saving does not pay back within the window, whose window was never declared, whose operation type has no execution path in this build, or whose savings cannot be compared across units is reported as `incomparable` rather than ranked. `defer` / `insufficient_evidence` are legitimate outcomes.

**Next.** `accept-capability` or `reject-capability` with the printed `result` — the choice is explicit and yours. A tie or a quality-degrading option may yield no recommendation, and deferring is a legitimate outcome.

### `orx accept-capability --recommendation JSON [--prediction ID] [--verify JSON] [--note TEXT] [--force]`

**EXPLICITLY** accept a recommendation and run the real offline operation on the prediction's OWN frozen scope (the only offline entry that changes knowledge). The operation TYPE decides what runs: `induce`/`revise` go through the existing induction path, `retire` removes the named target entry, and any other type is REFUSED — a candidate is never executed as a different operation under its name. A scope is never widened by re-reading the current bank.

**An operation runs what it DECLARED.** A candidate may carry the relations its operation forms, in `operation.config["relations"]` (each an `induce --relation` payload). When it does, acceptance submits exactly those relations — the operation the agent described is the operation that runs, instead of a generic statistical induction that leaves the technique unwritten. With no declared relations the statistical refresh runs as before (creating/refreshing the cell's quality/cost claim). The knowledge delta reports whichever happened, so the caller can see that a relation was created or revised, not merely a mean.

**Next.** Run `evaluate-capability` later, once qualified tasks have accumulated. Failures: the operation runs on the prediction's OWN frozen scope and an operation type this build cannot carry out is refused; the maintenance fact is already bound on `result.maintenance_binding`, so do not call `induce`/`retire` again.

### `orx reject-capability --recommendation JSON [--prediction ID] [--reason TEXT]`

Explicitly decline or defer a recommendation: no operation runs and NO knowledge changes. The decision is recorded with its reason so the history is auditable.

**Next.** Nothing further for this decision: rejection has no side effect and the prediction is left unbound.

### `orx bind-capability --prediction ID [--adoption-action ACTION_ID]`

**Stage 1 of the capability feedback.** Bind the REAL maintenance FACT: whether the operation happened, what knowledge actually changed (created / revised / **retired** entries — a retirement IS a knowledge change), the admission verdict of what it produced, the REAL cost, the operation's own end time, and whether the scope used matches the scope predicted. Idempotent by prediction. It CANNOT set `effect_verified` — a verified entry is a knowledge change, not evidence that future performance improved.

An ONLINE capability gain is bound the same way: a strategy-outcome prediction that claimed an H+ is archived as a trace, and `bind-capability --prediction sp_...` binds the real attempt it was made for (no `predict-capability` round is needed). `result.prediction_source` says which kind the id resolved to (`stored` / `online_trace`).

**Next.** `evaluate-capability --prediction ID` once later real tasks exist. Use this only when the maintenance action was performed outside `accept-capability`; otherwise the fact is already bound.

### `orx evaluate-capability --prediction ID [--tasks ID,...] [--paired JSON] [--allow-descriptive]`

**Stage 2 of the capability feedback.** Judge the prediction against REAL later-task results, or record a pre-arranged paired comparison. Only CLOSED episodes of tasks whose WORK really ran AFTER the operation — its real execution time, not its close-out timestamp, and under the knowledge entries the operation produced — fall inside the prediction's FROZEN targeting and are NOT part of its own experience scope participate. The sample is counted in **TASK-EPISODES**, not prediction records: ten predictions bound to one execution are ONE independent truth and cannot satisfy a ten-task horizon. The declared horizon must be reached or the result stays `pending`. A paired record is READ AND USED (its treated-minus-reference difference is the observed change) — merely existing is not attribution, and a pair taken on another metric/unit, or citing a task that never ran, is reported as unusable. `observed_improvement` is the only state that sets `effect_verified`; a pending horizon, a descriptive movement, insufficient evidence and a refutation are all first-class outcomes, and only a FINAL verdict short-circuits a repeat call.

An online H+ (`sp_...`) is evaluated through the same call, materialized on demand from its trace: the claim, target and conditions are the ORIGINAL ones (the framework never re-predicts after seeing the result), and an unlearned or unexecuted gain stays `pending`. `result.prediction_source` reports `online_trace`.

**Next.** Nothing further — the verdict is the end of the maintenance loop. Failures: an insufficient sample base reports the deficit; `--allow-descriptive` downgrades the claim explicitly rather than silently widening it.

## Stage 8 — Correct the record

### `orx exclude-execution --execution EXECUTION_ID --reason "..." [--superseded-by EXECUTION_ID]`

Withdraw a **wrong execution fact** from the evidence set. A retained fact is never rewritten and a withdrawn one is never deleted — it must stop counting. The row is preserved for audit, its `source` becomes `excluded`, and the reason is recorded on the fact under `execution_features.correction`. Every statistics / induction / trigger / retrieval path requires `source == "executed"`, so the fact drops out of **all** of them at once; its vector is removed from the execution index immediately. `--superseded-by` names the corrected re-run that replaces it (a link, never an inference — the correction is always your explicit statement). Derived layers pick the change up at the next `orx induce`. This is the ONLY way to un-count a fact: `record` never rewrites, and the fact's `execution_id` stays visible in `inspect --bank experience`.

**Next.** `calibration --rebuild` if the episode has left the window, or rely on the automatic republish while it is still in it. The row survives and stops counting, and `restore-execution` reverses it. (This is a WITHDRAWAL — distinct from the evidence window's eviction, which merely lets a source reference expire and never revokes knowledge.)

### `orx restore-execution --execution EXECUTION_ID --reason "..."`

Reverse an exclusion — a second explicit statement, because an exclusion can itself be wrong. The fact counts as evidence again (source back to `executed`) and the correction history keeps both decisions. Re-indexing is deliberately NOT automatic: run `orx rebuild-index --layer execution` to re-embed it, then `orx induce` to refresh the derived layers.

**Next.** The same republish rule applies; the fact counts again and the occurrence tally is re-derived from current facts.

### `orx retire --entry ID --reason "..."`

Your explicit, irreversible confirmation: moves an entry to the cold archive. Its vector leaves the retrieval index with it, so a later `recall` can never surface the retired advice.

**Next.** Nothing — retirement is irreversible. Use `inspect --bank strategic --status suspect|dormant` to find candidates before acting, and reserve this for genuine drift or dead ends.

## Stage 9 — Retrieval index

### `orx rebuild-index [--layer both|execution|strategic] [--dry-run]`

Explicit retrieval-index maintenance — the only command that embeds in bulk and the only place index vectors are (re)created in volume. It is needed in exactly three situations:

1. **First build.** Until an index exists, `recall` reports `degraded` ("index missing; run `orx rebuild-index`") and uses the structural channel only.
2. **Embedding-model change.** An index built by another `model_id` is refused WHOLESALE — vectors from two models are not comparable, so it is never partially reused.
3. **Settling a deferred sync.** If an embedding call failed during `record` / `induce`, that memory is unindexed (`result.index_sync.state = "deferred"`) until a rebuild picks it up.

Not a routine path: `record` refreshes the execution item, `induce` refreshes the knowledge items, and `retire` drops the retired vector.

`--dry-run` counts what would be indexed and **writes nothing** — no embedding call, no index file, not even the index directory. Content that cannot be indexed is reported as `unindexable` (a record whose text was never captured is not indexed). The index is derived data: rebuilding it never changes a fact or an entry.

Result: `{dry_run, layers: {"execution_evidence"|"strategic_knowledge": {items, model_id, dimension, unindexable}}, backend}`. Without a configured backend and without `--dry-run`, the command fails with exit code 2 and an explicit message rather than silently doing nothing.

**Next.** Re-read `recall`, or check health with `doctor`. Failures: `--dry-run` counts what would change without writing; a model change invalidates the old vectors, so the rebuild is required rather than optional after switching embedding models.
