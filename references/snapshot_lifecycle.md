# Prediction inputs, frozen facts, and explicit maintenance

OR-Harness keeps two memory layers: execution evidence and strategic knowledge. Prediction inputs are temporary working data. Predictions, calibration and runtime facts support those layers; they do not accumulate another full view of the memory banks at every action.

## Responsibilities

| Object | Stored content | Lifetime |
| --- | --- | --- |
| In-memory `BeliefSnapshot` | The current P/X/H/B view assembled for prediction | Caller-owned temporary object |
| `frozen_conditions` | Problem/progress value references, identity, real/hypothetical source, compact budget aggregates and knowledge-availability revision | Small facts used by action/prediction references; no coverage, knowledge bodies or bank statistics |
| `condition_values` | Immutable, content-addressed problem/progress values and actually referenced evidence, knowledge, targets, feedback and prompts | Shared historical values; no full-library content versions |
| `prediction_contexts` (`wm-context/4`) | Small audit manifest, frozen condition reference, selected content versions and input digest | Survives release of the full working input; existing calibration archival can move manifests |
| `runtime_inputs` | One candidate-related working input per decision, shared by its candidates; large problem content uses a shared version | Released after `choose-next` or successful episode close; expired inputs are removed on subsequent context writes |
| `knowledge_availability` | Entry identity, predicate/presence changes and revision order, without knowledge bodies | Historical applicability index for H+; availability is separate from adoption and effectiveness |
| Execution/action records | Real attempts, outcomes/checks, costs/masks, adoption/reuse versions, condition and prediction links | Existing evidence/episode policies apply |
| Published calibration, pairs and individual evaluations | Existing independent calibration/feedback records | Existing windows, eligibility, observation versions and archive rules apply |

`snapshot()` still returns a current in-memory view. It persists only a lightweight condition receipt. `get_snapshot()` reads that receipt with `coverage.conditions_only=true` and a missing-full-view explanation, or reads a complete legacy snapshot if that is what was stored. It never rebuilds historical knowledge from today's bank. `snapshot(..., persist=False)` and `context --no-persist` do not persist prediction state.

Action pre/post IDs and prediction `input_snapshot_id` remain usable as condition references, so old clients can resolve identity, progress and budget facts. New rows are not appended to `belief_snapshots`. Task progress uses the problem identity: writing the post-selection model preserves progress, while changing the actual problem resets it.

Execution evidence remains one real attempt per record. `method_planned` is never promoted to `method_actual`. A missing adoption declaration differs from `used_entries=[]`; available, displayed, adopted and effective are separate facts. New adoption and reuse declarations reference the content observed before the attempt. Missing old versions remain unknown.

Scripts, result files observed after attempts, captured output clips and internal CIR artifacts can be referenced under `HOME/artifacts/<prefix>/<sha256>`. Paths are relative to that home and carry SHA-256 and byte counts. A result file reference does not certify the result; task/solver checks still determine acceptance. Output clips may be truncated. Artifact I/O failure is reported without turning an observed execution into a fabricated failure.

## Context reuse and expiry

```bash
# Explicit context creation is optional; plan-next normally creates one itself.
orx --home /work/run context --task task.json --episode e1
orx --home /work/run context --context-id ctx_ID
orx --home /work/run predict-strategy --task task.json --candidate candidate.json --episode e1 --context ctx_ID

# Available after consumption/expiry as well as while active:
orx --home /work/run context --context-id ctx_ID --audit
```

During the valid lifetime, reuse sends the original stored input and never reruns recall against today's memory. Default limits:

- `OR_HARNESS_CONTEXT_TTL_S=86400`: input reuse lifetime, fixed when persisted.
- `OR_HARNESS_ACTIVE_CONTEXTS=128`: maximum active input count in a home.
- `OR_HARNESS_ACTIVE_CONTEXT_BYTES=16777216`: active JSON payload capacity; large shared problem versions are not repeatedly counted as inline runtime payloads.
- `OR_HARNESS_CONTEXT_CALL_LEASE_S=3600`: protection for a provider call in progress; a crashed call cannot pin its working input forever.

Capacity pressure explicitly expires older unleased inputs. If a new input cannot fit while other calls are in progress, it is refused. Active calls are protected until their call lease expires; provider network I/O holds no SQLite write transaction. Cleanup runs on context writes and at consumption boundaries; there is no background timer.

`choose-next` releases the comparison input after recording the choice/rejection and progress. Execution and later binding/calibration continue from frozen conditions and the prediction. A pending close with unfinished actions does not release inputs. Successful episode close releases remaining inputs of that task/episode.

Reading or reusing a released/expired input raises `ContextExpired` (`StorageError`), and CLI reuse returns exit code 2. Use `--audit` to inspect its manifest, or start a new decision. An empty context is never returned as a substitute. Python objects already held by a caller remain their old value; they are not refreshed from the bank.

A strategy provider call saves an `invalid` prediction with `call_state=in_flight` before starting, under the same prediction ID that the returned result will update. A process interruption therefore leaves an honest unknown result/usage and durable conditions. Successful and failed responses preserve real usage and all identity/request references on their first returned-result save. No automatic retry is implied by an interrupted call.

## Compatibility and model-input changes

Existing `wm-context/3`, execution payloads and full legacy snapshots remain readable without migrating historical content or rebuilding the bank. The reader preserves the original format version. Unsupported versions raise an explicit error. New `wm-context/4` serializes `conditions` rather than a duplicate complete `snapshot`; X/B live once in `solving_context`. Normal new writes do not return to the old full-snapshot path.

Condition lookup prefers new frozen facts, then the original stored legacy data. Budget comparison prefers the prediction's `frozen_budget_state` and falls back to its condition/snapshot reference. Problem binding uses each execution's own precondition identity. Internal CIR remains absent from provider input. Unknown historical text, budget, knowledge availability or actual method is not filled from today's values. An untimed legacy execution carries `execution_features.created_at_unknown`; its numeric timestamp is only a serialization sentinel, and H+ needs an observed record/action time. Missing budget history withholds probability scoring while retaining real cost/masks; an explicitly undeclared budget is distinct from missing history. A missing requested task-text version is not replaced with another version.

Storage and input projection are validated separately:

- A: storage decoupling. The fixed-memory fixture compares actual retrieval identities/content, candidate statistics, targets and key provider fields with the captured local v3 baseline, ignoring only generated identities/times, the declared context schema version and the separately tested diagnostic projection.
- B: calibration diagnostic projection. Context/model input carries `validity_correction_summary` (count, kinds and field counts) rather than complete `validity_corrections`. Counts describe the published population before candidate filtering. Individual stored/derived evaluations and the published diagnostic summary retain full corrections independently.
- The legacy `predict_outcome` writer also projects its state to candidate-related knowledge/statistics and omits growing support ID lists. That changes legacy provider input; it does not change how old predictions are read. The current `wm-so/1` provider never needed the complete stored snapshot.

Model/metric/unit/scope/observation-version isolation, per-field eligibility, windows and sample thresholds remain unchanged. Request diagnostics measure characters, not tokens. This change does not promise identical stochastic model responses or reduced request size on every fixture.

## Fixed-memory testing with per-question copies

Finish building the base and stop writing to it, including index updates. Create a fresh home from the same base for every test question:

```bash
orx --home /work/fixed-base fork-memory --destination /work/test-A
orx --home /work/test-A plan-next --task A.json --episode test --candidates candidates-A.json
# choose-next, execute, record, check-task and close-episode all use /work/test-A.
# Retries for A continue in this same home and inherit A's progress/budget.

orx --home /work/fixed-base fork-memory --destination /work/test-B
# All B operations use /work/test-B, never /work/test-A.
```

`fork-memory` refuses an existing destination and copies the database through SQLite's consistent backup API, including WAL data. It copies the base index, archives and immutable artifacts. Source schema/history is not initialized or rewritten. The source must be quiescent so the copied sidecar indexes describe the same fixed base. A normal home remains writable within a question; do not merge test results into the base. Only disabling induction does not freeze retrieval, statistics, consultation metadata or published feedback.

## Explicit old-view cleanup

Preview opens SQLite read-only and does not initialize `Store` or rewrite payloads:

```bash
orx --home /work/bank-copy maintain-snapshots

# Explicit application to a COPY, with a new backup path:
orx --home /work/bank-copy maintain-snapshots --apply --backup /work/backups/bank-before.db

# Optional physical reclamation, separately requested:
orx --home /work/bank-copy maintain-snapshots --apply --backup /work/backups/bank-before-vacuum.db --vacuum

# Explicitly preview expiry of inactive, unclosed LEGACY views:
orx --home /work/bank-copy maintain-snapshots --expire-open-days 30
```

Default candidates are verifiably closed snapshots with recoverable identity/profile and conditions. Running actions, staged attempts, active inputs, selected predictions awaiting binding, late task checks, pending H+, unsupported contexts and unresolved old references are reported as protections. Missing conditions are not guessed.

`--expire-open-days N` is an explicit legacy-view expiry policy, not permission to discard attempts or finish actions. It considers only tasks with no newer recorded activity. Add it to an explicit `--apply --backup ...` command to release recoverable inactive views; real pending attempts/actions remain pending. Unresolved references and missing applicability evidence remain protected. H+ availability and adoption never become verified by expiry.

Application creates and verifies a new consistent SQLite backup while holding the store write lock, then rechecks dependencies and atomically converts eligible views into small conditions before removing the old full rows. Historical availability IDs are shared separately; selected legacy context values are preserved as content versions before its full input expires. An unresolved archive pins shared old knowledge versions; this maintenance operation does not guess archive liveness or garbage-collect new cross-reference values/artifact files.

Reports distinguish candidate gross payload bytes, actual payload bytes before/after application, SQLite page/freelist counts and physical DB file bytes. Deleting JSON does not by itself shrink the SQLite file. `--vacuum` is optional; retained pages/WAL are not proof of failure. Rerunning is idempotent but always requires a new backup path. Restore a backup into a separate home, with its original sidecars, for a rollback; older code must not be used to silently ignore new-format records.

No ordinary open, prediction or close path cleans user legacy full snapshots. Main/DS bank cleanup is an independent operation explicitly initiated by the user after testing a copy.
