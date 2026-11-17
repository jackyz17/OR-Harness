# Saved prediction inputs and memory maintenance

Use this page to recover a saved input, run fixed-memory tests or maintain old snapshots. Use the [prediction reference](world_model_contract.md) for request material and forecast fields.

## Context reuse and expiry

| Record | What is stored and for how long |
|---|---|
| Current `BeliefSnapshot` | In-memory view of the task, progress, capability evidence and budget. `snapshot` persists a compact condition receipt |
| `runtime_inputs` | Working input shared by a decision's candidates. Released on successful close-out or expired by reuse limits |
| `prediction_contexts` | Saved input manifest, selected versions and digest for audit. Available after working-input release, subject to calibration archival |
| `frozen_conditions` and `condition_values` | Shared, immutable problem, progress and referenced evidence values. Action and prediction links resolve these facts |

A close-out with unfinished actions keeps its working input. A completed close-out releases it. For a released or expired context, read `orx context --context-id CTX --audit` or start a new decision. Ordinary reuse requires an active input and reports `ContextExpired` (CLI exit 2) otherwise. See [input identity and audit](world_model_contract.md#4-inspect-a-prediction-and-its-input).

| Setting | Default |
|---|---|
| `OR_HARNESS_CONTEXT_TTL_S` | 86,400 seconds for reuse |
| `OR_HARNESS_ACTIVE_CONTEXTS` | 128 active inputs per home |
| `OR_HARNESS_ACTIVE_CONTEXT_BYTES` | 16,777,216 bytes of active payloads per home |
| `OR_HARNESS_CONTEXT_CALL_LEASE_S` | 3,600 seconds of protection for an in-progress provider call |

Context writes and consumption apply the limits. Capacity pressure expires older unleased inputs first. A new input is refused if it still exceeds capacity after eligible inputs expire.

## Read historical records

Read the saved record's version and original conditions. New `wm-context/4` contexts reference compact conditions. Old `wm-context/3` and complete legacy snapshots use their saved content. Reading a compact snapshot receipt reports `coverage.conditions_only=true`. Missing historical knowledge, task text or measurements are reported as gaps. Archived calibration uses its own [retention rules](episode_closeout.md#6-retention-three-separate-scopes).

| Stored item | How to interpret it |
|---|---|
| Context or snapshot | Read the saved input, frozen values and version references for that decision. Use `--audit` for a released input's manifest |
| Unversioned prediction | Read through `orx contract --payload prediction.json`. `legacy_view` reports mapped fields and gaps. Old quality maps to `benefit.value`, old failure probability to the `task_failure` risk event, and confidence to model-reported uncertainty. Cost keeps its measured mask |
| Legacy action spec | `measurement_scope="attempt"` maps to attempt scope while retaining the declared parameters. Declare the intended scope explicitly when the old scope cannot be mapped |
| Execution or artifact | Follow saved execution, action, condition and artifact references. The task check records acceptance, and exact adopted rule versions belong to the execution receipt |

Use the saved `task_text_digest` and rule/body references to identify the content seen at the time. See [rule-version use](induction.md#existing-entry-use-and-utility) for adoption and verification details.

## Fixed-memory testing with per-question copies

Finish building the base and stop writes, including index updates, before copying. Give each test question a fresh home from that same base.

```bash
orx --home /work/fixed-base fork-memory --destination /work/test-A
orx --home /work/test-A plan-next --task A.json --episode test --candidates candidates-A.json

orx --home /work/fixed-base fork-memory --destination /work/test-B
```

Use the question's home for execution, task-derived checks, retries, recording and close-out. Lock its final answer before reference scoring and keep reference inputs and results in the evaluator. `fork-memory` uses a consistent SQLite backup and copies the matching index, archive and artifacts. Supply a new destination for each question.

## Preview and compact old snapshots

Use a copy of the bank for maintenance. A preview opens the database read-only.

```bash
orx --home /work/bank-copy maintain-snapshots
orx --home /work/bank-copy maintain-snapshots --apply --backup /work/backups/bank-before.db
```

- `--apply --backup NEW_FILE` rechecks dependencies, verifies a new backup and compacts eligible full legacy views into saved conditions.
- `--expire-open-days N` adds eligible inactive, unclosed legacy views to the preview or an explicit apply. Active inputs, unfinished actions, staged attempts, late checks, pending induction and unresolved references remain protected.
- `--vacuum` requests physical reclamation separately. Read the report's before/after payload bytes and SQLite page counts to judge logical and physical savings.

Use a new backup path for another apply. To roll back, restore the backup and its matching sidecars in a separate home. See [maintenance validation](../docs/world_model/snapshot_lifecycle_implementation.md) for storage details.
