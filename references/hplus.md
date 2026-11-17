# Frozen H+ and evidence-backed learning receipts

The normal solving path is `profile / recall -> plan-next -> execute --prediction -> check-task -> record -> close-episode -> consume frozen H+`. Follow [SKILL.md](../SKILL.md) for the six steps and [induction.md](induction.md#complete-review-submissions) for the canonical complete submission and empty-review branches.

## Predict and freeze the investment decision

H+ is the pre-execution binary induction switch in each candidate's same world-model answer inside `plan-next`. `capability_gain.decision` is the JSON number 0 or 1 (`wm-hplus-binary/1`). G/C/R determines U; H+ never enters it.

With 1, name a reusable increment/boundary in `claim` and a concrete fact in `verification_conditions`; with 0, give a reason in `basis`. Judge what this execution plus one review could add relative to the prediction-time library, what can be checked, and which future step would change. An adaptation of a known method may qualify. Empty recall does not establish novelty. A valid 1 missing explanatory detail still needs an honest review. Missing/illegal decisions schedule 0 with distinct diagnostics; historical three-state records keep their original meanings.

Give the solving agent the frozen verification condition. Actual method receipts, captured artifacts and `check-task` supply facts. Optional `hplus_verification` records completion with the original prediction ID, condition index, source and references. An unfinished verification remains unknown and does not cancel a required review.

## Dispatch, consume and recover

```bash
orx close-episode --task T --episode EP --terminal completed
# J comes from result.induction_schedule.pending_ids.
orx induction-status --task T --episode EP
orx induction-material --induction-id J
# Submit the appropriate complete review branch from induction.md using J.
```

The framework dispatches one durable job per actually bound positive prediction/episode. The outer agent reads material and submits a short rule plus review or a reasoned empty review. Failure and insufficient material do not cancel 1. H+=0 stores execution and existing-entry use, skips new induction, and can still supply another job's comparison. Unexecuted candidates stay unobserved. `process_induction_jobs(reviewer, ...)` is the equivalent Python callback; no hidden model call writes a method.

Repeated close shows still-pending jobs and does not duplicate them. Before declaring building complete or starting the next training task, read `induction-status`: `result.completed` means no required job remains pending. It does not imply a positive learning result. `result.signal_counts` separates explicit 1/0, missing, invalid and legacy signals; `result.observation_counts` separates supported positive/negative/unknown actual reviews.

Keep J across interruption, material queries and identical retries. Each material query reads current evidence; the triggering execution stays fixed as controls paginate. Job/session references temporarily retain necessary evidence across eviction and restart. Completion retains a compact receipt reference, its real induction action and exact rule versions; reads expand the receipt without storing its whole body twice. Legacy full results remain readable.

A relation's `induction_id` or material session identifies formal review even for `add_evidence`. Ownership conflicts precede writes. An old identical evidence submission cannot settle a different job. A different completed-job submission is a readable conflict; an identical original retry returns the same visible receipt/action and retries deferred indexing. An unassociated existing-entry maintenance operation stays separate and needs no empty review. Frozen libraries reject either write path.

## Keep publication, learning and hypothesis truth distinct

The agent chooses publication/candidate state, revision, merge or evidence addition. See the complete A/B derivation and exact commands in [induction.md](induction.md#complete-review-submissions). Existing cases with reliable derivation or small executable checks may justify a scoped cold-start rule. A third real task plus boundary and regression tests is not a compulsory gate. Later checks retain their actual scope.

Learning receipts (`wm-hplus-review/2`) preserve the original hypothesis, actual induction action, conclusion, supported fact references and exact rule versions. A review supplies `outcome` and `comparison`, or the rule's `mechanism_comparison`:

| Learning observation | Required support |
|---|---|
| 1 | A real new/revised body plus a sourced sufficient substantive-increment conclusion and valid evidence; derivation/artifact inspection/small checks cover only their stated scope |
| 0 | A supported sufficient comparison establishes `already_covered` or `no_increment` against available prediction-time knowledge and actual work |
| unknown | Insufficient material, missing/invalid support, unexecuted work or no supported review conclusion |

A new published entry is not by itself a learning positive. An empty review can settle J with an unknown observation. Prediction H+=0 is the decision to skip new induction; learning label 0 follows a real supported review. Original hypothesis truth is separate: a refuted hypothesis may yield a useful boundary increment.

The default comparison basis uses recorded checks. `basis:"derivation"` names the retained reasoning over actual cited inputs; `basis:"artifact_inspection"` uses captured artifacts. References must resolve. `artifacts.*` points to captured readable artifacts of that execution; `task_check.*` points to a real report/check field. A library/snapshot reference alone is not an executed outcome. The receipt preserves the supporting check/version/coverage without certifying the agent's mathematics or causal judgment.

A late first check can supply missing support; correction/withdrawal changes later observations while retaining original history. A failed check can support a counterexample/boundary conclusion within its scope. Excluding the sole support makes later observation unknown; restoring re-reads valid evidence. Publication and utility remain author-managed. Explicit utility `refuting` has the current demotion behavior documented in [induction.md](induction.md#existing-entry-use-and-utility).

## Next-task feedback and audit

In the same memory home, the next `plan-next` automatically selects a few applicable settled receipts available before its cutoff. Related mechanism/structure and known labels guide selection within the existing feedback budget. The short view carries the hypothesis, checked fact references, conclusion, exact versions, verification state and a deviation reason; long comparisons remain behind references. An unavailable legacy receipt becomes unknown individually without blocking other feedback.

The frozen context can retain a receipt that the actual provider request omits for size. Inspect `result.prediction.prediction.trace.model_info.request_omissions` after `orx inspect --bank predictions --prediction P`. Read the saved context manifest with `orx context --context-id CTX --audit` for frozen feedback references. These are different audit questions. See [prediction_context.md](prediction_context.md) and [snapshot_lifecycle.md](snapshot_lifecycle.md).

## Existing experiment and historical paths

Historical capability/effect records remain inspectable. `evaluate-capability` assesses later performance under its separate legacy protocol, not this review-increment label. Existing H+/NoH+/random build-and-freeze experiment entry points remain available in separate homes. NoH+ opportunity review uses the prediction-time knowledge snapshot; a skipped formal H+=0 is not an observed no-increment review. Large frozen testing, checker isolation, splits and experimental statistics are outside this documentation change.
