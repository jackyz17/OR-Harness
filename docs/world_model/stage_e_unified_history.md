# Stage E/1 — unified historical retrieval, local cases and global statistics

## What changed

G/C/R previously selected their evidence from three different
condition-grouping systems: G scanned paired feedback for keyword overlap, C
grouped cost windows by `phase`/family/solver/structure, R grouped technical
attempts by solver interface + structure. Sparse histories fragmented into
singleton groups, so decisions with real comparable executions often got no
numerical reference.

Stage E/1 adds ONE retrieval base (`src/or_harness/world_model/history_retrieval.py`):

```
problem text ─┐
candidate method ─┤→ embedding query per view → existing execution index
conditions ───────┘        ↓
                     merge by execution (per-view scores)
                        ↓  per-view cosine floor
                     task-diversified pool (≤30, one row per task, then a
                     valuable repair attempt)
                        ↓
        attach record facts + closed-evaluation facts
                        ↓
        G / C / R light rerank (≤10 local cases each)
                        ↓
        GLOBAL G/C/R statistics (pre-decision closed window)
                        ↓
   frozen into PredictionContext.history_retrieval → provider blocks
```

No second vector store, no re-embedding of history, no new persistence.

## Configuration (actual values)

| Key | Default | Env override |
|---|---|---|
| max local cases per signal | 10 | `OR_HARNESS_HISTORY_MAX_CASES` |
| pool size | 30 | `OR_HARNESS_HISTORY_POOL` |
| same-task attempts | 2 | `OR_HARNESS_HISTORY_TASK_ATTEMPTS` |
| problem-view floor | 0.62 (local-hashing) | `OR_HARNESS_HISTORY_THRESHOLD` |
| candidate-method-view floor | 0.60 (local-hashing) | `OR_HARNESS_HISTORY_METHOD_THRESHOLD` |
| conditions-view floor | 0.30 (local-hashing) | `OR_HARNESS_HISTORY_CONDITIONS_THRESHOLD` |
| method-relevance floor (`similar_method`) | 0.62 | `OR_HARNESS_HISTORY_METHOD_RELEVANCE_THRESHOLD` |
| G/C/R evidence ceiling | 24000 chars | `OR_HARNESS_GCR_EVIDENCE_CHARS` |

A row is admitted when **any** query view reaches **its own** floor.
`similarity.overall` is the largest `score/floor` ratio: a relative ranking
value, not a cross-view comparable cosine.

### Why per-view floors

A task-text query is a subset of the execution document (which also contains
the family/outcome and the method), so its cosine is systematically lower than
a document-to-document comparison. One global threshold would either miss
same-task executions or admit unrelated ones through the loosest view.

The local-hashing values were measured on the repository review-bank history
(12 executions / 8 tasks; see `docs/world_model/history_threshold_calibration.md`
for the full tables):

| view | related range | unrelated range | chosen floor |
|---|---|---|---|
| problem (task-text query vs doc) | same task 0.843–0.944 (long text); short task texts fall to ~0.5–0.7 | different family 0.403–0.804 | 0.62 |
| candidate method | self 0.58–0.71, same-task other 0.65–0.68 | other task 0.24–0.70 (no clean separation) | 0.60 (weak) |
| conditions | same task 0.15–0.33 | different family 0.06–0.22 | 0.30 |

The sample is small and the backend is LEXICAL. The values are cautious
starting points, not universal constants; any other embedding backend must
re-derive them. A remote/OpenAI-compatible backend has NO verified calibration
in this repository: its provisional floors are reported with
`threshold_verified: false`.

## Signal rerank

- **G**: requires an embedding similarity (pool admission) and labels the case
  `same_task` / `similar_method` / `similar_conditions` / `task_similar`.
  `similar_method` needs the CANDIDATE-METHOD view to reach its own floor; a
  generic word overlap is never reported as a method match. Cases without a
  recorded label are kept with `g.observed: null` and a reason (an execution
  error is not a G modelling failure). `recorded_cause` only from recorded
  attribution.
- **C**: the ONLY hard constraint is the measurement scope
  (`remaining_to_task_end`, complete window). No phase. `strict` = same known
  family and no known structure/solver/config difference. A KNOWN difference
  (e.g. a HiGHS candidate looking at a CBC history) or an unknown condition
  keeps the case as `broader_reference` with `differences`/
  `unknown_conditions` and a lower `relevance_weight` — it is NOT discarded.
  Four dimensions only; missing dimensions are named, never zero-filled.
  Reusable artifacts are optional context, not a phase. A remaining-task cost
  window is one reference (`covers_execution_ids`), never one case per
  execution in the window.
- **R**: reranked by same solver interface, then candidate-method similarity,
  then general similarity. Each case carries the three events' labels on one
  bound attempt; `evidence_status` distinguishes `closed_pair` (calibration
  material) from `execution_only` (visible execution fact, never a calibration
  sample). A closed pair's G and R labels are attributed ONLY to the bound
  execution (`bound_execution_id`), never to every execution in the same
  remaining-task window; the C window is carried by ONE
  `cost_carrier_execution_id` and lists `covers_execution_ids`.

## Global statistics

Computed from the pre-decision closed calibration window only (same model
identity, no future observation):

- **G** (`wm-g-global/1`): observed/predicted/scored counts, distinct
  tasks/episodes, observed/predicted mean, MAE, signed bias direction, Brier,
  AUC when both classes exist, reliability bins when n ≥ 8, unknown counts with
  reasons. ALL prediction-calibration values (means, MAE, bias, Brier, AUC,
  bins) use the SAME scored `(prediction, observation)` pairs; an observed
  label without a prediction is counted separately and never paired by array
  position. `signed_error = predicted - observed`, so a POSITIVE signed mean
  means `over_predicted` (the model predicted higher).
- **C** (`wm-c-global/1`): per active dimension n, p50, p90, predicted-vs-actual
  sample count, median absolute/relative error, mean log ratio and error
  direction, Spearman rank correlation, incomplete observation count and
  reasons. Retry observations under the legacy rule are not re-read as the new
  target.
- **R** (`wm-r-global/1`): per event occurred/not_occurred/unknown,
  known-label rate, reported/scorable probabilities, mean Brier and
  discrimination (mean probability on positives minus negatives when both
  classes exist). `technical_r_statistics` now also publishes
  `mean_probability_observed` / `mean_probability_not_observed` /
  `discrimination` per event.

The globals are BACKGROUND: they are never a candidate-specific prior, never a
correction multiplier, and never a joint/max probability for R.

## Budget and delivery

- Each signal keeps at most 10 local cases in the stored selection; the
  provider blocks are compact.
- Per-block trimming caps the case tail, refreshes the coverage counts
  (`reference_coverage` for C, `selection` for G/R) and records every omission
  under `budget_trim`; global scalars survive the trim.
- `history_retrieval` in the provider view records the backend, the actual
  per-view floors, the query names, admission/pool counts and degradation.
  The long threshold rationale and the full pool stay in the stored context.
- When the embedding backend is missing/unusable, `status: degraded` and the
  reason are explicit. The global statistics are still provided; the retired
  keyword/condition matching is NOT substituted.

## Known limits

- Threshold calibration is small-sample and lexical only; a real embedding
  backend needs the same labeled-pair experiment. No production value is
  claimed here.
- Cross-family analogy detection is as good as the embedding backend. With
  local-hashing, reworded analogies fall below the floor and are honestly not
  retrieved.
- The pool is stored in the frozen context (bounded, but larger than the old
  summary). It is not a new database.
- A dimension that is never completely observed stays incomplete forever; no
  estimate is substituted.

## Attribution and validity notes

- The task-vector fast path re-applies the full-document recall validity rules:
  the record must still exist, still have `source=executed`, must not postdate
  the frozen moment, and its CURRENT retrieval document digest must match the
  indexed one. Dropped rows are counted by reason in `accounting`
  (`excluded_invalid_source`, `excluded_stale_index`, `excluded_missing_record`,
  `excluded_unreadable_task_text`, `excluded_after_cutoff`).
- A closed pair's G/R labels belong to its bound execution only; a
  remaining-task cost window is ONE reference carried by one execution.

## Index requirement

The problem view prefers the execution item's `task_vector` (an auxiliary
task-text-only embedding written by `IndexSynchronizer`). An index built before
that field existed still works through the full-document fallback, and the fact
is recorded as `accounting.problem_view_channel: legacy_full_document`; the
same-task score is then diluted by the method/outcome suffix. Run
`orx rebuild-index --layer execution` after upgrading to enable the task-text
channel. The auxiliary vector is part of the EXISTING index file — no second
vector database is created.
