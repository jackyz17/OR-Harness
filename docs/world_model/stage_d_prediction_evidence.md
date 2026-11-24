# Stage D: G/C/R prediction-evidence selection and delivery

Status: implemented on the `wm-so/1` service without changing its output
protocol, G/C/R definitions, U weights, H+, or Strategic Knowledge.

## 1. Defects addressed

1. **G did not foreground why a candidate's method fit the current task.**
   Global calibration is a diagnostic of past error, but the provider view
   gave no compact, candidate-specific view of recorded method cases.
2. **C returned no reference at all when `phase` was `unknown`.** The old
   strict filter rejected every group, even same-remaining-scope history whose
   phase or auxiliary condition was merely unrecorded.
3. **R's condition groups were delivered without the broader event base
   rates** that `technical_r_statistics()` already computes for every closed
   window.

This stage is deliberately evidence-only: it does not fit a calibrator, move
a probability, or turn a base rate into a prediction.

A companion contract correction reduces the ACTIVE C prediction targets to
four remaining-task dimensions: `remaining_latency_s`, `output_tokens`,
`tool_calls`, `retries`. `latency_s`, `solver_runtime_s` and raw
`llm_tokens` remain historical execution observations for diagnostics/audit,
but they are not prediction targets, U terms, C-reference dimensions or new C
calibration dimensions. See section 3a.

## 2. G — candidate method/condition evidence

`prediction_context.g_candidate_evidence` is derived on read from the
already-selected `prediction_execution_pairs`; it adds no second history store
and no new storage.

- **Only observed G labels** are cases. A crash, timeout or unobserved
  execution is not converted to G=0; it remains visible in the pair block for
  R/C.
- **Explicit comparability.** `comparison_role` is
  `strict_condition_match` (same task, or known same family+structure) or
  `partial_condition_similarity` (shared method terms / one known condition
  only). A missing structural value is listed as unknown, not treated as a
  match.
- **Recorded causes only.** `recorded_cause` is copied from
  `benefit.attribution`; no cause is inferred from `task_check=failed`, an
  execution status, or free prose.
- **Not a correction.** The block carries
  `global_calibration_role: diagnostic_only`; it contains no correction
  coefficient and no post-hoc probability adjustment.
- **No double counting.** One case per `evaluation_id`; same-task method
  comparisons stay in `same_task_comparisons` and are not copied into new
  cases.

The selection ranks same-task and known matching conditions first, then
method match, then labeled failures. It is capped (default 4) and the full
body stays in the pair block / evaluation store (`detail_ref`).

## 3. C — strict vs broader references

`cost_evidence` now exposes two lists:

| Grade | Meaning | Use |
|---|---|---|
| `reference_ranges` / `cases` | The existing strict rule: same remaining scope, phase, family, known solving limits and known structure | Comparable range, eligible for the shared decision cost scale |
| `broader_reference_ranges` / `broader_reference_cases` | Same `remaining_to_task_end` measurement scope, but phase and/or an auxiliary condition is unknown | Weak reference only; clearly labelled |
| unavailable | A known phase, family, solver, config or structure conflict, incomplete window, or different scope | Not shown |

`reference_coverage` reports `strict` / `broader_reference` /
`unavailable` counts and the current phase. `evidence_status` is one of
`strict`, `strict_and_broader`, `broader_reference_only`,
`insufficient_evidence`.

Rules preserved:

- `phase=unknown` is **not guessed** as `initial_modelling` or
  `existing_model`. It can receive a broader reference whose
  `grade_reasons` says the current phase is unknown.
- No cost measurement scopes are mixed. The strict/broader lists are separate
  and no group is relabelled.
- Each group/case keeps `n_distinct_episodes`, per-dimension
  `n`/`n_distinct_episodes`/p50/p75/min/max, observation scope and bounded
  source refs (`n_source_refs`, `source_refs_ref`).
- A dimension with no reliable observation remains absent (unknown), never a
  zero.
- `decision_cost_scales()` reads **only** strict `reference_ranges`; a broader
  reference cannot enter a multi-candidate common scale.

## 3a. Active C prediction contract (`wm-c-predict/4`)

A new wm-so/1 prediction may name ONLY:

| Dimension | Meaning |
|---|---|
| `remaining_latency_s` | wall-clock from this decision's anchor to task termination, including later modelling, coding, tool calls, solving, checking, repair and retries |
| `output_tokens` | execution-model tokens GENERATED from this decision to task termination (never input+tokens) |
| `tool_calls` | tool calls in the SAME remaining task window |
| `retries` | additional solve attempts after the bound baseline attempt |

`latency_s` (one sandbox attempt's wall-clock) and `solver_runtime_s` (the
solver's inner runtime) are rejected with an explicit contract error and
point to the four active names; they are never renamed into
`remaining_latency_s`. Raw `llm_tokens` is also not an active target.

- New predictions persist an `ExpectedCost.prediction_contract` marker,
  `wm-c-predict/4`.
- U (`score_strategy_outcome_predictions`) intersects declared dims with the
  four active targets, and prices only a common remaining-to-task_end scope.
- C close-out (`_predicted_cost_dims`) and `conditional_cost` prediction
  coverage score only the four active targets.
- C reference selection drops legacy observation dimensions from
  `cost_evidence` ranges/cases; the stored `observed_fact` keeps them for
  execution diagnostics/audit.
- `decision_cost_scales()` scales only active dimensions, so old
  latency/solver observations cannot enter a shared U scale.
- Archived six-dimensional cost predictions remain readable
  (`ExpectedCost.prediction_contract is None`), are never converted, and are
  excluded from new U/C calibration.

## 4. R — global event base rates

`implementation_r_evidence.global_event_base_rates` is a separate block built
from `technical_r_statistics()["events"]`:

- per event (`implementation_failure`, `timeout`, `environment_failure`):
  `occurred`, `not_occurred`, `unknown`, `n_known_labels`, `occurrence_rate`;
- `n_attempts`, `observation_rule_version`, source/scope/rate-basis notes;
- `support: measured_global_base_rate` or `insufficient_evidence`.

The denominator is known labels only. An unknown label is not a
non-occurrence, and a rate with no known labels stays `None`. Only recorded
executed attempts enter the population: an unexecuted candidate and a
prediction-stage request timeout have no attempt record. Events are never
summed; there is no joint/max probability. When no condition group matches,
the block still provides this wide background with
`support: global_base_rates_only`.

## 5. Delivery, budget and compatibility

- `g_candidate_evidence` is a provider-only block and participates in the
  existing G/C/R evidence ceiling. When the **whole** request is still over
  its bound, G/C/R evidence is reduced to grades/counts plus a `detail_ref`
  before the core is declared unsendable; the stored context keeps the full
  bodies.
- C's `reference_coverage` is refreshed after budget trimming, so a shortened
  weak-reference list cannot keep a stale "strict" count.
- No persisted schema, calibration summary, frozen context, U weight, H+
  trigger, or Strategic Knowledge interface changed. Old summaries without
  the new fields read as unavailable/empty, not as errors.

## 6. Typical provider-view comparison

Before (phase unknown):

```json
"cost_evidence": {
  "current_conditions": {"phase": "unknown"},
  "reference_ranges": [],
  "cases": [],
  "missing": ["decision phase unavailable"]
}
```

After:

```json
"cost_evidence": {
  "current_conditions": {"phase": "unknown"},
  "reference_ranges": [],
  "broader_reference_ranges": [{
    "reference_grade": "broader_reference",
    "grade_reasons": ["current phase unknown", "..."],
    "n_distinct_episodes": 2,
    "dimensions": {"tool_calls": {"p50": 8, "p75": 9, "n": 2}}
  }],
  "cases": [],
  "broader_reference_cases": [],
  "reference_coverage": {
    "strict": {"n_groups_available": 0},
    "broader_reference": {"n_groups_available": 1},
    "unavailable": false
  },
  "evidence_status": "broader_reference_only",
  "missing": ["decision phase unavailable: broader references only, ..."]
}
```

Coverage for the fixture matrix used by the stage tests:

| Scenario | strict | broader | unavailable |
|---|---:|---:|---:|
| phase known and matching, no conflicts | 1 | 0 | no |
| phase unknown, same-scope history | 0 | 1 | no |
| known phase mismatch | 0 | 0 | yes |
| known solver conflict | 0 | 0 | yes |
| no history at all | 0 | 0 | yes |

## 7. Tests

New deterministic suite:
`tests/harness/test_stage_d_prediction_evidence.py` covers G ranking/causes/
unknown labels/metric separation, C strict/broader/conflicts/scales/unknown
dimensions/attempt-latency separation, R global rates/denominators/legacy
readability, provider-request delivery, and the active four-dimensional C
contract (rejection of `latency_s`/`solver_runtime_s`, independent missing
dimensions, old six-dimensional read compatibility, and U excluding archived
observation dimensions).

Affected existing suites were updated only where the stage intentionally
changes behavior (`phase=unknown` now has an explicitly graded broader
reference) and run with the new suite:

```bash
PYTHONPATH=src:tests/harness python -m unittest \
  tests.harness.test_stage_d_prediction_evidence \
  tests.harness.test_g_decoupling \
  tests.harness.test_c_signal_review \
  tests.harness.test_r13_remaining_cost \
  tests.harness.test_r_technical \
  tests.harness.test_wm_calibration_loop \
  tests.harness.test_history_feedback_budget
```

---

> **Superseded by stage E/1 (unified historical retrieval).** The evidence
> selection described here (G shared-method keywords, C condition groups with a
> phase classification, R solver-interface condition groups) has been replaced
> by `src/or_harness/world_model/history_retrieval.py`. See
> `docs/world_model/stage_e_unified_history.md`: LOCAL cases come from one
> embedding-recalled pool reranked per signal, GLOBAL statistics are separate,
> and C no longer uses a modelling phase. The four-dimensional C prediction
> contract (`wm-c-predict/4`) and the observation rules in this document are
> unchanged.
