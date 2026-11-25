# build6.4 implementation observations: validation and handoff

Date: 2026-11-26 (Asia/Shanghai). Baseline: local HEAD
`50caffbdb7ea53c66182af2482994f7cf318abf2`.

The complete contract, actual bindings, host fields and runnable API example,
deduplication, evidence paths, version isolation and Runner responsibilities are
in [the build6.4 handoff](../../references/build64_implementation.md).

## Delivered contracts

| Target | Before | After |
| --- | --- | --- |
| implementation_failure | Bound formal submission failure | Evidenced substantive defect in the actual selected candidate implementation/execution/necessary technical verification window, including repaired defects |
| R observation/scoring | `wm-r-technical/1`, `wm-r-max-event/1` | `wm-r-implementation-window/1`, `wm-r-max-event/2` |
| retries | Additional formal attempts after a bound attempt | Corrective/retry rounds actually started from this decision through task termination, including later strategy rework |
| retries observation | `wm-retry-additional-attempts/1` | `wm-retry-corrective-rounds/1` |

Actual selection uses a real `choose_next` action for the scored winner. It
does not fabricate or consume an execution. Each new formal decision ends the
earlier implementation window. Each retries observation instead uses its own
prediction's decision anchor and the confirmed task-end boundary. No final
execute success is automatically converted into a clean implementation window;
incomplete retries coverage exposes a lower bound without a calibrated actual.

`implementation_facts`, `implementation_coverage` and `correction_coverage` are
optional fields on existing action outcomes. Reports require explicit observer
provenance, actual times, stable event/round identities and evidence references.
Executor and host duplicates share `execution:EX:technical`; continuous repair
and formal resubmission count once. A selected pre-execute abandonment remains
visible in `selected_without_execution`, without a fabricated formal Evaluation.

The same facts are projected through execution retrieval and the shared history
pool into actual provider requests, reflection and existing H+=1 induction
material. Failing versions, repair diffs and final verification remain referenced.
Earlier abandoned selections can accompany a later real execution as contextual
evidence; their errors do not label that later candidate. H+=0 retains facts and
creates no induction job. G, other C dimensions/weights, R aggregation, U, H+
publication standards, retrieval algorithms/windows and formal execution quotas
remain unchanged.

Timezone runtime data is mounted read-only from explicit system/Conda runtime
directories. The implementation does not trust arbitrary `PYTHONTZPATH` values.
A recognized unavailable runtime data dependency is an environment cause;
explicit execution-contract violations retain their implementation attribution.

## Prompt and provenance

The exact resolved prompt diff is [build64_prompt.diff](build64_prompt.diff).
Machine-readable sizes and the existing prompt-version digest are in
[build64_prompt_lengths.json](build64_prompt_lengths.json).

| Portion | Before characters | After characters | Before whitespace words | After whitespace words |
| --- | ---: | ---: | ---: | ---: |
| Actual joint system prompt | 29,224 | 29,275 | 4,190 | 4,117 |
| H+ paragraph | 5,522 | 5,970 | 740 | 789 |

Word counts are whitespace splits, not model tokens. R guidance was consolidated
to offset the added R/C/H+ semantics and retain the existing full-request budget
regression. Existing prompt_ref/prompt_version/context/request-digest tracing is
reused; there is no parallel version registry or additional provider call.

## Regression evidence

The machine-readable report is [build64_regression.json](build64_regression.json).
The new mechanism tests are in
[test_build64_implementation.py](../../tests/harness/test_build64_implementation.py).
They exercise the real planner, actions, binding, Executor and material paths
with scripted forecasts. They prove contract/transport behavior, not improved
model prediction quality.

Successful verification runs include:

- 266 compatibility/calibration/budget tests: implementation windows, formal
  compatibility, new retry observations, legacy grouping, actual material and
  provider projections, and existing episode closeout.
- 163 mechanism, R, history/budget, H+ semantic and prediction-reasoning tests.
- 46 tests in the NumPy/Pandas/SciPy environment, including the actual confined
  LP/reduction/MILP composite strategy and timezone runtime dependency access.
- 29 C aggregation/weight tests after annotating the synthetic retry distribution
  with the current observation version. The normalization and weights are unchanged.
- 88 feature/R/H+ tests after distinguishing separate recovered defects inside
  one execution from its terminal formal error. A terminal error retains its
  shared Executor identity; distinct internal defects retain their own host IDs.

The larger 511-test sweep initially caught that one unversioned synthetic C
fixture; its complete module passed after correction. Other sweep results and
full-suite/baseline comparison are recorded in the JSON report. Passing counts
above overlap and must not be added into a distinct-test total.

The full fixed-snapshot run covered **2,159 tests: 28 failures, 6 errors and 2
optional numerical skips**. Its failing IDs exactly match those reproduced from
the baseline; no additional failing ID remained. The final narrow internal-event
identity correction was verified by the 88-test follow-up. The JSON retains
both the full-suite snapshot hash and final source hash so these validation
scopes are explicit.

Baseline failures were independently reproduced from a temporary `git archive`
of the original HEAD, using the same selected failing test methods and localhost
permissions. There were 28 assertion failures and 6 errors already present,
including six obsolete H+ literal-string expectations, retired CLI surfaces,
snapshot fixtures and earlier rule-lifecycle assertions. Stored historical
predictions/evaluations/executions were not migrated or rewritten to satisfy tests.

## Runner integration and remaining validation

The external Runner still must record selection before implementation; report
confirmed defects and actually started corrective rounds with stable IDs/times;
capture failing scripts, diffs and checks; report honest complete coverage;
finish/close the exact episode; and complete frozen positive H+ review jobs.
It must preserve existing formal execution quotas and reference-answer separation.
Internally recovered defects need structured host evidence rather than stdout
keyword inference. Existing attempt cost/budget retry increments must not be
backfilled with corrective-round totals.

Original A143 and the complete 27-task/34-execution production experiment were
not replayed. Real Agent forecasting discrimination and the quality/rate of
implementation knowledge publication remain unmeasured. These are not claims
made by scripted tests.

The framework can retain honest Unknown/lower-bound evidence immediately. A
formal library run claiming complete new R negatives or calibrated corrective
counts requires the Runner integration and a real end-to-end pilot first; that
coverage claim is blocked until then. The known full-suite failures also mean
the repository cannot be described as globally green. They are listed separately
from this change's corrected regressions.
