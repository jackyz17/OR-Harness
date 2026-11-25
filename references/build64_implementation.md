# build6.4 implementation observations and corrective costs

This is an extension of existing actions and evidence projections. It adds no
event database, model call, transcript analyser, retrieval algorithm or formal
execution permission. Deploy one fixed code/prompt revision for the run.

## Contracts and versions

| Target | Archived definition | Current definition |
| --- | --- | --- |
| R.implementation_failure | Technical failure of the bound formal submission; a protocol-valid success could label 0 | At least one evidenced substantive implementation defect in the selected candidate window, including repaired pre-execute and recovered internal defects |
| R observation/scoring | `wm-r-technical/1`, `wm-r-max-event/1` (earlier single-event scoring also readable) | `wm-r-implementation-window/1`, `wm-r-max-event/2` |
| C.retries | Additional formal solve attempts after the bound attempt, `wm-retry-additional-attempts/1` | Additional corrective/retry rounds actually started from the prediction's decision anchor to task termination, `wm-retry-corrective-rounds/1` |

Implementation labels: **1** needs one confirmed qualifying defect; **0** needs
actual implementation and sufficiently complete coverage without such a defect;
otherwise **Unknown**. Successful execute alone cannot establish 0. A defect can
be an actual API/type/index/parsing bug, incorrect implementation of the selected
mathematical model, or broken code/result delivery. Mathematical strategy
inadequacy remains G. Routine probes, expected exceptions, unverified suspicions,
duplicate logs and confirmed environment/framework faults are excluded from this R.

Timeout still means harness interruption of the bound formal execution before a
usable result. Environment_failure still concerns the bound formal execution.
All three event names, probability ranges, maximum aggregation and U coefficients
stay unchanged. The original `technical_outcome` remains a separate formal
diagnostic with its original observation version. Normal timezone runtime data
is mounted read-only; recognized unavailable timezone data is an environment
dependency failure. Explicit file-contract violations remain implementation defects.

Retries counts **started extra correction**, not errors or tool invocations.
Initial coding, planned composite steps, ordinary probing and clean checks add
nothing. A confirmed error with no started repair adds no retry. Substantive
repair, reimplementation, failed-acceptance rework and environment/framework
rework can count. Nonnegative expected fractions, zero and Unknown are valid;
there is no default or floor of 0.1. Other C dimensions, weights and normalization
are unchanged. Recorded execution cost/budget retry increments remain the
independent `or-retry-increment/1` formal-attempt diagnostic.

## Bindings and boundaries

1. Call `plan_next` and take its scored selected prediction P. Before actual
   implementation, call `choose_next(decision_action_id, prediction_id=P)`.
   Its returned action ID S is the real selection anchor for implementation.
   Repeating the same choice before execution reuses S. Execute can still
   auto-select, but that later anchor cannot observe earlier unreported work.
2. Implementation facts name S; P/task/episode are checked and filled from S.
   S must belong to the real scored decision. No execution is fabricated or
   consumed by reporting a local implementation failure.
3. The implementation window ends at candidate implementation/necessary technical
   verification completion or the next `plan_next` action's start, whichever
   comes first. Selection before/after a new decision creates different windows
   even for the same method. Facts after the next decision are refused for S.
4. Each retries window starts at **that prediction's own plan decision action**,
   not the first execution or the selection of some other candidate. It ends
   at the confirmed `finish_task` boundary. It includes later candidates' started
   rework, but excludes corrections already started before this anchor. Report
   complete corrective coverage for the whole remaining window; otherwise the
   observation exposes a lower bound and stays excluded from numeric calibration.

A selected implementation abandoned before execute retains its actual facts and
R window label. Closeout reports it in
`calibration_summary.technical_r.selected_without_execution`. It has no fabricated
formal Evaluation, G, timeout/environment label or induction job. Its R forecast
is not a Brier sample in the current Evaluation channel: this is an explicit
evaluation coverage gap. Its action facts can be read directly and remain in
current-task progress. Cross-task embedding discovery still requires a real
Execution Evidence record; this extension does not add another searchable bank.

## Minimal host reports

Use existing `ORHarness.report_action('model' | 'verify', ...)` or `end_action`
with optional outcome keys below. CLI uses `orx action --report model --task
t.json --episode ep1 --outcome report.json`. `--outcome` accepts JSON or a file;
no new CLI command or output contract is introduced. Existing `host_usage` can
accompany the report for the other cost dimensions.

```python
import time

choice = h.choose_next(plan['decision_action_id'],
                       prediction_id=plan['selected_prediction_id'])
selection_id = choice['action_id']
selected_at = h.actions.get(selection_id).started_at
anchor = h.actions.get(plan['decision_action_id']).started_at

# Supply actual recorded timestamps and stable IDs from the host, not guessed
# commands/exception keywords. Capture the failing version before overwriting it.
failed_at = time.time()
# ...actual failing targeted test, then started substantive correction...
repair_started = time.time()
h.report_action('model', task, 'ep1', outcome={
    'implementation_facts': [{
        'selection_action_id': selection_id,
        'event_id': 'host-defect-17',
        'occurred_at': failed_at,
        'phase': 'pre_execute',
        'classification': 'implementation',
        'confirmed': True,
        'requires_correction': True,
        'observer': 'runner-session-42',
        'summary': 'A targeted API test rejected native numeric scalars.',
        'evidence_refs': ['artifact:failure.py', 'host-log:session42/test17'],
        'correction': {
            'round_id': 'host-correction-8',
            'started_at': repair_started,
            'method': 'Normalize scalar types before constructing expressions.',
            'evidence_refs': ['artifact:repair.diff', 'artifact:fixed.py'],
            'result': 'targeted test passed',
            'verification_refs': ['host-log:session42/test18'],
        },
    }],
})

# After implementation/execution/necessary technical verification finishes:
stop = time.time()
h.report_action('verify', task, 'ep1', started_at=stop, ended_at=stop, outcome={
    'implementation_coverage': {
        'selection_action_id': selection_id,
        'from': selected_at, 'through': stop,
        'implemented': True, 'complete': True, 'candidate_complete': True,
        'observer': 'runner-session-42',
        'evidence_refs': ['host-log:session42/implementation-span'],
    },
})

# Commit/finish under the existing workflow. Only claim complete when the host
# actually observed all started corrective rounds from anchor through stop.
stop = time.time()
h.report_action('finish_task', task, 'ep1', started_at=stop, ended_at=stop, outcome={
    'correction_coverage': {
        'from': anchor, 'through': stop, 'complete': True,
        'observer': 'runner-session-42',
        'evidence_refs': ['host-log:session42/complete-correction-span'],
    },
})
```

`phase`: implementation / pre_execute / execution / technical_verification.
`classification`: implementation / environment / framework / acceptance /
mathematical / other / unknown. Missing/unconfirmed responsibility remains
Unknown. `obstructed: true` can establish a substantive obstacle instead of
`requires_correction`. Mark expected probes/handling `excluded: true`, or simply
omit them from defect facts. Omit `correction` if no additional round started.
All references locate concise evidence; host reports are labelled `host_report`
with an explicit observer, never relabelled as Executor observations. A report
accepts at most 32 small facts (6,000 characters each), using references for
full scripts, tracebacks and diffs.

For a defect inside a formal execution, also set `execution_id: EX`. The
framework verifies EX belongs to S. The terminal formal implementation error
uses `execution:EX:technical`, shared with Executor. A separate recovered internal
defect keeps its stable host `event_id`; it must not be merged with another bug
merely because both occurred in EX. On a successful formal result this is the
default. If EX also ended with a terminal implementation error, set
`formal_error: false` for distinct earlier recovered defects; use
`formal_error: true` for the same terminal error. A captured occurrence time
inside EX is retained; an outside logging timestamp uses receipt completion.
Stdout containing an exception is never automatically treated as a new defect.

## Deduplication and evidence delivery

- Defects deduplicate by `event_id` within the exact task/episode. A formal error
  reported by Executor and host uses the same canonical identity and retains
  both provenances. Conflicting classifications/selection bindings stay visible
  and cannot produce a clean negative.
- Corrections deduplicate by stable `correction.round_id` and actual start time.
  Several edits/tools, or repeated event reports within a round, count once.
  Conflicting round times make completeness unknown. To associate a formal
  resubmission with that same round, report `correction.retry_action_id` using
  the real execute action ID in the same task/episode. This can be a later report
  of the same event/round, preserving its original occurrence/start times and
  adding the actual retry action ID and supporting references. It grants no
  execution permission. A proven linked round started before a later decision
  is not charged again at resubmission.
  Time overlap or the same failed event alone cannot prove that a repair owns
  the next formal retry. If an intervening repair has no reliable association,
  retain incomplete coverage and the count of evidenced started rounds as a
  lower bound, even when the host reported complete coverage. Abandoning an old
  repair and actually starting reimplementation needs distinct round IDs; link
  the resubmission to the round that performed it. Automatic formal-retry facts
  supplement explicit reports only when the actual failed attempt/resubmission
  establishes an additional round without an ambiguous reported repair.
- Defect references preserve failing scripts/tests; correction references carry
  the diff/fixed script and observable result; verification references locate
  final checks. Reflection and induction material include the same facts.
- Real execution projections read the associated actions into
  `implementation_evidence`: vector Execution Evidence → frozen retrieval hits
  → actual provider request; shared history pool → candidate R cases; and
  `compact_execution` → H+=1 induction material. This is a read-time projection,
  not a rewrite of archived execution records. Large material is bounded and
  identifies omitted facts and source action references.
- H+=0 keeps exactly the same real evidence but creates no induction job. H+=1
  requests existing review after an actual bound execution; it does not promise
  a published rule. The frozen switch, fields and publication standards do not
  change. Implementation rules use the existing When/Do/Expect representation.

## Compatibility, reporting and Runner duties

Current observations carry the new versions in the existing prediction trace
and request convention. Prompt text/digest/context tracing uses the existing
mechanism; no second version registry is added. Old predictions are evaluated
under their own formal convention, without migration. Calibration groups split
by R observation/scoring and retries version; historical formal diagnostics are
under `technical_r.historical_formal`. Current global/local numeric feedback
does not convert old R zeros or formal retry totals. Original stored predictions,
evaluations and receipts stay unchanged, including when late facts refresh a
derived current view.

`conditional_cost.retries` reports MAE, signed error (actual minus predicted),
predicted/actual means, prediction variance, prediction distributions conditional
on retry/no-retry, positive-positive log-error coverage, and descriptive constant
0/0.1 baseline MAEs. No epsilon relative error is manufactured for zeros.

Runner must select before implementation, identify actual defects/started rounds,
keep stable IDs/timestamps, capture failing versions/diffs/checks, report genuine
coverage, close the correct episode and complete required H+=1 reviews. It must
retain formal retry limits and reference-answer separation. The framework does
not monitor arbitrary host tool calls, retrospectively infer repairs from logs,
or authorize solving outside formal execution. Recovered internal defects need
explicit structured host evidence. Generic cost amendment `retries` still refers
to formal attempt accounting; do not backfill it with corrective-round totals.

Scripted tests prove binding, observation and material transport, not improved
forecast discrimination or knowledge quality. A real Runner/provider pilot is
still needed before claiming complete production coverage or improved predictions.
Without a Runner reporting these facts and coverage, current R negatives and
complete retries calibration remain unavailable; this blocks a build that claims
complete implementation/correction observations. It does not prevent retaining
honest Unknown/lower-bound evidence under the existing workflow.
