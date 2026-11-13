# Binary H+: prediction, building work and factual feedback

New strategy predictions request `capability_gain.decision`, a JSON number
**0 or 1**, in the SAME G/C/R call (`wm-hplus-binary/1`). It is the induction
switch. It never enters solving utility. G keeps its existing benefit meaning.

The model sees the candidate, related execution evidence, current short rules
and relevant settled receipts. A 1 means a specific reusable rule/boundary
may be missing: give one sentence in `claim`, one check in
`verification_conditions`, and evidence in `basis`. A known method applied
to a new structure can qualify. Already covered, expected duplicate and
insufficiently grounded opportunities receive 0 with one reason in `basis`.
Use `applies_to` for future premises and existing per-metric `expected_changes`
when useful. No new numerical H+ formula or utility weight is introduced.

```json
{"capability_gain":{
  "decision":1,
  "claim":"The current upper-endpoint rule needs a sign-aware minimization boundary.",
  "basis":["candidate steps","retrieved maximum-only rule"],
  "applies_to":["integer feasible intervals with objective monotone in X"],
  "verification_conditions":[{
    "condition":"Inspect which X endpoint is selected and check the independently computed integer optimum.",
    "check_basis":"captured script, solution variables and check-task"
  }]
}}
```

Only finite JSON numbers equal to 0 or 1 are valid; booleans, strings, other
numbers and non-finite values are illegal. Missing/illegal values schedule
0 and have separate `decision_status=missing|invalid`, unlike explicit 0.
The authoritative frozen value is `trace.model_info.hplus.decision`; the
framework adds the prediction/input/source references. A valid 1 missing
claim/check detail stays 1 with gaps. Model verification flags are reset.
Historical `assessment=expected|none|insufficient_basis`, missing blocks and
old post-close allow/skip/pending judgements retain their original meanings.
They do not supply or override the new switch.

## Normal CLI/API handoff

Keep one home and episode throughout. The runner should already supply its
known execution context. Candidate selection still uses G/C/R.

```bash
orx --home BANK --induction-mode hplus --world-model URL::MODEL \
  predict-strategy --task task.json --episode E --candidate candidate.json
# P is result.prediction_id. Hand its concrete verification to the solving agent.
orx --home BANK execute --task task.json --episode E --prediction P \
  --code solve.py --workspace ws
orx --home BANK check-task EX --check '{"reference_objective":90,"reference_source":"bench_declared"}'
orx --home BANK record --from-staged EX
orx --home BANK close-episode --task T --episode E --terminal completed
orx --home BANK induction-status --task T --episode E
# For EVERY J in result.induction_schedule.pending_ids:
orx --home BANK induction-material --task T --induction-id J
orx --home BANK induce --induction-id J --relation '{"when":"Fixed Y gives an integer X interval with monotone objective.","do":"Enumerate Y, select the X endpoint by objective direction and coefficient sign, then compare objectives. Disable reduction on nonmonotone objectives.","expect":"Preserve the original integer optimum.","evidence":["EX"]}'
```

`--relation` takes inline JSON; when using a file pass its JSON text through
the calling application (or use the Python API). `check-task --check` likewise
uses inline JSON on its existing interface. `--review` accepts inline JSON or
a file. `--task` on induce remains unsupported; the job identifies its task.

New prediction homes automatically enter H+ mode. Explicit `--induction-mode
hplus` fixes the mode from the first command; `OR_INDUCTION_MODE` and Python
`ORHarness(induction_mode="hplus")` are equivalent. Mode is persistent and
cannot be switched into manual/NoH+ to bypass an existing H+ branch. Control
branches use separately forked homes. Pure old/manual work retains the prior
explicit transition.

Selection (`choose-next`) surfaces the frozen signal/check. Execution stores
the original prediction ID and `hplus_verification_plan`; report actual
method performance with the matching `OR_ACTION_ID`. An optional field in
the EXISTING check record reports the learning check:

```json
{"reference_objective":90,"reference_source":"bench_declared",
 "hplus_verification":[{
   "prediction_id":"P","condition_index":0,"state":"checked",
   "source":"reviewer inspected the captured endpoint enumeration and checker",
   "evidence_refs":["artifacts.submitted_script","task_check.checks"]
 }]}
```

This attests only to the named check; the original checker still covers
only its declared items. Missing completion remains unknown. 1 still
requires induction after a failed attempt, changed hypothesis or unfinished
check. A prediction without a real execution is unobserved and creates no job.
Separate actually bound candidates in one episode own separate signals/jobs.

Close-out saves all task/usage facts and existing rule-use results, then
creates jobs BEFORE raw evidence retention. It does not author a method or
call an induction model. The outer building agent consumes
`induction_schedule.next_actions`; completion of the TASK is separate from
completion of LIBRARY BUILDING. The handoff contract is testable:

- Every pending positive job is read and submitted once, or completed as an
  empty review; `induction-status.completed` must be true before freezing.
- Material and submission authorize the SAME `induction_id`, not another
  candidate's signal. One unambiguous matching evidence/material ID can
  resolve it automatically. Multiple jobs need an explicit ID.
- A 0 cannot authorize either formal material or a write. Its actual
  execution may nevertheless be included as a positive job's contrast.
- Retries reuse the durable material session, induction action and receipt.
  A different submission cannot reuse a completed job. A crash before commit
  leaves no partial rule/receipt; restart uses the same pending action.

Python building runners can consume this contract directly:

```python
closed = h.close_episode(task_id, episode_id)
def reviewer(material, job):
    # Existing outer agent/inducer reads actual methods, contrasts and rules.
    # It returns normal induce arguments: relations, notes and optional review.
    return {"notes": ["Checked available facts; independent evidence is missing."],
            "review": {"outcome": "insufficient_material",
                       "hypothesis_outcome": "unknown"}}
handoff = h.process_induction_jobs(reviewer, task_id=task_id, episode_id=episode_id)
assert handoff["completed"]
```

The library enforces authorization and durable completion, and the supplied
outer callback supplies the actual induction reasoning. Ignoring the handoff
leaves visible pending jobs; close-out alone cannot claim building is finished.

## Receipts, versions and verification

A submission uses the existing short when/do/expect and explicit
create/add_evidence/revise/merge operations. New/substantively changed bodies
remain candidates pending independent transfer, changed-premise boundaries
and required historical regressions. Trials declare the exact adopted version;
actual method/checker evidence qualifies later observations. See
[induction.md](induction.md).

If review finds no rule to write, complete the positive job without a relation:

```bash
orx --home BANK induce --induction-id J --note "Actual evidence reviewed" \
  --review '{"outcome":"insufficient_material","conclusion":"No supported reusable update yet.","hypothesis_outcome":"unknown"}'
```

Receipt outcomes cover `knowledge_submitted`, `new_rule`, `revised`,
`evidence_added`, `already_covered`, `no_increment`, `hypothesis_refuted` and
`insufficient_material`. Actual rule updates and versions come from committed
framework results, not caller-typed IDs. The receipt links prediction, solving
action, REAL induction action, material, original input/baseline, source facts,
check completion, conclusion and exact versions. Source scope contains the
original and displayed/cited derivation cases. Future targets use predicted
premises/family, not a source-task whitelist.

Hypothesis truth is separate from knowledge increment: a false hypothesis can
yield a useful checked boundary. Empty review is not a rule. Unverified writes
are not confirmed learning. Sufficient negative comparison supplies:

```json
{"outcome":"already_covered","hypothesis_outcome":"refuted",
 "conclusion":"The performed mechanism adds no rule beyond this baseline.",
 "comparison":{"sufficient":true,"source":"reviewer inspected frozen rules and actual checked artifacts",
   "assessment":"Necessary premises, actions and effect match the existing rule.",
   "evidence_refs":["SOURCE_FACT_REF_FROM_JOB","BASELINE_SNAPSHOT_REF_FROM_JOB"]}}
```

The framework requires real referenced baseline/fact evidence, actual method
and checker support; it cannot prove natural-language equivalence. An
unsupported comparison stays unknown. Cost-saving claims additionally require
the existing same-task, equivalent-acceptance, actual-condition and measured-
scope comparison rules. Adding a correct rule does not prove the original
cost-saving hypothesis or a causal performance improvement.

The real induction completion automatically supplies stage-1 maintenance
binding. Solving alone returns `not_adopted`; a fabricated action association
cannot replace the completed job. Legacy later-performance evaluation remains
available separately and new online effects require actual use of the exact
produced version with method/checker coverage.

## Feedback and controls

`hplus_feedback` selects a few applicable settled receipts by known family,
coarse structure and mechanism terms within the existing context budget.
Its original hypothesis, actual facts, conclusion, verification and bias
reason are frozen before the next prediction. Independently validated
substantive increment=1; sourced sufficient coverage/no-increment comparison=0;
unexecuted/insufficient/unvalidated=unknown. Old stance/effect statistics remain
separate and readable. No retrospective signal changes are made.

The runnable callback experiment entry uses the SAME initial library, task
stream, predictor, inducer and acceptance/verification functions in three
separate homes: `hplus`, `nohplus` and equal-trigger-budget `random`. NoH+
reviews 0 signals against their prediction-time library snapshots. Random
selection uses a seed independent of outcomes and matches the realized H+
trigger count. After training, freeze each library and `fork-memory` afresh
for each common independent test question. No training results from those
test copies are returned to the source library.

```bash
# Scripted false-positive/false-negative protocol fixture, real integer solves:
PYTHONPATH=src python -m or_harness.experiments.hplus --output /tmp/hplus-new-run --seed 7
# Positive minimization revision, with independent optimum 90:
PYTHONPATH=src python -m or_harness.experiments.hplus --output /tmp/hplus-positive-run --positive-min
```

Directories must be new. `summary.json` reports triggers, receipts, empty
reviews, measured induction wall time, unknown tokens/tool calls, independently
validated version increments, zero-signal missed opportunities, source input
refs, and frozen test results. The fixture deliberately scripts decisions and
reviews; it validates plumbing and makes no predictor/industrial-quality claim.

For actual experiments import `run_comparison` from
`or_harness.experiments.hplus`. Supply real `provider_factory`,
`propose(h, task)`, `attempt(h, task, prediction)` and
`review(h, material, job)` callbacks. The attempt callback uses ordinary
execute/check-task; the driver checks binding, records every actual attempt,
closes, drains required jobs and refuses missing completion. This minimal
driver has one predicted attempt per question; a richer existing retry runner
uses the same `process_induction_jobs` handoff at its episode boundary.
