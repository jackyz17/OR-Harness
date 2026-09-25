# Episode close-out and experience calibration

**Status: implemented.** This is the fourth phase of the reconstruction: after the unified contract (phase 1), the frozen prediction input context (phase 2) and the strategy-outcome prediction service (phase 3, wm-so/1), this phase closes the loop — a real episode ends, its bound predictions are evaluated against their real outcomes field by field, and the aggregate becomes experience calibration that later episodes read.

- Close-out record version: `wm-closeout/1`; calibration summary version: `wm-calib/2`; risk-event vocabulary: `wm-events/2`
- Python module: `or_harness.world_model.episode_closeout`
- API: `ORHarness.close_episode` / `episode_closeout_record` / `get_strategy_evaluation` / `strategy_prediction_evaluations` / `calibration_summary` / `archive_calibration` / `calibration_retention`
- CLI: `orx close-episode` / `orx calibration` / `orx inspect --bank evaluations` / `orx archive-calibration` / `orx inspect --bank retention`
- Tests: `tests/harness/test_episode_closeout.py`, `tests/harness/test_calibration_v2.py`
- Runnable example: [`references/examples/episode_closeout.py`](examples/episode_closeout.py)

Related: the prediction service in [`references/strategy_outcome.md`](strategy_outcome.md), the input context in [`references/prediction_context.md`](prediction_context.md), the contract objects in [`references/world_model_contract.md`](world_model_contract.md).

---

## 1. The loop this phase completes

```
profile / recall -> freeze ONE context
    -> predict candidates (wm-so/1) -> compare -> EXPLICIT choice
    -> model / execute / repair / verify (the agent's work)
    -> bind the real action to the chosen candidate's prediction
    -> finish_task / close-episode            <-- THIS PHASE
    -> per-field evaluation of every bound prediction
    -> experience calibration published (closed episodes only)
    -> LATER episodes' contexts read the published summary
```

Three units stay distinct, and the close-out never conflates them:

- **attempt** — one `execute_strategy` call, one solver invocation;
- **strategy execution window** — one selection round of one strategy: modelling, possibly several attempts, repair, verification. The same strategy chosen again later (or switched away and back) is a DIFFERENT round (`round_index`), never aggregated into one sample;
- **episode** — the whole task interaction, closed once, honestly.

## 2. Closing an episode

```bash
orx close-episode --task t1 --episode ep1 [--terminal completed|failed|aborted|budget_exhausted] \
    [--finish-action ACTION_ID] [--min-samples N]
```

What it does (in order):

1. **refuses while actions are still running**: an episode with a `running` action returns `state="pending"` with the `unfinished_actions` listed — a running scope has no final numbers, and closing anyway would freeze a record their results could never enter (a re-close just returns the stored record). End the running actions (or let them finish), then close;
2. **summarizes and evaluates** every BOUND strategy-outcome prediction of the episode against its real outcome (see §3–4);
3. **records the close-out once** — idempotent: re-closing (after a restart, or by mistake) returns the stored record and counts nothing twice;
4. **publishes the calibration summary** (see §5) — built AFTER the evaluations and the close-out record are persisted, so the first close-out's own return already includes this round's samples.

What it never does: run a solver, call the prediction model, trigger induction, or rewrite a stored prediction. `--terminal failed|aborted| budget_exhausted` closes an honest failure as a failure — only `completed` claims success. `--finish-action` links the close-out to the `finish_task` action you recorded (`orx action --report finish_task`); the close-out does not create a second task lifecycle.

**A close-out is the end of the episode, NOT a certification of the answer.** The normal flow is check-and-repair FIRST, then close: a task-result verdict (`orx check-task`) belongs before the close-out. Closing with unchecked answers is legitimate (the check may not be runnable yet), so the close-out never refuses on that ground — but it reports `task_checks = {n_executions, verdicts, unchecked, note}` so "the episode ended" is never misread as "the answer was validated". An execution whose answer a check CONFIRMED does not satisfy the task is `failed`: it contributes `0.0` to the benefit observation (the gate is recorded in `evaluations[].benefit.task_check_gated`, so "predicted 0.8, observed 0.0" can be read as a disqualified answer rather than a bad solve), while its real cost still counts in the scope's totals. A refused close-out (a running action) returns `closeout: null` with `state="pending"` — the CLI reports it as an open episode (exit 2), never as a crash.

## 3. The real outcome summary

For each bound prediction, the close-out derives what ACTUALLY happened from the action log and the execution records — never from the prediction:

- **benefit observations** use the prediction's OWN declared metric/unit/baseline (the yardstick is frozen at prediction time; it is never re-chosen after the result is known). This build has an observation adapter for exactly ONE metric — the solver's normalized gap (`optimal` → 1.0; otherwise `1-gap`). A prediction declaring any other metric (a business cost-saving ratio, a completion rate) is reported `scope_mismatch`: the solver's 1-gap number is never re-labelled as a metric it did not measure. A feasible solution with no gap/bound is the 0.5 heuristic — reported `unverified`, never scored. Other benefit kinds have no observation adapter either: reported, not evaluated.
- **window benefit rule** (declared before evaluation, applied uniformly): the LAST qualified in-scope attempt's solution is the window's benefit observation. All in-scope failed retries' measured costs still count.
- **cost** is scoped: an ATTEMPT-scope prediction is compared against the bound action's own execution; a STRATEGY-WINDOW-scope prediction is compared against the WHOLE window of its selection round — every in-scope execution, failed retries included (an 11s-then-17s window reports 28s, not 17s). Modelling/verification/other attempts are auxiliary overhead — real spend, reported separately, never charged to the predicted scope. Per-dimension totals carry completeness (`complete` only when every in-scope item measured the dimension); `latency_s` is never summed.
- **risk events** are observed by the **framework**, independently of what the model predicted, and each event has ONE observation unit. `task_check_failed` reflects the CHECK RESULT and nothing more (kind, covered scope and basis travel with the label) — a failed check does not mean the model was wrong. `environment_failure` / `implementation_failure` are the executor's own recorded `error_class` (a legacy record without one stays unknown — the cause is never inferred from prose). `solver_reported_infeasible` is a FACT about the solver's verdict, never by itself a strategy failure (correctly identifying that the original problem is infeasible is a valid outcome). `timeout` is an observation, not a verdict on the strategy. `budget_exhausted` is decided by the episode-scoped ledger. The retired names `model_invalid` / `no_feasible_solution` / `model_failure` get NO label and NO alias — see the vocabulary table in §5.
- **verification**: solver `optimal` does not prove business requirements; the summary reports the verification evidence that exists (verify actions, executor checks), nothing more. A task-level check (`orx check-task`) is part of that evidence, and a `failed` verdict replaces the solver-side quality with `0.0` for the benefit observation: the calibration channel compares against the TASK's outcome, so a relaxed answer's solver-side optimum is not scored as if the task had been solved.

## 4. The per-field evaluation

Each field is judged on its own eligibility — never a blanket verdict:

| Field | Compared when | Excluded when (and why it matters) |
|---|---|---|
| benefit | the prediction declared a value AND the same metric was observed under the same scope | no baseline/value predicted (`not_predicted`); no qualified observation (`unobserved`); heuristic-only quality (`unverified`); other metric kinds (`scope_mismatch`); binding identity not established (`identity_mismatch`). The DECLARED metric/unit always travel with the block, so an unobservable sample still groups by what it said it was predicting |
| cost | per dimension: predicted AND completely measured on the real scope; error = absolute + directed `log(actual/predicted)` | partially-measured totals are excluded (an incomplete total is not a truth); zero predicted or actual produces no relative ratio (a substituted value would be a fabrication) |
| risk | per event: a predicted probability AND a known label → Brier score | no probability (`not_predicted`); unknown label (never a 0 label by default); a RETIRED event name (no label, no alias); events are never averaged across names |
| interval | the prediction saved an interval AND an observed value exists → covered yes/no + `width` | no interval saved, or nothing comparable to cover |

**Directed feedback.** `benefit.signed_error = observed - predicted` (positive = the prediction was too LOW) and `cost.per_dim[dim].log_ratio = log(actual/predicted)` (positive = cost was under-predicted) let a later prediction tell an over-estimate from an under-estimate, not merely "off by this much". The absolute errors are kept unchanged alongside them.

Excluded fields are **neither hits nor misses**: they never enter a denominator. An evaluation with nothing comparable is `excluded` with the per-field reasons; a scope still running is `pending`. The FROZEN prediction is read, never rewritten — no post-hoc "better" prediction is regenerated closer to the result.

**Unknown identity is not a match.** The binding records `binding_mismatch` (a KNOWN disagreement) and `binding_unknown` (a field the executed action could not observe — no episode recorded, a config key the action log never carries) separately. A mismatch blocks the evaluation outright; an unknown makes the fields that depend on it unevaluable. Re-binding the same action re-evaluates the comparability (an action still running at first bind may have completed since).

## 5. The experience calibration summary

```bash
orx calibration [--min-samples N] [--rebuild]
```

Aggregated from the **WINDOW** — the newest N closed task-episodes (default 50, `OR_CALIBRATION_WINDOW`). The window is the sample set: it bounds both the statistics and the work a close-out does, so a prediction reads a small published summary instead of scanning history. An open episode never calibrates anything, least of all itself.

Grouped by `strategy_outcome|<model identity>|<metric>|<unit>|<scope>` — a different metric, unit, scope **or predicting model** is a DIFFERENT group, never pooled. Each group reports:

| Statistic | Meaning |
|---|---|
| `n_samples` / `n_distinct_episodes` / `correlated_predictions` | Prediction-observation pairs, independent episodes, and the pairs that repeat one truth. The threshold counts DISTINCT EPISODES (a `(task_id, episode_id)` pair), so five re-planning predictions over one execution cannot cross it |
| `mean_benefit_abs_error` / `mean_benefit_signed_error` | Magnitude and DIRECTION of benefit error (signed positive = historically under-predicted) |
| `mean_cost_log_error` / `mean_cost_log_ratio` per dimension | Magnitude and DIRECTION of cost error (log-ratio positive = historically under-predicted cost) |
| `interval_coverage` / `mean_interval_width` | Coverage alone cannot tell a tight interval from a mile-wide one |
| `mean_brier_by_event` / `n_brier_samples_by_event` | Per EVENT NAME, never pooled across names |
| `mean_predicted_probability` / `scored_occurrence_rate` | Both over the SAME denominator (predictions with a probability AND a label), so their difference is a meaningful over/under-estimate signal |

A group below the sample minimum (`--min-samples`, default 5) reports `insufficient_evidence` with `reliability: null` — no figure is invented. The summary is labelled `applicability: "global_diagnostic"`: it carries NO strategy or problem-condition breakdown, so it is NOT a claim about the conditional bias of the candidate currently under consideration.

This is a **measured record, not a promise**: writing the summary does not claim future predictions improve, and it is NOT a fitted calibrator or a model weight (training-free is unchanged). It is kept SEPARATE from the legacy knowledge-prediction reliability (`prediction_class_reliability`): a knowledge hit rate never proves OR strategy-outcome accuracy.

### Two counting units

The summary reports two numbers that are never conflated:

- **prediction-observation pairs** (`groups`): one per eligible prediction. This is the sample for judging a probability.
- **observation units** (`occurrence`): the framework's own count of how often an event really happened — once per execution (or per episode for `budget_exhausted`), however many predictions were bound to it. Each entry reports `n_observation_units`, `n_occurred` / `n_not_occurred` / `n_unknown` and `unit_occurrence_rate` over its OWN denominator (labelled observation units). It is reported separately and must never be subtracted from a `mean_predicted_probability` computed over a different sample set.

**Labels are PER EXECUTION.** An episode whose two executions had different outcomes (one timed out, one passed its check) is ONE occurrence and ONE absence — a rate of 50%, not 100%. A scope-level verdict assigned to every execution in the scope would report every run as a failure whenever one failed.

**Observation is derived from the CURRENT facts.** `occurrence` is rebuilt from the executions the window's episodes really produced on every read, NOT from the labels frozen in stored evaluations:
- an episode with NO bound prediction still produced real executions, so its observations still count;
- a late task check (in either direction) is reflected immediately;
- a withdrawn (`exclude-execution`) fact still counts as an observation — it really ran — while leaving the prediction-comparison statistics.

An event the model never predicted still counts in `occurrence` and can never produce a Brier score.

### The risk-event vocabulary (`wm-events/2`)

| Event | Unit | Source | Note |
|---|---|---|---|
| `environment_failure` | execution | `error_class == 'environment'` | A record with no recorded class stays UNKNOWN |
| `implementation_failure` | execution | `error_class == 'model'` | The harness's own code failed — NOT a modelling error |
| `timeout` | execution | `status == 'timeout'` | An observation, not a verdict on the strategy |
| `solver_reported_infeasible` | execution | `status == 'infeasible'` | A verdict, not by itself a failure |
| `task_check_failed` | execution | `task_check.state` | The CHECK RESULT only; kind/scope/basis travel with it |
| `budget_exhausted` | episode | the declared budget view | Only a scope matching the ledger's (the episode) can be scored |

**Retired** (no label, no alias): `model_invalid` (a solver error or a failed check cannot establish that the MODEL was wrong — use `task_check_failed` for the check result), `no_feasible_solution` (ambiguous between a reported infeasibility and a failure to find a solution), `model_failure` (renamed `implementation_failure`). The prediction prompt lists ONLY the observable names and explicitly forbids inventing the retired ones.

**Scope matching.** The budget ledger is episode-scoped, so an attempt- or strategy-window-scope prediction is NOT scored against it — the label stays unknown with an explicit `scope_mismatch` basis, while the framework still records the FACT under `occurrence`. This round deliberately builds no per-attempt budget system.

**When later episodes see it.** A NEW `PredictionContext` carries the published summary's CONTENT under `strategy_outcome_calibration` — a SINGLE-ROW read, no history scan. The block is filtered twice: by the ATTACHED provider's model identity (another model's errors are not evidence about this one, and the withheld keys are named under `withheld_groups`) and, in the request, by the candidate's scope. Stored contexts never change: episode 1's context keeps what it froze.

**Corrections republish, and correct the RIGHT field.** A late task check, an exclusion or a restore on an episode still IN the window rebuilds and republishes the summary. The stored evaluation is never rewritten; the summary uses a LIVE re-derivation instead:
- a check that now FAILS re-derives the benefit observation to `0.0` **and keeps the measured cost** — a wrong answer still cost what it cost, so the correction must not take a valid measurement with it;
- a check that was WITHDRAWN restores the un-gated observation;
- an execution WITHDRAWN from the evidence set removes the sample entirely (there is nothing left to calibrate against);
- the correction is SCOPED to the executions the evaluation actually compared, so a late verdict on one execution does not disqualify a sibling's evaluation.

Publication is atomic: the evaluations, the close-out record and the registry row are written in ONE transaction, then the summary and its `published` flag in a second. A crash between them leaves `published=0` (or a record with no registry row), which the next close or `orx calibration --rebuild` detects and finishes without re-counting.

### Per-statistic thresholds

Every statistic carries its OWN evidence verdict (`benefit_evidence`, `cost_evidence[dim]`, `interval_evidence`, `brier_evidence_by_event[event]`), because a group of 50 episodes where only ONE predicted a given risk has exactly ONE sample for that risk. A group whose statistics are unevenly supported reports `basis: "partially_measured"` and names them under `under_sampled_statistics`; `min_samples_basis` states that the threshold is applied per statistic. A statistic nobody predicted is **absent** (no verdict), not "insufficient evidence" — "nobody asked" is not "too little data".

## 6. Retention: three separate scopes

```bash
orx inspect --bank retention
orx archive-calibration [--dry-run]
```

One "retention" number cannot honestly bound the online database, decide which episodes calibrate, and cap the archive at the same time, so the three questions are answered separately (all configurable, all recorded on the summary):

| Scope | Question | Default | Env var |
|---|---|---|---|
| **window** | Which closed task-episodes calibrate? | 50 | `OR_CALIBRATION_WINDOW` |
| **late-check grace** | How long is ONLINE detail kept for a possible late check? Applies ONLY to episodes with an unchecked execution | 30 days | `OR_CALIBRATION_LATE_CHECK_GRACE_DAYS` |
| **archive caps** | What history is kept on disk? Per file / total / age — whichever bites first evicts the oldest file | 64 MB / 1 GB / 365 days | `OR_CALIBRATION_ARCHIVE_*` |

An episode whose executions all carry a verdict is NOT held by the grace period: it leaves the online set as soon as it drops out of the window. Only episodes genuinely awaiting a verdict are kept — there is no blanket extra retention.

**Only DETAIL is archived** (evaluation, prediction and frozen-context payloads) into `{home}/archive/calibration/calibration-NNNN.jsonl`. The **close-out registry tombstone stays online permanently**: it is what keeps a repeated close idempotent and the window locatable after the payloads are gone, so archiving can never cause a duplicate close or a double count. Re-running the archive pass is idempotent AND resumable: a record already on disk is not written again, but its ONLINE copy is still removed and its episode still marked — so an interrupted pass ("file written, delete not done") can finish its cleanup on the next run. Restoring an archived payload is possible for AUDIT, but a restored payload never re-enters the calibration automatically — the window is decided by the registry's `closed_at` ordering.

**Automatic maintenance.** Every close-out runs a LIGHT check (one indexed count of registry rows outside the window and not yet archived). Only when that count crosses `OR_CALIBRATION_AUTO_ARCHIVE_THRESHOLD` (default 200) does an archive pass run, so retention is maintained without archiving on every close.

**Old stores are backfilled.** A store written before the registry existed has `episode_closeout|…` records but no rows. `calibration_window` backfills them once, idempotently, using each close-out's OWN `created_at` as `closed_at` — so the window ordering is historical, not "whenever the migration ran", and an old database's episodes are not silently invisible.

## 7. Window rounds (the identity extension)

`ORHarness.strategy_execution_window(..., round_index=N)` (and the `win::task::episode::strategy::rN` window id) selects ONE selection round of a (task, episode, strategy): the actions from the Nth recorded choice of that strategy up to the next recorded choice of any strategy. The same strategy chosen twice — or switched away and back — is TWO windows with TWO ids; a round that does not exist is reported not-comparable with the reason. Legacy three-part window ids still parse (`round_index=None`).

## 8. Known limitations

- **Planner suggestion vs close-out evaluation (open)**: a candidate with no comparable benefit or failing quality checks can still be SUGGESTED by the planner's conservative yardstick. The close-out's evaluation does NOT depend on the planner's utility or the suggestion — eligibility is decided by the prediction–outcome match and the measurement basis only. Automatic recommendation reliability is not yet claimed.
- **Scoring rules are decision rules, not measurements (open)**: "unknown cost charged the peak share / unknown risk the full weight" are DECISION RULES, not measured facts. The calibration summary reports measured errors only; it never presents those rules as observations.
- **No framework channel can adjudicate a MODELLING error (open)**: a task check reports whether its DECLARED bases held, not whether the model was wrong — the cause (misread task, implementation bug, unmet requirement, wrong reference) is the agent's diagnosis. That is why `model_invalid` is retired rather than derived, and why a risk the framework cannot observe keeps label `unknown`.
- **Budget risk is episode-scoped only (open)**: this round keeps the episode-level over-budget FACT and marks narrower-scope predictions `scope_mismatch`. A per-attempt budget system is deliberately out of scope.
- Late-arriving verification: an evaluation, once written at close-out, is never silently revised — the STORED evaluation is the honest record of what was known then. A task-result check that arrives LATER does change later USE, and the correction is FIELD-SCOPED: the benefit observation is re-derived (to `0.0` for a confirmed-wrong answer) while the measured cost is preserved, the correction is applied only to the execution it names, and the published summary is republished when the episode is still in the window. Rewriting the stored evaluation itself remains refused.
- **Archived detail cannot receive a late correction (open)**: once an episode's detail has been archived (past the grace period), a later task check still annotates the EXECUTION (facts are never archived), but that episode's calibration correction channel is closed. The grace period is the bound on how long that channel stays open — there is no permanent retention path.
- **Constraint satisfaction is only checkable when declared (open)**: the framework does not parse natural-language constraints, so a task check covers the bases you declare (a reference value, a status, integrality, a declared objective recomputation, explicit probes). Undeclared constraints stay in `scope.unchecked` — a `passed` verdict is not proof that the model represents the task.
- **Task-level verdicts are per-execution (open)**: `task_check` annotates one execution's answer. There is no episode-level "the task was solved" flag, deliberately: an episode may contain a disqualified attempt and a repaired success, and collapsing them into one verdict would lose exactly the distinction this layer exists to preserve.

## 9. Verification checklist for a consuming agent

- [ ] Did the episode really end? `close-episode` on an episode with running actions is REFUSED (`state="pending"`, `closeout: null`): end them first, then close — their results could never enter a frozen record.
- [ ] Is the terminal state honest? `failed`/`aborted`/`budget_exhausted` are endings, not failures of the close-out.
- [ ] Am I reading the close-out as a certification of the answer? It is not: read `task_checks` — `unchecked` executions have UNKNOWN validity, and a `failed` one was confirmed not to satisfy the task. A task check passing is also not a knowledge claim passing (`induce --verify` is a separate gate).
- [ ] Am I reading a failed task check as "the model was wrong"? It is not: `task_check_failed` reports the CHECK RESULT, and the cause is your diagnosis. The framework deliberately has no `model_invalid` label.
- [ ] Am I reading `solver_reported_infeasible` as a failure? It is a verdict — correctly identifying an infeasible ORIGINAL problem is a valid outcome.
- [ ] Am I reading an `excluded` evaluation as a miss? Excluded is neither hit nor miss — check `exclusion_reasons`. A withdrawn fact still counts as an OBSERVATION (it really ran); it leaves the prediction-comparison statistics only.
- [ ] Am I reading a late correction as "the sample was dropped"? A failed check RE-DERIVES the benefit (to 0.0) and keeps the measured cost; only a withdrawn execution removes the sample.
- [ ] Am I subtracting `unit_occurrence_rate` from `mean_predicted_probability`? Do NOT: they have different denominators. Compare `scored_occurrence_rate` with `mean_predicted_probability` instead — those share a sample set.
- [ ] Is one execution being counted as several occurrences? Labels are per execution: a scope with one timeout and one success is ONE occurrence, rate 50%. Check `n_observation_units` against the real execution count.
- [ ] Am I reading a group of many episodes as "every statistic is measured"? Check the per-statistic verdicts (`benefit_evidence`, `cost_evidence`, `interval_evidence`, `brier_evidence_by_event`) and `under_sampled_statistics`: a risk only one episode predicted has ONE sample whatever the group size.
- [ ] Am I reading `insufficient_evidence` as a bad reliability? It is the honest unknown: too few samples, no figure claimed. A statistic nobody predicted is ABSENT, not under-sampled.
- [ ] Am I reading the calibration as the CURRENT candidate's conditional bias? It is not: `applicability` says `global_diagnostic` — there is no per-strategy breakdown.
- [ ] Am I comparing another model's error statistics to this model? The context filters by model identity; a withheld group is named under `withheld_groups`.
- [ ] Did a later episode's context change? It should not: stored contexts are frozen; only NEW contexts read the published summary.
- [ ] Am I counting one truth twice? Re-planning predictions bound to the same outcome are marked correlated; the DISTINCT-EPISODE count is the sample base and the threshold counts episodes, never predictions. Occurrence rates are counted by observation unit, so several predictions over one execution are ONE event.
- [ ] Am I treating the calibration as a capability gain? It is a record of past errors. H evidence about the world model comes only from these real evaluations — never from the model's self-assessment.
- [ ] Did a correction reach later predictions? A late check/exclusion/restore on a WINDOW episode republishes the summary; `orx calibration --rebuild` is the explicit repair path.
- [ ] Is the archive bounded? Check `orx inspect --bank retention`: the per-file, total and age caps are all enforced — a total cap is what makes "bounded" true.
