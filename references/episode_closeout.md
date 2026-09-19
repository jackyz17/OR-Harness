# Episode close-out and experience calibration (world-model M4)

**Status: implemented.** This is the fourth phase of the reconstruction:
after the unified contract (phase 1), the frozen prediction input context
(phase 2) and the strategy-outcome prediction service (phase 3, wm-so/1),
this phase closes the loop — a real episode ends, its bound predictions
are evaluated against their real outcomes field by field, and the
aggregate becomes experience calibration that later episodes read.

- Close-out record version: `wm-closeout/1`; calibration summary version:
  `wm-calib/1`
- Python module: `or_harness.world_model.episode_closeout`
- API: `ORHarness.close_episode` / `episode_closeout_record` /
  `get_strategy_evaluation` / `strategy_prediction_evaluations` /
  `calibration_summary`
- CLI: `orx close-episode` / `orx calibration` / `orx evaluations`
- Tests: `tests/harness/test_episode_closeout.py`
- Runnable example: [`references/examples/episode_closeout.py`](examples/episode_closeout.py)

Related: the prediction service in
[`references/strategy_outcome.md`](strategy_outcome.md), the input context
in [`references/prediction_context.md`](prediction_context.md), the
contract objects in [`references/world_model_contract.md`](world_model_contract.md).

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
- **strategy execution window** — one selection round of one strategy:
  modelling, possibly several attempts, repair, verification. The same
  strategy chosen again later (or switched away and back) is a DIFFERENT
  round (`round_index`), never aggregated into one sample;
- **episode** — the whole task interaction, closed once, honestly.

## 2. Closing an episode

```bash
orx close-episode --task t1 --episode ep1 [--terminal completed|failed|aborted|budget_exhausted] \
    [--finish-action ACTION_ID] [--min-samples N]
```

What it does (in order):

1. **refuses while actions are still running**: an episode with a
   `running` action returns `state="pending"` with the
   `unfinished_actions` listed — a running scope has no final numbers,
   and closing anyway would freeze a record their results could never
   enter (a re-close just returns the stored record). End the running
   actions (or let them finish), then close;
2. **summarizes and evaluates** every BOUND strategy-outcome prediction
   of the episode against its real outcome (see §3–4);
3. **records the close-out once** — idempotent: re-closing (after a
   restart, or by mistake) returns the stored record and counts nothing
   twice;
4. **publishes the calibration summary** (see §5) — built AFTER the
   evaluations and the close-out record are persisted, so the first
   close-out's own return already includes this round's samples.

What it never does: run a solver, call the prediction model, trigger
induction, or rewrite a stored prediction. `--terminal failed|aborted|
budget_exhausted` closes an honest failure as a failure — only
`completed` claims success. `--finish-action` links the close-out to the
`finish_task` action you recorded (`orx action --report finish_task`);
the close-out does not create a second task lifecycle.

## 3. The real outcome summary

For each bound prediction, the close-out derives what ACTUALLY happened
from the action log and the execution records — never from the
prediction:

- **benefit observations** use the prediction's OWN declared
  metric/unit/baseline (the yardstick is frozen at prediction time; it is
  never re-chosen after the result is known). This build has an
  observation adapter for exactly ONE metric — the solver's normalized
  gap (`optimal` → 1.0; otherwise `1-gap`). A prediction declaring any
  other metric (a business cost-saving ratio, a completion rate) is
  reported `scope_mismatch`: the solver's 1-gap number is never
  re-labelled as a metric it did not measure. A feasible solution with
  no gap/bound is the 0.5 heuristic — reported `unverified`, never
  scored. Other benefit kinds have no observation adapter either:
  reported, not evaluated.
- **window benefit rule** (declared before evaluation, applied
  uniformly): the LAST qualified in-scope attempt's solution is the
  window's benefit observation. All in-scope failed retries' measured
  costs still count.
- **cost** is scoped: an ATTEMPT-scope prediction is compared against
  the bound action's own execution; a STRATEGY-WINDOW-scope prediction
  is compared against the WHOLE window of its selection round — every
  in-scope execution, failed retries included (an 11s-then-17s window
  reports 28s, not 17s). Modelling/verification/other attempts are
  auxiliary overhead — real spend, reported separately, never charged to
  the predicted scope. Per-dimension totals carry completeness
  (`complete` only when every in-scope item measured the dimension);
  `latency_s` is never summed.
- **risk events** get labels `occurred` / `not_occurred` / unknown —
  and only events with a real OBSERVATION CHANNEL can ever be labelled:
  the in-scope executions' statuses and failure classes
  (`model_invalid`, `no_feasible_solution`, `timeout`,
  `environment_failure`/`model_failure`) and the episode's budget view
  (`budget_exhausted`). A business risk with no observation channel keeps
  label `unknown` whatever the logs show — "no failure log" is not
  evidence it did not happen — and is excluded from scoring. A completed
  scope with no such observed event is a `not_occurred` label of ONE
  trajectory (the basis says so).
- **verification**: solver `optimal` does not prove business
  requirements; the summary reports the verification evidence that
  exists (verify actions, executor checks), nothing more.

## 4. The per-field evaluation

Each field is judged on its own eligibility — never a blanket verdict:

| Field | Compared when | Excluded when (and why it matters) |
|---|---|---|
| benefit | the prediction declared a value AND the same metric was observed under the same scope | no baseline/value predicted (`not_predicted`); no qualified observation (`unobserved`); heuristic-only quality (`unverified`); other metric kinds (`scope_mismatch`); binding identity not established (`identity_mismatch`) |
| cost | per dimension: predicted AND completely measured on the real scope; error = absolute + log-ratio | partially-measured totals are excluded (an incomplete total is not a truth); zero predicted or actual produces no relative error |
| risk | per event: a predicted probability AND a known label → Brier score | no probability (`not_predicted`); unknown label (never a 0 label by default); events are never averaged across names |
| interval | the prediction saved an interval AND an observed value exists → covered yes/no | no interval saved, or nothing comparable to cover |

Excluded fields are **neither hits nor misses**: they never enter a
denominator. An evaluation with nothing comparable is `excluded` with the
per-field reasons; a scope still running is `pending`. The FROZEN
prediction is read, never rewritten — no post-hoc "better" prediction is
regenerated closer to the result.

**Unknown identity is not a match.** The M3 binding records
`binding_mismatch` (a KNOWN disagreement) and `binding_unknown` (a field
the executed action could not observe — no episode recorded, a config key
the action log never carries) separately. A mismatch blocks the
evaluation outright; an unknown makes the fields that depend on it
unevaluable. Re-binding the same action re-evaluates the comparability
(an action still running at first bind may have completed since).

## 5. The experience calibration summary

```bash
orx calibration [--min-samples N]
```

Aggregated from **closed episodes only** — an open episode never
calibrates anything, least of all itself. Grouped by
(metric, unit, scope) so different definitions never mix (the same
metric under a different unit or scope is a DIFFERENT group), risk
events are scored PER EVENT NAME (different events are different random
variables; their Brier scores are reported separately, never pooled
into one mean), and each group reports: sample count, DISTINCT episode
count, mean absolute benefit error, mean per-dimension cost log-error,
interval coverage rate, and per-event Brier means. The sample threshold
counts DISTINCT EPISODES (independent truths) — one truth bound to
several re-planning predictions is marked `correlated_predictions`,
never counted as independent samples, so five predictions over one
execution cannot cross the threshold. An episode's identity is the full
`(task_id, episode_id)` pair, because episode ids are chosen per task:
five different tasks that each used `ep1` are five independent
task-episodes, and deduplicating on the bare `episode_id` would collapse
them into one sample. A group below the sample minimum
(`--min-samples`, default 5, effective value and basis recorded on the
summary) reports `insufficient_evidence` with `reliability: null` — no
figure is invented.

This is a **measured record, not a promise**: writing the summary does
not claim future predictions improve, and it is NOT a fitted calibrator
or a model weight (training-free is unchanged). It is kept SEPARATE from
the legacy knowledge-prediction reliability (`prediction_class_reliability`):
a knowledge hit rate never proves OR strategy-outcome accuracy.

**When later episodes see it.** A NEW `PredictionContext` carries the
published summary's CONTENT (groups, counts, sources — not an id) under
`strategy_outcome_calibration`, sent to the provider as part of the
context. Stored contexts never change: episode 1's context keeps what it
froze, episode 2's context reads what episode 1 published. An active
episode does not see its own not-yet-closed feedback.

## 6. Window rounds (the identity extension)

`ORHarness.strategy_execution_window(..., round_index=N)` (and the
`win::task::episode::strategy::rN` window id) selects ONE selection round
of a (task, episode, strategy): the actions from the Nth recorded choice
of that strategy up to the next recorded choice of any strategy. The same
strategy chosen twice — or switched away and back — is TWO windows with
TWO ids; a round that does not exist is reported not-comparable with the
reason. Legacy three-part window ids still parse (`round_index=None`).

## 7. Known limitations (deliberately deferred to M6)

- **M3 item 3 (open)**: a candidate with no comparable benefit or failing
  quality checks can still be SUGGESTED by the planner's conservative
  yardstick. The close-out's evaluation does NOT depend on the planner's
  utility or the suggestion — eligibility is decided by the
  prediction–outcome match and the measurement basis only. Automatic
  recommendation reliability lands in M6.
- **M3 item 4 (open)**: the scoring rules "unknown cost charged the peak
  share / unknown risk the full weight" are DECISION RULES, not measured
  facts. The calibration summary reports measured errors only; it never
  presents those rules as observations.
- Late-arriving verification: an evaluation, once written at close-out,
  is not silently revised. A late verification fact requires an explicit
  new close-out of a LATER episode (or a manual re-derivation); this
  build refuses to rewrite a frozen result and reports the stored one.

## 8. Verification checklist for a consuming agent

- [ ] Did the episode really end? `close-episode` on an episode with
      running actions is REFUSED (`state="pending"`): end them first,
      then close — their results could never enter a frozen record.
- [ ] Is the terminal state honest? `failed`/`aborted`/`budget_exhausted`
      are endings, not failures of the close-out.
- [ ] Am I reading an `excluded` evaluation as a miss? Excluded is
      neither hit nor miss — check `exclusion_reasons`.
- [ ] Am I reading `insufficient_evidence` as a bad reliability? It is
      the honest unknown: too few samples, no figure claimed.
- [ ] Did a later episode's context change? It should not: stored
      contexts are frozen; only NEW contexts read the published summary.
- [ ] Am I counting one truth twice? Re-planning predictions bound to
      the same outcome are marked correlated; the DISTINCT-EPISODE count
      is the sample base and the threshold counts episodes, never
      predictions.
- [ ] Am I treating the calibration as a capability gain? It is a record
      of past errors. H evidence about the world model comes only from
      these real evaluations — never from the model's self-assessment.
