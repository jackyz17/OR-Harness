# Stage F — risk-ordered self-check and stricter H+ induction

> Review-round fixes (proportionate verification guidance, `SKILL.md` §4 trim, induction navigation, benchmark-example boundary and relaxed documentation tests) are documented in [stage_f_review_fixes.md](stage_f_review_fixes.md). Where wording below differs, that document is authoritative.

This stage addresses two behaviour problems in the solving agent:

- **Self-check too heavy.** Build4 showed 28/30 tasks with an independent
  re-solve path and ~2 verification-script runs per task on average. Some
  failures had already been checked by several methods yet still missed
  objective, domain or constraint-semantics errors.
- **H+ and offline induction too permissive.** 27/37 H+=1 triggers produced 21
  `new_rule` and 5 `revision` reviews. Many were ordinary modelling steps, and
  only one review explicitly recorded that its pre-registered verification was
  completed.

The change is deliberately guidance-first: `SKILL.md`, the H+ prompt in
`strategy_prediction.py`, `references/induction.md` and aligned references. No
new framework layer, tool, agent, classifier or knowledge type is introduced,
and the G/C/R retrieval, U scoring, H+ binary contract, episode/execution
binding, induction scheduling and knowledge storage are unchanged.

## 1. Self-check guidance — before / after

`SKILL.md` §3b previously presented one flow ("choose a verification method")
and a minimum list of dimensions. It mentioned that `optimal` only solves the
written model, but did not state the model-vs-task distinction as a rule, did
not bound when an independent check is justified, and did not say how a failed
or errored check is handled. `references/commands.md` repeated the same
single-flow framing.

After the change, `SKILL.md` §3b keeps the two questions, the base/enhanced
decision, the flow and the `check-task` contract in a compact form (the entry
doc went from 298 to 262 lines), while the detailed check lists, the
model-vs-task explanation, failure handling and all worked examples live in the
new [references/self_check.md](../../references/self_check.md). The
`check-task` interface is unchanged. The substantive change:

| Aspect | Before | After |
|---|---|---|
| Framing | One flow: "choose a verification method: hand calculation, code review, an independent verification script, enumeration, a bound/dual, or declared deterministic checks" | Two layers: a **necessary base check** on every task, then **enhanced verification only when needed** |
| Base check | A minimum list in one paragraph | Explicit dimensions: objective meaning, variable meaning/domain, units and billing range, key constraints (coupling), feasibility, objective/residual/status agreement, optimality evidence when required |
| Default re-solve | Not prohibited; the flow invited a verification method per task | "Do NOT run a complete independent re-solve as a default step for every task"; a second algorithm is still correct when it is genuinely how the problem gets a reliable answer |
| Model vs task | "The solver's `optimal` status only says the model you wrote was solved" | Explicit rule: a solver optimum proves optimality of the written model; a second solver on the same wrong model returns the same wrong answer; a dual/enumeration can share a wrong interpretation — check model-to-statement correspondence first |
| Failed check | "a bad check or missing evidence is corrected and re-run on the SAME execution" | Adds: analyse whether the task reading, model, implementation or check is wrong; `script_error`/`assertion_violations` is never a pass; an unhandled failed verification cannot be marked done or passed |
| `insufficient` | "Neither a pass nor a failure" | Adds: it does not by itself require a complete re-solve |
| H+ verification | Mentioned only in `induction.md` | §3b and `commands.md` state that `verification_conditions` test the knowledge hypothesis, not a second task requirement; reuse evidence, record `not_checked`/`unavailable`, never run a second full solve just to show the condition complete |

The flow diagram now asks "is there a concrete reason for enhanced
verification?" and runs targeted checks until the evidence is sufficient before the same
finding-location decision tree. The reference adds two short examples for the
new rules: "two solvers agree on the same wrong model" and "a failed
verification script is not a pass"; the existing B018/0087 examples moved there
as well. The NL4Opt walkthrough in `SKILL.md` now says the small exhaustive task
needs only the base check and must not add a second solver run.

`references/commands.md` §5 keeps a one-paragraph summary next to `execute` and
links to the reference, and `references/episode_closeout.md` states that a
framework `passed` covers only declared checks and that a failed check or broken
verification is corrected on the same execution, never silently accepted.

## 2. H+ prediction — tightening the trigger

The removed loophole was the prompt sentence:

> Reusable modelling knowledge also includes translating the task statement
> into decision variables, objectives and constraints, together with the
> assumptions and applicability boundaries of that translation.

It is replaced by an explicit rule set in
`src/or_harness/world_model/strategy_prediction.py` (system prompt,
`capability_gain` block):

- **One question.** After execution and review, is there a concrete, reasonable
  and CHECKABLE opportunity to learn a reusable method, mechanism or
  applicability boundary that the library does not cover and that could change
  a future modelling or solving decision?
- **What H+ is not.** Not a difficulty rating, not a success forecast, not
  "a trick can be summarised", not "the task has new variables/constraints",
  not "a reusable-sounding rule can be written".
- **Standard translation is not an increment.** Naming the objective, capacity
  constraint, integer declarations, time index, ordinary LP/MILP solve and
  routine result check as task-necessary steps that do NOT justify 1.
- **Positive triggers.** An easily confused business constraint mechanism with
  clear applicability conditions; a published method that fails under a
  structure this run can test; a structure-preserving reduction; a transferable
  solving-strategy combination; a reusable solver-API/implementation failure
  mechanism with a reliable repair; a substantive uncovered condition or
  operating step. The test is whether the finding can change how a future agent
  MODELS or EXECUTES a task.
- **Cold start both ways.** An empty library is not a reason for 0; "no
  published rule covers it" is only necessary background, never sufficient.
- **Output shape unchanged.** One sentence in `claim` covering mechanism,
  addition, checkable fact and future decision; one check in
  `verification_conditions` feasible during this run (no later large
  experiment as a substitute; affordable small-scale checks are allowed); `basis`/`applies_to`/
  `expected_changes` as before. No new schema.
- **H+=0.** Candidate-specific reason in `basis`, `claim` omitted. A rule ID,
  version and applicable conditions are cited only for "already covered"; never
  invented, and never inferred from similarity.

The frozen binary semantics are unchanged: H+ is still outside U, H+=1 still
always schedules exactly one review after a bound real attempt (including
failure and incomplete verification), and the review can conclude that nothing
was learned.

### Judgement examples used by the new prompt (from the requirement set)

| Situation | Expected H+ | Why |
|---|---|---|
| Translate a stated capacity limit into one linear constraint | 0 | standard task translation; no new mechanism |
| Standard LP/MILP solve, no increment proposed | 0 | solving work, nothing reusable to add |
| Empty library, but only an ordinary modelling step available | 0 | cold start does not turn a standard step into an increment |
| A concrete, checkable structural reduction | 1 may be justified | reusable mechanism with a feasible check |
| New boundary where a published method fails | 1 may be justified | checkable applicability boundary |
| A published applicable rule fully covers the method | 0, cite the real rule | covered; cite actual ID/version/conditions |
| A method might work but no concrete checkable hypothesis exists | 0 | no concrete increment |
| A transferable solver-API failure/repair mechanism | 1 may be justified | reusable implementation fact with an observed repair |

The examples test the principles; they do not force 1 for complex tasks or 0
for simple ones.

## 3. Review standard — refusing an increment

`references/induction.md` now states that H+=1 schedules a review but does not
prove that knowledge was gained. The review compares, in order: the frozen
claim and conditions, the actual execution/checks, the existing rules and
history in the material, and whether the verification condition was actually
completed. It then chooses `new_rule`, `revision`, `evidence_added`,
`already_covered`, `no_new_knowledge` or `insufficient_evidence` from the
evidence. `new_rule`/`revision` are never required to settle a positive job;
writing no rule is a successful review when the facts support that conclusion.

Key additions:

- **Hypothesis outcome stays separate.** Four cases are named explicitly:
  claim supported, claim refuted, claim not sufficiently verified, and a
  different finding discovered during the run. Case 4 is not reported as case
  1. The existing `hypothesis_outcome` / `deviation_reason` / verification
  records / `basis` are reused; no second hypothesis-status system is added.
  An unfinished condition stays `not_checked`/`unavailable`/unknown and is
  never upgraded to checked to close the job.
- **Durable knowledge boundary.** A published rule is used BEFORE solving a
  future task, so `when` must be decidable at that decision point and `do`
  executable. Benchmark scores, a test set's reference answers, hidden-reference
  differences, evaluation-result-dependent strategies, test-specific patches
  and "use the reference's domain" rules must not be stored. Reference answers
  and checkers may be cited as POST-HOC verification evidence only. No
  `benchmark_specific` or `modelling_translation` knowledge type is added.
- **Verification tied to claim type.** Exact derivations scope a rule without
  independent task repeats; empirical success/cost/transfer claims need
  independent tasks with comparison; single-run technical facts stay scoped
  facts; unverified generalisations stay hypotheses. No second complete solve
  is demanded for every entry.
- **Worked flows** for `new_rule`, `no_new_knowledge` and
  `insufficient_evidence`, including what each receipt looks like and why a
  no-write result is a completed review.

`SKILL.md` §5, `references/concepts.md`,
`references/world_model_contract.md` and `references/episode_closeout.md` were
aligned with the same standard; the receipt/feedback semantics are unchanged.

## 4. Interface compatibility

No contract, storage or scheduling change:

- `capability_gain` keeps `decision`, `claim`, `basis`, `applies_to`,
  `expected_changes`, `verification_conditions`; the binary H+ protocol
  (`wm-hplus-binary/1`) and its scheduling/receipt paths are untouched.
- The compact induction submission still accepts the same six `result` values
  and keeps `hypothesis_outcome`/`deviation_reason`; `normalize_submission`,
  `induce`, `induction-material`, `induction-status` and
  `orx induce --submission` are unchanged.
- G/C/R retrieval, definitions, cost normalisation, U and early/late
  coefficients, execution evidence, strategic knowledge storage, task/episode
  binding and benchmark evaluation rules are untouched.
- Single-candidate and multi-candidate planning follow the same path; the new
  prose only changes what the agent/reviewer is told to consider.

## 5. Files and core locations

Modified:

- `SKILL.md` — §3b compacted to the two questions, the base/enhanced decision,
  the flow and the `check-task` contract (298 → 262 lines; detail moved out);
  §5 review standard condensed with links; NL4Opt base-check note.
- `src/or_harness/world_model/strategy_prediction.py` — the
  `STRATEGY_OUTCOME_SYSTEM_PROMPT` `capability_gain` instructions (prompt text
  only; no code path or schema change).
- `references/commands.md` — §5 keeps a one-paragraph self-check summary, the
  result-state analysis and the failed-verification/H+ notes, and links to the
  new reference.
- `references/induction.md` — increment definition and non-triggers, cold-start
  nuance, review decisions, hypothesis/result separation, durable knowledge
  boundary, verification-claim matching, worked flows.
- `references/concepts.md`, `references/world_model_contract.md`,
  `references/episode_closeout.md` — aligned H+ definition, review outcome and
  self-check/verification wording.
- `README.md`, `README_zh.md`, `references/examples/_check_docs.py` — reference
  map and agent-facing doc list include `self_check.md`.
- `tests/harness/test_history_feedback_budget.py` — the end-to-end
  history-first fixture budget moved from 26,000 to 28,000 tokens (the system
  prompt grew ~1.7k chars; 26k had been within ~200 tokens of the send bound,
  so the fixture was recalibrated above the protected core while still forcing
  history-first trimming).

New:

- `references/self_check.md` — the detailed self-check reference: two-layer
  criteria, model-vs-task rules, H+ verification conditions, failed/insufficient
  handling, report reading and all worked examples (B018, 0087, 0312, failed
  verification script).
- `tests/harness/test_stage_f_selfcheck_induction.py` — 38 tests: SKILL
  conciseness/link contract, self-check reference contract, H+ prompt contract,
  induction-review contract, receipt-level review flows, and interface
  invariance.

## 6. Tests

Targeted regression run from the repo root with
`TMPDIR=$PWD/.tmp_tests PYTHONPATH=src:tests/harness:. python -m unittest ...`:

| Module | Result |
|---|---|
| `test_stage_f_selfcheck_induction` (new) | 38 / 38 OK (review round) |
| `test_hplus_binary` + `test_induction_submission` + `test_hplus` + `test_hplus_experiments` | 53 / 53 OK |
| `test_task_check` + `test_episode_closeout` + `test_offline_induction` + `test_r14_induction` | 130 / 130 OK |
| `test_history_feedback_budget` | 25 / 25 OK |
| `test_calibration_v2` | 63 / 63 OK |
| `test_numbered_entries` + `test_r_technical` + `test_documented_result_keys` + `test_command_surface` | 175 ran; 171 OK, 4 pre-existing failures/errors on the clean base `c387cae` (verified in a separate worktree) |
| `test_prediction_context` | 105 / 105 OK |
| `test_strategy_outcome` + `test_stage_e_history_retrieval` | 108 / 108 OK |
| `references/examples/_check_docs.py` | OK (flags, symbols, links, result keys; includes `self_check.md`) |
| `test_world_model_contract.TestDocumentationMatchesCode` | 2 / 2 OK |
| `python -m py_compile` on the edited module | OK |

Pre-existing failures confirmed on the clean base (`c387cae`), unrelated to
this change: `test_documented_result_keys.test_inspect_capability`,
`test_command_surface.test_every_removed_query_is_reachable_through_inspect`,
and two `test_command_surface.TestSameConventionCompletionCompares` tests.

## 7. Remaining limits / what still needs real-agent validation

- The change is guidance and prompt wording. Whether a real model actually
  stops per-task full re-solves, and whether H+ positive rate falls, must be
  measured with a real agent experiment; no large experiment was run here.
- The H+ judgement examples are principles, not a classifier: the framework
  still cannot decide whether a `when` condition is semantically "standard
  translation". The review agent is the judge.
- The system prompt grew by ~1.7k characters (24,997 → 26,654). Deployment
  budgets at the default 64k input budget are unaffected; a deliberately tiny
  26k-token fixture had to be recalibrated (see above). `OR_HARNESS_INPUT_BUDGET_TOKENS`
  can be raised for tighter deployments if needed.
- The self-check failure handling is not mechanically enforced: no new code
  prevents an agent from ignoring a failed verification script, but the
  `script_error`/`assertion_violations` evidence and the close-out report make
  it visible, and the guidance now names it as a self-check failure.
- `SKILL.md` stays a flow document: detailed criteria, report fields and worked
  examples are in `references/self_check.md`, and a regression test enforces a
  workflow stages and reference links plus the absence of the most detailed
  reference-only material; it has no fixed line budget and prints a
  non-blocking size note instead.
