# Stage F review fixes — guidance precision and documentation structure

This round addresses the four review findings without changing the framework:
self-check verification is now sufficiency-driven instead of capped at one
check, H+ verification may use affordable proportionate evidence instead of an
absolute ban on a complete re-solve, `SKILL.md` hands low-frequency knowledge
maintenance to references, and the documentation tests stop depending on an
arbitrary line budget and exact sentences.

No G/C/R, U, H+ binary/scheduling, knowledge-schema, benchmark or CLI business
code changed.

## 1. Self-check: minimum sufficient verification, not a one-check cap

Problem: the flow said `run ONE targeted independent check`, which reads as a
hard limit even when a complex model has several independent doubts.

Before (SKILL §3b):

> Add an enhanced check ... only when there is a concrete reason ...
> Do not run a complete independent re-solve as a default step for every task.

```
run ONE targeted independent check and read the ACTUAL evidence
```

After:

> Use the minimum sufficient verification: keep adding targeted checks while an
> unresolved doubt can still change the conclusion, and stop once the evidence
> is sufficient. Do not run a complete independent re-solve as a default step
> for every task, and do not treat a fixed number of checks as an automatic
> limit.

```
run the cheapest targeted check for the open doubt; read the ACTUAL evidence
   |
   v
does a substantive doubt still block the conclusion? -- yes --> another targeted check
   |
   no
```

`references/self_check.md` Layer 2 now states that verification is
sufficiency-driven, not count-limited: address independent doubts separately,
do not repeat an already-established part, and use a complete re-solve only when
it is the right way to resolve a specific doubt. `references/commands.md`
repeats the same rule next to `check-task`. Failure handling is unchanged: a
script error or assertion failure is never a pass; a check/evidence problem is
fixed on the same execution, and a real model/implementation problem gets a new
`plan-next`.

## 2. H+: proportionate verification evidence

Problem: the prediction prompt and induction reference banned verification
conditions that need a complete re-solve, which also blocked legitimate cheap
checks such as a small full enumeration or an original-versus-reduced
comparison.

Prompt before:

> do not propose a condition that needs a later large experiment or a second
> complete solve of the task.

Prompt after:

> Match the check to the claimed mechanism with MINIMUM SUFFICIENT evidence: a
> complete re-solve is NOT the default, and a later large experiment cannot
> substitute for evidence from THIS run; but an affordable small-scale full
> enumeration, an original-versus-reduced-method comparison, or a targeted
> reproduction IS allowed when it verifies the key mechanism.

`references/induction.md` after:

> Match the check to the claimed mechanism with minimum sufficient evidence: a
> complete re-solve is not the default, and a later large experiment cannot
> substitute for evidence from this run; but an affordable small-scale full
> enumeration, an original-versus-reduced-method comparison, or a targeted
> reproduction can verify the mechanism.

The same wording is aligned in `SKILL.md`, `references/self_check.md` and
`references/commands.md`, so no file says "second complete solve is invalid" any
more. Preserved: H+=1 still needs a concrete checkable increment hypothesis;
the condition must be completable within this run's reasonable scope; an
unfinished condition is recorded `not_checked`/`unavailable`, never upgraded to
checked, and never faked for publication. The binary switch, mandatory review,
verification records and compact submission contract are untouched.

## 3. SKILL.md structure: main flow in, knowledge maintenance out

`SKILL.md` remains the flow document (259 lines). §1, §2, §3a, §3b, §5, §6, the
top diagram, the NL4Opt example and References are retained. Changes:

- **§4 trimmed.** The entry-usage field explanation, full verdict taxonomy,
  `keep`/`demote`/`promote`, retire/restore, `status_view`, `utility` and
  `forward_calibration` details moved out (they already exist in
  `references/induction.md#existing-entry-use-and-utility`, which now also
  documents the `result.entry_usage` block). The Skill keeps: call
  `close-episode`; review the adoptions of this episode; record a judgement
  when the evidence supports one; the `--attribute-effect` call; H+ = 0 also
  works; detailed rules are linked.
- **§5 kept short.** It now states explicitly that H+=1 requires one review and
  H+=0 adds no job, and keeps material → compare → result → `orx induce
  --submission` → `--dry-run`. Version management and long examples stay in
  `induction.md`.
- **§3b kept short.** Two questions, base vs enhanced decision, sufficiency
  wording, main failure branches, `check-task`/`record`, and links to
  `self_check.md` and `commands.md`.
- **References section** lists `self_check.md` so the detailed path is one hop
  from the Skill.

## 4. induction.md: navigation, no duplicate cold start

- Added a short `## Contents` block linking the main sections and subsections.
- Merged the duplicated cold-start sentences into one paragraph in
  "Predict H+ at cold start"; the standard steps are still explicitly not an
  increment, the cold start still does not force 0, and rule coverage is still
  never inferred from similarity alone.
- Preserved: the no-increment rule for ordinary modelling, the independence of
  H+=1 from publication, the four hypothesis-outcome cases, the durable-
  knowledge boundary (no benchmark scores/reference answers as a rule premise),
  `when`/`do`/`expect`, the existing result types, worked flows and the
  verification/publication boundaries.
- `result.entry_usage` reading moved here from the Skill; the duplicate
  `effect_review.pending` sentence was merged.

## 5. Self-check examples no longer use a hidden benchmark answer

`references/self_check.md` example 0087 now judges the domain from the task
statement:

> the task states continuous variables (no integrality requirement) ... Judge
> the variable domain from the task statement, not from any external answer: do
> not add an integer constraint because some outside value happens to be
> integral — that would change the task.

The examples section also states that a reference answer or checker result is
used only AFTER commitment as offline evidence, never as a modelling premise.
The boundary remains in `induction.md`: post-hoc checker/reference results can
support a review but cannot become the applicability premise of a future rule.

## 6. Documentation tests: no line budget, no exact-sentence dependence

`tests/harness/test_stage_f_selfcheck_induction.py` was rewritten (38 tests):

- removed `assertLessEqual(len(skill.splitlines()), 280)`; the size check is now
  a non-blocking print when the Skill exceeds 300 lines;
- checks the main workflow stages and workflow-diagram stages are present;
- checks every `references/` link in the Skill resolves, and every internal
  markdown anchor across the agent-facing docs points at a real heading;
- checks the CLI commands/options the Skill cites exist in the real parser;
- checks the low-frequency entry-management vocabulary lives in the induction
  reference, not the Skill;
- asserts semantic term clusters, not full sentences (sufficiency-driven
  verification, model-before-solver, failed/insufficient handling, H+ condition
  separation, proportionate H+ evidence, standard-translation non-trigger, cold
  start, induction no-increment outcomes, durable-knowledge boundary);
- keeps the executable receipt-flow tests: `no_new_knowledge` with a refuted
  hypothesis, `insufficient_evidence`, refuted hypothesis plus a separate new
  rule, sourced-vs-unsourced condition states, and H+=0 not creating a job;
- keeps the interface-invariance tests (binary H+ version, review outcomes,
  `capability_gain` fields, compact submission results, H+ outside U).

A stale anchor in `references/episode_closeout.md` (`wm-events3` →
`wm-events4`) was fixed so the anchor test can be meaningful.

## 7. Files and locations

- `SKILL.md` — §3b sufficiency wording/diagram, §4 knowledge-maintenance trim,
  §5 explicit H+=1/H+=0 line and `orx induction-status` mention.
- `references/self_check.md` — Layer 2 sufficiency rule, proportionate H+
  evidence, corrected 0087 example, post-hoc-evidence note.
- `references/induction.md` — Contents map, merged cold-start paragraph,
  proportionate H+ verification wording, `entry_usage` paragraph and duplicate
  removal.
- `references/commands.md` — sufficiency wording and proportionate H+
  verification wording.
- `references/episode_closeout.md` — fixed `wm-events/4` anchor.
- `src/or_harness/world_model/strategy_prediction.py` — one prompt passage in
  `capability_gain`: minimum-sufficient evidence rule replacing the absolute
  second-solve ban. No code-path change.
- `tests/harness/test_stage_f_selfcheck_induction.py` — rewritten as above.
- `docs/world_model/stage_f_selfcheck_induction.md` — pointer to this document;
  superseded wording noted.

## 8. Tests

Per the review instruction, only the changed test module was run (plus the
documentation consistency script), because this round is documentation/prompt
work:

| Check | Result |
|---|---|
| `test_stage_f_selfcheck_induction` | **38 / 38 OK** |
| `references/examples/_check_docs.py` | OK (flags, symbols, links, result keys) |

Unchanged suites were not run. Known pre-existing failures from the previous
round are unchanged by this round (no production code path changed):
`test_documented_result_keys.test_inspect_capability`,
`test_command_surface.test_every_removed_query_is_reachable_through_inspect`,
two `test_command_surface.TestSameConventionCompletionCompares` tests.

## 9. Still needs real-agent validation

- Whether a real solver agent now stops after sufficient evidence instead of
  adding another full solve, and whether it adds a second targeted check when a
  genuine independent doubt remains.
- Whether H+ verifiers actually use affordable small-scale checks instead of
  either avoiding them or over-running large re-solves.
- The H+ positive rate and self-check effort are not targets of this round and
  were not measured; a real experiment is still required.
