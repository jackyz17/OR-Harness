# Agent-led induction and existing-entry management

The agent chooses the method, comparison purpose, evidence scope and publication.
The framework keeps execution facts, immutable rule versions, references and
retry state. G/C/R still selects solving strategies; the frozen H+ only decides
whether to spend a new induction round.

## Close and consume H+

Use `predict-strategy` → execute → check → record → close. A real bound H+=1
must complete one review, including after failure or missing checks. H+=0 skips
new induction but keeps execution evidence and ordinary existing-rule use.
A prediction never executed remains unobserved. Missing/invalid H+ schedules
0 with its separate diagnostic; historical three-state records keep their meaning.

```bash
orx close-episode --task T --episode E
orx induction-status --task T --episode E
orx induction-material --induction-id J
orx induce --induction-id J --review '{"outcome":"insufficient_material","conclusion":"Actual facts reviewed; support is insufficient."}' --note "Reviewed available evidence."
```

`J` is returned in `induction_schedule.pending_ids`. The outer building agent
consumes each pending job with a short submission or reasoned empty review.
`process_induction_jobs(reviewer, ...)` provides the same callback handoff.
Close-out never invents a rule or calls a second hidden model. A resumed runner
reuses the job, actual induction action and final receipt. Omitted episode
resolves only one unique actual round; ambiguity returns readable existing
execution choices (`close-episode --task T --execution EX`).

An empty review is submitted without `--relation`, with a reason or structured
review receipt. Placeholder claims such as "no new knowledge" are not reusable
rules and are rejected rather than stored as method bodies.

## Query current material

```bash
orx induction-material --induction-id J --purpose modelling
orx induction-material --induction-id J --purpose implementation --query "endpoint direction and fallback" --related-top-k 3
orx induction-material --induction-id J --executions ex_A,ex_B --limit 2
orx induction-material --induction-id J --cursor 'CURSOR_FROM_BUDGET'
```

Every call reads current evidence. There is no query-key cache or saved full
response to replay. The primary case is J's actual triggering execution;
newer same-task attempts remain selectable controls. In a formal job, the
anchor stays first while `limit`/cursor paginate controls, so an older anchor
cannot hide newer attempts. Manual batches retain ordinary newest-first paging.

The default `induction-material/2` pack contains actual methods and their sources,
plans separately, original checks/unchecked scope, costs and provenance, failures,
related cases and short existing rules. Source links and exact original-input
matches are advisory overlap information. The agent decides what they establish;
text-number heuristics, file paths, family labels and mother-task fields do not
rule on transfer or publication. Repeated references to an execution are not new
physical uses. Similarity only finds candidates.

The 32000-character default counts the entire result and CLI envelope, controlled
by `OR_HARNESS_INDUCTION_MATERIAL_CHARS`. Oversized content is a whole-value
reference. `budget.next_cursor` reports omitted controls. Use:

```bash
orx inspect --bank experience --execution ex_A
orx inspect --bank strategic --entry 3 --rule-version candidate
orx inspect --bank values --value 'rule_fact:...'
```

Pending jobs temporarily hold their trigger and selected/delivered necessary
cases through normal window eviction and restart. Sessions retain IDs and fact
references rather than full packs. Completion releases full temporary execution
copies; minimal original facts/checker references remain for audit. Ordinary
`check-task` does not freeze a second full record. Other raw executions continue
through the normal evidence window. Historical saved packs remain readable.

## Submit a short rule and choose publication

```bash
orx induce --induction-id J --relation '{"when":"For fixed Y, feasible integer X forms an interval and the objective is monotone in X.","do":"Enumerate Y, derive both X endpoints and select by coefficient sign and objective direction. Disable this reduction for nonmonotone objectives and use full enumeration.","expect":"Preserve the stated integer optimum and constraints.","evidence":["ex_A","ex_B","ex_constructed"],"publication":{"state":"published","reason":"Agent checked the endpoint derivation, enumerations and nonmonotone fallback.","scope":["stated integer-interval mechanism and listed checks"]}}'
```

`publication` is one small author choice: `state:"published"` or `"candidate"`,
a short `reason`, and optional declared `scope`. Omission leaves a new body as
candidate; it does not trigger a framework content judgment. Publishing offers
that exact version in normal retrieval immediately. Actual verification scope
is reported beside the author's judgment. A solver status, a numerical reference
comparison and a method proof cover different claims; the report keeps them apart.

Derivation, independent benchmarks, exhaustive constructed cases and counterexamples
can support their stated conclusions. The agent can publish an endpoint mechanism
from A/B plus derivation and a constructed boundary without waiting for a third
real task. A later real C can then be recorded as transfer. One unchecked
observation can remain candidate. Transfer/boundary/regression are optional
review purposes, not a uniform publication checklist.

| Operation | Behavior |
|---|---|
| New paragraphs and evidence | New candidate body; explicit publication may offer it now |
| `entry_id`, `operation:"revise"`, paragraphs | New immutable version under the same ID; the active version remains until chosen otherwise |
| `entry_id`, `operation:"add_evidence"` | Reuse the body/version; append evidence or optional validation |
| `entry_id`, `operation:"merge"`, `merge_entry_ids`, reason | Agent selects and confirms the target; confirmed sources leave normal recall when the target is offered |

Safe formatting equivalence reuses an existing body. More complex equivalence is
an agent judgment. There is no automatic list of old successes that must be
re-run. Review a few discriminating cases, re-run where needed, or explain why
recorded derivation/facts suffice. Original checker versions remain traceable.
A reference-only `check-task --check '{"reference_objective":2699}'` establishes
that numerical comparison, with its declared source; it does not fabricate
constraint coverage. Missing acceptance is unknown.

Check that each declared assertion matches the claim it supports. A failed
assertion is not automatically a refutation of the whole rule. Record what the
check actually establishes; the agent decides attribution and utility management.
Existing assertion kinds such as `probe`, `status`, `comparison`, `code_changed`
and `code_unchanged` retain their declared scope; a changed code hash is not proof
that the method improved or that the original task was accepted.

## Existing-entry validation and utility need no positive job

At execution declare the actual adopted version with `--used-entry-ids 3@2`.
Omitted adoption is unknown; `''` declares none. The frozen reference preserves
that exact version. Historical v1 facts may support an agent's v2 compatibility
review, but only an actual v2 adoption generates a v2 use event.

Normal check/record/close saves use and scope. Optional `rule_assessments` can
record the agent's verdict and purpose:

```json
{"entry_id":"3","version":2,"purpose":"transfer","verdict":"supported",
 "source":"agent inspected the actual script, original inputs and declared checks",
 "assessment":"The chosen endpoint matches the objective direction; only the listed checks are claimed.",
 "evidence_refs":["artifacts.submitted_script","artifacts.workspace_result"]}
```

Existing-body evidence addition, validation or publication selection can use
`induce --relation` without a new job, provided it selects an existing entry and
contains no new/changed body. The existing `--attribute-effect` is the utility
management path:

```bash
orx induce --attribute-effect '{"entry_id":"3","execution_id":"ex_USED","verdict":"helped","lifecycle_action":"keep","note":"Agent reviewed the actual adopted use and declared acceptance."}'
orx retire --entry 3 --reason 'Agent concludes this method should leave normal recall.'
```

`promote`, `demote`, `keep` use the same lifecycle for legacy and versioned
entries. Management names real adopted evidence and a short reason. The version
comes from frozen adoption; multiple versions require choosing an existing one.
Management of an old version does not silently demote a different current version.
Pure maintenance returns `maintenance_only:true`, state differences and index
results without a new induction action, positive job or empty review receipt.
New/changed bodies still need authorized induction. Final retirement uses the
existing cold archive.

## Cost and corrections are factual inputs

The agent selects a cost baseline and decides comparability and attribution.
Optional `comparator_execution_id` and `cost_dimensions` on assessment/validation
return both actual methods, measurement scopes, execution conditions, acceptance
rules, values and provenance. No baseline search, dimension substitution or
framework advantage verdict occurs. `cost_eligibility` distinguishes measured,
estimated, partial and unknown values. For 9→10 calls and 100→25 tokens, the
report gives +1 call and −75 tokens separately; it does not say calls decreased.

Omitted optional assessments on unchanged facts preserve the prior assessment.
Explicit correction/withdrawal remains visible. `exclude-execution` and
`restore-execution` change evidence validity; current material omits excluded
facts and future H+ support reflects their validity. They do not trigger global
rule revalidation, automatic pause/recovery or publication. The agent evaluates
counterexamples and manages the entry. Valid failed executions remain useful
facts, not records to exclude merely for failure. Historical inputs, original
checks and exact adopted versions remain auditable.

For learning conclusions and the frozen switch see [hplus.md](hplus.md).
Runnable A/B publication and later C management:
`PYTHONPATH=src python docs/validation/run_review_eight_demo.py`.
