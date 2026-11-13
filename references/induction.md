# Induction: compare executions, freeze a rule, test transfer

The normal interaction is one bounded review round and one short submission. The
agent identifies a mechanism and its necessary premises; the framework
budgets material, resolves evidence, freezes versions and records checks.
Execution evidence and reusable rule bodies remain separate.

## Close and consume the frozen H+

The SAME `predict-strategy` call fixes `capability_gain.decision` to 0 or 1.
A bound, actually executed 1 MUST be reviewed once after task close-out,
even after failure, insufficient checking or a disproved hypothesis. A 0
skips induction; omitted/illegal signals schedule 0 with separate diagnostics.
Historical three-state/missing predictions retain their original meanings.
G/C/R strategy scoring is unchanged.

```bash
orx close-episode --task T --episode E --terminal completed
orx induction-status --task T --episode E
orx induction-material --task T --induction-id J
orx induce --induction-id J --relation '{"when":"...","do":"...","expect":"...","evidence":["ex_REAL"]}'
# Or complete the same positive job without writing a rule:
orx induce --induction-id J --review '{"outcome":"insufficient_material","conclusion":"Actual evidence reviewed; no independent check yet."}' --note "..."
```

`J` comes from `close-episode.result.induction_schedule.pending_ids`.
The outer building agent MUST consume the handoff and require
`induction-status.completed:true` before freezing. Python runners can use
`process_induction_jobs(reviewer, task_id=T, episode_id=E)`; its callback
returns normal `induce` arguments. No hidden model or automatic rule author
is introduced. Both material and writing authorize the same positive job.
An explicit ID or a matching material/evidence reference resolves it; ambiguous
jobs require an ID. A negative execution can be a positive job's contrast.

New prediction homes automatically enter H+ mode; `--induction-mode hplus`
can fix it explicitly. NoH+ and seeded equal-budget random controls use
separate homes. A post-close allow/skip/pending judgement cannot flip the
new signal. The old explicit transition survives only for old/manual work.
Detailed signal, receipt, recovery and experiment examples: [hplus.md](hplus.md).

## Read a small comparison pack

```bash
orx induction-material --task T --induction-id J
orx induction-material --task T --induction-id J --purpose implementation
orx induction-material --induction-id J --query "endpoint direction and fallback" --related-top-k 3
orx induction-material --induction-id J --executions ex_A,ex_B
```

The default `induction-material/2` response contains the current task,
related executions, short existing rules and their states. `purpose=modelling`
prioritizes other source problems sharing constraint relationships;
`implementation` prioritizes actual failures/repairs and implementation
contrasts. Recorded methods, saved problem text and retained structure feed
one existing embedding/index lookup. Similarity, family and structural
buckets identify candidates; they never establish necessary premises or
method equivalence. Unavailable retrieval differs from a successful search
with no hits. Local hashing is a deterministic test backend, not a semantic
retrieval claim.

Rows keep source identity, actual method and its source, a separately labelled
plan, decisive checks/failures, measured resources/masks/scope and execution
IDs. The primary case is the positive job's bound execution, even if the task
later ran another candidate. Absent performed-method evidence is unknown.
Original instance identity, frozen raw inputs and structured mathematical
content identify same-instance controls across renaming/rewording. One-sided
parent/source links are retained. Text-only concrete duplicate hints remain
pending confirmation; structure, method and family alone never imply a
duplicate. Independent isomorphic inputs can validate transfer. Ordinary
tasks need no invented mother-task field.

`OR_HARNESS_INDUCTION_MATERIAL_CHARS` defaults to 32000. The entire compact
JSON result is counted, including history, old knowledge, H+, diagnostics,
metadata and (for CLI) the `result`/`summary` envelope. `budget.chars_used`
is the actual serialized character count, not a token/byte estimate. Selection
precedes expansion. Oversized methods/text/rules are whole-value references,
never misleading prefixes. The pack protects a current case, cross-source
comparisons and relevant old rules before filling spare space. Counts and
`budget.next_cursor` report omissions; use `--cursor` only when more evidence
is needed. Changing purpose/query/count/selected executions creates another
session; an identical request reuses its pack. All sessions actually delivered
for this job, the triggering execution and submitted citations form its
derivation scope. Another round's reads do not enter it. Manual legacy work
may pass `--episode E` to select the initiating allow/skip/pending decision.

```bash
orx inspect --bank experience --execution ex_123
orx inspect --bank strategic --entry 3
orx inspect --bank strategic --entry 3 --rule-version candidate
orx inspect --bank values --value 'rule_fact:...'
# Optional legacy expanded diagnostic; not the bounded normal pack:
orx induction-material --task T --view detail
```

## Submit when / do / expect

```bash
orx induce --induction-id J --relation '{"when":"For fixed Y, feasible integer X is an interval and the objective is monotone in X.","do":"Enumerate Y, derive both X endpoints, select the endpoint from the coefficient sign and objective direction, and compare objectives. Disable this reduction for a nonmonotone objective and enumerate the full feasible set.","expect":"Preserve the original integer optimum and all stated constraints.","evidence":["ex_123"],"material_id":"induction_material:..."}'
```

Use real IDs from the read. In H+ mode, `material_id` and even `induction_id`
may be omitted when displayed/cited evidence uniquely identifies a positive
job, including when the first or only citation is a historical contrast.
Conflicting job/session references fail before writing. All three paragraphs are nonempty,
complete and executable; supporting detail belongs behind references. The
framework allocates the entry number and default subject, resolves facts and
roles, derives support scope and writes a candidate version. Necessary
premises belong in `when`; family/cell discovery does not auto-expand them.
Optional `predicates` narrow discovery but do not replace premise assessment.

| Operation | Input | Result |
|---|---|---|
| New rule | paragraphs + evidence IDs | New candidate, unless its canonical body already exists |
| Add evidence | `entry_id`, `operation:"add_evidence"`, evidence | Same body/version, independent evidence events |
| Revise | `entry_id`, `operation:"revise"`, complete paragraphs | New candidate version under the same number |
| Merge | `entry_id`, `operation:"merge"`, `merge_entry_ids`, `merge_reason` | Explicit source links; source rules stay available until the target and required source regressions pass |

Exact repeats are idempotent. Same-mechanism new cases should explicitly
select the original ID. Different names/families do not force different IDs;
similarity does not automatically merge meanings. `target_entry` and
`target_entry_id` are compatibility aliases for the explicit `entry_id`.
Changing the body requires revision or merge. Safe formatting equivalence
reuses the existing body/version. A substantive content change advances
the version. Ordinary recall exposes one active version. A pending revision
does not replace the old active version, and a suspended candidate cannot be
promoted through a maturity flag. To record no new rule, use `orx induce
--note "..."` without `--relation`.

## Check transfer, boundaries and regressions

Freezing a new rule is followed by a targeted trial or a historical holdout.
At execution, declare `--used-entry-ids 3@1`; an explicit candidate version is
permitted. The adoption snapshot freezes that exact immutable body. `''`
means declared none; an omitted option means undeclared. Matching names or
recall alone never prove adoption or faithful use.

Record a matching `method_performed` receipt with the execution's
`OR_ACTION_ID`, or a sourced inspection of captured artifacts. `check-task`
may attach a small optional `rule_assessments` list:

```json
{"entry_id":"3","version":1,"applicability":"applies",
 "method_fidelity":"faithful","effect_coverage":"complete",
 "source":"reviewer inspected the captured script and original requirements",
 "assessment":"The objective sign selects the lower endpoint; both constraints and the independently derived optimum were checked.",
 "evidence_refs":["artifacts.submitted_script","artifacts.workspace_result"]}
```

This is an attributable assessment with auditable sources, not automatic
natural-language proof checking. A plan, equal step text, solver `optimal`
or a checker `passed` alone supplies none of the missing premise/fidelity/
coverage facts. The original checker must actually cover the claimed effect;
a self-derived objective reference alone is insufficient.

- **Transfer:** a real independent source problem not displayed or cited
  during derivation. Same-source retries/seeds/branches and identical original
  content remain controls. A historical holdout needs an actual inspected
  application of the same mechanism and full effect coverage.
- **Boundary:** a real or small constructed test with changed necessary
  premises, `applicability:"not_applicable"`, and `response:"disable"` or
  `"fallback"`, with the response and result checked. Its scope is labelled
  `boundary_test`, not broader industrial evidence.
- **Regression:** representative cases group identical frozen inputs and
  acceptance rules, rather than requiring every old execution ID. A normal
  rerun adopting the new version with equivalent acceptance settles the old
  case, retaining old/new execution references and `actual_version_use` basis.
  Changed data, weaker acceptance or a v1 adoption leaves the v2 gap open.
  Revision and merge share these requirements. Frozen old uses retain their
  old version. Explicit historical reviews are separately marked
  `historical_review` and can
  be submitted as `validation:[{"kind":"regression","execution_id":"...",
  "assessment":{...}}]` on an `add_evidence` operation for the new version.
- **Counterexample:** only an attributable in-domain failure of the actual
  rule (`failure_source:"rule"`) can refute it. Code deviation, unknown
  adoption, missing coverage and unassigned failures stay insufficient.

Close-out and late `check-task` register checks safely; repeated close-out or
submission does not duplicate a physical use/check. Independent transfer,
boundary and required regressions must all pass before the candidate becomes
active. A credible active-version counterexample pauses it. New uncertainty
is recorded without inventing a refutation or deleting prior frozen proof.
Close-out freezes decisive rule-check facts before automatic retention can
evict executions, including an immediately expiring evidence window.
`check-task` also refreshes existing dependent checks. Omitting optional
assessments with unchanged facts and acceptance preserves the existing review;
an explicit `rule_assessments:[]` or matching `withdrawn:true` withdraws it.
Changed facts require renewed assessment. `exclude_execution` and
`restore_execution` update dependent publication and future H+ support, with
distinct events for each real pause/recovery. Prior checks and issued inputs
remain auditable. Window expiry does not mean withdrawal.
Normal `record` cost updates and `amend-cost` also recompute dependent cost
comparisons; changed measured values or provenance cannot leave stale savings
support published. Publication changes synchronize the strategic index.

Use `effect_scope:"technical"` only for an explicitly technical implementation
rule: its check reads executor/protocol facts (`wm-r-technical/1`), not a
business optimum. Use `effect_scope:"cost"` for a cost advantage. A cost
comparison needs the same frozen task, different observed actual methods,
comparable actual solving conditions, equivalent acceptance rules and trusted
measured dimensions on the same scope. The claimed dimensions come from the
assessment or rule. A unique recorded comparator is associated automatically;
multiple distinct baselines return choices, resolved by optional
`comparator_execution_id` in the assessment. `agent_estimate` never becomes
measured evidence through its mask. Existing-version use/check registration
works with H+=0 and after the original job completed. An isolated pass or
unknown cost never proves an advantage. G/C/R scoring is unchanged.

## Storage and historical data

V2 root rows hold pointers/state. Immutable `condition_values` hold bodies,
assessments and decisive facts. `rule_versions`, `rule_checks`, `rule_events`
associate checks, uses, effects and source aliases with exact versions.
Repeated claim text references the body; genuinely different historical
assertions keep their own references. Frozen decisive facts remain readable
after the execution evidence window evicts the original row.
Pending triggers and delivered contrasts retain a full immutable execution
reference, including original task, actual method, artifact/checker sources.
Material, citation resolution and `inspect --bank experience --execution EX`
read that reference after eviction/restart without reinserting the raw bank
row. Explicit validity changes still apply to retained evidence. The raw
window continues cleaning other episodes; this does not bound all referenced
knowledge storage.

New execution rows encode fixed boilerplate as short states and reuse the
method receipt body by reference. Audit reads render the original fixed
messages. Runtime values, source/conflicts, failure details, absent optional
predictions, measured zero, masks and scopes round-trip unchanged.

Unversioned legacy JSON keeps its original publication/verification semantics
and is labelled historical in short views. Reading it does not migrate it.
A deliberate revision preserves its original version before adding a v2
candidate. The old `induce --relation` claim/evidence/check grammar remains an
input adapter; newly submitted claims use v2 candidate publication. A declared
assertion audit (`verified`, `fact_checked`, `insufficient_evidence`, `refuted`)
is separate from transfer validation. Legacy lifecycle diagnostics still run;
new-version publication cannot be bypassed through a legacy lifecycle action.

Check that each declared assertion matches the claim it is meant to support.
`probe`, `status`, `comparison`, `code_changed` and `code_unchanged` remain
available. For a claim about changing a formulation, `code_changed` checks
different captured code hashes; `code_unchanged` checks the opposite. A failed
assertion is not automatically a refutation of the natural-language rule.
Placeholder claims such as "no new knowledge" and empty method steps are
refused without writing. Record that outcome as an empty review instead.

Maintenance actions record actual evidence IDs and committed deltas. Only
changed rule documents are embedded; evidence/use logs do not re-embed the
body. Failed index synchronization is deferred without rolling back facts;
retrying an affected submission/close-out retries those IDs. `rebuild-index`
remains an explicit repair tool. Production migration/cleanup is separate.
