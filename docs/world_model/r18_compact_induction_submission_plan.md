# R18 · compact agent induction submission

## 1. Goal

The induction agent should answer only: what rule was found, when it applies,
what to do, what effect is expected, which actual evidence supports it, why the
conclusion follows, and whether to publish. The framework keeps the existing
evidence validation, H+ authorisation/receipt, versioned rule writer,
publication/verification separation, adoption and effect-attribution paths.

No new knowledge lifecycle and no new storage schema are introduced. The
compact entry is a normaliser in front of `induce(relations=..., review=...)`.

## 2. Minimal protocol

```json
{
  "result": "new_rule",
  "rule": {
    "when": "...",
    "do": "...",
    "expect": "..."
  },
  "basis": {
    "kind": "derivation",
    "refs": ["EX", "rule_fact:..."],
    "analysis": "..."
  },
  "publication": {"state": "published", "reason": "..."}
}
```

For a revision add `"entry_id"` (and optionally `"version"`). For
`evidence_added` add `"entry_id"`/`"version"` and omit `rule`. For
`already_covered` / `no_new_knowledge` only `result` and `basis` are needed.
For `insufficient_evidence` the basis may be omitted. Optional
`hypothesis_outcome` and `deviation_reason` remain separate from `result`.

`rule` accepts `effect_scope` and `subject`; `predicates` is deliberately not
part of the protocol because `when` is the one semantic applicability
statement and the existing writer stores an empty machine-predicate set.

## 3. Internal mapping

| Compact field | Existing field |
|---|---|
| `result` | `review.outcome` via one table |
| `rule.when/do/expect` | relation body |
| `rule.effect_scope/subject` | relation body / strategy subject |
| `entry_id`, `version`, `result=revision` | `operation:"revise"` target and version |
| `entry_id`, `version`, `result=evidence_added` | `operation:"add_evidence"` target and version |
| `basis.kind` | `review.comparison.basis` |
| `basis.refs` | relation `evidence` plus `review.comparison.evidence_refs` |
| `basis.analysis` | `review.conclusion` and `review.comparison.assessment` |
| `publication` | relation `publication` |

The result table is the only place the learning outcome is declared:

| `result` | `review.outcome` | `sufficient` | `substantive` | relation |
|---|---|---|---|---|
| `new_rule` | `new_rule` | true | true | create with `rule` |
| `revision` | `revised` | true | true | revise with `rule` |
| `evidence_added` | `evidence_added` | true | false | add_evidence |
| `already_covered` | `already_covered` | true | false | none |
| `no_new_knowledge` | `no_increment` | true | false | none |
| `insufficient_evidence` | `insufficient_material` | false | false | none |

These are translations of the agent's declared result, not framework
verification. The existing `supported_comparison` still checks that each
reference resolves, is a real executed fact, is time-valid, still supports the
conclusion and is of the declared basis type. An `insufficient_evidence`
submission can never become a label-1 observation through this mapping.

## 4. Evidence association

`basis.refs` is the only evidence list the agent writes. The normaliser keeps
every reference for the learning comparison and derives the execution ids the
versioned writer needs:

- a bare id / `execution:<id>` is an execution id;
- a `rule_fact:` (or other saved value with an `execution_id`) is resolved to
  its recorded execution;
- `artifacts.*` / `task_check.*` belongs to the induction job's trigger
  execution and requires `--induction-id`;
- a snapshot-only reference does not by itself authorize a written rule.

Task consistency, execution existence and support validity stay in the
existing `_reference` / `_resolve_claim_evidence` code. Source is derived from
the associated execution identities; timestamps and rule versions are written
by the existing receipt/version machinery. The seed-family metadata consumed
by the legacy claim reader is now derived alongside seed evidence, so the same
recorded facts render the same identity in compact and legacy reads.

## 5. Lifecycle and failure behaviour

The compact call uses `induce` unchanged after normalisation:

- H+ = 1: `submission_job` / `require_job` validates the job, material read,
  positive signal and authorisation before any write.
- H+ = 0: the skipped job cannot authorize a compact write.
- one SQLite transaction covers the rule version, evidence events, receipt and
  learning-review event; a failure leaves the job pending and reuses the
  reserved action id.
- the same compact payload has a stable fingerprint and returns the same
  receipt; a conflicting payload is refused.
- `dry_run` exercises the same validation without rules, receipts, events or
  index entries.
- publication is applied by the existing `update_publication`; verification is
  only what the existing writer actually checked.

## 6. Compatibility

`--relation`/`--review` remain accepted for merge, explicit assertion audits,
and existing-entry utility management. The compact entry and the legacy entry
cannot be mixed in one call. No table, column or serialized historical object
is changed; old rules and databases remain readable.

## 7. Field reduction

The same minimal new-rule case:

| Interface | Payloads | Leaf fields |
|---|---:|---:|
| legacy `--relation` + `--review` | 2 JSON arguments | 14 |
| compact `--submission` | 1 JSON argument | 9 |

The legacy count includes `when/do/expect`, `evidence`, publication state and
reason, `outcome`, `conclusion`, and comparison
`sufficient/substantive/basis/source/assessment/evidence_refs`. The compact
count is `result`, `when/do/expect`, `basis.kind/refs/analysis`, publication
state and reason. Evidence is named once instead of in both the relation and
the comparison; `source` is derived, and `conclusion`/`assessment` share the
one `analysis`. A revision adds `entry_id`/`version` in both interfaces.

## 8. Tests

`tests/harness/test_induction_submission.py` covers new rule success,
candidate default, published choice, revision, evidence-only addition,
`already_covered`, `no_new_knowledge`, `insufficient_evidence`, missing and bad references, artifact-to-execution
association, outer-runner integration, dry-run then commit, repeated
idempotency, mid-transaction failure recovery, H+ = 0 refusal, CLI submission,
mixed-interface rejection, schema rejection, compact use outside H+,
supplied-but-insufficient basis, and an isolated-fork comparison of compact
versus legacy payloads on the same job
(rule body, predicates, versions, publication, verification state, receipt
outcome/comparison, rule updates and rule-event history).
