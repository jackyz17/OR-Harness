# R20 · revision field inheritance

## Effect scope

`_body()` used `raw.get("effect_scope", "task_check")`, so any revision that
omitted the field was silently reset to `task_check` even when the prior
version was `technical`. `validate_case()` has a technical-observation branch,
so this changed the evidence basis used by later validation.

Resolution:

- explicit nonempty `effect_scope` wins;
- omission on a revision inherits the selected prior version's value;
- a new rule (no prior body) keeps the historical `task_check` default;
- an explicitly supplied empty/non-text value is refused, not silently
  defaulted.

## Predicates

`raw.get("predicates") or conditions...` treated an explicit `{}` as
"missing", so a revision could not clear retained machine conditions.

Resolution:

- omission (`predicates` absent, `conditions.predicates` absent, or
  `predicates: null`) inherits the prior version;
- explicit `predicates: {}` or `conditions.predicates: {}` clears it;
- explicit non-empty predicates replace it;
- a new rule with no declaration gets `{}`;
- non-object declarations are refused.

The compact interface does not expose predicates, so its revisions inherit
the selected version's machine conditions. The legacy advanced form remains
the way to clear or replace them explicitly.

## Tests

- `test_rule_versions.TestRuleLifecycle.test_technical_revision_inherits_scope_and_predicates_can_be_cleared`
  covers inherit / technical validation / explicit clear / explicit replace.
- `test_induction_submission.TestCompactSubmission.test_compact_revision_inherits_technical_effect_scope`
  covers the compact revision path.
