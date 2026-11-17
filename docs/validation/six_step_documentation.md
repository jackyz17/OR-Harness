# Six-step solving documentation review

Review baseline: `main@3809a24`.
The directory has no usable Git metadata: `git status --short` and
`git rev-parse --short HEAD` both report a non-repository. Of the baseline's
tracked files, all 215 present locally matched their baseline blobs before
editing; 77 files were absent from this partial checkout. No newer local
implementation was substituted or inferred. Existing local-only documents are
preserved.

## Scope

- `SKILL.md` preserves the historical five-section organization and complete
  ASCII flowchart while covering all six workflow stages: shared task input and
  retrieval; candidates and U; execution/checking/recording; close-out and H+
  dispatch; comparison and review; actual version adoption, optional utility
  management and later feedback. The diagram continues to the next task after
  required jobs settle. Historical layout does not restore obsolete commands.
- Both README files and all edited agent documents are in English, as requested.
- `references/induction.md` is the single full new/revised submission example,
  including both `--relation` and `--review`, plus covered/no-increment and
  insufficient-material branches. SKILL and H+ link to it.
- Publication, required-job completion and learning observations are separate.
  The documents use the current author-led scoped publication behavior rather
  than an obligatory third-task/transfer/boundary/regression bundle.
- Actual implementation boundaries are explicit: `--memory-mode none` still
  runs vector retrieval; an effect verdict `refuting` without a lifecycle action
  can demote the adopted version; stored feedback may be absent from the actual
  provider request after budget trimming.
- In production Python, only `src/or_harness/cli.py` changed. Its AST is identical
  after string contents are removed: help and operational text changed, with no
  parsing, scoring, execution, checking, induction or storage logic change.
- Runnable documentation now uses the saved U-selected prediction directly and
  uses `text` in the shared task file. Historical pairing/automatic-admission
  audits are identified as historical, not normal authorization instructions.
  The existing A/B fixture is reused unchanged. The optional material-size audit
  body was restored unchanged; no pressure-test workflow extension is delivered.

Cold-start prompt policy, old rule-projection defaults and large frozen-evaluation
protocols are unchanged. The input-context reference records the existing runtime
wording and directs readers to actual publication/scope fields.

## Checks and observed results

| Check | Observed result |
|---|---|
| Minimal `task.json` copied from SKILL | `profile` and `recall` accepted the same file and task ID; absent CIR was explicit; unavailable embedding returned `degraded`, separately from empty recommendations |
| Normal single/multiple candidate CLI example | Passed through plan, original-P execution, check, record, close and H+ settlement; no manual selection or duplicate forecast |
| Complete new-rule command extracted from the Markdown | Explicit publication succeeded, job completed and learning count was positive=1; exact retry returned the same receipt |
| Covered command extracted from Markdown | Real source facts plus available prediction-time snapshot supported completion with negative=1 |
| Insufficient command extracted from Markdown | Empty review completed its job with unknown=1 |
| Next-task feedback | The next provider request included two settled related receipt rows; saved prediction exposed request omissions and context audit remained readable |
| Actual adopted version | Strategic inspect returned `entries[0].rule.view_version=1` |
| `--memory-mode none` | Structural rows=0 while vector retrieval returned 3 execution hits and 1 rule hit in the isolated fixture |
| Default `refuting` effect | The explicitly selected isolated use changed the adopted active version to paused; this is documented as a real side effect |
| Existing A/B fixture | A=2699, B=90, constructed nonmonotone case=0; rule published before later C=196; learning label=1 |
| Mechanical documentation check | All documented flags, commands, imports, links, 91 result keys and 15 example paths passed |
| CLI source boundary and language | Only string contents changed in the production CLI; all edited Markdown is English |
| Whitespace | No diagnostics from the snapshot-based `git diff --no-index --check` |

The extracted new-rule command bound job `ij_39356fb3ded60d875bb1d5c5` to
prediction `sp_b087fdbc1e2a`, solve action `ac_bf788b3f1120` and actual induction
action `ac_d7afa0e3d549`. Solve and induction actions are distinct.

Validation commands included:

```bash
PYTHONPATH=src python3 references/examples/_check_docs.py
PYTHONPATH=src python3 docs/validation/run_review_eight_demo.py
PYTHONPATH=src python3 references/examples/strategy_outcome.py
PYTHONPATH=src python3 references/examples/method_evidence.py
PYTHONPATH=src python3 references/examples/no_catalog.py
PYTHONPATH=src:tests/harness python3 -m unittest -q \
  test_end_to_end_cli.TestEndToEndCLI.test_normal_scored_cli_single_and_multiple_candidates
git diff --no-index --check /tmp/or-harness-six-step-docs-before /tmp/or-harness-six-step-docs-after
```

The three Markdown submissions were extracted and run through the real CLI in
one isolated temporary home with the existing fixed provider. The scratch audit
and its raw results are `/tmp/verify_six_step_doc_commands.py` and
`/tmp/six-step-docs-field-audit.json`; the retained A/B output is
`/tmp/six-step-existing-ab.json`. These fixtures establish wiring and declared
evidence scope, not model discrimination or empirical savings.

Earlier in this turn, before the instruction to limit testing, 198 related tests
passed in 45.226 seconds. No full `tests/harness` run was performed. Subsequent
checks were limited to documentation mechanics, small fixed-provider examples
and source/whitespace inspection. The native `git diff --check` could not operate
without repository metadata, so its no-index equivalent compared the before and
after snapshots.

## Changed files

- `SKILL.md`, `README.md`, `README_zh.md`
- `references/commands.md`, `references/modeling.md`
- `references/strategy_outcome.md`, `references/episode_closeout.md`
- `references/induction.md`, `references/hplus.md`
- `references/prediction_context.md`, `references/snapshot_lifecycle.md`
- `references/concepts.md`
- `src/or_harness/cli.py` (text only)
- `references/examples/_check_docs.py` (documented flag/example registration)
- `references/examples/unified_solving.py` (shared task text field)
- `references/examples/strategy_outcome.py` (direct scored selection)
- `references/examples/episode_closeout.py` (historical-audit description)
- `references/examples/method_evidence.py`, `references/examples/no_catalog.py`
  (publication and inspection explanations)
- `references/examples/versioned_induction.py` (historical-audit docstring only)
- This report

The complete induction command offered for review is in
[the canonical submission section](../../references/induction.md#new-or-substantively-revised-method).
