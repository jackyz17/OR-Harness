# Frozen H+ and evidence-backed learning receipts

H+ is the pre-execution binary investment switch in the SAME `predict-strategy`
answer. G/C/R still selects the solving strategy. Use `capability_gain.decision`
as the JSON number 0 or 1 (`wm-hplus-binary/1`). With 1, provide a short learning
hypothesis in `claim` and a concrete `verification_conditions` check; with 0,
state a reason in `basis`. Missing/illegal decisions schedule 0 with distinct
diagnostics. Old three-state predictions retain their original protocol.

The question is what reusable method, applicability boundary or verifiable
effect this execution and one review can add relative to the current library,
which fact can be checked, and which future step would change. Empty recall
alone does not establish novelty. A valid 1 with missing detail still requires
one honest review; actual evidence decides the learning receipt.

## The normal handoff

```bash
orx predict-strategy --task task.json --episode E --candidate candidate.json
orx execute --task task.json --prediction P --code solve.py --workspace ws
orx check-task EX --check '{"reference_objective":90,"reference_source":"bench_declared"}'
orx record --from-staged EX
orx close-episode --task T --episode E
orx induction-material --induction-id J
orx induce --induction-id J --relation '{"when":"...","do":"...","expect":"...","evidence":["EX"],"publication":{"state":"published","reason":"Agent checked the stated mechanism.","scope":["listed evidence only"]}}'
```

`J` comes from `induction_schedule.pending_ids`. The outer building agent must
settle each positive executed job with a short submission or a reasoned empty
review. Failure and insufficient material do not cancel 1. H+=0 stores the
execution and existing-entry use, skips new induction, and may still supply
another review's contrast or utility evidence. Unexecuted predictions remain
unobserved. `process_induction_jobs(reviewer, ...)` runs the same callback handoff;
the framework does not author a rule or make an extra hidden model call.

Give the solving agent the chosen frozen verification condition. Actual method,
artifacts and `check-task` provide the evidence. Optional `hplus_verification`
records its completion with the original prediction ID, condition index, source
and actual references. Absence stays unknown and does not cancel the review.

Every material call is a current query. Purpose/query/case IDs/cursor may change;
the triggering execution stays fixed while controls paginate independently.
Jobs keep trigger/ownership, necessary evidence/session references and final
submission/receipt, not cached full packs. Full pending evidence survives normal
window eviction/restart, and is released after completion while minimum audit
facts remain. Old packs/receipts stay readable. The runner passes the existing
job/material references automatically; ambiguous cases choose an existing object.
A shared historical evidence ID cannot dispatch an old identical submission to
a new job: an exact retry returns its original receipt. Explicit conflicts name
the differing payload fields and the receipt to inspect.

Repeated close reuses the actually bound prediction/action/episode identity.
Omitted episode auto-resolves only one real round; `--execution EX` selects an
existing run when needed. Two connections reread state inside transactions;
a late material write cannot reset a completed job. Before declaring building
complete, require `induction-status.completed:true`.

## Separate publication and learning

The induction agent chooses publication, candidate status, revision, merge or
only evidence addition. There is no transfer/boundary/regression prerequisite.
Derivation and small executable checks can support a scoped cold-start rule;
later independent tasks add transfer observations. See [induction.md](induction.md).

Learning receipts (`wm-hplus-review/2`) preserve original prediction/hypothesis,
actual induction action, conclusion/reason, valid fact references and exact rule
versions. The agent supplies `review.outcome` and `review.comparison`, or a
relation's `mechanism_comparison`. For example:

```json
{"outcome":"new_rule","hypothesis_outcome":"supported",
 "conclusion":"Endpoint direction and a disable boundary are supported in the stated scope.",
 "comparison":{"sufficient":true,"substantive":true,"basis":"derivation",
   "source":"inducer checked endpoint algebra and captured actual enumerations",
   "assessment":"At fixed Y, monotonicity and direction select the interval endpoint. A nonmonotone constructed case uses full enumeration.",
   "evidence_refs":["ex_A","ex_B","ex_constructed"]}}
```

- **1:** an actual new/revised body plus the agent's sourced, sufficient
  substantive-increment conclusion and valid support. Derivation or small
  runnable checks can confirm only their checked scope; a third real task and
  publication are not prerequisites.
- **0:** a sourced, supported `already_covered` or `no_increment` conclusion.
  Adding evidence may accompany this conclusion without changing the body.
- **Unknown:** insufficient support, missing actual review, unchecked required
  facts, or invalid/excluded sole support. An empty review can finish a job
  while leaving its learning observation unknown.

The default comparison basis uses existing checks; a late checker may make that
same recorded conclusion usable in a later observation. `basis:"derivation"`
names the agent's retained derivation over cited actual inputs;
`basis:"artifact_inspection"` uses actual captured artifacts. References must
exist. A context/library reference alone is not an executed result. Numerical
checks report their exact declared coverage; the framework does not certify
the agent's mathematical or causal judgment.

Hash changes, no recalled old rule, current publication and later retirement do
not infer increments. Safe exact/formatting dedup is retained. Synonym rewriting
without supported review is unknown. Truth of the original hypothesis is separate
from the useful method/boundary actually learned; a refuted hypothesis can still
support new knowledge.

An empty negative review can use existing job references:

```bash
orx induce --induction-id J --note 'Agent compared the actual checked workflow.' --review '{"outcome":"already_covered","conclusion":"No reusable increment beyond the existing mechanism.","comparison":{"sufficient":true,"substantive":false,"source":"agent compared frozen rules and actual checks","assessment":"The performed mechanism is already covered in the stated scope.","evidence_refs":["SOURCE_FACT_REF_FROM_JOB","BASELINE_SNAPSHOT_REF_FROM_JOB"]}}'
```

## Later facts, utility and prediction feedback

Later corrections affect only later observations. `exclude-execution` removes
support from current material/feedback; restoring reads valid sources again.
The entry's published status remains the agent's management decision, and its
previously supported learning result is not revoked merely because the agent
retires the method for utility reasons. Original receipts/check versions and
already-issued prediction inputs remain traceable.

Existing-entry utility management is independent of new induction. H+=0 can
record use/check/effects and `promote`, `demote`, `keep`, or `retire` with real
adoption references and a reason. Cost baselines and causal benefit judgments
belong to the agent; raw per-dimension observations are provided separately.
No empty induction receipt is needed for pure maintenance.

Within the existing feedback budget, select related settled positive/negative
examples before unknowns. Send a short hypothesis, conclusion, effective fact
references/check identifiers, reason and exact versions. Long comparisons stay
behind receipt/value references. If a compact row still does not fit, try the
next candidate rather than ending selection. Feedback uses only observations
recorded before the new prediction's cutoff.

## Historical/control paths

Historical receipts and three-state capability records remain inspectable;
legacy `evaluate-capability` later-performance attribution has a separate
meaning from this induction conclusion. Historical full material packs can be
read for audit, but new pending calls always query current evidence.

Existing H+/NoH+/random build-and-freeze experiment commands remain available
in separate homes. This change does not redesign their frozen-library lifecycle,
experiment splits/missed-opportunity statistics, or `evaluate-capability`
cross-execution attribution (the deferred P1-9 through P1-11). Old experiment
field names are not evidence of independent verification under this new label
protocol. Use real attributable samples to assess discrimination; substitute
providers validate the chain.

Runnable normal-flow demonstration:

```bash
PYTHONPATH=src python docs/validation/run_review_eight_demo.py
```

It publishes from A/B plus derivation and a constructed boundary before real C,
then records C use and performs lightweight `keep`, with an actual next provider
request fragment in its JSON output.
