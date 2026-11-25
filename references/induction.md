# Offline induction and existing-entry management

Inspect `implementation_evidence` alongside mathematical methods: compare failing versions, repair diffs and verified outcomes to identify transferable API, numeric/domain, parsing/index, diagnosis/repair or status/protocol mechanisms that change a future decision. Use the referenced artifacts to distinguish confirmed repairs from workarounds or unresolved causes. H+=0 retains execution evidence for later recall.

Compare recorded executions to extract a method, test its scope, and submit the review. Use [SKILL.md](../SKILL.md) for solving and close-out.

## Contents

- [When to start induction](#when-to-start-induction)
  - [What counts as a knowledge increment](#what-counts-as-a-knowledge-increment)
  - [Predict H+ at cold start](#predict-h-at-cold-start)
- [Take a required review](#take-a-required-review)
  - [Review decisions](#review-decisions)
  - [Keep the hypothesis outcome separate from the learning result](#keep-the-hypothesis-outcome-separate-from-the-learning-result)
  - [Durable knowledge must be usable by a future decision](#durable-knowledge-must-be-usable-by-a-future-decision)
  - [Match the verification claim to the rule's claim](#match-the-verification-claim-to-the-rules-claim)
- [Minimal submissions](#minimal-submissions)
- [Read material and choose comparisons](#read-material-and-choose-comparisons)
- [Complete review submissions](#complete-review-submissions)
- [Existing-entry use and utility](#existing-entry-use-and-utility)
- [Scoped checks, costs and corrections](#scoped-checks-costs-and-corrections)
- [Learning feedback and experiment modes](#learning-feedback-and-experiment-modes)

## When to start induction

`plan-next` freezes `capability_gain.decision` as the JSON number `0` or `1` (`wm-hplus-binary/1`). G, C and R determine U; H+ never enters it. The frozen switch only decides whether the completed execution must be REVIEWED afterwards.

### What counts as a knowledge increment

H+ is not a judgement that the task is difficult, that the candidate will succeed, that a trick can be summarised, or that the task uses new variables or constraints. It asks one question:

> In the current Harness knowledge state, could executing this candidate and its reasonably foreseeable checking, diagnosis and verification provide a concrete, reasonable and testable opportunity to produce, supplement, correct or independently validate transferable strategy knowledge that could improve a future agent's modelling, solving, implementation or verification decisions?

H+ anticipates the opportunity for [generalizing from cases](#generalize-the-method-from-cases); offline review carries out that generalization. Before execution, identify a consequential decision, align mathematical or operational roles when relevant history exists, and abstract entities, coefficients or a failure/repair difference into a possible mechanism. Ask what future decision it could improve, why it might work, and what minimal reasonable check this candidate could provide to support, refine or challenge it. This is a learning hypothesis, not a completed proof or cross-task validation; no prior executions are required.

A positive signal predicts an opportunity worth review, not a guarantee of publishable knowledge. Keep `claim` to a concise learning hypothesis, ground it in the frozen input and current knowledge through `basis`, and use `verification_conditions` for a minimum feasible check. It need not prove the mechanism, cross-task transfer, multi-task validation or eventual publication in advance. Formal knowledge validation and publication standards remain the offline review's responsibility.

Distinguish a method's familiarity in operations research, its coverage in the current strategy library, and its usefulness to a future agent. Textbook practice, a standard OR technique, a generally known algorithm or a conventional solver/verification method is background, not a reason by itself for H+=0. Specific applicability conditions, operating mechanisms, boundaries or implementation experience can still offer learning value.

Turning the task statement into decision variables, an objective and constraints is NECESSARY SOLVING WORK, not a knowledge increment by itself. These do not justify H+=1 on their own:

- writing a profit-maximisation objective because the task asks for maximum profit;
- writing a capacity constraint because the task gives a capacity;
- declaring variables integer because the task requires integers;
- adding a time index because the task has periods;
- building an ordinary LP/MILP model and calling a solver successfully;
- running a routine constraint or result check on the model just built.

A positive signal is possible for a transferable modelling mechanism or structural reduction, a classical method's structural applicability, a solving-strategy combination with a specific purpose, a reusable API or implementation mechanism, a transferable check or optimality-verification method, a rule's scope extension/correction/counterexample, or meaningful independent validation of an existing weakly supported mechanism. The test is whether evidence from this candidate's reasonable process could improve a future decision, not whether the method is new to the domain.

Checks and diagnostics need not be the main solver step to offer learning value. For example, with few facilities and a transportation LP remaining after fixing openings, affordable opening-subset enumeration plus transportation LPs can independently check a facility-location MILP optimum and offer a reusable verification strategy. This is eligible only when reasonably foreseeable for the current candidate. Do not borrow another candidate's H+, automatically combine unselected candidates' learning value, or use later execution facts to revise the frozen forecast.

For H+=1, `claim` states a concise potential reusable mechanism and the future decision it could change. `basis` explains task-structural grounds, current knowledge coverage and, when relevant history contributes, the shared mathematical or operational relationship. `verification_conditions` names the minimum feasible check that could support, refine or challenge that same mechanism in THIS run. A complete re-solve is not the default, and a later large experiment cannot substitute for evidence from this run; but an affordable small-scale full enumeration, an original-versus-reduced-method comparison, or a targeted reproduction can verify the mechanism. A 1 requests a review; it does not guarantee a rule or a publication.

For H+=0, `basis` gives the candidate-specific reason and `claim` is omitted. A covered 0 requires the actual visible published rule's **When** (conditions, mathematical structure and prerequisites), **Do** (core mathematical or operational mechanism) and **Expect** (effect, guarantee or verification purpose) to cover the hypothesis's substantive content, with no meaningful new evidence from this run. Cite the rule ID/version and compare these three aspects. Partial coverage may leave new conditions, reductions, guarantees, implementation details or failure boundaries to learn; missing coverage alone is still insufficient for H+=1.

Execution Evidence is induction material, not completed induction; Strategic Knowledge records the transferable mechanism actually abstracted and saved. Only currently visible **published strategy rules** can establish existing-knowledge coverage. Repeated successes, familiar solvers, similar objectives/terms, structural cells and embedding hits cannot substitute for one. When several executions support a concrete transferable mechanism but no corresponding rule exists, they support the opportunity for review instead of negating it. Do not justify H+=0 solely with "existing execution evidence already covers the method", "historical successful executions" or "existing solver capability". Keep these execution cases visible; if no learning mechanism exists, explain that lack of value rather than claiming instance records are strategy knowledge.

A bounded integer-enumeration rule does not automatically cover continuous-LP vertex optimality, an LP-relaxation integrality certificate or monotonicity-based variable reduction: their premises and mathematical arguments differ, even when they all involve optimisation or checking.

### Predict H+ at cold start

An empty library or missing historical reviews neither supplies a positive hypothesis nor prevents one grounded in the task, candidate and reasonably obtainable evidence. Use H+=0 if no concrete reasonable hypothesis can be identified; a method's being generally known is not itself that reason. Look for current-library gaps, corrections or meaningful independent evidence rather than demanding domain novelty.

Compare the proposed increment with published strategy rules that apply to the candidate in the frozen prediction input. Similar executions, matching structural cells and structural recommendations help find material and checks; they do not establish rule coverage by themselves. A structural recommendation requires confirmed or partial machine support; a rule whose predicates are empty or unmeasured is not offered there and remains discoverable through text retrieval. The **when** paragraph is the ONE semantic applicability statement: the framework checks only deterministic structural conditions (`predicates`, optional; empty means no machine-checkable condition, never unconditional applicability) and never decides whether `when` holds. A different `family` label alone is not a contradiction — a shared mathematical mechanism may transfer. When a recommendation contains a published rule, read its `when`/`do`/`expect` and decide applicability; a retrieval hit is not adoption. For an already-covered 0, cite its visible ID, version and applicable conditions in `basis`.

For example, take a small finite task whose full enumeration fits this run's budget. With no published applicable rule, the candidate enumerates Y and selects the lower endpoint of each finite integer X interval because the minimization objective increases with X. Its `capability_gain` can be:

```json
{
  "decision": 1,
  "claim": "For fixed Y and a minimization objective increasing in X, selecting the lower feasible integer endpoint can replace full X enumeration.",
  "basis": ["The original task has finite integer bounds; the candidate uses lower-endpoint enumeration and the strategy-rule view is empty."],
  "verification_conditions": [{
    "condition": "Check monotonicity and that the chosen feasible endpoints preserve the fixed-Y optimum; reuse an existing comparison if available.",
    "check_basis": "The original constraints and objective, the endpoint derivation and the captured implementation; an affordable finite comparison may supplement these."
  }]
}
```

For a covered control, suppose the frozen input actually shows published rule `12@3`, including its finite-interval premises, endpoint choice by objective direction and fallback for nonmonotone objectives. If the candidate repeats that method under those conditions, return 0 and omit `claim`:

```json
{
  "decision": 0,
  "basis": ["Published rule 12@3 already covers finite integer X intervals at fixed Y, lower-endpoint selection for this increasing minimization objective and full-enumeration fallback when monotonicity fails; this run adds no scope finding or meaningful independent evidence."]
}
```

Use only rule IDs and versions actually present in the input; the control ID above is illustrative. If no concrete reusable increment with a check feasible during this run can be identified, return 0 with that specific reason and omit `claim`. Missing review history alone is not that reason. A 1 schedules a factual review; the review decides whether anything was learned and whether a rule can be published.

For a bare modelling translation, suppose the task states a capacity limit and the candidate simply writes a linear capacity constraint. No specific transferable mechanism, boundary or operational finding beyond that translation has been identified. Return 0:

```json
{
  "decision": 0,
  "basis": ["The candidate translates a stated capacity limit into a linear constraint. That is required solving work, not a reusable increment, and no concrete mechanism or method boundary that this run could establish was identified."]
}
```

When a condition is checked, add an item to the `hplus_verification` list in the `check-task --check` JSON. Supply the original `prediction_id`, the zero-based `condition_index`, `state:"checked"`, a `source`, and locatable `evidence_refs` such as `artifacts.submitted_script` and `task_check.checks`. Use `finding` to record what the check showed. An omitted state or missing support leaves the condition unknown. Use `not_checked` or `unavailable` honestly when the run could not complete the condition; never mark it checked to close the review, and never run a large complete re-solve merely to make it show as checked — an affordable small-scale check that verifies the mechanism is allowed.

During H+ building, close-out schedules one required review per bound executed positive prediction, including failed attempts and incomplete verification. Missing or invalid decisions schedule as 0 with their own diagnostics. Executed H+ = 0 attempts remain available as comparisons.

## Take a required review

Read each J from `close-episode` under `result.induction_schedule.pending_ids`. Use `induction-status` to resume interrupted work. Settle the returned jobs before continuing the building stream.

```bash
orx induction-status --task T --episode EP
orx induction-material --induction-id J
```

H+=1 schedules the review; it does not prove that knowledge was gained. The review must be allowed to conclude that nothing was learned.

### Review decisions

Extract a mechanism that changes a future decision, rather than repackaging the execution history as a generic workflow. Compare, in this order:

1. the frozen `claim` and each `verification_condition`;
2. what the actual execution and checks established — the captured script, result, check report, attached verification and the material cases;
3. the relevant existing rules and historical executions delivered in the material (including the frozen prediction-time baseline);
4. whether the verification condition was actually completed.

Then choose the learning result from the evidence, not from the trigger:

- `new_rule`: a genuinely new reusable method or boundary with enough evidence.
- `revision`: an existing rule's scope or body is wrong or incomplete and the corrected version is supported.
- `evidence_added`: an existing unchanged rule gains real support; no new body.
- `already_covered`: the actual published When/Do/Expect fully cover the proposed mechanism, with no supported increment or meaningful new evidence; cite the rule version and applicable conditions.
- `no_new_knowledge`: the execution was a normal modelling/solving run without a reusable increment, or the hypothesis dissolved on inspection.
- `insufficient_evidence`: the material cannot support either a rule or a supported no-increment conclusion.

`new_rule` and `revision` are never required to settle a positive job. Writing no rule is a successful review when the facts support that conclusion. Do not split one method into several renamed fragments to grow the count, and do not merge methods whose mechanisms differ just to keep the count low.

### Keep the hypothesis outcome separate from the learning result

The review must also say what happened to the ORIGINAL prediction-time hypothesis, using the existing `hypothesis_outcome` (`supported`, `refuted`, `unknown`) and `deviation_reason`. Four cases are different:

1. the claim was supported;
2. the claim was refuted;
3. the claim was not sufficiently verified;
4. the execution revealed different new knowledge unrelated to the claim.

Do not present case 4 as case 1. If the original hypothesis was not verified but an incidental, well-evidenced technique is worth inducing, analyse that finding separately; the original claim's outcome is still recorded as `refuted` or `unknown`. An unfinished verification condition stays `not_checked`, `unavailable` or unknown in the receipt; it is never upgraded to checked to close the job. Use the existing `hypothesis_outcome`, `deviation_reason`, verification records and `basis`; do not invent a second hypothesis-status system.

### Durable knowledge must be usable by a future decision

A published strategy rule is used BEFORE solving a future task. It must be decidable and executable at that decision point:

- **when**: applicability conditions or a method boundary a future agent can judge from real task inputs and the candidate;
- **do**: concrete modelling or implementation steps that can be executed;
- **expect**: the guarantee or measured effect the evidence supports.

Prefer the supported mechanism that removes work or prevents mistakes: a search-space reduction, feasible-set/optimum-preserving variable reduction, dominance, monotonicity, valid bound, analytic certificate, applicability/failure boundary, or reusable implementation repair. Keep its premises, core operation, future decision and evidence explicit. If the frozen claim concerned a reduction, assess that reduction first; do not silently replace it with "enumerate everything and independently enumerate again". Another finding may be published only with its own support and an honest original-hypothesis outcome in the existing receipt.

Strategic knowledge may store reusable modelling methods, solving strategies with applicability conditions, structural reductions and speed-ups, transferable code/API experience, and evidenced method-failure boundaries. It must NOT store benchmark scores, a specific test set's reference answers, a particular task's difference from a hidden reference, strategies that only work once the evaluation result is known, patches made only to pass a particular test, or rules of the form "if the benchmark reference uses domain X, use domain X". Reference answers, checkers and experiment results may be cited as POST-HOC VERIFICATION EVIDENCE, but they cannot be a premise of the strategy itself unless that condition is a real task input the future agent can see.

Do not add knowledge types such as `benchmark_specific` or `modelling_translation`, and do not create a separate classification or storage for them.

### Match the verification claim to the rule's claim

Keep the existing verified / insufficient / refuted semantics and publication flow. Distinguish:

- an exactly derived mathematical property: a valid derivation scopes it; one solid derivation can support the rule without independent task repeats;
- an empirical success, cost or transfer claim: it needs independent tasks with effective comparison before it is described as a verified general regularity;
- a technical fact observed in a single run: keep it as a scoped technical fact;
- an unverified generalisation: keep it as a hypothesis, not a verified rule.

A single observation may be retained as evidence but must not be advertised as a proven cross-task advantage. Finite integer enumeration establishes a finite search, not a measured speed advantage over MILP. Independent optima may have different solution points: check feasibility and the same optimal objective, not equality of the entire solution vector. Continuous-LP vertex arguments and integer enumeration require different premises and proofs. When a cheap certificate already suffices, do not demand a second complete solve; use further checks only for an unresolved claim. Keep the existing multi-task evidence, verification and publication standards.

## Minimal submissions

Compare cases, form a supported conclusion, then submit it with the same J.
J comes from `close-episode` under `result.induction_schedule.pending_ids`.
One JSON object carries the whole review: the learning result, the rule, one
evidence-and-argument basis, and the publication intention.

```bash
orx induce --induction-id J \
  --submission '{
    "result":"new_rule",
    "rule":{
      "when":"Y is finitely enumerable; at fixed Y, feasible integer X forms a finite interval and the objective is monotone in X.",
      "do":"Skip empty intervals. Choose the lower or upper X endpoint according to objective direction and coefficient sign, then compare all Y. Enumerate X fully when the objective is not monotone.",
      "expect":"Preserve the optimum under these conditions while reducing X enumeration."
    },
    "basis":{
      "kind":"derivation",
      "refs":["EX","SOURCE_FACT_REF"],
      "analysis":"For each feasible fixed Y, monotonicity puts the best X at the selected endpoint; comparing all Y preserves the optimum."
    },
    "publication":{
      "state":"published",
      "reason":"The recorded enumeration and the endpoint derivation support this monotone finite-interval scope."
    }
  }'
```

`rule` is still **when / do / expect**. `when` is the one semantic
applicability statement; do not resubmit `predicates` or duplicate the rule
under another field. The framework stores an empty machine-predicate set, the
same short-rule body and version metadata as the existing writer.

`basis` is the single place actual evidence is named. `kind` is the evidence
type: `derivation` (an argument over real inputs/outcomes), `recorded_checks`
(a sourced checker result) or `artifact_inspection` (a captured artifact).
`refs` must name real references from the material or this memory. An
execution id, a fact such as `SOURCE_FACT_REF`, and a captured
`artifacts.*`/`task_check.*` reference all resolve to the recorded execution
identity; the existing H+ validation still checks that the reference exists,
belongs to a real executed fact, is time-valid and still supports the
conclusion. `analysis` is the short substantive reason, not a second summary
of the rule.

`result` is the only learning-result field:

| `result` | What is written | Receipt outcome | Learning label |
|---|---|---|---|
| `new_rule` | a new candidate version from `rule` | `new_rule` | 1 when the basis is supported and the version is a real increment |
| `revision` | a candidate version of `entry_id` (`version` optional) from `rule` | `revised` | 1 when the basis is supported and the new version is a real increment |
| `evidence_added` | evidence on an unchanged `entry_id` version; no `rule` | `evidence_added` | 0 when the basis is supported and no substantive increment is claimed |
| `already_covered` | no rule; a supported conclusion over the frozen rule/scope | `already_covered` | 0 when the basis is supported |
| `no_new_knowledge` | no rule; a supported no-increment conclusion | `no_increment` | 0 when the basis is supported |
| `insufficient_evidence` | no rule; the review is recorded with unknown learning | `insufficient_material` | unknown |

`new_rule` and `revision` are the only results that write a rule.
`evidence_added`, `already_covered` and `no_new_knowledge` require `basis`;
`insufficient_evidence` may omit it. Do not write a rule to say there is no
new knowledge: use the matching no-write result. The table lists what each
result writes; it is not a preference order. For a positive trigger, choose the
result the evidence supports, including a no-write result.

`publication` is the agent's scoped usage choice, not a verification claim.
Omit it to leave the new body as a candidate. `state` is
`published` or `candidate`; `reason` is required. Publishing does not make the
rule verified, and verification is whatever the existing writer actually
checked; a candidate version can still be adopted and later revised.

To revise an existing rule, add the real `entry_id` (and optionally the
`version` being revised):

```bash
orx induce --induction-id J \
  --submission '{
    "result":"revision",
    "entry_id":"ENTRY",
    "version":1,
    "rule":{
      "when":"The objective is monotone in X at fixed Y and the feasible X set is a finite interval.",
      "do":"Choose the endpoint from the objective direction and coefficient sign; disable the reduction when monotonicity fails.",
      "expect":"Preserve the optimum within this sign-aware scope."
    },
    "basis":{"kind":"derivation","refs":["EX","SOURCE_FACT_REF"],
             "analysis":"The earlier body only covered maximization; the checked minimization case selects the opposite endpoint."},
    "publication":{"state":"candidate","reason":"Keep the active version until independent use checks the revision."}
  }'
```

When a revision omits `effect_scope` or machine `predicates`, both inherit
the selected prior version; an explicit `predicates:{}` in the legacy
advanced form clears the old machine conditions, and an explicit
`effect_scope` replaces it. A new rule with no prior version still gets the
`task_check` default. The compact interface does not expose predicates, so its
revisions inherit them from the version being revised.

When the existing rule already covers the executed method, cite the frozen
rule/scope and the actual fact, and write no rule:

```bash
orx induce --induction-id J \
  --submission '{
    "result":"already_covered",
    "basis":{"kind":"recorded_checks",
             "refs":["SOURCE_FACT_REF","BASELINE_SNAPSHOT_REF"],
             "analysis":"The executed steps and assumptions match the rule version present in the prediction-time snapshot."}
  }'
```

Use `no_new_knowledge` for a supported review that finds no reusable
addition. Use `insufficient_evidence` when the material cannot support either
a rule or a no-increment conclusion:

```bash
orx induce --induction-id J \
  --submission '{"result":"insufficient_evidence",
    "deviation_reason":"The objective is available, but the record lacks resource-level flows needed to check the activation constraints."}'
```

An optional `hypothesis_outcome` (`supported`, `refuted` or `unknown`) and
`deviation_reason` can be added when the H+ hypothesis itself was tested; they
are separate from `result`, which records what knowledge was written.

Optionally add `--dry-run` to the same submission command when unsure about
its input. It checks the result/rule/entry selection, evidence references and
publication choice without writing rules, receipts, events or index entries.
Fix a reported field or use a real reference from the material, then remove
`--dry-run` to submit with the same J. A passing preview leaves J pending.
`would_create`, `would_update` and `would_complete` describe expected work;
`saved` and `published` count actual writes and remain zero in a preview.
Insufficient learning support is reported separately from invalid input and
does not prevent a factual review from completing. Repeating the same compact
submission returns the same receipt; a conflicting payload is refused. After
submission, read `result.induction_receipt` and use `induction-status` to
confirm completion.

### Legacy advanced interface

`--relation` and `--review` remain accepted for compatibility and advanced
operations that the compact interface deliberately does not expose, such as
merge and explicit assertion audits. Do not combine them with `--submission`
in one call.

```bash
orx induce --induction-id J \
  --relation '{"when":"...","do":"...","expect":"...","evidence":["EX"],
               "publication":{"state":"candidate","reason":"..."}}' \
  --review '{"outcome":"new_rule",
             "comparison":{"sufficient":true,"substantive":true,
                           "basis":"derivation","source":"...",
                           "assessment":"...","evidence_refs":["EX"]}}'
```

The legacy spelling uses `review.outcome` instead of `result`, separate
`relation.evidence` and `comparison.evidence_refs`, and explicit
`sufficient`/`substantive` booleans. Those are exactly the fields the compact
interface translates; use one interface consistently. For merge use
`operation:"merge"` with `merge_entry_ids` and `merge_reason`; for assertion
audits use `--verify` or an embedded `check`. Existing-entry use and utility
judgements still use `--attribute-effect` (no compact equivalent).

## Read material and choose comparisons

Read the default pack, then follow the question raised by the evidence. `--purpose` sets the search focus for that read. You can change the focus, request known execution IDs, or page through attempts while keeping the same trigger and J.

```bash
orx induction-material --induction-id J --purpose modelling --query "MECHANISM_KEYWORDS"
orx induction-material --induction-id J --purpose implementation --query "ERROR_OR_API_KEYWORDS"
orx induction-material --induction-id J --executions EX1,EX2
orx induction-material --induction-id J --cursor CURSOR_FROM_BUDGET
```

| Field | How to use it |
|---|---|
| `material[]` | The trigger first, then current-task controls. Read `method`, `method_source`, outcomes, checks, and measured costs. |
| `related_history.executions[]` | Candidate comparisons from history, including failures. Compare the original constraints, actual methods, and check coverage. |
| `existing_knowledge` | Short views of relevant rules in the current library during ordinary H+ building. Use them to find possible revisions or duplicates. |
| `comparison_table` | An index of `execution_id`, `relationship`, and `check`. Read the case itself for its method. |
| `induction_gate` | The frozen hypothesis, verification conditions, `source_fact_ref`, and prediction-time `baseline_snapshot_ref`. |
| `budget.next_cursor` | The next page of current-task controls. |

Use `--purpose modelling` for model structure or `--purpose implementation` for code and repairs. Compare retries to diagnose a failure and independent cases to test how the method travels. Pending reviews retain delivered execution facts. The default response budget is 32,000 characters (`OR_HARNESS_INDUCTION_MATERIAL_CHARS`). Follow `inspect_hint` to open a compact or reference-only case.

```bash
orx inspect --bank experience --execution EX1
orx inspect --bank strategic --entry ENTRY --rule-version candidate
orx inspect --bank values --value VALUE_REF
```

After answering, cite available reference answers, checker results, or user feedback with their source and coverage. Use original-constraint checks or a derivation to support claims about feasibility and why a step transfers.

## Complete review submissions

A positive trigger does not require a new rule. The review first decides whether
there is a reusable increment at all; `already_covered`, `no_new_knowledge` and
`insufficient_evidence` are legitimate conclusions of a complete review.

### Generalize the method from cases

Start from the frozen H+ hypothesis and what the execution showed. Identify a specific decision in the model or implementation and the effect it might have on feasibility, objective value, runtime, or reliability. Choose the next comparison by what would clarify that claim.

Let the evidence determine what to record. A review can yield a modeling or solving method, an implementation technique, or a rule that combines steps that work together. Submit separate rules for independently useful findings with their own evidence. When the library already covers the method or evidence is insufficient, complete the review with that conclusion.

- **Find what changed the result.** A successful run gives a method worth inspecting. A failure and its repair on the same input can identify the step responsible. Compare what the model expressed, what the code actually did, and what the checks observed.
- **Align the roles across problems.** Look for another task that poses the same decision under different domain details. Map the decision variables, the constraints that connect them, and the quantity or guarantee they affect. Keep the relationship that the cases share and express case-specific capacities or coefficients as parameters.
- **Explain why the step works.** Derive the effect from the original constraints and objective. Keep the assumptions the argument uses and phrase them as facts the next agent can check. A derivation can support a scoped rule when only one case is available.
- **Try to break the claim.** Use a similar failed execution, an independent task, or a small constructed case that changes a condition relevant to the explanation. For an exact reformulation, check which feasible decisions and objective values it preserves. For an empirical speed or quality claim, compare measured outcomes under comparable conditions. If the claim fails, narrow its scope or add a fallback.
- **Record the reusable decision.** Put the applicability conditions and any relevant failure boundary in **when**. Put the method and the checks needed to apply it in **do**. State the guarantee or measured effect the evidence supports in **expect**. Check the relevant existing rule before choosing to add, revise, or supplement it. If the evidence leaves a decisive question open, record what would resolve it.

### Induce implementation methods

Use `--purpose implementation` when the reusable change concerns code or a solver interface, such as data parsing, index mapping, API calls, or result delivery. Compare the submitted scripts, error or protocol receipts, and actual results. A failed attempt and a repair on the same problem, model, and settings can isolate the step that mattered. Look for the same failure mechanism in another task or implementation. Generalize the reusable code step, the interface assumptions it depends on, and the check that detects failure.

Match validation to the claim. Use `execution_features.technical_outcome` and captured artifacts to check technical delivery. For a claim about the computed answer, check the original task. For a cost claim, compare measured values under matching conditions.

#### Implementation example · Deliver solver results

Use the computed objectives from the container and transshipment examples as inputs to a controlled script test. Change the delivery code between attempts.

| Case | Failure | Repair and observed result |
|---|---|---|
| Container production | The script prints 43,200 and exits without writing `result.json`. The executor records a missing-result protocol failure. | Write `{"status":"optimal","objective_value":43200}` in `result.json` in the execution workspace. The executor reads a valid result. |
| Transshipment | The script writes `{"status":"optimal","objective":685}`. The executor reports that the objective field is missing. | Use `objective_value` for the numeric objective. The executor reads a valid result with value 685. |

The shared result writer can be a short rule.

- **when** The solve script hands results to the executor through the workspace file.
- **do** Map the solver's actual status and available solution in one place. Write a fresh `result.json`. For `optimal` or `feasible`, include a finite numeric `objective_value`. When the solver establishes infeasibility, deliver `{"status":"infeasible"}` and any available diagnostics.
- **expect** The executor can read both the solution and infeasibility branches under the result protocol.

For this claim, use `"effect_scope":"technical"` inside the compact `rule`. For the learning basis, use `"kind":"artifact_inspection"` with the actual before/after executions and their captured artifacts.

### Modeling example · Fixed setup costs in SIRL IndustryOR

These teaching problems are IndustryOR cases 12, 44, and 77 from SIRL. They illustrate one route from recorded cases to a transferable rule. Read the problem statements, then compare the checked outcomes and the constraints that explain them.

#### Example 1 · Container production with substitution

A factory must meet its orders for six container sizes. Each size has its own production equipment. Producing any positive quantity of a size triggers a one-time setup fee of 1,200 yuan for that equipment.

| Size ID | Volume (cm³) | Orders (units) | Variable cost (yuan/unit) |
|---|---:|---:|---:|
| 1 | 1,500 | 500 | 5 |
| 2 | 2,500 | 550 | 8 |
| 3 | 4,000 | 700 | 10 |
| 4 | 6,000 | 900 | 12 |
| 5 | 9,000 | 400 | 16 |
| 6 | 12,000 | 300 | 18 |

One container can fill one order of the same or a smaller required volume. Choose the production quantities and their allocation to orders. Meet all orders and minimize variable production costs plus setup fees.

#### Example 2 · Transshipment through stations with fixed fees

Send material from two origins to two destinations. Each shipment passes through exactly one station. Destination 1 requires 80 units and destination 2 requires 120. The tables give supply limits, station capacity, and the unit shipping costs on both legs.

| Origin | Available supply | To station 1 | To station 2 |
|---|---:|---:|---:|
| 1 | 100 | 2 | 3 |
| 2 | 150 | 4 | 1 |

| Station | Throughput capacity | One-time fee if used | To destination 1 | To destination 2 |
|---|---:|---:|---:|---:|
| 1 | 100 | 10 | 3 | 2 |
| 2 | 100 | 15 | 1 | 4 |

Choose route flows that meet demand, conserve flow at each station, and respect supply and capacity. Minimize the costs of both shipping legs plus the fees of used stations.

#### Example 3 · Part assignment with mandatory machines

Process one part of each type from 1 to 10. Assign each part to one machine. Using a machine incurs its setup fee once. The per-part processing costs and setup fees are in yuan.

| Machine | Setup fee | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 100 | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 |
| B | 135 | 15 | 25 | 35 | 45 | 55 | 65 | 75 | 85 | 95 | 105 |
| C | 200 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 | 110 |

Exactly one of parts 1 and 2 must use A. The other uses B or C. Parts 3, 4 and 5 are assigned to A, B and C respectively. Machine C can handle at most three parts. Formulate an integer model that minimizes total processing and setup costs.

#### Compare the cases and test the boundary

| Evidence | Inference |
|---|---|
| **Containers.** Selecting a plan by variable cost alone, then scoring it with all fees, costs 43,700 yuan. Joint optimization costs 43,200 yuan by producing size 3 for size 2 orders and size 6 for size 5 orders. | The choice is whether to pay a setup fee to avoid more expensive substitution. Link the setup flag to all production of that size. Bound production by the total demand the size can serve. |
| **Transshipment.** The optimum costs 685. Each station carries 100 units. In every feasible plan, demand of 200 fills the two capacities of 100. | The quantity controlled by the flag is total station throughput. Use physical capacity for its bound and keep inbound equal to outbound. The network constraints already force both stations open. |
| **Part assignment.** The optimum costs 1,005 yuan, of which 435 is setup. The three mandatory assignments force all machines to be used in every feasible plan. | A different constraint proves the same simplification. Fix the three flags to one and carry the fees as constants. Keep the part assignment and machine limits. |

The common model uses total nonnegative use `q_j`, a valid upper bound `U_j`, a binary flag `y_j`, and a one-time positive fee `F_j`. The link `q_j <= U_j*y_j` makes any positive use require activation, and `F_j*y_j` charges once. With that fee and no other need to activate an unused resource, cost minimization permits zero activation at zero use. Fixing a flag to one is justified when the original constraints force positive use in every feasible plan.

Probe that last premise by raising station 2's capacity from 100 to 200 in the transshipment example. Route 50 units from origin 1 and 150 from origin 2 through station 2, then send 80 and 120 to the destinations. This satisfies the modified constraints with station 1 closed, so the original proof for fixing both flags no longer applies. The bounded-use linkage still applies.

For the submission below, `EX_CONTAINERS`, `EX_TRANSFER`, and `EX_PARTS` stand for real `result.execution_id` values from running and checking those problems in the same memory home. J is the review ID from the positive trigger. Choose the publication scope from the comparisons and derivation you can support.

### New or substantively revised method

This submission uses the fixed-setup-cost modeling example.

```bash
orx induce --induction-id J \
  --submission '{
    "result":"new_rule",
    "rule":{
      "when":"In a cost minimization model, a resource incurs a positive one-time fee when used. Its total use is bounded and nonnegative, and the setup flag governs that use and fee.",
      "do":"Use a binary activation variable y_j for each optional resource j. Charge F_j*y_j once and constrain its total use q_j by q_j <= U_j*y_j. Set U_j from capacity or reachable demand. Check total use, original flow balances, and assignment constraints. Fix y_j to one and keep F_j as a constant when the original constraints force positive use in every feasible plan.",
      "expect":"Choose resource use jointly with variable cost and account for each setup fee once. Fixing a mandatory activation preserves feasibility and total cost."
    },
    "basis":{
      "kind":"derivation",
      "refs":["EX_CONTAINERS","EX_TRANSFER","EX_PARTS"],
      "analysis":"The container comparison isolates the effect of joint setup and variable costs. Transshipment maps total use to throughput. Capacity in transshipment and designated parts in assignment independently show how original constraints can force activation."
    },
    "publication":{
      "state":"published",
      "reason":"Substitution and station flow support the use-to-activation link. Capacity bounds and mandatory assignments show how to prove a flag can be fixed.",
      "scope":["Positive fixed charges on bounded resource use; forced activation only when implied by the original constraints"]
    }
  }'
```

`--submission` writes **when / do / expect**, the evidence identities, the
learning comparison and the publication choice. Omit `publication` to leave a
new body as a candidate. To revise, use `"result":"revision"` with the real
`entry_id` (and optional `version`); the active version stays available while
the revision is a candidate. `merge` is a legacy advanced operation: use
`--relation` with `operation:"merge"`, `merge_entry_ids` and `merge_reason`.
Source rules stay available until the target version is published. Identical
bodies reuse their version.

### Already covered or no reusable increment

Inspect `source_fact_ref` and `baseline_snapshot_ref` from
`result.induction_gate`. For `already_covered`, identify the rule version in
the prediction-time snapshot that covers the executed method and cite both
references. If the available evidence leaves prior coverage unresolved,
submit `insufficient_evidence`.

```bash
orx induce --induction-id J \
  --submission '{
    "result":"already_covered",
    "basis":{
      "kind":"recorded_checks",
      "refs":["SOURCE_FACT_REF","BASELINE_SNAPSHOT_REF"],
      "analysis":"The executed steps and assumptions match the rule version in the frozen snapshot."
    }
  }'
```

Use `no_new_knowledge` for a supported review that finds no reusable
addition. To attach evidence to an unchanged rule in the same J, use
`"result":"evidence_added","entry_id":"ENTRY","version":VERSION` with the new
execution in `basis.refs`.

### Insufficient material

```bash
orx induce --induction-id J \
  --submission '{"result":"insufficient_evidence",
    "deviation_reason":"The objective is available, but the execution record lacks resource-level flows needed to check the activation constraints."}'
```

This completes the review with an unknown learning label. Use it when
evidence is insufficient for either a rule or a no-increment conclusion.

### Worked review flows

The three outcomes below use the same review protocol and differ only in what
the evidence supports. EX and J are real IDs; SOURCE_FACT_REF and
BASELINE_SNAPSHOT_REF come from `result.induction_gate`.

**new_rule — a specific mechanism survived review.** The frozen claim proposed
that capacity shared between a training phase and a production phase is easy to
deduct in only one phase. The run's base check, the attached script and the
comparison with the failed one-phase attempt isolate the step. Read the
material, then submit:

```bash
orx induce --induction-id J \
  --submission '{
    "result":"new_rule",
    "rule":{
      "when":"A model has a training phase and a production phase that share one resource capacity, and the task states the resource is unavailable in both phases.",
      "do":"Deduct that resource capacity from every phase in which it is unavailable, not only the production phase. Recompute resource usage in both phases and keep the same phase scope in the check.",
      "expect":"Preserves the stated availability constraint across the whole horizon."
    },
    "basis":{"kind":"derivation","refs":["EX","SOURCE_FACT_REF"],
             "analysis":"The captured one-phase and two-phase runs differ only in the capacity deduction; the two-phase version preserves the stated availability constraint and the objective."},
    "hypothesis_outcome":"supported",
    "publication":{"state":"published","reason":"The derivation and the recorded comparison support this scope."}
  }'
```

Read `result.induction_receipt`: `outcome=new_rule`, the written version and
`verification[].state` for each condition. A condition that was not actually
checked stays `unknown` even when the rule is accepted.

**no_new_knowledge — the hypothesis dissolved.** The frozen claim suggested that
a standard capacity constraint might hide a modelling trap. The execution shows
the statement gave the limit directly, the code wrote `sum(q) <= cap`, and the
recorded check matched the task. There is no reusable increment. Submit:

```bash
orx induce --induction-id J \
  --submission '{
    "result":"no_new_knowledge",
    "hypothesis_outcome":"refuted",
    "deviation_reason":"The capacity limit was stated directly; no boundary or mechanism was tested.",
    "basis":{"kind":"recorded_checks","refs":["EX","task_check.checks"],
             "analysis":"The recorded check only confirms the stated capacity limit; it provides no specific reusable mechanism, scope correction or independent verification finding."}
  }'
```

Receipt: `outcome=no_increment`, learning label 0 when the no-increment
conclusion is supported, and no rule. `hypothesis_outcome` still records
`refuted`; the hypothesis outcome and the learning result are separate fields.

**insufficient_evidence — the condition was not completed.** The frozen claim
asked whether a two-solver comparison would expose a numerical boundary, but
the second solver was unavailable in the sandbox and the run kept only one
solver's result. The self-check committed what it could; the H+ condition is
not checked. Submit:

```bash
orx induce --induction-id J \
  --submission '{"result":"insufficient_evidence",
    "hypothesis_outcome":"unknown",
    "deviation_reason":"The second solver was unavailable; no independent boundary check exists in this execution."}'
```

Receipt: `outcome=insufficient_material`, learning label unknown, no rule, and
the verification item stays `unknown`. This is a COMPLETED review, not a failed
one; it must not be reported as a verified hypothesis. If a run produces both a
refuted original claim and an unrelated supported finding, record the refuted
outcome and submit the supported finding as its own `new_rule` with its own
basis.

### Read publication, completion and learning separately

| Result | Field |
|---|---|
| Version offered for use | `induce` → `result.relations[].publication.published` |
| Committed review, evidence and rule versions | `induce` → `result.induction_receipt` |
| Required jobs remaining | `induction-status` → `result.completed` and `result.pending_ids` |
| Prediction-time decisions and diagnostics | `induction-status` → `result.signal_counts` |
| Learning observations | `induction-status` → `result.observation_counts` |

A `wm-hplus-review/2` receipt supports label 1 for an actual new version with a substantive, sourced comparison. Label 0 requires a supported no-increment conclusion. In a compact submission the comparison is built from `basis.kind`, `basis.refs` and `basis.analysis`; `source` is derived from the associated execution identities. Use `recorded_checks` for sourced checker results, `derivation` for reasoning over actual inputs, or `artifact_inspection` for captured artifacts. The legacy path spells out `source`, `assessment` and `evidence_refs` directly. A refuted hypothesis can support a new boundary rule.

Read material for J before submitting. If `index_sync.state` is `deferred`, retry the identical submission to resume indexing with the saved receipt. A completed J accepts only the same submission on retry.

## Existing-entry use and utility

For an adopted rule, pass `execute --used-entry-ids ENTRY@VERSION`. An omitted
declaration leaves use unknown; pass `--used-entry-ids ''` to declare no
adoption. The version is resolved from the frozen `used_entries` declaration,
so you do not re-type it for every judgement.

At `close-episode`, read `result.entry_usage`: `entries[]` lists the adopted
`(entry, version)` with read-time utility totals and an `inspect_ref`, and
`effect_review.pending` names adoptions with no current valid judgement. Record
the judgement in the same close-out call (works for H+ = 0):

Read the entry's UTILITY summary with `orx inspect --bank strategic --entry N`
(`result.utility`, and the same block inside `rule_history.utility`):

- `totals.adoptions` — distinct `(entry, version, execution_id)` adoptions;
- `totals.judged` — adoptions with a CURRENT VALID effect judgement;
- `totals.unjudged` — adoptions with no judgement at all;
- `totals.{helped,neutral,unrelated,refuting}` — your own latest judgements;
- `totals.invalidated` — judgements whose supporting execution was excluded/
  withdrawn (the adoption and the historical judgement stay recorded);
- `unattributed_judgments` — judgements that cannot be tied to an adopted
  version; they are never guessed onto the current version;
- `latest_management`, `status`, and per-version `evidence` references.

Three facts stay separate in `status_view`: publication (is it offered?),
verification (how far was the CLAIM checked?) and usage (how did actual use
turn out?). `validated`/`legacy_status` is the historical maturity label, not
a proof of usefulness; a passing check or a publication does not prove that
using the entry helps, and the forward track's `hit_rate`/`n_predictions`
belongs to `forward_calibration` — `n_predictions = 0` means "not estimated",
not a measured zero effect. A failed task check is not `refuting`; a technical
check failure is evidence, not a utility verdict.

Record an effect judgement with the SAME close-out call (works for H+ = 0):

```bash
orx close-episode --task T --episode E --terminal completed \
  --attribute-effect '{"entry_id":"2","execution_id":"EX_USED","verdict":"helped","note":"The rule fixed the price unit before the objective was written; the recorded check then matched the reference scale."}'
```

A `refuting` judgement keeps its existing default (pause that version); a
confirmed in-scope error can be paused immediately, without waiting for a
fixed number of failures. Use `keep`/`demote`/`promote` for an explicit
management decision. Leave the judgement unevaluated when the evidence does
not support one — missing or low-sample use is NOT evidence of uselessness.
If several judgements are neutral/unrelated/refuting, or another rule or the
documented procedure now covers the case, consider revising or retiring the
entry — your decision, not an automatic threshold.

Add a sourced transfer, boundary or counterexample observation in `check-task` through `rule_assessments`. Each item identifies the `entry_id`, `version`, `purpose`, `verdict`, `source`, `assessment` and `evidence_refs`.

To add evidence or change publication for an existing body, use the compact `evidence_added` result (or `revision` when the body changes) with its real `entry_id` and `version`. Explicit assertion audits and merge remain available through the legacy `--relation`/`--verify` path. For utility judgments use the actual adoption execution.

```bash
orx induce --attribute-effect '{"entry_id":"ENTRY","execution_id":"EX_USED","verdict":"helped","lifecycle_action":"keep","note":"The adopted activation rule matched the recorded constraints and checks."}'
orx retire --entry ENTRY --reason 'The rule should leave ordinary recall under current operating conditions.'
```

| Action | Effect |
|---|---|
| `keep` | Records the judgment and keeps the selected version and status. |
| `demote` | Suspends the selected version and removes it from ordinary recall if active. |
| `promote` | Activates the selected version after identity and merge checks. |
| `retire` | Moves the entry to the cold archive. |

A management decision applies to the version the cited execution ACTUALLY
adopted: a problem found in v1 does not demote a corrected v2 that is now
active. A changed statistic never restores a paused version by itself —
restore is an explicit `promote` (or a new version with an explicit
publication decision). `retire` accepts `--evidence-execution-id` (repeatable)
so the cold-archive card keeps the stated reason, the cited facts and the
read-time usage summary; retire/restore maintain the recall index through the
same synchronization path.

`verdict:"refuting"` defaults to demotion when `lifecycle_action` is omitted. Specify `keep` to record the judgment while retaining the version. The primary `execution_id` identifies the frozen adopted version. Existing-body maintenance returns `maintenance_only:true`.

## Scoped checks, costs and corrections

Use task checks and derivations that cover the actual effect claimed. For cost comparisons, supply `comparator_execution_id` and optional `cost_dimensions`. Compare the observed settings, acceptance criteria, measured dimensions and their sources. A missing optional baseline is reported as `cost_comparison.state:unavailable`.

Keep failed attempts as contrasts. Use `exclude-execution` and `restore-execution` to correct evidence validity. Later checks can supply missing support. A learning label can return to unknown if its supporting facts are withdrawn.

## Learning feedback and experiment modes

Later `plan-next` calls in the same memory home select applicable receipts recorded before the decision cutoff. Read [prediction input audit](world_model_contract.md#4-inspect-a-prediction-and-its-input) to see which feedback reached the provider.

Use a supported covered/no-increment conclusion as negative feedback only for a specifically comparable mechanism and its premises. Insufficient evidence, unfinished or unobserved induction, and execution failures that prevent verification leave the hypothesis unknown. Empty feedback does not imply no learning opportunity; none of these unknowns justifies globally tightening H+.

Ordinary prediction homes enable H+ by default. To set it explicitly, use global `--induction-mode hplus` or `OR_INDUCTION_MODE=hplus`. Settle required reviews before freezing the library. Use separate homes for [H+, NoH+ and random controls](../docs/validation/hplus_binary.md) and follow the [fixed-memory test procedure](snapshot_lifecycle.md#fixed-memory-testing-with-per-question-copies).
