# R19 · structural retrieval support and verification propagation

## P1.1 · empty predicates do not occupy structural recommendation slots

`StrategicBank.matching()` remains the bank-level non-conflict view: it keeps
`not_specified` / `unknown` entries readable with their machine state. The
SELECTOR's structural recall is narrower because it is a claim about this
task's structure.

`Selector._matching_entries()` now offers only entries whose deterministic
machine state is `compatible` or `partial`:

- `compatible`: every recorded machine condition is satisfied (a necessary,
  not sufficient, structural match);
- `partial`: at least one recorded condition is satisfied and another is
  unknown; reported with its incomplete state and warning;
- `not_specified` (empty predicates) and `unknown` (unmeasured/semantic-only
  conditions) are NOT structural matches and do not consume a `top` slot.

The text/embedding channel is unchanged: it still discovers empty-predicate
and unknown-condition rules by their text. A rule can therefore be absent from
`recommendations` while present in `vector_recall.strategic_knowledge`.

Recall now reports `recall_budget.omitted_without_machine_support` and
`recommendations_basis.omitted_without_machine_support`. When no structural
row survives, the basis text names those entries and points at the semantic
channel instead of claiming "no memory".

## P1.2 · verification state is independent of current applicability

The entry's historical `verification_state` is carried explicitly on the
structural `Recommendation` and on the structural retrieval hit. Applicability
remains a separate layer:

- a verified rule is `verified_knowledge` even when the current task requires
  the agent's `when` assessment (`requires_agent_assessment` is not a
  downgrade);
- `classify_evidence` reads the explicit top-level state first, and falls back
  to the nested `knowledge.verification_state` carried by older/frozen shapes;
- the provider projection reclassifies from merged content, so a
  structural-only hit keeps its verified/legacy/unverified class.

Publication, verification, machine compatibility and semantic assessment stay
separate facts.

## Tests

- `test_vector_recall.TestStructuralSupportSelection`:
  - 29 empty-predicate entries do not crowd a confirmed rule out of `top=3`;
  - omission counts are reported;
  - an empty-predicate rule is still returned by semantic retrieval.
- `test_prediction_context.TestRetrievalEvidence`:
  - a verified entry that is structural-only (not indexed) stays
    `verified_knowledge` in the retrieval view and in the provider view;
  - `classify_evidence` keeps the nested verification state for old shapes.
