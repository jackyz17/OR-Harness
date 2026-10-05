"""The documented top-level ``result`` keys, one source, checked two ways.

The Skill and the command reference tell the agent which fields to read from
a command's JSON line (``result.execution_id`` and so on). Those paths used
to drift from the code with nothing to catch it: a rename in the CLI left
the docs pointing at a key that no longer existed, and an agent that
followed them read ``None``.

This module is the bridge between the two checks:

* ``references/examples/_check_docs.py`` (static) asserts that every
  ``result.<key>`` the agent-facing docs mention is DECLARED here — so an
  invented key fails immediately, at the doc edit;
* ``tests/harness/test_documented_result_keys.py`` (runtime) runs each
  command against a hermetic harness and asserts every DECLARED key really
  appears in the output — so a declaration that the code does not honour
  fails in CI.

Both halves are needed: the static half catches a typo in the docs, the
runtime half catches a change in the code. Keys are TOP-LEVEL only; a
nested path (``result.report.conclusion``) is declared as ``report`` here
and its internals are the owning command's business.
"""

from __future__ import annotations

from typing import Dict, Set

#: The Skill tables are the source of truth for WHICH keys matter; this map
#: records only whether a key, when cited for a command, is REQUIRED (always
#: emitted) or CONDITIONAL (emitted only under a stated condition, and noted
#: as such in the docs). The runtime test reads the Skill table, so the two
#: can never drift; this map exists to tell the test what to insist on.
#:
#: ``command -> {"required": {...}, "conditional": {...}}``. A key cited in
#: the docs that is in NEITHER set is a docs bug the runtime test reports.
DOCUMENTED_RESULT_KEYS: Dict[str, Dict[str, Set[str]]] = {
    "profile": {
        "required": {"coupling", "profile", "derivation"},
        "conditional": set(),
    },
    "recall": {
        "required": {"recommendations", "recommendations_basis", "profile"},
        "conditional": {"held_claims", "vector_recall", "degraded",
                        "available_solver_families", "solver_advisories",
                        "task_digest", "evidence_candidates",
                        "evidence_candidates_note"},
    },
    "context": {
        "required": {"context_id"},
        "conditional": set(),
    },
    "predict-strategy": {
        # The CLI wraps the prediction: the id is top-level, the rest (status,
        # benefit, cost, trace, capability_gain) ride inside ``prediction``.
        "required": {"prediction_id", "prediction"},
        "conditional": {"failure", "effective_parameters"},
    },
    "plan-next": {
        # The plan object rides under ``plan``: candidates, status, ids and
        # costs are inside it, not at the top level.
        "required": {"plan", "decision_action_id", "effective_parameters"},
        "conditional": {"protocol"},
    },
    "choose-next": {
        # ``deviation`` and ``rejected`` are ALWAYS present (None / False when
        # they do not apply), so they are required, not conditional.
        "required": {"selected", "status", "deviation", "rejected"},
        "conditional": set(),
    },
    "execute": {
        "required": {"execution_id", "execution", "action_id"},
        "conditional": {"prediction_binding"},
    },
    "check-task": {
        "required": {"state", "report", "staged", "execution_id",
                     "action_id"},
        "conditional": {"next", "reflection_material"},
    },
    "record": {
        "required": {"recorded", "cost_completeness", "induction_hints",
                     "prediction_checks", "index_sync", "execution_id"},
        "conditional": {"cost_feedback", "unrecorded_staged_executions"},
    },
    "close-episode": {
        "required": {"task_checks", "evaluations", "calibration_summary",
                     "closeout", "evidence_window", "retention",
                     "already_closed"},
        "conditional": set(),
    },
    "compare-capability": {
        "required": {"recommendation", "basis", "rule", "comparisons",
                     "incomparable"},
        "conditional": {"selected_prediction_id", "selected_operation_type"},
    },
    "evaluate-capability": {
        "required": {"evaluation", "already_evaluated"},
        "conditional": {"prediction_source"},
    },
    "bind-capability": {
        "required": {"binding", "already_bound"},
        "conditional": {"state", "prediction_source"},
    },
    "induce": {
        "required": {"results"},
        "conditional": {"relations", "revisions", "action", "created",
                        "updated", "skipped", "cost_claim_withheld"},
    },
    "induction-candidates": {
        "required": {"candidates"},
        "conditional": {"count"},
    },
    "induction-material": {
        "required": set(),
        "conditional": {"material", "bundle_id", "evidence"},
    },
    "review-material": {
        "required": {"count", "material", "existing_knowledge", "budget"},
        "conditional": {"tasks", "n_distinct_tasks", "check_states",
                        "total_completed", "n_attempts"},
    },
    "predict-capability": {
        "required": {"prediction_id"},
        "conditional": {"status", "expected_changes", "horizon"},
    },
    "accept-capability": {
        "required": {"maintenance_binding"},
        "conditional": {"adoption_action_id", "already_accepted"},
    },
    "reject-capability": {
        "required": set(),
        "conditional": {"rejected", "recommendation_id"},
    },
    "inspect": {
        "required": {"bank"},
        "conditional": {"count", "recommendations", "entries", "evaluation",
                        "evaluations", "prediction", "online_gains",
                        "n_predictions", "predictions"},
    },
}


def required_keys(command: str) -> Set[str]:
    """The keys ``command`` must ALWAYS emit (empty when not covered)."""
    entry = DOCUMENTED_RESULT_KEYS.get(command)
    return set(entry["required"]) if entry else set()


def declared_keys(command: str) -> Set[str]:
    """Every key the docs may cite for ``command`` (required + conditional)."""
    entry = DOCUMENTED_RESULT_KEYS.get(command)
    if not entry:
        return set()
    return set(entry["required"]) | set(entry["conditional"])


def all_declared_keys() -> Set[str]:
    """Every key any command declares, for the static doc scan."""
    keys: Set[str] = set()
    for entry in DOCUMENTED_RESULT_KEYS.values():
        keys |= set(entry["required"]) | set(entry["conditional"])
    return keys
