"""Is a recorded cost number usable as a MEASURED fact?

A cost dimension can be ``measured`` in the storage sense (the mask says so)
and still not be usable as a calibration truth:

* an ``agent_estimate`` is a DECLARATION — 20000 declared tokens are not
  20000 observed tokens, and scoring a prediction against a declaration
  teaches the model nothing about the world;
* a completion-only figure (``token_basis == "completion_only"``) is a
  LOWER BOUND on the true total — comparing it with a full-口径 prediction
  is a unit error;
* a dimension marked measured by nobody/differently is UNKNOWN, and unknown
  is never zero.

Before this module every consumer read ``CostVector.measured_dims()``
alone, so a declared estimate and a provider measurement were
indistinguishable. The rule lives HERE, once:

* :func:`cost_eligibility` — the verdict for ONE (record, dimension).

Consumers apply it by SKIPPING a non-measured value; they do not build a
second report of what they skipped.

The rules (deliberately small; no new bank, no new lifecycle):

* ``provider_usage``  -> measured (a real observation from the provider);
* ``agent_observed``  -> measured (the harness really counted it);
* ``agent_estimate``  -> estimate (kept for display, never a truth);
* no provenance entry -> measured (the executor's OWN observation:
  ``latency_s`` / ``solver_runtime_s`` are measured by the framework, and
  a record that never went through ``update_cost`` has no declaration);
* no measurement at all -> unknown;
* an explicit 0 from a TRUSTED source is a valid measurement (a real zero
  is evidence); 0 from an estimate is still not a truth.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

from or_harness.core.schema import COST_DIMENSIONS

#: Eligibility verdicts for one (record, dimension).
MEASURED = "measured"
ESTIMATE = "estimate"
PARTIAL = "partial"
UNKNOWN = "unknown"

#: Sources that are real observations of the number.
_TRUSTED_SOURCES = ("provider_usage", "agent_observed")

#: Token口径 values that are a LOWER BOUND rather than a complete total.
_PARTIAL_TOKEN_BASES = ("completion_only", "prompt_only")


def _provenance_of(record: Any, dim: str) -> Dict[str, Any]:
    provenance = (getattr(record, "execution_features", None) or {}) \
        .get("cost_provenance") or {}
    entry = provenance.get(dim)
    return dict(entry) if isinstance(entry, dict) else {}


def cost_eligibility(record: Any, dim: str) -> str:
    """The eligibility of ONE record's cost dimension.

    Returns one of :data:`MEASURED` / :data:`ESTIMATE` / :data:`PARTIAL` /
    :data:`UNKNOWN`. Only :data:`MEASURED` may enter a calibrated actual or
    a learning evidence mean; the others are excluded WITH this reason.
    """
    if dim not in COST_DIMENSIONS:
        return UNKNOWN
    cost = getattr(record, "cost", None)
    if cost is None or dim not in cost.measured_dims():
        return UNKNOWN
    entry = _provenance_of(record, dim)
    source = entry.get("source")
    if source == "agent_estimate":
        return ESTIMATE
    if source is not None and source not in _TRUSTED_SOURCES:
        # An unknown source is a declaration until proven otherwise.
        return ESTIMATE
    # A token count recorded from a SINGLE side is a lower bound, never a
    # complete total: it is a real observation of PART of the quantity.
    if dim == "llm_tokens":
        basis = entry.get("basis")
        if basis in _PARTIAL_TOKEN_BASES:
            return PARTIAL
    return MEASURED
