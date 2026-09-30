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
indistinguishable. The rule lives HERE, once, and each consumer calls it:

* :func:`cost_eligibility` — the verdict for ONE (record, dimension).
* :func:`eligible_cost_records` — the records whose dimension may be used
  as a measured fact.

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


def cost_eligibility_reason(record: Any, dim: str) -> str:
    """A short, actionable reason for a non-measured verdict."""
    verdict = cost_eligibility(record, dim)
    entry = _provenance_of(record, dim)
    if verdict == ESTIMATE:
        return (f"{dim} was DECLARED (source={entry.get('source')!r}), not "
                "observed: it is shown but never used as a calibration "
                "truth or a measured cost evidence claim")
    if verdict == PARTIAL:
        return (f"{dim} was recorded under the token口径 "
                f"{entry.get('basis')!r}, which is a LOWER bound (the "
                "provider reported only one side): it cannot stand in for a "
                "complete total")
    return (f"{dim} was not measured on this record: unknown is never "
            "treated as zero")


def eligible_cost_records(records: Iterable[Any],
                          dim: str) -> list:
    """The records whose ``dim`` may be used as a MEASURED fact."""
    return [r for r in (records or [])
            if cost_eligibility(r, dim) == MEASURED]


def record_cost_exclusions(records: Iterable[Any]) -> Dict[str, Any]:
    """Per-dimension exclusions for a group of records, with counts.

    ``{dim: {"n_measured", "n_total", "excluded": {verdict: n}}}`` — only
    dimensions that had at least one excluded record are listed, so a
    clean group reports nothing.
    """
    records = list(records or [])
    out: Dict[str, Any] = {}
    for dim in COST_DIMENSIONS:
        excluded: Dict[str, int] = {}
        n_measured = 0
        for rec in records:
            verdict = cost_eligibility(rec, dim)
            if verdict == MEASURED:
                n_measured += 1
            else:
                excluded[verdict] = excluded.get(verdict, 0) + 1
        # "unknown" alone is not an exclusion (the dimension was simply not
        # measured there); a declared/partial value IS.
        non_truth = {k: v for k, v in excluded.items() if k != UNKNOWN}
        if non_truth:
            out[dim] = {
                "n_measured": n_measured,
                "n_total": len(records),
                "excluded": non_truth,
                "note": ("only measured values may stand as the real total; "
                         "declared estimates and single-side token figures "
                         "are excluded from it (they remain visible in the "
                         "per-dimension breakdown)"),
            }
    return out


def screen_aggregate(aggregate: Dict[str, Any],
                     records: Iterable[Any]) -> Dict[str, Any]:
    """Fold declaration screening into an ``aggregate_costs`` result.

    The aggregate's ``total`` comes from the records' ``measured_dims()``
    mask. This narrows it to the ELIGIBLE records and keeps the excluded
    values visible under ``excluded``. Returns a NEW dict; the input is not
    modified.

    Only dimensions that actually had a non-truth contributor (a DECLARED
    estimate or a single-side token figure) are reported — a dimension
    nobody measured is simply unknown, not "screened". The per-dimension
    exclusions are computed in ONE pass over the records.
    """
    from or_harness.core.schema import accumulate_measured_costs

    records = list(records or [])
    if not records:
        return {}
    # One pass: classify every record for every dimension.
    by_dim: Dict[str, Dict[str, int]] = {
        dim: {} for dim in COST_DIMENSIONS}
    totals: Dict[str, Dict[str, float]] = {
        dim: {d: 0.0 for d in COST_DIMENSIONS} for dim in COST_DIMENSIONS}
    counts: Dict[str, Dict[str, int]] = {
        dim: {d: 0 for d in COST_DIMENSIONS} for dim in COST_DIMENSIONS}
    for rec in records:
        measured = rec.cost.measured_dims()
        for dim in COST_DIMENSIONS:
            verdict = cost_eligibility(rec, dim)
            by_dim[dim][verdict] = by_dim[dim].get(verdict, 0) + 1
            if verdict == MEASURED and dim in measured:
                accumulate_measured_costs([rec.cost], totals[dim],
                                          counts[dim])
    screened: Dict[str, Any] = {}
    for dim in COST_DIMENSIONS:
        non_truth = {k: v for k, v in by_dim[dim].items()
                     if k not in (MEASURED, UNKNOWN)}
        if not non_truth:
            continue
        n_eligible = by_dim[dim].get(MEASURED, 0)
        screened[dim] = {
            "total": (round(totals[dim][dim], 6) if counts[dim][dim] else None),
            "n_measured": counts[dim][dim],
            "n_eligible_items": n_eligible,
            "n_items": len(records),
            "excluded": non_truth,
            "note": ("only measured values may stand as the real total; "
                     "declared estimates and single-side token figures are "
                     "excluded from it (they remain visible in the "
                     "per-dimension breakdown)"),
        }
    return screened
