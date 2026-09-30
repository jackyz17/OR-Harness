"""Which identity problems block which part of a comparison.

The binding layer (``bind_strategy_outcome``) records two DIFFERENT kinds
of identity problem:

* a **mismatch** — the executed action KNOWN to differ from the predicted
  candidate (another task, another solver, another value for a config key);
  the run was not the predicted run.
* an **unknown** — the executed action could not CONFIRM a field (no
  episode on the action, a config key the run never reported back).

Both must reach the close-out, and the close-out must decide ONE FIELD AT
A TIME what may still be compared. Collapsing the two into "the identity
is not established, discard the sample" threw away real observations for a
missing receipt, which is what this module exists to prevent.

The rule is a tiny, FRAMEWORK-OWNED TABLE — deliberately not a dependency
engine over field names:

* a problem with the PROBLEM/TASK identity (``task_id``, ``action_type``,
  ``timing``, ``window_round``) invalidates everything, including cost: the
  sample is not this prediction's at all.
* a problem with the EXECUTION CONDITIONS (``solver``, ``strategy_id``,
  ``effective_input``, any ``config`` key) invalidates the OUTCOME
  dimensions (benefit / risk / interval) but PRESERVES cost: a different
  condition really did run and really did cost what it cost, and the user's
  rule is that the spend and the failure stay on record.
* an UNKNOWN config key changes nothing unless it names the APPROACH (see
  :data:`APPROACH_CONFIG_KEYS`). An unreported ``time_limit`` or ``seed``
  does not invalidate an observation; an unreported ``relaxation`` does,
  because the answer may not be the predicted approach's answer.
* an unknown ``episode_id`` changes nothing: the observation is the bound
  action's own execution, and a MISSING label is a receipt the run did not
  fill in, not a proven mismatch.
* a KNOWN ``episode_id`` MISMATCH is different from an unknown one: the
  prediction declared one episode and the execution is on record as a
  DIFFERENT one, so the execution is provably not the predicted run's. It
  invalidates everything — the same standing as a wrong ``task_id``. Real
  runs from different episodes must never be paired.
"""

from __future__ import annotations

from typing import Any, Dict, FrozenSet, Mapping, Optional

#: The four comparison blocks an evaluation may carry.
EVALUATION_DIMENSIONS = ("benefit", "cost", "risk", "interval")

_ALL: FrozenSet[str] = frozenset(EVALUATION_DIMENSIONS)
#: Everything except cost. A different execution condition leaves the spend
#: on record while making the quality/risk comparison meaningless.
_OUTCOME: FrozenSet[str] = frozenset({"benefit", "risk", "interval"})
_BENEFIT_ONLY: FrozenSet[str] = frozenset({"benefit"})
#: A benefit and the interval drawn around the SAME value are two readings
#: of ONE observation. They must share one eligibility: if the benefit may
#: not be scored, an interval "covering" the same value would be scoring it
#: anyway. ``risk`` is left alone — it is a per-event label, not the same
#: number.
_BENEFIT_AND_INTERVAL: FrozenSet[str] = frozenset({"benefit", "interval"})
_NONE: FrozenSet[str] = frozenset()

#: Config keys that name the APPROACH rather than the effort spent. An
#: UNCONFIRMED value for one of these blocks the benefit comparison (the
#: answer may not be the predicted approach's answer); every other config
#: key leaves the comparison intact. Config keys that merely tune effort
#: (``time_limit``, ``mip_gap``, ``seed``, ``threads``) are deliberately
#: absent: an unreported budget never discards a real observation.
APPROACH_CONFIG_KEYS: FrozenSet[str] = frozenset({
    "method", "method_performed", "algorithm", "formulation",
    "relaxation", "integer", "integrality", "decomposition",
    "heuristic", "cutting_plane", "warm_start",
})

#: Dimension impact of a KNOWN disagreement, by identity field.
#:
#: A disagreement about WHICH THING was predicted (the task, the action
#: type, the strategy, the problem input, the timing) rejects the sample
#: entirely: the run was simply not the predicted run, and comparing any of
#: its numbers would teach the model about a thing it never predicted.
#:
#: A disagreement about the CONDITIONS the predicted strategy ran under (a
#: different solver, a different value for a config key) is a DEVIATION: the
#: strategy really ran and really spent what it spent, so the outcome
#: dimensions go (the quality is not the original condition's performance)
#: while cost stays.
MISMATCH_IMPACT: Dict[str, FrozenSet[str]] = {
    "action_type": _ALL,
    "task_id": _ALL,
    "timing": _ALL,
    "window_round": _ALL,
    "strategy_id": _ALL,
    "effective_input": _ALL,
    "episode_id": _ALL,
    "solver": _OUTCOME,
    # A config key is handled separately (see ``binding_attribution``):
    # ANY config disagreement is a condition deviation.
    "config": _OUTCOME,
}

#: Dimension impact of an UNCONFIRMED (unknown) field.
#:
#: An unknown field is a MISSING RECEIPT, not a disagreement. It blocks
#: only the comparison it really makes unreadable: an unreported effort
#: knob (``time_limit``, ``seed``) blocks nothing; an unreported APPROACH
#: knob blocks the benefit. A task-level unknown that leaves the outcome
#: unreadable blocks the outcome and keeps cost.
UNKNOWN_IMPACT: Dict[str, FrozenSet[str]] = {
    "action_type": _NONE,
    "task_id": _NONE,
    "timing": _NONE,
    "window_round": _NONE,
    "episode_id": _NONE,
    "effective_input": _OUTCOME,
    "strategy_id": _NONE,
    "solver": _NONE,
    "config": _NONE,
}

#: A mismatch on a field this table does not know is treated as a
#: task-level problem (block everything): a KNOWN disagreement is never
#: waved through just because its name is new.
_DEFAULT_MISMATCH_IMPACT = _ALL


def _field_impact(field: str, *, kind: str,
                  config_key: str = "") -> FrozenSet[str]:
    if kind == "mismatch":
        if config_key:
            return MISMATCH_IMPACT["config"]
        return MISMATCH_IMPACT.get(field, _DEFAULT_MISMATCH_IMPACT)
    # unknown
    if config_key:
        return (_BENEFIT_ONLY if config_key in APPROACH_CONFIG_KEYS
                else _NONE)
    return UNKNOWN_IMPACT.get(field, _NONE)


def binding_attribution(mismatch: Mapping[str, Any],
                        unknown: Mapping[str, Any],
                        method_observed: Any = None) -> Dict[str, Any]:
    """Translate the binding's mismatch/unknown record into per-dimension
    blocking.

    Returns::

        {
          "blocked": {dimension: [{"field": ..., "kind": ...}, ...]},
          "fields": {"<field>": {"kind", "blocks", "reason"}},
          "notes": [...],
        }

    ``blocked`` holds only the dimensions that must NOT be compared;
    a dimension absent from a list is free to be evaluated. The reason
    strings are short and actionable — they are shown to the agent.

    ``method_observed`` is the binding's own ``compare_methods`` record (a
    performed method that really differs from the plan). It blocks the
    BENEFIT and its INTERVAL only: the answer is not scored as the planned
    method's result, while the cost and any failure stay on record.
    """
    blocked: Dict[str, list] = {dim: [] for dim in EVALUATION_DIMENSIONS}
    fields: Dict[str, Dict[str, Any]] = {}
    notes: list = []

    def _add(field: str, kind: str, dims: FrozenSet[str], reason: str
             ) -> None:
        for dim in dims:
            blocked[dim].append({"field": field, "kind": kind})
        fields[field] = {"kind": kind, "blocks": sorted(dims),
                         "reason": reason}

    for name, entry in (mismatch or {}).items():
        if name == "config":
            for key in (entry or {}):
                _add(f"config.{key}", "mismatch",
                     _field_impact("config", kind="mismatch",
                                   config_key=str(key)),
                     f"the run used a different value for the config key "
                     f"{key!r}: this is not the predicted configuration, so "
                     "the outcome comparison is not this prediction's truth "
                     "(the real cost is still kept)")
            continue
        _add(str(name), "mismatch", _field_impact(str(name), kind="mismatch"),
             _mismatch_reason(str(name)))

    for name, entry in (unknown or {}).items():
        if name == "config":
            for key in (entry or {}):
                key = str(key)
                if f"config.{key}" in fields:
                    continue          # a known disagreement already governs
                _add(f"config.{key}", "unknown",
                     _field_impact("config", kind="unknown", config_key=key),
                     _unknown_config_reason(key))
            continue
        name = str(name)
        if name in fields:
            continue
        _add(name, "unknown", _field_impact(name, kind="unknown"),
             _unknown_reason(name))

    # A performed method that differs from the plan. It is neither a
    # mismatch nor an unknown: the run really happened, but the answer is
    # not the planned method's answer.
    deviation = method_deviation(method_observed)
    if deviation is not None and "method" not in fields:
        _add("method", "deviation", _BENEFIT_AND_INTERVAL,
             method_deviation_note(deviation))

    soft = sorted(f for f, e in fields.items() if not e["blocks"])
    if soft:
        notes.append(
            "these identity fields could not be confirmed but block "
            "nothing: " + ", ".join(soft) +
            " — the observation is the bound action's own execution, and a "
            "missing receipt is a caveat, not grounds to discard a real "
            "attempt")
    return {"blocked": blocked, "fields": fields, "notes": notes, "deviation":
            deviation}


def blocked_dimensions(attribution: Mapping[str, Any]) -> Dict[str, Any]:
    """The dimension -> blocking-field map, with a compact reason per dim."""
    out: Dict[str, Any] = {}
    for dim, entries in (attribution.get("blocked") or {}).items():
        if entries:
            out[dim] = list(entries)
    return out


def eligibility_for(entries: list) -> str:
    """The eligibility value for a blocked dimension.

    A KNOWN disagreement is ``identity_mismatch``; an unconfirmed field that
    still blocks the dimension is ``identity_unknown``. The two are
    different facts and a reader must be able to tell them apart.
    """
    if any(str(e.get("kind")) == "mismatch" for e in (entries or [])):
        return "identity_mismatch"
    return "identity_unknown"


def _mismatch_reason(field: str) -> str:
    if field in ("task_id", "action_type"):
        return ("the bound action belongs to a different task/action type: "
                "this sample is not this prediction's at all")
    if field == "timing":
        return ("the prediction was made after the action began: hindsight "
                "is not a prediction")
    if field == "window_round":
        return ("the bound action is not inside the declared selection "
                "round's window")
    if field == "strategy_id":
        return ("the run executed a different strategy: this sample is not "
                "the predicted strategy's outcome at all")
    if field == "solver":
        return ("the run used a different solver: the quality is not the "
                "predicted solver's performance (the real cost and any "
                "failure are still kept)")
    if field == "episode_id":
        return ("the run happened in a different episode: it is a different "
                "real run, not this prediction's execution, so no dimension "
                "is scored against it")
    if field == "effective_input":
        return ("the problem changed between the prediction and the "
                "execution: this sample is not the predicted problem's "
                "outcome at all")
    return ("the executed action differs from the predicted candidate on "
            "this field")


def _unknown_reason(field: str) -> str:
    if field == "episode_id":
        return ("the executed action records no episode id, so the "
                "prediction's episode claim cannot be confirmed — this "
                "blocks nothing: the observation is the bound action's own "
                "execution")
    if field == "effective_input":
        return ("the problem identity the execution ran under could not be "
                "established: the outcome may belong to another problem "
                "input (the real cost is still kept)")
    return ("the executed action could not confirm this identity field; it "
            "blocks nothing by itself")

def _unknown_config_reason(key: str) -> str:
    if key in APPROACH_CONFIG_KEYS:
        return (f"the run did not report {key!r}, which names the APPROACH: "
                "the answer may not be the predicted approach's answer, so "
                "the benefit is not scored (the real cost is still kept). "
                f"Have solve.py read {key!r} back into result.json's "
                "`config` to confirm it")
    return (f"the run did not report {key!r}: it is a caveat, not grounds "
            "to discard the attempt — this blocks nothing")


# ---------------------------------------------------------------------------
# method deviation: a plan the run did not carry out
# ---------------------------------------------------------------------------

#: Verdicts from :func:`or_harness.core.schema.compare_methods` that mean
#: the performed method was NOT the planned method. ``unknown`` is not one
#: of them: an unobserved performance is not evidence of a deviation.
METHOD_DEVIATION_VERDICTS = ("mismatch",)


def method_deviation(method_observed: Any) -> Optional[Dict[str, Any]]:
    """Read a ``compare_methods`` observation, or None when there is none.

    The comparison record is produced by the BINDING (``method_observed``
    in the prediction's ``trace.model_info``) and is the ONLY honest source
    of "the run did not carry out the plan". Returns a small, actionable
    summary when the performed method really differs from the plan; returns
    ``None`` when either side is unknown (an absent plan or an unobserved
    performance is NOT a deviation).
    """
    if not isinstance(method_observed, dict):
        return None
    verdict = str(method_observed.get("verdict") or "")
    if verdict not in METHOD_DEVIATION_VERDICTS:
        return None
    return {
        "verdict": verdict,
        "planned_name": method_observed.get("planned_name"),
        "actual_name": method_observed.get("actual_name"),
        "planned_steps": list(method_observed.get("planned_steps") or []),
        "actual_steps": list(method_observed.get("actual_steps") or []),
        "reason": str(method_observed.get("reason") or ""),
    }


def method_deviation_note(deviation: Mapping[str, Any]) -> str:
    """A short note explaining what a method deviation DOES and does not do."""
    planned = deviation.get("planned_name") or "(unnamed plan)"
    actual = deviation.get("actual_name") or "(unnamed performance)"
    return (f"the run performed {actual!r}, not the planned {planned!r}: the "
            "answer and the interval drawn around it are not scored as the "
            "planned method's benefit (the cost and any failure really "
            "happened and stay on record). Nothing about the planned method "
            "can be concluded from this attempt")


# ---------------------------------------------------------------------------
# entry-point field discipline for a candidate's config
# ---------------------------------------------------------------------------

#: Config keys that are really the METHOD, not an execution parameter. A
#: caller that puts one of these in ``config`` has confused two fields
#: (``method`` is the approach, ``config`` is what the execution runs with).
METHOD_AS_CONFIG_KEYS: FrozenSet[str] = frozenset({
    "method", "method_planned", "method_performed", "method_actual",
})

#: Config keys that carry no execution information at all (identity or
#: bookkeeping): they can never be observed back from a run, so predicting
#: them would always read as an unknown. Refused at entry so the caller
#: never spends an attempt on a sample that cannot be confirmed.
NON_EXECUTION_CONFIG_KEYS: FrozenSet[str] = frozenset({
    "task_id", "episode_id", "strategy_id", "action_type", "prediction_id",
    "action_id", "window_id", "scope", "config", "benefit", "risk", "cost",
})


def check_candidate_config(config: Any, method: Any = None
                           ) -> Dict[str, Any]:
    """Validate a candidate's ``config`` at ENTRY, not after the run.

    Returns::

        {
          "config": <normalized config>,   # method keys moved out
          "method": <merged method>,       # method keys folded in
          "notes": [str, ...],             # what was normalized, and why
          "errors": [str, ...],            # unambiguous misuse, refused now
        }

    Two kinds of misuse are caught BEFORE an attempt is spent:

    * a METHOD in ``config`` (``method`` / ``method_performed`` /
      ``method_planned`` / ``method_actual``) is folded into the ``method``
      field when that is unambiguous, and reported. The framework has ONE
      field for the approach; the config is what the run executes with.
    * a NON-EXECUTION key (``task_id``, ``scope``, ...) is an error: it can
      never be observed back from a run, so keeping it would guarantee an
      ``identity_unknown`` for a value nobody could ever confirm.

    Everything else (``time_limit``, ``mip_gap``, ``seed``, ...) passes
    through untouched: a config key the run does not report back is a
    caveat, not a mistake.
    """
    if not isinstance(config, dict):
        return {"config": {}, "method": method, "notes": [], "errors": []}
    normalized: Dict[str, Any] = {}
    method_keys: Dict[str, Any] = {}
    errors: list = []
    notes: list = []
    for key, value in config.items():
        key = str(key)
        if key in METHOD_AS_CONFIG_KEYS:
            method_keys[key] = value
            continue
        if key in NON_EXECUTION_CONFIG_KEYS:
            errors.append(
                f"config key {key!r} is not an execution parameter: it can "
                "never be read back from a run, so predicting it would "
                "always be an unconfirmable unknown. Put it in its own "
                "field (or drop it) instead of `config`")
            continue
        normalized[key] = value
    merged_method = method
    if method_keys:
        # The approach belongs in ``method``. Fold it in when unambiguous —
        # i.e. when the caller did not already declare a method, and the
        # supplied value is a method-shaped mapping (or a bare string name).
        candidate_method = (method_keys.get("method")
                            or method_keys.get("method_planned")
                            or method_keys.get("method_performed")
                            or method_keys.get("method_actual"))
        if method in (None, {}, "") and candidate_method not in (None, "", {}):
            merged_method = candidate_method
            notes.append(
                "the config key(s) " +
                ", ".join(sorted(method_keys)) +
                " described the APPROACH, so they were moved to the "
                "candidate's `method` field; `config` holds execution "
                "parameters only")
        else:
            notes.append(
                "the config key(s) " + ", ".join(sorted(method_keys)) +
                " describe the APPROACH and were ignored: the candidate "
                "already declares a `method`. `config` is for execution "
                "parameters only")
    return {"config": normalized, "method": merged_method,
            "notes": notes, "errors": errors}


def normalize_config_method(config: Any) -> Dict[str, Any]:
    """Fold a METHOD out of ``config`` using the SAME rule as
    :func:`check_candidate_config`, WITHOUT needing the candidate's existing
    method.

    :meth:`CandidateRef.from_dict` uses this so a stored candidate whose
    payload carried ``config.method`` as a bare string (a legacy/foreign
    shape) is normalized on the way IN, rather than crashing later when its
    ``config`` mapping is read. It is deliberately a thin wrapper so the two
    entry points can never apply different rules.
    """
    result = check_candidate_config(config, method=None)
    return {"config": result["config"], "method": result["method"],
            "errors": result["errors"]}


