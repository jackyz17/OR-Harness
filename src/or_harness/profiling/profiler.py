"""Deterministic problem profiling.

Same input, same output — no randomness, no time dependence. Coupling
dimensions come from three channels, in priority order:

1. CIR (best): the task JSON's optional top-level ``coupling`` field — the
   pre-model Coupling-Aware Intermediate Representation.  Scalar dims are
   derived from its structure: resource relations, temporal indexes,
   network indexes.  The CIR's relational structure is the primary
   representation; these scalars are its derived ProblemSignature.
2. structured spec: explicit fields (time_periods, resources, entities...).
3. harness-supplied: ``annotations.coupling`` — the harness's own estimate.

semantic_coupling is NEVER derived as a scalar: the CIR's relations are the
semantic understanding; the scalar always stays the harness's call.

**The profile is IDENTITY, and identity is frozen before the strategy.**
Only pre-strategy inputs may define it. The ``model`` field and a solve
script are POST-strategy artifacts: the strategy may legitimately change
the formulation (decomposition, rolling horizon, relaxation), so letting
them move the structural key would make the same problem land in a
different cell depending on how it was later solved — and the key that
retrieved the evidence would no longer be the key the execution is filed
under. Both are therefore DIAGNOSTIC only: the model is verified (L1/L2)
and its coupling is reported as an observation, never used as the key.

When a supplied value conflicts with the winning derivation across a bin
boundary, the profile carries ``coupling_warnings`` so the harness can
reconsider BEFORE writing solver code. No NLP subsystem, no graph
database.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from or_harness.core.schema import (
    COUPLING_FEATURES,
    ProblemProfile,
    SCALE_FEATURES,
)
from or_harness.profiling.model_syntax import (
    ModelReport,
    coupling_from_model,
    verify_model,
)

#: Dimensions measurable from structure (semantic_coupling excluded).
STRUCTURAL_DIMENSIONS = ("resource_coupling", "temporal_coupling", "route_complexity")

#: Diagnostic threshold for the supplied-vs-derived cross-check: how far apart
#: the two values may be before the report flags them. Purely advisory — the
#: derived value always wins, and applicability is read off evidence.
CROSS_CHECK_MIN_DISAGREEMENT = 0.2


def profile_task(task: Dict[str, Any],
                 cir: Optional[Any] = None) -> ProblemProfile:
    """Build a ProblemProfile from a task JSON document.

    There is no solve-script parameter: a script is a POST-strategy
    artifact and may never move the identity key. To inspect a
    formulation's structure, write it as the ``model`` field instead — it
    is verified and reported as ``model_coupling``.

    Task schema::

        {
          "task_id": "...",            # required
          "family": "...",             # required
          "coupling": {...},           # optional CIR (pre-model coupling
                                         understanding); when present it is
                                         the BEST source for the scalar
                                         signature dimensions
          "model": "SETS: ...",        # optional GAMS-style representation
                                         (see references/modeling.md); verified
                                         and reported as a DIAGNOSTIC
                                         (never a coupling source)
          "spec": {                    # optional structured description
            "n_vars": 1000, "n_constraints": 500, "n_int_vars": 1000,
            "density": 0.01,
            "time_periods": 24, "horizon": 24,
            "resources": ["crew", "vehicle"],
            "entities": [{"name": "arc", "indexes": ["i", "j", "t"]}, ...]
          },
          "annotations": {             # optional harness knowledge
            "coupling": {"resource_coupling": 0.9, ...}
          }
        }

    Derivation priority for the structural dimensions:
    CIR (task ``coupling`` field or explicit ``cir`` argument) > spec >
    supplied.  ``semantic_coupling`` is never derived — the CIR's relations
    are the semantic understanding; the scalar stays the harness's call.
    """
    if not isinstance(task, dict):
        raise ValueError("task must be a JSON object")
    task_id = task.get("task_id")
    family = task.get("family")
    if not task_id or not family:
        raise ValueError("task requires 'task_id' and 'family'")
    spec = dict(task.get("spec") or {})
    annotations = dict(task.get("annotations") or {})
    model_text = task.get("model")

    # The CIR is a pre-model artifact: when the caller does not pass one
    # explicitly, read it from the task's optional ``coupling`` field.
    # This is what wires recall()/execute() to coupling-aware signatures.
    if cir is None:
        coupling_data = task.get("coupling")
        if isinstance(coupling_data, dict):
            from or_harness.core.coupling import CouplingAwareIR
            cir = CouplingAwareIR.from_dict(coupling_data)

    model_report: Optional[ModelReport] = None
    model_coupling: Dict[str, Optional[float]] = {}
    if isinstance(model_text, str) and model_text.strip():
        model_report = verify_model(model_text)
        if model_report.parsed is not None:
            model_coupling = coupling_from_model(model_report.parsed)

    cir_coupling: Dict[str, Optional[float]] = {}
    if cir is not None:
        from or_harness.core.coupling import coupling_from_cir
        cir_coupling = coupling_from_cir(cir)

    supplied = _supplied_coupling(task, annotations)
    derived = _derive_coupling(spec)
    # Merge per dimension by priority: CIR > spec > supplied. This is the
    # IDENTITY key, so only PRE-strategy inputs participate — the model and
    # the solve script are post-strategy artifacts and are never a source
    # here (see the module docstring). semantic_coupling is never derived.
    coupling: Dict[str, Optional[float]] = {}
    origin: Dict[str, str] = {}
    for f in COUPLING_FEATURES:
        if f == "semantic_coupling":
            coupling[f] = _clamp01(supplied.get(f))
            origin[f] = "supplied" if f in supplied else "null"
            continue
        if cir_coupling.get(f) is not None:
            coupling[f] = cir_coupling[f]
            origin[f] = "cir"
        elif derived.get(f) is not None:
            coupling[f] = derived[f]
            origin[f] = "spec"
        elif f in supplied:
            coupling[f] = _clamp01(supplied[f])
            origin[f] = "supplied"
        else:
            coupling[f] = None
            origin[f] = "null"

    source = "harness_supplied" if (supplied and origin.get("resource_coupling")
                                    == "supplied") else "derived"

    warnings = _cross_check(supplied, coupling, origin)

    scale = {f: float(spec[f]) for f in SCALE_FEATURES if f in spec}
    risk = dict(spec.get("risk_features") or {})
    profile = ProblemProfile(
        problem_id=str(task_id), family=str(family), scale_features=scale,
        semantic_coupling=coupling["semantic_coupling"],
        resource_coupling=coupling["resource_coupling"],
        temporal_coupling=coupling["temporal_coupling"],
        route_complexity=coupling["route_complexity"],
        risk_features=risk, source=source, annotations=annotations)
    # Derivation report + warnings ride along in annotations (schema-stable).
    report: Dict[str, Any] = {"origin": origin}
    if model_report is not None:
        report["model_verification"] = model_report.to_dict()
    if model_coupling:
        # DIAGNOSTIC only: what the model's structure says about coupling.
        # Reported so a divergence from the identity key is VISIBLE, but
        # deliberately not used to compute it (the model is post-strategy).
        report["model_coupling"] = dict(model_coupling)
    if warnings:
        report["coupling_warnings"] = warnings
    profile.annotations["profiling"] = report
    return profile


def derivation_report(profile: ProblemProfile) -> Dict[str, Any]:
    """The per-dimension derivation report (origin/value/note) for CLI output."""
    profiling = profile.annotations.get("profiling") or {}
    origin = profiling.get("origin") or {}
    report: Dict[str, Any] = {}
    for f in COUPLING_FEATURES:
        value = getattr(profile, f)
        entry: Dict[str, Any] = {
            "value": value,
            "origin": origin.get(f, "unknown"),
        }
        if value is None:
            entry["note"] = _null_note(f, origin.get(f))
        report[f] = entry
    if "model_verification" in profiling:
        report["model_verification"] = profiling["model_verification"]
    if "coupling_warnings" in profiling:
        report["coupling_warnings"] = profiling["coupling_warnings"]
    if "model_coupling" in profiling:
        report["model_coupling"] = profiling["model_coupling"]
    return report


# ---------------------------------------------------------------------------
# cross-check: supplied vs derived, before the harness writes solver code
# ---------------------------------------------------------------------------


def _cross_check(supplied: Dict[str, float],
                 coupling: Dict[str, Optional[float]],
                 origin: Dict[str, str]) -> List[Dict[str, Any]]:
    """Warn when a supplied value and the winning structural derivation
    (CIR > spec) disagree materially — the harness should reconsider
    before writing solve.py.  The winning value is used for grouping either
    way (applicability is read off evidence, so this is a diagnostic, not a
    grouping rule)."""
    warnings: List[Dict[str, Any]] = []
    for f in STRUCTURAL_DIMENSIONS:
        if f not in supplied:
            continue
        structural = coupling.get(f)
        if structural is None:
            continue
        gap = abs(float(supplied[f]) - float(structural))
        if gap >= CROSS_CHECK_MIN_DISAGREEMENT:
            warnings.append({
                "dimension": f,
                "supplied": round(float(supplied[f]), 4),
                "derived": round(float(structural), 4),
                "gap": round(gap, 4),
                "origin": origin.get(f, "unknown"),
                "message": (
                    f"you supplied {f}={supplied[f]:.2f} but structural "
                    f"derivation says {structural:.2f} (gap {gap:.2f}) — "
                    f"reconsider before writing solver code (the derived "
                    f"value is what gets used)"),
            })
    return warnings


def _null_note(dimension: str, origin: str) -> str:
    if dimension == "semantic_coupling":
        return ("semantic coupling is never derived (business semantics are "
                "invisible to structure); supply annotations.coupling."
                "semantic_coupling to keep it in the profile — it informs "
                "your own judgment and never conditions the statistics")
    if origin == "supplied":
        return "supplied"
    return ("no structural signal found: provide the 'model' field (best), "
            "solver code (--code), or documented spec fields "
            "(time_periods/horizon, resources+entities, routes/network)")


# ---------------------------------------------------------------------------
# harness-supplied channel
# ---------------------------------------------------------------------------


def _supplied_coupling(task: Dict[str, Any],
                       annotations: Dict[str, Any]) -> Dict[str, float]:
    merged: Dict[str, float] = {}
    for container in (task, annotations, annotations.get("coupling") or {}):
        if not isinstance(container, dict):
            continue
        for f in COUPLING_FEATURES:
            if f in container and container[f] is not None:
                merged[f] = float(container[f])
    return merged


# ---------------------------------------------------------------------------
# derived channel (code AST + spec)
# ---------------------------------------------------------------------------


def _derive_coupling(spec: Dict[str, Any]) -> Dict[str, Optional[float]]:
    signals: Dict[str, float] = {}
    signals.update(_spec_signals(spec))

    resource = _combine([
        _ratio(spec.get("resources"), spec.get("entities")),
        signals.get("shared_resource_ratio"),
    ])
    temporal = _combine([
        _temporal_from_spec(spec),
        signals.get("temporal_index_ratio"),
    ])
    route = _combine([
        _route_from_spec(spec),
        signals.get("cooccurrence_rate"),
    ])
    return {
        "semantic_coupling": None,  # never derived — always harness-supplied
        "resource_coupling": resource,
        "temporal_coupling": temporal,
        "route_complexity": route,
    }


def _spec_signals(spec: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    entities = spec.get("entities") or []
    if isinstance(entities, list) and entities:
        indexed = sum(1 for e in entities
                      if isinstance(e, dict) and len(e.get("indexes") or []) >= 2)
        out["cooccurrence_rate"] = indexed / len(entities)
    return out


def _clamp01(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return max(0.0, min(1.0, float(value)))


def _ratio(numerator: Any, denominator: Any) -> Optional[float]:
    if not isinstance(numerator, list) or not isinstance(denominator, list):
        return None
    if not denominator:
        return None
    return min(1.0, len(numerator) / max(1, len(denominator)))


def _temporal_from_spec(spec: Dict[str, Any]) -> Optional[float]:
    periods = spec.get("time_periods", spec.get("horizon"))
    if not isinstance(periods, (int, float)) or periods <= 0:
        return None
    # More periods -> stronger temporal structure; saturates around 50.
    return min(1.0, float(periods) / 50.0)


def _route_from_spec(spec: Dict[str, Any]) -> Optional[float]:
    if "routes" in spec and isinstance(spec["routes"], (int, float)):
        return min(1.0, float(spec["routes"]) / 500.0)
    if spec.get("network") is True:
        return 0.8
    return None


def _combine(values) -> Optional[float]:
    present = [v for v in values if v is not None]
    if not present:
        return None
    return round(sum(present) / len(present), 6)
