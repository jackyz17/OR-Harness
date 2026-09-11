"""Deterministic problem profiling.

Same input, same output — no randomness, no time dependence. Coupling
dimensions come from five channels, in priority order:

1. CIR (best): the task JSON's optional top-level ``coupling`` field — the
   pre-model Coupling-Aware Intermediate Representation.  Scalar dims are
   derived from its structure: resource relations, temporal indexes,
   network indexes.  The CIR's relational structure is the primary
   representation; these scalars are its derived ProblemSignature.
2. model representation: the optional top-level ``model`` field — a
   GAMS-style five-block DSL the harness writes BEFORE solve.py
   (SKILL.md convention). resource_coupling = fraction of variables
   appearing in more than one constraint; verified (L1/L2) for free.
3. solve-script AST (fallback at execute time): variable co-occurrence,
   shared-resource ratios, temporal index structure.
4. structured spec: explicit fields (time_periods, resources, entities...).
5. harness-supplied: ``annotations.coupling`` — the harness's own estimate.

semantic_coupling is NEVER derived as a scalar: the CIR's relations are the
semantic understanding; the scalar always stays the harness's call.

When a supplied value conflicts with the winning derivation across a bin
boundary, the profile carries ``coupling_warnings`` so the harness can
reconsider BEFORE writing solver code. When both CIR and model are present,
inconsistencies surface as ``cir_warnings``. No NLP subsystem, no graph
database.
"""

from __future__ import annotations

import ast
from typing import Any, Dict, List, Optional

from or_harness.core.schema import (
    COUPLING_FEATURES,
    FINE_BIN_EDGES,
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


def profile_task(task: Dict[str, Any], code: Optional[str] = None,
                 cir: Optional[Any] = None) -> ProblemProfile:
    """Build a ProblemProfile from a task JSON document.

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
                                         and used as a coupling source
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
    CIR (task ``coupling`` field or explicit ``cir`` argument) > model >
    code/spec > supplied.  ``semantic_coupling`` is never derived — the
    CIR's relations are the semantic understanding; the scalar stays the
    harness's call.
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
    derived = _derive_coupling(spec, code)
    # Merge per dimension by priority: CIR > model > code/spec > supplied.
    # semantic_coupling is never derived — supplied only.
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
        elif model_coupling.get(f) is not None:
            coupling[f] = model_coupling[f]
            origin[f] = "model"
        elif derived.get(f) is not None:
            coupling[f] = derived[f]
            origin[f] = "code" if code else "spec"
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
    if warnings:
        report["coupling_warnings"] = warnings
    # CIR ↔ model cross-check (optional): the CIR is a pre-model artifact;
    # the model may later be used to verify it, but is never required to
    # create one.  When both are present, inconsistencies are flagged here.
    if cir is not None:
        from or_harness.core.coupling import cross_check_cir_model
        cir_warnings = cross_check_cir_model(cir, model_report.parsed if model_report else None)
        if cir_warnings:
            report["cir_warnings"] = cir_warnings
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
    if "cir_warnings" in profiling:
        report["cir_warnings"] = profiling["cir_warnings"]
    return report


# ---------------------------------------------------------------------------
# cross-check: supplied vs derived, before the harness writes solver code
# ---------------------------------------------------------------------------


def _cross_check(supplied: Dict[str, float],
                 coupling: Dict[str, Optional[float]],
                 origin: Dict[str, str]) -> List[Dict[str, Any]]:
    """Warn when a supplied value and the winning structural derivation
    (CIR > model > code/spec) disagree across a bin boundary — the harness
    should reconsider before writing solve.py.  The winning value is used
    for grouping either way."""
    warnings: List[Dict[str, Any]] = []
    for f in STRUCTURAL_DIMENSIONS:
        if f not in supplied:
            continue
        structural = coupling.get(f)
        if structural is None:
            continue
        supplied_bin = _bin_index(supplied[f])
        derived_bin = _bin_index(structural)
        if supplied_bin != derived_bin:
            warnings.append({
                "dimension": f,
                "supplied": round(float(supplied[f]), 4),
                "derived": round(float(structural), 4),
                "origin": origin.get(f, "unknown"),
                "message": (
                    f"you supplied {f}={supplied[f]:.2f} but structural "
                    f"derivation says {structural:.2f}; these fall in "
                    f"different similarity bins — reconsider before writing "
                    f"solver code (the derived value will be used for "
                    f"grouping)"),
            })
    return warnings


def _bin_index(value: float) -> int:
    v = max(0.0, min(1.0, float(value)))
    for i, (lo, hi) in enumerate(zip(FINE_BIN_EDGES, FINE_BIN_EDGES[1:])):
        if lo <= v <= hi:
            return i
    return len(FINE_BIN_EDGES) - 2


def _null_note(dimension: str, origin: str) -> str:
    if dimension == "semantic_coupling":
        return ("semantic coupling is never derived (business semantics are "
                "invisible to structure); supply annotations.coupling."
                "semantic_coupling if you want it grouped on")
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


def _derive_coupling(spec: Dict[str, Any], code: Optional[str]) -> Dict[str, Optional[float]]:
    signals: Dict[str, float] = {}
    if code:
        signals.update(_code_signals(code))
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


def _code_signals(code: str) -> Dict[str, float]:
    """Deterministic AST signals from a solve script.

    - shared_resource_ratio: fraction of subscripted variables whose index set
      mentions a resource-like name (capacity/resource/crew/vehicle/...).
    - temporal_index_ratio: fraction of subscripted variables indexed by a
      time-like name (t/time/period/day/...).
    - cooccurrence_rate: share of constraint-ish statements referencing >= 2
      distinct subscripted variables (variables co-occur in constraints).
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {}
    resource_names = {"capacity", "resource", "crew", "vehicle", "machine",
                      "worker", "budget", "stock", "inventory"}
    temporal_names = {"t", "time", "period", "day", "week", "month", "hour",
                      "stage", "slot", "shift"}
    subscripted = 0
    with_resource = 0
    with_time = 0
    constraint_statements = 0
    multi_var_statements = 0

    def index_names(node: ast.AST) -> set:
        names = set()
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name):
                names.add(sub.id.lower())
        return names

    statements = list(ast.walk(tree))
    for node in statements:
        if isinstance(node, ast.Subscript):
            subscripted += 1
            names = index_names(node.slice)
            if names & resource_names:
                with_resource += 1
            if names & temporal_names:
                with_time += 1
        elif isinstance(node, (ast.Assign, ast.Expr)):
            vars_in_stmt = set()
            for sub in ast.walk(node):
                if isinstance(sub, ast.Subscript) and isinstance(sub.value, ast.Name):
                    vars_in_stmt.add(sub.value.id)
            if vars_in_stmt:
                constraint_statements += 1
                if len(vars_in_stmt) >= 2:
                    multi_var_statements += 1
    out: Dict[str, float] = {}
    if subscripted:
        out["shared_resource_ratio"] = with_resource / subscripted
        out["temporal_index_ratio"] = with_time / subscripted
    if constraint_statements:
        out["cooccurrence_rate"] = multi_var_statements / constraint_statements
    return out


# ---------------------------------------------------------------------------
# small deterministic helpers
# ---------------------------------------------------------------------------


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
