"""Deterministic problem profiling.

Same input, same output — no randomness, no time dependence. Coupling
dimensions come from two channels:

1. harness-supplied: the outer agent already understands the problem and
   provides coupling values via the task file's ``annotations.coupling`` (or
   top-level coupling keys). The profile is then marked
   ``source: harness_supplied``.
2. derived: the profiler estimates coupling deterministically from the
   structured spec (explicit ``coupling`` hints are NOT required) and,
   optionally, from the solve script's AST (variable co-occurrence, shared
   resource-variable ratios, temporal index structure).

No NLP subsystem, no semantic graph. These four scalars are the quantitative
analogue of the legacy alignment layer's canonical roles.
"""

from __future__ import annotations

import ast
from typing import Any, Dict, Optional

from or_harness.core.schema import COUPLING_FEATURES, ProblemProfile, SCALE_FEATURES


def profile_task(task: Dict[str, Any], code: Optional[str] = None) -> ProblemProfile:
    """Build a ProblemProfile from a task JSON document.

    Task schema::

        {
          "task_id": "...",            # required
          "family": "...",             # required
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
    """
    if not isinstance(task, dict):
        raise ValueError("task must be a JSON object")
    task_id = task.get("task_id")
    family = task.get("family")
    if not task_id or not family:
        raise ValueError("task requires 'task_id' and 'family'")
    spec = dict(task.get("spec") or {})
    annotations = dict(task.get("annotations") or {})

    supplied = _supplied_coupling(task, annotations)
    if supplied:
        coupling = {f: _clamp01(supplied.get(f)) for f in COUPLING_FEATURES}
        source = "harness_supplied"
    else:
        coupling = _derive_coupling(spec, code)
        source = "derived"

    scale = {f: float(spec[f]) for f in SCALE_FEATURES if f in spec}
    risk = dict(spec.get("risk_features") or {})
    return ProblemProfile(
        problem_id=str(task_id), family=str(family), scale_features=scale,
        semantic_coupling=coupling["semantic_coupling"],
        resource_coupling=coupling["resource_coupling"],
        temporal_coupling=coupling["temporal_coupling"],
        route_complexity=coupling["route_complexity"],
        risk_features=risk, source=source, annotations=annotations)


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
# derived channel
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
    semantic = _semantic_from_spec(spec, resource, temporal, route)
    return {
        "semantic_coupling": semantic,
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


def _semantic_from_spec(spec: Dict[str, Any], *components: Optional[float]) -> Optional[float]:
    """Semantic coupling proxy: entity/interaction richness blended with the
    other coupling dimensions (highly coupled problems are semantically rich)."""
    entities = spec.get("entities")
    entity_richness = (min(1.0, len(entities) / 10.0)
                       if isinstance(entities, list) and entities else None)
    parts = [c for c in (entity_richness, *components) if c is not None]
    return (sum(parts) / len(parts)) if parts else None


def _combine(values) -> Optional[float]:
    present = [v for v in values if v is not None]
    if not present:
        return None
    return round(sum(present) / len(present), 6)
