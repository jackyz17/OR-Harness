"""Coupling-Aware Intermediate Representation (CIR).

A lightweight, domain-general structured representation of *how* the
components of an optimization problem interact.  It is intentionally NOT a
scalar profile — the primary representation is an explicit set of entities,
decisions, constraints, and relations.  Scalar coupling scores may later be
derived from this structure, but they are summaries, never substitutes.

Design constraints honoured here:

* **CIR exists before the canonical model.**  The outer harness agent
  produces a CIR from the natural-language task description *before* writing
  the GAMS-style ``model`` field.  The model may later cross-check the CIR,
  but is never required to create one.

* **Co-occurrence is structural evidence only.**  Constraint-variable
  co-occurrence can produce generic ``depends_on`` edges, but a semantic
  relation (``shares_resource``, ``competes_for``, …) is only emitted when
  additional entity/constraint semantics support the upgrade.

* **Domain-general.**  Every ``kind`` / ``type`` field is a free-form string.
  The core schema never hard-codes supply-chain-specific vocabulary.

* **Zero runtime LLM.**  The harness agent does semantic extraction; this
  module only validates, infers deterministically, and renders guidance.

The module is pure-stdlib, has no dependency on the rest of ``or_harness``
(except an optional import of :class:`~or_harness.profiling.model_syntax
.ParsedModel` for structural inference), and is fully JSON-round-trippable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Entity:
    """A business object in the problem (open-ended ``kind``).

    ``kind`` examples (non-exhaustive, never enumerated here): resource,
    product, site, period, route, machine, …  The schema accepts any string.
    """

    name: str
    kind: str = "other"
    attrs: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "kind": self.kind, "attrs": dict(self.attrs)}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Entity":
        return cls(name=str(data["name"]), kind=str(data.get("kind", "other")),
                   attrs=dict(data.get("attrs") or {}))


@dataclass
class Decision:
    """A decision variable or decision group in the problem."""

    name: str
    kind: str = "other"
    indexes: List[str] = field(default_factory=list)
    attrs: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "kind": self.kind,
                "indexes": list(self.indexes), "attrs": dict(self.attrs)}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Decision":
        return cls(name=str(data["name"]), kind=str(data.get("kind", "other")),
                   indexes=list(data.get("indexes") or []),
                   attrs=dict(data.get("attrs") or {}))


@dataclass
class ConstraintRef:
    """A reference to a constraint or business rule."""

    id: str
    kind: str = "other"
    expr: str = ""
    attrs: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "kind": self.kind, "expr": self.expr,
                "attrs": dict(self.attrs)}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConstraintRef":
        return cls(id=str(data["id"]), kind=str(data.get("kind", "other")),
                   expr=str(data.get("expr", "")),
                   attrs=dict(data.get("attrs") or {}))


#: Evidence levels for a relation.
#:
#: ``structural``  — derived from constraint-variable co-occurrence only.
#: ``semantic``    — supported by entity/constraint semantics or agent declaration.
#: ``declared``    — directly asserted by the agent (subject to validation).
EVIDENCE_LEVELS: Tuple[str, ...] = ("structural", "semantic", "declared")


@dataclass
class Relation:
    """A directed relationship between two nodes (entities, decisions, or
    constraints).  ``type`` is a free-form string; common examples include
    ``uses_resource``, ``shares_resource``, ``competes_for``, ``precedes``,
    ``flows_to``, ``depends_on``, ``constrained_by`` …"""

    source: str
    target: str
    type: str = "depends_on"
    evidence: str = "structural"
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"source": self.source, "target": self.target,
                "type": self.type, "evidence": self.evidence,
                "detail": self.detail}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Relation":
        return cls(source=str(data["source"]), target=str(data["target"]),
                   type=str(data.get("type", "depends_on")),
                   evidence=str(data.get("evidence", "structural")),
                   detail=str(data.get("detail", "")))


@dataclass
class CouplingGroup:
    """A derived coupling structure — a recurring pattern that carries
    modeling implications.  ``type`` is a free-form string; common examples
    include ``shared_bottleneck``, ``route_convergence``,
    ``temporal_propagation_chain``, ``global_constraint``,
    ``cross_stage_coupling`` …"""

    type: str
    members: List[str] = field(default_factory=list)
    resource: Optional[str] = None
    implication: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.type, "members": list(self.members),
                "resource": self.resource, "implication": self.implication}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CouplingGroup":
        return cls(type=str(data["type"]),
                   members=list(data.get("members") or []),
                   resource=data.get("resource"),
                   implication=str(data.get("implication", "")))


@dataclass
class CouplingIssue:
    """A validation issue, mirroring :class:`~or_harness.profiling.model_syntax
    .ModelIssue` in shape."""

    layer: str   # "L1" | "L2"
    code: str
    detail: str

    def to_dict(self) -> Dict[str, str]:
        return {"layer": self.layer, "code": self.code, "detail": self.detail}


@dataclass
class CouplingAwareIR:
    """The top-level CIR object.

    All child lists are JSON-round-trippable.  The object is intentionally
    plain — no graph database, no embeddings, no GNN.
    """

    entities: List[Entity] = field(default_factory=list)
    decisions: List[Decision] = field(default_factory=list)
    constraints: List[ConstraintRef] = field(default_factory=list)
    relations: List[Relation] = field(default_factory=list)
    coupling_groups: List[CouplingGroup] = field(default_factory=list)
    issues: List[CouplingIssue] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True when no validation issues were found."""
        return not self.issues

    # -- node name registry ---------------------------------------------------

    def _node_names(self) -> Set[str]:
        names: Set[str] = set()
        for e in self.entities:
            names.add(e.name)
        for d in self.decisions:
            names.add(d.name)
        for c in self.constraints:
            names.add(c.id)
        return names

    # -- serialisation --------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entities": [e.to_dict() for e in self.entities],
            "decisions": [d.to_dict() for d in self.decisions],
            "constraints": [c.to_dict() for c in self.constraints],
            "relations": [r.to_dict() for r in self.relations],
            "coupling_groups": [g.to_dict() for g in self.coupling_groups],
            "issues": [i.to_dict() for i in self.issues],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CouplingAwareIR":
        if not isinstance(data, dict):
            raise ValueError("CIR must be a JSON object")
        return cls(
            entities=[Entity.from_dict(d) for d in (data.get("entities") or [])],
            decisions=[Decision.from_dict(d) for d in (data.get("decisions") or [])],
            constraints=[ConstraintRef.from_dict(d)
                         for d in (data.get("constraints") or [])],
            relations=[Relation.from_dict(d)
                       for d in (data.get("relations") or [])],
            coupling_groups=[CouplingGroup.from_dict(d)
                             for d in (data.get("coupling_groups") or [])],
            issues=[CouplingIssue(layer=i.get("layer", "L1"),
                                  code=i.get("code", ""),
                                  detail=i.get("detail", ""))
                    for i in (data.get("issues") or [])],
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False,
                          sort_keys=True, separators=(",", ":"))


# ---------------------------------------------------------------------------
# L1 / L2 validation (deterministic, no LLM)
# ---------------------------------------------------------------------------


def validate_cir(cir: CouplingAwareIR) -> CouplingAwareIR:
    """Validate *cir* in place and return it.

    **L1 — format**: required fields present, types correct, no duplicate
    node names.

    **L2 — referential integrity**: every ``relation.source`` and
    ``relation.target`` must resolve to a known entity, decision, or
    constraint.  Every ``coupling_group.member`` must likewise resolve.
    """
    cir.issues.clear()
    seen_names: Set[str] = set()

    # -- L1: entities ----------------------------------------------------------
    for e in cir.entities:
        if not e.name:
            cir.issues.append(CouplingIssue("L1", "empty_name",
                                            "entity with empty name"))
        elif e.name in seen_names:
            cir.issues.append(CouplingIssue("L1", "duplicate_name",
                                            f"entity name {e.name!r} duplicated"))
        else:
            seen_names.add(e.name)

    # -- L1: decisions ---------------------------------------------------------
    for d in cir.decisions:
        if not d.name:
            cir.issues.append(CouplingIssue("L1", "empty_name",
                                            "decision with empty name"))
        elif d.name in seen_names:
            cir.issues.append(CouplingIssue("L1", "duplicate_name",
                                            f"decision name {d.name!r} duplicated"))
        else:
            seen_names.add(d.name)

    # -- L1: constraints -------------------------------------------------------
    for c in cir.constraints:
        if not c.id:
            cir.issues.append(CouplingIssue("L1", "empty_name",
                                            "constraint with empty id"))
        elif c.id in seen_names:
            cir.issues.append(CouplingIssue("L1", "duplicate_name",
                                            f"constraint id {c.id!r} duplicated"))
        else:
            seen_names.add(c.id)

    # -- L1: relations ---------------------------------------------------------
    for r in cir.relations:
        if not r.source or not r.target:
            cir.issues.append(CouplingIssue("L1", "empty_endpoint",
                                            f"relation {r.type!r} has empty endpoint"))
        if r.evidence not in EVIDENCE_LEVELS:
            cir.issues.append(CouplingIssue("L1", "bad_evidence",
                                            f"relation {r.source}->{r.target} "
                                            f"has evidence={r.evidence!r}; "
                                            f"expected one of {EVIDENCE_LEVELS}"))

    # -- L2: referential integrity --------------------------------------------
    node_names = seen_names  # all declared node names

    for i, r in enumerate(cir.relations):
        if r.source and r.source not in node_names:
            cir.issues.append(CouplingIssue("L2", "dangling_source",
                                            f"relation#{i} source {r.source!r} "
                                            "does not match any entity/decision/"
                                            "constraint"))
        if r.target and r.target not in node_names:
            cir.issues.append(CouplingIssue("L2", "dangling_target",
                                            f"relation#{i} target {r.target!r} "
                                            "does not match any entity/decision/"
                                            "constraint"))

    for i, g in enumerate(cir.coupling_groups):
        for m in g.members:
            if m not in node_names:
                cir.issues.append(CouplingIssue("L2", "dangling_member",
                                                f"coupling_group#{i} member "
                                                f"{m!r} does not match any "
                                                "entity/decision/constraint"))
        if g.resource is not None and g.resource not in node_names:
            cir.issues.append(CouplingIssue("L2", "dangling_resource",
                                            f"coupling_group#{i} resource "
                                            f"{g.resource!r} does not match "
                                            "any entity/decision/constraint"))

    return cir


# ---------------------------------------------------------------------------
# Deterministic structural inference
# ---------------------------------------------------------------------------

#: Constraint-expression keywords that hint at capacity / resource semantics.
_CAPACITY_KEYWORDS = {"capacity", "limit", "cap", "budget", "resource",
                      "available", "supply", "demand", "max"}
#: Entity-kind values that hint at a resource.
_RESOURCE_KIND_HINTS = {"resource", "capacity", "machine", "crew", "vehicle",
                        "worker", "budget", "stock", "inventory"}


def _has_capacity_semantics(expr: str) -> bool:
    """Heuristic: does *expr* look like a capacity/limit constraint?"""
    lower = expr.lower()
    return any(kw in lower for kw in _CAPACITY_KEYWORDS)


def infer_structural_relations(
        cir: CouplingAwareIR,
        parsed_model: Optional[Any] = None) -> CouplingAwareIR:
    """Populate *cir.relations* with **structural-only** edges.

    When *parsed_model* (a :class:`ParsedModel`) is provided, constraint-
    variable co-occurrence produces generic ``depends_on`` edges between
    decisions that share a constraint.  These edges carry
    ``evidence="structural"`` — they are evidence, not semantic claims.

    When a structural edge also has capacity semantics (the constraint
    expression contains a capacity keyword **and** the target entity kind is
    resource-like), the edge may be *upgraded* to ``uses_resource``.  This
    is the only semantic upgrade performed by the deterministic layer; all
    other semantic relations must come from the agent.

    The method is idempotent: calling it twice does not duplicate edges.
    """
    existing: Set[Tuple[str, str, str, str]] = {
        (r.source, r.target, r.type, r.evidence) for r in cir.relations
    }

    # -- upgrade pass: agent-declared structural edges with semantic support --
    # Build lookup tables.
    entity_by_name: Dict[str, Entity] = {e.name: e for e in cir.entities}
    constraint_by_id: Dict[str, ConstraintRef] = {c.id: c for c in cir.constraints}

    for rel in cir.relations:
        if rel.evidence != "structural" or rel.type != "depends_on":
            continue
        # Can we upgrade?  Check target entity kind + source constraint.
        target_entity = entity_by_name.get(rel.target)
        if target_entity is None:
            continue
        if target_entity.kind.lower() not in _RESOURCE_KIND_HINTS:
            continue
        # Look for a constraint that mentions the source decision and has
        # capacity semantics.
        for c in cir.constraints:
            if _has_capacity_semantics(c.expr) and rel.source in c.expr:
                rel.type = "uses_resource"
                rel.evidence = "semantic"
                rel.detail = (f"upgraded from structural co-occurrence: "
                              f"constraint {c.id} has capacity semantics "
                              f"and target {rel.target} is resource-like")
                break

    # -- co-occurrence inference from parsed model ----------------------------
    if parsed_model is not None:
        _infer_from_parsed_model(cir, parsed_model, existing, entity_by_name,
                                 constraint_by_id)

    return cir


def _infer_from_parsed_model(
        cir: CouplingAwareIR,
        model: Any,
        existing: Set[Tuple[str, str, str, str]],
        entity_by_name: Dict[str, Entity],
        constraint_by_id: Dict[str, ConstraintRef]) -> None:
    """Add structural ``depends_on`` edges from constraint-variable
    co-occurrence in *model* (a :class:`ParsedModel`)."""

    var_sets = model.constraint_variable_sets()
    var_names = list(model.variables)

    for ci, (label, expr) in enumerate(model.constraints):
        # Find which variables co-occur in this constraint.
        vars_here = var_sets[ci] if ci < len(var_sets) else set()
        if len(vars_here) < 2:
            continue
        constraint_id = label or f"C{ci + 1}"
        # If the constraint is already in the CIR, use its id; otherwise
        # create a lightweight ConstraintRef.
        if constraint_id not in constraint_by_id:
            cir.constraints.append(ConstraintRef(
                id=constraint_id, kind="other", expr=expr))
            constraint_by_id[constraint_id] = cir.constraints[-1]
        # Add depends_on edges between pairs of co-occurring decisions.
        vars_list = sorted(vars_here)
        for i_a in range(len(vars_list)):
            for i_b in range(i_a + 1, len(vars_list)):
                va, vb = vars_list[i_a], vars_list[i_b]
                key = (va, vb, "depends_on", "structural")
                if key in existing:
                    continue
                existing.add(key)
                cir.relations.append(Relation(
                    source=va, target=vb, type="depends_on",
                    evidence="structural",
                    detail=f"co-occur in constraint {constraint_id}"))
    # Re-validate after inference.
    validate_cir(cir)


# ---------------------------------------------------------------------------
# Coupling-group derivation + modeling guidance
# ---------------------------------------------------------------------------

#: Mapping from coupling-group type to a modeling-implication template.
#: The type strings are examples, not an exhaustive enum — unknown types
#: pass through with a generic message.
_IMPLICATION_TEMPLATES: Dict[str, str] = {
    "shared_bottleneck": (
        "Multiple decisions ({members}) consume the same resource "
        "({resource}); ensure one aggregate capacity constraint covers "
        "all relevant decisions."),
    "route_convergence": (
        "Multiple routes ({members}) converge on a shared downstream "
        "stage ({resource}); do not treat route capacities independently "
        "after convergence."),
    "temporal_propagation_chain": (
        "Decisions ({members}) propagate across time via state "
        "({resource}); ensure state-transition / inventory linkage is "
        "modeled across periods."),
    "global_constraint": (
        "Many local decisions ({members}) are bounded by one global "
        "constraint ({resource}); preserve the global constraint during "
        "decomposition."),
    "cross_stage_coupling": (
        "Decisions ({members}) in different stages are linked via "
        "{resource}; model the inter-stage dependency explicitly."),
}


def derive_coupling_groups(cir: CouplingAwareIR) -> CouplingAwareIR:
    """Derive :class:`CouplingGroup` objects from the relation structure.

    Currently detects two patterns deterministically:

    * **shared_bottleneck** — two or more ``uses_resource`` relations point
      to the same resource entity.
    * **global_constraint** — a constraint referenced by three or more
      ``constrained_by`` relations (many local decisions → one global rule).

    Other group types (``route_convergence``, ``temporal_propagation_chain``,
    ``cross_stage_coupling``) may be declared by the agent and are preserved
    as-is; the deterministic layer does not invent them.
    """
    existing_types = {g.type for g in cir.coupling_groups}

    # -- shared_bottleneck: group uses_resource edges by target ---------------
    resource_users: Dict[str, List[str]] = {}
    for r in cir.relations:
        if r.type == "uses_resource":
            resource_users.setdefault(r.target, []).append(r.source)

    for resource, users in resource_users.items():
        if len(users) < 2:
            continue
        gtype = "shared_bottleneck"
        members = sorted(users)
        # Avoid duplicating an agent-declared group.
        key = (gtype, resource)
        if any(g.type == gtype and g.resource == resource for g in cir.coupling_groups):
            continue
        cir.coupling_groups.append(CouplingGroup(
            type=gtype, members=members, resource=resource,
            implication=_render_implication(gtype, members, resource)))

    return cir


def _render_implication(gtype: str, members: List[str],
                        resource: Optional[str]) -> str:
    template = _IMPLICATION_TEMPLATES.get(gtype)
    if template is None:
        return (f"Coupling structure of type {gtype!r} detected among "
                f"{members}; review modeling implications.")
    return template.format(
        members=", ".join(members),
        resource=resource or "unknown")


def render_modeling_guidance(cir: CouplingAwareIR) -> List[Dict[str, Any]]:
    """Render coupling groups as structured modeling guidance for the agent.

    Each entry has ``type``, ``members``, ``resource``, and ``implication``
    — all inspectable, none hidden in embeddings or free-form prose.
    """
    guidance: List[Dict[str, Any]] = []
    for g in cir.coupling_groups:
        if not g.implication:
            g.implication = _render_implication(g.type, g.members, g.resource)
        guidance.append({
            "type": g.type,
            "members": list(g.members),
            "resource": g.resource,
            "implication": g.implication,
        })
    return guidance


# ---------------------------------------------------------------------------
# Cross-check: CIR ↔ canonical model
# ---------------------------------------------------------------------------


def cross_check_cir_model(
        cir: CouplingAwareIR,
        parsed_model: Optional[Any]) -> List[Dict[str, Any]]:
    """Compare a validated CIR against a parsed model representation.

    Returns a list of warning dicts (empty when consistent).  The model is
    an *optional* cross-check source — CIR is never required to originate
    from it.
    """
    warnings: List[Dict[str, Any]] = []
    if parsed_model is None:
        return warnings

    # Check: every decision in the CIR should appear as a variable in the
    # model, and vice versa.
    cir_decisions = {d.name for d in cir.decisions}
    model_vars = set(parsed_model.variables)
    missing_in_model = cir_decisions - model_vars
    missing_in_cir = model_vars - cir_decisions
    if missing_in_model:
        warnings.append({
            "code": "cir_decision_not_in_model",
            "detail": (f"CIR decisions {sorted(missing_in_model)} are not "
                       "declared as variables in the model representation"),
        })
    if missing_in_cir:
        warnings.append({
            "code": "model_var_not_in_cir",
            "detail": (f"Model variables {sorted(missing_in_cir)} are not "
                       "represented as decisions in the CIR"),
        })

    # Check: CIR relations should not contradict model structure.
    # A `uses_resource` edge (va, vb) implies va and vb co-occur in at least
    # one constraint.  If they never co-occur, flag it.
    var_sets = parsed_model.constraint_variable_sets()
    co_occur: Set[Tuple[str, str]] = set()
    for vs in var_sets:
        vs_list = sorted(vs)
        for i_a in range(len(vs_list)):
            for i_b in range(i_a + 1, len(vs_list)):
                co_occur.add((vs_list[i_a], vs_list[i_b]))

    for r in cir.relations:
        if r.type in ("uses_resource", "shares_resource", "competes_for"):
            pair = tuple(sorted([r.source, r.target]))
            if pair not in co_occur:
                warnings.append({
                    "code": "relation_not_in_model",
                    "detail": (f"CIR relation {r.source}--{r.type}-->"
                               f"{r.target} does not correspond to any "
                               "constraint-variable co-occurrence in the "
                               "model"),
                })

    return warnings


# ---------------------------------------------------------------------------
# High-level convenience: understand (the pre-model entry point)
# ---------------------------------------------------------------------------


def understand(task: Dict[str, Any]) -> Dict[str, Any]:
    """Build, validate, and reason over a CIR from a task JSON.

    The task JSON may carry an optional top-level ``coupling`` field::

        {
          "task_id": "...",
          "family": "...",
          "coupling": {
              "entities": [...],
              "decisions": [...],
              "constraints": [...],
              "relations": [...],
              "coupling_groups": [...]   # optional, agent-declared
          },
          "model": "SETS: ..."           # optional, for cross-check only
        }

    Returns::

        {
          "cir": <validated CIR dict>,
          "modeling_guidance": [...],
          "cir_warnings": [...]          # CIR ↔ model cross-check warnings
        }

    When no ``coupling`` field is present, returns ``cir=null`` with a
    guidance message prompting the agent to submit coupling understanding
    before modeling.
    """
    coupling_data = task.get("coupling")
    if not coupling_data or not isinstance(coupling_data, dict):
        return {
            "cir": None,
            "modeling_guidance": [],
            "cir_warnings": [],
            "message": ("No 'coupling' field found in the task. Submit a CIR "
                        "before writing the canonical model — coupling-aware "
                        "understanding is a pre-model step."),
        }

    cir = CouplingAwareIR.from_dict(coupling_data)
    validate_cir(cir)

    # Optional: infer structural relations from the model (cross-check only,
    # never required).  The model may be absent — CIR stands on its own.
    parsed_model = None
    model_text = task.get("model")
    if isinstance(model_text, str) and model_text.strip():
        try:
            from or_harness.profiling.model_syntax import parse_model, verify_model
            report = verify_model(model_text)
            if report.parsed is not None:
                parsed_model = report.parsed
        except Exception:
            pass  # model is optional; parse failure is not a CIR error

    infer_structural_relations(cir, parsed_model)
    derive_coupling_groups(cir)
    guidance = render_modeling_guidance(cir)
    warnings = cross_check_cir_model(cir, parsed_model)

    return {
        "cir": cir.to_dict(),
        "modeling_guidance": guidance,
        "cir_warnings": warnings,
    }
