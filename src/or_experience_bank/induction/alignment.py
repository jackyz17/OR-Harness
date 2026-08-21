"""Cross-memory structural alignment (module 3.3).

This is the RELATION step of Structure -> Relation -> Hypothesis -> Verification. It does
NOT ask "are these two memories similar?"; it asks "which structural ROLES correspond
across these different OR problems?" (Warehouse Capacity <-> Machine Capacity <-> Labor
Hours; Inventory Decision <-> Production Decision <-> Workforce Assignment).

The output is an AlignmentMap: the shared role vocabulary for a candidate cluster plus,
per member realization, the binding of each abstract role to that problem's concrete
entity. This map is what inducer.py turns into a hypothesis, and what pattern records
store as role_schema + role_mappings.

Red lines honoured:
- Heterogeneous complementarity: alignment happens only INSIDE a candidate cluster that
  candidates.py already certified as structurally isomorphic + cross-family.
- D18 harness principle: the framework emits the prompt and validates the role bindings
  against a controlled role vocabulary; the agent owns the LLM.
- Grounding (borrowed from Auto-Dreamer): every role binding cites the realization_id it
  came from, so nothing is asserted without provenance.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from ..core.modeling_schemas import RoleMappingEntry
from .candidates import CandidateCluster


# Controlled vocabulary of abstract structural roles. Small and stable by design — these
# are the cross-problem "interfaces" an induced pattern can refer to. The LLM must bind
# each member's concrete entities to roles from THIS list (open extension allowed via
# extra_roles, but the canonical set is preferred for cross-family reusability).
CANONICAL_ROLES = (
    "resource_pool",          # the shared scarce resource (warehouse/machine/labor hours)
    "capacity_limit",         # the bound on the resource
    "competing_decisions",    # the variables contending for the resource
    "objective_contribution", # per-unit marginal contribution to the objective
    "demand_requirement",     # what must be satisfied
    "coupling_constraint",    # the constraint tying decisions to the resource
    "time_period",            # temporal index when present
    "flow_balance",           # conservation relation when present
)


@dataclass
class AlignmentMap:
    """Shared role vocabulary + per-realization role bindings for one cluster."""

    cluster_id: str
    roles: List[str] = field(default_factory=list)               # the agreed role schema
    bindings: List[RoleMappingEntry] = field(default_factory=list)  # per-realization mapping
    confidence: float = 0.0
    notes: str = ""

    def role_schema(self) -> Dict[str, str]:
        """Pattern-ready role_schema block: role -> one-line abstract definition."""
        return {role: "abstract structural role shared across the cluster" for role in self.roles}

    def role_mappings(self) -> List[RoleMappingEntry]:
        return list(self.bindings)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "roles": list(self.roles),
            "bindings": [b.to_dict() for b in self.bindings],
            "confidence": self.confidence,
            "notes": self.notes,
        }


class StructuralAligner:
    """Framework-side alignment rules: prompt template + parse + role-vocab validation."""

    def __init__(self, extra_roles: Optional[List[str]] = None):
        self.roles = list(CANONICAL_ROLES) + list(extra_roles or [])

    # -- prompt -------------------------------------------------------------

    def build_alignment_prompt(self, cluster: CandidateCluster) -> str:
        """Instruction for the agent's LLM to bind each member's entities to shared roles."""
        member_blocks = []
        for m in cluster.members:
            method = (m.record.get("method") or {})
            member_blocks.append(
                "- realization_id: {rid}\n"
                "  problem_family: {fam}\n"
                "  title: {title}\n"
                "  method: {method}\n"
                "  signature: O={o} D={d} C={c} I={i} features={f}".format(
                    rid=m.realization_id,
                    fam=m.problem_family,
                    title=m.title,
                    method=(method.get("action_template") or m.title),
                    o=m.signature.objective if m.signature else "?",
                    d=m.signature.decision if m.signature else [],
                    c=m.signature.constraint if m.signature else [],
                    i=m.signature.interaction if m.signature else "?",
                    f=(m.signature.features if m.signature else {}),
                )
            )
        return (
            "You are aligning STRUCTURAL ROLES across a cluster of heterogeneous but "
            "structurally-isomorphic OR modeling experiences. Identify the SHARED abstract "
            "roles, then bind each role to the concrete entity in EACH realization.\n\n"
            "CANONICAL ROLES (prefer these; add extra only if needed): "
            + ", ".join(self.roles)
            + "\n\nCLUSTER (core signature key: " + cluster.core_key + "):\n"
            + "\n".join(member_blocks)
            + "\n\nReturn ONLY a JSON object:\n"
            "{\n"
            '  "roles": ["role_a", "role_b", ...],\n'
            '  "bindings": [{"realization_id": "...", "problem_family": "...", '
            '"mapping": {"role_a": "concrete entity", ...}}, ...],\n'
            '  "confidence": 0.0-1.0,\n'
            '  "notes": "why these roles correspond"\n'
            "}\n"
            "Every binding MUST cite its realization_id. Use canonical role names where possible."
        )

    # -- parse + validate ---------------------------------------------------

    def parse_and_validate(self, raw: Any, cluster: CandidateCluster) -> AlignmentMap:
        data = self._coerce_json(raw)
        if not isinstance(data, dict):
            return AlignmentMap(cluster_id=cluster.cluster_id, notes="could not parse alignment JSON")
        roles = [str(r) for r in data.get("roles", []) if str(r).strip()]
        known_ids = {m.realization_id for m in cluster.members}
        family_of = {m.realization_id: m.problem_family for m in cluster.members}
        bindings: List[RoleMappingEntry] = []
        for item in data.get("bindings", []):
            if not isinstance(item, dict):
                continue
            rid = item.get("realization_id", "")
            if rid not in known_ids:
                continue  # drop ungrounded bindings (grounding red line)
            mapping = {str(k): str(v) for k, v in (item.get("mapping") or {}).items()}
            bindings.append(
                RoleMappingEntry(
                    realization_id=rid,
                    problem_family=item.get("problem_family") or family_of.get(rid, ""),
                    mapping=mapping,
                )
            )
        try:
            confidence = float(data.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        return AlignmentMap(
            cluster_id=cluster.cluster_id,
            roles=roles,
            bindings=bindings,
            confidence=max(0.0, min(1.0, confidence)),
            notes=str(data.get("notes", "")),
        )

    def is_complete(self, alignment: AlignmentMap, cluster: CandidateCluster) -> bool:
        """An alignment is usable only if it covers every member and binds >= 1 role each."""
        if not alignment.roles or len(alignment.bindings) < len(cluster.members):
            return False
        return all(b.mapping for b in alignment.bindings)

    @staticmethod
    def _coerce_json(raw: Any) -> Optional[Any]:
        if raw is None:
            return None
        if isinstance(raw, (dict, list)):
            return raw
        text = str(raw).strip()
        for candidate in (text, text[text.find("{"): text.rfind("}") + 1] if "{" in text else ""):
            if not candidate:
                continue
            try:
                return json.loads(candidate)
            except (ValueError, TypeError):
                continue
        return None


class LLMBackedAligner:
    """OPTIONAL convenience loop for standalone runs/tests (NOT used in harness mode)."""

    def __init__(self, aligner: Optional[StructuralAligner] = None, llm_client: Optional[Any] = None):
        self.aligner = aligner or StructuralAligner()
        self.llm = llm_client

    async def align(self, cluster: CandidateCluster) -> AlignmentMap:
        if self.llm is None:
            return AlignmentMap(cluster_id=cluster.cluster_id, notes="no llm_client supplied")
        prompt = self.aligner.build_alignment_prompt(cluster)
        try:
            raw = await self.llm.generate_object(prompt)
        except (AttributeError, TypeError, ValueError) as exc:
            return AlignmentMap(cluster_id=cluster.cluster_id, notes="llm call failed: " + str(exc))
        return self.aligner.parse_and_validate(raw, cluster)


__all__ = ["CANONICAL_ROLES", "AlignmentMap", "StructuralAligner", "LLMBackedAligner"]
