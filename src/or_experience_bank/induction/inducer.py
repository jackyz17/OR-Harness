"""Pattern induction: hypothesis generation (module 3.4).

This is the HYPOTHESIS step of Structure -> Relation -> Hypothesis -> Verification. Given
an AlignmentMap (the shared structural roles discovered across a heterogeneous cluster)
plus the source realizations, it generates candidate general OR principles {P1..Pn}.

CRITICAL red line: a produced principle is a HYPOTHESIS (status="hypothesis"), NOT
knowledge. It has not survived counterexample search or solver transfer validation yet.
We deliberately do NOT let the LLM "summarize the memories" — the prompt is constrained
by the already-extracted structural relation (roles + bindings), so induction is grounded
in Structure, not free text.

A cluster may yield MULTIPLE candidate principles (resource-allocation / bottleneck-first /
marginal-value ...). Minimum-Explanation preference is enforced downstream: each hypothesis
carries a complexity estimate that feeds the scoring penalty (lambda*K) in validation.py.

D18 harness principle: the framework emits the prompt + validates structure; the agent owns
the LLM.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .alignment import AlignmentMap
from .candidates import CandidateCluster


@dataclass
class PrincipleHypothesis:
    """A candidate induced principle. Status is ALWAYS 'hypothesis' until validated."""

    statement: str = ""                       # the general principle (P0)
    structural_pattern: str = ""              # the shared structure it abstracts
    roles_used: List[str] = field(default_factory=list)
    source_realization_ids: List[str] = field(default_factory=list)
    applicability_conditions: List[str] = field(default_factory=list)
    complexity: float = 0.0                   # Minimum-Explanation estimate (feeds lambda*K)
    status: str = "hypothesis"                # hypothesis -> validated | refuted (never born validated)
    hypothesis_id: str = ""

    def is_hypothesis(self) -> bool:
        return self.status == "hypothesis"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "statement": self.statement,
            "structural_pattern": self.structural_pattern,
            "roles_used": list(self.roles_used),
            "source_realization_ids": list(self.source_realization_ids),
            "applicability_conditions": list(self.applicability_conditions),
            "complexity": self.complexity,
            "status": self.status,
        }


class PatternInducer:
    """Framework-side induction rules: prompt template + parse + structural validation."""

    # -- prompt -------------------------------------------------------------

    def build_induction_prompt(self, cluster: CandidateCluster, alignment: AlignmentMap) -> str:
        binding_lines = []
        for b in alignment.bindings:
            mapping = ", ".join("{}={}".format(k, v) for k, v in b.mapping.items())
            binding_lines.append(
                "  - {fam} ({rid}): {mapping}".format(fam=b.problem_family, rid=b.realization_id, mapping=mapping)
            )
        return (
            "You are inducing GENERAL OR PRINCIPLES from a cross-problem structural analogy. "
            "The structural roles have ALREADY been aligned across heterogeneous problems; your "
            "job is to state the transferable optimization principle that the correspondence reveals. "
            "Do NOT summarize the examples — state a principle that would transfer to an UNSEEN "
            "OR problem sharing the structure.\n\n"
            "SHARED ROLES: " + ", ".join(alignment.roles) + "\n"
            "STRUCTURAL CORRESPONDENCE:\n" + "\n".join(binding_lines) + "\n"
            "CLUSTER CORE SIGNATURE: " + cluster.core_key + "\n\n"
            "Return ONLY a JSON list of 1-3 candidate principles. Each:\n"
            "{\n"
            '  "statement": "the general principle (e.g. when competing decisions share a scarce '
            'resource with marginal objective contribution, prioritize higher marginal contribution '
            'subject to feasibility and coupling constraints)",\n'
            '  "structural_pattern": "the abstract structure it captures",\n'
            '  "roles_used": ["..."],\n'
            '  "applicability_conditions": ["when ..."],\n'
            '  "complexity": 0.0-1.0  (lower = simpler/more minimal explanation)\n'
            "}\n"
            "Prefer the MINIMAL principle that explains the correspondence. Mark nothing as proven."
        )

    # -- parse + validate ---------------------------------------------------

    def parse_and_validate(
        self, raw: Any, cluster: CandidateCluster, alignment: AlignmentMap
    ) -> List[PrincipleHypothesis]:
        data = self._coerce_json(raw)
        if isinstance(data, dict):  # tolerate a single-object answer
            data = [data]
        if not isinstance(data, list):
            return []
        source_ids = [m.realization_id for m in cluster.members]
        hypotheses: List[PrincipleHypothesis] = []
        for idx, item in enumerate(data):
            if not isinstance(item, dict):
                continue
            statement = str(item.get("statement", "")).strip()
            if not statement:
                continue  # a principle must actually say something
            hypotheses.append(
                PrincipleHypothesis(
                    hypothesis_id="hyp_{}_{}".format(cluster.cluster_id, idx),
                    statement=statement,
                    structural_pattern=str(item.get("structural_pattern", "")),
                    roles_used=[str(r) for r in item.get("roles_used", alignment.roles)],
                    source_realization_ids=list(source_ids),
                    applicability_conditions=[str(c) for c in item.get("applicability_conditions", [])],
                    complexity=self._clamp(item.get("complexity", 0.5)),
                    status="hypothesis",  # enforced: never born validated
                )
            )
        return hypotheses

    @staticmethod
    def _clamp(value: Any) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return 0.5

    @staticmethod
    def _coerce_json(raw: Any) -> Optional[Any]:
        if raw is None:
            return None
        if isinstance(raw, (dict, list)):
            return raw
        text = str(raw).strip()
        # a list answer
        if "[" in text:
            candidate = text[text.find("["): text.rfind("]") + 1]
            try:
                return json.loads(candidate)
            except (ValueError, TypeError):
                pass
        if "{" in text:
            candidate = text[text.find("{"): text.rfind("}") + 1]
            try:
                return json.loads(candidate)
            except (ValueError, TypeError):
                pass
        return None


class LLMBackedInducer:
    """OPTIONAL convenience loop for standalone runs/tests (NOT used in harness mode)."""

    def __init__(self, inducer: Optional[PatternInducer] = None, llm_client: Optional[Any] = None):
        self.inducer = inducer or PatternInducer()
        self.llm = llm_client

    async def induce(
        self, cluster: CandidateCluster, alignment: AlignmentMap
    ) -> List[PrincipleHypothesis]:
        if self.llm is None:
            return []
        prompt = self.inducer.build_induction_prompt(cluster, alignment)
        try:
            raw = await self.llm.generate_object(prompt)
        except (AttributeError, TypeError, ValueError):
            return []
        return self.inducer.parse_and_validate(raw, cluster, alignment)


__all__ = ["PrincipleHypothesis", "PatternInducer", "LLMBackedInducer"]
