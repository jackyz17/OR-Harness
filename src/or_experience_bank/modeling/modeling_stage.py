"""Structured modeling stage: produce a verified <model> before any solver branch runs
(Phase 1, module 1.1, D17/D18).

Flow inserted into solve() BEFORE branch creation:

    problem -> LLM generates <think>+<model> -> ModelingGate (L1 format + L2 structural,
    optional L3 semantic) -> on failure feed issues back and regenerate (<= max_rounds)
    -> a verified model text + validated StructuralSignature.

Only after the gate passes do solver branches turn the verified model into
solver-specific code. The modeling LLM call goes through the injected llm_client (the
harness agent owns the LLM; this stage only orchestrates generate -> validate -> retry).

Phase 4.1 (module 4.1): optional planning_priors (validated patterns + similar
realizations from the Modeling Bank) are injected into the prompt as priors. The LLM is
instructed to cite any pattern it actually applies with `[uses Pn]` inside <think> ; the
framework parses those citations (D18: LLM declares, framework validates) so that
utility attribution on a successful solve is precise, not a guess.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .modeling_contract import ModelingGate, SemanticValidator, parse_modeling_output
from ..core.modeling_schemas import StructuralSignature
from .signature_extractor import LLMBackedExtractor, SignatureExtractor

# Citation pattern the LLM emits in <think> : [uses E1] (and optionally [uses E1, E2]).
# All records are peers — unified citation tags (En) for both directly-solved and
# induced records. The framework parses these so utility attribution is precise.
USES_PATTERN = re.compile(r"\[uses\s+(E\d+(?:\s*,\s*E\d+)*)\]", re.IGNORECASE)


@dataclass
class ModelingStageResult:
    """Outcome of the structured-modeling stage."""

    success: bool
    think: str = ""
    model: str = ""
    signature: Optional[StructuralSignature] = None
    rounds_used: int = 0
    issues: List[Dict[str, str]] = field(default_factory=list)
    raw_output: str = ""
    cited_principle_ids: List[str] = field(default_factory=list)  # [uses En] parsed (unified)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "think": self.think,
            "model": self.model,
            "signature": self.signature.to_dict() if self.signature else None,
            "rounds_used": self.rounds_used,
            "issues": list(self.issues),
            "cited_principle_ids": list(self.cited_principle_ids),
        }


class StructuredModelingStage:
    """Runs the think->model->verify loop that must pass BEFORE multi-branch codegen."""

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        gate: Optional[ModelingGate] = None,
        extractor: Optional[SignatureExtractor] = None,
        max_rounds: int = 3,
    ):
        self.llm = llm_client
        # L3 semantic judge runs only when an llm is available; otherwise it is a no-op.
        semantic = SemanticValidator(llm_client)
        self.gate = gate or ModelingGate(semantic_validator=semantic, max_rounds=max_rounds)
        self.extractor = extractor or SignatureExtractor()
        self.max_rounds = max(1, max_rounds)

    def build_modeling_prompt(
        self,
        problem: str,
        issues: Optional[List[Dict[str, str]]] = None,
        planning_priors: Optional[Any] = None,
    ) -> str:
        """Prompt for the agent's LLM to produce <think> /<model> in GAMS-style DSL."""
        feedback = ""
        if issues:
            feedback = (
                "\n\nThe previous model FAILED verification. Fix these issues and return a "
                "corrected <model>:\n- "
                + "\n- ".join("[{}] {}: {}".format(i.get("layer"), i.get("type"), i.get("detail")) for i in issues)
            )
        priors_block = ""
        if planning_priors is not None and not planning_priors.is_empty():
            priors_block = self._format_planning_priors(planning_priors)
        return (
            "You are an OR modeling expert. Analyze the problem, then formalize it as a "
            "mathematical model using the GAMS-style blocks SETS / PARAMETERS / VARIABLES / "
            "OBJECTIVE / CONSTRAINTS with symbolic indexing (e.g. x[i,t]). Declare set members "
            "inline like 'i in Animals = {cow, sheep}'. Output EXACTLY two blocks:\n"
            "<think> your analysis</think>\n<model>the five blocks</model>\n\n"
            + priors_block
            + "PROBLEM:\n" + problem + feedback
        )

    def _format_planning_priors(self, priors: Any) -> str:
        """Format retrieved records as a compact planning-prior block (unified [En])."""
        lines = ["=== Past modeling experiences (reference) ==="]
        if priors.records:
            for index, record in enumerate(priors.records, start=1):
                method = (record.get("method") or {}).get("action_template") or record.get("title") or ""
                aspect = record.get("modeling_aspect") or ""
                lines.append("[E{index}] [{aspect}] {title}: {method}".format(
                    index=index, aspect=aspect, title=record.get("title", ""), method=method[:240]))
            lines.append(
                "If you apply an experience in your model, cite it inside <think> with "
                "[uses En]. Cite only experiences you actually used."
            )
        else:
            lines.append("(none available)")
        return "\n".join(lines) + "\n\n"

    async def run(
        self,
        problem: str,
        planning_priors: Optional[Any] = None,
    ) -> ModelingStageResult:
        """Generate + verify loop. Returns a verified model or the last failure."""
        if self.llm is None:
            return ModelingStageResult(success=False, issues=[
                {"layer": "stage", "type": "no_llm", "detail": "no llm_client supplied"}
            ])

        issues: Optional[List[Dict[str, str]]] = None
        last_report = None
        last_think = ""
        for round_number in range(1, self.max_rounds + 1):
            prompt = self.build_modeling_prompt(problem, issues, planning_priors)
            raw = await self.llm.generate_text(prompt)
            parsed_any = parse_modeling_output(raw)
            last_think = parsed_any["think"] or ""
            # Extract a signature (framework validates; agent already produced JSON
            # via llm).
            signature = await self._extract_signature(raw)
            report = await self.gate.check(problem, raw, signature)
            last_report = report
            if report.passed:
                parsed = parse_modeling_output(raw)
                return ModelingStageResult(
                    success=True,
                    think=parsed["think"] or "",
                    model=parsed["model"] or "",
                    signature=signature,
                    rounds_used=round_number,
                    raw_output=raw,
                    cited_principle_ids=self.extract_cited_principle_ids(
                        parsed["think"] or "", planning_priors
                    ),
                )
            issues = [issue.to_dict() for issue in report.issues]

        return ModelingStageResult(
            success=False,
            rounds_used=self.max_rounds,
            issues=issues or [],
            raw_output="",
            cited_principle_ids=self.extract_cited_principle_ids(
                last_think, planning_priors
            ),
        )

    @staticmethod
    def extract_cited_principle_ids(
        think_text: str,
        planning_priors: Optional[Any],
    ) -> List[str]:
        """Parse the LLM's [uses En] citations out of <think> (LLM declares, framework validates).

        Unmatched or unknown tags are ignored; only tags present in the injected priors map
        back to experience ids, so the LLM cannot invent a citation.
        """
        if not think_text or planning_priors is None:
            return []
        labels = planning_priors.labels or {}
        cited: List[str] = []
        for match in USES_PATTERN.finditer(think_text):
            for tag in re.split(r"\s*,\s*", match.group(1)):
                tag = tag.upper()
                experience_id = labels.get(tag)
                if experience_id and experience_id not in cited:
                    cited.append(experience_id)
        return cited

    async def _extract_signature(
        self, raw_modeling_output: str
    ) -> Optional[StructuralSignature]:
        """Ask the agent's LLM for a signature JSON, then framework-validate it."""
        parsed = parse_modeling_output(raw_modeling_output)
        if not parsed["model"]:
            return None
        backed = LLMBackedExtractor(self.extractor, self.llm)
        result = await backed.extract(parsed["model"])
        return result.signature if result.valid else None


__all__ = ["ModelingStageResult", "StructuredModelingStage"]
