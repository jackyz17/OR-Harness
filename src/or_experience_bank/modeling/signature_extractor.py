"""Structural-signature extraction under the harness architecture (module 0.6, D6/D18).

Harness principle: the FRAMEWORK holds rules (signature schema, controlled vocabulary,
validation, parse tolerance); the AGENT holds the LLM. The framework never calls an LLM
itself — it exposes prompt template + parse + validation. An OPTIONAL LLM-backed wrapper
drives the generate->validate->retry loop for standalone runs/tests.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..core.modeling_schemas import (
    CONSTRAINT_STRUCTURES,
    DECISION_STRUCTURES,
    INTERACTION_COUPLINGS,
    OBJECTIVE_STRUCTURES,
    RECOMMENDED_FEATURE_KEYS,
    SignatureValidationError,
    StructuralSignature,
)


@dataclass
class SignatureResult:
    """Outcome of parsing+validating an agent-produced signature."""

    valid: bool
    signature: Optional[StructuralSignature] = None
    errors: List[str] = field(default_factory=list)
    retry_hint: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "signature": self.signature.to_dict() if self.signature else None,
            "errors": list(self.errors),
            "retry_hint": self.retry_hint,
        }


class SignatureExtractor:
    """Framework-side signature rules: prompt template + parse + vocabulary validation.

    Holds NO llm. The harness agent generates the signature JSON; this class validates it.
    Open feature slots are NOT validated (Plan B promise, D9).
    """

    def __init__(self, max_retries: int = 2):
        self.max_retries = max(0, max_retries)

    def vocabulary_summary(self) -> str:
        return (
            "objective (one of): " + ", ".join(OBJECTIVE_STRUCTURES) + "\n"
            "decision (subset of): " + ", ".join(DECISION_STRUCTURES) + "\n"
            "constraint (subset of): " + ", ".join(CONSTRAINT_STRUCTURES) + "\n"
            "interaction (one of): " + ", ".join(INTERACTION_COUPLINGS) + "\n"
            "features (open slots; recommended keys): " + ", ".join(RECOMMENDED_FEATURE_KEYS)
        )

    def build_extraction_prompt(self, model_text: str, retry_errors: Optional[List[str]] = None) -> str:
        """Standard instruction for the agent's LLM. Pass retry_errors to fix prior invalid values."""
        retry_block = ""
        if retry_errors:
            retry_block = (
                "\n\nYour previous answer used out-of-vocabulary values:\n- "
                + "\n- ".join(retry_errors)
                + "\nCorrect ONLY these to values from the allowed vocabularies."
            )
        return (
            "Extract a structural signature from the optimization MODEL below. "
            "Return ONLY a JSON object with keys: objective, decision, constraint, "
            "interaction, features. Use EXACTLY the allowed values for the four core "
            "dimensions; the features object is open (any descriptive keys).\n\n"
            "ALLOWED VOCABULARIES:\n" + self.vocabulary_summary() + "\n\n"
            "MODEL:\n" + model_text + retry_block
        )

    def parse_and_validate(self, raw: Any) -> SignatureResult:
        """Parse agent output into a StructuralSignature, validating core-dim values."""
        data = self._coerce_json(raw)
        if data is None:
            return SignatureResult(
                valid=False,
                errors=["could not parse a JSON object from agent output"],
                retry_hint="Return a single JSON object with keys objective/decision/constraint/interaction/features.",
            )
        if not isinstance(data, dict):
            return SignatureResult(valid=False, errors=["signature JSON is not an object"])
        try:
            signature = StructuralSignature.from_dict(data)
        except SignatureValidationError as exc:
            return SignatureResult(
                valid=False,
                errors=[str(exc)],
                retry_hint="Fix the out-of-vocabulary value(s) and resubmit the JSON.",
            )
        except (TypeError, ValueError) as exc:
            return SignatureResult(valid=False, errors=["malformed signature: " + str(exc)])
        return SignatureResult(valid=True, signature=signature)

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


class LLMBackedExtractor:
    """OPTIONAL convenience loop for standalone runs/tests (NOT used in harness mode)."""

    def __init__(self, extractor: Optional[SignatureExtractor] = None, llm_client: Optional[Any] = None):
        self.extractor = extractor or SignatureExtractor()
        self.llm = llm_client

    async def extract(self, model_text: str) -> SignatureResult:
        if self.llm is None:
            return SignatureResult(valid=False, errors=["no llm_client supplied (harness mode: agent generates)"])
        errors: List[str] = []
        for attempt in range(self.extractor.max_retries + 1):
            prompt = self.extractor.build_extraction_prompt(model_text, retry_errors=errors or None)
            try:
                raw = await self.llm.generate_object(prompt)
            except (AttributeError, TypeError, ValueError) as exc:
                return SignatureResult(valid=False, errors=["llm call failed: " + str(exc)])
            result = self.extractor.parse_and_validate(raw)
            if result.valid:
                return result
            errors = result.errors
        return SignatureResult(valid=False, errors=errors, retry_hint="exhausted retries")


__all__ = ["SignatureResult", "SignatureExtractor", "LLMBackedExtractor"]
