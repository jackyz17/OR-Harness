"""Structural encoding for the induction pipeline (module 3.2).

Batch-encodes realizations into a canonical form that candidates.py can index. The
heavy lifting (prompt template + controlled-vocabulary validation) is delegated to
modeling/signature_extractor.SignatureExtractor — this module is the OFFLINE batch
driver around it.

D18 harness principle: the framework emits the prompt and validates; the agent owns the
LLM. We therefore split encoding into two layers:
- Framework (no LLM): detect whether a record already carries a valid signature and,
  if not, build the extraction prompt + parse/validate whatever the agent returns.
- Optional LLM-backed driver: an injectable llm_client loop for standalone runs/tests.

Records that already have a valid signature are reused verbatim (encoding is idempotent
and never rewrites the store — append-only red line).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from ..core.modeling_schemas import StructuralSignature
from ..modeling.signature_extractor import SignatureExtractor


@dataclass
class EncodingResult:
    """Outcome of ensuring one realization carries a usable structural signature."""

    realization_id: str
    status: str                              # "reused" | "encoded" | "needs_llm" | "invalid"
    signature: Optional[StructuralSignature] = None
    prompt: Optional[str] = None             # populated when status == "needs_llm"
    errors: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status in ("reused", "encoded") and self.signature is not None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "realization_id": self.realization_id,
            "status": self.status,
            "signature": self.signature.to_dict() if self.signature else None,
            "errors": list(self.errors),
        }


class StructuralEncoder:
    """Framework-side batch encoder: reuse valid signatures, else prep LLM extraction."""

    def __init__(self, extractor: Optional[SignatureExtractor] = None):
        self.extractor = extractor or SignatureExtractor()

    # -- single-record framework API --------------------------------------

    def existing_signature(self, record: Mapping[str, Any]) -> Optional[StructuralSignature]:
        """Return the record's signature iff one is actually present AND validates.

        A bare {} is NOT treated as an existing signature: StructuralSignature.from_dict
        would happily synthesize schema defaults (linear/independent), which carries no
        real structural information. We require the raw signature block to exist and to
        carry at least one genuine core-dim signal (a multi-valued decision/constraint
        entry or any open feature) before we trust it.
        """
        raw = (record.get("math_scope") or {}).get("structural_signature")
        if not isinstance(raw, Mapping) or not raw:
            return None
        try:
            signature = StructuralSignature.from_dict(raw)
        except Exception:
            return None
        has_signal = bool(signature.decision or signature.constraint or signature.features)
        return signature if has_signal else None

    def model_text_of(self, record: Mapping[str, Any]) -> str:
        """Best available text to extract a signature FROM (model body, else retrieval text)."""
        method = record.get("method") or {}
        for key in ("action_template", "rationale"):
            if method.get(key):
                return str(method[key])
        return str(record.get("retrieval_text") or record.get("title") or "")

    def encode(self, record: Mapping[str, Any]) -> EncodingResult:
        """Reuse a valid signature, or emit the extraction prompt for the agent's LLM."""
        rid = record.get("experience_id", "")
        existing = self.existing_signature(record)
        if existing is not None:
            return EncodingResult(realization_id=rid, status="reused", signature=existing)
        prompt = self.extractor.build_extraction_prompt(self.model_text_of(record))
        return EncodingResult(realization_id=rid, status="needs_llm", prompt=prompt)

    def submit(self, record: Mapping[str, Any], raw: Any) -> EncodingResult:
        """Validate the agent's LLM output for a record previously marked needs_llm."""
        rid = record.get("experience_id", "")
        result = self.extractor.parse_and_validate(raw)
        if result.valid:
            return EncodingResult(realization_id=rid, status="encoded", signature=result.signature)
        return EncodingResult(realization_id=rid, status="invalid", errors=result.errors)

    # -- batch framework API ----------------------------------------------

    def encode_batch(self, records: List[Mapping[str, Any]]) -> List[EncodingResult]:
        return [self.encode(r) for r in records]

    def signatures_ready(self, records: List[Mapping[str, Any]]) -> Dict[str, StructuralSignature]:
        """Map realization_id -> signature for every record that already has a valid one."""
        ready: Dict[str, StructuralSignature] = {}
        for r in records:
            sig = self.existing_signature(r)
            if sig is not None:
                ready[r.get("experience_id", "")] = sig
        return ready


class LLMBackedEncoder:
    """OPTIONAL convenience loop for standalone runs/tests (NOT used in harness mode)."""

    def __init__(self, encoder: Optional[StructuralEncoder] = None, llm_client: Optional[Any] = None):
        self.encoder = encoder or StructuralEncoder()
        self.llm = llm_client

    async def encode(self, record: Mapping[str, Any]) -> EncodingResult:
        result = self.encoder.encode(record)
        if result.status != "needs_llm":
            return result
        if self.llm is None:
            return EncodingResult(
                realization_id=result.realization_id,
                status="invalid",
                errors=["no llm_client supplied (harness mode: agent generates)"],
            )
        errors: List[str] = []
        for _ in range(self.encoder.extractor.max_retries + 1):
            prompt = result.prompt if not errors else self.encoder.extractor.build_extraction_prompt(
                self.encoder.model_text_of(record), retry_errors=errors
            )
            try:
                raw = await self.llm.generate_object(prompt)
            except (AttributeError, TypeError, ValueError) as exc:
                return EncodingResult(
                    realization_id=result.realization_id, status="invalid",
                    errors=["llm call failed: " + str(exc)],
                )
            submitted = self.encoder.submit(record, raw)
            if submitted.ok:
                return submitted
            errors = submitted.errors
        return EncodingResult(
            realization_id=result.realization_id, status="invalid",
            errors=errors or ["exhausted retries"],
        )

    async def encode_batch(self, records: List[Mapping[str, Any]]) -> List[EncodingResult]:
        return [await self.encode(r) for r in records]


__all__ = ["EncodingResult", "StructuralEncoder", "LLMBackedEncoder"]
