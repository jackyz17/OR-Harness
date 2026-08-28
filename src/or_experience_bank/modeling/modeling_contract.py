"""Three-stage modeling contract with verification moved BEFORE code generation
(module 0.1, Decision D17).

Flow:  problem -> <think> -> <model> -> [verify model] -> multi-branch <python>
                                          | failed
                                          +--> back to think/model (repair loop <=3)

The <model> body uses a GAMS-style lightweight DSL (Option A): it borrows GAMS's
SETS/PARAMETERS/VARIABLES/OBJECTIVE/CONSTRAINTS block discipline plus symbolic
indexing ``x[i,t]``, but carries no execution semantics and does not depend on a
commercial GAMS interpreter. The syntax is defined behind a pluggable ``ModelSyntax``
interface so a full GAMS grammar (Option B) can be swapped in later without changing
the gate/flow above it.

Three verification layers, ALL without executing code:
  L1 FormatValidator      tags + five blocks present (deterministic)
  L2 StructuralValidator  symbol cross-reference (declared vs referenced) + signature
                          dimension derivation vs extracted signature (deterministic)
  L3 SemanticValidator    LLM-as-a-Judge for missing/spurious constraints (returns
                          structured issues); pluggable, no-op without an LLM client
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol


# ---------------------------------------------------------------------------
# ModelSyntax: pluggable grammar interface (Option A now, Option B = full GAMS later)
# ---------------------------------------------------------------------------

@dataclass
class ParsedModel:
    """A parsed <model> body: five required GAMS blocks + optional AUXILIARY block."""

    sets: List[str] = field(default_factory=list)
    parameters: List[str] = field(default_factory=list)
    variables: List[str] = field(default_factory=list)
    objective: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    auxiliary: List[str] = field(default_factory=list)

    def blocks(self) -> Dict[str, List[str]]:
        return {
            "SETS": self.sets,
            "PARAMETERS": self.parameters,
            "VARIABLES": self.variables,
            "OBJECTIVE": self.objective,
            "CONSTRAINTS": self.constraints,
            "AUXILIARY": self.auxiliary,
        }


class ModelSyntax(Protocol):
    """Grammar contract for a <model> body. Implementations must be able to split the
    body into the five blocks and extract declared symbols. Option B (full GAMS) is an
    alternative implementation of this same interface."""

    name: str
    required_blocks: List[str]

    def split_blocks(self, model_text: str) -> ParsedModel:
        ...

    def declared_symbols(self, parsed: ParsedModel) -> Dict[str, Dict[str, Any]]:
        """symbol -> {kind, index_dim, vtype} for SETS/PARAMETERS/VARIABLES."""
        ...

    def referenced_symbols(self, parsed: ParsedModel) -> List[str]:
        """symbols referenced by OBJECTIVE/CONSTRAINTS."""
        ...


# ---------------------------------------------------------------------------
# Option A: GAMS-style lightweight syntax
# ---------------------------------------------------------------------------

_BLOCK_HEADER = re.compile(r"^\s*(SETS|PARAMETERS|VARIABLES|OBJECTIVE|CONSTRAINTS|AUXILIARY)\s*:?\s*$", re.I)
_SYMBOL_DECL = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)((?:\[[^\]]*\])?)")
_INDEX_CONTENT = re.compile(r"\[([^\]]*)\]")
_VTYPE = re.compile(r"\b(binary|integer|continuous)\b", re.I)


class GamsStyleSyntax:
    """Lightweight GAMS-discipline grammar for the <model> body (Option A, D17)."""

    name = "gams-style-v1"
    required_blocks = ["SETS", "PARAMETERS", "VARIABLES", "OBJECTIVE", "CONSTRAINTS"]
    optional_blocks = ["AUXILIARY"]

    def split_blocks(self, model_text: str) -> ParsedModel:
        parsed = ParsedModel()
        current: Optional[str] = None
        block_map = {
            "SETS": parsed.sets,
            "PARAMETERS": parsed.parameters,
            "VARIABLES": parsed.variables,
            "OBJECTIVE": parsed.objective,
            "CONSTRAINTS": parsed.constraints,
            "AUXILIARY": parsed.auxiliary,
        }
        for raw in model_text.splitlines():
            line = raw.rstrip()
            header = _BLOCK_HEADER.match(line)
            if header:
                current = header.group(1).upper()
                continue
            if current and line.strip() and not line.strip().startswith("#"):
                block_map[current].append(line.strip())
        return parsed

    def declared_symbols(self, parsed: ParsedModel) -> Dict[str, Dict[str, Any]]:
        symbols: Dict[str, Dict[str, Any]] = {}
        for line in parsed.sets:
            match = _SYMBOL_DECL.match(line)
            if match:
                symbols[match.group(1)] = {"kind": "set", "index_dim": 0, "vtype": None}
            # register set members (e.g. "i in Animals = {cow, sheep, chicken}") so
            # literal indices like x['cow'] resolve to a declared member.
            for member in self._set_members(line):
                symbols.setdefault(member, {"kind": "set_member", "index_dim": 0, "vtype": None})
        for line in parsed.parameters:
            match = _SYMBOL_DECL.match(line)
            if match:
                symbols[match.group(1)] = {
                    "kind": "parameter",
                    "index_dim": self._index_dim(match.group(2)),
                    "vtype": None,
                }
        for line in parsed.variables:
            match = _SYMBOL_DECL.match(line)
            if match:
                vtype = _VTYPE.search(line)
                symbols[match.group(1)] = {
                    "kind": "variable",
                    "index_dim": self._index_dim(match.group(2)),
                    "vtype": vtype.group(1).lower() if vtype else "continuous",
                }
        # AUXILIARY block: declare symbols that appear on the LHS of "name = expr".
        # These are auxiliary variables (e.g. P_success = 1 - prod(i, 1-P[i])).
        # They are treated as declared variables so L2 does not flag them.
        for line in parsed.auxiliary:
            match = _SYMBOL_DECL.match(line)
            if match:
                symbols.setdefault(match.group(1), {
                    "kind": "variable",
                    "index_dim": self._index_dim(match.group(2)),
                    "vtype": "continuous",
                })
        return symbols

    @staticmethod
    def _set_members(set_line: str) -> List[str]:
        """Parse members from a set declaration like "i in Animals = {cow, sheep}"."""
        brace = re.search(r"\{([^}]*)\}", set_line)
        if not brace:
            return []
        return [m.strip().strip("'\"") for m in brace.group(1).split(",") if m.strip()]

    def referenced_symbols(self, parsed: ParsedModel) -> List[str]:
        refs: List[str] = []
        for line in parsed.objective + parsed.constraints + parsed.auxiliary:
            for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", line):
                refs.append(token)
        return refs

    @staticmethod
    def _index_dim(index_part: str) -> int:
        if not index_part:
            return 0
        content = _INDEX_CONTENT.search(index_part)
        if not content:
            return 0
        return len([part for part in content.group(1).split(",") if part.strip()])


# Tokens that legitimately appear in OBJECTIVE/CONSTRAINTS/AUXILIARY but are NOT model symbols:
# math keywords, summation/product notation, common math functions, and constraint labels.
_RESERVED = {
    "minimize", "maximize", "subject", "to", "sum", "sigma", "forall", "in",
    "s", "t", "st", "and", "or", "e", "pi", "le", "ge", "eq", "leq", "geq",
    # Common math functions that may appear in nonlinear objectives/constraints
    "prod", "exp", "log", "sqrt", "abs", "max", "min", "pow",
    # Common boolean/probability operators
    "not", "true", "false", "if", "then", "else",
}
_SUM_TOKEN = re.compile(r"^sum_?\{?.*$", re.I)
_CONSTRAINT_LABEL = re.compile(r"^C\d+$")


# ---------------------------------------------------------------------------
# Verification issues
# ---------------------------------------------------------------------------

@dataclass
class ModelIssue:
    layer: str            # format | structural | semantic
    type: str             # e.g. missing_block / undefined_symbol / signature_mismatch
    detail: str

    def to_dict(self) -> Dict[str, str]:
        return {"layer": self.layer, "type": self.type, "detail": self.detail}


@dataclass
class ModelValidationReport:
    passed: bool
    issues: List[ModelIssue] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"passed": self.passed, "issues": [issue.to_dict() for issue in self.issues]}


# ---------------------------------------------------------------------------
# Layer 1: format
# ---------------------------------------------------------------------------

class FormatValidator:
    """L1: tags present + five required blocks present and non-empty (deterministic)."""

    def __init__(self, syntax: Optional[ModelSyntax] = None):
        self.syntax = syntax or GamsStyleSyntax()

    def validate(self, think: Optional[str], model: Optional[str]) -> ModelValidationReport:
        issues: List[ModelIssue] = []
        if think is None:
            issues.append(ModelIssue("format", "missing_tag", "missing <think> block"))
        if model is None:
            issues.append(ModelIssue("format", "missing_tag", "missing <model> block"))
            return ModelValidationReport(passed=False, issues=issues)
        parsed = self.syntax.split_blocks(model)
        for block in self.syntax.required_blocks:
            if not parsed.blocks()[block]:
                issues.append(ModelIssue("format", "missing_block", "block {} is empty or absent".format(block)))
        return ModelValidationReport(passed=not issues, issues=issues)


# ---------------------------------------------------------------------------
# Layer 2: structural
# ---------------------------------------------------------------------------

class StructuralValidator:
    """L2: symbol cross-reference (declared vs referenced) + signature consistency."""

    def __init__(self, syntax: Optional[ModelSyntax] = None):
        self.syntax = syntax or GamsStyleSyntax()

    def validate(self, model: str, signature: Optional[Any] = None) -> ModelValidationReport:
        issues: List[ModelIssue] = []
        parsed = self.syntax.split_blocks(model)
        declared = self.syntax.declared_symbols(parsed)
        declared_names = set(declared) | _RESERVED
        index_positions = self._index_positions(parsed)
        for ref in set(self.syntax.referenced_symbols(parsed)):
            if self._is_noise(ref):
                continue
            if ref in declared_names:
                continue
            if len(ref) == 1 and ref in index_positions:
                continue  # single-letter used only as an index (i, j, t)
            issues.append(
                ModelIssue("structural", "undefined_symbol", "symbol {!r} is referenced but not declared".format(ref))
            )
        if signature is not None:
            issues.extend(self._check_signature(declared, signature))
        return ModelValidationReport(passed=not issues, issues=issues)

    @staticmethod
    def _index_positions(parsed: ParsedModel) -> set:
        """Letters that appear inside index brackets (declared sets make them indices)."""
        positions = set()
        for line in parsed.objective + parsed.constraints + parsed.auxiliary:
            for content in _INDEX_CONTENT.findall(line):
                for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", content):
                    positions.add(token)
        return positions

    @staticmethod
    def _is_noise(token: str) -> bool:
        """Summation notation and constraint labels are not model symbols."""
        return bool(_SUM_TOKEN.match(token) or _CONSTRAINT_LABEL.match(token))

    def _check_signature(self, declared: Dict[str, Dict[str, Any]], signature: Any) -> List[ModelIssue]:
        """Deterministically derive decision-structure hints and compare to signature."""
        issues: List[ModelIssue] = []
        variables = {name: m for name, m in declared.items() if m["kind"] == "variable"}
        has_binary = any(m["vtype"] == "binary" for m in variables.values())
        max_index = max((m["index_dim"] for m in variables.values()), default=0)
        sig_decision = set(getattr(signature, "decision", []) or [])
        if has_binary and "binary_assignment" not in sig_decision:
            issues.append(
                ModelIssue("structural", "signature_mismatch",
                           "model declares binary variables but signature.decision lacks 'binary_assignment'")
            )
        if max_index >= 2 and not any(d.startswith("multi_index") for d in sig_decision):
            issues.append(
                ModelIssue("structural", "signature_mismatch",
                           "model has {}-dim indexed variables but signature.decision lacks a multi_index tag".format(max_index))
            )
        return issues


# ---------------------------------------------------------------------------
# Layer 3: semantic (LLM-as-a-Judge, pluggable)
# ---------------------------------------------------------------------------

class SemanticValidator:
    """L3: LLM judge for missing/spurious constraints. No-op without an LLM client."""

    def __init__(self, llm_client: Optional[Any] = None):
        self.llm = llm_client

    async def validate(self, problem: str, model: str) -> ModelValidationReport:
        if self.llm is None:
            return ModelValidationReport(passed=True, issues=[])
        prompt = (
            "You are an OR modeling checker. Compare the PROBLEM and the MODEL. "
            "Report ONLY genuine modeling defects as a JSON array of objects with keys "
            "'type' and 'detail'. Check: (1) constraints required by the problem but "
            "missing from the model; (2) spurious or duplicated constraints; (3) known "
            "problem-family pitfalls (TSP subtour elimination, VRP capacity). "
            "Return an empty array if the model is faithful.\n\n"
            "PROBLEM:\n" + problem + "\n\nMODEL:\n" + model
        )
        issues: List[ModelIssue] = []
        try:
            raw = await self.llm.generate_object(prompt)
            for item in raw if isinstance(raw, list) else []:
                if isinstance(item, dict) and item.get("detail"):
                    issues.append(
                        ModelIssue("semantic", str(item.get("type", "semantic_defect")), str(item["detail"]))
                    )
        except (TypeError, ValueError, AttributeError):
            return ModelValidationReport(passed=True, issues=[])
        return ModelValidationReport(passed=not issues, issues=issues)


# ---------------------------------------------------------------------------
# Output parsing (<think>/<model>); <python> deferred to branch stage
#
# TWO accepted marker syntaxes (2026-08-26):
#   1. Square-bracket markers (PREFERRED for harness agents):
#          [THINK]...[/THINK]  [MODEL]...[/MODEL]
#      Some harnesses (Hermes-class chat pipelines) reserve/consume <think>
#      tags as their own reasoning-channel markers, so angle-bracket tags
#      cannot survive the trip from the agent to model.txt. Square brackets
#      pass through every known pipeline untouched.
#   2. Angle-bracket XML tags (legacy, still accepted):
#          <think>...</think>  <model>...</model>
# The parser accepts either (or a mix) and normalizes to the same output.
# ---------------------------------------------------------------------------

_TAG = re.compile(r"<(think|model)>(.*?)</\1>", re.S | re.I)
_BRACKET = re.compile(r"\[(think|model)\](.*?)\[/\1\]", re.S | re.I)


def parse_modeling_output(text: str) -> Dict[str, Optional[str]]:
    """Extract think/model bodies from either marker syntax.

    Accepts [THINK]...[/THINK]/[MODEL]...[/MODEL] (harness-safe, preferred)
    and <think>...</think>/<model>...</model> (legacy). Square-bracket hits
    take precedence when both appear. Returns None for any absent block.
    """
    text = text or ""
    found: Dict[str, str] = {}
    for name, body in _BRACKET.findall(text):
        found[name.lower()] = body.strip()
    if not found:
        for name, body in _TAG.findall(text):
            found[name.lower()] = body.strip()
    return {"think": found.get("think"), "model": found.get("model")}


# ---------------------------------------------------------------------------
# ModelingGate: orchestrate L1/L2 (+L3) and the repair loop
# ---------------------------------------------------------------------------

class ModelingGate:
    """Runs the three verification layers over a <think>/<model> candidate and decides
    whether the model may proceed to multi-branch code generation (D17)."""

    def __init__(
        self,
        syntax: Optional[ModelSyntax] = None,
        semantic_validator: Optional[SemanticValidator] = None,
        max_rounds: int = 3,
    ):
        self.syntax = syntax or GamsStyleSyntax()
        self.format_validator = FormatValidator(self.syntax)
        self.structural_validator = StructuralValidator(self.syntax)
        self.semantic_validator = semantic_validator or SemanticValidator()
        self.max_rounds = max(1, max_rounds)

    def check_static(self, problem: str, raw_output: str, signature: Optional[Any] = None) -> ModelValidationReport:
        """L1 + L2 only (synchronous, no LLM). Use for fast gating and in tests."""
        parsed = parse_modeling_output(raw_output)
        report = self.format_validator.validate(parsed["think"], parsed["model"])
        if not report.passed:
            return report
        structural = self.structural_validator.validate(parsed["model"], signature)
        issues = report.issues + structural.issues
        return ModelValidationReport(passed=not issues, issues=issues)

    async def check(self, problem: str, raw_output: str, signature: Optional[Any] = None) -> ModelValidationReport:
        """Full L1+L2+L3 pipeline."""
        static_report = self.check_static(problem, raw_output, signature)
        if not static_report.passed:
            return static_report
        parsed = parse_modeling_output(raw_output)
        semantic = await self.semantic_validator.validate(problem, parsed["model"])
        issues = static_report.issues + semantic.issues
        return ModelValidationReport(passed=not issues, issues=issues)


__all__ = [
    "ParsedModel",
    "ModelSyntax",
    "GamsStyleSyntax",
    "ModelIssue",
    "ModelValidationReport",
    "FormatValidator",
    "StructuralValidator",
    "SemanticValidator",
    "ModelingGate",
    "parse_modeling_output",
]
