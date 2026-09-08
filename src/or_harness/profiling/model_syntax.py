"""GAMS-style model representation: parsing and deterministic verification.

The harness agent is asked (by SKILL.md convention, never forced) to write a
model representation BEFORE writing solve.py, and to carry it in the task JSON
as the optional top-level ``model`` field. This module gives that
representation two things the framework can do without any LLM:

1. Verification (L1 format + L2 symbol cross-reference) — structural modeling
   errors are caught before they become solver-code retries (retries are a
   CostVector dimension by design).
2. Coupling derivation — the declared CONSTRAINTS are the cleanest possible
   input for structural coupling measurement (cleaner than solver-code AST,
   which mixes implementation detail with model structure).

Migrated from the legacy modeling contract, minus everything that belonged to
the framework-driven LLM loop (THINK/MODEL markers, L3 semantic judge, repair
rounds): the harness writes the model, the framework only checks it.

DSL (five blocks, headers case-insensitive, optional trailing colon):

    SETS:
     i in Projects = {p0, p1}
     j in Resources = {r0, r1}
    PARAMETERS:
     E[i,j]
     limit[j]
    VARIABLES:
     x[i,j] continuous >= 0
    OBJECTIVE:
     maximize sum(i, sum(j, (E[i,j] - alpha * C[i,j]) * x[i,j]))
    CONSTRAINTS:
     C1: sum(i, x[i,j]) <= limit[j]
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

BLOCK_NAMES = ("SETS", "PARAMETERS", "VARIABLES", "OBJECTIVE", "CONSTRAINTS")

_HEADER_RE = re.compile(r"^\s*(SETS|PARAMETERS|VARIABLES|OBJECTIVE|CONSTRAINTS)\s*:?\s*$",
                        re.IGNORECASE)
_SET_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s+in\s+([A-Za-z_]\w*)\s*=\s*\{(.*)\}\s*$",
                     re.IGNORECASE)
_SYMBOL_RE = re.compile(r"^([A-Za-z_]\w*)((?:\[[^\]]*\])?)\s*(.*)$")
_VAR_TYPE_RE = re.compile(r"\b(binary|integer|continuous)\b", re.IGNORECASE)
_LABEL_RE = re.compile(r"^C\d+$")

#: Tokens never treated as undeclared symbols in expressions.
_NOISE_PATTERNS = [
    re.compile(r"^sum_?\{?.*$", re.IGNORECASE),
    re.compile(r"^prod_?\{?.*$", re.IGNORECASE),
    re.compile(r"^C\d+$", re.IGNORECASE),
]
_NOISE_WORDS = {
    "minimize", "maximize", "subject", "to", "sum", "sigma", "forall", "in",
    "and", "or", "le", "ge", "eq", "leq", "geq",
    "exp", "log", "sqrt", "abs", "max", "min", "pow",
}


@dataclass
class ParsedModel:
    """Structured view of the five-block model representation."""

    sets: Dict[str, List[str]] = field(default_factory=dict)          # name -> members
    set_names: Dict[str, str] = field(default_factory=dict)           # index -> set name
    parameters: Dict[str, Optional[str]] = field(default_factory=dict)  # name -> index expr
    variables: Dict[str, str] = field(default_factory=dict)            # name -> type
    var_indices: Dict[str, str] = field(default_factory=dict)          # name -> index expr
    objective: str = ""
    constraints: List[Tuple[str, str]] = field(default_factory=list)   # (label, expr)

    def constraint_variable_sets(self) -> List[Set[str]]:
        """Variables referenced by each constraint — the coupling substrate."""
        result: List[Set[str]] = []
        for _label, expr in self.constraints:
            result.append({v for v in self.variables if _references(expr, v)})
        return result


@dataclass
class ModelIssue:
    layer: str  # "L1" | "L2"
    code: str
    detail: str

    def to_dict(self) -> Dict[str, str]:
        return {"layer": self.layer, "code": self.code, "detail": self.detail}


@dataclass
class ModelReport:
    parsed: Optional[ParsedModel] = None
    issues: List[ModelIssue] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.issues

    def to_dict(self) -> Dict[str, object]:
        return {"passed": self.passed,
                "issues": [i.to_dict() for i in self.issues]}


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------


def parse_model(text: str) -> ParsedModel:
    """Split the five blocks and populate the structured view.

    Lenient by design: structural problems are reported by verify_model, not
    by raising here.
    """
    model = ParsedModel()
    current: Optional[str] = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        header = _HEADER_RE.match(line)
        if header:
            current = header.group(1).upper()
            continue
        if current is None:
            continue  # content before any header: ignored (L1 reports empties)
        if current == "SETS":
            m = _SET_RE.match(line)
            if m:
                index, set_name, members = m.group(1), m.group(2), m.group(3)
                member_list = [s.strip().strip("'\"") for s in members.split(",")
                               if s.strip()]
                model.sets[set_name] = member_list
                model.set_names[index] = set_name
        elif current == "PARAMETERS":
            m = _SYMBOL_RE.match(line)
            if m:
                model.parameters[m.group(1)] = m.group(2) or None
        elif current == "VARIABLES":
            m = _SYMBOL_RE.match(line)
            if m:
                type_match = _VAR_TYPE_RE.search(m.group(3) or "")
                model.variables[m.group(1)] = (type_match.group(1).lower()
                                               if type_match else "continuous")
                model.var_indices[m.group(1)] = m.group(2) or ""
        elif current == "OBJECTIVE":
            model.objective = (model.objective + " " + line).strip()
        elif current == "CONSTRAINTS":
            if ":" in line:
                label, expr = line.split(":", 1)
                model.constraints.append((label.strip(), expr.strip()))
            else:
                model.constraints.append(("", line))
    return model


# ---------------------------------------------------------------------------
# verification (deterministic, no LLM)
# ---------------------------------------------------------------------------


def verify_model(text: str) -> ModelReport:
    """L1 (format) + L2 (symbol cross-reference) verification."""
    report = ModelReport()
    if not text or not text.strip():
        report.issues.append(ModelIssue("L1", "empty_model",
                                        "model representation is empty"))
        return report
    model = parse_model(text)
    report.parsed = model

    # L1: all five blocks non-empty.
    if not model.sets:
        report.issues.append(ModelIssue("L1", "missing_block",
                                        "SETS block is empty or absent"))
    if not model.parameters:
        report.issues.append(ModelIssue("L1", "missing_block",
                                        "PARAMETERS block is empty or absent"))
    if not model.variables:
        report.issues.append(ModelIssue("L1", "missing_block",
                                        "VARIABLES block is empty or absent"))
    if not model.objective:
        report.issues.append(ModelIssue("L1", "missing_block",
                                        "OBJECTIVE block is empty or absent"))
    if not model.constraints:
        report.issues.append(ModelIssue("L1", "missing_block",
                                        "CONSTRAINTS block is empty or absent"))

    # L2: constraint labels must be C1, C2, ...
    for label, _expr in model.constraints:
        if not label:
            report.issues.append(ModelIssue(
                "L2", "missing_label",
                "constraint without a label; use C1, C2, ..."))
        elif not _LABEL_RE.match(label):
            report.issues.append(ModelIssue(
                "L2", "bad_label",
                f"constraint label {label!r} must match C1, C2, ..."))

    # L2: every symbol referenced in OBJECTIVE/CONSTRAINTS must be declared.
    declared: Set[str] = set(model.parameters) | set(model.variables)
    declared |= set(model.sets)
    declared |= {m for members in model.sets.values() for m in members}
    for where, expr in [("OBJECTIVE", model.objective)] + \
            [(label or f"constraint#{i}", expr)
             for i, (label, expr) in enumerate(model.constraints, 1)]:
        for symbol in _referenced_symbols(expr):
            if symbol in declared or _is_noise(symbol):
                continue
            if len(symbol) == 1 and _inside_index_brackets(expr, symbol):
                continue  # single-letter index variable
            report.issues.append(ModelIssue(
                "L2", "undefined_symbol",
                f"{symbol!r} is referenced in {where} but not declared"))
    return report


# ---------------------------------------------------------------------------
# coupling derivation from the model representation
# ---------------------------------------------------------------------------


def coupling_from_model(model: ParsedModel) -> Dict[str, Optional[float]]:
    """Structural coupling measured from the declared model.

    resource_coupling: fraction of decision variables appearing in more than
    one constraint (0 = constraints independent; 1 = fully coupled). This is
    the operational definition — the same quantity the solver-code AST fallback
    approximates, measured here from the model itself.

    temporal_coupling: fraction of variables whose declared indices include a
    temporal set (name containing time/period/stage/day/hour) or whose index
    expression uses t.

    route_complexity: fraction of variables whose indices reference a
    network-ish set (arc/edge/node/road/link) — a conservative proxy.

    semantic_coupling: NOT derived — business semantics are invisible to
    structure; it always stays the harness's call.
    """
    if not model.variables or not model.constraints:
        return {"resource_coupling": None, "temporal_coupling": None,
                "route_complexity": None, "semantic_coupling": None}

    var_sets = model.constraint_variable_sets()
    n_vars = len(model.variables)
    shared = sum(1 for v in model.variables
                 if sum(1 for s in var_sets if v in s) > 1)
    rc = shared / n_vars

    temporal_sets = {idx for idx, set_name in model.set_names.items()
                     if re.search(r"time|period|stage|day|hour|week|month",
                                  set_name, re.IGNORECASE)}
    network_sets = {idx for idx, set_name in model.set_names.items()
                    if re.search(r"arc|edge|road|link|route|leg", set_name,
                                 re.IGNORECASE)}
    tc_count = rx_count = 0
    for name, index_expr in model.var_indices.items():
        indices = _index_names(index_expr)
        if any(i in temporal_sets or i.lower() == "t" for i in indices):
            tc_count += 1
        if any(i in network_sets for i in indices):
            rx_count += 1
    return {
        "resource_coupling": round(rc, 4),
        "temporal_coupling": round(tc_count / n_vars, 4),
        "route_complexity": round(rx_count / n_vars, 4),
        "semantic_coupling": None,
    }


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------


def _references(expr: str, symbol: str) -> bool:
    return re.search(rf"\b{re.escape(symbol)}\b", expr) is not None


def _referenced_symbols(expr: str) -> List[str]:
    return re.findall(r"[A-Za-z_]\w*", expr)


def _is_noise(token: str) -> bool:
    if token.lower() in _NOISE_WORDS:
        return True
    return any(p.match(token) for p in _NOISE_PATTERNS)


def _inside_index_brackets(expr: str, symbol: str) -> bool:
    for bracket in re.findall(r"\[([^\]]*)\]", expr):
        if symbol in re.findall(r"[A-Za-z_]\w*", bracket):
            return True
    return False


def _index_names(index_expr: str) -> List[str]:
    if not index_expr:
        return []
    inner = index_expr.strip()[1:-1] if index_expr.strip().startswith("[") \
        else index_expr.strip()
    return [t.strip() for t in inner.split(",") if t.strip()]
