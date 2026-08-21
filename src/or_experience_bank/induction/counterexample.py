"""Counterexample search: solver-backed refutation (module 3.5).

A hypothesis is not knowledge until it survives an attempt to BREAK it. This module
generates potential failure conditions for a candidate principle, then tries to confirm
a genuine counterexample by EXECUTING a small refutation problem through a solver.

Red line (anti self-judgment): the LLM proposes failure conditions and writes the
refutation code, but it NEVER decides whether the principle actually failed. That verdict
comes only from SafePythonExecutor execution evidence. "LLM proposes, solver disposes."

Flow per hypothesis:
  1. propose_failure_conditions(hypothesis)   -> LLM: fixed-charge / nonconvex / min-batch ...
  2. build_refutation(hypothesis, condition)  -> LLM: small Python program that instantiates
     the condition and prints whether the principle's prediction holds (verdict JSON).
  3. SafePythonExecutor.execute(program)      -> execution evidence (the actual verdict).
  4. verdict = refuted iff execution succeeded AND the program reported the principle fails.

Confirmed counterexamples shrink the pattern's applicability_conditions and are recorded
(CounterexampleRecord); they never silently delete the hypothesis (append-only red line).

D18 harness principle: the framework emits prompts + parses + owns the executor verdict;
the agent owns the LLM. The executor is injectable so tests can supply a stub.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.modeling_schemas import CounterexampleRecord
from .inducer import PrincipleHypothesis


@dataclass
class RefutationAttempt:
    """One (failure_condition, refutation_program) pair + its execution verdict."""

    condition: str = ""
    program: str = ""
    executed: bool = False
    execution_status: str = ""          # executor status: ok/error/timeout/...
    principle_failed: Optional[bool] = None  # parsed from program verdict; None if unexecuted
    evidence: str = ""                  # human-readable execution summary

    @property
    def is_counterexample(self) -> bool:
        # confirmed only when EXECUTED and the program reported the principle fails
        return self.executed and self.principle_failed is True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "condition": self.condition,
            "executed": self.executed,
            "execution_status": self.execution_status,
            "principle_failed": self.principle_failed,
            "is_counterexample": self.is_counterexample,
            "evidence": self.evidence,
        }


@dataclass
class RefutationResult:
    """Aggregate refutation outcome for one hypothesis."""

    hypothesis_id: str
    attempts: List[RefutationAttempt] = field(default_factory=list)
    surviving_conditions: List[str] = field(default_factory=list)  # conditions that did NOT break it
    counterexamples: List[CounterexampleRecord] = field(default_factory=list)
    shrunk_applicability: List[str] = field(default_factory=list)   # narrowed applicability

    @property
    def refuted(self) -> bool:
        return any(a.is_counterexample for a in self.attempts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "refuted": self.refuted,
            "attempts": [a.to_dict() for a in self.attempts],
            "counterexamples": [c.to_dict() for c in self.counterexamples],
            "shrunk_applicability": list(self.shrunk_applicability),
        }


class CounterexampleSearcher:
    """Framework-side counterexample rules: prompts + verdict parsing + executor ownership."""

    def __init__(self, executor: Optional[Any] = None, solver: str = "highs"):
        # executor is any object with async execute(code_path, workspace, solver)
        # -> SolverExecutionResult. Default None => harness supplies one at runtime.
        self.executor = executor
        self.solver = solver

    # -- prompts ------------------------------------------------------------

    def build_failure_conditions_prompt(self, hypothesis: PrincipleHypothesis) -> str:
        return (
            "A candidate OR optimization principle is shown below. List the structural conditions "
            "under which it would FAIL. Think of: fixed/setup (fixed-charge) costs, nonconvex or "
            "nonlinear coupling, minimum batch sizes, indivisibilities, precedence interactions, "
            "degenerate ties in marginal contribution.\n\n"
            "PRINCIPLE: " + hypothesis.statement + "\n"
            "CURRENT APPLICABILITY: " + "; ".join(hypothesis.applicability_conditions or ["(none)"]) + "\n\n"
            "Return ONLY a JSON list of short failure-condition strings (at most 4)."
        )

    def build_refutation_prompt(self, hypothesis: PrincipleHypothesis, condition: str) -> str:
        return (
            "Write a SMALL self-contained Python program that instantiates the failure condition "
            "below and checks whether the principle's prediction actually FAILS. The program must "
            "be runnable with the standard library only and must print, as its LAST stdout line, a "
            "JSON object: {\"principle_failed\": true|false, \"evidence\": \"...\"}.\n\n"
            "PRINCIPLE: " + hypothesis.statement + "\n"
            "FAILURE CONDITION TO INSTANTIATE: " + condition + "\n\n"
            "Return ONLY the Python source code, no markdown fences."
        )

    # -- parsing ------------------------------------------------------------

    def parse_failure_conditions(self, raw: Any) -> List[str]:
        data = self._coerce_json(raw)
        if isinstance(data, dict):
            data = data.get("conditions", [])
        if not isinstance(data, list):
            return []
        return [str(c).strip() for c in data if str(c).strip()][:4]

    def parse_verdict(self, stdout: str) -> Optional[bool]:
        """Extract the program's self-reported verdict from its last JSON stdout line."""
        for line in reversed([ln for ln in stdout.splitlines() if ln.strip()]):
            try:
                obj = json.loads(line)
            except (ValueError, TypeError):
                continue
            if isinstance(obj, dict) and "principle_failed" in obj:
                return bool(obj["principle_failed"])
        return None

    # -- executor-driven verdict (the anti self-judgment step) -------------

    async def run_refutation(self, attempt: RefutationAttempt, workspace: Path) -> RefutationAttempt:
        """Execute the refutation program; the EXECUTOR decides failure, not the LLM."""
        if self.executor is None:
            attempt.evidence = "no executor supplied (harness mode: agent runs the program)"
            return attempt
        workspace = Path(workspace)
        workspace.mkdir(parents=True, exist_ok=True)
        code_path = workspace / "refutation.py"
        code_path.write_text(attempt.program, encoding="utf-8")
        try:
            result = await self.executor.execute(code_path, workspace, self.solver)
        except Exception as exc:  # executor failure is NOT a counterexample
            attempt.executed = True
            attempt.execution_status = "executor_error"
            attempt.evidence = "executor raised: " + str(exc)
            return attempt
        attempt.executed = True
        attempt.execution_status = getattr(result, "status", "unknown")
        stdout = getattr(result, "stdout", "") or ""
        attempt.principle_failed = self.parse_verdict(stdout)
        attempt.evidence = (
            "status={status} verdict={verdict}".format(
                status=attempt.execution_status, verdict=attempt.principle_failed
            )
        )
        return attempt

    def aggregate(self, hypothesis: PrincipleHypothesis, attempts: List[RefutationAttempt]) -> RefutationResult:
        """Summarize attempts into counterexamples + shrunk applicability (append-only)."""
        result = RefutationResult(hypothesis_id=hypothesis.hypothesis_id, attempts=attempts)
        for a in attempts:
            if a.is_counterexample:
                result.counterexamples.append(
                    CounterexampleRecord(summary=a.condition, solver_evidence=a.evidence)
                )
                result.shrunk_applicability.append("NOT when: " + a.condition)
            elif a.executed and a.principle_failed is False:
                result.surviving_conditions.append(a.condition)
        return result

    @staticmethod
    def _coerce_json(raw: Any) -> Optional[Any]:
        if raw is None:
            return None
        if isinstance(raw, (dict, list)):
            return raw
        text = str(raw).strip()
        if "[" in text:
            try:
                return json.loads(text[text.find("["): text.rfind("]") + 1])
            except (ValueError, TypeError):
                pass
        if "{" in text:
            try:
                return json.loads(text[text.find("{"): text.rfind("}") + 1])
            except (ValueError, TypeError):
                pass
        return None


class LLMBackedCounterexampleSearcher:
    """OPTIONAL convenience loop for standalone runs/tests (NOT used in harness mode)."""

    def __init__(
        self,
        searcher: Optional[CounterexampleSearcher] = None,
        llm_client: Optional[Any] = None,
    ):
        self.searcher = searcher or CounterexampleSearcher()
        self.llm = llm_client

    async def search(self, hypothesis: PrincipleHypothesis, workspace: Path) -> RefutationResult:
        attempts: List[RefutationAttempt] = []
        if self.llm is None:
            return self.searcher.aggregate(hypothesis, attempts)
        cond_raw = await self.llm.generate_object(self.searcher.build_failure_conditions_prompt(hypothesis))
        for condition in self.searcher.parse_failure_conditions(cond_raw):
            program = await self.llm.generate_text(self.searcher.build_refutation_prompt(hypothesis, condition))
            attempt = RefutationAttempt(condition=condition, program=program)
            attempt = await self.searcher.run_refutation(attempt, Path(workspace) / _safe(condition))
            attempts.append(attempt)
        return self.searcher.aggregate(hypothesis, attempts)


def _safe(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text)[:40] or "case"


__all__ = [
    "CounterexampleSearcher",
    "LLMBackedCounterexampleSearcher",
    "RefutationAttempt",
    "RefutationResult",
]
