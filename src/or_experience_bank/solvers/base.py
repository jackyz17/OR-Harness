"""Solver adapter contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict

from ..solving.execution import SafePythonExecutor
from ..core.schemas import AvailabilityResult, SolverExecutionResult


class SolverAdapter(ABC):
    name = "unknown"
    solver_family = "unknown"
    api = "unknown"

    def __init__(self, executor: SafePythonExecutor = None):
        self.executor = executor or SafePythonExecutor()

    @abstractmethod
    def is_available(self) -> AvailabilityResult:
        raise NotImplementedError

    def validate_environment(self) -> AvailabilityResult:
        return self.is_available()

    def build_generation_context(self, problem_spec: Dict[str, Any], attempt_number: int) -> Dict[str, Any]:
        return {
            "solver": self.name,
            "solver_family": self.solver_family,
            "api": self.api,
            "attempt_number": attempt_number,
            "result_contract": "Write result.json in the current working directory with fields: status, solver, objective_sense, objective_value, objective_bound, mip_gap, runtime_seconds, variables, diagnostics, message.",
            "blocked_imports": ["subprocess", "socket", "urllib", "http", "requests", "pathlib", "shutil"],
            "allowed_imports": ["os", "os.path", "json", "sys", "math", "itertools", "collections"],
        }

    async def execute(self, code_path: Path, workspace: Path, attempt_number: int) -> SolverExecutionResult:
        return await self.executor.execute(code_path, workspace, self.name)

    def normalize_feedback(self, result: SolverExecutionResult) -> str:
        return result.normalized_error or result.message or result.status

    def parse_result(self, result: SolverExecutionResult) -> SolverExecutionResult:
        return result

