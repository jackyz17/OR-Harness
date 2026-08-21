"""Deterministic adapter used only when explicitly injected by tests/demo."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import List, Optional

from .base import SolverAdapter
from ..core.schemas import AvailabilityResult, SolverExecutionResult


class MockSolverAdapter(SolverAdapter):
    solver_family = "mock"
    api = "mock"

    def __init__(self, name: str = "mock", outcomes: Optional[List[SolverExecutionResult]] = None, delay: float = 0.0):
        super().__init__()
        self.name = name
        self.outcomes = list(outcomes or [
            SolverExecutionResult(
                status="optimal", solver=name, exit_code=0, objective_sense="minimize",
                objective_value=1.0, runtime_seconds=0.01, variables={"x": 1},
            )
        ])
        self.delay = delay
        self.started = []
        self.finished = []

    def is_available(self) -> AvailabilityResult:
        return AvailabilityResult(True, "explicit mock adapter")

    async def execute(self, code_path: Path, workspace: Path, attempt_number: int) -> SolverExecutionResult:
        self.started.append((attempt_number, str(workspace)))
        if self.delay:
            await asyncio.sleep(self.delay)
        index = min(attempt_number - 1, len(self.outcomes) - 1)
        source = self.outcomes[index]
        result = SolverExecutionResult(**source.__dict__)
        result.solver = self.name
        self.finished.append((attempt_number, str(workspace)))
        return result

