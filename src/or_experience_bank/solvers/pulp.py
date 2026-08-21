from __future__ import annotations

import importlib.util

from .base import SolverAdapter
from ..core.schemas import AvailabilityResult, TerminationReason


class PuLPAdapter(SolverAdapter):
    """PuLP modeling-and-solving branch (Option 1: treated as its own branch).

    PuLP is a modeling framework; this branch uses the PuLP API with its default
    bundled solver (CBC). It tests PuLP-API code generation quality as a branch."""

    name = "pulp"
    solver_family = "milp"
    api = "pulp"

    def is_available(self) -> AvailabilityResult:
        if importlib.util.find_spec("pulp") is None:
            return AvailabilityResult(False, "pulp is not installed", TerminationReason.SOLVER_UNAVAILABLE.value)
        return AvailabilityResult(True, "PuLP is available (default CBC solver)")
