from __future__ import annotations

import importlib.util

from .base import SolverAdapter
from ..core.schemas import AvailabilityResult, TerminationReason


class ORToolsAdapter(SolverAdapter):
    name = "ortools"
    solver_family = "cp_sat"
    api = "ortools.cp_model"

    def is_available(self) -> AvailabilityResult:
        if importlib.util.find_spec("ortools") is None:
            return AvailabilityResult(False, "ortools is not installed", TerminationReason.SOLVER_UNAVAILABLE.value)
        return AvailabilityResult(True, "OR-Tools is available")

