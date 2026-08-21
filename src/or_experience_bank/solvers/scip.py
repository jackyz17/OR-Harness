from __future__ import annotations

import importlib.util

from .base import SolverAdapter
from ..core.schemas import AvailabilityResult, TerminationReason


class SCIPAdapter(SolverAdapter):
    name = "scip"
    solver_family = "milp"
    api = "pyscipopt"

    def is_available(self) -> AvailabilityResult:
        if importlib.util.find_spec("pyscipopt") is None:
            return AvailabilityResult(False, "pyscipopt is not installed", TerminationReason.SOLVER_UNAVAILABLE.value)
        return AvailabilityResult(True, "PySCIPOpt is available")

