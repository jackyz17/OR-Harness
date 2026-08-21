from __future__ import annotations

import importlib.util

from .base import SolverAdapter
from ..core.schemas import AvailabilityResult, TerminationReason


class HiGHSAdapter(SolverAdapter):
    """HiGHS solver via highspy. Open source (no license check, per project decision)."""

    name = "highs"
    solver_family = "milp"
    api = "highspy"

    def is_available(self) -> AvailabilityResult:
        if importlib.util.find_spec("highspy") is None:
            return AvailabilityResult(False, "highspy is not installed", TerminationReason.SOLVER_UNAVAILABLE.value)
        return AvailabilityResult(True, "HiGHS (highspy) is available")
