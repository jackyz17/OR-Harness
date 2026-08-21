from __future__ import annotations

import importlib.util

from .base import SolverAdapter
from ..core.schemas import AvailabilityResult, TerminationReason


class COPTAdapter(SolverAdapter):
    """COPT solver via coptpy. Treated as install-only check (open handling, per project
    decision): we detect the module and do not perform a license probe here."""

    name = "copt"
    solver_family = "milp"
    api = "coptpy"

    def is_available(self) -> AvailabilityResult:
        if importlib.util.find_spec("coptpy") is None:
            return AvailabilityResult(False, "coptpy is not installed", TerminationReason.SOLVER_UNAVAILABLE.value)
        return AvailabilityResult(True, "COPT (coptpy) is available")
