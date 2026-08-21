from __future__ import annotations

import importlib.util

from .base import SolverAdapter
from ..core.schemas import AvailabilityResult, TerminationReason


class PyomoAdapter(SolverAdapter):
    """Pyomo modeling-and-solving branch (Option 1: treated as its own branch).

    Pyomo is a modeling framework that can connect to many backend solvers. As a branch
    it tests Pyomo-API code generation quality; availability requires both pyomo and at
    least one usable backend solver."""

    name = "pyomo"
    solver_family = "milp"
    api = "pyomo"

    def is_available(self) -> AvailabilityResult:
        if importlib.util.find_spec("pyomo") is None:
            return AvailabilityResult(False, "pyomo is not installed", TerminationReason.SOLVER_UNAVAILABLE.value)
        # Pyomo needs a backend solver to actually optimize. Check a common set.
        backends = [m for m in ("highspy", "gurobipy", "pyscipopt", "coptpy", "pulp")
                    if importlib.util.find_spec(m) is not None]
        if not backends:
            return AvailabilityResult(
                False,
                "pyomo is installed but no backend solver (highspy/gurobipy/pyscipopt/coptpy/pulp) is available",
                TerminationReason.SOLVER_UNAVAILABLE.value,
            )
        return AvailabilityResult(True, "Pyomo is available with backends: " + ", ".join(backends))
