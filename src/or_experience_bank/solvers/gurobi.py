from __future__ import annotations

import importlib.util

from .base import SolverAdapter
from ..core.schemas import AvailabilityResult, TerminationReason


class GurobiAdapter(SolverAdapter):
    name = "gurobi"
    solver_family = "milp"
    api = "gurobipy"

    def is_available(self) -> AvailabilityResult:
        if importlib.util.find_spec("gurobipy") is None:
            return AvailabilityResult(False, "gurobipy is not installed", TerminationReason.SOLVER_UNAVAILABLE.value)
        try:
            import gurobipy as gp

            env = gp.Env(empty=True)
            env.setParam("OutputFlag", 0)
            env.start()
            env.dispose()
            return AvailabilityResult(True, "gurobipy and license are available")
        except Exception as exc:
            message = str(exc).lower()
            reason = TerminationReason.LICENSE_ERROR.value if "license" in message else TerminationReason.SOLVER_UNAVAILABLE.value
            return AvailabilityResult(False, type(exc).__name__ + ": " + str(exc)[:300], reason)

