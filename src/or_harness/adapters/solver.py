"""Solver adapters: thin availability probes over seven solver bindings.

The harness chooses the concrete solver per situation (strategies only carry a
solver *family* hint); these adapters answer exactly one question: "is this
solver usable in this environment?" Zero-dependency probing — solver packages
are optional extras, discovered at runtime by ``orx doctor``.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Dict, List, Optional

API_HINTS = {
    "highs": {"entrypoint": "highspy.Highs", "domains": "continuous, integer, binary",
              "constraints": "linear", "time_limit_api": "setOptionValue('time_limit', seconds)",
              "execution_mode": "in-process"},
    "pulp": {"entrypoint": "pulp.LpProblem / LpVariable", "domains": "continuous, integer, binary",
             "constraints": "linear", "time_limit_api": "backend-dependent",
             "execution_mode": "external CBC launch is blocked; use an in-process backend"},
    "ortools": {"entrypoint": "ortools.sat.python.cp_model.CpModel / CpSolver",
                "domains": "integer, boolean", "constraints": "CP-SAT linear, logical, scheduling",
                "time_limit_api": "solver.parameters.max_time_in_seconds", "execution_mode": "in-process"},
    "scip": {"entrypoint": "pyscipopt.Model", "domains": "continuous, integer, binary",
             "constraints": "linear and supported nonlinear", "time_limit_api": "setParam('limits/time', seconds)",
             "execution_mode": "in-process"},
    "copt": {"entrypoint": "coptpy.Envr().createModel()", "domains": "continuous, integer, binary",
             "constraints": "binding-dependent; confirm required type", "time_limit_api": "model.setParam('TimeLimit', seconds)",
             "execution_mode": "in-process; license availability unknown"},
    "pyomo": {"entrypoint": "pyomo.environ.ConcreteModel / SolverFactory",
              "domains": "continuous, integer, binary", "constraints": "backend-dependent",
              "time_limit_api": "backend-dependent", "execution_mode": "external solver launches are blocked"},
    "gurobi": {"entrypoint": "gurobipy.Model", "domains": "continuous, integer, binary",
               "constraints": "binding-dependent; confirm required type", "time_limit_api": "model.Params.TimeLimit",
               "execution_mode": "in-process; remote/license-service networking is blocked"},
}


@dataclass
class Availability:
    name: str
    solver_family: str
    api: str
    available: bool
    message: str

    def to_dict(self) -> Dict[str, object]:
        return {"name": self.name, "solver_family": self.solver_family,
                "api": self.api, "available": self.available,
                "message": self.message, "probe_scope": "module_spec_only",
                "runtime_usable": None, "sandbox_usable": None,
                "api_hint": API_HINTS.get(self.name),
                "note": "module discovery does not confirm imports, backend, license or execution in the sandbox"}


class SolverAdapter:
    """Minimal contract: metadata + availability probe."""

    name = "unknown"
    solver_family = "unknown"
    api = "unknown"

    def is_available(self) -> Availability:
        raise NotImplementedError

    def _spec_probe(self, ok_message: str, missing_message: Optional[str] = None
                    ) -> Availability:
        if importlib.util.find_spec(self.api) is None:
            return Availability(self.name, self.solver_family, self.api, False,
                                missing_message or f"{self.api} is not installed")
        return Availability(self.name, self.solver_family, self.api, True, ok_message)


class HiGHSAdapter(SolverAdapter):
    name, solver_family, api = "highs", "milp", "highspy"

    def is_available(self) -> Availability:
        return self._spec_probe("HiGHS (highspy) is available")


class PuLPAdapter(SolverAdapter):
    name, solver_family, api = "pulp", "milp", "pulp"

    def is_available(self) -> Availability:
        return self._spec_probe("PuLP is available (bundled CBC backend)")


class ORToolsAdapter(SolverAdapter):
    name, solver_family, api = "ortools", "cp_sat", "ortools"

    def is_available(self) -> Availability:
        if importlib.util.find_spec("ortools") is None:
            return Availability(self.name, self.solver_family, self.api, False,
                                "ortools is not installed")
        if importlib.util.find_spec("ortools.sat.python.cp_model") is None:
            return Availability(self.name, self.solver_family, self.api, False,
                                "ortools CP-SAT module is missing")
        return Availability(self.name, self.solver_family, self.api, True,
                            "OR-Tools CP-SAT is available")


class SCIPAdapter(SolverAdapter):
    name, solver_family, api = "scip", "milp", "pyscipopt"

    def is_available(self) -> Availability:
        return self._spec_probe("SCIP (pyscipopt) is available")


class COPTAdapter(SolverAdapter):
    name, solver_family, api = "copt", "milp", "coptpy"

    def is_available(self) -> Availability:
        return self._spec_probe("COPT (coptpy) is available")


class PyomoAdapter(SolverAdapter):
    name, solver_family, api = "pyomo", "milp", "pyomo"

    def is_available(self) -> Availability:
        try:
            found = importlib.util.find_spec("pyomo.environ") is not None
        except ModuleNotFoundError:
            found = False  # parent package itself is missing
        if not found:
            return Availability(self.name, self.solver_family, self.api, False,
                                "pyomo is not installed")
        return Availability(self.name, self.solver_family, self.api, True,
                            "Pyomo is available (a backend solver is also required)")


class GurobiAdapter(SolverAdapter):
    name, solver_family, api = "gurobi", "milp", "gurobipy"

    def is_available(self) -> Availability:
        return self._spec_probe("gurobipy module found; license/runtime not tested")


def default_adapters() -> List[SolverAdapter]:
    return [HiGHSAdapter(), PuLPAdapter(), ORToolsAdapter(), SCIPAdapter(),
            COPTAdapter(), PyomoAdapter(), GurobiAdapter()]


def probe_all(adapters: Optional[List[SolverAdapter]] = None) -> List[Availability]:
    return [a.is_available() for a in (adapters or default_adapters())]


def available_families(adapters: Optional[List[SolverAdapter]] = None) -> Dict[str, List[str]]:
    """family -> [available solver names]; lets the harness map a strategy's
    solver_family hint onto a concrete, usable solver."""
    families: Dict[str, List[str]] = {}
    for avail in probe_all(adapters):
        if avail.available:
            families.setdefault(avail.solver_family, []).append(avail.name)
    return families
