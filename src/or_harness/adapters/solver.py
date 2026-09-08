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
                "message": self.message}


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
        if importlib.util.find_spec("gurobipy") is None:
            return Availability(self.name, self.solver_family, self.api, False,
                                "gurobipy is not installed")
        try:  # license probe: starting an environment fails without a license
            import gurobipy as gp

            env = gp.Env(empty=True)
            env.setParam("OutputFlag", 0)
            env.start()
            env.dispose()
            return Availability(self.name, self.solver_family, self.api, True,
                                "gurobipy and license are available")
        except Exception as exc:  # pragma: no cover - environment specific
            return Availability(self.name, self.solver_family, self.api, False,
                                f"{type(exc).__name__}: {str(exc)[:200]}")


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
