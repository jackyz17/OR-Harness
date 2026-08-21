"""Solver discovery and explicit adapter registration."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from .base import SolverAdapter
from .copt import COPTAdapter
from .gurobi import GurobiAdapter
from .highs import HiGHSAdapter
from .ortools import ORToolsAdapter
from .pulp import PuLPAdapter
from .pyomo import PyomoAdapter
from .scip import SCIPAdapter
from ..core.schemas import AvailabilityResult


class SolverRegistry:
    def __init__(self, adapters: Optional[Iterable[SolverAdapter]] = None):
        defaults = adapters if adapters is not None else [
            GurobiAdapter(), SCIPAdapter(), HiGHSAdapter(), COPTAdapter(),
            ORToolsAdapter(), PuLPAdapter(), PyomoAdapter(),
        ]
        self._adapters: Dict[str, SolverAdapter] = {adapter.name: adapter for adapter in defaults}

    def register(self, adapter: SolverAdapter) -> None:
        if adapter.name in self._adapters:
            raise ValueError("Solver already registered: " + adapter.name)
        self._adapters[adapter.name] = adapter

    def get(self, name: str) -> SolverAdapter:
        if name not in self._adapters:
            raise KeyError("Unknown solver: " + name)
        return self._adapters[name]

    def available(self, names: Iterable[str]) -> tuple:
        selected: List[SolverAdapter] = []
        unavailable: Dict[str, AvailabilityResult] = {}
        for name in names:
            try:
                adapter = self.get(name)
            except KeyError:
                unavailable[name] = AvailabilityResult(False, "unsupported solver", "unsupported_solver")
                continue
            result = adapter.validate_environment()
            if result.available:
                selected.append(adapter)
            else:
                unavailable[name] = result
        return selected, unavailable

    def names(self) -> List[str]:
        return sorted(self._adapters)
