from __future__ import annotations

import unittest

from helpers import SRC_DIR
from or_experience_bank.core.schemas import AvailabilityResult
from or_experience_bank.solvers.base import SolverAdapter
from or_experience_bank.solvers.mock import MockSolverAdapter
from or_experience_bank.solvers.registry import SolverRegistry


class Unavailable(SolverAdapter):
    name = "missing"
    solver_family = "milp"
    api = "missing"

    def is_available(self):
        return AvailabilityResult(False, "license unavailable", "license_error")


class RegistryTests(unittest.TestCase):
    def test_unavailable_is_skipped_and_mock_is_available(self):
        registry = SolverRegistry([Unavailable(), MockSolverAdapter("mock")])
        available, unavailable = registry.available(["missing", "mock"])
        self.assertEqual([item.name for item in available], ["mock"])
        self.assertEqual(unavailable["missing"].termination_reason, "license_error")
        self.assertTrue(registry.get("mock").is_available().available)

    def test_default_registry_has_all_seven_solvers(self):
        registry = SolverRegistry()
        names = registry.names()
        for expected in ("gurobi", "scip", "highs", "copt", "ortools", "pulp", "pyomo"):
            self.assertIn(expected, names)

    def test_new_adapters_have_family_and_api(self):
        registry = SolverRegistry()
        for name in ("highs", "copt", "pulp", "pyomo"):
            adapter = registry.get(name)
            self.assertEqual(adapter.solver_family, "milp", name)
            self.assertTrue(adapter.api, name)

    def test_new_adapters_availability_returns_result(self):
        # Availability depends on whether the package is installed; we only assert the
        # call returns a well-formed AvailabilityResult without raising (no install here).
        registry = SolverRegistry()
        for name in ("highs", "copt", "pulp", "pyomo"):
            result = registry.get(name).is_available()
            self.assertIsInstance(result.available, bool, name)
            self.assertTrue(result.reason, name)

