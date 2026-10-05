"""Solver-failure facts: classification and environment advisories.

The induction DETECTORS (strategy_contrast / intervention_recovery /
structural_reproduction / advantage_reversal) were removed — the framework no
longer manufactures induction labels from online pattern hits. What remains
is purely factual and recomputable, and this module tests exactly that.
"""
import unittest

from helpers import HarnessTestCase

from or_harness.core.schema import FailureRecord
from or_harness.strategy.experience_bank import ExperienceBank
from or_harness.strategy.triggers import classify_failure, solver_advisories


class TestFailureClassification(HarnessTestCase):
    def setUp(self):
        super().setUp()
        self.bank = ExperienceBank(self.store)

    def failed(self, execution_id, solver, error):
        rec = self.make_record(execution_id=execution_id, task_id="t1",
                               feasible=False, status="error",
                               solver={"name": solver, "code_hash": "x"})
        rec.failures = [FailureRecord(attempt=1, error=error)]
        return rec

    def test_environment_vs_model(self):
        env = self.failed("ex_e", "pulp",
                          "security policy: blocked import subprocess")
        mod = self.failed("ex_m", "highs", "TypeError: bad operand")
        self.assertEqual(classify_failure(env), "environment")
        self.assertEqual(classify_failure(mod), "model")

    def test_missing_module_is_environment(self):
        rec = self.failed("ex_n", "highs",
                          "ModuleNotFoundError: No module named 'highspy'")
        self.assertEqual(classify_failure(rec), "environment")


class TestSolverAdvisories(HarnessTestCase):
    def setUp(self):
        super().setUp()
        self.bank = ExperienceBank(self.store)

    def test_only_environment_failures_listed(self):
        env_fail = self.make_record(
            execution_id="ex_a", task_id="t1", feasible=False, status="error",
            solver={"name": "pulp", "code_hash": "x"},
            failures=[FailureRecord(
                1, "security policy: blocked import subprocess")])
        model_fail = self.make_record(
            execution_id="ex_b", task_id="t2", feasible=False, status="error",
            solver={"name": "highs", "code_hash": "y"},
            failures=[FailureRecord(1, "TypeError: unsupported operand")])
        ok = self.make_record(execution_id="ex_c", task_id="t3",
                              solver={"name": "ortools", "code_hash": "z"})
        for rec in (env_fail, model_fail, ok):
            self.bank.append(rec)
        advisories = solver_advisories(self.bank)
        self.assertEqual(len(advisories), 1)  # pulp only
        self.assertEqual(advisories[0]["solver"], "pulp")
        self.assertEqual(advisories[0]["environment_failures"], 1)

    def test_empty_when_no_failures(self):
        self.bank.append(self.make_record(execution_id="ex_ok"))
        self.assertEqual(solver_advisories(self.bank), [])


class TestNoDetectorSurface(unittest.TestCase):
    """The removed detectors must not reappear on the module."""

    def test_detectors_are_gone(self):
        import or_harness.strategy.triggers as triggers
        for name in ("check_triggers", "InductionHint", "PATTERNS",
                     "evidence_execution_ids"):
            self.assertFalse(hasattr(triggers, name),
                             f"{name} must not exist on the triggers module")

    def test_retained_facts_are_present(self):
        import or_harness.strategy.triggers as triggers
        self.assertTrue(callable(triggers.classify_failure))
        self.assertTrue(callable(triggers.solver_advisories))
