"""Modeling-repair material tests (2026-09-29 defects, detector-free).

The induction DETECTORS were removed, so this module no longer tests that a
repair "fires a hint". What it still pins is the MATERIAL the framework
records and reports for a modeling-repair story:

- a task-check failure is a REAL failure (the answer was wrong though the
  solver was happy), and a passed answer is not a prior failure;
- the material report names what the evidence carries (a performed method?
  a passed check?) and what is ``missing`` — a report, not an admission
  verdict;
- an excluded execution no longer counts as evidence (at the statistics
  layer that every material reader uses).

Run with the same conventions as the rest of the suite:
    PYTHONPATH=src python -m unittest discover -s tests/harness -p test_method_repair.py -q
"""
import io
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tests.harness.helpers import HarnessTestCase  # noqa: E402

from or_harness.api import ORHarness  # noqa: E402
from or_harness.core.schema import ExecutionRecord  # noqa: E402
from or_harness.world_model.provider import WorldModelProvider  # noqa: E402

#: The per-box model (WRONG): the cost is folded into a linear per-box rate.
PER_BOX_METHOD = {"name": "per-box cost insertion",
                  "steps": ["cost_per_box = 10", "min 10 * boxes"]}
#: The integer-car model (RIGHT): an integer vehicle variable connects it.
INTEGER_CAR_METHOD = {
    "name": "integer vehicle variable",
    "steps": ["int car", "boxes <= 10 * car", "min 100 * car"]}

#: The transport task: 15 boxes, 10 per car, 100 per car.
TASK = {"task_id": "transport", "family": "routing",
        "description": "ship 15 boxes, 10 per car, 100 per car",
        "annotations": {"coupling": {"resource_coupling": 0.9,
                                     "temporal_coupling": 0.1}}}


class NoProvider(WorldModelProvider):
    """A provider that must never be called here."""

    name = "noop"

    def predict(self, request, timeout_s=None):
        raise AssertionError("no model call expected")


class RepairCase(HarnessTestCase):
    def harness(self, **kwargs):
        h = ORHarness(home=self.home, world_model=NoProvider(), **kwargs)
        self.addCleanup(h.close)
        return h

    def _record(self, h, execution_id, *, feasible, status, objective,
                method=None, method_planned=None, task_check=None,
                solver="highs", task_id="transport", strategy_id="S-transport"):
        rec = ExecutionRecord(
            execution_id=execution_id, task_id=task_id,
            strategy_id=strategy_id,
            profile_snapshot=h.profile(dict(TASK, task_id=task_id)),
            quality={"feasible": feasible, "objective": objective,
                     "gap": 0.0, "status": status},
            solver={"name": solver},
            method_actual=method, method_planned=method_planned)
        h.record(rec)
        if task_check is not None:
            h.check_task_result(execution_id,
                                {"reference_objective": task_check})
        return rec


# ---------------------------------------------------------------------------
# A plan is not a performed method — the material report says so
# ---------------------------------------------------------------------------

class TestMaterialReportNamesWhatItCarries(RepairCase):
    def test_planned_only_evidence_reports_planned_basis(self):
        """A candidate whose evidence reports only PLANNED methods reports
        basis ``planned_only`` and names ``method_performed`` as missing."""
        h = self.harness()
        for i in range(2):
            h.record(ExecutionRecord(
                execution_id=f"m{i}", task_id=f"T{i}", strategy_id="S-plan",
                profile_snapshot=h.profile(dict(TASK, task_id=f"T{i}")),
                quality={"feasible": True, "objective": 1.0, "gap": 0.0,
                         "status": "optimal"},
                solver={"name": "highs"},
                method_planned=PER_BOX_METHOD))
        material = h.induction_material(strategy_id="S-plan")
        self.assertTrue(material["count"])
        entry = material["material"][0]
        self.assertEqual(entry["method"]["basis"], "planned_only")
        self.assertIsNotNone(entry["method"]["planned"])
        self.assertIsNone(entry["method"]["actual"])

    def test_performed_method_reports_performed_basis(self):
        h = self.harness()
        for i in range(2):
            h.record(ExecutionRecord(
                execution_id=f"p{i}", task_id=f"T{i}", strategy_id="S-perf",
                profile_snapshot=h.profile(dict(TASK, task_id=f"T{i}")),
                quality={"feasible": True, "objective": 1.0, "gap": 0.0,
                         "status": "optimal"},
                solver={"name": "highs"},
                method_actual=INTEGER_CAR_METHOD))
        material = h.induction_material(strategy_id="S-perf")
        entry = material["material"][0]
        self.assertEqual(entry["method"]["basis"], "performed")
        self.assertIsNotNone(entry["method"]["actual"])


# ---------------------------------------------------------------------------
# A task-check failure is a real failure
# ---------------------------------------------------------------------------

class TestTaskCheckFailureIsAPriorFailure(RepairCase):
    def test_wrong_answer_then_fix_is_recorded(self):
        """The solver reports `optimal` and `feasible`, but the ANSWER is
        wrong (150 where the task needs 200). The check must record `failed`
        on the first and `passed` on the fixed second attempt."""
        h = self.harness()
        self._record(h, "c1", feasible=True, status="optimal",
                     objective=150.0, method=PER_BOX_METHOD,
                     task_check=200.0)   # reference 200 -> observed 150 fails
        first = h.bank.get("c1")
        self.assertEqual(
            first.execution_features["task_check"]["state"], "failed")
        self._record(h, "c2", feasible=True, status="optimal",
                     objective=200.0, method=INTEGER_CAR_METHOD,
                     task_check=200.0)   # reference 200 -> passes
        second = h.bank.get("c2")
        self.assertEqual(
            second.execution_features["task_check"]["state"], "passed")
        # The failed answer IS a prior failure; the passed one is not.
        self.assertEqual(
            [r.execution_id for r in h._prior_failures(second)], ["c1"])

    def test_a_passed_answer_does_not_count_as_prior_failure(self):
        h = self.harness()
        self._record(h, "d1", feasible=True, status="optimal",
                     objective=200.0, method=PER_BOX_METHOD,
                     task_check=200.0)
        self._record(h, "d2", feasible=True, status="optimal",
                     objective=200.0, method=INTEGER_CAR_METHOD,
                     task_check=200.0)
        prior = h._prior_failures(h.bank.get("d2"))
        self.assertEqual(prior, [])


# ---------------------------------------------------------------------------
# Excluded executions do not count as evidence
# ---------------------------------------------------------------------------

class TestExcludedExecutionLeavesEvidence(RepairCase):
    def test_excluded_attempt_drops_out_of_the_material(self):
        h = self.harness()
        for i in range(2):
            self._record(h, f"x{i}", feasible=True, status="optimal",
                         objective=10.0, method=INTEGER_CAR_METHOD,
                         task_id=f"X{i}", strategy_id="S-x")
        h.exclude_execution("x0", reason="wrong fact")
        material = h.induction_material(strategy_id="S-x")
        ids = {m["execution_id"] for m in material["material"]}
        self.assertNotIn("x0", ids)
        self.assertIn("x1", ids)


# ---------------------------------------------------------------------------
# normalize_method is idempotent
# ---------------------------------------------------------------------------

class TestNormalizeMethodIdempotent(unittest.TestCase):
    def test_a_round_trip_through_a_record_is_stable(self):
        """An unknown key stays at ONE ``extra`` level throughout the record
        round trip (to_dict -> from_dict, three times)."""
        record = ExecutionRecord(
            execution_id="r1", task_id="t1", strategy_id="S1",
            profile_snapshot=self._profile(),
            quality={"feasible": True, "objective": 1.0, "gap": 0.0,
                     "status": "optimal"},
            method_actual={"name": "m", "steps": ["s"],
                           "artifacts": {"ref": "solve.py"}})
        first = ExecutionRecord.from_dict(record.to_dict())
        second = ExecutionRecord.from_dict(first.to_dict())
        third = ExecutionRecord.from_dict(second.to_dict())
        self.assertEqual(first.method_actual, second.method_actual)
        self.assertEqual(second.method_actual, third.method_actual)
        self.assertEqual(
            first.method_actual["extra"]["artifacts"], {"ref": "solve.py"})
        self.assertNotIn("extra", first.method_actual["extra"])

    def _profile(self):
        from or_harness.core.schema import ProblemProfile
        return ProblemProfile(
            problem_id="t1", family="routing",
            scale_features={"n_vars": 1.0, "n_constraints": 1.0,
                            "n_int_vars": 1.0, "density": 0.01},
            semantic_coupling=0.5, resource_coupling=0.5,
            temporal_coupling=0.5, route_complexity=0.5)


if __name__ == "__main__":
    unittest.main()
