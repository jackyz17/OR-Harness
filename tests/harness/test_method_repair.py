"""Regression tests for the eight reviewed defects (2026-09-29).

Each test pins ONE of the defects against the transport-modeling story: a
first model that is solved wrongly (per-box cost, answer costs 150 where the
task needs 200), then fixed by adding integer vehicle variables, then
abstracted into a reusable technique.

Run with the same conventions as the rest of the suite:
    PYTHONPATH=src python -m unittest tests.harness.test_method_repair -q
"""
import io
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tests.harness.helpers import HarnessTestCase  # noqa: E402

from or_harness.api import ORHarness  # noqa: E402
from or_harness.core.schema import ExecutionRecord  # noqa: E402
from or_harness.strategy.triggers import evidence_execution_ids  # noqa: E402
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
                method=None, task_check=None, solver="highs",
                task_id="transport", strategy_id="S-transport",
                created_offset=None):
        rec = ExecutionRecord(
            execution_id=execution_id, task_id=task_id,
            strategy_id=strategy_id,
            profile_snapshot=h.profile(dict(TASK, task_id=task_id)),
            quality={"feasible": feasible, "objective": objective,
                     "gap": 0.0, "status": status},
            solver={"name": solver},
            method_actual=method)
        if created_offset is not None:
            rec.created_at = 1_000_000.0 + created_offset
        h.record(rec)
        if task_check is not None:
            h.check_task_result(execution_id,
                                {"reference_objective": task_check})
        return rec


# ---------------------------------------------------------------------------
# #2 a plan is not an intervention
# ---------------------------------------------------------------------------

class TestPlanIsNotAnIntervention(RepairCase):
    def test_plan_only_change_does_not_fire_recovery(self):
        """Two attempts, NO performed method on either, only the plan
        differs: the detector must stay silent. Treating a plan change as a
        fix manufactured recoveries out of intent alone."""
        h = self.harness()
        h.record(ExecutionRecord(
            execution_id="a1", task_id="transport", strategy_id="S1",
            profile_snapshot=h.profile(TASK),
            quality={"feasible": False, "objective": 0.0, "gap": 0.0,
                     "status": "error"},
            solver={"name": "highs"},
            method_planned=PER_BOX_METHOD))
        h.record(ExecutionRecord(
            execution_id="a2", task_id="transport", strategy_id="S1",
            profile_snapshot=h.profile(TASK),
            quality={"feasible": True, "objective": 200.0, "gap": 0.0,
                     "status": "optimal"},
            solver={"name": "highs"},
            method_planned=INTEGER_CAR_METHOD))
        hints = (h.bank.get("a2").execution_features
                 .get("induction_hints") or [])
        self.assertFalse(
            [x for x in hints if x["pattern"] == "intervention_recovery"])

    def test_material_state_rejects_planned_only_evidence(self):
        """A candidate whose evidence reports only PLANNED methods is
        ``insufficient``: a plan is intent, not a performed method."""
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
        state = material["material"][0]["material_state"]
        self.assertEqual(state["state"], "insufficient")
        self.assertIn("PLANNED", state["reason"])

    # The POSITIVE counterpart — BOTH attempts report a performed method, so
    # the difference DOES fire, with basis='performed_method' — lives in
    # test_triggers.TestInterventionRecovery.test_same_solver_modeling_fix_fires,
    # which also covers the declared-intervention path. It is not repeated here.


# ---------------------------------------------------------------------------
# #3 a task-check failure is a real failure
# ---------------------------------------------------------------------------

class TestTaskCheckFailureIsAPriorFailure(RepairCase):
    def test_wrong_answer_then_fix_is_detected(self):
        """The solver reports `optimal` and `feasible`, but the ANSWER is
        wrong (150 where the task needs 200). The fix keeps the solver and
        adds integer vehicle variables, and the new answer passes. The
        recovery must be detected even though nothing was ever infeasible."""
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
        prior = h._prior_failures(second)
        self.assertEqual([r.execution_id for r in prior], ["c1"])
        hints = second.execution_features.get("induction_hints") or []
        recovery = [x for x in hints
                    if x["pattern"] == "intervention_recovery"]
        self.assertTrue(recovery, "the modeling repair must be detected")

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

    def test_a_failed_answer_cannot_evidence_a_recovery(self):
        """A 'success' whose task check FAILED is not a success: the OFFLINE
        candidate is withdrawn once the check refutes the premise (the
        online hint fired at record time, before the check existed)."""
        h = self.harness()
        self._record(h, "e1", feasible=False, status="error",
                     objective=0.0, method=PER_BOX_METHOD)
        self._record(h, "e2", feasible=True, status="optimal",
                     objective=150.0, method=INTEGER_CAR_METHOD,
                     task_check=200.0)   # still wrong
        self.assertEqual(
            h.bank.get("e2").execution_features["task_check"]["state"],
            "failed")
        candidates = h.induction_candidates()
        unavailable = [b for b in candidates
                       if b["kind"] == "pattern_unavailable"]
        self.assertTrue(unavailable,
                        "the refuted recovery premise must be withdrawn")
        self.assertIn("task check", unavailable[0]["material_problem"])
        live = [b for b in candidates if b["kind"] == "pattern"]
        self.assertFalse([b for b in live
                          if b["pattern"] == "intervention_recovery"])


# ---------------------------------------------------------------------------
# #4 advantage_reversal evidence reaches an offline candidate
# ---------------------------------------------------------------------------

class TestAdvantageReversalReachesOffline(unittest.TestCase):
    def test_nested_cell_ids_are_extracted(self):
        ev = {"kind": "advantage_reversal", "family": "routing",
              "advantageous_cell": {"execution_ids": ["a1", "a2"]},
              "adverse_cell": {"execution_ids": ["a3", "a4"]}}
        self.assertEqual(evidence_execution_ids(ev), ["a1", "a2", "a3", "a4"])

    def test_every_detector_shape_is_covered(self):
        # contrast / reproduction: a {strategy: [ids]} map
        self.assertEqual(
            evidence_execution_ids({"execution_ids": {"S1": ["x1"],
                                                      "S2": ["x2"]}}),
            ["x1", "x2"])
        # within-execution: a flat id
        self.assertEqual(evidence_execution_ids({"execution_id": "s"}),
                         ["s"])
        # advantage reversal: the two nested cells
        self.assertEqual(
            evidence_execution_ids({
                "advantageous_cell": {"execution_ids": ["a"]},
                "adverse_cell": {"execution_ids": ["b"]}}),
            ["a", "b"])


# ---------------------------------------------------------------------------
# #5 an excluded side collapses the candidate, it does not survive
# ---------------------------------------------------------------------------

class TestExcludedSideCollapsesTheCandidate(RepairCase):
    def test_losing_a_whole_side_yields_unavailable(self):
        h = self.harness()
        h.record(ExecutionRecord(
            execution_id="f1", task_id="transport", strategy_id="S1",
            profile_snapshot=h.profile(TASK),
            quality={"feasible": False, "objective": 0.0, "gap": 0.0,
                     "status": "error"},
            solver={"name": "highs"},
            method_actual=PER_BOX_METHOD))
        h.record(ExecutionRecord(
            execution_id="f2", task_id="transport", strategy_id="S1",
            profile_snapshot=h.profile(TASK),
            quality={"feasible": True, "objective": 200.0, "gap": 0.0,
                     "status": "optimal"},
            solver={"name": "highs"},
            method_actual=INTEGER_CAR_METHOD))
        self.assertTrue([x for x in
                         (h.bank.get("f2").execution_features
                          .get("induction_hints") or [])
                         if x["pattern"] == "intervention_recovery"])
        # Exclude the FAILED side: the recovery chain loses its "before".
        h.exclude_execution("f1", reason="repro")
        candidates = h.induction_candidates()
        unavailable = [b for b in candidates
                       if b["kind"] == "pattern_unavailable"]
        self.assertTrue(unavailable,
                        "a candidate whose side was excluded must be "
                        "reported as unavailable")
        self.assertTrue(unavailable[0]["material_problem"])
        self.assertIn("failed", unavailable[0]["material_problem"])
        state = h.induction_material(
            bundle_id=unavailable[0]["bundle_id"])["material"][0][
                "material_state"]
        self.assertEqual(state["state"], "unavailable")
        # And NO live candidate carries the dead id.
        for bundle in candidates:
            self.assertNotIn("f1", bundle["execution_ids"])

    def test_losing_one_of_several_on_a_side_keeps_the_candidate(self):
        """A contrast that loses ONE record from a side of >=3 still holds,
        with the dead id pruned from its evidence — the side is not gone."""
        h = self.harness()
        for i in range(3):
            h.record(ExecutionRecord(
                execution_id=f"gA{i}", task_id=f"ga{i}", strategy_id="SA",
                profile_snapshot=h.profile(dict(TASK, task_id=f"ga{i}")),
                quality={"feasible": True, "objective": 100.0,
                         "gap": 0.6, "status": "optimal"},
                solver={"name": "highs"},
                method_actual={"name": "method a", "steps": [f"sa{i}"]}))
        for i in range(2):
            h.record(ExecutionRecord(
                execution_id=f"gB{i}", task_id=f"gb{i}", strategy_id="SB",
                profile_snapshot=h.profile(dict(TASK, task_id=f"gb{i}")),
                quality={"feasible": True, "objective": 100.0,
                         "gap": 0.0, "status": "optimal"},
                solver={"name": "highs"},
                method_actual={"name": "method b", "steps": [f"sb{i}"]}))
        h.exclude_execution("gA0", reason="repro")
        candidates = [b for b in h.induction_candidates()
                      if b["kind"] == "pattern"]
        contrast = [b for b in candidates
                    if b["pattern"] == "strategy_contrast"]
        self.assertTrue(contrast)
        self.assertNotIn("gA0", contrast[0]["execution_ids"])
        self.assertIn("gA1", contrast[0]["execution_ids"])
        self.assertTrue(any("pruned" in r
                            for r in contrast[0]["trigger_reasons"]))


# ---------------------------------------------------------------------------
# #8 normalize_method is idempotent
# ---------------------------------------------------------------------------

class TestNormalizeMethodIdempotent(unittest.TestCase):
    def test_a_round_trip_through_a_record_is_stable(self):
        """#8: normalizing an ALREADY-normalized method must be a no-op.

        Verified through the real record path (to_dict -> from_dict, three
        times) because that is where the nesting happened; an unknown key
        stays at ONE ``extra`` level throughout."""
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
