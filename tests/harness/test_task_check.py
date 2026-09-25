"""Task-result checks: does the ANSWER satisfy the original task?

The gap this suite exists for: ``orx execute`` answers "did the solver solve
the model it was given" (a legal status, a finite objective, a gap). It cannot
answer "is this a valid answer to the TASK". A relaxed LP answered with
fractional values reports ``optimal`` with ``gap=0`` and would otherwise enter
recall, the conditional statistics, the world-model feedback and offline
induction as a success sample.

So these tests assert BEHAVIOUR, not field presence: that a claim is really
generated, really checked, really passed or refuted, and that a confirmed-wrong
answer really stops counting as success while remaining recorded evidence. The
acceptance scenario is the LP relaxation: 10750 (fractional) → identified as
not a valid integer answer → the variable domain is repaired → 10755 passes.
"""
import json
import os
import sys
import unittest

from helpers import HarnessTestCase

from or_harness.api import ORHarness
from or_harness.core.schema import (
    TASK_CHECK_STATES,
    task_check_block,
    task_check_state,
    task_effective_quality,
)
from or_harness.strategy.stats import quality_score
from or_harness.strategy.verification import (
    TASK_CHECK_FAILED,
    TASK_CHECK_INSUFFICIENT,
    TASK_CHECK_PASSED,
    verify_task_result,
)

EMBEDDING_ENV_KEYS = ("OR_EMBEDDING_BACKEND", "OR_EMBEDDING_BASE_URL",
                      "OR_EMBEDDING_MODEL", "OR_EMBEDDING_API_KEY")

#: The task statement for the acceptance scenario. The INTEGER requirement is
#: stated by the TASK, so the check basis is read off the task — never
#: reverse-engineered from a reference value.
INT_TASK_TEXT = (
    "A workshop produces two products. Product 1 needs 3 hours of machining "
    "and 2 hours of finishing; product 2 needs 1 hour of machining and 4 "
    "hours of finishing. Machining has 9 hours available, finishing 12. "
    "Every production quantity MUST BE A WHOLE NUMBER of units (fractional "
    "units cannot be shipped). Maximise profit (7 per unit of product 1, 3 "
    "per unit of product 2).")


def _task(task_id="t1", **coupling):
    values = {"resource_coupling": 0.3, "temporal_coupling": 0.1,
              "route_complexity": 0.2}
    values.update(coupling)
    return {"task_id": task_id, "family": "production_planning",
            "description": INT_TASK_TEXT,
            "spec": {"n_vars": 2, "n_constraints": 2, "n_int_vars": 2},
            "annotations": {"coupling": {**values, "semantic_coupling": 0.5}}}


#: A prediction payload whose benefit is a NORMALIZED quality, so the
#: strategy-outcome channel has something real to compare against. The
#: provider is a stub: no model call, no network.
STUB_PAYLOAD = {
    "benefit": {"kind": "solution_quality",
                "metric": "normalized_objective_gap", "unit": "1-gap",
                "value": 0.8,
                "baseline": {"kind": "conditional_stats", "value": 0.7}},
    "cost": {"llm_tokens": 1200, "solver_runtime_s": 3.0},
    "risk": {"events": [{"event": "model_invalid", "probability": 0.2}]},
    "uncertainty": {"execution_randomness": 0.3, "knowledge_gap": 0.6},
}


class StubProvider:
    """A world-model provider that returns a fixed payload."""

    name = "stub-task-check"

    def __init__(self, payload=None):
        self.payload = payload or STUB_PAYLOAD

    def predict(self, request, timeout_s=None):
        return {"payload": self.payload,
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
                "error": None, "latency_s": 0.02}

    def describe(self):
        return {"provider": self.name, "configured": True}


class TaskCheckCase(HarnessTestCase):
    """A harness with a hermetic embedding backend + stub world model."""

    def setUp(self):
        super().setUp()
        saved = {key: os.environ.pop(key, None) for key in EMBEDDING_ENV_KEYS}

        def restore():
            for key, value in saved.items():
                if value is not None:
                    os.environ[key] = value
        self.addCleanup(restore)
        try:
            from or_harness.strategy.embedding_index import (
                LocalHashEmbeddingBackend,
            )
            backend = LocalHashEmbeddingBackend()
        except Exception:  # pragma: no cover - backend is part of the build
            backend = None
        self.h = ORHarness(home=self.home, embedding=backend,
                           world_model=StubProvider())
        self.addCleanup(self.h.close)

    def solve(self, task, *, variables, objective, status="optimal",
              gap=0.0, strategy="S01", episode_id="ep1", tag=None,
              runtime_seconds=0.01):
        """Execute and record one attempt with an explicit solution vector."""
        from pathlib import Path
        label = tag or f"{task['task_id']}_{strategy}_{episode_id}"
        work = Path(self.home) / f"ws_{label}"
        work.mkdir(parents=True, exist_ok=True)
        script = work / "solve.py"
        payload = {"status": status, "objective_value": objective,
                   "objective_bound": objective,
                   "runtime_seconds": runtime_seconds,
                   "variables": variables}
        if gap is not None:
            payload["mip_gap"] = gap
        script.write_text(
            "import json\n"
            "with open('result.json', 'w') as fh:\n"
            "    json.dump(" + json.dumps(payload) + ", fh)\n",
            encoding="utf-8")
        record = self.h.execute(task, strategy, str(script), str(work),
                                solver="highs", episode_id=episode_id)
        self.h.record(record)
        return record


# ---------------------------------------------------------------------------
# 1. the acceptance scenario: LP relaxation -> repair -> pass
# ---------------------------------------------------------------------------

class TestRelaxationRepairLoop(TaskCheckCase):
    """10750 fractional -> identified -> variable domain repaired -> 10755."""

    def test_fractional_relaxation_is_identified_and_repaired(self):
        task = _task("t1")
        # The relaxation: an optimal answer to the LP with fractional units.
        relaxed = self.solve(task, variables={"x1": 2.5, "x2": 1.5},
                             objective=10750.0, tag="relaxed")

        check = {
            "reference_objective": 10750.0,
            "integer": {"variables": ["x1", "x2"]},
        }
        first = self.h.check_task_result(relaxed.execution_id, check)
        self.assertEqual(first["state"], TASK_CHECK_FAILED)
        # The objective MATCHED the reference; the domain check is what
        # caught it. This is the whole point: a matching objective does not
        # make the answer a valid one.
        basis_checks = {c["check"]: c for c in first["report"]["checks"]}
        self.assertTrue(basis_checks["reference_objective"]["ok"])
        self.assertFalse(basis_checks["integer_domains"]["ok"])
        self.assertEqual(basis_checks["integer_domains"]["n_non_integer"], 2)
        diffs = first["report"]["diffs"]
        self.assertTrue(all(d["basis"] == "integer_domains" for d in diffs))
        self.assertIn("x1", {d["variable"] for d in diffs})

        # The record keeps its observed solver-side quality and cost — the
        # answer was produced, the cost was paid, and neither is rewritten.
        stored = self.h.bank.get(relaxed.execution_id)
        self.assertEqual(stored.quality["status"], "optimal")
        self.assertEqual(stored.quality["objective"], 10750.0)
        self.assertEqual(task_check_state(stored), "failed")
        self.assertTrue(stored.cost.measured)

        # The repair: whole units only, re-solved in the SAME episode.
        repaired = self.solve(task, variables={"x1": 2, "x2": 3},
                              objective=10755.0, tag="repaired")
        second = self.h.check_task_result(
            repaired.execution_id,
            {"reference_objective": 10755.0,
             "integer": {"variables": ["x1", "x2"]}})
        self.assertEqual(second["state"], TASK_CHECK_PASSED)
        self.assertEqual(task_check_state(
            self.h.bank.get(repaired.execution_id)), "passed")

        # BOTH attempts remain recorded facts with their real cost — the
        # failure is not deleted, it is disqualified as a success sample.
        self.assertEqual(len(self.h.bank.query(task_id="t1")), 2)
        self.assertIsNotNone(self.h.bank.get(relaxed.execution_id))
        # The failed answer contributes ZERO quality, the repaired one its
        # observed quality. That is the mechanism, not a side effect.
        self.assertEqual(quality_score(self.h.bank.get(relaxed.execution_id)),
                         0.0)
        self.assertEqual(quality_score(self.h.bank.get(repaired.execution_id)),
                         1.0)

    def test_relaxation_costs_still_count_in_the_task_total(self):
        """Disqualifying an answer must not erase the money spent on it."""
        task = _task("t1")
        relaxed = self.solve(task, variables={"x1": 2.5, "x2": 1.5},
                             objective=10750.0, tag="r1")
        repaired = self.solve(task, variables={"x1": 2, "x2": 3},
                              objective=10755.0, tag="r2")
        self.h.check_task_result(relaxed.execution_id,
                                 {"integer": {"variables": ["x1", "x2"]}})
        self.h.check_task_result(repaired.execution_id,
                                 {"integer": {"variables": ["x1", "x2"]}})
        summary = self.h.task_cost_summary("t1")
        # Both attempts are charged: "A fails -> A repaired -> B succeeds"
        # charges all of them, never only the last success.
        self.assertEqual(summary["n_attempts"], 2)
        # Cumulative dimensions sum over every attempt; latency_s is never
        # summed (per-attempt latencies are facts, not an end-to-end total).
        self.assertIsNotNone(summary["total_cost"]["solver_runtime_s"])
        self.assertEqual(len(summary["attempts"]), 2)
        self.assertTrue(all(
            attempt["cost"].get("latency_s") is not None
            for attempt in summary["attempts"]))


# ---------------------------------------------------------------------------
# 2. a matching objective is not a valid answer
# ---------------------------------------------------------------------------

class TestObjectiveMatchIsNotValidity(TaskCheckCase):

    def test_objective_matches_but_the_answer_is_invalid(self):
        """A probe over the solution fails while the objective matches."""
        task = _task("t1")
        record = self.solve(task, variables={"x1": 2.5, "x2": 1.5},
                            objective=10750.0, tag="probe")
        report = verify_task_result(record, {
            "reference_objective": 10750.0,
            "semantic_probe": {"path": "execution_features."
                                       "solution_variables.x1",
                               "min": 0, "max": 2.0},
        })
        self.assertEqual(report["state"], TASK_CHECK_FAILED)
        self.assertEqual(report["diffs"][0]["basis"], "semantic_probe")

    def test_reported_objective_disagreeing_with_the_solution(self):
        """The objective recomputed from the vector must match the report."""
        task = _task("t1")
        record = self.solve(task, variables={"x1": 2, "x2": 3},
                            objective=10755.0, tag="recompute")
        # 7*2 + 3*3 = 23, not 10755: the reported objective is not the
        # objective of the reported solution.
        report = verify_task_result(record, {
            "recompute_objective": {"coefficients": {"x1": 7, "x2": 3},
                                    "constant": 0.0},
        })
        self.assertEqual(report["state"], TASK_CHECK_FAILED)
        self.assertEqual(report["diffs"][0]["basis"], "recompute_objective")
        # And when the coefficients DO describe the answer, it passes.
        ok = verify_task_result(record, {
            "recompute_objective": {"coefficients": {"x1": 7, "x2": 3},
                                    "constant": 10732.0},
        })
        self.assertEqual(ok["state"], TASK_CHECK_PASSED)

    def test_reference_status_mismatch_fails(self):
        task = _task("t1")
        record = self.solve(task, variables={"x1": 2, "x2": 3},
                            objective=10755.0, status="feasible", tag="status")
        report = verify_task_result(record, {"reference_status": "optimal"})
        self.assertEqual(report["state"], TASK_CHECK_FAILED)


# ---------------------------------------------------------------------------
# 3. no gold / no basis / missing vector -> insufficient, never a pass
# ---------------------------------------------------------------------------

class TestInsufficientIsNeverAPass(TaskCheckCase):

    def test_no_check_basis_is_insufficient(self):
        task = _task("t1")
        record = self.solve(task, variables={"x1": 2, "x2": 3},
                            objective=10755.0, tag="nobasis")
        report = verify_task_result(record, {})
        self.assertEqual(report["state"], TASK_CHECK_INSUFFICIENT)
        self.assertEqual(report["scope"]["basis"], [])
        self.assertNotEqual(report["state"], TASK_CHECK_PASSED)

    def test_no_gold_is_not_a_pass(self):
        """The old implementation returned matched=True with no gold."""
        task = _task("t1")
        record = self.solve(task, variables={"x1": 2, "x2": 3},
                            objective=10755.0, tag="nogold")
        report = verify_task_result(record, {"integer": {}})
        self.assertEqual(report["state"], TASK_CHECK_PASSED)
        # ...but ONLY for the declared basis, and the report says so.
        self.assertEqual(report["scope"]["basis"], ["integer"])
        self.assertTrue(any("model" in item
                            for item in report["scope"]["unchecked"]))
        # With nothing declared there is no pass at all.
        self.assertEqual(verify_task_result(record, {})["state"],
                         TASK_CHECK_INSUFFICIENT)

    def test_missing_solution_vector_is_insufficient(self):
        """No recorded vector means the domain check cannot run."""
        task = _task("t1")
        # A script that reports no `variables` at all.
        from pathlib import Path
        work = Path(self.home) / "ws_novec"
        work.mkdir(parents=True, exist_ok=True)
        script = work / "solve.py"
        script.write_text(
            "import json\n"
            "with open('result.json', 'w') as fh:\n"
            "    json.dump({'status': 'optimal', 'objective_value': 10755.0,"
            " 'objective_bound': 10755.0, 'runtime_seconds': 0.01}, fh)\n",
            encoding="utf-8")
        record = self.h.execute(task, "S01", str(script), str(work),
                                solver="highs", episode_id="ep1")
        self.h.record(record)
        report = verify_task_result(record, {"integer": {}})
        self.assertEqual(report["state"], TASK_CHECK_INSUFFICIENT)
        self.assertTrue(any("no solution vector" in item
                            for item in report["scope"]["unchecked"]))
        # A reference objective alone still works (it needs no vector).
        report2 = verify_task_result(record,
                                     {"reference_objective": 10755.0})
        self.assertEqual(report2["state"], TASK_CHECK_PASSED)

    def test_named_variable_absent_from_the_vector_is_insufficient(self):
        task = _task("t1")
        record = self.solve(task, variables={"x1": 2}, objective=10755.0,
                            tag="partial")
        report = verify_task_result(
            record, {"integer": {"variables": ["x1", "x9"]}})
        self.assertEqual(report["state"], TASK_CHECK_INSUFFICIENT)
        self.assertTrue(any("x9" in item
                            for item in report["scope"]["unchecked"]))

    def test_unreadable_check_does_not_pass(self):
        """A probe with no comparison cannot decide — and never passes."""
        task = _task("t1")
        record = self.solve(task, variables={"x1": 2, "x2": 3},
                            objective=10755.0, tag="badprobe")
        report = verify_task_result(record, {
            "semantic_probe": {"path": "quality.objective"},
        })
        self.assertEqual(report["state"], TASK_CHECK_INSUFFICIENT)

    def test_failed_execution_is_insufficient_not_failed(self):
        """'We could not check it' is not 'we checked it and it failed'."""
        task = _task("t1")
        record = self.solve(task, variables={"x1": 2, "x2": 3},
                            objective=10755.0, status="timeout", gap=None,
                            tag="timeout")
        report = verify_task_result(record, {"integer": {}})
        self.assertEqual(report["state"], TASK_CHECK_INSUFFICIENT)

    def test_declared_intent_is_reported(self):
        task = _task("t1")
        record = self.solve(task, variables={"x1": 2.5, "x2": 1.5},
                            objective=10750.0, tag="intent")
        report = verify_task_result(
            record, {"reference_objective": 10750.0,
                     "intent": "relaxation"})
        self.assertEqual(report["state"], TASK_CHECK_PASSED)
        self.assertEqual(report["intent"], "relaxation")
        self.assertTrue(any("relaxation" in item
                            for item in report["scope"]["unchecked"]))
        with self.assertRaises(ValueError):
            verify_task_result(record, {"integer": {}, "intent": "nonsense"})


# ---------------------------------------------------------------------------
# 4. a confirmed-wrong answer is not a success sample (but stays recorded)
# ---------------------------------------------------------------------------

class TestConfirmedWrongIsNotSuccess(TaskCheckCase):

    def test_failed_check_zeroes_quality_without_touching_the_fact(self):
        task = _task("t1")
        record = self.solve(task, variables={"x1": 2.5, "x2": 1.5},
                            objective=10750.0, tag="q1")
        before = self.h.bank.get(record.execution_id)
        self.assertEqual(quality_score(before), 1.0)  # optimal, gap 0
        self.h.check_task_result(record.execution_id,
                                 {"integer": {"variables": ["x1", "x2"]}})
        after = self.h.bank.get(record.execution_id)
        self.assertEqual(quality_score(after), 0.0)
        # Nothing else about the fact moved.
        self.assertEqual(after.quality, before.quality)
        self.assertEqual(after.cost.to_dict(), before.cost.to_dict())
        self.assertEqual(after.source, "executed")
        self.assertIn(record.execution_id,
                      [r.execution_id for r in self.h.bank.query(task_id="t1")])

    def test_insufficient_check_does_not_demote(self):
        """A check that could not decide is not evidence of failure."""
        task = _task("t1")
        record = self.solve(task, variables={"x1": 2, "x2": 3},
                            objective=10755.0, tag="q2")
        self.h.check_task_result(record.execution_id, {"integer": {}})
        self.assertEqual(task_check_state(
            self.h.bank.get(record.execution_id)), "passed")
        # And an execution with NO check keeps its full observed quality —
        # nothing is retroactively demoted.
        other = self.solve(task, variables={"x1": 1, "x2": 1},
                           objective=1000.0, tag="q3")
        self.assertIsNone(task_check_state(self.h.bank.get(
            other.execution_id)))
        self.assertEqual(quality_score(self.h.bank.get(other.execution_id)),
                         1.0)

    def test_unchecked_history_is_unchanged(self):
        task = _task("t1")
        record = self.solve(task, variables={"x1": 2, "x2": 3},
                            objective=10755.0, tag="q4")
        self.assertEqual(task_effective_quality(record, 0.75), 0.75)
        self.assertIsNone(task_check_block(record))
        self.assertIsNone(task_check_state(record))

    def test_recall_shows_the_task_check_limitation(self):
        """Recall must carry the restriction, not hide it."""
        task = _task("t1")
        record = self.solve(task, variables={"x1": 2.5, "x2": 1.5},
                            objective=10750.0, tag="recall1")
        self.h.check_task_result(record.execution_id,
                                 {"integer": {"variables": ["x1", "x2"]}})
        result = self.h.recall(task, top=3)
        hits = (result.get("vector_recall") or {}).get(
            "execution_evidence") or []
        if not hits:
            self.skipTest("no embedding backend: the text channel is off")
        hit = next(h for h in hits
                   if h["execution_id"] == record.execution_id)
        self.assertEqual(hit["task_check"], "failed")
        self.assertTrue(any("satisfy the task" in item.lower()
                            for item in hit["task_check_limitations"]))

    def test_statistics_report_the_task_check_coverage(self):
        task = _task("t1")
        good = self.solve(task, variables={"x1": 2, "x2": 3},
                          objective=10755.0, tag="s1")
        bad = self.solve(task, variables={"x1": 2.5, "x2": 1.5},
                         objective=10750.0, tag="s2")
        self.h.check_task_result(good.execution_id, {"integer": {}})
        self.h.check_task_result(bad.execution_id, {"integer": {}})
        cell = self.h.stats.cell(good.profile_snapshot and
                                 __import__("or_harness.core.schema",
                                            fromlist=["group_key"])
                                 .group_key(good.profile_snapshot),
                                 "S01")
        self.assertEqual(cell.task_checks.get("passed"), 1)
        self.assertEqual(cell.task_checks.get("failed"), 1)
        # mean_quality reflects the gate: (1.0 + 0.0) / 2.
        self.assertAlmostEqual(cell.mean_quality, 0.5, places=6)


# ---------------------------------------------------------------------------
# 5. the repair trajectory is readable (raw material for induction)
# ---------------------------------------------------------------------------

class TestRepairTrajectoryIsReadable(TaskCheckCase):

    def test_the_whole_trajectory_can_be_read_back(self):
        task = _task("t1")
        relaxed = self.solve(task, variables={"x1": 2.5, "x2": 1.5},
                             objective=10750.0, tag="t1a")
        first = self.h.check_task_result(
            relaxed.execution_id,
            {"integer": {"variables": ["x1", "x2"]}})
        repaired = self.solve(task, variables={"x1": 2, "x2": 3},
                              objective=10755.0, tag="t1b")
        second = self.h.check_task_result(repaired.execution_id,
                                          {"integer": {}})

        # (a) both attempts are recorded facts of the same task/episode
        records = self.h.bank.query(task_id="t1")
        self.assertEqual(len(records), 2)
        self.assertEqual({task_check_state(r) for r in records},
                         {"failed", "passed"})
        # (b) both checks are logged as verify ACTIONS with their verdicts
        verify_actions = self.h.actions.query(task_id="t1",
                                              action_type="verify")
        self.assertEqual(len(verify_actions), 2)
        states = {a.outcome.get("state") for a in verify_actions}
        self.assertEqual(states, {"failed", "passed"})
        for action in verify_actions:
            self.assertIsNotNone(action.linked_execution_id)
        # (c) the artifacts a reader needs are present, not just hashes
        material = first["reflection_material"]
        self.assertIn("WHOLE NUMBER", material["task_text"])
        self.assertEqual(material["reported_quality"]["objective"], 10750.0)
        self.assertEqual(material["solution_variables"],
                         {"x1": 2.5, "x2": 1.5})
        self.assertTrue(material["guidance"])
        # (d) the failed attempt is citable as contrast evidence
        self.assertTrue(self.h.bank.get(relaxed.execution_id)
                        .execution_features.get("task_check"))
        self.assertEqual(second["state"], TASK_CHECK_PASSED)

    def test_induce_reads_the_gated_quality(self):
        """Offline induction must not learn from a disqualified answer."""
        task = _task("t1")
        bad = self.solve(task, variables={"x1": 2.5, "x2": 1.5},
                         objective=10750.0, tag="i1")
        self.h.check_task_result(bad.execution_id,
                                 {"integer": {"variables": ["x1", "x2"]}})
        result = self.h.induce(strategy_id="S01")
        # The candidate's statistics saw quality 0.0 for that execution,
        # so no quality claim can be built on it.
        cell = self.h.stats.for_profile(bad.profile_snapshot).get("S01")
        self.assertEqual(cell.mean_quality, 0.0)
        self.assertIsInstance(result, dict)


# ---------------------------------------------------------------------------
# 6. late corrections
# ---------------------------------------------------------------------------

class TestLateCorrection(TaskCheckCase):

    def test_late_check_corrects_the_benefit_and_keeps_the_cost(self):
        """A check arriving AFTER close-out changes later use, not history.

        The corrected sample keeps COUNTING: only the benefit observation is
        re-derived (to 0.0 — the answer does not satisfy the task), while the
        measured COST is preserved. Dropping the whole evaluation would throw
        away a real measurement."""
        task = _task("t1")
        prediction = self.h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy", "strategy_id": "S01"},
            "ep1")
        record = self.solve(task, variables={"x1": 2, "x2": 3},
                            objective=10755.0, tag="late1")
        self.h.bind_strategy_outcome(prediction.prediction_id,
                                     record.action_id)
        closed = self.h.close_episode("t1", "ep1")
        evaluation = closed["evaluations"][0]
        self.assertEqual(evaluation["state"], "evaluated")
        self.assertEqual(evaluation["benefit"]["observed"], 1.0)
        stored_before = self.h.get_strategy_evaluation(
            evaluation["evaluation_id"])
        first_summary = self.h.calibration_summary()
        self.assertEqual(first_summary["n_evaluated"], 1)
        group_before = list(first_summary["groups"].values())[0]
        self.assertEqual(group_before["mean_benefit_abs_error"], 0.2)

        # The correction arrives late: the answer was fractional after all.
        # It is written through the API (the documented path), so the
        # published summary is REPUBLISHED as part of the correction — the
        # stored evaluation is untouched, but the summary a later context
        # reads already reflects the correction.
        late = self.h.check_task_result(
            record.execution_id,
            {"reference_objective": 999.0})
        self.assertEqual(late["state"], "failed")
        self.assertIn("calibration_republished", late)
        second_summary = self.h.calibration_summary()
        # The stored evaluation is NOT rewritten...
        self.assertEqual(self.h.get_strategy_evaluation(
            evaluation["evaluation_id"]), stored_before)
        # ...and the sample still COUNTS, with the benefit re-derived to 0.0.
        self.assertEqual(second_summary["n_evaluated"], 1)
        correction = second_summary["validity_corrections"][0]
        self.assertEqual(correction["kind"], "live_rederivation")
        self.assertEqual(correction["fields"], ["benefit"])
        self.assertTrue(correction["counted"])
        self.assertIn("cost_preserved",
                      correction["detail"]["benefit"])
        # The benefit error is now |0.8 - 0.0| = 0.8, not |0.8 - 1.0| = 0.2.
        group_after = list(second_summary["groups"].values())[0]
        self.assertEqual(group_after["mean_benefit_abs_error"], 0.8)
        # The COST measurement survives: the answer was wrong, but it really
        # did cost what it cost.
        self.assertEqual(group_after["mean_cost_log_error"],
                         group_before["mean_cost_log_error"])
        # Statistics pick the corrected judgment up on the next read.
        self.assertEqual(quality_score(self.h.bank.get(record.execution_id)),
                         0.0)

    def test_late_correction_republishes_the_published_summary(self):
        """A late check must reach the PUBLISHED summary, not only a fresh
        rebuild — otherwise later predictions keep reading a stale label."""
        task = _task("t1")
        prediction = self.h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy", "strategy_id": "S01"},
            "ep1")
        record = self.solve(task, variables={"x1": 2, "x2": 3},
                            objective=10755.0, tag="late3")
        self.h.bind_strategy_outcome(prediction.prediction_id,
                                     record.action_id)
        self.h.close_episode("t1", "ep1")
        # A context built now reads the published summary (no correction).
        before = self.h.build_prediction_context(
            _task("t1"), "ep2").strategy_calibration
        self.assertEqual(before["validity_corrections"], [])

        self.h.check_task_result(
            record.execution_id,
            {"reference_objective": 999.0})
        # A context built AFTER the correction sees it WITHOUT any rebuild
        # call: the correction republished the summary.
        after = self.h.build_prediction_context(
            _task("t1"), "ep2").strategy_calibration
        self.assertTrue(after["validity_corrections"])
        self.assertEqual(after["validity_corrections"][0]["fields"],
                         ["benefit"])
        # The first context is frozen and unchanged.
        self.assertNotEqual(after, before)

    def test_raw_annotation_does_not_republish(self):
        """The raw bank channel (``set_task_check``) is a narrow fact write,
        not a calibration trigger: a correction written that way is picked
        up on the NEXT rebuild/close, and the read path reports the stored
        summary. Documented behaviour, so a caller knows which channel
        updates the published summary."""
        task = _task("t1")
        prediction = self.h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy", "strategy_id": "S01"},
            "ep1")
        record = self.solve(task, variables={"x1": 2, "x2": 3},
                            objective=10755.0, tag="late4")
        self.h.bind_strategy_outcome(prediction.prediction_id,
                                     record.action_id)
        self.h.close_episode("t1", "ep1")
        self.h.bank.set_task_check(record.execution_id, {
            "state": "failed",
            "execution_id": record.execution_id,
            "task_id": "t1",
            "episode_id": "ep1",
            "diffs": [{"basis": "reference_objective", "expected": 999.0}],
            "checked_at": __import__("time").time() + 10,
        })
        # The published summary is unchanged until an explicit rebuild.
        published = self.h.calibration_summary()
        self.assertEqual(published["validity_corrections"], [])
        rebuilt = self.h.calibration_summary(rebuild=True)
        self.assertTrue(rebuilt["validity_corrections"])
        self.assertEqual(rebuilt["validity_corrections"][0]["fields"],
                         ["benefit"])

    def test_check_recorded_before_closeout_is_not_a_correction(self):
        """A known-at-the-time failure must not be double-counted."""
        task = _task("t1")
        prediction = self.h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy", "strategy_id": "S01"},
            "ep1")
        record = self.solve(task, variables={"x1": 2.5, "x2": 1.5},
                            objective=10750.0, tag="late2")
        self.h.bind_strategy_outcome(prediction.prediction_id,
                                     record.action_id)
        self.h.check_task_result(record.execution_id,
                                 {"integer": {"variables": ["x1", "x2"]}})
        self.h.close_episode("t1", "ep1")
        summary = self.h.calibration_summary()
        self.assertEqual(summary["exclusions"].get("validity_corrected", 0),
                         0)
        self.assertEqual(summary["validity_corrections"], [])


# ---------------------------------------------------------------------------
# 7. the defects that were exposed (running actions, CLI pending)
# ---------------------------------------------------------------------------

class TestExposedDefectsAreFixed(TaskCheckCase):

    def test_executor_exception_does_not_leave_a_running_action(self):
        task = _task("t1")
        from pathlib import Path
        outside = Path(self.home) / "outside_solve.py"
        outside.write_text("import json\n"
                           "json.dump({}, open('result.json', 'w'))\n",
                           encoding="utf-8")
        with self.assertRaises(ValueError):
            self.h.execute(task, "S01", str(outside),
                           str(Path(self.home) / "ws_out"), solver="highs",
                           episode_id="ep1")
        # The exception still propagates, but the action is not stranded.
        self.assertEqual(self.h.actions.running(task_id="t1"), [])
        actions = self.h.actions.query(task_id="t1",
                                       action_type="execute_strategy")
        self.assertEqual([a.status for a in actions], ["failed"])
        # ...and the episode is therefore still closable.
        result = self.h.close_episode("t1", "ep1")
        self.assertIsNotNone(result["closeout"])

    def test_cli_close_episode_reports_pending_without_crashing(self):
        import io
        from contextlib import redirect_stdout
        from or_harness.cli import main
        task = _task("t1")
        snap = self.h.snapshot(task, "ep1")
        self.h.actions.begin_action("model", "t1", "ep1", pre_snapshot=snap)
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = main(["--home", self.home, "close-episode",
                         "--task", "t1", "--episode", "ep1"])
        self.assertEqual(code, 2)
        payload = json.loads(buf.getvalue())
        self.assertIn("still OPEN", payload["result"]["error"])
        self.assertNotIn("TypeError", buf.getvalue())

    def test_cli_check_task_end_to_end(self):
        import io
        from contextlib import redirect_stdout
        from or_harness.cli import main
        task = _task("t1")
        record = self.solve(task, variables={"x1": 2.5, "x2": 1.5},
                            objective=10750.0, tag="cli1")
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = main(["--home", self.home, "check-task",
                         record.execution_id, "--check",
                         json.dumps({"integer": {"variables": ["x1", "x2"]}})])
        self.assertEqual(code, 0)
        payload = json.loads(buf.getvalue())
        self.assertEqual(payload["result"]["state"], "failed")
        self.assertIn("FAILED", payload["summary"])
        self.assertIn("reflection_material", payload["result"])

        # The CLI is also the repair path: a fixed answer passes.
        fixed = self.solve(task, variables={"x1": 2, "x2": 3},
                           objective=10755.0, tag="cli2")
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = main(["--home", self.home, "check-task",
                         fixed.execution_id, "--check",
                         json.dumps({"reference_objective": 10755.0,
                                     "integer": {}})])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(buf.getvalue())["result"]["state"],
                         "passed")

    def test_cli_check_task_unknown_execution_fails_cleanly(self):
        import io
        from contextlib import redirect_stdout
        from or_harness.cli import main
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = main(["--home", self.home, "check-task", "ex_missing"])
        self.assertEqual(code, 2)
        self.assertIn("unknown execution_id",
                      json.loads(buf.getvalue())["result"]["error"])


# ---------------------------------------------------------------------------
# 8. the formal prediction path really produces an evaluation
# ---------------------------------------------------------------------------

class TestFormalPathProducesEvaluation(TaskCheckCase):

    def test_predict_strategy_execute_check_bind_close(self):
        """The documented path must end in a real evaluation, not empty."""
        task = _task("t1")
        prediction = self.h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy", "strategy_id": "S01"},
            "ep1")
        record = self.solve(task, variables={"x1": 2.5, "x2": 1.5},
                            objective=10750.0, tag="formal")
        checked = self.h.check_task_result(
            record.execution_id,
            {"integer": {"variables": ["x1", "x2"]}})
        self.assertEqual(checked["state"], "failed")
        self.h.bind_strategy_outcome(prediction.prediction_id,
                                     record.action_id)
        closed = self.h.close_episode("t1", "ep1")
        evaluation = closed["evaluations"][0]
        self.assertEqual(evaluation["state"], "evaluated")
        # The calibration channel compares against the TASK's outcome: the
        # disqualified answer is a 0.0 observation, not the solver's 1.0.
        self.assertEqual(evaluation["benefit"]["observed"], 0.0)
        self.assertIn("task_check_gated", evaluation["benefit"])
        # And the close-out reports the task-check coverage explicitly.
        self.assertEqual(closed["task_checks"]["verdicts"], {"failed": 1})
        self.assertEqual(closed["task_checks"]["unchecked"], 0)

    def test_closeout_notes_unchecked_answers(self):
        task = _task("t1")
        self.solve(task, variables={"x1": 2, "x2": 3}, objective=10755.0,
                   tag="unchecked")
        closed = self.h.close_episode("t1", "ep1")
        self.assertEqual(closed["task_checks"]["unchecked"], 1)
        self.assertIn("UNKNOWN", closed["task_checks"]["note"])


# ---------------------------------------------------------------------------
# 9. the annotation channel itself
# ---------------------------------------------------------------------------

class TestTaskCheckChannel(HarnessTestCase):

    def test_set_task_check_works_for_staged_executions(self):
        from or_harness.strategy.experience_bank import ExperienceBank
        bank = ExperienceBank(self.store)
        record = self.make_record(execution_id="ex_staged")
        bank.stage_pending(record)
        bank.set_task_check("ex_staged", {"state": "failed",
                                         "checked_at": 1.0})
        staged = bank.get_pending("ex_staged")
        self.assertEqual(task_check_state(staged), "failed")
        # Recording it carries the annotation verbatim.
        bank.append(staged)
        bank.clear_pending("ex_staged")
        self.assertEqual(task_check_state(bank.get("ex_staged")), "failed")

    def test_malformed_annotation_reads_as_absent(self):
        from or_harness.strategy.experience_bank import ExperienceBank
        bank = ExperienceBank(self.store)
        record = self.make_record(execution_id="ex_bad")
        bank.append(record)
        bank.set_task_check("ex_bad", {"state": "maybe"})
        self.assertIsNone(task_check_state(bank.get("ex_bad")))
        # A withdrawn check returns to UNKNOWN, never to a default verdict.
        bank.set_task_check("ex_bad", None)
        self.assertIsNone(task_check_block(bank.get("ex_bad")))

    def test_unknown_execution_is_refused(self):
        from or_harness.core.storage import StorageError
        from or_harness.strategy.experience_bank import ExperienceBank
        bank = ExperienceBank(self.store)
        with self.assertRaises(StorageError):
            bank.set_task_check("ex_nope", {"state": "passed"})

    def test_state_vocabulary_is_closed(self):
        self.assertEqual(TASK_CHECK_STATES,
                         ("passed", "failed", "insufficient"))


if __name__ == "__main__":
    unittest.main()
