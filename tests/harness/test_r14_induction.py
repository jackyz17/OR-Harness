"""r14: whole-record induction material, joint H+ in the material, completion
observed across the acceptance branches, the shared decision anchor and the
real task-end boundary, adoption/attribution maintenance, and interval
semantics by kind.

Each class pins ONE r14 requirement, using NO real world model and NO new run
batch (scriptable/fixture providers only).

1. ``build_induction_material`` returns the SELECTED records WHOLE (task text,
   method steps/why, task-check fields, failures, trajectory) and references
   task texts by VERSION (echoed once), without recovering the full CIR.
2. The joint H+ saved with a strategy prediction reaches the material, with
   the four stances kept apart.
3. ``_observe_completion`` records 0 for an attempt that RAN and failed with
   no usable result (including ``task_check=insufficient``), while an
   unexecuted / unchecked-with-a-result attempt stays unknown.
4. The remaining span uses the SHARED decision anchor and the REAL task end
   (``finish_task``), not the per-prediction timestamp or the last execute.
5. Adoptions enter maintenance (counted apart from hits), and an agent
   ``refuting`` attribution drives the existing demotion.
6. A ``mean`` interval is not covered/uncovered by one observation.
"""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tests.harness.helpers import HarnessTestCase  # noqa: E402

from or_harness.api import ORHarness  # noqa: E402
from or_harness.core.schema import (  # noqa: E402
    ExecutionRecord, ProblemProfile, StrategicEntry, CostVector)
from or_harness.strategy.embedding_index import (  # noqa: E402
    LocalHashEmbeddingBackend)


def _profile(problem_id="t1", family="production_planning"):
    return ProblemProfile(problem_id=problem_id, family=family,
                          resource_coupling=0.5, temporal_coupling=0.2,
                          route_complexity=0.5)


RICH_TASK_TEXT = (
    "A workshop makes two products. Product 1 uses 50 per unit, product 2 "
    "uses 30 per unit. Upper bounds: product 1 at most 700, product 2 at "
    "most 500. Product 1 also needs 2 hours of finishing and product 2 "
    "needs 4 hours; finishing has 12 hours available. Every quantity must "
    "be a whole number. Maximise total profit across both products.")


class TestWholeRecordMaterial(HarnessTestCase):
    """The selected record is returned WHOLE, not summarised."""

    def test_task_text_method_and_check_are_not_truncated(self):
        h = ORHarness(home=self.home,
                      embedding=LocalHashEmbeddingBackend())
        self.addCleanup(h.close)
        task = {"task_id": "t1", "family": "production_planning",
                "description": RICH_TASK_TEXT,
                "spec": {"n_vars": 2, "n_constraints": 3, "n_int_vars": 2}}
        h.capture_task_text(task)
        from or_harness.world_model.state import task_text_digest
        rec = ExecutionRecord(
            execution_id="ex_rich", task_id="t1", strategy_id="S1",
            profile_snapshot=_profile("t1"),
            quality={"feasible": True, "objective": 100.0, "gap": 0.0,
                     "status": "optimal"},
            cost=CostVector(llm_tokens=10, measured={"llm_tokens"}),
            solver={"name": "highs", "code_hash": "h"}, source="executed",
            task_text_digest=task_text_digest(task))
        rec.method_actual = {
            "name": "bound-then-enumerate",
            "steps": [f"step {i}: derive bound {i}" for i in range(1, 7)],
            "why": "the bound makes the enumeration finite",
            "fallback": "fall back to a full MILP when the bound fails",
        }
        rec.execution_features["task_check"] = {
            "state": "failed",
            "scope": {"basis": "objective <= 90"},
            "reference_source": "agent",
            "conclusion": "the objective exceeds the declared cap",
            "checked": [{"check": f"probe {i}", "ok": False,
                         "reason": f"detail {i}"} for i in range(10)],
        }
        rec.trajectory = []
        rec.failures = []
        h.record(rec)
        out = h.induction_material()
        entry = out["material"][0]
        # Full task text (not a 280-char excerpt): the coefficients and
        # bounds survive.
        self.assertIsInstance(entry["task_text"], str)
        self.assertIn("50 per unit", entry["task_text"])
        self.assertIn("700", entry["task_text"])
        self.assertIn("500", entry["task_text"])
        self.assertGreater(len(entry["task_text"]), 280)
        # All six method steps and the why/fallback survive.
        actual = entry["method"]["actual"]
        self.assertEqual(actual["n_steps"], 6)
        self.assertEqual(len(actual["steps"]), 6)
        self.assertIn("why", actual)
        self.assertIn("fallback", actual)
        # The task check's 10 items all travel (no 8-item cap).
        self.assertEqual(len(entry["task_check"]["checked"]), 10)
        # The response-level version map echoes the text once.
        self.assertEqual(len(out["task_texts"]), 1)
        self.assertTrue(entry["task_text_ref"].get("task_text_digest"))

    def test_no_full_cir_is_recovered(self):
        h = ORHarness(home=self.home,
                      embedding=LocalHashEmbeddingBackend())
        self.addCleanup(h.close)
        from or_harness.world_model.state import task_text_digest
        task = {"task_id": "t1", "family": "f", "description": RICH_TASK_TEXT,
                "spec": {"n_vars": 2, "n_constraints": 2, "n_int_vars": 2}}
        h.capture_task_text(task)
        cir = {"constraints": [{"id": "C1", "kind": "capacity",
                                "expr": "x1 <= 700"}],
               "decisions": [{"id": "D1", "name": "x1"}],
               "entities": [], "relations": [], "coupling_groups": []}
        rec = ExecutionRecord(
            execution_id="ex_cir", task_id="t1", strategy_id="S1",
            profile_snapshot=_profile("t1"),
            quality={"feasible": True, "objective": 1.0, "gap": 0.0,
                     "status": "optimal"},
            cost=CostVector(llm_tokens=1, measured={"llm_tokens"}),
            solver={"name": "highs", "code_hash": "h"}, source="executed",
            task_text_digest=task_text_digest(task), cir_snapshot=cir)
        h.record(rec)
        entry = h.induction_material()["material"][0]
        cir_out = entry["problem"]["cir"]
        # The retained constraint expression travels (existing structure).
        self.assertEqual(cir_out["elements"]["constraints"][0]["expr"],
                         "x1 <= 700")
        # The COUNTS are still reported; no full-CIR-only keys were invented.
        self.assertEqual(cir_out["n_constraints"], 1)
        self.assertNotIn("issues", cir_out)  # none were present

    def test_related_history_shares_the_batch_expansion(self):
        h = ORHarness(home=self.home,
                      embedding=LocalHashEmbeddingBackend())
        self.addCleanup(h.close)
        from or_harness.world_model.state import task_text_digest
        for tid, sid in (("t1", "bound_then_enum"), ("t2", "enum_bound")):
            task = {"task_id": tid, "family": "f",
                    "description": f"solve {tid} by bounding then enumerating",
                    "spec": {"n_vars": 2, "n_constraints": 2, "n_int_vars": 2}}
            h.capture_task_text(task)
            rec = ExecutionRecord(
                execution_id=f"ex_{tid}", task_id=tid, strategy_id=sid,
                profile_snapshot=_profile(tid),
                quality={"feasible": True, "objective": 1.0, "gap": 0.0,
                         "status": "optimal"},
                cost=CostVector(llm_tokens=1, measured={"llm_tokens"}),
                solver={"name": "highs", "code_hash": "h"}, source="executed",
                task_text_digest=task_text_digest(task))
            rec.method_actual = {"name": sid, "steps": ["bound", "enumerate"]}
            h.record(rec)
        out = h.induction_material(task_id="t1", related_top_k=3)
        rh = out["related_history"]
        self.assertTrue(rh["enabled"])
        batch_keys = set(out["material"][0].keys())
        hit = rh["executions"][0]
        # The hit carries the SAME expansion fields as a batch entry (plus
        # the discovery metadata), not a lesser summary. Batch-only
        # bookkeeping (attempt/independent_task) is added by the batch loop,
        # so it is compared modulo that bookkeeping plus the discovery block.
        bookkeeping = {"attempts_of_task", "attempt_index", "independent_task",
                       "cursor"}
        for key in batch_keys - bookkeeping:
            self.assertIn(key, hit)
        self.assertIn("discovery", hit)


class TestJointHPlusInMaterial(HarnessTestCase):
    """A saved joint H+ reaches the material without gating anything."""

    def _harness(self, payload):
        from or_harness.world_model.provider import WorldModelProvider

        class _Provider(WorldModelProvider):
            name = "stub-hplus"

            def __init__(self):
                self.payload = payload

            def predict(self, request, timeout_s=None):
                return {"payload": self.payload,
                        "usage": {"prompt_tokens": 10,
                                  "completion_tokens": 5},
                        "error": None, "latency_s": 0.01}

        h = ORHarness(home=self.home, world_model=_Provider(),
                      embedding=LocalHashEmbeddingBackend())
        self.addCleanup(h.close)
        return h

    def test_saved_hplus_is_surfaced(self):
        h = self._harness({
            "benefit": {"kind": "effective_completion",
                        "metric": "task_result_check_passed",
                        "unit": "boolean", "value": 0.8,
                        "baseline": {"kind": "declared", "value": 0.5}},
            "cost": {"llm_tokens": 10},
            "capability_gain": {
                "assessment": "expected",
                "claim": "the bound generalises to a second task",
                "applies_to": ["bounded enumeration"]},
        })
        task = {"task_id": "t1", "family": "f", "description": "solve",
                "spec": {"n_vars": 2, "n_constraints": 2, "n_int_vars": 2}}
        prediction = h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy", "strategy_id": "S1"},
            "ep1")
        self.assertIsNotNone(prediction.capability_gain)
        rec = self.make_record(execution_id="ex_h", task_id="t1",
                               strategy_id="S1", profile=_profile("t1"))
        h.record(rec)
        out = h.induction_material()
        hplus = out["joint_hplus"]
        self.assertGreaterEqual(hplus["n"], 1)
        item = hplus["items"][0]
        self.assertEqual(item["assessment"], "expected")
        self.assertEqual(item["claim"], "the bound generalises to a second task")
        self.assertFalse(item["executed"])  # never bound to an execution

    def test_none_stance_is_surfaced_and_does_not_block(self):
        h = self._harness({
            "benefit": {"kind": "effective_completion",
                        "metric": "task_result_check_passed",
                        "unit": "boolean", "value": 0.5,
                        "baseline": {"kind": "declared", "value": 0.5}},
            "capability_gain": {"assessment": "none",
                                "basis": ["nothing new here"]},
        })
        task = {"task_id": "t1", "family": "f", "description": "solve",
                "spec": {"n_vars": 2, "n_constraints": 2, "n_int_vars": 2}}
        h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy", "strategy_id": "S1"},
            "ep1")
        h.record(self.make_record(execution_id="ex_n", task_id="t1",
                                  strategy_id="S1", profile=_profile("t1")))
        out = h.induction_material()
        self.assertEqual(out["joint_hplus"]["n_none"], 1)
        # `none` does not block: the material is still returned.
        self.assertGreaterEqual(out["count"], 1)


class TestCompletionAcrossBranches(HarnessTestCase):
    """Completion is observed across the acceptance branches."""

    def _harness(self):
        h = ORHarness(home=self.home,
                      embedding=LocalHashEmbeddingBackend())
        self.addCleanup(h.close)
        return h

    def test_ran_and_failed_without_result_observes_zero(self):
        from or_harness.world_model.episode_closeout import (
            _observe_completion, RealOutcomeSummary)
        rec = self.make_record(execution_id="ex_crash", task_id="t1",
                               feasible=False, objective=None, status="error",
                               profile=_profile("t1"))
        rec.execution_features["executed"] = True
        summary = RealOutcomeSummary(prediction_id="p", task_id="t1",
                                     episode_id="ep1", strategy_id="S1")
        summary.eligibility = {}
        _observe_completion(summary, [rec], scope="attempt")
        self.assertEqual(summary.benefit["observed"], 0.0)
        self.assertEqual(summary.benefit["basis"], "confirmed_failed_no_result")

    def test_ran_and_failed_with_insufficient_check_observes_zero(self):
        from or_harness.world_model.episode_closeout import (
            _observe_completion, RealOutcomeSummary)
        rec = self.make_record(execution_id="ex_ins", task_id="t1",
                               feasible=False, objective=None, status="error",
                               profile=_profile("t1"))
        rec.execution_features["executed"] = True
        rec.execution_features["task_check"] = {"state": "insufficient",
                                                "scope": {}}
        summary = RealOutcomeSummary(prediction_id="p", task_id="t1",
                                     episode_id="ep1", strategy_id="S1")
        summary.eligibility = {}
        _observe_completion(summary, [rec], scope="attempt")
        self.assertEqual(summary.benefit["observed"], 0.0)
        # The answer's validity stays UNKNOWN.
        self.assertIn("benefit_answer_validity", summary.eligibility)

    def test_never_executed_stays_unknown(self):
        from or_harness.world_model.episode_closeout import (
            _observe_completion, RealOutcomeSummary)
        rec = self.make_record(execution_id="ex_refused", task_id="t1",
                               feasible=False, objective=None, status="error",
                               profile=_profile("t1"))
        rec.execution_features["executed"] = False  # sandbox refusal
        summary = RealOutcomeSummary(prediction_id="p", task_id="t1",
                                     episode_id="ep1", strategy_id="S1")
        summary.eligibility = {}
        _observe_completion(summary, [rec], scope="attempt")
        self.assertNotIn("observed", summary.benefit)
        self.assertEqual(summary.eligibility["benefit"]["eligibility"],
                         "unobserved")

    def test_window_scope_does_not_score_a_first_failure(self):
        from or_harness.world_model.episode_closeout import (
            _observe_completion, RealOutcomeSummary)
        rec = self.make_record(execution_id="ex_w", task_id="t1",
                               feasible=False, objective=None, status="error",
                               profile=_profile("t1"))
        rec.execution_features["executed"] = True
        summary = RealOutcomeSummary(prediction_id="p", task_id="t1",
                                     episode_id="ep1", strategy_id="S1")
        summary.eligibility = {}
        # A window scope is judged by how it ENDED, never by a first failure.
        _observe_completion(summary, [rec], scope="strategy_window")
        self.assertNotIn("observed", summary.benefit)


class TestSharedAnchorAndTaskEnd(unittest.TestCase):
    """Unit checks on the anchor and the task-end boundary helpers."""

    def test_finish_task_is_the_task_end(self):
        from or_harness.world_model.episode_closeout import (
            _task_end_boundary)
        from or_harness.world_model.actions import ActionRecord

        class _Actions:
            def __init__(self, records):
                self._records = records

            def query(self, **kw):
                return list(self._records)

        actions = [
            ActionRecord(action_id="a1", action_type="execute_strategy",
                         task_id="t1", episode_id="ep1", started_at=100.0,
                         ended_at=110.0, status="completed"),
            ActionRecord(action_id="a2", action_type="verify", task_id="t1",
                         episode_id="ep1", started_at=120.0, ended_at=140.0,
                         status="completed"),
            ActionRecord(action_id="a3", action_type="finish_task",
                         task_id="t1", episode_id="ep1", started_at=145.0,
                         ended_at=150.0, status="completed"),
        ]

        class _H:
            def __init__(self):
                self.actions = _Actions(actions)

        end, basis = _task_end_boundary(_H(), None, "ep1", 100.0)
        self.assertEqual(end, 150.0)
        self.assertEqual(basis, "finish_task")

    def test_without_finish_task_the_boundary_is_flagged(self):
        from or_harness.world_model.episode_closeout import (
            _task_end_boundary)
        from or_harness.world_model.actions import ActionRecord

        class _Actions:
            def __init__(self, records):
                self._records = records

            def query(self, **kw):
                return list(self._records)

        actions = [
            ActionRecord(action_id="a1", action_type="execute_strategy",
                         task_id="t1", episode_id="ep1", started_at=100.0,
                         ended_at=110.0, status="completed"),
            ActionRecord(action_id="a2", action_type="verify", task_id="t1",
                         episode_id="ep1", started_at=120.0, ended_at=140.0,
                         status="completed"),
        ]

        class _H:
            def __init__(self):
                self.actions = _Actions(actions)

        end, basis = _task_end_boundary(_H(), None, "ep1", 100.0)
        self.assertEqual(end, 140.0)
        self.assertEqual(basis, "last_action_not_task_end")

    def test_late_offline_actions_do_not_extend_the_task(self):
        """An action after the close-out (a later offline pass) must not push
        the original task's end forward."""
        from or_harness.world_model.episode_closeout import (
            _task_end_boundary)
        from or_harness.world_model.actions import ActionRecord

        class _Actions:
            def __init__(self, records):
                self._records = records

            def query(self, **kw):
                return list(self._records)

        actions = [
            ActionRecord(action_id="a1", action_type="finish_task",
                         task_id="t1", episode_id="ep1", started_at=145.0,
                         ended_at=150.0, status="completed"),
            # A later action (offline induction / audit) after close-out.
            ActionRecord(action_id="a2", action_type="verify", task_id="t1",
                         episode_id="ep1", started_at=200.0,
                         ended_at=500.0, status="completed"),
        ]

        class _H:
            def __init__(self):
                self.actions = _Actions(actions)

        # Close-out happened at t=160.
        end, basis = _task_end_boundary(_H(), None, "ep1", 100.0,
                                        closeout_created_at=160.0)
        self.assertEqual(end, 150.0)
        self.assertEqual(basis, "finish_task")

    def test_shared_anchor_uses_the_decision_action(self):
        """Two predictions with DIFFERENT created_at share ONE decision
        anchor (the select_strategy action's started_at)."""
        from or_harness.world_model.episode_closeout import _decision_anchor
        from or_harness.world_model.actions import ActionRecord

        class _Actions:
            def __init__(self, record):
                self._record = record

            def get(self, action_id):
                return self._record if action_id == self._record.action_id \
                    else None

        decision = ActionRecord(action_id="dec1",
                                action_type="select_strategy", task_id="t1",
                                episode_id="ep1", started_at=100.0,
                                status="completed")

        class _H:
            def __init__(self):
                self.actions = _Actions(decision)

        class _Trace:
            def __init__(self, created, decision_id):
                self.created_at = created
                self.model_info = ({"decision_action_id": decision_id}
                                   if decision_id else {})

        class _P:
            def __init__(self, created, decision_id):
                self.trace = _Trace(created, decision_id)

        a, basis = _decision_anchor(_H(), _P(105.0, "dec1"), None)
        b, _ = _decision_anchor(_H(), _P(130.0, "dec1"), None)
        self.assertEqual(a, 100.0)
        self.assertEqual(b, 100.0)  # NOT the per-prediction created_at
        self.assertEqual(basis, "decision_action")
        # A legacy prediction with no decision id flags the fallback.
        c, c_basis = _decision_anchor(_H(), _P(140.0, None), None)
        self.assertEqual(c, 140.0)
        self.assertEqual(c_basis, "prediction_created_at_fallback")


class TestAdoptionMaintenance(HarnessTestCase):
    """Adoptions and agent attribution reach the maintenance report."""

    def _qualified_entry(self, h):
        entry = StrategicEntry(
            entry_id="placeholder", strategy_id="qualitative_trick",
            pattern={"predicates": {"family": "f"}})
        # quality_estimated stays False: a claim-only, qualitative technique.
        h.sbank.add(entry)
        return h.sbank.list()[0]

    def _engine(self, h):
        from or_harness.strategy.induction import InductionEngine
        from or_harness.strategy.stats import ConditionalStats
        return InductionEngine(ConditionalStats(h.bank), h.sbank)

    def test_qualitative_adoption_is_counted_and_reported(self):
        h = ORHarness(home=self.home,
                      embedding=LocalHashEmbeddingBackend())
        self.addCleanup(h.close)
        entry = self._qualified_entry(h)
        # An execution that DECLARED adopting the entry, whose strategy NAME
        # differs (adoption is by number, never by name).
        rec = self.make_record(execution_id="ex_adopt", task_id="t1",
                               strategy_id="a_different_name",
                               profile=_profile("t1"))
        rec.execution_features["used_entries"] = {
            "entry_ids": [entry.entry_id],
            "known_entry_ids": [entry.entry_id],
            "unknown_entry_ids": [],
        }
        h.record(rec)
        report = self._engine(h).revise()
        item = [r for r in report if r["entry_id"] == entry.entry_id][0]
        self.assertEqual(item["adoptions"]["n_adoptions"], 1)
        self.assertEqual(item["forward"]["n_predictions"], 0,
                         "an adoption is not a hit/miss")
        stored = h.sbank.get(entry.entry_id)
        self.assertEqual(stored.prediction_track.n_adoptions, 1)
        # No numeric prediction -> still a `candidate` (adoption never
        # promotes on its own).
        self.assertEqual(stored.status, "candidate")

    def test_refuting_attribution_demotes_via_existing_path(self):
        h = ORHarness(home=self.home,
                      embedding=LocalHashEmbeddingBackend())
        self.addCleanup(h.close)
        entry = self._qualified_entry(h)
        h.induce(effect_attributions=[
            {"entry_id": entry.entry_id, "verdict": "refuting",
             "note": "using it did not help on this instance"}])
        stored = h.sbank.get(entry.entry_id)
        self.assertEqual(len(stored.effect_attribution), 1)
        self.assertEqual(stored.effect_attribution[0]["by"], "agent")
        report = self._engine(h).revise()
        item = [r for r in report if r["entry_id"] == entry.entry_id][0]
        self.assertTrue(item["transitions"])
        self.assertEqual(h.sbank.get(entry.entry_id).status, "suspect")


class TestIntervalByKind(unittest.TestCase):
    """A mean interval is not decided by a single observation."""

    def test_mean_interval_is_unobservable_on_one_value(self):
        from or_harness.world_model.contracts import BenefitEstimate, \
            BaselineStatement
        from or_harness.world_model.episode_closeout import (
            evaluate_strategy_prediction, RealOutcomeSummary)
        from or_harness.world_model.contracts import CandidateRef
        from or_harness.world_model.contracts import \
            StrategyOutcomePrediction, PredictionTrace
        candidate = CandidateRef(task_id="t1", episode_id="ep1",
                                 strategy_id="S1",
                                 action_type="execute_strategy")
        prediction = StrategyOutcomePrediction(
            prediction_id="p", candidate=candidate)
        prediction.status = "valid"
        prediction.benefit = BenefitEstimate(
            kind="effective_completion", metric="task_result_check_passed",
            unit="boolean", value=0.85, interval=[0.8, 0.9],
            interval_kind="mean", interval_coverage=0.9,
            baseline=BaselineStatement(kind="declared", value=0.5))
        prediction.trace = PredictionTrace(prediction_kind="strategy_outcome")
        prediction.trace.comparable = True
        summary = RealOutcomeSummary(prediction_id="p", task_id="t1",
                                     episode_id="ep1", strategy_id="S1")
        summary.benefit = {"kind": "effective_completion",
                           "metric": "task_result_check_passed",
                           "unit": "boolean", "observed": 1.0,
                           "eligibility": "evaluable"}
        summary.eligibility = {"benefit": {"eligibility": "evaluable"}}
        summary.scope_status = "completed"
        evaluation = evaluate_strategy_prediction(prediction, summary)
        self.assertEqual(evaluation.interval["eligibility"], "unobservable")
        self.assertNotIn("covered", evaluation.interval)


if __name__ == "__main__":
    unittest.main()
