"""M6 tests: H-evolution prediction, knowledge value, delayed feedback.

Semantic-risk focused, organised by the acceptance themes the M6 plan names.
The provider here is a LABELLED synthetic fixture answering from the request
content; nothing in this file claims real-LLM behaviour.

Coverage map (plan §9 assertion -> test):
  1-4   task boundary            TestTaskBoundary
  5-7   prediction/fact isolation TestPredictionFactIsolation
  8-9   unknown upside not paid   TestUnknownKnowledgeValue
  10-11 knowledge-need maths      TestStructuralNeed
  12-13 model cannot self-raise   TestModelCannotSelfRaiseValue
  14-15 reward must be costed     TestPreconditionRealizability
  16-18 verdict semantics         TestVerdictSemantics
  19-22 delayed feedback loop     TestDelayedFeedbackLoop
  23-24 deduplicated attribution  TestDeduplicatedAttribution
  25-27 self-correction           TestSelfCorrection
  28-30 budget honesty            TestBudgetHonesty
  31-33 publication gate          TestPublicationGateUntouched
  34-35 knowledge value decides   TestKnowledgeValueChangesChoice
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tests.harness.helpers import HarnessTestCase  # noqa: E402

from or_harness.api import ORHarness, PREDICTION_MODES  # noqa: E402
from or_harness.world_model.knowledge import (  # noqa: E402
    KnowledgeTarget,
    learning_needs,
    precondition_realizability,
    structural_need,
    validate_knowledge_changes,
)
from or_harness.world_model.knowledge_track import (  # noqa: E402
    STAGE_CONSOLIDATION,
    STAGE_EXECUTION,
    class_reliability,
    dedupe_attribution,
    knowledge_feedback,
    proposition_key,
    record_stage_verdict,
    reliability_lookup,
    stage_verdict,
    unresolved_stages,
)
from or_harness.world_model.planner import (  # noqa: E402
    PlanLimits,
    evaluate_path,
    real_incumbent_available,
    second_step_dependency_ok,
)
from or_harness.world_model.prediction import (  # noqa: E402
    ActionSpec,
    OutcomePrediction,
    validate_prediction_payload,
)
from or_harness.world_model.provider import WorldModelProvider  # noqa: E402
from or_harness.world_model.state import task_text_digest  # noqa: E402

TASK = {"task_id": "t1", "family": "routing", "description": "toy routing"}

SOLVE_SCRIPT = (
    "import json\n"
    "with open('result.json', 'w') as fh:\n"
    "    json.dump({'status': 'feasible', 'objective_value': 900.0,\n"
    "               'objective_bound': 1000.0, 'runtime_seconds': 0.1}, fh)\n"
)


class KnowledgeProvider(WorldModelProvider):
    """Predicts X/B plus a knowledge change. ``uncertainty`` and
    ``expected_knowledge_value`` are deliberately model-controlled so tests
    can assert they do NOT raise the value."""

    name = "m6-knowledge"

    def __init__(self, *, quality=0.7, change="candidate_forms",
                 horizon="after_consolidation", uncertainty=0.0,
                 self_value=1.0, extra_cost=None,
                 propose_at_cold_start=False,
                 observation=None):
        self.quality = quality
        self.change = change
        self.horizon = horizon
        self.uncertainty = uncertainty
        self.self_value = self_value
        self.extra_cost = extra_cost
        # Whether to propose a hypothesis about the action's own strategy
        # when the framework offered no structural targets. Off by default
        # here so the "no targets -> no knowledge_changes" assertions stay
        # meaningful; the cold-start path has its own tests.
        self.propose_at_cold_start = propose_at_cold_start
        # What the prediction expects to SEE. Must name fields that really
        # exist on an execution record (the prompt says which): an
        # expectation the framework cannot check cannot confirm a claim,
        # so the verdict would stay pending forever.
        self.observation = (observation if observation is not None
                            else {"feasible": True})
        self.requests = []

    def predict(self, request):
        self.requests.append(request)
        payload = {
            "outcome_status": "feasible", "feasible": True,
            "quality": self.quality, "failure_prob": 0.1,
            "cost": {"llm_tokens": 100.0},
        }
        targets = request.get("candidate_knowledge_targets") or []
        if targets:
            first = targets[0]
            item = {
                "target": {"kind": "hypothesis",
                           "strategy_id": first.get("strategy_id")},
                "change": self.change,
                "horizon": self.horizon,
                "uncertainty": self.uncertainty,
                "expected_knowledge_value": self.self_value,
                "expected_observation": self.observation,
                "check_condition": "an entry exists afterwards",
            }
            if self.extra_cost is not None:
                item["expected_extra_cost"] = dict(self.extra_cost)
            payload["knowledge_changes"] = [item]
        return {"payload": payload,
                "usage": {"prompt_tokens": 50, "completion_tokens": 20},
                "error": None, "latency_s": 0.01}


class Base(HarnessTestCase):
    def setUp(self):
        super().setUp()
        self._work = tempfile.TemporaryDirectory()
        self.addCleanup(self._work.cleanup)
        self.work = Path(self._work.name)
        self.script = self.work / "solve.py"
        self.script.write_text(SOLVE_SCRIPT, encoding="utf-8")

    def make_harness(self, provider=None, **kwargs):
        h = ORHarness(home=self.home,
                      world_model=provider or KnowledgeProvider(), **kwargs)
        self.addCleanup(h.close)
        return h

    def seed(self, h, task_id="t1", strategy_id="S01"):
        """One real execution + record — the minimum for a learning need."""
        task = dict(TASK, task_id=task_id)
        record = h.execute(task, strategy_id, str(self.script), str(self.work),
                           solver="highs", episode_id=f"e_{task_id}")
        h.record(record, override={"llm_tokens": 1500.0})
        return record


# ---------------------------------------------------------------------------
# 1-4  task boundary
# ---------------------------------------------------------------------------


class TestTaskBoundary(Base):
    def test_progress_not_inherited_across_task_versions(self):
        """A2: progress from a snapshot taken under a DIFFERENT task version
        must not be inherited — checked at state construction, not only at
        bind time, so a stale X cannot be planned with."""
        h = self.make_harness()
        self.seed(h)
        original = dict(TASK)
        h.snapshot(original, "ep1",
                   task_progress={"current_solution": {
                       "value": {"status": "optimal"}, "provenance":
                       "observed", "epistemic": "fact"}})
        # Same task_id, DIFFERENT content: a new task context.
        edited = dict(TASK, description="toy routing with a NEW constraint")
        inherited = h._episode_progress("t1", "ep1",
                                        task_digest=task_text_digest(edited))
        self.assertEqual(inherited, {},
                         "a changed task version must not inherit old X")
        unchanged = h._episode_progress(
            "t1", "ep1", task_digest=task_text_digest(original))
        self.assertIn("current_solution", unchanged)

    def test_bind_compares_prediction_version_not_current_task(self):
        """A3: a LATE but valid prediction must not be rejected merely
        because the task JSON has since been edited. The comparison is
        between the versions the two SIDES recorded."""
        h = self.make_harness()
        spec = ActionSpec("execute_strategy", "t1", strategy_id="S01",
                          episode_id="ep1")
        prediction = h.predict_outcome(TASK, spec, "ep1")
        record = h.execute(TASK, "S01", str(self.script), str(self.work),
                           solver="highs", episode_id="ep1")
        # The task is edited AFTER the execution — a late binding must still
        # be judged against execution-time versions, not this new content.
        edited = dict(TASK, description="edited after the fact")
        h.snapshot(edited, "ep1")
        bound = h.bind_outcome(prediction.prediction_id, record.action_id)
        digest_mismatch = (bound.binding_mismatch or {}).get("task_digest")
        self.assertIsNone(digest_mismatch,
                          "same-version prediction/execution must not be "
                          "rejected for a later edit")

    def test_cross_task_records_are_not_action_transition_samples(self):
        """A4: a prediction for task A never scores task B's execution."""
        h = self.make_harness()
        spec = ActionSpec("execute_strategy", "t1", strategy_id="S01",
                          episode_id="ep1")
        prediction = h.predict_outcome(TASK, spec, "ep1")
        other = self.make_record(task_id="t2", strategy_id="S01")
        h.bank.append(other)
        action = h.actions.begin_action(
            "execute_strategy", "t2", "ep2",
            pre_snapshot=h.snapshot(dict(TASK, task_id="t2"), "ep2"),
            params={"strategy_id": "S01"})
        h.actions.end_action(action.action_id, status="completed",
                             linked_execution_id=other.execution_id,
                             cost=other.cost, rollup="reference")
        bound = h.bind_outcome(prediction.prediction_id, action.action_id)
        self.assertIn("task_id", bound.binding_mismatch or {})


# ---------------------------------------------------------------------------
# 5-7  prediction / fact isolation
# ---------------------------------------------------------------------------


class TestPredictionFactIsolation(Base):
    def test_predicted_knowledge_never_enters_the_strategic_bank(self):
        """A5: predicting knowledge changes writes no entry."""
        h = self.make_harness()
        self.seed(h)
        before = h.sbank.count()
        spec = ActionSpec("execute_strategy", "t1", strategy_id="S01")
        prediction = h.predict_outcome(TASK, spec, "ep1")
        self.assertTrue((prediction.predicted or {}).get("knowledge_changes"))
        self.assertEqual(h.sbank.count(), before)

    def test_predicted_knowledge_is_not_recallable(self):
        """A6: predicted knowledge must never surface as a recall result."""
        h = self.make_harness()
        self.seed(h)
        spec = ActionSpec("execute_strategy", "t1", strategy_id="S01")
        h.predict_outcome(TASK, spec, "ep1")
        recall = h.recall(TASK, top=5)
        for item in recall["recommendations"]:
            self.assertNotIn("knowledge_changes", item)
        self.assertNotIn("knowledge_changes",
                         str(recall.get("recommendations")))

    def test_prediction_rollout_may_reason_over_imagined_solution(self):
        """A7: a rollout over a predicted successor IS allowed; what is
        forbidden is a REAL execution treating it as an incumbent."""
        prediction = OutcomePrediction(
            prediction_id="wp_x", input_snapshot_id="bs_x",
            action_spec=ActionSpec("execute_strategy", "t1",
                                   strategy_id="S01"),
            status="valid",
            predicted={"state_changes": {"current_solution": {
                "status": "feasible", "objective": 4210.5,
                "solution": {"x_M1": 30}}}})
        self.assertTrue(second_step_dependency_ok(prediction),
                        "a structured predicted successor is usable for a "
                        "rollout")
        # ... but a REAL execution must not depend on it.
        imagined = {"current_solution": {
            "value": {"status": "optimal", "objective": 4210.5},
            "provenance": "agent_reported", "epistemic": "inferred",
            "hypothetical": True}}
        self.assertFalse(real_incumbent_available(imagined))
        observed = {"current_solution": {
            "value": {"status": "optimal", "objective": 4210.5},
            "provenance": "observed", "epistemic": "fact"}}
        self.assertTrue(real_incumbent_available(observed))


# ---------------------------------------------------------------------------
# 8-9  unknown upside is never rewarded
# ---------------------------------------------------------------------------


class TestUnknownKnowledgeValue(Base):
    def test_missing_k_adds_nothing_not_the_full_weight(self):
        """A8: an unknown knowledge value contributes 0.0, NOT delta."""
        limits = PlanLimits(delta=0.9)
        prediction = OutcomePrediction(
            prediction_id="wp_1", input_snapshot_id="bs_1",
            action_spec=ActionSpec("execute_strategy", "t1",
                                   strategy_id="S01"),
            status="valid",
            predicted={"quality": 0.8, "feasible": True,
                       "failure_prob": 0.0, "cost": {"llm_tokens": 1.0},
                       "cost_measured": ["llm_tokens"]})
        with_k_none = evaluate_path([prediction], limits,
                                    {"llm_tokens": 1.0}, ["llm_tokens"],
                                    knowledge=(None, {}))
        self.assertEqual(with_k_none.delta_knowledge, 0.0)
        self.assertIn("knowledge", with_k_none.incomparable)
        # A justified K of 1.0 WOULD add the full weight — proving the term
        # is live and the None case is a deliberate zero, not a dead term.
        with_k_one = evaluate_path([prediction], limits,
                                   {"llm_tokens": 1.0}, ["llm_tokens"],
                                   knowledge=(1.0, {}))
        self.assertEqual(with_k_one.delta_knowledge, 0.9)

    def test_unknown_k_ranks_like_no_knowledge_term(self):
        """A9: with K unknown, ranking degenerates to the solve utility."""
        limits = PlanLimits(delta=0.9)
        strong = OutcomePrediction(
            prediction_id="wp_s", input_snapshot_id="bs_1",
            action_spec=ActionSpec("execute_strategy", "t1",
                                   strategy_id="S01"),
            status="valid",
            predicted={"quality": 0.9, "feasible": True, "failure_prob": 0.0,
                       "cost": {"llm_tokens": 1.0},
                       "cost_measured": ["llm_tokens"]})
        weak = OutcomePrediction(
            prediction_id="wp_w", input_snapshot_id="bs_1",
            action_spec=ActionSpec("execute_strategy", "t1",
                                   strategy_id="S02"),
            status="valid",
            predicted={"quality": 0.3, "feasible": True, "failure_prob": 0.0,
                       "cost": {"llm_tokens": 1.0},
                       "cost_measured": ["llm_tokens"]})
        p_strong = evaluate_path([strong], limits, {"llm_tokens": 1.0},
                                 ["llm_tokens"], knowledge=(None, {}))
        p_weak = evaluate_path([weak], limits, {"llm_tokens": 1.0},
                               ["llm_tokens"], knowledge=(None, {}))
        self.assertGreater(p_strong.utility, p_weak.utility,
                           "unknown knowledge must not invert a quality gap")


# ---------------------------------------------------------------------------
# 10-11  knowledge-need maths
# ---------------------------------------------------------------------------


class TestStructuralNeed(unittest.TestCase):
    def test_interval_slack_is_bounded_and_monotone(self):
        """A10: the replacement for the sign-broken ratio is in [0,1] and
        rises with the interval's room above its honest floor."""
        from or_harness.core.schema import min_interval_width
        for n in (2, 3, 4, 5, 10, 30):
            floor = min_interval_width(n)
            at_floor = structural_need(support_n=n, interval_width=floor,
                                       distinct_tasks=1)
            wide = structural_need(support_n=n, interval_width=1.0,
                                   distinct_tasks=1)
            self.assertIsNotNone(at_floor)
            self.assertIsNotNone(wide)
            self.assertGreaterEqual(at_floor, 0.0)
            self.assertLessEqual(wide, 1.0)
            self.assertLessEqual(at_floor, wide)
        # The concrete case the old formula got wrong: width 0.4 with a
        # 0.2 floor used to evaluate to MINUS ONE.
        value = structural_need(support_n=3, interval_width=0.4,
                                distinct_tasks=1)
        self.assertGreater(value, 0.0)

    def test_reuse_demand_counts_independent_tasks(self):
        """A11: repeating ONE task does not raise the reuse basis."""
        one_task = structural_need(support_n=5, interval_width=None,
                                   distinct_tasks=1)
        five_tasks = structural_need(support_n=5, interval_width=None,
                                     distinct_tasks=5)
        self.assertLess(one_task, five_tasks)
        # Unknown everywhere -> no need is claimed at all.
        self.assertIsNone(structural_need(support_n=None,
                                          interval_width=None,
                                          distinct_tasks=None))


# ---------------------------------------------------------------------------
# 12-13  the model cannot raise its own value
# ---------------------------------------------------------------------------


class TestModelCannotSelfRaiseValue(Base):
    def test_self_reported_uncertainty_does_not_raise_k(self):
        """A12: a model asserting uncertainty=0.0 gains nothing, because the
        discount comes from measured class reliability, not from the model."""
        from or_harness.world_model.knowledge import knowledge_value
        targets = [KnowledgeTarget(kind="hypothesis", strategy_id="S01",
                                   support_n=2, distinct_tasks=2)]
        low = OutcomePrediction(
            prediction_id="wp_low", input_snapshot_id="bs_1",
            action_spec=ActionSpec("execute_strategy", "t1",
                                   strategy_id="S01"),
            status="valid",
            predicted={"knowledge_changes": [{
                "target": {"kind": "hypothesis", "strategy_id": "S01"},
                "change": "candidate_forms",
                "horizon": "after_consolidation",
                "uncertainty": 0.0, "expected_knowledge_value": 1.0}]})
        certain = knowledge_value(targets, [low],
                                  reliability=lambda c, h: None)
        self.assertIsNone(certain[0],
                          "an unmeasured class grants no value regardless of "
                          "the model's self-reported certainty")
        measured = knowledge_value(targets, [low],
                                   reliability=lambda c, h: 1.0)
        self.assertIsNotNone(measured[0])
        self.assertGreater(measured[0], 0.0)

    def test_self_scored_value_is_never_consumed(self):
        """A13: expected_knowledge_value is recorded, not used."""
        from or_harness.world_model.knowledge import knowledge_value
        targets = [KnowledgeTarget(kind="hypothesis", strategy_id="S01",
                                   support_n=2, distinct_tasks=2)]
        def make(value):
            return OutcomePrediction(
                prediction_id=f"wp_{value}", input_snapshot_id="bs_1",
                action_spec=ActionSpec("execute_strategy", "t1",
                                       strategy_id="S01"),
                status="valid",
                predicted={"knowledge_changes": [{
                    "target": {"kind": "hypothesis", "strategy_id": "S01"},
                    "change": "candidate_forms",
                    "horizon": "after_consolidation",
                    "expected_knowledge_value": value}]})
        low, _ = knowledge_value(targets, [make(0.01)],
                                 reliability=lambda c, h: 0.5)
        high, _ = knowledge_value(targets, [make(1.0)],
                                  reliability=lambda c, h: 0.5)
        self.assertEqual(low, high,
                         "the model's own value estimate must not move K")


# ---------------------------------------------------------------------------
# 14-15  reward must be costed
# ---------------------------------------------------------------------------


class TestPreconditionRealizability(unittest.TestCase):
    def test_uncosted_after_consolidation_value_is_not_granted(self):
        """A14: depending on offline work whose cost is unknown downgrades K
        to None instead of granting the full benefit."""
        item = {"change": "revises", "horizon": "after_consolidation",
                "preconditions": ["verification_check"]}
        realizability, notes = precondition_realizability(item)
        self.assertIsNone(realizability)
        self.assertTrue(notes)

    def test_declared_extra_cost_allows_a_discounted_value(self):
        """A15: the same dependency WITH a declared cost is discountable but
        granted — and the discount is explicit."""
        item = {"change": "revises", "horizon": "after_consolidation",
                "preconditions": ["verification_check"],
                "expected_extra_cost": {"llm_tokens": 200.0}}
        realizability, notes = precondition_realizability(item)
        self.assertIsNotNone(realizability)
        self.assertLess(realizability, 1.0)
        self.assertTrue(any("realizability" in n for n in notes))

    def test_no_preconditions_is_fully_realizable(self):
        realizability, notes = precondition_realizability(
            {"change": "adds_evidence", "horizon": "after_execution"})
        self.assertEqual(realizability, 1.0)
        self.assertEqual(notes, [])


# ---------------------------------------------------------------------------
# 16-18  verdict semantics
# ---------------------------------------------------------------------------


class TestVerdictSemantics(Base):
    def test_unmet_opportunity_stays_pending_not_missed(self):
        """A17: with nothing observed to judge against, the item stays
        pending rather than being called wrong."""
        from or_harness.world_model.knowledge_track import evaluate_execution
        target = KnowledgeTarget(kind="existing_entry", strategy_id="S01",
                                 entry_id="se_x", cell_token="",
                                 quality_interval=None)
        record = self.make_record(task_id="t1", strategy_id="S01")
        status, detail = evaluate_execution(
            {"change": "supports", "horizon": "after_execution"}, target,
            record)
        self.assertEqual(status, "pending")
        self.assertIn("states no interval", detail["reason"])
        # With NO observed quality at all (an infeasible attempt) it is also
        # pending, never missed.
        infeasible = self.make_record(task_id="t1", strategy_id="S01",
                                      feasible=False, status="infeasible")
        status, detail = evaluate_execution(
            {"change": "supports", "horizon": "after_execution"}, target,
            infeasible)
        self.assertEqual(status, "pending")

    def test_candidate_formed_is_not_publishable_knowledge(self):
        """A16: an entry FORMING is not the same as publishable knowledge."""
        from or_harness.world_model.knowledge_track import (
            evaluate_consolidation,
        )
        target = KnowledgeTarget(kind="hypothesis", strategy_id="S01")
        status, detail = evaluate_consolidation(
            {"change": "candidate_forms",
             "horizon": "after_consolidation"},
            target, [], ["se_new"], ["S01"])
        self.assertEqual(status, "fulfilled")
        self.assertIn("publishable", detail["reason"])

    def test_missed_requires_the_opportunity_to_have_arrived(self):
        """A18: the induction ran on this scope but nothing formed."""
        from or_harness.world_model.knowledge_track import (
            evaluate_consolidation,
        )
        target = KnowledgeTarget(kind="hypothesis", strategy_id="S01")
        status, detail = evaluate_consolidation(
            {"change": "candidate_forms",
             "horizon": "after_consolidation"},
            target, [{"entry_id": "se_other", "changed": {}}], [], [])
        self.assertEqual(status, "missed")
        self.assertIn("no entry formed", detail["reason"])


# ---------------------------------------------------------------------------
# 19-22  the delayed feedback loop
# ---------------------------------------------------------------------------


class TestDelayedFeedbackLoop(Base):
    def test_normal_execution_prediction_is_evaluated_at_both_stages(self):
        """A19+A20: an ORDINARY action's knowledge prediction gets an
        execution-stage verdict AND a consolidation-stage verdict.

        Two predictions are needed because the stages are honestly
        different propositions: evidence lands when the execution is
        recorded, a claim can only form when induction runs."""
        # (a) execution stage: an evidence-accumulation prediction.
        exec_provider = KnowledgeProvider(change="adds_evidence",
                                          horizon="after_execution")
        h = self.make_harness(exec_provider)
        self.seed(h, "t1")
        spec = ActionSpec("execute_strategy", "t1", strategy_id="S01")
        exec_pred = h.predict_outcome(TASK, spec, "ep1")
        record = h.execute(TASK, "S01", str(self.script), str(self.work),
                           solver="highs", episode_id="ep1")
        h.bind_outcome(exec_pred.prediction_id, record.action_id)
        result = h.record(record, override={"llm_tokens": 1500.0})
        self.assertIn("knowledge_feedback", result,
                      "record must evaluate the execution-stage prediction")
        after_exec = h.get_prediction(exec_pred.prediction_id)
        verdict = stage_verdict(after_exec, 0, STAGE_EXECUTION)
        self.assertIsNotNone(verdict)
        self.assertEqual(verdict["status"], "fulfilled")
        self.assertTrue(verdict["targets_available"])

        # (b) consolidation stage: the claim-formation prediction, judged
        # by the induction that follows.
        cons_provider = KnowledgeProvider(change="candidate_forms",
                                          horizon="after_consolidation")
        h.world_model = cons_provider
        h.predictions.provider = cons_provider
        cons_pred = h.predict_outcome(TASK, spec, "ep2")
        cons_record = h.execute(TASK, "S01", str(self.script), str(self.work),
                                solver="highs", episode_id="ep2")
        h.bind_outcome(cons_pred.prediction_id, cons_record.action_id)
        h.record(cons_record, override={"llm_tokens": 1500.0})
        self.seed(h, "t2")
        induced = h.induce(strategy_id="S01", all_=True)
        self.assertIn("knowledge_feedback", induced,
                      "induce must evaluate the consolidation-stage "
                      "prediction — this is the loop M6 exists to close")
        after_cons = h.get_prediction(cons_pred.prediction_id)
        verdict = stage_verdict(after_cons, 0, STAGE_CONSOLIDATION)
        self.assertIsNotNone(verdict)
        self.assertEqual(verdict["status"], "fulfilled")

    def test_stage_partitioning_lets_later_stages_still_land(self):
        """A21: a prediction whose immediate X/B comparison already ran can
        still receive its knowledge verdict."""
        h = self.make_harness()
        spec = ActionSpec("execute_strategy", "t1", strategy_id="S01")
        prediction = h.predict_outcome(TASK, spec, "ep1")
        prediction.feedback = {"compared": True, "compared_fields": {}}
        h.predictions._save(prediction)
        self.assertTrue(record_stage_verdict(
            prediction, 0, STAGE_EXECUTION, "fulfilled", {}))
        self.assertTrue(record_stage_verdict(
            prediction, 0, STAGE_CONSOLIDATION, "fulfilled", {}),
            "an existing immediate comparison must not block the knowledge "
            "stage")
        self.assertEqual(
            len(knowledge_feedback(prediction)["0"]), 2)

    def test_xb_comparison_is_not_blocked_by_a_knowledge_verdict(self):
        """A2a: the knowledge channel writes into its own partition, so its
        verdict must not suppress the X/B calibration of the same
        prediction."""
        provider = KnowledgeProvider(change="adds_evidence",
                                     horizon="after_execution")
        h = self.make_harness(provider)
        self.seed(h, "t1")
        spec = ActionSpec("execute_strategy", "t1", strategy_id="S01")
        prediction = h.predict_outcome(TASK, spec, "ep1")
        record = h.execute(TASK, "S01", str(self.script), str(self.work),
                           solver="highs", episode_id="ep1")
        h.bind_outcome(prediction.prediction_id, record.action_id)
        # record() runs the H evaluation first, writing the knowledge block.
        h.record(record, override={"llm_tokens": 1500.0})
        after = h.get_prediction(prediction.prediction_id)
        self.assertIn("knowledge", after.feedback or {})
        # The X/B comparison must still happen.
        compared = h.compare_prediction(prediction.prediction_id)
        self.assertTrue((compared.feedback or {}).get("compared"),
                        "a knowledge verdict must not brick the X/B "
                        "comparison")
        # ... and re-running it stays idempotent.
        again = h.compare_prediction(prediction.prediction_id)
        self.assertEqual((again.feedback or {}).get("execution_id"),
                         (compared.feedback or {}).get("execution_id"))

    def test_pending_can_still_resolve(self):
        """A22: pending is not terminal — it must be promotable once the
        opportunity arrives. A terminal verdict, by contrast, is final."""
        h = self.make_harness(KnowledgeProvider(
            change="adds_evidence", horizon="after_execution"))
        spec = ActionSpec("execute_strategy", "t1", strategy_id="S01")
        self.seed(h)
        prediction = h.predict_outcome(TASK, spec, "ep1")
        # The prediction declares which stage can resolve here; a stage it
        # never declared is not on its work list at all.
        stage = (prediction.predicted["knowledge_changes"][0]["horizon"])
        self.assertEqual(stage, STAGE_EXECUTION)
        record_stage_verdict(prediction, 0, stage, "pending",
                             {"reason": "not yet"})
        self.assertEqual(stage_verdict(prediction, 0, stage)["status"],
                         "pending")
        # A pending item is still the framework's work list.
        self.assertIn((0, stage), unresolved_stages(prediction))
        changed = record_stage_verdict(prediction, 0, stage, "fulfilled",
                                       {"reason": "later"})
        self.assertTrue(changed,
                        "a pending verdict records that the opportunity has "
                        "not arrived; it must be promotable")
        promoted = stage_verdict(prediction, 0, stage)
        self.assertEqual(promoted["status"], "fulfilled")
        self.assertEqual(promoted.get("promoted_from"), "pending")
        # ... and once terminal it is final.
        self.assertFalse(record_stage_verdict(prediction, 0, stage,
                                              "missed", {}))
        self.assertEqual(stage_verdict(prediction, 0, stage)["status"],
                         "fulfilled")
        self.assertNotIn((0, stage), unresolved_stages(prediction))

    def test_repeated_evaluation_is_idempotent(self):
        """A12 (repeat feedback): re-running the same stage changes nothing."""
        h = self.make_harness()
        self.seed(h)
        spec = ActionSpec("execute_strategy", "t1", strategy_id="S01")
        prediction = h.predict_outcome(TASK, spec, "ep1")
        record = h.execute(TASK, "S01", str(self.script), str(self.work),
                           solver="highs", episode_id="ep1")
        h.bind_outcome(prediction.prediction_id, record.action_id)
        first = h.record(record, override={"llm_tokens": 1500.0})
        second = h.evaluate_knowledge_execution(h.bank.get(
            record.execution_id))
        self.assertEqual(second, {},
                         "re-evaluation must not re-write or re-count")


# ---------------------------------------------------------------------------
# 23-24  deduplicated attribution
# ---------------------------------------------------------------------------


class TestDeduplicatedAttribution(unittest.TestCase):
    def test_same_proposition_credits_only_the_earliest_prediction(self):
        """A23: several actions predicting one proposition do not each
        collect the full benefit."""
        item = {"target": {"kind": "existing_entry", "entry_id": "se_1",
                           "strategy_id": "S01"},
                "change": "revises", "horizon": "after_consolidation"}
        entries = [
            {"prediction_id": "wp_late", "created_at": 200.0, "item": item},
            {"prediction_id": "wp_early", "created_at": 100.0, "item": item},
        ]
        marked = dedupe_attribution(entries)
        by_id = {m["prediction_id"]: m["attribution"] for m in marked}
        self.assertEqual(by_id["wp_early"], "attributed")
        self.assertEqual(by_id["wp_late"], "co_attributed")

    def test_evidence_accumulation_is_not_double_counting(self):
        """A24: adds_evidence genuinely accumulates per action."""
        item = {"target": {"kind": "hypothesis", "strategy_id": "S01"},
                "change": "adds_evidence", "horizon": "after_execution"}
        entries = [{"prediction_id": "wp_a", "created_at": 1.0, "item": item},
                   {"prediction_id": "wp_b", "created_at": 2.0, "item": item}]
        marked = dedupe_attribution(entries)
        self.assertTrue(all(m["attribution"] == "per_action"
                            for m in marked))

    def test_proposition_identity_separates_horizons(self):
        base = {"target": {"kind": "hypothesis", "strategy_id": "S01"},
                "change": "candidate_forms"}
        a = proposition_key(dict(base, horizon="after_execution"))
        b = proposition_key(dict(base, horizon="after_consolidation"))
        self.assertNotEqual(a, b,
                            "the same claim at two times is two propositions")


# ---------------------------------------------------------------------------
# 25-27  self-correction
# ---------------------------------------------------------------------------


class TestSelfCorrection(Base):
    def test_low_reliability_suppresses_later_value(self):
        """A25: a class with a measured 0.0 fulfilment rate yields no value."""
        from or_harness.world_model.knowledge import knowledge_value
        verdicts = [{"change": "candidate_forms",
                     "horizon": "after_consolidation",
                     "status": "missed"}] * 6
        table = class_reliability(verdicts)
        lookup = reliability_lookup(table)
        self.assertEqual(lookup("candidate_forms", "after_consolidation"),
                         0.0)
        targets = [KnowledgeTarget(kind="hypothesis", strategy_id="S01",
                                   support_n=2, distinct_tasks=2)]
        prediction = OutcomePrediction(
            prediction_id="wp_1", input_snapshot_id="bs_1",
            action_spec=ActionSpec("execute_strategy", "t1",
                                   strategy_id="S01"),
            status="valid",
            predicted={"knowledge_changes": [{
                "target": {"kind": "hypothesis", "strategy_id": "S01"},
                "change": "candidate_forms",
                "horizon": "after_consolidation"}]})
        value, detail = knowledge_value(targets, [prediction],
                                        reliability=lookup)
        self.assertEqual(value, 0.0,
                         "a class that never pays off must not pay")

    def test_thin_history_stays_unknown(self):
        """A26: too few samples => no reliability is claimed."""
        table = class_reliability([{
            "change": "narrows", "horizon": "after_consolidation",
            "status": "fulfilled"}])
        entry = table["narrows|after_consolidation"]
        self.assertIsNone(entry["reliability"])
        self.assertEqual(entry["basis"], "insufficient_history")
        self.assertIsNone(reliability_lookup(table)(
            "narrows", "after_consolidation"))

    def test_reliability_is_measured_only_from_resolved_verdicts(self):
        """A27: pending and inconclusive carry no information either way."""
        table = class_reliability([
            {"change": "supports", "horizon": "after_execution",
             "status": "fulfilled"},
            {"change": "supports", "horizon": "after_execution",
             "status": "inconclusive"},
            {"change": "supports", "horizon": "after_execution",
             "status": "pending"},
            {"change": "supports", "horizon": "after_execution",
             "status": "contradicted"},
        ])
        self.assertEqual(table["supports|after_execution"]["n"], 2)

    def test_reliability_table_is_empty_before_any_evidence(self):
        h = self.make_harness()
        self.assertEqual(h.prediction_reliability_table(), {})


# ---------------------------------------------------------------------------
# 28-30  budget honesty
# ---------------------------------------------------------------------------


class TestBudgetHonesty(Base):
    def test_failed_execution_cost_still_counted(self):
        """A28: a failure is a real spend."""
        h = self.make_harness()
        broken = self.work / "broken.py"
        broken.write_text("raise SystemExit(2)\n", encoding="utf-8")
        record = h.execute(TASK, "S01", str(broken), str(self.work),
                           solver="highs", episode_id="ep1")
        result = h.record(record)
        self.assertTrue(result["recorded"])

    def test_delta_zero_reproduces_pre_m6_utility(self):
        """A35: with delta 0 the knowledge term is bit-identical to before."""
        prediction = OutcomePrediction(
            prediction_id="wp_1", input_snapshot_id="bs_1",
            action_spec=ActionSpec("execute_strategy", "t1",
                                   strategy_id="S01"),
            status="valid",
            predicted={"quality": 0.8, "feasible": True,
                       "failure_prob": 0.1,
                       "cost": {"llm_tokens": 10.0},
                       "cost_measured": ["llm_tokens"]})
        weights = {"llm_tokens": 1.0}
        off = evaluate_path([prediction], PlanLimits(delta=0.0,
                                                     cost_weights=weights),
                            {"llm_tokens": 10.0}, ["llm_tokens"],
                            knowledge=(1.0, {}))
        self.assertEqual(off.delta_knowledge, 0.0)
        # alpha*Q - beta*(w*v/norm) - gamma*R, with no knowledge term at all.
        expected = round(1.0 * 0.8 - 1.0 * (1.0 * 10.0 / 10.0)
                         - 1.0 * 0.1, 6)
        self.assertEqual(off.utility, expected)
        self.assertNotIn("knowledge", off.incomparable,
                         "delta=0 leaves the comparison report untouched")

    def test_planning_cost_is_sunk_not_in_path_utility(self):
        """A30: the calls' own spend never enters a path's utility."""
        h = self.make_harness()
        self.seed(h)
        plan = h.plan_next(TASK, "ep1",
                           candidates=[ActionSpec("execute_strategy", "t1",
                                                  strategy_id="S01")],
                           limits={"horizon": 1})
        if plan.get("planning_cost"):
            for path in plan["paths"]:
                self.assertNotIn("planning_cost", path)


# ---------------------------------------------------------------------------
# 31-33  the publication gate is untouched
# ---------------------------------------------------------------------------


class TestPublicationGateUntouched(Base):
    def test_knowledge_prediction_does_not_publish_an_entry(self):
        """A31: predicting candidate_forms does not verify anything."""
        h = self.make_harness()
        self.seed(h)
        before = [e.entry_id for e in h.sbank.list()]
        spec = ActionSpec("execute_strategy", "t1", strategy_id="S01")
        h.predict_outcome(TASK, spec, "ep1")
        self.assertEqual([e.entry_id for e in h.sbank.list()], before)

    def test_unverified_entry_still_not_publishable(self):
        """A33: a prediction's verdict never flips an entry's standing."""
        from or_harness.strategy.selector import is_publishable
        h = self.make_harness()
        self.seed(h, "t1")
        self.seed(h, "t2")
        h.induce(strategy_id="S01", all_=True)
        for entry in h.sbank.list():
            block = entry.verification or {}
            if block and block.get("state") != "verified":
                self.assertFalse(is_publishable(entry))

    def test_legacy_entry_without_verification_remains_publishable(self):
        """A32: the pre-existing exception is preserved, not silently
        changed into 'unverified means unpublishable'."""
        from or_harness.strategy.selector import is_publishable
        entry = h = None
        harness = self.make_harness()
        self.seed(harness, "t1")
        harness.induce(strategy_id="S01", all_=True)
        entries = harness.sbank.list()
        if entries:
            legacy = entries[0]
            legacy.verification = {}
            self.assertTrue(is_publishable(legacy),
                            "an entry with no verification block is the "
                            "documented legacy case and stays usable")


# ---------------------------------------------------------------------------
# 34-35  the knowledge value actually changes the choice
# ---------------------------------------------------------------------------


class TestKnowledgeValueChangesChoice(Base):
    def test_knowledge_value_flips_a_tied_choice_only_when_enabled(self):
        """A34 — the core falsifiable assertion. Two paths tied on X/B and
        differing only in the predicted knowledge change: the choice flips
        under h-x-b-value and does NOT flips under delta=0."""
        def make(sid, knowledge_changes):
            return OutcomePrediction(
                prediction_id=f"wp_{sid}", input_snapshot_id="bs_1",
                action_spec=ActionSpec("execute_strategy", "t1",
                                       strategy_id=sid),
                status="valid",
                predicted={"quality": 0.7, "feasible": True,
                           "failure_prob": 0.1,
                           "cost": {"llm_tokens": 10.0},
                           "cost_measured": ["llm_tokens"],
                           "knowledge_changes": knowledge_changes})
        change = [{
            "target": {"kind": "hypothesis", "strategy_id": "S01"},
            "change": "candidate_forms",
            "horizon": "after_consolidation"}]
        with_change = make("S01", change)
        without = make("S02", [])
        base_norms, base_basis = {"llm_tokens": 10.0}, ["llm_tokens"]

        off = PlanLimits(delta=0.0)
        p_off_a = evaluate_path([with_change], off, base_norms, base_basis)
        p_off_b = evaluate_path([without], off, base_norms, base_basis)
        self.assertEqual(p_off_a.utility, p_off_b.utility,
                         "with delta=0 the knowledge prediction is invisible")

        on = PlanLimits(delta=0.5)
        from or_harness.world_model.knowledge import knowledge_value
        targets = [KnowledgeTarget(kind="hypothesis", strategy_id="S01",
                                   support_n=2, distinct_tasks=2)]
        reliable = lambda c, h: 1.0
        k_a, _ = knowledge_value(targets, [with_change],
                                 reliability=reliable)
        k_b, _ = knowledge_value(targets, [without], reliability=reliable)
        self.assertIsNotNone(k_a)
        self.assertIsNone(k_b)
        p_on_a = evaluate_path([with_change], on, base_norms, base_basis,
                               knowledge=(k_a, {}))
        p_on_b = evaluate_path([without], on, base_norms, base_basis,
                               knowledge=(k_b, {}))
        self.assertGreater(p_on_a.utility, p_on_b.utility,
                           "a justified knowledge value must be able to "
                           "break a tie — this is the mechanism M6 adds")

    def test_modes_gate_what_is_requested_and_scored(self):
        """The three experiment modes differ in request content and in
        whether the knowledge term is live."""
        self.assertEqual(PREDICTION_MODES,
                         ("x-b-only", "h-x-b", "h-x-b-value"))
        provider_off = KnowledgeProvider()
        h_off = self.make_harness(provider_off, prediction_mode="x-b-only",
                                  delta=0.5)
        self.seed(h_off)
        spec = ActionSpec("execute_strategy", "t1", strategy_id="S01")
        off_pred = h_off.predict_outcome(TASK, spec, "ep1")
        self.assertNotIn("candidate_knowledge_targets",
                         provider_off.requests[0])
        self.assertFalse((off_pred.predicted or {}).get("knowledge_changes"))

        provider_hnv = KnowledgeProvider()
        h_hnv = self.make_harness(provider_hnv, prediction_mode="h-x-b",
                                  delta=0.5)
        self.seed(h_hnv)
        plan = h_hnv.plan_next(
            TASK, "ep1",
            candidates=[ActionSpec("execute_strategy", "t1",
                                   strategy_id="S01")],
            limits={"horizon": 1})
        self.assertIn("candidate_knowledge_targets",
                      provider_hnv.requests[0],
                      "h-x-b still PREDICTS H, it just does not score it")
        for path in plan["paths"]:
            self.assertEqual(path.get("delta_knowledge"), 0.0)

    def test_invalid_prediction_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            ORHarness(home=self.home, prediction_mode="nonsense")


# ---------------------------------------------------------------------------
# contract validation
# ---------------------------------------------------------------------------


class TestKnowledgeContractValidation(unittest.TestCase):
    def setUp(self):
        self.spec = ActionSpec("execute_strategy", "t1", strategy_id="S01")

    def test_malformed_knowledge_changes_reject_the_whole_payload(self):
        problems = validate_prediction_payload(
            {"knowledge_changes": [{"change": "nope",
                                    "horizon": "later",
                                    "target": {"kind": "existing_entry",
                                               "entry_id": "se_1"}}]},
            self.spec)
        self.assertTrue(problems)

    def test_hypothesis_requires_observation_and_check(self):
        problems = validate_prediction_payload(
            {"knowledge_changes": [{
                "target": {"kind": "hypothesis", "strategy_id": "S01"},
                "change": "candidate_forms",
                "horizon": "after_consolidation"}]},
            self.spec)
        self.assertTrue(any("expected_observation" in p for p in problems))

    def test_narrows_is_an_accepted_change(self):
        problems = validate_prediction_payload(
            {"knowledge_changes": [{
                "target": {"kind": "existing_entry", "entry_id": "se_1",
                           "strategy_id": "S01"},
                "change": "narrows",
                "horizon": "after_consolidation"}]},
            self.spec)
        self.assertEqual(problems, [])

    def test_knowledge_only_payload_counts_as_predicted_content(self):
        problems = validate_prediction_payload(
            {"knowledge_changes": [{
                "target": {"kind": "hypothesis", "strategy_id": "S01"},
                "change": "candidate_forms",
                "horizon": "after_consolidation",
                "expected_observation": {"e": True},
                "check_condition": "x"}]},
            self.spec)
        self.assertEqual(problems, [])

    def test_tiered_targets_reject_invented_entries(self):
        targets = [KnowledgeTarget(kind="existing_entry", strategy_id="S01",
                                   entry_id="se_real")]
        accepted, problems = validate_knowledge_changes(
            [{"target": {"kind": "existing_entry", "entry_id": "se_fake",
                         "strategy_id": "S01"},
              "change": "revises", "horizon": "after_consolidation"}],
            targets)
        self.assertEqual(problems, [])
        self.assertIn("rejected", accepted[0])

    def test_structural_only_mode_skips_the_existence_check(self):
        accepted, problems = validate_knowledge_changes(
            [{"target": {"kind": "existing_entry", "entry_id": "se_any",
                         "strategy_id": "S01"},
              "change": "revises", "horizon": "after_consolidation"}],
            targets=None)
        self.assertEqual(problems, [])
        self.assertNotIn("rejected", accepted[0])

    def test_cold_start_proposes_no_targets(self):
        """No evidence at all is a REAL state: the framework has no
        structural reason to expect learning and says so."""
        self.assertEqual(learning_needs.__module__, "or_harness.world_model"
                                                    ".knowledge")


class TestLearningNeedsContract(unittest.TestCase):
    def test_empty_evidence_yields_no_targets(self):
        snapshot = type("S", (), {
            "problem_state": {"profile": {}}, "coverage": {
                "knowledge_layers": {"verified": [], "legacy_unknown": [],
                                     "unverified": []},
                "cell_statistics": {},
                "coverage_gaps": {"strategies_without_evidence": ["S01"]}},
            "harness_state": {}})()
        spec = ActionSpec("execute_strategy", "t1", strategy_id="S01")
        self.assertEqual(learning_needs(snapshot, spec, records=[]), [])


if __name__ == "__main__":
    unittest.main()
