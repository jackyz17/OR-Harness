"""M6 closeout regressions: the six defects found in the M6 commit.

Each test reproduces the reported failure mode and asserts the fixed
behaviour. The reproduction cases came from a review of commit 5578e6f;
they are grouped here by defect rather than by module because each one
spans several layers (prediction -> execution -> induction -> feedback).

  1  unevaluated predictions counted as knowledge hits (unbound / not in
     the induction's scope / mismatched binding) + attribution not deduped
  2  feedback stages blocking each other; pending not promotable; an
     induction that changed nothing leaving predictions unevaluated forever
  3  the real HTTP prompt carrying no knowledge_changes contract; cold start
     rejecting a sound hypothesis about the action's own strategy
  4  verdicts not checking the predicted proposition (expected observation,
     preconditions, honest measurement instead of the 0.5 placeholder)
  5  the enabling extra cost never entering scoring
  6  bind_induction_outcome unusable end to end
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tests.harness.helpers import HarnessTestCase  # noqa: E402

from or_harness.api import ORHarness  # noqa: E402
from or_harness.core.storage import StorageError  # noqa: E402
from or_harness.world_model.knowledge import (  # noqa: E402
    KnowledgeTarget,
    knowledge_value,
    validate_knowledge_changes,
)
from or_harness.world_model.knowledge_track import (  # noqa: E402
    STAGE_CONSOLIDATION,
    STAGE_EXECUTION,
    class_reliability,
    evaluate_consolidation,
    evaluate_execution,
    record_stage_verdict,
    reliability_lookup,
    stage_verdict,
    unresolved_stages,
)
from or_harness.world_model.planner import (  # noqa: E402
    PlanLimits,
    evaluate_path,
)
from or_harness.world_model.prediction import (  # noqa: E402
    ActionSpec,
    OutcomePrediction,
)
from or_harness.world_model.provider import (  # noqa: E402
    SYSTEM_PROMPT,
    WorldModelProvider,
)

TASK = {"task_id": "t1", "family": "routing", "description": "toy"}

SOLVE = ("import json\n"
         "with open('result.json', 'w') as fh:\n"
         "    json.dump({'status': 'feasible', 'objective_value': 900.0,\n"
         "               'objective_bound': 1000.0, 'runtime_seconds': 0.1},"
         " fh)\n")


class ScriptedProvider(WorldModelProvider):
    """Provider returning a knowledge change when targets are offered, with
    configurable change/horizon/observation/preconditions/extra cost."""

    name = "m6-closeout"

    def __init__(self, *, change="candidate_forms",
                 horizon="after_consolidation", observation=None,
                 check_condition="the observation holds", preconditions=None,
                 extra_cost=None, quality=0.7, propose_at_cold_start=True):
        self.change = change
        self.horizon = horizon
        self.observation = observation
        self.check_condition = check_condition
        self.preconditions = preconditions
        self.extra_cost = extra_cost
        self.quality = quality
        # Real models may propose a hypothesis about the action's own
        # strategy even when no structural target was offered (the prompt
        # says so explicitly). Default on, so the cold start is exercised.
        self.propose_at_cold_start = propose_at_cold_start
        self.requests = []

    def predict(self, request):
        self.requests.append(request)
        payload = {"outcome_status": "feasible", "feasible": True,
                   "quality": self.quality, "cost": {"llm_tokens": 100.0}}
        targets = request.get("candidate_knowledge_targets") or []
        if targets or self.propose_at_cold_start:
            first = (targets[0] if targets
                     else {"kind": "hypothesis", "strategy_id": "S01"})
            item = {
                "target": ({"kind": "existing_entry",
                            "entry_id": first.get("entry_id"),
                            "strategy_id": first.get("strategy_id")}
                           if first.get("kind") == "existing_entry"
                           else {"kind": "hypothesis",
                                 "strategy_id": first.get("strategy_id")}),
                "change": self.change,
                "horizon": self.horizon,
            }
            if item["target"]["kind"] == "hypothesis":
                item["expected_observation"] = (
                    self.observation if self.observation is not None
                    else {"new_entry": True})
                item["check_condition"] = self.check_condition
            if self.preconditions:
                item["preconditions"] = list(self.preconditions)
            if self.extra_cost is not None:
                item["expected_extra_cost"] = dict(self.extra_cost)
            payload["knowledge_changes"] = [item]
        return {"payload": payload, "usage": {"completion_tokens": 20},
                "error": None, "latency_s": 0.01}


class Base(HarnessTestCase):
    def setUp(self):
        super().setUp()
        self._work = tempfile.TemporaryDirectory()
        self.addCleanup(self._work.cleanup)
        self.work = Path(self._work.name)
        self.script = self.work / "solve.py"
        self.script.write_text(SOLVE, encoding="utf-8")

    def make_harness(self, provider=None, **kwargs):
        h = ORHarness(home=self.home,
                      world_model=provider or ScriptedProvider(), **kwargs)
        self.addCleanup(h.close)
        return h

    def seed(self, h, task_id="t1", strategy="S01"):
        rec = h.execute(dict(TASK, task_id=task_id), strategy,
                        str(self.script), str(self.work), solver="highs",
                        episode_id=f"e_{task_id}")
        h.record(rec, override={"llm_tokens": 1500.0})
        return rec

    def executed_prediction(self, h, task_id="t1", strategy="S01",
                            episode="ep1", bind=True):
        """A prediction whose action really ran and was banked."""
        spec = ActionSpec("execute_strategy", task_id, strategy_id=strategy,
                          episode_id=episode)
        prediction = h.predict_outcome(dict(TASK, task_id=task_id), spec,
                                       episode)
        rec = h.execute(dict(TASK, task_id=task_id), strategy,
                        str(self.script), str(self.work), solver="highs",
                        episode_id=episode)
        if bind:
            h.bind_outcome(prediction.prediction_id, rec.action_id)
        h.record(rec, override={"llm_tokens": 1500.0})
        return prediction, rec


# ---------------------------------------------------------------------------
# DEFECT 1 — unevaluated predictions must not become knowledge hits
# ---------------------------------------------------------------------------


class TestNoCreditWithoutExecution(Base):
    def test_unbound_predictions_are_never_credited(self):
        """D1a: five predictions that never ran must not all become
        `fulfilled` on a later induction (the reported repro inflated the
        class reliability to 1.0)."""
        h = self.make_harness(ScriptedProvider(
            change="candidate_forms", horizon="after_consolidation"))
        self.seed(h, "t1")
        spec = ActionSpec("execute_strategy", "t1", strategy_id="S01")
        predictions = [h.predict_outcome(TASK, spec, f"ep{i}")
                       for i in range(5)]
        for p in predictions:
            self.assertIsNone(p.bound_action_id)
        self.seed(h, "t2")
        result = h.induce(strategy_id="S01", all_=True)
        self.assertNotIn("knowledge_feedback", result,
                         "no unexecuted prediction may be evaluated")
        for p in predictions:
            after = h.get_prediction(p.prediction_id)
            self.assertNotIn(STAGE_CONSOLIDATION,
                             (after.feedback or {}).get("knowledge", {})
                             .get("0", {}))
            # Still pending, so it can resolve once it IS executed.
            self.assertIn((0, STAGE_CONSOLIDATION),
                          unresolved_stages(after))
        self.assertEqual(h.prediction_reliability_table(), {},
                         "an empty history must not become a perfect score")

    def test_mismatched_binding_is_never_scored_as_a_knowledge_hit(self):
        """D1b: predicted S01, executed S02 — the binding already reports the
        strategy mismatch, so the H channel must refuse it too."""
        h = self.make_harness(ScriptedProvider(
            change="adds_evidence", horizon="after_execution"))
        self.seed(h, "t1")
        spec = ActionSpec("execute_strategy", "t1", strategy_id="S01")
        prediction = h.predict_outcome(TASK, spec, "ep1")
        rec = h.execute(TASK, "S02", str(self.script), str(self.work),
                        solver="highs", episode_id="ep1")
        h.record(rec, override={"llm_tokens": 1500.0})
        bound = h.bind_outcome(prediction.prediction_id, rec.action_id)
        self.assertIn("strategy_id", bound.binding_mismatch or {})
        self.assertEqual(h.evaluate_knowledge_execution(
            h.bank.get(rec.execution_id)), {},
            "evidence about a different candidate cannot resolve this claim")
        self.assertIsNone(stage_verdict(
            h.get_prediction(prediction.prediction_id), 0, STAGE_EXECUTION))

    def test_execution_outside_the_induction_scope_is_not_credited(self):
        """D1c: an induction consolidating OTHER evidence gives no verdict to
        a prediction about an execution it never saw."""
        h = self.make_harness(ScriptedProvider(
            change="candidate_forms", horizon="after_consolidation"))
        self.seed(h, "t1")
        prediction, rec = self.executed_prediction(h, "t1")
        self.seed(h, "t2")
        # An explicit scope naming a DIFFERENT execution only.
        other = self.seed(h, "t3")
        result = h.induce(strategy_id="S01", all_=True,
                          execution_ids=[other.execution_id])
        feedback = (result.get("knowledge_feedback") or {})
        self.assertNotIn(prediction.prediction_id, feedback,
                         "an execution outside the named scope must not "
                         "resolve the prediction")

    def test_induction_scope_is_recorded_in_the_verdict(self):
        """The verdict carries which execution and scope justified it."""
        h = self.make_harness(ScriptedProvider(
            change="candidate_forms", horizon="after_consolidation"))
        self.seed(h, "t1")
        prediction, rec = self.executed_prediction(h, "t1")
        self.seed(h, "t2")
        h.induce(strategy_id="S01", all_=True)
        verdict = stage_verdict(h.get_prediction(prediction.prediction_id),
                                0, STAGE_CONSOLIDATION)
        self.assertIsNotNone(verdict)
        self.assertEqual(verdict.get("consolidated_execution"),
                         rec.execution_id)
        self.assertIn("scope_basis", verdict)


class TestAttributionDeduplication(unittest.TestCase):
    def test_one_outcome_is_credited_once_per_class(self):
        """D1d: several actions predicting the SAME proposition must not each
        score a hit for the single event that resolved it."""
        proposition = ["se_1", "revises", "after_consolidation"]
        records = [{"change": "revises", "horizon": "after_consolidation",
                    "status": "fulfilled", "proposition": proposition,
                    "prediction_id": f"wp_{i}", "created_at": float(i)}
                   for i in range(6)]
        table = class_reliability(records)
        entry = table["revises|after_consolidation"]
        self.assertEqual(entry["n"], 1,
                         "six actions on one proposition is ONE outcome")
        # Without the attribution rule the class would look perfect.
        self.assertNotEqual(entry.get("reliability"), 1.0)

    def test_evidence_accumulation_still_counts_per_action(self):
        """adds_evidence genuinely accumulates: not double counting."""
        records = [{"change": "adds_evidence", "horizon": "after_execution",
                    "status": "fulfilled",
                    "proposition": ["S01", "adds_evidence",
                                    "after_execution"],
                    "prediction_id": f"wp_{i}", "created_at": float(i)}
                   for i in range(6)]
        table = class_reliability(records)
        self.assertEqual(table["adds_evidence|after_execution"]["n"], 6)


# ---------------------------------------------------------------------------
# DEFECT 2 — stages must not block each other
# ---------------------------------------------------------------------------


class TestFeedbackStagesIndependent(Base):
    def test_record_then_compare_works(self):
        """D2a: record() writes the H verdict into its own partition; the X/B
        comparison must still run afterwards."""
        h = self.make_harness(ScriptedProvider(
            change="adds_evidence", horizon="after_execution"))
        self.seed(h, "t1")
        spec = ActionSpec("execute_strategy", "t1", strategy_id="S01")
        prediction = h.predict_outcome(TASK, spec, "ep1")
        rec = h.execute(TASK, "S01", str(self.script), str(self.work),
                        solver="highs", episode_id="ep1")
        h.bind_outcome(prediction.prediction_id, rec.action_id)
        h.record(rec, override={"llm_tokens": 1500.0})
        compared = h.compare_prediction(prediction.prediction_id)
        self.assertTrue((compared.feedback or {}).get("compared"))
        # Idempotent replay of the X/B stage.
        again = h.compare_prediction(prediction.prediction_id)
        self.assertEqual((again.feedback or {}).get("execution_id"),
                         (compared.feedback or {}).get("execution_id"))

    def test_pending_is_promotable_and_terminal_is_final(self):
        """D2b: pending -> fulfilled must be allowed; fulfilled is then
        final."""
        h = self.make_harness(ScriptedProvider(
            change="adds_evidence", horizon="after_execution"))
        spec = ActionSpec("execute_strategy", "t1", strategy_id="S01")
        prediction = h.predict_outcome(TASK, spec, "ep1")
        # The stage that can resolve here is the one the prediction declares.
        stage = prediction.predicted["knowledge_changes"][0]["horizon"]
        self.assertEqual(stage, STAGE_EXECUTION)
        self.assertTrue(record_stage_verdict(
            prediction, 0, stage, "pending", {"reason": "later"}))
        self.assertIn((0, stage), unresolved_stages(prediction),
                      "pending stays on the work list")
        self.assertTrue(record_stage_verdict(
            prediction, 0, stage, "fulfilled", {"reason": "now"}))
        self.assertEqual(stage_verdict(prediction, 0, stage)["status"],
                         "fulfilled")
        self.assertFalse(record_stage_verdict(
            prediction, 0, stage, "missed", {}),
            "a settled judgement is never rewritten")
        self.assertNotIn((0, stage), unresolved_stages(prediction))

    def test_induction_that_changed_nothing_still_evaluates(self):
        """D2c: an induction over the prediction's own evidence with no entry
        change is the NEGATIVE outcome, not a reason to skip forever."""
        h = self.make_harness(ScriptedProvider(
            change="candidate_forms", horizon="after_consolidation"))
        self.seed(h, "t1")
        prediction, rec = self.executed_prediction(h, "t1")
        # Threshold-free: this induction runs for the predicted strategy over
        # its own recorded evidence but (S01 already covered) changes
        # nothing for it.
        result = h.induce(strategy_id="S01", all_=False)
        feedback = result.get("knowledge_feedback") or {}
        verdict = stage_verdict(h.get_prediction(prediction.prediction_id),
                                0, STAGE_CONSOLIDATION)
        self.assertIn(prediction.prediction_id, feedback)
        self.assertEqual(verdict["status"], "missed",
                         "the opportunity arrived and nothing formed")
        self.assertTrue(verdict.get("induction_changed_nothing"))


# ---------------------------------------------------------------------------
# DEFECT 3 — the real prompt must carry the contract
# ---------------------------------------------------------------------------


class TestPromptContract(unittest.TestCase):
    def test_system_prompt_documents_knowledge_changes(self):
        """D3a: a real model can only produce valid H predictions if the
        production prompt actually asks for them."""
        for token in ("knowledge_changes", "target", "change", "horizon",
                      "expected_observation", "check_condition",
                      "preconditions", "prediction_basis",
                      "candidate_knowledge_targets"):
            self.assertIn(token, SYSTEM_PROMPT,
                          f"the prompt must document {token!r}")
        for kind in ("existing_entry", "hypothesis"):
            self.assertIn(kind, SYSTEM_PROMPT)
        for change in ("adds_evidence", "supports", "revises", "refutes",
                       "candidate_forms", "narrows"):
            self.assertIn(change, SYSTEM_PROMPT)

    def test_prompt_keeps_the_no_solving_rule(self):
        """The earlier G10 fix must survive the contract expansion."""
        self.assertIn("Do NOT solve the problem", SYSTEM_PROMPT)


class TestColdStartHypothesis(Base):
    def test_a_hypothesis_about_this_actions_strategy_is_admitted(self):
        """D3b: with NO structural targets proposed, a hypothesis naming the
        action's own strategy must still be accepted."""
        accepted, problems = validate_knowledge_changes(
            [{"target": {"kind": "hypothesis", "strategy_id": "S01"},
              "change": "candidate_forms",
              "horizon": "after_consolidation",
              "expected_observation": {"new_entry": True},
              "check_condition": "an entry exists afterwards"}],
            targets=[], action_strategy_id="S01")
        self.assertEqual(problems, [])
        self.assertEqual(len(accepted), 1)
        self.assertNotIn("rejected", accepted[0])

    def test_a_hypothesis_about_an_unrelated_strategy_is_still_rejected(self):
        accepted, _ = validate_knowledge_changes(
            [{"target": {"kind": "hypothesis", "strategy_id": "S99"},
              "change": "candidate_forms",
              "horizon": "after_consolidation",
              "expected_observation": {"new_entry": True},
              "check_condition": "x"}],
            targets=[], action_strategy_id="S01")
        self.assertIn("rejected", accepted[0])

    def test_invented_existing_entry_is_still_rejected(self):
        """The relaxation must not open the door to invented knowledge."""
        accepted, _ = validate_knowledge_changes(
            [{"target": {"kind": "existing_entry", "entry_id": "se_fake",
                         "strategy_id": "S01"},
              "change": "revises", "horizon": "after_consolidation"}],
            targets=[KnowledgeTarget(kind="existing_entry",
                                     strategy_id="S01",
                                     entry_id="se_real")],
            action_strategy_id="S01")
        self.assertIn("rejected", accepted[0])

    def test_cold_start_prediction_is_accepted_end_to_end(self):
        h = self.make_harness(ScriptedProvider(
            change="candidate_forms", horizon="after_consolidation"))
        # No evidence at all: the framework proposes no targets.
        spec = ActionSpec("execute_strategy", "t1", strategy_id="S01")
        prediction = h.predict_outcome(TASK, spec, "ep1")
        changes = (prediction.predicted or {}).get("knowledge_changes") or []
        self.assertEqual(len(changes), 1,
                         "a sound hypothesis about the action's own strategy "
                         "must survive a cold start")
        self.assertNotIn("rejected", changes[0])


# ---------------------------------------------------------------------------
# DEFECT 4 — the verdict must judge the predicted proposition
# ---------------------------------------------------------------------------


class TestVerdictJudgesTheProposition(Base):
    def test_unmet_precondition_is_pending_not_fulfilled(self):
        """D4a: a hypothesis requiring a contrast run, with no contrast run,
        must not be fulfilled merely because a record appeared."""
        target = KnowledgeTarget(kind="hypothesis", strategy_id="S01")
        record = self.make_record(task_id="t1", strategy_id="S01")
        status, detail = evaluate_execution(
            {"change": "candidate_forms", "horizon": "after_execution",
             "preconditions": ["contrast_execution"]},
            target, record)
        self.assertEqual(status, "pending")
        self.assertIn("contrast_execution", detail["unmet_preconditions"])

    def test_contradicted_expected_observation_is_contradicted(self):
        target = KnowledgeTarget(kind="hypothesis", strategy_id="S01")
        record = self.make_record(task_id="t1", strategy_id="S01",
                                  feasible=False, status="infeasible")
        status, detail = evaluate_execution(
            {"change": "adds_evidence", "horizon": "after_execution",
             "expected_observation": {"feasible": True}},
            target, record)
        self.assertEqual(status, "contradicted")

    def test_uncheckable_observation_is_not_silently_fulfilled(self):
        target = KnowledgeTarget(kind="hypothesis", strategy_id="S01")
        record = self.make_record(task_id="t1", strategy_id="S01")
        status, detail = evaluate_execution(
            {"change": "adds_evidence", "horizon": "after_execution",
             "expected_observation": {"no_such_field": 1}},
            target, record)
        self.assertEqual(status, "pending")
        self.assertIn("could not be checked", detail["reason"])

    def test_placeholder_quality_is_not_treated_as_measured(self):
        """D4b: a feasible result with no gap/optimality basis has NO measured
        quality, so a support claim cannot be judged against it."""
        target = KnowledgeTarget(kind="existing_entry", strategy_id="S01",
                                 entry_id="se_1", cell_token="",
                                 quality_interval=(0.0, 0.5))
        record = self.make_record(task_id="t1", strategy_id="S01",
                                  feasible=True, objective=100.0)
        record.quality.pop("gap", None)
        record.quality["status"] = "feasible"   # no gap, no optimality
        status, detail = evaluate_execution(
            {"change": "supports", "horizon": "after_execution"},
            target, record)
        self.assertEqual(status, "pending")
        self.assertIn("not measured", detail["reason"])

    def test_optimal_status_is_a_real_measurement(self):
        target = KnowledgeTarget(kind="existing_entry", strategy_id="S01",
                                 entry_id="se_1", cell_token="",
                                 quality_interval=(0.5, 1.0))
        record = self.make_record(task_id="t1", strategy_id="S01")
        record.quality.pop("gap", None)
        record.quality["status"] = "optimal"
        status, detail = evaluate_execution(
            {"change": "supports", "horizon": "after_execution"},
            target, record)
        self.assertEqual(status, "fulfilled")
        self.assertEqual(detail["observed_quality"], 1.0)
        self.assertEqual(detail["quality_basis"], "optimal status")


# ---------------------------------------------------------------------------
# DEFECT 5 — the enabling extra cost must reach the score
# ---------------------------------------------------------------------------


class TestExtraCostEntersScoring(Base):
    def test_extra_cost_changes_the_utility(self):
        """D5: raising the declared enabling cost from 1 to 1e9 tokens must
        change K / the path utility."""
        table = {"revises|after_consolidation": {
            "n": 50, "reliability": 1.0, "basis": "measured"}}
        lookup = reliability_lookup(table)
        targets = [KnowledgeTarget(kind="hypothesis", strategy_id="S01",
                                   support_n=2, distinct_tasks=2)]

        def build(extra):
            return OutcomePrediction(
                prediction_id=f"wp_{extra}", input_snapshot_id="bs_1",
                action_spec=ActionSpec("execute_strategy", "t1",
                                       strategy_id="S01"),
                status="valid",
                predicted={"quality": 0.7, "feasible": True,
                           "failure_prob": 0.1,
                           "cost": {"llm_tokens": 10.0},
                           "cost_measured": ["llm_tokens"],
                           "knowledge_changes": [{
                               "target": {"kind": "hypothesis",
                                          "strategy_id": "S01"},
                               "change": "revises",
                               "horizon": "after_consolidation",
                               "preconditions": ["contrast_execution"],
                               "expected_extra_cost": {"llm_tokens": extra}}]})

        limits = PlanLimits(delta=0.5, cost_weights={"llm_tokens": 1.0})
        norms, basis = {"llm_tokens": 10.0}, ["llm_tokens"]
        cheap_k, cheap_detail = knowledge_value(targets, [build(1.0)],
                                                reliability=lookup)
        dear_k, dear_detail = knowledge_value(targets, [build(1e9)],
                                              reliability=lookup)
        self.assertIsNotNone(cheap_k)
        self.assertEqual(cheap_k, dear_k,
                         "the knowledge VALUE itself is unchanged")
        cheap_path = evaluate_path([build(1.0)], limits, norms, basis,
                                   knowledge=(cheap_k, cheap_detail))
        dear_path = evaluate_path([build(1e9)], limits, norms, basis,
                                  knowledge=(dear_k, dear_detail))
        self.assertLess(dear_path.utility, cheap_path.utility,
                        "the enabling spend must make the path worse")
        charged = dear_path.knowledge_detail.get("extra_cost_charged")
        self.assertIsNotNone(charged,
                             "the charge must be reported, not silent")
        self.assertGreater(charged["normalized"], 0.0)

    def test_uncosted_after_consolidation_gain_is_still_refused(self):
        targets = [KnowledgeTarget(kind="hypothesis", strategy_id="S01",
                                   support_n=2, distinct_tasks=2)]
        prediction = OutcomePrediction(
            prediction_id="wp_x", input_snapshot_id="bs_1",
            action_spec=ActionSpec("execute_strategy", "t1",
                                   strategy_id="S01"),
            status="valid",
            predicted={"knowledge_changes": [{
                "target": {"kind": "hypothesis", "strategy_id": "S01"},
                "change": "revises", "horizon": "after_consolidation",
                "preconditions": ["verification_check"]}]})
        value, _ = knowledge_value(
            targets, [prediction],
            reliability={"revises|after_consolidation": 1.0}.get or (
                lambda c, h: 1.0))
        self.assertIsNone(value,
                          "no declared enabling cost => no granted value")


# ---------------------------------------------------------------------------
# DEFECT 6 — the induction binding must be usable
# ---------------------------------------------------------------------------


class TestInductionBindingUsable(Base):
    def test_assess_accept_bind_chain(self):
        """D6: the documented chain must work end to end."""
        h = self.make_harness(ScriptedProvider())
        self.seed(h, "t1")
        self.seed(h, "t2")
        bundles = h.induction_candidates()
        self.assertTrue(bundles, "two tasks of evidence must yield a bundle")
        assessment = h.assess_induction(bundles[0])
        assessment_id = assessment["assessment_id"]
        accepted = h.accept_induction(assessment)
        self.assertTrue(accepted.get("adoption_action_id"))
        # The assessment id is discoverable where it was actually written.
        decision = h.actions.get(assessment["decision_action_id"])
        self.assertEqual((decision.outcome or {}).get("assessment_id"),
                         assessment_id)
        result = h.bind_induction_outcome(assessment_id)
        self.assertTrue(result.get("compared"))
        self.assertIn(result["verdict"]["status"],
                      ("fulfilled", "missed", "inconclusive"))

    def test_binding_is_idempotent(self):
        h = self.make_harness(ScriptedProvider())
        self.seed(h, "t1")
        self.seed(h, "t2")
        assessment = h.assess_induction(h.induction_candidates()[0])
        h.accept_induction(assessment)
        first = h.bind_induction_outcome(assessment["assessment_id"])
        second = h.bind_induction_outcome(assessment["assessment_id"])
        self.assertEqual(first["verdict"]["status"],
                         second["verdict"]["status"])
        self.assertIn("already bound", second.get("note", ""))

    def test_unknown_assessment_is_still_an_error(self):
        h = self.make_harness(ScriptedProvider())
        with self.assertRaises(StorageError):
            h.bind_induction_outcome("ia_does_not_exist")

    def test_acceptance_carries_the_induction_result_and_scope(self):
        """The adoption record must hold what the binding reads: the
        transition AND the evidence scope."""
        h = self.make_harness(ScriptedProvider())
        rec1 = self.seed(h, "t1")
        rec2 = self.seed(h, "t2")
        assessment = h.assess_induction(h.induction_candidates()[0])
        accepted = h.accept_induction(assessment)
        adoption = h.actions.get(accepted["adoption_action_id"])
        linked = (adoption.outcome or {}).get("induction_result") or {}
        self.assertEqual(linked.get("assessment_id"),
                         assessment["assessment_id"])
        self.assertIn("knowledge_delta", linked)
        self.assertTrue(linked.get("execution_ids"),
                        "the consolidated scope must be recorded")


# ---------------------------------------------------------------------------
# The consolidation verdict itself
# ---------------------------------------------------------------------------


class TestConsolidationVerdict(unittest.TestCase):
    def test_candidate_forms_fulfilled_only_for_the_named_strategy(self):
        target = KnowledgeTarget(kind="hypothesis", strategy_id="S01")
        status, detail = evaluate_consolidation(
            {"change": "candidate_forms", "horizon": "after_consolidation",
             "expected_observation": {"new_entry": True}},
            target, [], ["se_new"], ["S02"])
        self.assertEqual(status, "missed")


if __name__ == "__main__":
    unittest.main()
