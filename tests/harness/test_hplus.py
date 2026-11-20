"""Same-call capability-gain payload compatibility for strategy predictions.

Cover shared G/C/R independence, missing values, legacy assessment serialization,
identity and capability evidence. Binary dispatch and receipt feedback are tested
through the current workflow in test_hplus_binary.
"""

from decision_fixtures import planned_prediction
import json
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tests.harness.helpers import HarnessTestCase  # noqa: E402

from or_harness.api import ORHarness  # noqa: E402
from or_harness.strategy.embedding_index import (  # noqa: E402
    LocalHashEmbeddingBackend,
)
from or_harness.world_model.contracts import (  # noqa: E402
    CapabilityGain,
    validate_strategy_outcome,
)
from or_harness.world_model.provider import WorldModelProvider  # noqa: E402
from or_harness.world_model.strategy_prediction import (  # noqa: E402
    build_strategy_outcome_request,
    parse_strategy_outcome_payload,
)

EMBEDDING_ENV_KEYS = ("OR_EMBEDDING_BACKEND", "OR_EMBEDDING_BASE_URL",
                      "OR_EMBEDDING_MODEL", "OR_EMBEDDING_API_KEY")

TASK = {"task_id": "t1", "family": "routing",
        "description": "A small routing instance.",
        "spec": {"n_vars": 4, "n_constraints": 3, "n_int_vars": 4}}

BASE_PAYLOAD = {
    "benefit": {"kind": "solution_quality",
                "metric": "normalized_objective_gap", "unit": "1-gap",
                "value": 0.8,
                "baseline": {"kind": "conditional_stats", "value": 0.7}},
    "cost": {"solver_runtime_s": 3.0},
}

GAIN_PAYLOAD = {
    "assessment": "expected",
    "claim": "builds a reusable warm-start structure",
    "applies_to": ["routing with high resource_coupling"],
    "expected_changes": [
        {"metric": "solver_runtime_s", "direction": "decrease",
         "value": 20, "unit": "seconds", "value_kind": "relative",
         "beneficial_direction": "decrease",
         "baseline": {"kind": "conditional_stats", "value": 30}}],
    "evidence_required": ["reuse on a similar task and solve at least as "
                          "well with less runtime"],
    "verification_conditions": [{"condition": "a later task shows a >=10% "
                                            "runtime drop"}],
    "degradation_risk": {"events": [{"event": "timeout", "probability": 0.1}]},
    "uncertainty": ["may not generalize to other structures"],
    "basis": ["capability.sources.m"],
}


class StubProvider(WorldModelProvider):
    name = "stub-hplus"

    def __init__(self, payload=None):
        self.payload = payload if payload is not None else dict(
            BASE_PAYLOAD, capability_gain=GAIN_PAYLOAD)
        self.requests = []

    def predict(self, request, timeout_s=None):
        self.requests.append(request)
        return {"payload": self.payload,
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
                "error": None, "latency_s": 0.01}


class HPlusCase(HarnessTestCase):
    def setUp(self):
        super().setUp()
        saved = {key: os.environ.pop(key, None) for key in EMBEDDING_ENV_KEYS}

        def restore():
            for key, value in saved.items():
                if value is not None:
                    os.environ[key] = value
        self.addCleanup(restore)
        self.backend = LocalHashEmbeddingBackend()
        self.provider = StubProvider()
        self.h = ORHarness(home=self.home, world_model=self.provider,
                           embedding=self.backend)
        self.addCleanup(self.h.close)


# ---------------------------------------------------------------------------
# 1 & 4. the same call, and no invented composite score
# ---------------------------------------------------------------------------


class TestSameCallAndShape(HPlusCase):

    def test_the_gain_is_parsed_from_the_same_answer(self):
        prediction = planned_prediction(self.h,
            TASK, {"action_type": "execute_strategy", "strategy_id": "S01"},
            "ep1", benefit_convention={'kind': 'solution_quality', 'metric': 'normalized_objective_gap', 'unit': '1-gap'})
        self.assertEqual(prediction.status, "valid")
        self.assertTrue(prediction.claims_capability_gain)
        gain = prediction.capability_gain
        self.assertEqual(gain.claim, GAIN_PAYLOAD["claim"])
        self.assertEqual(gain.applies_to, GAIN_PAYLOAD["applies_to"])
        # ONE model call carried benefit + cost + gain.
        self.assertEqual(len(self.provider.requests), 1)

    def test_the_gain_keeps_each_metric_in_its_own_unit(self):
        prediction = planned_prediction(self.h,
            TASK, {"action_type": "execute_strategy", "strategy_id": "S01"},
            "ep1", benefit_convention={'kind': 'solution_quality', 'metric': 'normalized_objective_gap', 'unit': '1-gap'})
        change = prediction.capability_gain.expected_changes[0]
        # Not a 0-1 score: the metric's own unit and a signed direction.
        self.assertEqual(change.metric, "solver_runtime_s")
        self.assertEqual(change.unit, "seconds")
        self.assertEqual(change.direction, "decrease")
        self.assertTrue(change.is_improvement)

    def test_the_request_whitelists_the_gain_field(self):
        context = self.h.build_prediction_context(TASK, "ep1")
        from or_harness.world_model.contracts import CandidateRef
        candidate = CandidateRef(action_type="execute_strategy",
                                 strategy_id="S01", task_id="t1",
                                 episode_id="ep1")
        request = build_strategy_outcome_request(context, candidate)
        self.assertIn("capability_gain",
                      request["output_contract"]["allowed_fields"])

    def test_the_gain_round_trips_through_storage(self):
        prediction = planned_prediction(self.h,
            TASK, {"action_type": "execute_strategy", "strategy_id": "S01"},
            "ep1", benefit_convention={'kind': 'solution_quality', 'metric': 'normalized_objective_gap', 'unit': '1-gap'})
        reloaded = self.h.get_strategy_outcome_prediction(
            prediction.prediction_id)
        self.assertTrue(reloaded.claims_capability_gain)
        self.assertEqual(reloaded.capability_gain.claim,
                         GAIN_PAYLOAD["claim"])
        # The verification conditions survive too.
        self.assertEqual(
            reloaded.capability_gain.verification_conditions[0].condition,
            "a later task shows a >=10% runtime drop")

    def test_a_single_string_basis_survives_storage_round_trip(self):
        # A model may answer a text-list field with ONE string. Parsing and
        # storing must keep it as one entry (never one entry per character),
        # and the frozen H+ decision must not change.
        self.provider.payload = dict(BASE_PAYLOAD, capability_gain=dict(
            GAIN_PAYLOAD, decision=1,
            basis="standard method",
            applies_to="routing with high resource_coupling"))
        prediction = planned_prediction(self.h,
            TASK, {"action_type": "execute_strategy", "strategy_id": "S01"},
            "ep1", benefit_convention={'kind': 'solution_quality', 'metric': 'normalized_objective_gap', 'unit': '1-gap'})
        self.assertEqual(prediction.capability_gain.basis,
                         ["standard method"])
        self.assertEqual(prediction.capability_gain.decision, 1)
        reloaded = self.h.get_strategy_outcome_prediction(
            prediction.prediction_id)
        self.assertEqual(reloaded.capability_gain.basis,
                         ["standard method"])
        self.assertEqual(reloaded.capability_gain.applies_to,
                         ["routing with high resource_coupling"])
        self.assertEqual(reloaded.capability_gain.decision, 1)


# ---------------------------------------------------------------------------
# 2. explanatory, never the utility
# ---------------------------------------------------------------------------


class TestGainIsNotUtility(HPlusCase):

    def test_the_plan_carries_the_gain_as_read_only_info(self):
        from or_harness.world_model.prediction import ActionSpec
        plan = self.h.plan_next(
            TASK, "ep1",
            candidates=[ActionSpec("execute_strategy", "t1",
                                   strategy_id="S01", solver="highs")],
            limits={"horizon": 1, 'benefit_kind': 'solution_quality', 'benefit_metric': 'normalized_objective_gap', 'benefit_unit': '1-gap'})
        self.assertEqual(plan["model_calls_made"], 1)
        entry = plan["candidates"][0]
        gain = entry.get("capability_gain")
        self.assertIsNotNone(gain)
        self.assertEqual(gain["kind"], "frozen_induction_signal")
        self.assertEqual(gain['explanation'],GAIN_PAYLOAD['claim'])
        self.assertEqual(gain['decision_status'],'missing')
        # The score itself is benefit/cost/risk only — no capability term.
        self.assertNotIn('H+',entry['score']['scoring_basis']['terms'])
        self.assertTrue(set(entry['score']['scoring_basis']['terms']) <= {'G','C','R'})

    def test_a_large_gain_does_not_change_the_ranking(self):
        # Two candidates with the SAME benefit/cost/risk but a gain on only
        # one: their utilities must be equal (the gain never enters).
        class TwoCandidateProvider(WorldModelProvider):
            name = "stub-two"

            def predict(self, request, timeout_s=None):
                sid = (request.get("candidate") or {}).get("strategy_id")
                payload = dict(BASE_PAYLOAD)
                if sid == "S01":
                    payload["capability_gain"] = dict(GAIN_PAYLOAD,decision=1)
                return {"payload": payload,
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                        "error": None, "latency_s": 0.01}

        h = ORHarness(home=self.home, world_model=TwoCandidateProvider(),
                      embedding=self.backend)
        self.addCleanup(h.close)
        from or_harness.world_model.prediction import ActionSpec
        plan = h.plan_next(
            TASK, "ep1",
            candidates=[ActionSpec("execute_strategy", "t1",
                                   strategy_id="S01", solver="highs"),
                        ActionSpec("execute_strategy", "t1",
                                   strategy_id="S02", solver="highs")],
            limits={"horizon": 1, 'benefit_kind': 'solution_quality', 'benefit_metric': 'normalized_objective_gap', 'benefit_unit': '1-gap'})
        utilities = {c["action_spec"]["strategy_id"]: c["score"]["utility"]
                     for c in plan["candidates"]}
        self.assertEqual(utilities["S01"], utilities["S02"],
                         "H+ must not enter the utility")
        # But the gain is still reported for the candidate that has one.
        gains = {c["action_spec"]["strategy_id"]:
                 c.get("capability_gain") for c in plan["candidates"]}
        self.assertIsNotNone(gains["S01"])
        self.assertEqual(gains['S01']['decision'],1)
        self.assertEqual(gains['S02']['decision'],0)
        self.assertEqual(gains['S02']['decision_status'],'missing')


# ---------------------------------------------------------------------------
# 3. absent / empty gain does not invalidate the rest
# ---------------------------------------------------------------------------


class TestGainAbsence(HPlusCase):

    def test_a_prediction_with_no_gain_block_records_why_it_is_absent(self):
        provider = StubProvider(payload=dict(BASE_PAYLOAD))
        h = ORHarness(home=self.home, world_model=provider,
                      embedding=self.backend)
        self.addCleanup(h.close)
        prediction = planned_prediction(h,
            TASK, {"action_type": "execute_strategy", "strategy_id": "S01"},
            "ep1", benefit_convention={'kind': 'solution_quality', 'metric': 'normalized_objective_gap', 'unit': '1-gap'})
        self.assertEqual(prediction.status, "valid")
        self.assertIsNone(prediction.capability_gain)
        self.assertFalse(prediction.claims_capability_gain)
        # A MISSING block is not a silent null: the contract now REQUIRES a
        # stance, so the framework records that the model did not state
        # one — never read as "no gain".
        unsupported = prediction.trace.unsupported_fields
        self.assertIn("capability_gain", unsupported)
        self.assertIn("not predicted by the model", unsupported[
            "capability_gain"])
        self.assertIn("stance", unsupported["capability_gain"])
        # The benefit/cost are unaffected.
        self.assertEqual(prediction.benefit.value, 0.8)
        self.assertIsNotNone(prediction.cost)

    def test_an_empty_gain_block_is_no_gain_not_a_positive(self):
        provider = StubProvider(payload=dict(BASE_PAYLOAD,
                                             capability_gain={}))
        h = ORHarness(home=self.home, world_model=provider,
                      embedding=self.backend)
        self.addCleanup(h.close)
        prediction = planned_prediction(h,
            TASK, {"action_type": "execute_strategy", "strategy_id": "S01"},
            "ep1", benefit_convention={'kind': 'solution_quality', 'metric': 'normalized_objective_gap', 'unit': '1-gap'})
        self.assertEqual(prediction.status, "valid")
        self.assertFalse(prediction.claims_capability_gain)
        self.assertTrue(any("no gain claimed" in n
                            for n in prediction.notes))
        # An empty block carries NO stance: it reads as unassessed, never
        # as "the model said none".
        self.assertFalse(prediction.capability_gain.assessed)
        self.assertTrue(any("no assessment" in n for n in prediction.notes))

    def test_a_malformed_gain_drops_only_the_gain_block(self):
        provider = StubProvider(payload=dict(
            BASE_PAYLOAD,
            capability_gain={"expected_changes": [{"direction": "increase"}]}))
        h = ORHarness(home=self.home, world_model=provider,
                      embedding=self.backend)
        self.addCleanup(h.close)
        prediction = planned_prediction(h,
            TASK, {"action_type": "execute_strategy", "strategy_id": "S01"},
            "ep1", benefit_convention={'kind': 'solution_quality', 'metric': 'normalized_objective_gap', 'unit': '1-gap'})
        # The malformed gain is DROPPED with a reason, and the valid
        # benefit/cost prediction it accompanied is KEPT: a local error in
        # the explanatory extra never destroys the forecast.
        self.assertEqual(prediction.status, "valid")
        self.assertIsNone(prediction.capability_gain)
        self.assertEqual(prediction.benefit.value, 0.8)
        self.assertIsNotNone(prediction.cost)
        unsupported = prediction.trace.unsupported_fields
        self.assertIn("capability_gain", unsupported)
        self.assertIn("malformed", unsupported["capability_gain"])
        self.assertTrue(any("only the gain block is affected" in n
                            for n in prediction.notes))

    def test_a_non_object_gain_drops_only_the_gain_block(self):
        provider = StubProvider(payload=dict(BASE_PAYLOAD,
                                             capability_gain="more gain!"))
        h = ORHarness(home=self.home, world_model=provider,
                      embedding=self.backend)
        self.addCleanup(h.close)
        prediction = planned_prediction(h,
            TASK, {"action_type": "execute_strategy", "strategy_id": "S01"},
            "ep1", benefit_convention={'kind': 'solution_quality', 'metric': 'normalized_objective_gap', 'unit': '1-gap'})
        self.assertEqual(prediction.status, "valid")
        self.assertIsNone(prediction.capability_gain)
        self.assertIn("malformed",
                      prediction.trace.unsupported_fields["capability_gain"])


# ---------------------------------------------------------------------------
# 3b. the stance is explicit: assessment must be stated and observable
# ---------------------------------------------------------------------------


class TestGainAssessment(HPlusCase):

    def _predict_with(self, gain_payload):
        provider = StubProvider(payload=dict(
            BASE_PAYLOAD, capability_gain=gain_payload))
        h = ORHarness(home=self.home, world_model=provider,
                      embedding=self.backend)
        self.addCleanup(h.close)
        return h, planned_prediction(h,
            TASK, {"action_type": "execute_strategy", "strategy_id": "S01"},
            "ep1", benefit_convention={'kind': 'solution_quality', 'metric': 'normalized_objective_gap', 'unit': '1-gap'})

    def test_all_three_assessments_parse_and_round_trip(self):
        for stance in ("expected", "none", "insufficient_basis"):
            with self.subTest(assessment=stance):
                payload = dict(GAIN_PAYLOAD, assessment=stance)
                h, prediction = self._predict_with(payload)
                self.assertEqual(prediction.status, "valid")
                gain = prediction.capability_gain
                self.assertEqual(gain.assessment, stance)
                self.assertTrue(gain.assessed)
                # Round-trip through storage keeps the stance.
                reloaded = h.get_strategy_outcome_prediction(
                    prediction.prediction_id)
                self.assertEqual(reloaded.capability_gain.assessment, stance)
                # And through the contract dict.
                self.assertEqual(gain.to_dict()["assessment"], stance)

    def test_an_illegal_assessment_drops_only_the_gain_block(self):
        h, prediction = self._predict_with(
            dict(GAIN_PAYLOAD, assessment="probably"))
        # An illegal stance is a MALFORMED block: dropped with its reason,
        # and the valid benefit/cost prediction beside it is KEPT.
        self.assertEqual(prediction.status, "valid")
        self.assertIsNone(prediction.capability_gain)
        self.assertEqual(prediction.benefit.value, 0.8)
        self.assertIsNotNone(prediction.cost)
        unsupported = prediction.trace.unsupported_fields
        self.assertIn("capability_gain", unsupported)
        self.assertIn("malformed", unsupported["capability_gain"])
        self.assertIn("assessment", unsupported["capability_gain"])
        self.assertTrue(any("only the gain block is affected" in n
                            for n in prediction.notes))

    def test_a_none_assessment_survives_prediction_storage(self):
        # A stated "no gain" is a real, falsifiable stance: it must be
        # archived like any other, or it would vanish exactly like a
        # silent null — the confusion the stance requirement removes.
        h, prediction = self._predict_with({"assessment": "none"})
        self.assertEqual(prediction.status, "valid")
        self.assertFalse(prediction.claims_capability_gain)
        self.assertEqual(prediction.capability_gain.assessment, "none")
        reloaded = h.get_strategy_outcome_prediction(prediction.prediction_id)
        self.assertEqual(reloaded.capability_gain.assessment, "none")

    def test_an_old_payload_without_assessment_reads_as_unassessed(self):
        # An old payload (no assessment field) is NOT retroactively read
        # as "no gain": it stays unassessed, and a claimed gain still
        # archives a trace.
        old_payload = {k: v for k, v in GAIN_PAYLOAD.items()
                       if k != "assessment"}
        h, prediction = self._predict_with(old_payload)
        self.assertEqual(prediction.status, "valid")
        gain = prediction.capability_gain
        self.assertEqual(gain.assessment, "")
        self.assertFalse(gain.assessed)
        self.assertTrue(gain.claimed,
                        "claimed semantics are unchanged by assessment")
        self.assertTrue(prediction.claims_capability_gain)
        # Round-trip: still unassessed after storage.
        reloaded = h.get_strategy_outcome_prediction(
            prediction.prediction_id)
        self.assertEqual(reloaded.capability_gain.assessment, "")
        self.assertFalse(reloaded.capability_gain.assessed)



# ---------------------------------------------------------------------------
# 5. a predicted gain is never a verified capability
# ---------------------------------------------------------------------------


class TestGainIsNotVerifiedCapability(HPlusCase):

    def test_nothing_is_written_into_the_capability_evidence(self):
        before = self.h.capability_evidence().to_dict()
        planned_prediction(self.h,
            TASK, {"action_type": "execute_strategy", "strategy_id": "S01"},
            "ep1", benefit_convention={'kind': 'solution_quality', 'metric': 'normalized_objective_gap', 'unit': '1-gap'})
        after = self.h.capability_evidence().to_dict()
        # A prediction changed no capability SOURCE state (the evidence
        # carries read timestamps, so compare the substantive parts).
        def _sources(evidence):
            return {name: {"status": source.get("status"),
                           "direct_evidence": source.get("direct_evidence")}
                    for name, source in (evidence.get("sources") or {}).items()}
        self.assertEqual(_sources(before), _sources(after))
        # And no evidence was CREATED by predicting.
        self.assertEqual(before.get("evidence_version"),
                         after.get("evidence_version")) \
            if "evidence_version" in before else None

    def test_the_gain_carries_no_effect_verified_flag(self):
        prediction = planned_prediction(self.h,
            TASK, {"action_type": "execute_strategy", "strategy_id": "S01"},
            "ep1", benefit_convention={'kind': 'solution_quality', 'metric': 'normalized_objective_gap', 'unit': '1-gap'})
        view = prediction.capability_gain.to_dict()
        # The online block has no "verified" state at all — verification is
        # the OFFLINE path's job.
        self.assertNotIn("effect_verified", view)
        self.assertFalse(
            prediction.capability_gain.verification_conditions[0]
            .effect_verified)


# ---------------------------------------------------------------------------
# 6. the framework fixes identity; a payload may not rewrite it
# ---------------------------------------------------------------------------


class TestIdentityStaysFixed(unittest.TestCase):

    def test_a_payload_cannot_rewrite_the_candidate(self):
        from or_harness.world_model.contracts import CandidateRef
        candidate = CandidateRef(action_type="execute_strategy",
                                 strategy_id="S01", task_id="t1")
        payload = dict(BASE_PAYLOAD, candidate={"strategy_id": "S99"},
                       task_id="hacked")
        prediction = parse_strategy_outcome_payload(payload, candidate)
        # The parsed prediction keeps the FRAMEWORK's candidate.
        self.assertEqual(prediction.candidate.strategy_id, "S01")


# ---------------------------------------------------------------------------
# 7. the contract validator
# ---------------------------------------------------------------------------


class TestCapabilityGainValidation(unittest.TestCase):

    def test_a_gain_value_without_a_baseline_is_refused(self):
        from or_harness.world_model.contracts import (
            CandidateRef,
            ExpectedChange,
            StrategyOutcomePrediction,
        )
        prediction = StrategyOutcomePrediction(
            candidate=CandidateRef(action_type="execute_strategy",
                                   strategy_id="S01", task_id="t1"),
            status="valid",
            capability_gain=CapabilityGain(
                claim="x",
                expected_changes=[ExpectedChange(
                    metric="solver_runtime_s", direction="decrease",
                    value=10.0)]))
        problems = validate_strategy_outcome(prediction)
        self.assertTrue(any("requires a baseline" in p for p in problems))

    def test_a_gain_with_no_metric_is_refused(self):
        from or_harness.world_model.contracts import (
            CandidateRef,
            ExpectedChange,
            StrategyOutcomePrediction,
        )
        prediction = StrategyOutcomePrediction(
            candidate=CandidateRef(action_type="execute_strategy",
                                   strategy_id="S01", task_id="t1"),
            status="valid",
            capability_gain=CapabilityGain(
                claim="x", expected_changes=[ExpectedChange(metric="")]))
        problems = validate_strategy_outcome(prediction)
        self.assertTrue(any("metric is required" in p for p in problems))


if __name__ == "__main__":
    unittest.main()
