"""Synthetic fixtures for C aggregation; these are NOT experiment results.

All numbers are hand-constructed to check arithmetic and configuration.
Planning tests use an in-process scripted provider and fresh temporary homes;
no experiment bank, execution history, network or real model is used.
"""
from __future__ import annotations

import copy
import contextlib
import io
import json
import math
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from or_harness.api import ORHarness
from or_harness.core import schema
from or_harness.core.schema import CostVector, PREDICTED_COST_DIMENSIONS
from or_harness.strategy.embedding_index import LocalHashEmbeddingBackend
from or_harness.world_model.contracts import (
    COST_PREDICTION_CONTRACT_VERSION,
    FAITHFUL_G_METRIC,
    FAITHFUL_G_OBSERVATION_VERSION,
    RETRIES_FUTURE_VERSION,
    R_SCORING_VERSION,
    BaselineStatement,
    BenefitEstimate,
    CandidateRef,
    CapabilityGain,
    ExpectedCost,
    PredictionTrace,
    RiskEvent,
    RiskStatement,
    StrategyOutcomePrediction,
)
from or_harness.world_model.history_retrieval import decision_cost_scales
from or_harness.world_model.planner import (
    PlanLimits,
    score_strategy_outcome_predictions,
)
from or_harness.world_model.provider import WorldModelProvider


# Synthetic fixtures only. Independent peaks are fixed before any prediction.
PEAKS = {"remaining_latency_s": 99.0, "output_tokens": 9999.0,
         "tool_calls": 15.0, "retries": 3.0}
OLD_WEIGHTS = {"remaining_latency_s": 0.1, "output_tokens": 1.0,
               "tool_calls": 0.2, "retries": 2.0}
NEW_WEIGHTS = {"remaining_latency_s": 1.0, "output_tokens": 1.0,
               "tool_calls": 0.5, "retries": 0.5}
LEGACY_WEIGHTS = {"llm_tokens": 1.0, "output_tokens": 1.0,
                  "tool_calls": 0.2, "solver_runtime_s": 0.5,
                  "retries": 2.0, "latency_s": 0.1,
                  "remaining_latency_s": 0.1}
# A normalizes to (0.5, 0.5, 0.5, 0); B to (0, 0, 0, 1).
SYNTHETIC_A = {"remaining_latency_s": 9.0, "output_tokens": 99.0,
               "tool_calls": 3.0, "retries": 0.0}
SYNTHETIC_B = {"remaining_latency_s": 0.0, "output_tokens": 0.0,
               "tool_calls": 0.0, "retries": 3.0}


def synthetic_prediction(name="a", costs=None, *, hplus=1):
    costs = dict(SYNTHETIC_A if costs is None else costs)
    return StrategyOutcomePrediction(
        prediction_id="synthetic-" + name,
        candidate=CandidateRef(action_type="execute_strategy",
                               strategy_id=name, task_id="synthetic-task"),
        status="valid",
        benefit=BenefitEstimate(
            kind="effective_completion", metric=FAITHFUL_G_METRIC,
            unit="probability", value=0.8,
            observation_rule_version=FAITHFUL_G_OBSERVATION_VERSION,
            baseline=BaselineStatement(kind="declared", value=0.0)),
        cost=ExpectedCost(
            expected=CostVector.from_dict({**costs, "measured": list(costs)}),
            measured=list(costs), scope="remaining_to_task_end",
            prediction_contract=COST_PREDICTION_CONTRACT_VERSION),
        risk=RiskStatement(events=[
            RiskEvent(event=event, probability=probability)
            for event, probability in (("implementation_failure", 0.2),
                                       ("timeout", 0.1),
                                       ("environment_failure", 0.05))]),
        capability_gain=CapabilityGain(decision=hplus),
        trace=PredictionTrace(prediction_kind="strategy_outcome", model_info={
            "risk_convention": {"scoring_rule_version": R_SCORING_VERSION},
            "cost_observation_rule_versions": {"retries": RETRIES_FUTURE_VERSION},
        }))


def synthetic_scores(predictions, **overrides):
    return score_strategy_outcome_predictions(
        predictions, PlanLimits(**{"alpha": 1.0, "beta": 1.0, "gamma": 1.0,
                                   "cost_scales": dict(PEAKS), **overrides}))


class TestSyntheticCAggregation(unittest.TestCase):
    def test_explicit_old_and_proposed_profiles_have_known_c_and_u(self):
        predictions = [synthetic_prediction("a"),
                       synthetic_prediction("b", SYNTHETIC_B)]
        # Independent arithmetic: old denominator=3.3, proposed denominator=3.
        for weights, expected_c in ((OLD_WEIGHTS, (13 / 66, 20 / 33)),
                                    (NEW_WEIGHTS, (5 / 12, 1 / 6))):
            with self.subTest(weights=weights):
                scores = synthetic_scores(predictions, cost_weights=weights)
                for score, c in zip(scores, expected_c):
                    self.assertEqual(score.cost_normalized, round(c, 6))
                    self.assertEqual(score.utility, round(0.8 - c - 0.2, 6))

    def test_default_profile_has_exactly_the_four_active_dimensions(self):
        self.assertEqual(schema.DEFAULT_PREDICTED_COST_WEIGHTS, NEW_WEIGHTS)
        self.assertEqual(set(schema.DEFAULT_PREDICTED_COST_WEIGHTS),
                         set(PREDICTED_COST_DIMENSIONS))
        score = synthetic_scores([synthetic_prediction()])[0]
        self.assertEqual(score.cost_normalized, round(5 / 12, 6))
        for dim, weight in NEW_WEIGHTS.items():
            self.assertAlmostEqual(score.scoring_basis["cost_weights"][dim], weight / 3)

    def test_each_dimension_is_monotone_and_clipped_under_both_profiles(self):
        for weights in (OLD_WEIGHTS, NEW_WEIGHTS):
            for dim, peak in PEAKS.items():
                values = []
                for value in (0.0, peak / 4, peak, peak * 2):
                    costs = {**SYNTHETIC_A, dim: value}
                    score = synthetic_scores([synthetic_prediction(costs=costs)],
                                             cost_weights=weights)[0]
                    values.append(score.cost_normalized)
                    self.assertGreaterEqual(score.cost_normalized, 0)
                    self.assertLessEqual(score.cost_normalized, 1)
                with self.subTest(weights=weights, dimension=dim):
                    self.assertEqual(values, sorted(values))
                    self.assertEqual(values[-1], values[-2])

    def test_missing_dimension_is_removed_for_every_candidate_and_reweighted(self):
        for missing in PREDICTED_COST_DIMENSIONS:
            with self.subTest(missing=missing):
                full = synthetic_prediction("full")
                partial = synthetic_prediction("partial", {
                    d: value for d, value in SYNTHETIC_A.items() if d != missing})
                scores = synthetic_scores([full, partial])
                self.assertEqual(scores[0].cost_normalized, scores[1].cost_normalized)
                self.assertEqual(scores[0].utility, scores[1].utility)
                for score in scores:
                    basis = score.scoring_basis
                    self.assertNotIn(missing, basis["cost_dimensions"])
                    self.assertEqual(basis["omitted_cost_dimensions"][missing]
                                     ["prediction_ids"], [partial.prediction_id])
                    total = sum(w for d, w in NEW_WEIGHTS.items() if d != missing)
                    for dim, weight in basis["cost_weights"].items():
                        self.assertAlmostEqual(weight, NEW_WEIGHTS[dim] / total)
                    self.assertAlmostEqual(sum(basis["cost_weights"].values()), 1)

    def test_explicit_weight_maps_replace_defaults_including_empty_and_zero(self):
        for weights, expected_dims in (({"retries": 7.0}, ["retries"]),
                                      ({"output_tokens": 0.0,
                                        "remaining_latency_s": 5.0},
                                       ["remaining_latency_s"]),
                                      ({}, [])):
            with self.subTest(weights=weights):
                original = copy.deepcopy(weights)
                score = synthetic_scores([synthetic_prediction()], cost_weights=weights)[0]
                self.assertEqual(score.scoring_basis["cost_dimensions"], expected_dims)
                self.assertEqual(weights, original)
                if not expected_dims:
                    self.assertIsNone(score.cost_normalized)
                    self.assertEqual(score.utility, 0.6)
                else:
                    self.assertEqual(score.scoring_basis["cost_weights"],
                                     {expected_dims[0]: 1.0})

    def test_absent_cost_omits_c_for_both_candidates(self):
        for absent_block in (False, True):
            with self.subTest(absent_block=absent_block):
                unknown = synthetic_prediction("unknown", costs={})
                if absent_block:
                    unknown.cost = None
                scores = synthetic_scores([synthetic_prediction(), unknown])
                for score in scores:
                    self.assertIsNone(score.cost_normalized)
                    self.assertEqual(score.utility, 0.6)
                    self.assertEqual(score.scoring_basis["cost_normalized_values"], {})

    def test_legacy_only_weight_config_warns_without_enabling_c_or_blocking_u(self):
        weights = {"llm_tokens": 1, "solver_runtime_s": 0.5, "latency_s": 0.1}
        score = synthetic_scores([synthetic_prediction()], cost_weights=weights)[0]
        diagnostic = score.scoring_basis["cost_weight_diagnostic"]
        self.assertEqual(diagnostic["ignored_legacy_dimensions"], sorted(weights))
        self.assertEqual(diagnostic["positive_active_dimensions"], [])
        self.assertEqual(diagnostic["active_dimensions_without_positive_weight"],
                         sorted(PREDICTED_COST_DIMENSIONS))
        self.assertIn(diagnostic["message"], score.notes)
        self.assertEqual(score.scoring_basis["cost_dimensions"], [])
        self.assertEqual(score.scoring_basis["cost_weights"], {})
        self.assertIsNone(score.cost_normalized)
        self.assertEqual(score.utility, 0.6)

    def test_mixed_legacy_weights_warn_and_still_price_only_retries(self):
        predictions = [synthetic_prediction("a"),
                       synthetic_prediction("b", SYNTHETIC_B)]
        weights = {"llm_tokens": 1, "retries": 2}
        before = copy.deepcopy(weights)
        mixed = synthetic_scores(predictions, cost_weights=weights)
        active_only = synthetic_scores(predictions, cost_weights={"retries": 2})
        self.assertEqual([s.cost_normalized for s in mixed], [0.0, 1.0])
        self.assertEqual([s.utility for s in mixed], [s.utility for s in active_only])
        self.assertEqual(weights, before)
        for score in mixed:
            diagnostic = score.scoring_basis["cost_weight_diagnostic"]
            self.assertEqual(diagnostic["ignored_legacy_dimensions"], ["llm_tokens"])
            self.assertEqual(diagnostic["positive_active_dimensions"], ["retries"])
            self.assertEqual(len(diagnostic["active_dimensions_without_positive_weight"]), 3)
            self.assertEqual(score.scoring_basis["cost_weights"], {"retries": 1.0})

    def test_intentional_active_partial_or_empty_maps_have_no_migration_warning(self):
        for weights in (None, {}, {"retries": 2}, OLD_WEIGHTS, NEW_WEIGHTS):
            with self.subTest(weights=weights):
                score = synthetic_scores([synthetic_prediction()], cost_weights=weights)[0]
                self.assertNotIn("cost_weight_diagnostic", score.scoring_basis)

    def test_migration_diagnostic_does_not_claim_configured_dimensions_are_charged(self):
        for retries in (0, 2):
            with self.subTest(retries=retries):
                score = synthetic_scores([synthetic_prediction()], cost_scales={},
                                         cost_weights={"llm_tokens": 1, "retries": retries})[0]
                diagnostic = score.scoring_basis["cost_weight_diagnostic"]
                self.assertEqual(diagnostic["positive_active_dimensions"],
                                 ["retries"] if retries else [])
                self.assertEqual(score.scoring_basis["cost_dimensions"], [])
                self.assertEqual(score.scoring_basis["cost_weights"], {})
                self.assertIsNone(score.cost_normalized)
                self.assertEqual(score.utility, 0.6)

    def test_nonpositive_or_invalid_weights_are_omitted_without_penalty(self):
        for weight in (0, -1, True, float("nan"), float("inf")):
            with self.subTest(weight=weight):
                score = synthetic_scores([synthetic_prediction()],
                                         cost_weights={"retries": weight})[0]
                self.assertIsNone(score.cost_normalized)
                self.assertEqual(score.utility, 0.6)
                self.assertIn("weight", score.scoring_basis[
                    "omitted_cost_dimensions"]["retries"]["reason"])

    def test_beta_zero_makes_u_and_selection_independent_of_internal_weights(self):
        predictions = [synthetic_prediction("a"),
                       synthetic_prediction("b", SYNTHETIC_B)]
        old = synthetic_scores(predictions, cost_weights=OLD_WEIGHTS, beta=0)
        new = synthetic_scores(predictions, cost_weights=NEW_WEIGHTS, beta=0)
        self.assertNotEqual(old[0].cost_normalized, new[0].cost_normalized)
        self.assertEqual([s.utility for s in old], [0.6, 0.6])
        self.assertEqual([s.utility for s in new], [0.6, 0.6])
        self.assertEqual(max(old, key=lambda s: s.utility).prediction_id,
                         max(new, key=lambda s: s.utility).prediction_id)

    def test_missing_scales_do_not_use_current_candidate_maxima(self):
        predictions = [synthetic_prediction("a"),
                       synthetic_prediction("b", SYNTHETIC_B)]
        scores = synthetic_scores(predictions, cost_scales={})
        for score in scores:
            self.assertIsNone(score.cost_normalized)
            self.assertEqual(score.utility, 0.6)
            self.assertEqual(score.scoring_basis["cost_weights"], {})
            self.assertEqual(score.scoring_basis["cost_scales"], {})
            self.assertEqual(score.scoring_basis["cost_scale_sources"], {})
            self.assertEqual(score.scoring_basis["cost_normalized_values"], {})
            self.assertEqual(score.scoring_basis["terms"], ["G", "R"])
            for dim in PEAKS:
                self.assertIn("scale", score.scoring_basis[
                    "omitted_cost_dimensions"][dim]["reason"])

    def test_invalid_or_wrong_scope_scale_is_excluded_and_weights_renormalized(self):
        invalid = (0, -1, None, True, float("nan"), float("inf"),
                   {"peak": 99, "scope": "attempt"})
        for scale in invalid:
            with self.subTest(scale=scale):
                score = synthetic_scores([synthetic_prediction()], cost_scales={
                    **PEAKS, "remaining_latency_s": scale})[0]
                basis = score.scoring_basis
                self.assertNotIn("remaining_latency_s", basis["cost_dimensions"])
                self.assertIn("scale", basis["omitted_cost_dimensions"]
                              ["remaining_latency_s"]["reason"])
                self.assertEqual(basis["cost_weights"], {
                    "output_tokens": 0.5, "tool_calls": 0.25, "retries": 0.25})

    def test_dominant_costs_are_never_penalized_in_reverse(self):
        for weights in (OLD_WEIGHTS, NEW_WEIGHTS):
            for dim in PEAKS:
                lower = {**SYNTHETIC_A, dim: 0.0}
                higher = {**SYNTHETIC_A, dim: PEAKS[dim]}
                low, high = synthetic_scores([
                    synthetic_prediction("lower", lower),
                    synthetic_prediction("higher", higher)], cost_weights=weights)
                with self.subTest(weights=weights, dimension=dim):
                    self.assertLess(low.cost_normalized, high.cost_normalized)
                    self.assertGreater(low.utility, high.utility)

    def test_legacy_dimensions_remain_readable_and_do_not_enter_u(self):
        prediction = synthetic_prediction(costs={
            **SYNTHETIC_A, "llm_tokens": 123456, "solver_runtime_s": 17,
            "latency_s": 23})
        prediction.cost.prediction_contract = None  # archived prediction shape
        saved = prediction.to_dict()
        restored = StrategyOutcomePrediction.from_dict(copy.deepcopy(saved))
        score = synthetic_scores([restored])[0]
        self.assertEqual(restored.to_dict(), saved)
        self.assertEqual(score.cost_normalized, round(5 / 12, 6))
        for dim in ("llm_tokens", "solver_runtime_s", "latency_s"):
            self.assertNotIn(dim, score.scoring_basis["cost_dimensions"])
            self.assertIn("historical", score.scoring_basis[
                "omitted_cost_dimensions"][dim]["reason"])

    def test_legacy_scalarization_defaults_and_record_format_are_unchanged(self):
        self.assertEqual(schema.DEFAULT_COST_WEIGHTS, LEGACY_WEIGHTS)
        vector = CostVector.from_dict({d: 1.0 for d in LEGACY_WEIGHTS})
        saved = vector.to_dict()
        self.assertAlmostEqual(vector.scalarize(), 4.9)
        self.assertEqual(CostVector.from_dict(saved).to_dict(), saved)

    def test_g_r_hplus_external_coefficients_and_inputs_are_unchanged(self):
        for hplus in (0, 1):
            prediction = synthetic_prediction(hplus=hplus)
            saved = copy.deepcopy(prediction.to_dict())
            old = synthetic_scores([prediction], cost_weights=OLD_WEIGHTS,
                                   alpha=0.7, beta=0.4, gamma=0.3)[0]
            new = synthetic_scores([prediction], cost_weights=NEW_WEIGHTS,
                                   alpha=0.7, beta=0.4, gamma=0.3)[0]
            self.assertEqual((old.benefit_value, old.risk_effective), (0.8, 0.2))
            self.assertEqual((new.benefit_value, new.risk_effective), (0.8, 0.2))
            for key in ("risk_events", "risk_aggregation", "risk_scoring_rule",
                        "risk_event_probabilities", "utility_weights"):
                self.assertEqual(old.scoring_basis[key], new.scoring_basis[key])
            self.assertEqual(new.scoring_basis["utility_weights"],
                             {"alpha": 0.7, "beta": 0.4, "gamma": 0.3})
            self.assertEqual(prediction.to_dict(), saved)

    def test_audit_values_reproduce_c_and_u_before_final_rounding(self):
        scales = {d: {"peak": p, "scope": "remaining_to_task_end",
                      "source": "synthetic_explicit_scale", "reference": "synthetic:peaks"}
                  for d, p in PEAKS.items()}
        prediction = synthetic_prediction()
        score = synthetic_scores([prediction], cost_scales=scales,
                                 alpha=0.7, beta=0.4, gamma=0.3)[0].to_dict()
        basis = score["scoring_basis"]
        for dim in basis["cost_dimensions"]:
            expected = min(1, max(0, math.log1p(score["cost_reported"]["values"][dim])
                                   / math.log1p(basis["cost_scales"][dim])))
            self.assertEqual(basis["cost_normalized_values"][dim], expected)
            self.assertEqual(basis["cost_scale_sources"][dim]["reference"], "synthetic:peaks")
        c = sum(basis["cost_weights"][d] * basis["cost_normalized_values"][d]
                for d in basis["cost_dimensions"])
        self.assertEqual(score["cost_normalized"], round(c, 6))
        self.assertEqual(score["utility"], round(0.7 * score["benefit_value"]
                         - 0.4 * c - 0.3 * score["risk_effective"], 6))

    def test_scale_source_uses_only_frozen_synthetic_p90_or_explicit_override(self):
        context = SimpleNamespace(created_at=123.0, history_retrieval={
            "global": {"c": {"dimensions": {
                d: {"p90": peak, "n": 7, "source_refs": ["synthetic:closed-window"]}
                for d, peak in PEAKS.items()}}}})
        candidates = [synthetic_prediction().candidate]
        scales = decision_cost_scales(context, candidates)
        self.assertEqual({d: s["peak"] for d, s in scales.items()}, PEAKS)
        self.assertTrue(all(s["source"] == "pre_decision_measured_history"
                            for s in scales.values()))
        scored = synthetic_scores([synthetic_prediction()], cost_scales=scales)[0]
        self.assertEqual(set(scored.scoring_basis["cost_dimensions"]), set(PEAKS))
        self.assertTrue(all(s["source"] == "pre_decision_measured_history" for s
                            in scored.scoring_basis["cost_scale_sources"].values()))
        self.assertEqual(scales, decision_cost_scales(context, []))
        override = decision_cost_scales(context, candidates, {"tool_calls": 31})
        self.assertEqual(override["tool_calls"]["peak"], 31)
        self.assertEqual(override["tool_calls"]["source"], "explicit_plan_limits")


class SyntheticProvider(WorldModelProvider):
    """An in-process fixture, with no network or execution calls."""
    name = "synthetic-c-weights"

    def __init__(self):
        self.requests = []

    def predict(self, request, timeout_s=None):
        self.requests.append(copy.deepcopy(request))
        strategy = request["candidate"]["strategy_id"]
        p = synthetic_prediction(strategy, SYNTHETIC_A if strategy == "a" else SYNTHETIC_B)
        payload = {"benefit": p.benefit.to_dict(),
                   "cost": p.cost.expected.to_dict(), "risk": p.risk.to_dict(),
                   "capability_gain": {"decision": 0, "basis": ["synthetic fixture only"]}}
        payload["cost"] = {d: payload["cost"][d] for d in PREDICTED_COST_DIMENSIONS}
        return {"payload": payload, "usage": {}, "error": None, "latency_s": 0.0}


class TestSyntheticCWeightPlanning(unittest.TestCase):
    def harness(self, **kwargs):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        provider = SyntheticProvider()
        harness = ORHarness(home=temporary.name, world_model=provider,
                            embedding=LocalHashEmbeddingBackend(), **kwargs)
        self.addCleanup(harness.close)
        return harness, provider

    @staticmethod
    def plan(harness, names=("a", "b"), **limits):
        return harness.plan_next(
            {"task_id": "synthetic-task", "family": "synthetic",
             "text": "Synthetic arithmetic fixture; no solve is executed."},
            "synthetic-episode", candidates=[
                {"strategy_id": name, "solver": "highs",
                 "method": {"name": "synthetic " + name, "steps": ["Synthetic fixture only"]}}
                for name in names], limits={"cost_scales": dict(PEAKS), **limits})

    def test_default_single_and_multi_candidate_scores_are_identical(self):
        harness, provider = self.harness()
        a = self.plan(harness, ("a",))
        b = self.plan(harness, ("b",))
        both = self.plan(harness)
        self.assertEqual(a["selected_score"]["cost_normalized"], round(5 / 12, 6))
        for single, row in zip((a, b), both["candidates"]):
            for key in ("cost_normalized", "utility", "benefit_value", "risk_effective"):
                self.assertEqual(single["selected_score"][key], row["score"][key])
        self.assertEqual(both["selected_prediction_id"], both["candidates"][1]["prediction_id"])
        self.assertEqual(both["limits"]["cost_weights"], NEW_WEIGHTS)
        self.assertEqual(harness.selector.cost_weights, LEGACY_WEIGHTS)
        self.assertEqual(len(provider.requests), 4)
        self.assertEqual(provider.requests[-1]["prediction_context"]["context_id"],
                         provider.requests[-2]["prediction_context"]["context_id"])
        recorded = harness.actions.get(both["decision_action_id"]).outcome
        self.assertEqual(recorded["selected_score"], both["selected_score"])

    def test_plan_override_wins_over_harness_override_and_defaults(self):
        harness, _ = self.harness(cost_weights=OLD_WEIGHTS)
        old = self.plan(harness)
        new = self.plan(harness, cost_weights=NEW_WEIGHTS)
        empty = self.plan(harness, cost_weights={})
        self.assertEqual(old["limits"]["cost_weights"], OLD_WEIGHTS)
        self.assertEqual(old["selected_prediction_id"], old["candidates"][0]["prediction_id"])
        self.assertEqual(new["limits"]["cost_weights"], NEW_WEIGHTS)
        self.assertEqual(new["selected_prediction_id"], new["candidates"][1]["prediction_id"])
        self.assertEqual(empty["limits"]["cost_weights"], {})
        self.assertIsNone(empty["selected_score"]["cost_normalized"])
        self.assertEqual(empty["selected_score"]["utility"], 0.6)

    def test_explicit_empty_harness_weights_do_not_fall_back_to_selector_defaults(self):
        harness, _ = self.harness(cost_weights={})
        plan = self.plan(harness)
        self.assertEqual(plan["limits"]["cost_weights"], {})
        self.assertIsNone(plan["selected_score"]["cost_normalized"])
        self.assertEqual(plan["selected_score"]["utility"], 0.6)

    def test_plan_override_controls_diagnostics_for_legacy_harness_configuration(self):
        weights = {"llm_tokens": 1, "retries": 2}
        harness, _ = self.harness(cost_weights=weights)
        inherited = self.plan(harness)
        self.assertIn("cost_weight_diagnostic", inherited["selected_score"]["scoring_basis"])
        overridden = self.plan(harness, cost_weights=NEW_WEIGHTS)
        self.assertNotIn("cost_weight_diagnostic", overridden["selected_score"]["scoring_basis"])
        self.assertEqual(inherited["limits"]["cost_weights"], weights)
        self.assertEqual(overridden["limits"]["cost_weights"], NEW_WEIGHTS)

    def test_empty_history_without_explicit_scales_reports_inactive_c(self):
        harness, _ = self.harness()
        plan = self.plan(harness, cost_scales={})
        self.assertEqual(plan["limits"]["cost_weights"], NEW_WEIGHTS)
        for candidate in plan["candidates"]:
            score = candidate["score"]
            basis = score["scoring_basis"]
            self.assertIsNone(score["cost_normalized"])
            self.assertEqual(score["utility"], 0.6)
            self.assertEqual(basis["terms"], ["G", "R"])
            for key in ("cost_dimensions", "cost_scale_sources", "cost_weights"):
                self.assertFalse(basis[key])
            for dim in PEAKS:
                self.assertIn("scale", basis["omitted_cost_dimensions"][dim]["reason"])

    def test_harness_external_coefficients_and_beta_zero_are_preserved(self):
        harness, _ = self.harness(alpha=0.7, beta=0.0, gamma=0.3)
        old = self.plan(harness, cost_weights=OLD_WEIGHTS)
        new = self.plan(harness, cost_weights=NEW_WEIGHTS)
        for plan in (old, new):
            self.assertEqual(plan["selected_score"]["utility"], 0.5)
            self.assertEqual(plan["selected_score"]["scoring_basis"]["utility_weights"],
                             {"alpha": 0.7, "beta": 0.0, "gamma": 0.3})
            self.assertEqual(plan["selected_prediction_id"], plan["candidates"][0]["prediction_id"])

    def test_explicit_planlimits_object_and_partial_harness_map_are_preserved(self):
        weights = {"retries": 7.0}
        harness, _ = self.harness(cost_weights=weights)
        weights["output_tokens"] = 50.0  # caller mutation cannot change the config
        inherited = self.plan(harness)
        self.assertEqual(inherited["limits"]["cost_weights"], {"retries": 7.0})
        self.assertEqual(inherited["selected_score"]["scoring_basis"]["cost_weights"],
                         {"retries": 1.0})
        limits = PlanLimits(cost_weights={}, cost_scales=dict(PEAKS))
        plan = harness.plan_next(
            {"task_id": "synthetic-task", "family": "synthetic",
             "text": "Synthetic arithmetic fixture only."},
            "synthetic-planlimits", candidates=[{"strategy_id": "a"}], limits=limits)
        self.assertEqual(plan["limits"]["cost_weights"], {})
        self.assertIsNone(plan["selected_score"]["cost_normalized"])

    def test_cli_can_switch_profiles_and_preserves_the_score_audit(self):
        from or_harness import cli
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        provider = SyntheticProvider()

        def build_harness(**kwargs):
            return ORHarness(**{**kwargs, "world_model": provider,
                                "embedding": LocalHashEmbeddingBackend()})

        task = {"task_id": "synthetic-cli", "family": "synthetic",
                "text": "Synthetic CLI fixture; no solve is executed."}
        for weights, selected_index in ((OLD_WEIGHTS, 0), (NEW_WEIGHTS, 1)):
            with self.subTest(weights=weights):
                output = io.StringIO()
                args = ["--home", temporary.name, "--cost-weights",
                        ",".join(f"{d}={w}" for d, w in weights.items()),
                        "plan-next", "--task", json.dumps(task),
                        "--episode", "synthetic-cli", "--candidates",
                        json.dumps([{"strategy_id": "a"}, {"strategy_id": "b"}]),
                        "--cost-scales", json.dumps(PEAKS)]
                with patch.object(cli, "ORHarness", side_effect=build_harness), \
                        contextlib.redirect_stdout(output):
                    status = cli.main(args)
                self.assertEqual(status, 0, output.getvalue())
                result = json.loads(output.getvalue())["result"]
                plan = result["plan"]
                self.assertEqual(plan["limits"]["cost_weights"], weights)
                self.assertEqual(result["selected_prediction_id"],
                                 plan["candidates"][selected_index]["prediction_id"])
                score = result["selected_score"]
                basis = score["scoring_basis"]
                self.assertEqual(set(basis["cost_normalized_values"]), set(PEAKS))
                c = sum(basis["cost_weights"][d] * basis["cost_normalized_values"][d]
                        for d in basis["cost_dimensions"])
                self.assertEqual(score["cost_normalized"], round(c, 6))
                self.assertEqual(score["utility"], round(0.8 - c - 0.2, 6))

    def test_cli_legacy_warning_is_nonblocking_visible_once_and_preserves_config(self):
        from or_harness import cli
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        provider = SyntheticProvider()

        def build_harness(**kwargs):
            return ORHarness(**{**kwargs, "world_model": provider,
                                "embedding": LocalHashEmbeddingBackend()})

        for weights in ({"llm_tokens": 1}, {"llm_tokens": 1, "retries": 2}):
            for scales in ({}, PEAKS):
                with self.subTest(weights=weights, scales=scales):
                    output = io.StringIO()
                    args = ["--home", temporary.name, "--cost-weights",
                            ",".join(f"{d}={w}" for d, w in weights.items()),
                            "plan-next", "--task", json.dumps({
                                "task_id": "synthetic-migration", "family": "synthetic",
                                "text": "Synthetic legacy-weight diagnostic fixture."}),
                            "--candidates", json.dumps([
                                {"strategy_id": "a"}, {"strategy_id": "b"}]),
                            "--cost-scales", json.dumps(scales)]
                    with patch.object(cli, "ORHarness", side_effect=build_harness), \
                            contextlib.redirect_stdout(output):
                        status = cli.main(args)
                    self.assertEqual(status, 0, output.getvalue())
                    response = json.loads(output.getvalue())
                    result = response["result"]
                    score = result["selected_score"]
                    diagnostic = score["scoring_basis"]["cost_weight_diagnostic"]
                    self.assertEqual(response["summary"].count(diagnostic["message"]), 1)
                    self.assertEqual(result["plan"]["limits"]["cost_weights"], weights)
                    self.assertIsNotNone(result["selected_prediction_id"])
                    if not scales or "retries" not in weights:
                        self.assertIsNone(score["cost_normalized"])
                        self.assertEqual(score["utility"], 0.6)
                    else:
                        self.assertEqual(score["scoring_basis"]["cost_weights"],
                                         {"retries": 1.0})


if __name__ == "__main__":
    unittest.main()
