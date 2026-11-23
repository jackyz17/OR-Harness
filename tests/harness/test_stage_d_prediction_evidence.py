"""Stage D: graded, candidate-relevant G/C/R prediction evidence.

Deterministic tests of the provider-only evidence selection. They use fixed
history summaries/records and the existing hermetic harness fixtures; no real
model call is made.
"""
import copy
import unittest
from types import SimpleNamespace

from c_fixtures import CCase, REUSE
from g_fixtures import COUPLED, SPLIT, GCase
from test_r_technical import RCase

from or_harness.core.schema import ExecutionRecord
from or_harness.world_model.contracts import (
    R_CONVENTION, R_OBSERVATION_VERSION,
)
from or_harness.world_model.cost_signal import (
    decision_cost_scales, select_cost_evidence,
)
from or_harness.world_model.g_signal import select_candidate_method_evidence
from or_harness.world_model.r_signal import (
    global_event_base_rates, select_r_evidence, technical_r_statistics,
)


# ---------------------------------------------------------------------------
# synthetic builders
# ---------------------------------------------------------------------------

def _g_pair(evaluation_id, *, task_id="t", family="assignment",
            cell="rc[0.75,1.00]|tc[unknown]|rx[unknown]", observed=0,
            attribution=None, metric="task_check_passed_given_faithful_execution",
            rule="wm-g-faithful/1", steps=None):
    benefit = {
        "metric": metric,
        "observation_rule_version": rule,
        "predicted": 0.7,
        "signed_error": 0.3 if observed == 1 else -0.7,
        "evidence_refs": [f"artifacts.{evaluation_id}"],
    }
    if observed is not None:
        benefit["observed"] = observed
    if attribution is not None:
        benefit["attribution"] = attribution
    return {
        "evaluation_id": evaluation_id,
        "task_id": task_id,
        "scope": "attempt",
        "conditions": {"task_id": task_id, "family": family,
                       **({"cell": cell} if cell else {})},
        "method_planned": {"name": "joint selection",
                           "steps": steps or ["select the same factory"]},
        "method_actual": {"name": "joint selection",
                          "steps": "same as method_planned.steps"},
        "benefit": benefit,
    }


def _candidate(method_steps=None):
    return SimpleNamespace(
        method={"name": "joint selection",
                "steps": method_steps or ["select the same factory"]},
        solver="highs", config={}, scope="attempt", task_id="t")


def _cost_group(*, phase="initial_modelling", family="assignment",
                solver="highs", structure=None, dimensions=None,
                source_refs=None):
    return {
        "conditions": {
            "scope": "remaining_to_task_end",
            "window_coverage": "complete",
            "phase": phase,
            "family": family,
            "structure": structure if structure is not None else {"problem_class": "milp"},
            "solver_conditions": {"constraints": {"executor": None,
                                                  "declared_budget": None}},
            "actual_solving_conditions": [{"solver": solver, "config": {}}],
            "retry_observation_rule_version": "wm-c-retries-legacy/1",
        },
        "observation_rule_version": "wm-c-window/1",
        "n_distinct_episodes": 2,
        "support": "sparse",
        "support_note": "fixture",
        "dimensions": dimensions or {
            "tool_calls": {"p50": 8, "p75": 9, "min": 8, "max": 9,
                           "n": 2, "n_distinct_episodes": 2}},
        "source_refs": source_refs or ["ex-c1", "ex-c2"],
    }


def _cost_context(groups, *, phase="unknown", family="assignment"):
    """A PredictionContext-shaped object for the pure selector."""
    solving = {"execution_context": {"phase": phase}} if phase != "unknown" else {}
    return SimpleNamespace(
        solving_context=solving,
        execution_constraints={},
        joint=SimpleNamespace(profile={"family": family,
                                       "problem_class": "milp"}),
        task_id="t",
        created_at=1.0,
        strategy_calibration={
            "conditional_cost": {"groups": copy.deepcopy(groups), "cases": []}},
    )


def _technical_record(execution_id, *, source="executed", label=1,
                      event_labels=None):
    rec = ExecutionRecord(
        execution_id=execution_id, task_id="t", strategy_id="S04",
        source=source,
        profile_snapshot=None,
        quality={"feasible": True, "objective": 1.0, "gap": 0.0,
                 "status": "optimal"},
        solver={"name": "highs"},
    )
    # ExecutionRecord requires a profile snapshot in practice; reuse a tiny
    # one through the public helper's shape.
    from or_harness.core.schema import ProblemProfile
    rec.profile_snapshot = ProblemProfile(
        problem_id="t", family="assignment",
        semantic_coupling=0.8, resource_coupling=0.9,
        temporal_coupling=0.1, route_complexity=0.8)
    labels = {"implementation_failure": label,
              "timeout": 0, "environment_failure": None}
    labels.update(event_labels or {})
    rec.execution_features = {
        "executed": True,
        "technical_outcome": {
            **R_CONVENTION,
            "observation_rule_version": R_OBSERVATION_VERSION,
            "source": "SafePythonExecutor",
            "label": label,
            "reason": "fixture technical label",
            "r_events": {name: {"label": value, "basis": "fixture"}
                         for name, value in labels.items()},
        },
    }
    return rec


def _r_context(*, solver="highs", family="assignment"):
    return SimpleNamespace(
        solving_context={},
        joint=SimpleNamespace(profile={"family": family,
                                       "problem_class": "milp"}),
    )


def _technical_summary(*, groups=None, events=None):
    return {
        "observation_rule_version": R_OBSERVATION_VERSION,
        "coverage": {"n_actual_attempts": 4},
        "groups": groups or [],
        "events": events or {},
    }


# ---------------------------------------------------------------------------
# G
# ---------------------------------------------------------------------------

class TestGCandidateMethodEvidence(unittest.TestCase):
    def test_same_task_case_ranks_first_and_points_back_to_pair(self):
        pairs = [
            _g_pair("ev-other", task_id="other", family="scheduling",
                    cell="rc[0.00,0.25]|tc[unknown]|rx[unknown]"),
            _g_pair("ev-same", task_id="t"),
        ]
        out = select_candidate_method_evidence(
            {"pairs": pairs}, candidate=_candidate(),
            profile={"family": "assignment", "problem_class": "milp"},
            convention={"metric": "task_check_passed_given_faithful_execution",
                        "observation_rule_version": "wm-g-faithful/1"},
            scope="attempt", task_id="t")
        self.assertEqual(out["cases"][0]["evaluation_id"], "ev-same")
        self.assertEqual(len({case["evaluation_id"] for case in out["cases"]}),
                         len(out["cases"]))
        case = out["cases"][0]
        self.assertEqual(case["comparison_role"], "strict_condition_match")
        self.assertEqual(case["condition_match"], "same_task")
        self.assertEqual(case["observed"], 0)
        self.assertIn("detail_ref", case)
        self.assertEqual(out["selection"]["n_pairs_available"], 2)
        self.assertEqual(out["selection"]["n_included"], 2)

    def test_recorded_cause_only_appears_when_attribution_exists(self):
        recorded = {"method_fidelity": "faithful",
                    "failure_source": "candidate_method",
                    "source": "fixture audit",
                    "assessment": "the candidate omitted the coupling"}
        pairs = [
            _g_pair("ev-recorded", attribution=recorded),
            _g_pair("ev-unrecorded"),
        ]
        out = select_candidate_method_evidence(
            {"pairs": pairs}, candidate=_candidate(),
            profile={"family": "assignment", "problem_class": "milp"},
            convention={"metric": "task_check_passed_given_faithful_execution"},
            scope="attempt", task_id="t")
        by_id = {case["evaluation_id"]: case for case in out["cases"]}
        self.assertEqual(by_id["ev-recorded"]["recorded_cause"]["failure_source"],
                         "candidate_method")
        self.assertNotIn("recorded_cause", by_id["ev-unrecorded"])

    def test_unobserved_execution_error_is_not_promoted_to_a_g_case(self):
        pairs = [_g_pair("ev-crash", observed=None)]
        out = select_candidate_method_evidence(
            {"pairs": pairs}, candidate=_candidate(),
            profile={"family": "assignment", "problem_class": "milp"},
            convention={"metric": "task_check_passed_given_faithful_execution"},
            scope="attempt", task_id="t")
        self.assertEqual(out["cases"], [])
        self.assertEqual(out["selection"]["n_condition_relevant"], 0)
        self.assertEqual(out["global_calibration_role"], "diagnostic_only")

    def test_unknown_structure_is_partial_not_a_match(self):
        pair = _g_pair("ev-no-cell", cell="")
        out = select_candidate_method_evidence(
            {"pairs": [pair]}, candidate=_candidate(),
            profile={"family": "assignment", "problem_class": "milp"},
            convention={"metric": "task_check_passed_given_faithful_execution"},
            scope="attempt", task_id="other")
        self.assertEqual(out["cases"][0]["comparison_role"],
                         "partial_condition_similarity")
        self.assertEqual(out["cases"][0]["condition_match"],
                         "same_family_only")

    def test_other_metric_is_never_merged_into_this_convention(self):
        pair = _g_pair("ev-other-metric", metric="normalized_objective_gap",
                       rule="wm-g-quality/1")
        out = select_candidate_method_evidence(
            {"pairs": [pair]}, candidate=_candidate(),
            profile={"family": "assignment", "problem_class": "milp"},
            convention={"metric": "task_check_passed_given_faithful_execution",
                        "observation_rule_version": "wm-g-faithful/1"},
            scope="attempt", task_id="t")
        self.assertEqual(out["cases"], [])

    def test_cases_are_unique_and_the_global_block_is_not_a_correction(self):
        pairs = [_g_pair("ev-1"), _g_pair("ev-2"), _g_pair("ev-3")]
        out = select_candidate_method_evidence(
            {"pairs": pairs}, candidate=_candidate(),
            profile={"family": "assignment", "problem_class": "milp"},
            convention={"metric": "task_check_passed_given_faithful_execution"},
            scope="attempt", task_id="t")
        ids = [case["evaluation_id"] for case in out["cases"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertLessEqual(len(ids), out["selection"]["n_condition_relevant"])
        self.assertEqual(out["global_calibration_role"], "diagnostic_only")
        for forbidden in ("adjustment", "correction", "calibrated_g"):
            self.assertNotIn(forbidden, out)
        self.assertNotIn("g_correction", out)


class TestGProviderView(GCase):
    def test_candidate_method_evidence_reaches_the_request(self):
        history = self.predict_g(SPLIT, "hist")
        record = self.run_g(history, split=True)
        self.check_g(record, fidelity="faithful", failure="candidate_method")
        self.close_g("hist")
        self.predict_g(COUPLED, "next")
        evidence = self.provider.requests[-1]["prediction_context"][
            "g_candidate_evidence"]
        self.assertTrue(evidence["cases"])
        case = evidence["cases"][0]
        self.assertEqual(case["condition_match"], "same_task")
        self.assertEqual(case["observed"], 0)
        self.assertEqual(case["recorded_cause"]["failure_source"],
                         "candidate_method")


# ---------------------------------------------------------------------------
# C
# ---------------------------------------------------------------------------

class TestCostReferenceGrades(unittest.TestCase):
    def test_phase_unknown_yields_explicit_broader_reference(self):
        summary = {"conditional_cost": {
            "groups": [_cost_group(phase="initial_modelling")], "cases": []}}
        evidence = select_cost_evidence(
            summary, _cost_context([]), _candidate())
        self.assertEqual(evidence["reference_ranges"], [])
        self.assertEqual(len(evidence["broader_reference_ranges"]), 1)
        group = evidence["broader_reference_ranges"][0]
        self.assertEqual(group["reference_grade"], "broader_reference")
        self.assertEqual(evidence["current_conditions"]["phase"], "unknown")
        self.assertTrue(any("current phase unknown" in reason
                            for reason in group["grade_reasons"]))
        self.assertEqual(evidence["evidence_status"], "broader_reference_only")
        self.assertEqual(evidence["reference_coverage"]["strict"]["n_groups_available"], 0)

    def test_known_phase_conflict_is_unavailable_not_broader(self):
        summary = {"conditional_cost": {
            "groups": [_cost_group(phase="existing_model")], "cases": []}}
        evidence = select_cost_evidence(
            summary, _cost_context([], phase="initial_modelling"), _candidate())
        self.assertEqual(evidence["reference_ranges"], [])
        self.assertEqual(evidence["broader_reference_ranges"], [])
        self.assertEqual(evidence["evidence_status"], "insufficient_evidence")

    def test_known_solver_conflict_is_unavailable_not_broader(self):
        summary = {"conditional_cost": {
            "groups": [_cost_group(phase="initial_modelling", solver="glpk")],
            "cases": []}}
        evidence = select_cost_evidence(
            summary, _cost_context([]), _candidate())
        self.assertEqual(evidence["reference_ranges"], [])
        self.assertEqual(evidence["broader_reference_ranges"], [])
        self.assertEqual(evidence["evidence_status"], "insufficient_evidence")
        self.assertTrue(evidence["reference_coverage"]["unavailable"])

    def test_known_phase_matches_are_strict_for_both_phases(self):
        for phase in ("initial_modelling", "existing_model"):
            with self.subTest(phase=phase):
                summary = {"conditional_cost": {
                    "groups": [_cost_group(phase=phase)], "cases": []}}
                evidence = select_cost_evidence(
                    summary, _cost_context([], phase=phase), _candidate())
                self.assertEqual(len(evidence["reference_ranges"]), 1)
                self.assertEqual(
                    evidence["reference_ranges"][0]["reference_grade"], "strict")
                self.assertEqual(evidence["broader_reference_ranges"], [])
                self.assertEqual(evidence["evidence_status"], "strict")

    def test_broader_references_do_not_pollute_shared_cost_scales(self):
        summary = {"conditional_cost": {
            "groups": [_cost_group(phase="initial_modelling")], "cases": []}}
        unknown = _cost_context([])
        unknown.strategy_calibration = summary
        scales = decision_cost_scales(unknown, [_candidate()])
        self.assertNotIn("tool_calls", scales)

        known = _cost_context([], phase="initial_modelling")
        known.strategy_calibration = summary
        scales = decision_cost_scales(known, [_candidate()])
        self.assertEqual(scales["tool_calls"]["peak"], 9)
        self.assertEqual(scales["tool_calls"]["source"],
                         "pre_decision_measured_history")

    def test_missing_dimension_stays_unknown(self):
        group = _cost_group(dimensions={
            "tool_calls": {"p50": 8, "p75": 9, "min": 8, "max": 9,
                           "n": 2, "n_distinct_episodes": 2}})
        evidence = select_cost_evidence(
            {"conditional_cost": {"groups": [group], "cases": []}},
            _cost_context([], phase="initial_modelling"), _candidate())
        shown = evidence["reference_ranges"][0]["dimensions"]
        self.assertIn("tool_calls", shown)
        self.assertNotIn("output_tokens", shown)
        self.assertNotIn("remaining_latency_s", shown)

    def test_attempt_latency_is_not_renamed_into_remaining_latency(self):
        group = _cost_group(dimensions={
            "latency_s": {"p50": 1.5, "p75": 2.0, "min": 1.0, "max": 2.0,
                          "n": 2, "n_distinct_episodes": 2}})
        evidence = select_cost_evidence(
            {"conditional_cost": {"groups": [group], "cases": []}},
            _cost_context([], phase="initial_modelling"), _candidate())
        dimensions = evidence["reference_ranges"][0]["dimensions"]
        # An attempt-latency observation is execution-only; it must not be
        # delivered as a remaining-task C reference or renamed.
        self.assertNotIn("latency_s", dimensions)
        self.assertNotIn("remaining_latency_s", dimensions)

    def test_no_history_reports_insufficient_evidence(self):
        evidence = select_cost_evidence(
            {"conditional_cost": {"groups": [], "cases": []}},
            _cost_context([], phase="initial_modelling"), _candidate())
        self.assertEqual(evidence["evidence_status"], "insufficient_evidence")
        self.assertTrue(evidence["reference_coverage"]["unavailable"])
        self.assertTrue(evidence["missing"])

    def test_case_refs_and_sample_counts_survive_selection(self):
        group = _cost_group(source_refs=["ex-1", "ex-2", "ex-3"])
        evidence = select_cost_evidence(
            {"conditional_cost": {"groups": [group], "cases": []}},
            _cost_context([], phase="initial_modelling"), _candidate())
        shown = evidence["reference_ranges"][0]
        self.assertEqual(shown["n_distinct_episodes"], 2)
        self.assertEqual(shown["n_source_refs"], 3)
        self.assertEqual(shown["source_refs"][:2], ["ex-1", "ex-2"])


# ---------------------------------------------------------------------------
# R
# ---------------------------------------------------------------------------

class TestRGlobalBaseRates(unittest.TestCase):
    R_EVENTS = ("implementation_failure", "timeout", "environment_failure")

    def test_condition_groups_and_global_rates_stay_separate(self):
        rates = {
            "implementation_failure": {"occurred": 2, "not_occurred": 3,
                                       "unknown": 1, "n_known_labels": 5,
                                       "occurrence_rate": 0.4},
            "timeout": {"occurred": 1, "not_occurred": 4, "unknown": 1,
                        "n_known_labels": 5, "occurrence_rate": 0.2},
            "environment_failure": {"occurred": 0, "not_occurred": 2,
                                    "unknown": 4, "n_known_labels": 2,
                                    "occurrence_rate": 0.0},
        }
        group = {
            "conditions": {"solver_interface": "highs",
                           "structure": {"problem_class": "milp"}},
            "observation_rule_version": R_OBSERVATION_VERSION,
            "support": "measured", "n_attempts": 3,
            "events": {name: {"occurred": 1, "not_occurred": 1, "unknown": 1,
                              "n_known_labels": 2, "occurrence_rate": 0.5}
                       for name in self.R_EVENTS},
            "source_refs": ["e1", "e2", "e3"],
            "technical_failures": 1, "technical_successes": 1,
            "unknown": 1, "n_known_labels": 2,
        }
        evidence = select_r_evidence(
            {"technical_r": _technical_summary(groups=[group], events=rates)},
            _r_context(), SimpleNamespace(solver="highs"))
        self.assertEqual(len(evidence["groups"]), 1)
        global_rates = evidence["global_event_base_rates"]
        self.assertEqual(global_rates["events"]["implementation_failure"][
            "occurred"], 2)
        self.assertEqual(global_rates["events"]["implementation_failure"][
            "n_known_labels"], 5)
        self.assertEqual(evidence["support"],
                         "applicable_facts_with_global_base_rates")
        self.assertNotIn("joint", evidence)
        self.assertNotIn("joint_event_probability", evidence)

    def test_no_condition_group_still_returns_global_background(self):
        rates = {
            "implementation_failure": {"occurred": 1, "not_occurred": 1,
                                       "unknown": 0, "n_known_labels": 2,
                                       "occurrence_rate": 0.5},
        }
        evidence = select_r_evidence(
            {"technical_r": _technical_summary(events=rates)},
            _r_context(), SimpleNamespace(solver="glpk"))
        self.assertEqual(evidence["groups"], [])
        self.assertEqual(evidence["support"], "global_base_rates_only")
        self.assertEqual(evidence["global_event_base_rates"]["events"][
            "implementation_failure"]["occurrence_rate"], 0.5)

    def test_unknown_label_is_excluded_from_rate_denominator(self):
        tech = _technical_summary(events={
            "implementation_failure": {"occurred": 2, "not_occurred": 3,
                                       "unknown": 7, "n_known_labels": 5,
                                       "occurrence_rate": 0.4},
        })
        rates = global_event_base_rates(tech)
        entry = rates["events"]["implementation_failure"]
        self.assertEqual(entry["n_known_labels"], 5)
        self.assertEqual(entry["occurrence_rate"], 0.4)
        self.assertNotIn("rate_with_unknown_as_negative", entry)

    def test_no_known_labels_keeps_rate_none(self):
        tech = _technical_summary(events={
            "environment_failure": {"occurred": 0, "not_occurred": 0,
                                    "unknown": 4, "n_known_labels": 0,
                                    "occurrence_rate": None},
        })
        rates = global_event_base_rates(tech)
        self.assertIsNone(rates["events"]["environment_failure"][
            "occurrence_rate"])
        self.assertEqual(rates["support"], "insufficient_evidence")

    def test_unexecuted_and_prediction_only_timeouts_are_not_attempts(self):
        executed = _technical_record("ex-executed", label=1,
                                     event_labels={"timeout": None})
        # The timeout event has no usable label on this attempt (the record
        # says timeout but not whether the harness interrupted it).
        executed.quality["status"] = "timeout"
        unexecuted = _technical_record("ex-unexecuted", source="agent_reported")
        stats = technical_r_statistics(
            {("t", "ep1"): [executed, unexecuted]}, [], min_samples=1)
        self.assertEqual(stats["coverage"]["n_actual_attempts"], 1)
        rates = global_event_base_rates(stats)
        self.assertEqual(rates["n_attempts"], 1)
        self.assertEqual(rates["events"]["implementation_failure"][
            "n_known_labels"], 1)
        # No record at all for a prediction-stage timeout: nothing to add.
        self.assertEqual(rates["events"]["timeout"]["n_known_labels"], 0)

    def test_three_events_keep_their_own_denominators(self):
        rates = {
            "implementation_failure": {"occurred": 1, "not_occurred": 0,
                                       "unknown": 2, "n_known_labels": 1,
                                       "occurrence_rate": 1.0},
            "timeout": {"occurred": 0, "not_occurred": 3, "unknown": 0,
                        "n_known_labels": 3, "occurrence_rate": 0.0},
            "environment_failure": {"occurred": 2, "not_occurred": 2,
                                    "unknown": 0, "n_known_labels": 4,
                                    "occurrence_rate": 0.5},
        }
        parsed = global_event_base_rates(_technical_summary(events=rates))
        self.assertEqual(set(parsed["events"]), set(self.R_EVENTS))
        self.assertEqual(parsed["events"]["implementation_failure"][
            "n_known_labels"], 1)
        self.assertEqual(parsed["events"]["timeout"]["n_known_labels"], 3)
        self.assertEqual(parsed["events"]["environment_failure"][
            "n_known_labels"], 4)
        self.assertIn("maximum", parsed["note"])

    def test_legacy_summary_without_events_remains_readable(self):
        evidence = select_r_evidence(
            {"technical_r": {"groups": [], "coverage": {}}},
            _r_context(), SimpleNamespace(solver="highs"))
        self.assertEqual(evidence["groups"], [])
        self.assertEqual(evidence["global_event_base_rates"], {})
        self.assertEqual(evidence["support"], "insufficient_evidence")


class TestRProviderView(RCase):
    def test_global_rates_reach_the_request_beside_condition_group(self):
        history = self.predict_r(episode="hist")
        self.body(history, text="raise ValueError('bad network indices')\n")
        self.close_g("hist")
        self.predict_r(episode="next")
        evidence = self.provider.requests[-1]["prediction_context"][
            "implementation_r_evidence"]
        self.assertTrue(evidence["groups"])
        rates = evidence["global_event_base_rates"]
        self.assertEqual(rates["events"]["implementation_failure"]["occurred"], 1)
        self.assertEqual(rates["events"]["implementation_failure"][
            "n_known_labels"], 1)
        self.assertNotIn("joint", evidence)


# ---------------------------------------------------------------------------
# Active four-dimensional C prediction contract
# ---------------------------------------------------------------------------

class TestCFourDimensionContract(unittest.TestCase):
    def test_new_prediction_accepts_only_the_four_active_dimensions(self):
        from or_harness.core.schema import PREDICTED_COST_DIMENSIONS
        self.assertEqual(tuple(PREDICTED_COST_DIMENSIONS),
                         ("remaining_latency_s", "output_tokens",
                          "tool_calls", "retries"))
        from or_harness.world_model.contracts import (
            COST_PREDICTION_CONTRACT_VERSION,
        )
        self.assertEqual(COST_PREDICTION_CONTRACT_VERSION,
                         "wm-c-predict/4")

    def test_latency_and_solver_runtime_are_explicit_contract_errors(self):
        from test_strategy_outcome import GOOD_PAYLOAD, StrategyCase
        from or_harness.api import ORHarness

        class Case(StrategyCase):
            def runTest(self):
                pass

        case = Case("runTest")
        case.setUp()
        try:
            case.provider.payload = copy.deepcopy(GOOD_PAYLOAD)
            case.provider.payload["cost"] = {
                "latency_s": 1.0, "solver_runtime_s": 2.0,
                "tool_calls": 3,
            }
            prediction = case.h.predict_strategy_outcome(
                {"task_id": "t1", "family": "routing",
                 "description": "toy routing"},
                {"action_type": "execute_strategy", "strategy_id": "S01"},
                "ep1",
                benefit_convention={"kind": "solution_quality",
                                    "metric": "normalized_objective_gap",
                                    "unit": "1-gap"})
            self.assertEqual(prediction.status, "invalid")
            joined = " ".join(prediction.notes)
            self.assertIn("historical observation dimension", joined)
            self.assertIn("latency_s", joined)
            self.assertIn("solver_runtime_s", joined)
            self.assertIn("remaining_latency_s", joined)
        finally:
            case.doCleanups()

    def test_missing_active_dimensions_stay_unknown_not_zero(self):
        from test_strategy_outcome import GOOD_PAYLOAD, StrategyCase

        class Case(StrategyCase):
            def runTest(self):
                pass

        case = Case("runTest")
        case.setUp()
        try:
            case.provider.payload = copy.deepcopy(GOOD_PAYLOAD)
            case.provider.payload["cost"] = {"tool_calls": 3}
            prediction = case.h.predict_strategy_outcome(
                {"task_id": "t1", "family": "routing",
                 "description": "toy routing"},
                {"action_type": "execute_strategy", "strategy_id": "S01"},
                "ep1",
                benefit_convention={"kind": "solution_quality",
                                    "metric": "normalized_objective_gap",
                                    "unit": "1-gap"})
            self.assertEqual(prediction.status, "valid")
            self.assertEqual(prediction.cost.measured, ["tool_calls"])
            self.assertEqual(prediction.cost.expected.measured_dims(),
                             {"tool_calls"})
            # No other active dimension was silently filled with zero.
            for dim in ("remaining_latency_s", "output_tokens", "retries"):
                self.assertNotIn(dim, prediction.cost.expected.measured_dims())
        finally:
            case.doCleanups()

    def test_u_ignores_archived_old_cost_dimensions(self):
        from or_harness.core.schema import CostVector
        from or_harness.world_model.contracts import (
            BaselineStatement, BenefitEstimate, CandidateRef, ExpectedCost,
            PredictionTrace, StrategyOutcomePrediction,
        )
        from or_harness.world_model.planner import (
            PlanLimits, score_strategy_outcome_predictions,
        )
        prediction = StrategyOutcomePrediction(
            candidate=CandidateRef(action_type="execute_strategy",
                                   strategy_id="S01", task_id="t1"),
            status="valid", service_available=True,
            provider_configured=True, service_implemented=True,
            benefit=BenefitEstimate(
                kind="solution_quality", metric="normalized_objective_gap",
                unit="1-gap", value=0.8,
                baseline=BaselineStatement(kind="conditional_stats",
                                           value=0.5)),
            cost=ExpectedCost(
                expected=CostVector(llm_tokens=100.0,
                                    solver_runtime_s=3.0,
                                    latency_s=1.0,
                                    measured={"llm_tokens",
                                              "solver_runtime_s",
                                              "latency_s"}),
                measured=["llm_tokens", "solver_runtime_s", "latency_s"],
                prediction_contract=None),
            trace=PredictionTrace(prediction_kind="strategy_outcome"))
        scores = score_strategy_outcome_predictions(
            [prediction], PlanLimits(
                alpha=1.0, beta=1.0, gamma=1.0))
        score = scores[0]
        self.assertEqual(score.scoring_basis["cost_dimensions"], [])
        self.assertNotIn("solver_runtime_s",
                         score.cost_reported["values"])
        self.assertIn("solver_runtime_s",
                      score.scoring_basis["omitted_cost_dimensions"])
        self.assertIsNotNone(score.utility)  # G only, no fabricated C

    def test_archived_six_dim_cost_round_trips_without_conversion(self):
        from or_harness.core.schema import CostVector
        from or_harness.world_model.contracts import (
            CandidateRef, ExpectedCost, PredictionTrace,
            StrategyOutcomePrediction, validate_strategy_outcome,
        )
        old = ExpectedCost(
            expected=CostVector(llm_tokens=10.0, solver_runtime_s=2.0,
                                latency_s=1.0,
                                measured={"llm_tokens", "solver_runtime_s",
                                          "latency_s"}),
            measured=["llm_tokens", "solver_runtime_s", "latency_s"])
        self.assertIsNone(old.prediction_contract)
        prediction = StrategyOutcomePrediction(
            candidate=CandidateRef(action_type="execute_strategy",
                                   strategy_id="S01", task_id="t1"),
            status="contract_only", cost=old,
            trace=PredictionTrace(prediction_kind="strategy_outcome"))
        dumped = prediction.to_dict()
        restored = StrategyOutcomePrediction.from_dict(dumped)
        self.assertEqual(restored.to_dict(), dumped)
        self.assertIsNone(restored.cost.prediction_contract)
        self.assertEqual(restored.cost.expected.solver_runtime_s, 2.0)
        # Archived data is readable, not re-validated as a new contract.
        self.assertEqual(validate_strategy_outcome(restored), [])
