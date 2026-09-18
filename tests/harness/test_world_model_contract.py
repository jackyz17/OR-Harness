"""Unified world-model contract (Phase 1) regressions.

Focus: the boundaries that matter in production, not the happy path only.

 1  round-trip serialization, provenance preservation, and unknown values
    that must NOT become a measured zero;
 2  frozen snapshots that do not change when a caller mutates its inputs;
 3  legacy unversioned payloads still readable, WITHOUT acquiring new
    semantics; an unknown contract version failing explicitly;
 4  a contract buildable with no complete mathematical model and partial
    unknowns;
 5  capability evidence that never fabricates an H score from counts;
 6  attempt vs strategy execution window never silently interchanged;
 7  risk and cost kept separate; prediction data never entering execution
    statistics or knowledge verification;
 8  recommendation / selection / execution separation, and the old
    binding, cost-dedup and feedback idempotency not regressing.
"""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tests.harness.helpers import HarnessTestCase  # noqa: E402

from or_harness.api import ORHarness  # noqa: E402
from or_harness.core.schema import CostVector  # noqa: E402
from or_harness.world_model.contracts import (  # noqa: E402
    CONTRACT_VERSION,
    LEGACY_CONTRACT_VERSION,
    LEGACY_UNMAPPABLE,
    BaselineStatement,
    BenefitEstimate,
    CandidateRef,
    CapabilityEvolutionPrediction,
    CapabilitySourceEvidence,
    EvidenceRef,
    ExpectedChange,
    ExpectedCost,
    ExperienceScope,
    HarnessCapabilityEvidence,
    LearningOperation,
    PredictionTrace,
    RiskEvent,
    RiskStatement,
    StrategyOutcomePrediction,
    TaskTargeting,
    UncertaintyStatement,
    UnsupportedContractVersion,
    VerificationCondition,
    capability_evidence_from_legacy_harness_state,
    detect_payload_version,
    legacy_prediction_view,
    load_contract_payload,
    prediction_kinds_for_mode,
    validate_capability_evolution,
    validate_capability_evidence,
    validate_strategy_outcome,
)
from or_harness.world_model.execution_window import (  # noqa: E402
    DEFAULT_AUXILIARY_ACTION_TYPES,
    DEFAULT_IN_SCOPE_ACTION_TYPES,
    build_execution_window,
    window_id_for,
)
from or_harness.world_model.prediction import ActionSpec  # noqa: E402

TASK = {"task_id": "t1", "family": "routing", "description": "toy routing",
        "spec": {"n_vars": 100, "n_constraints": 50},
        "annotations": {"coupling": {"resource_coupling": 0.9,
                                     "temporal_coupling": 0.1,
                                     "route_complexity": 0.85}}}


def _benefit(value=0.8, kind="solution_quality"):
    return BenefitEstimate(
        kind=kind, metric="normalized_objective_gap", unit="1-gap",
        value=value,
        baseline=BaselineStatement(kind="conditional_stats", value=0.7))


class TestContractRoundTrip(HarnessTestCase):
    """Requirement 1: round-trip, provenance, unknown != zero."""

    def test_strategy_outcome_round_trip_is_lossless(self):
        prediction = StrategyOutcomePrediction(
            candidate=CandidateRef(action_type="execute_strategy",
                                   strategy_id="S01", task_id="t1",
                                   scope="attempt"),
            status="contract_only",
            benefit=_benefit(),
            cost=ExpectedCost(
                expected=CostVector(llm_tokens=1200.0,
                                    solver_runtime_s=3.0,
                                    measured={"llm_tokens",
                                              "solver_runtime_s"}),
                measured=["llm_tokens", "solver_runtime_s"]),
            risk=RiskStatement(events=[RiskEvent(event="task_failure",
                                                 probability=0.2)]),
            uncertainty=UncertaintyStatement(
                source="framework_heuristic", knowledge_gap=0.6),
            trace=PredictionTrace(
                prediction_kind="strategy_outcome",
                input_snapshot_id="bs_x", input_version="digest1",
                evidence_basis=[EvidenceRef("execution", "ex_1", "support")],
                unsupported_fields={"retries": "no evidence"}),
        )
        dumped = prediction.to_dict()
        reloaded = StrategyOutcomePrediction.from_dict(dumped)
        self.assertEqual(reloaded.to_dict(), dumped)

    def test_capability_evolution_round_trip_is_lossless(self):
        prediction = CapabilityEvolutionPrediction(
            current_evidence=HarnessCapabilityEvidence(
                version="cap1",
                sources={"m": CapabilitySourceEvidence(
                    source="m", status="indirect_evidence",
                    evidence=[EvidenceRef("knowledge_layer", "verified")])}),
            candidate_operation=LearningOperation(
                operation_type="induce", strategy_id="S01",
                scope=ExperienceScope(execution_ids=["ex_1"],
                                      task_ids=["t1", "t2"])),
            status="contract_only",
            task_targeting=TaskTargeting(description="routing",
                                         family="routing"),
            baseline=BaselineStatement(kind="no_knowledge", value=0.5),
            horizon="next 10 matching tasks", horizon_tasks=10,
            expected_changes=[ExpectedChange(
                metric="mean_quality", direction="increase", value=0.05,
                baseline=BaselineStatement(kind="no_knowledge"))],
            learning_cost=ExpectedCost(
                expected=CostVector(llm_tokens=4000.0,
                                    measured={"llm_tokens"}),
                measured=["llm_tokens"]),
            degradation_risk=RiskStatement(events=[RiskEvent(
                event="overgeneralization", probability=0.3)]),
            verification_conditions=[VerificationCondition(
                condition="interval holds on 5 unseen tasks",
                evaluable=True, check_basis="induce --verify")],
            trace=PredictionTrace(prediction_kind="capability_evolution"),
        )
        dumped = prediction.to_dict()
        reloaded = CapabilityEvolutionPrediction.from_dict(dumped)
        self.assertEqual(reloaded.to_dict(), dumped)
        # distinct-task count is DERIVED, never re-typed by the caller
        self.assertEqual(
            reloaded.candidate_operation.scope.distinct_tasks, 2)

    def test_unknown_is_never_a_measured_zero(self):
        """An absent benefit/cost stays absent; the measured mask survives."""
        prediction = StrategyOutcomePrediction(
            candidate=CandidateRef(action_type="execute_strategy",
                                   strategy_id="S01", task_id="t1"),
            status="contract_only",
            cost=ExpectedCost(
                expected=CostVector(llm_tokens=100.0, measured={"llm_tokens"}),
                measured=["llm_tokens"]),
            trace=PredictionTrace(prediction_kind="strategy_outcome"))
        dumped = prediction.to_dict()
        self.assertIsNone(dumped["benefit"])
        self.assertIsNone(dumped["uncertainty"])
        self.assertEqual(dumped["cost"]["expected_measured"],
                         ["llm_tokens"])
        reloaded = StrategyOutcomePrediction.from_dict(dumped)
        measured = reloaded.cost.expected.measured_dims()
        self.assertEqual(measured, {"llm_tokens"})
        self.assertNotIn("solver_runtime_s", measured)

    def test_unsupported_fields_are_preserved_with_reasons(self):
        prediction = StrategyOutcomePrediction(
            candidate=CandidateRef(action_type="execute_strategy",
                                   strategy_id="S01", task_id="t1"),
            status="contract_only",
            trace=PredictionTrace(
                prediction_kind="strategy_outcome",
                unsupported_fields={"risk.severity": "no loss model",
                                    "cost.retries": "not observed"}))
        dumped = prediction.to_dict()
        self.assertEqual(
            StrategyOutcomePrediction.from_dict(dumped).to_dict(), dumped)

    def test_evidence_basis_requires_ref_type_and_id(self):
        with self.assertRaises(ValueError):
            EvidenceRef.from_dict({"ref_type": "execution"})
        with self.assertRaises(ValueError):
            EvidenceRef.from_dict({"ref_id": "ex_1"})
        ref = EvidenceRef.from_dict({"ref_type": "entry", "ref_id": "se_1"})
        self.assertEqual(ref.to_dict(),
                         {"ref_type": "entry", "ref_id": "se_1",
                          "note": None})


class TestValidationRules(HarnessTestCase):
    """The rules that keep a contract honest."""

    def _base(self, **kwargs):
        defaults = dict(
            candidate=CandidateRef(action_type="execute_strategy",
                                   strategy_id="S01", task_id="t1"),
            status="contract_only",
            trace=PredictionTrace(prediction_kind="strategy_outcome",
                                  input_snapshot_id="bs_1"))
        defaults.update(kwargs)
        return StrategyOutcomePrediction(**defaults)

    def test_benefit_value_without_baseline_is_rejected(self):
        prediction = self._base(benefit=BenefitEstimate(
            kind="solution_quality", metric="q", value=0.8))
        self.assertTrue(any("baseline" in p
                            for p in validate_strategy_outcome(prediction)))

    def test_valid_status_requires_a_service(self):
        prediction = self._base(status="valid", benefit=_benefit())
        self.assertTrue(any("prediction service" in p
                            for p in validate_strategy_outcome(prediction)))

    def test_contract_only_contradicting_service_is_rejected(self):
        prediction = self._base(service_available=True)
        self.assertTrue(any("contradicts" in p
                            for p in validate_strategy_outcome(prediction)))

    def test_window_scope_requires_a_recorded_window(self):
        prediction = self._base(candidate=CandidateRef(
            action_type="execute_strategy", strategy_id="S01", task_id="t1",
            scope="strategy_window"))
        self.assertTrue(any("window_id" in p
                            for p in validate_strategy_outcome(prediction)))

    def test_model_self_report_is_not_a_calibrated_probability(self):
        prediction = self._base(
            uncertainty=UncertaintyStatement(source="model_self_report",
                                             knowledge_gap=0.4))
        self.assertTrue(any("model_self_report" in p
                            for p in validate_strategy_outcome(prediction)))

    def test_execute_strategy_candidate_requires_strategy_id(self):
        prediction = self._base(candidate=CandidateRef(
            action_type="execute_strategy", task_id="t1"))
        self.assertTrue(any("strategy_id" in p
                            for p in validate_strategy_outcome(prediction)))

    def test_risk_and_cost_are_separate_fields(self):
        """A risk event may not smuggle a cost dimension."""
        prediction = self._base(risk=RiskStatement(events=[RiskEvent(
            event="rework", probability=0.3)]))
        dumped = prediction.to_dict()
        self.assertIn("risk", dumped)
        self.assertIn("cost", dumped)
        self.assertNotIn("cost", dumped["risk"])
        self.assertIsNone(dumped["cost"])

    def test_capability_evolution_requires_horizon_and_conditions(self):
        prediction = CapabilityEvolutionPrediction(
            current_evidence=HarnessCapabilityEvidence(),
            candidate_operation=LearningOperation(operation_type="induce"),
            status="valid", service_available=True,
            horizon="")
        problems = validate_capability_evolution(prediction)
        self.assertTrue(any("horizon" in p for p in problems))
        self.assertTrue(any("task_targeting" in p for p in problems))
        self.assertTrue(any("baseline" in p for p in problems))
        self.assertTrue(any("verification_conditions" in p for p in problems))
        self.assertTrue(any("expected_change" in p for p in problems))

    def test_expected_change_value_with_unknown_direction_is_rejected(self):
        prediction = CapabilityEvolutionPrediction(
            current_evidence=HarnessCapabilityEvidence(),
            candidate_operation=LearningOperation(operation_type="induce"),
            status="contract_only", horizon="next task",
            expected_changes=[ExpectedChange(metric="q", direction="unknown",
                                             value=0.1)])
        self.assertTrue(any("contradictory" in p for p in
                            validate_capability_evolution(prediction)))

    def test_invalid_operation_type_is_rejected(self):
        with self.assertRaises(ValueError):
            LearningOperation(operation_type="finetune_model")

    def test_invalid_scope_is_rejected(self):
        with self.assertRaises(ValueError):
            CandidateRef(action_type="execute_strategy", scope="task_total")

    def test_prediction_kinds_for_legacy_modes(self):
        self.assertEqual(prediction_kinds_for_mode("x-b-only"),
                         ("strategy_outcome",))
        self.assertEqual(prediction_kinds_for_mode("h-x-b-value"),
                         ("strategy_outcome", "capability_evolution"))
        with self.assertRaises(ValueError):
            prediction_kinds_for_mode("nonsense")


class TestCapabilityEvidenceNoScore(HarnessTestCase):
    """Requirement 5: H evidence is never turned into a composite score."""

    def test_legacy_harness_state_maps_to_indirect_evidence_only(self):
        evidence = capability_evidence_from_legacy_harness_state({
            "knowledge": {"verified": [{"entry_id": "se_1"}],
                          "legacy_unknown": [], "unverified": []},
            "experience": {"task_execution_count": 3,
                           "total_executions": 9},
            "tool_config": {"available_solver_families": {"milp": ["highs"]},
                            "executor_timeout_seconds": 30},
        })
        self.assertEqual(evidence.score_scheme, "no_composite_score")
        self.assertEqual(evidence.sources["m"].status, "indirect_evidence")
        self.assertEqual(evidence.sources["t"].status, "indirect_evidence")
        # Nothing observed these three: they stay no_evidence rather than
        # being filled from an unrelated count.
        for source in ("w_or", "pi", "r"):
            self.assertEqual(evidence.sources[source].status, "no_evidence")
        dumped = evidence.to_dict()
        self.assertNotIn("composite_score", dumped)
        self.assertNotIn("h_score", dumped)

    def test_composite_score_scheme_is_rejected(self):
        evidence = HarnessCapabilityEvidence(score_scheme="weighted_sum_v1")
        problems = validate_capability_evidence(evidence)
        self.assertTrue(any("composite" in p for p in problems))

    def test_direct_evidence_requires_a_reference(self):
        evidence = HarnessCapabilityEvidence(sources={
            "w_or": CapabilitySourceEvidence(source="w_or",
                                             status="direct_evidence")})
        problems = validate_capability_evidence(evidence)
        self.assertTrue(any("direct_evidence" in p for p in problems))

    def test_no_complete_model_still_builds_capability_contract(self):
        """A task with no 'model' field is a normal state."""
        h = ORHarness(home=self.home)
        self.addCleanup(h.close)
        self.assertNotIn("model", TASK)
        prediction = h.build_capability_evolution_contract(
            TASK, LearningOperation(operation_type="induce",
                                    strategy_id="S01"),
            horizon="next 10 matching tasks")
        self.assertEqual(prediction.status, "contract_only")
        self.assertFalse(prediction.service_available)
        self.assertIn("contract_only", " ".join(prediction.notes))
        self.assertEqual(prediction.current_evidence.score_scheme,
                         "no_composite_score")

    def test_capability_evidence_without_task_says_not_consulted(self):
        h = ORHarness(home=self.home)
        self.addCleanup(h.close)
        evidence = h.capability_evidence()
        self.assertEqual(evidence.sources["m"].status,
                         "indirect_evidence")
        self.assertTrue(any("NOT consulted" in n
                            for n in evidence.notes))


class TestLegacyCompatibility(HarnessTestCase):
    """Requirement 3: legacy payloads stay readable, without new semantics."""

    LEGACY = {
        "prediction_id": "wp_legacy",
        "input_snapshot_id": "bs_legacy",
        "action_spec": {"action_type": "execute_strategy",
                        "strategy_id": "S01", "task_id": "t1",
                        "measurement_scope": "attempt"},
        "status": "valid",
        "predicted": {"outcome_status": "feasible", "feasible": True,
                      "quality": 0.8, "failure_prob": 0.2,
                      "cost": {"llm_tokens": 100.0},
                      "cost_measured": ["llm_tokens"],
                      "state_changes": {"current_solution": {
                          "status": "feasible"}},
                      "knowledge_changes": [{"change": "supports"}]},
        "confidence": 0.6,
        "evidence_basis": ["coverage.cell_statistics.S01"],
        "unsupported_fields": {"retries": "no evidence"},
        "call_cost": {"llm_tokens": 50.0},
        "created_at": 1.0,
    }

    def test_unversioned_payload_is_identified_as_legacy(self):
        self.assertEqual(detect_payload_version(self.LEGACY),
                         LEGACY_CONTRACT_VERSION)
        self.assertEqual(detect_payload_version({}),
                         LEGACY_CONTRACT_VERSION)

    def test_legacy_payload_loads_as_legacy_not_as_current(self):
        with self.assertRaises(UnsupportedContractVersion):
            load_contract_payload(self.LEGACY)

    def test_legacy_view_preserves_old_semantics(self):
        view = legacy_prediction_view(self.LEGACY)
        mapped = view["mapped_fields"]
        self.assertFalse(view["is_current_contract"])
        self.assertEqual(mapped["candidate.expected_status"], "feasible")
        self.assertEqual(mapped["benefit.value"], 0.8)
        self.assertEqual(mapped["cost.expected"], {"llm_tokens": 100.0})
        self.assertEqual(mapped["risk.events"][0]["probability"], 0.2)
        # No invented severity, no invented capability increment.
        self.assertIsNone(mapped["risk.events"][0]["severity"])
        self.assertNotIn("capability_increment", mapped)
        self.assertTrue(any("severity" in g for g in view["gaps"]))

    def test_legacy_view_never_derives_a_capability_level(self):
        view = legacy_prediction_view(self.LEGACY)
        evidence = view["capability_evidence"]
        self.assertEqual(evidence["score_scheme"], "no_composite_score")
        for source in ("w_or", "pi", "r", "t"):
            self.assertEqual(evidence["sources"][source]["status"],
                             "no_evidence")
        # A legacy payload carries evidence references, so M is INDIRECT
        # evidence at most — never a level.
        self.assertEqual(evidence["sources"]["m"]["status"],
                         "indirect_evidence")

    def test_unmappable_content_is_named(self):
        view = legacy_prediction_view(self.LEGACY)
        for key in ("harness_state.knowledge", "budget_state",
                    "state_changes (field-level detail)"):
            self.assertIn(key, view["unmappable"])
        self.assertEqual(set(view["unmappable"]), set(LEGACY_UNMAPPABLE))

    def test_unknown_version_fails_explicitly(self):
        payload = {"contract_version": "wm-contract/9",
                   "prediction_id": "x",
                   "prediction_type": "strategy_outcome"}
        self.assertEqual(detect_payload_version(payload),
                         "unknown/wm-contract/9")
        with self.assertRaises(UnsupportedContractVersion):
            load_contract_payload(payload)

    def test_api_read_returns_unsupported_without_parsing(self):
        result = ORHarness.read_prediction_payload(
            {"contract_version": "wm-contract/9"})
        self.assertFalse(result["supported"])
        self.assertNotIn("contract", result)
        self.assertNotIn("legacy_view", result)

    def test_api_read_legacy_returns_the_view(self):
        result = ORHarness.read_prediction_payload(self.LEGACY)
        self.assertTrue(result["supported"])
        self.assertTrue(result["legacy"])
        self.assertEqual(result["contract_version"],
                         LEGACY_CONTRACT_VERSION)

    def test_current_contract_payload_round_trips_through_read(self):
        prediction = StrategyOutcomePrediction(
            candidate=CandidateRef(action_type="execute_strategy",
                                   strategy_id="S01", task_id="t1"),
            status="contract_only", benefit=_benefit(),
            trace=PredictionTrace(prediction_kind="strategy_outcome",
                                  input_snapshot_id="bs_1"))
        result = ORHarness.read_prediction_payload(prediction.to_dict())
        self.assertTrue(result["supported"])
        self.assertFalse(result["legacy"])
        self.assertEqual(result["contract_version"], CONTRACT_VERSION)
        self.assertEqual(result["contract"], prediction.to_dict())

    def test_malformed_current_payload_is_unsupported_not_a_crash(self):
        result = ORHarness.read_prediction_payload({
            "contract_version": CONTRACT_VERSION, "prediction_id": "x",
            "prediction_type": "not_a_kind"})
        self.assertFalse(result["supported"])
        self.assertIn("not_a_kind", result["error"])


class TestExecutionWindow(HarnessTestCase):
    """Requirement 6: attempt vs strategy window are never interchanged."""

    class _Action:
        def __init__(self, action_id, action_type, *, status="completed",
                     strategy_id=None, source="executed", rollup="own",
                     cost=None, execution_id=None, started_at=0.0,
                     episode_id="ep1", task_id="t1"):
            self.action_id = action_id
            self.action_type = action_type
            self.task_id = task_id
            self.episode_id = episode_id
            self.source = source
            self.status = status
            self.rollup = rollup
            self.cost = cost
            self.linked_execution_id = execution_id
            self.started_at = started_at
            self.ended_at = started_at + 1.0
            self.params = ({"strategy_id": strategy_id}
                           if strategy_id else {})

    def _window(self, actions, **kwargs):
        return build_execution_window(actions, task_id="t1", episode_id="ep1",
                                      strategy_id="S01", **kwargs)

    def test_window_with_no_attempt_is_not_comparable(self):
        window = self._window([])
        self.assertFalse(window.comparable)
        self.assertTrue(any("no in-scope executed attempt" in r
                            for r in window.not_comparable_reasons))

    def test_attempt_without_linked_execution_is_not_comparable(self):
        window = self._window([self._Action("ac_1", "execute_strategy",
                                            strategy_id="S01",
                                            execution_id=None)])
        self.assertFalse(window.comparable)
        self.assertTrue(any("no linked execution" in r
                            for r in window.not_comparable_reasons))

    def test_real_attempt_makes_the_window_comparable(self):
        window = self._window([self._Action(
            "ac_1", "execute_strategy", strategy_id="S01",
            execution_id="ex_1")])
        self.assertTrue(window.comparable)
        self.assertEqual(window.n_attempts, 1)
        self.assertEqual(window.window_id,
                         window_id_for("t1", "ep1", "S01"))

    def test_auxiliary_actions_are_reported_not_folded_in(self):
        window = self._window([
            self._Action("ac_m", "model", cost=CostVector(
                llm_tokens=500.0, measured={"llm_tokens"})),
            self._Action("ac_1", "execute_strategy", strategy_id="S01",
                         execution_id="ex_1",
                         cost=CostVector(solver_runtime_s=2.0,
                                         measured={"solver_runtime_s"})),
            self._Action("ac_v", "verify"),
        ])
        self.assertEqual(window.n_attempts, 1)
        self.assertEqual([a.action_type for a in window.auxiliary_actions],
                         ["model", "verify"])
        # The auxiliary spend is visible and separate — the window never
        # claims modeling was free.
        self.assertEqual(window.auxiliary_cost["llm_tokens"]["total"], 500.0)
        self.assertTrue(window.auxiliary_cost["llm_tokens"]["partial"])
        self.assertEqual(window.attempt_cost["solver_runtime_s"]["total"],
                         2.0)
        self.assertIsNone(window.attempt_cost["llm_tokens"]["total"])

    def test_reference_rollup_is_never_summed_twice(self):
        window = self._window([
            self._Action("ac_1", "execute_strategy", strategy_id="S01",
                         execution_id="ex_1"),
            self._Action("ac_v", "verify", rollup="reference",
                         cost=CostVector(llm_tokens=999.0,
                                         measured={"llm_tokens"})),
        ])
        # The reference-cost action is excluded from the aggregation
        # entirely: its spend is already counted on its child.
        self.assertIsNone(window.auxiliary_cost["llm_tokens"]["total"])
        self.assertEqual(window.auxiliary_cost["llm_tokens"]["n_items"], 0)

    def test_partial_measurement_is_reported_not_hidden(self):
        window = self._window([
            self._Action("ac_1", "execute_strategy", strategy_id="S01",
                         execution_id="ex_1",
                         cost=CostVector(llm_tokens=10.0,
                                         measured={"llm_tokens"}),
                         started_at=0.0),
            self._Action("ac_2", "execute_strategy", strategy_id="S01",
                         execution_id="ex_2",
                         cost=CostVector(solver_runtime_s=1.0,
                                         measured={"solver_runtime_s"}),
                         started_at=5.0),
        ])
        tokens = window.attempt_cost["llm_tokens"]
        self.assertEqual(tokens["total"], 10.0)
        self.assertEqual(tokens["n_measured"], 1)
        self.assertEqual(tokens["n_items"], 2)
        self.assertTrue(tokens["partial"])
        self.assertFalse(tokens["complete"])
        # A dimension NO attempt measured is unknown, never a zero total.
        self.assertIsNone(window.attempt_cost["retries"]["total"])
        self.assertEqual(window.attempt_cost["retries"]["n_measured"], 0)

    def test_hypothetical_actions_are_excluded(self):
        window = self._window([self._Action(
            "ac_h", "execute_strategy", strategy_id="S01",
            execution_id="ex_h", source="hypothetical")])
        self.assertEqual(window.n_attempts, 0)
        self.assertFalse(window.comparable)

    def test_other_strategy_actions_are_excluded(self):
        window = self._window([
            self._Action("ac_1", "execute_strategy", strategy_id="S01",
                         execution_id="ex_1"),
            self._Action("ac_2", "execute_strategy", strategy_id="S02",
                         execution_id="ex_2"),
        ])
        self.assertEqual(window.n_attempts, 1)
        self.assertTrue(any("another strategy" in n
                            for n in window.notes))

    def test_overlapping_scope_and_auxiliary_is_rejected(self):
        with self.assertRaises(ValueError):
            self._window([], in_scope_action_types=["verify"],
                         auxiliary_action_types=["verify"])

    def test_unknown_action_type_is_rejected(self):
        with self.assertRaises(ValueError):
            self._window([], in_scope_action_types=["invent_action"])

    def test_window_scope_candidate_gets_the_real_window_id(self):
        h = ORHarness(home=self.home)
        self.addCleanup(h.close)
        prediction = h.build_strategy_outcome_contract(
            TASK, CandidateRef(action_type="execute_strategy",
                               strategy_id="S01", task_id="t1",
                               scope="strategy_window"), "ep1")
        self.assertEqual(prediction.candidate.window_id,
                         window_id_for("t1", "ep1", "S01"))
        self.assertFalse(prediction.trace.comparable)
        self.assertTrue(prediction.trace.not_comparable_reasons)

    def test_default_scope_is_the_solve_attempt(self):
        self.assertEqual(DEFAULT_IN_SCOPE_ACTION_TYPES,
                         ("execute_strategy",))
        self.assertIn("model", DEFAULT_AUXILIARY_ACTION_TYPES)
        self.assertIn("verify", DEFAULT_AUXILIARY_ACTION_TYPES)


class TestApiContracts(HarnessTestCase):
    """Requirements 4/8: buildable with unknowns; separation intact."""

    def _harness(self):
        h = ORHarness(home=self.home)
        self.addCleanup(h.close)
        return h

    def test_contract_builds_without_a_model_and_with_unknowns(self):
        h = self._harness()
        task = {"task_id": "t_unknown", "family": "routing",
                "description": "vague task"}
        prediction = h.build_strategy_outcome_contract(
            task, ActionSpec(action_type="execute_strategy",
                             task_id="t_unknown", strategy_id="S01"),
            "ep1")
        self.assertEqual(prediction.status, "contract_only")
        self.assertIsNone(prediction.benefit)
        self.assertIsNone(prediction.cost)
        self.assertIsNone(prediction.risk)
        self.assertEqual(prediction.trace.input_version is not None, True)

    def test_build_derives_the_trace_instead_of_asking_the_caller(self):
        h = self._harness()
        prediction = h.build_strategy_outcome_contract(
            TASK, CandidateRef(action_type="execute_strategy",
                               strategy_id="S01"), "ep1")
        self.assertEqual(prediction.contract_version, CONTRACT_VERSION)
        self.assertEqual(prediction.prediction_type, "strategy_outcome")
        self.assertTrue(prediction.trace.input_snapshot_id.startswith("bs_"))
        self.assertGreater(prediction.trace.created_at, 0)
        self.assertEqual(prediction.trace.prediction_version,
                         "not-attached")

    def test_candidate_from_another_task_is_rejected(self):
        h = self._harness()
        with self.assertRaises(ValueError):
            h.build_strategy_outcome_contract(
                TASK, CandidateRef(action_type="execute_strategy",
                                   strategy_id="S01", task_id="other"),
                "ep1")

    def test_service_availability_follows_the_provider(self):
        h = self._harness()
        self.assertFalse(h.prediction_service_available("strategy_outcome"))
        self.assertFalse(
            h.prediction_service_available("capability_evolution"))
        with self.assertRaises(ValueError):
            h.prediction_service_available("nonsense")

    def test_snapshot_is_frozen_against_caller_mutation(self):
        h = self._harness()
        candidate = CandidateRef(action_type="execute_strategy",
                                 strategy_id="S01", task_id="t1",
                                 config={"time_limit": 60})
        prediction = h.build_strategy_outcome_contract(TASK, candidate,
                                                       "ep1")
        candidate.config["time_limit"] = 9999
        self.assertEqual(prediction.candidate.config["time_limit"], 60)

    def test_contract_never_writes_a_fact_or_knowledge(self):
        """Building a contract must not touch the banks."""
        h = self._harness()
        before_exec = h.bank.count()
        before_entries = len(h.sbank.all()) if hasattr(h.sbank, "all") else None
        h.build_strategy_outcome_contract(
            TASK, CandidateRef(action_type="execute_strategy",
                               strategy_id="S01"), "ep1")
        h.build_capability_evolution_contract(
            TASK, LearningOperation(operation_type="induce",
                                    strategy_id="S01"),
            horizon="next task")
        self.assertEqual(h.bank.count(), before_exec)
        if before_entries is not None:
            self.assertEqual(len(h.sbank.all()), before_entries)

    def test_prediction_data_never_enters_execution_statistics(self):
        """A contract build leaves the conditional statistics untouched."""
        h = self._harness()
        profile = h.profile(TASK)
        before = h.stats.for_profile(profile)
        h.build_strategy_outcome_contract(
            TASK, CandidateRef(action_type="execute_strategy",
                               strategy_id="S01"), "ep1")
        after = h.stats.for_profile(profile)
        self.assertEqual(sorted(before), sorted(after))
        for sid in before:
            self.assertEqual(before[sid].n, after[sid].n)

    def test_legacy_prediction_path_is_unchanged(self):
        """The old OutcomePrediction path still returns not_configured."""
        h = self._harness()
        spec = ActionSpec(action_type="execute_strategy", task_id="t1",
                          strategy_id="S01")
        prediction = h.predict_outcome(TASK, spec, "ep1")
        self.assertEqual(prediction.status, "not_configured")
        # And it is NOT reported as a current-contract prediction.
        self.assertEqual(detect_payload_version(prediction.to_dict()),
                         LEGACY_CONTRACT_VERSION)


class TestCliContract(HarnessTestCase):
    """The documented CLI surface must actually exist and run."""

    def _run(self, argv):
        from or_harness.cli import main
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = main(["--home", self.home] + argv)
        return code, json.loads(buf.getvalue())

    def test_contract_command_builds_strategy_outcome(self):
        code, payload = self._run([
            "contract", "--kind", "strategy_outcome",
            "--task", json.dumps(TASK),
            "--spec", json.dumps({"action_type": "execute_strategy",
                                  "strategy_id": "S01",
                                  "scope": "attempt"}),
            "--episode", "ep1"])
        self.assertEqual(code, 0)
        self.assertEqual(payload["result"]["prediction_type"],
                         "strategy_outcome")
        self.assertEqual(payload["result"]["status"], "contract_only")

    def test_contract_command_reads_legacy_payload(self):
        code, payload = self._run([
            "contract", "--payload",
            json.dumps(TestLegacyCompatibility.LEGACY)])
        self.assertEqual(code, 0)
        self.assertEqual(payload["result"]["contract_version"],
                         LEGACY_CONTRACT_VERSION)
        self.assertTrue(payload["result"]["legacy"])

    def test_contract_command_fails_on_unknown_version(self):
        code, payload = self._run([
            "contract", "--payload",
            json.dumps({"contract_version": "wm-contract/9"})])
        self.assertEqual(code, 2)
        self.assertIn("error", payload["result"])

    def test_contract_command_builds_capability_evolution(self):
        code, payload = self._run([
            "contract", "--kind", "capability_evolution",
            "--operation", json.dumps({"operation_type": "induce",
                                       "strategy_id": "S01"}),
            "--horizon", "next 10 matching tasks",
            "--verification", json.dumps({
                "condition": "interval holds on unseen tasks",
                "evaluable": True})])
        self.assertEqual(code, 0)
        self.assertEqual(payload["result"]["status"], "contract_only")
        self.assertFalse(payload["result"]["service_available"])

    def test_contract_without_required_task_fails_cleanly(self):
        code, payload = self._run([
            "contract", "--kind", "strategy_outcome",
            "--spec", json.dumps({"action_type": "execute_strategy",
                                  "strategy_id": "S01"})])
        self.assertEqual(code, 2)
        self.assertIn("requires --task", payload["result"]["error"])


class TestDocumentationMatchesCode(unittest.TestCase):
    """Requirement: documented flags/symbols/links must really exist.

    The docs are a deliverable, so drift between them and the code is a
    defect, not a cosmetic issue: an agent following a documented flag that
    does not exist wastes a whole attempt.
    """

    def test_documented_cli_flags_and_symbols_exist(self):
        import subprocess
        root = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                            "..", ".."))
        script = os.path.join(root, "references", "examples", "_check_docs.py")
        self.assertTrue(os.path.exists(script), "doc checker is missing")
        env = dict(os.environ, PYTHONPATH=os.path.join(root, "src"))
        completed = subprocess.run(
            [sys.executable, script], cwd=root, env=env,
            capture_output=True, text=True)
        self.assertEqual(
            completed.returncode, 0,
            f"documentation drifted from the code:\n{completed.stdout}\n"
            f"{completed.stderr}")

    def test_contract_reference_doc_is_linked_from_the_agent_entry(self):
        from pathlib import Path
        root = Path(os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "..")))
        skill = (root / "SKILL.md").read_text(encoding="utf-8")
        self.assertIn("references/world_model_contract.md", skill)
        # The entry doc must stay short and agent-facing: no internal
        # database or research-symbol vocabulary.
        for forbidden in ("world_model_predictions", "SQLite",
                          "CREATE TABLE"):
            self.assertNotIn(forbidden, skill)

    def test_readme_documents_contract_only_honestly(self):
        from pathlib import Path
        root = Path(os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "..")))
        for name in ("README.md", "README_zh.md"):
            text = (root / name).read_text(encoding="utf-8")
            self.assertIn("contract_only", text,
                          f"{name} must state the service-not-attached state")

    def test_docs_do_not_claim_the_whole_reconstruction_is_done(self):
        from pathlib import Path
        root = Path(os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "..")))
        contract_doc = (root / "references" / "world_model_contract.md"
                        ).read_text(encoding="utf-8")
        self.assertIn("## 9. What is NOT in this phase", contract_doc)
        for absent in ("no task-closing scheduler",
                       "no H evaluation system",
                       "no multi-step latent rollouts"):
            self.assertIn(absent, contract_doc)

    def test_runnable_example_passes(self):
        import subprocess
        root = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                            "..", ".."))
        script = os.path.join(root, "references", "examples",
                              "contract_roundtrip.py")
        env = dict(os.environ, PYTHONPATH=os.path.join(root, "src"))
        completed = subprocess.run(
            [sys.executable, script], cwd=root, env=env,
            capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0,
                         f"documented example failed:\n{completed.stdout}\n"
                         f"{completed.stderr}")
        self.assertIn("All assertions passed", completed.stdout)


if __name__ == "__main__":
    unittest.main()
