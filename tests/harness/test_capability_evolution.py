"""World-model M5 tests: capability-evolution prediction (wm-ce/1), the
offline-improvement decision, and the two-stage capability feedback.

What this file asserts, one behaviour per test class:

1. the protocol really reaches the provider: a ``wm-ce/1`` request selects
   the capability prompt and schema on a REAL local HTTP stub, the payload
   parses into a ``CapabilityEvolutionPrediction``, and it survives a save
   and a restore;
2. the framework's identity is not negotiable: a payload that restates the
   operation, the baseline, the horizon or ``effect_verified`` has those
   keys IGNORED and the attempt recorded; the model's numbers never become
   a calibrated probability;
3. honest failure states: not configured, all-omitted, malformed nested
   types, NaN/Infinity, out-of-range probabilities and one candidate's
   failure each produce a distinguishable result, and a failed call keeps
   the usage it consumed;
4. the comparison is bounded and explainable: only quantified, quality-safe
   savings are ranked; unquantified and incomparable candidates are
   reported; ``defer`` is a first-class result; a per-task saving is never
   extrapolated without a declared task count;
5. only an EXPLICIT acceptance runs the operation, and it runs on the
   prediction's OWN scope; a rejection touches no knowledge;
6. facts are not effects: binding the real maintenance fact never sets
   ``effect_verified``, and only a real later-task observation under a
   comparable setup does;
7. the capability predictor can never prove its own W_OR improvement, and
   repeated evaluation never doubles a sample.
"""
import io
import json
import os
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tests.harness.helpers import HarnessTestCase  # noqa: E402

from or_harness.api import ORHarness  # noqa: E402
from or_harness.core.storage import StorageError  # noqa: E402
from or_harness.strategy.embedding_index import (  # noqa: E402
    LocalHashEmbeddingBackend,
)
from or_harness.world_model.capability_evolution import (  # noqa: E402
    CAPABILITY_EVOLUTION_PROTOCOL_VERSION,
    CAPABILITY_EVOLUTION_SYSTEM_PROMPT,
)
from or_harness.world_model.contracts import (  # noqa: E402
    BaselineStatement,
    LearningOperation,
)
from or_harness.world_model.provider import (  # noqa: E402
    HttpChatProvider,
    WorldModelProvider,
)

EMBEDDING_ENV_KEYS = ("OR_EMBEDDING_BACKEND", "OR_EMBEDDING_BASE_URL",
                      "OR_EMBEDDING_MODEL", "OR_EMBEDDING_API_KEY")

REQ_TEXT = ("A distribution centre must be loaded before the delivery window "
            "opens; demand 100 units may not be deferred.")


def _task(task_id="t1", **coupling):
    # The defaults mirror helpers.make_profile, which is what the seeded
    # evidence records are built with. A later task whose structure landed
    # in a DIFFERENT cell than the evidence would be outside the claim's
    # frozen target, and the evaluation would (correctly) refuse to count
    # it — so the fixture keeps them in one cell on purpose.
    values = {"resource_coupling": 0.9, "temporal_coupling": 0.1,
              "route_complexity": 0.85}
    values.update(coupling)
    return {"task_id": task_id, "family": "routing",
            "description": REQ_TEXT,
            "spec": {"n_vars": 100, "n_constraints": 50, "n_int_vars": 100},
            "annotations": {"coupling": {**values, "semantic_coupling": 0.8}}}


#: A well-formed capability payload: one quantified, quality-safe saving.
#: The learning cost is expressed in the SAVING's own unit (seconds), since
#: a comparison only nets quantities in the same currency — and it is kept
#: below the saving so the fixture is net-positive.
GOOD_PAYLOAD = {
    "expected_changes": [
        {"metric": "resource_cost", "unit": "s", "direction": "decrease",
         "value": -1.5, "beneficial_direction": "decrease",
         "value_kind": "absolute"},
        {"metric": "normalized_solution_quality", "unit": "1-gap",
         "direction": "unchanged", "beneficial_direction": "increase"},
    ],
    "learning_cost": {"solver_runtime_s": 0.2},
    "degradation_risk": {"events": [
        {"event": "overgeneralized_entry", "probability": 0.2},
        {"event": "revised_performance_regression"},
    ]},
    "uncertainty": {"knowledge_gap": 0.5, "execution_randomness": 0.2},
    "verification_conditions": [
        {"condition": "later matching tasks need less solver time",
         "check_basis": "solver_runtime_s on the next 5 unseen tasks",
         "evaluable": True},
    ],
    "evidence_basis": ["capability_evidence.sources.m",
                       "learning_material.executions[0]"],
}


def _bundle(bundle_id="cb_1", kind="new_claim", strategy_id="S04",
            execution_ids=("ex1", "ex2"), tasks=("t1", "t2")):
    # The cell the fixture's OWN profiles quantize to (helpers.make_profile
    # defaults: rc=0.9, tc=0.1, rx=0.85). A cell token that does not
    # correspond to the executions behind the bundle would declare a target
    # the evidence never came from — and the evaluation would (correctly)
    # refuse to count those tasks as matching.
    cell = "rc[0.75,1.00]|tc[0.00,0.25]|rx[0.75,1.00]"
    return {
        "bundle_id": bundle_id,
        "kind": kind,
        "strategy_id": strategy_id,
        "family": "routing",
        "cell_token": cell,
        "group_key": f"family=routing|{cell}",
        "execution_ids": list(execution_ids),
        "tasks": list(tasks),
        "n_supporting": len(execution_ids),
        "trigger_reasons": ["sufficient independent evidence"],
        "mean_quality": 0.72,
        "mean_cost": {"solver_runtime_s": 5.0},
        "cost_measured": ["solver_runtime_s"],
        "failure_rate": 0.0,
        "target_entry_id": None,
        "entry_before": None,
        "created_at": 0.0,
    }


class StubProvider(WorldModelProvider):
    """A stub provider returning a fixed payload, recording requests."""

    name = "stub-m5"

    def __init__(self, payload=None, usage=None):
        self.payload = GOOD_PAYLOAD if payload is None else payload
        self.usage = usage if usage is not None else {
            "prompt_tokens": 100, "completion_tokens": 50}
        self.requests = []

    def predict(self, request, timeout_s=None):
        self.requests.append(request)
        return {"payload": self.payload, "usage": self.usage,
                "error": None, "latency_s": 0.02}


class M5Case(HarnessTestCase):
    """A harness with an injected local embedding backend (hermetic)."""

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

    def predict(self, *, payload=None, bundle=None, horizon="next 10 tasks",
                horizon_tasks=10, operation=None):
        provider = StubProvider(payload=payload) if payload is not None \
            else self.provider
        h = self.h if provider is self.provider else ORHarness(
            home=self.home, world_model=provider, embedding=self.backend)
        if h is not self.h:
            self.addCleanup(h.close)
        prediction = h.predict_capability_evolution(
            operation or {"operation_type": "induce", "strategy_id": "S04"},
            bundle=bundle if bundle is not None else _bundle(),
            horizon=horizon, horizon_tasks=horizon_tasks)
        return h, prediction
    def seed_executions(self, strategy="S04", tasks=("t1", "t2")):
        """Real executed records matching the bundle's scope."""
        for index, task_id in enumerate(tasks):
            self.h.bank.append(self.make_record(
                execution_id=f"ex{index + 1}", task_id=task_id,
                strategy_id=strategy, gap=0.05,
                profile=self.make_profile(problem_id=task_id)))


# ---------------------------------------------------------------------------
# 1. the protocol really reaches the provider
# ---------------------------------------------------------------------------


class TestProtocolReachesProvider(M5Case):

    def test_request_names_the_capability_protocol(self):
        _, prediction = self.predict()
        request = self.provider.requests[0]
        self.assertEqual(request.get("prediction_protocol"),
                         CAPABILITY_EVOLUTION_PROTOCOL_VERSION)
        self.assertEqual(request["request_kind"],
                         "capability_evolution_prediction")
        self.assertEqual(prediction.status, "valid")

    def test_provider_receives_real_evidence_content_not_ids(self):
        self.seed_executions()
        _, prediction = self.predict()
        request = self.provider.requests[0]
        material = request["learning_material"]
        self.assertTrue(material["executions"],
                        "the real execution content must travel")
        self.assertTrue(material["executions"][0]["available"])
        self.assertEqual(material["executions"][0]["task_id"], "t1")
        self.assertIn("cost", material["executions"][0])
        # The capability evidence travels with its per-source detail.
        self.assertIn("sources", request["capability_evidence"])

    def test_framework_fixes_identity_in_the_request(self):
        _, prediction = self.predict()
        request = self.provider.requests[0]
        self.assertEqual(request["candidate_operation"]["operation_type"],
                         "induce")
        self.assertEqual(request["experience_scope"]["execution_ids"],
                         ["ex1", "ex2"])
        self.assertEqual(request["task_targeting"]["family"], "routing")
        self.assertEqual(request["baseline"]["kind"], "conditional_stats")
        self.assertEqual(request["horizon"], "next 10 tasks")
        self.assertIn("forbidden_fields", request["output_contract"])

    def test_prediction_survives_save_and_restore(self):
        _, prediction = self.predict()
        restored = self.h.get_capability_evolution_prediction(
            prediction.prediction_id)
        self.assertIsNotNone(restored)
        self.assertEqual(restored.to_dict(), prediction.to_dict())
        self.assertEqual(restored.status, "valid")
        self.assertEqual(restored.horizon, prediction.horizon)
        self.assertEqual(restored.horizon_tasks, prediction.horizon_tasks)
        self.assertEqual(
            [c.to_dict() for c in restored.expected_changes],
            [c.to_dict() for c in prediction.expected_changes])
        self.assertEqual(restored.experience_scope.to_dict(),
                         prediction.experience_scope.to_dict())
        self.assertEqual(restored.current_evidence.to_dict(),
                         prediction.current_evidence.to_dict())

    def test_capability_records_are_not_readable_as_or_predictions(self):
        """The two generations live in separate tables: a capability record
        must never be deserialized as a strategy-outcome prediction."""
        _, prediction = self.predict()
        self.assertEqual(self.h.store.count_contract_predictions(), 0,
                         "no capability record may land in the "
                         "strategy-outcome table")
        self.assertEqual(self.h.store.count_capability_predictions(), 1)
        self.assertEqual(self.h.strategy_outcome_predictions(), [])
        self.assertEqual(
            len(self.h.capability_evolution_predictions()), 1)

    def test_local_http_stub_gets_the_capability_prompt(self):
        """A REAL HTTP round trip: the stub server confirms the capability
        protocol selected the capability prompt and schema."""
        captured = {}

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                captured["system"] = body["messages"][0]["content"]
                captured["request"] = json.loads(
                    body["messages"][1]["content"])
                reply = json.dumps({
                    "choices": [{"message": {"content": json.dumps(
                        GOOD_PAYLOAD)}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 20},
                }).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(reply)))
                self.end_headers()
                self.wfile.write(reply)

            def log_message(self, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.shutdown)

        provider = HttpChatProvider(
            base_url=f"http://127.0.0.1:{server.server_port}",
            model="stub-model", api_key="test-key")
        h = ORHarness(home=self.home, world_model=provider,
                      embedding=self.backend)
        self.addCleanup(h.close)
        prediction = h.predict_capability_evolution(
            {"operation_type": "induce", "strategy_id": "S04"},
            bundle=_bundle(), horizon="next 10 tasks", horizon_tasks=10)
        # The capability prompt really was selected on the wire.
        self.assertIn("wm-ce/1", captured["system"])
        self.assertEqual(captured["system"],
                         CAPABILITY_EVOLUTION_SYSTEM_PROMPT)
        self.assertEqual(
            captured["request"]["prediction_protocol"],
            CAPABILITY_EVOLUTION_PROTOCOL_VERSION)
        self.assertEqual(prediction.status, "valid")
        self.assertEqual(len(prediction.expected_changes), 2)

    def test_unconfigured_provider_is_contract_only(self):
        h = ORHarness(home=self.home, embedding=self.backend)
        self.addCleanup(h.close)
        prediction = h.predict_capability_evolution(
            {"operation_type": "induce", "strategy_id": "S04"},
            bundle=_bundle(), horizon="next 10 tasks")
        self.assertEqual(prediction.status, "contract_only")
        self.assertFalse(prediction.service_available)
        self.assertFalse(prediction.prediction_made)


# ---------------------------------------------------------------------------
# 2. the framework's identity is not negotiable
# ---------------------------------------------------------------------------


class TestIdentityIsFixed(M5Case):

    def test_payload_cannot_restate_the_operation(self):
        payload = json.loads(json.dumps(GOOD_PAYLOAD))
        payload["candidate_operation"] = {
            "operation_type": "retire", "strategy_id": "S99"}
        payload["experience_scope"] = {"execution_ids": ["hacked"]}
        payload["baseline"] = {"kind": "declared", "value": 0.99}
        payload["horizon"] = "forever"
        _, prediction = self.predict(payload=payload)
        self.assertEqual(prediction.candidate_operation.operation_type,
                         "induce")
        self.assertEqual(prediction.candidate_operation.strategy_id, "S04")
        self.assertEqual(prediction.experience_scope.execution_ids,
                         ["ex1", "ex2"])
        self.assertEqual(prediction.horizon, "next 10 tasks")
        self.assertNotEqual(prediction.baseline.value, 0.99)
        overrides = prediction.trace.model_info["attempted_field_overrides"]
        for key in ("candidate_operation", "experience_scope", "baseline",
                    "horizon"):
            self.assertIn(key, overrides)
        self.assertTrue(any("IGNORED" in n for n in prediction.notes))

    def test_payload_cannot_set_effect_verified(self):
        payload = json.loads(json.dumps(GOOD_PAYLOAD))
        payload["status"] = "valid"
        payload["effect_verified"] = True
        payload["verification_conditions"] = [
            {"condition": "already verified", "effect_verified": True,
             "fact_bound": True, "evaluable": True}]
        _, prediction = self.predict(payload=payload)
        condition = prediction.verification_conditions[0]
        self.assertFalse(condition.effect_verified,
                         "the model may never assert a verified effect")
        self.assertFalse(condition.fact_bound)
        self.assertTrue(condition.prediction_made)
        self.assertIn("status", prediction.trace.model_info[
            "attempted_field_overrides"])

    def test_self_reported_uncertainty_is_not_calibrated(self):
        _, prediction = self.predict()
        self.assertEqual(prediction.uncertainty.source, "model_self_report")
        self.assertIsNone(prediction.uncertainty.knowledge_gap,
                          "a self-reported number is never stored as a "
                          "measured probability")
        joined = " ".join(prediction.uncertainty.notes)
        self.assertIn("UNCALIBRATED", joined)
        self.assertIn("0.5", joined)

    def test_omitted_beneficial_direction_is_not_a_free_benefit(self):
        """A signed change with no stated improvement direction must not be
        read as good news — otherwise the operation gets a free benefit."""
        payload = {"expected_changes": [
            {"metric": "resource_cost", "direction": "decrease",
             "value": -2.0}]}
        _, prediction = self.predict(payload=payload)
        change = prediction.expected_changes[0]
        self.assertIsNone(change.is_improvement)
        self.assertEqual(change.beneficial_direction, "either")
        self.assertTrue(any("beneficial_direction" in k for k in
                            prediction.trace.unsupported_fields))


# ---------------------------------------------------------------------------
# 3. honest failure states
# ---------------------------------------------------------------------------


class TestHonestFailures(M5Case):

    def test_all_omitted_is_a_recorded_non_prediction(self):
        _, prediction = self.predict(payload={})
        self.assertEqual(prediction.status, "contract_only")
        self.assertFalse(prediction.prediction_made)
        self.assertTrue(any("no usable expected_change" in n
                            for n in prediction.notes))

    def test_cost_only_is_still_not_a_performance_claim(self):
        payload = {"learning_cost": {"llm_tokens": 100}}
        _, prediction = self.predict(payload=payload)
        self.assertEqual(prediction.status, "contract_only")
        self.assertIsNotNone(prediction.learning_cost)
        self.assertFalse(prediction.prediction_made)

    def test_malformed_nested_types_are_explicit_problems(self):
        payload = {
            "expected_changes": [
                {"metric": "resource_cost", "direction": "sideways"},
                {"direction": "decrease", "value": -1.0},
                "not-an-object",
            ],
            "degradation_risk": {"events": [
                {"event": "x", "probability": 1.5},
            ]},
            "learning_cost": {"not_a_dimension": 5},
        }
        _, prediction = self.predict(payload=payload)
        self.assertEqual(prediction.status, "invalid")
        joined = " ".join(prediction.notes)
        self.assertIn("direction must be", joined)
        self.assertIn("metric is required", joined)
        self.assertIn("must be a JSON object", joined)
        self.assertIn("probability must be in [0, 1]", joined)
        self.assertIn("unknown learning_cost dimension", joined)

    def test_nan_and_infinity_are_rejected(self):
        payload = {"expected_changes": [
            {"metric": "resource_cost", "direction": "decrease",
             "value": float("inf"), "beneficial_direction": "decrease"}]}
        _, prediction = self.predict(payload=payload)
        self.assertEqual(prediction.status, "invalid")
        self.assertTrue(any("finite" in n for n in prediction.notes))

    def test_relative_value_must_be_a_ratio(self):
        payload = {"expected_changes": [
            {"metric": "resource_cost", "direction": "decrease",
             "value": -20.0, "value_kind": "relative",
             "beneficial_direction": "decrease"}]}
        _, prediction = self.predict(payload=payload)
        self.assertEqual(prediction.status, "invalid")
        self.assertTrue(any("RATIO" in n for n in prediction.notes))

    def test_value_without_a_baseline_is_not_falsifiable(self):
        payload = {"expected_changes": [
            {"metric": "resource_cost", "direction": "decrease",
             "value": -1.0, "beneficial_direction": "decrease"}]}
        h = ORHarness(home=self.home, world_model=StubProvider(payload),
                      embedding=self.backend)
        self.addCleanup(h.close)
        # No bundle -> no frozen baseline at all.
        prediction = h.predict_capability_evolution(
            {"operation_type": "induce", "strategy_id": "S04"},
            horizon="next 10 tasks")
        self.assertEqual(prediction.status, "invalid")
        self.assertTrue(any("baseline" in n for n in prediction.notes))

    def test_provider_error_keeps_its_consumed_usage(self):
        class Boom(WorldModelProvider):
            name = "boom"

            def predict(self, request, timeout_s=None):
                raise RuntimeError("model is down")

        h = ORHarness(home=self.home, world_model=Boom(),
                      embedding=self.backend)
        self.addCleanup(h.close)
        prediction = h.predict_capability_evolution(
            {"operation_type": "induce", "strategy_id": "S04"},
            bundle=_bundle(), horizon="next 10 tasks")
        self.assertEqual(prediction.status, "provider_error")
        self.assertFalse(prediction.prediction_made)
        self.assertIn("RuntimeError", prediction.trace.model_info["error"])

    def test_failure_is_persisted_with_its_call_cost(self):
        payload = {"expected_changes": [{"metric": "", "direction": "x"}]}
        _, prediction = self.predict(payload=payload)
        self.assertEqual(prediction.status, "invalid")
        stored = self.h.get_capability_evolution_prediction(
            prediction.prediction_id)
        self.assertIsNotNone(stored, "a failed call is still a record")
        self.assertIsNotNone(stored.trace.call_cost)
        self.assertEqual(stored.trace.call_cost.llm_tokens, 50.0)


# ---------------------------------------------------------------------------
# 4. the comparison is bounded and explainable
# ---------------------------------------------------------------------------


class TestComparison(M5Case):

    def _saving(self, value, *, metric="resource_cost", unit="s",
                beneficial="decrease", direction="decrease", **extra):
        change = {"metric": metric, "unit": unit, "direction": direction,
                  "value": value, "beneficial_direction": beneficial}
        change.update(extra)
        # The learning cost is in the SAVING's own unit: a comparison only
        # nets quantities expressed in the same currency, so a fixture
        # that mixes them would (correctly) be reported as incomparable.
        # It is kept small so the fixture is net-POSITIVE: an operation
        # that costs as much as it saves is not a recommendation.
        return {"expected_changes": [change],
                "learning_cost": {"solver_runtime_s": 0.1},
                "verification_conditions": [
                    {"condition": "later matching tasks need less solver "
                                  "time", "evaluable": True}]}

    def test_largest_quantified_saving_is_recommended(self):
        small = self.predict(payload=self._saving(-1.0))[1]
        large = self.predict(payload=self._saving(-4.0))[1]
        result = self.h.compare_capability_evolution(
            [small.prediction_id, large.prediction_id], horizon_tasks=10)
        self.assertEqual(result["recommendation"], "accept")
        self.assertEqual(result["selected_prediction_id"],
                         large.prediction_id)
        self.assertEqual(result["rule"]["no_universal_score"], True)
        entry = [c for c in result["comparisons"]
                 if c["prediction_id"] == large.prediction_id][0]
        self.assertEqual(entry["saving"]["per_task"], 4.0)
        self.assertEqual(entry["saving"]["total"], 40.0)

    def test_quality_degradation_is_never_auto_ranked(self):
        payload = {"expected_changes": [
            {"metric": "resource_cost", "unit": "s", "direction": "decrease",
             "value": -9.0, "beneficial_direction": "decrease"},
            {"metric": "normalized_solution_quality", "unit": "1-gap",
             "direction": "decrease", "beneficial_direction": "increase"},
        ], "verification_conditions": [{"condition": "quality holds",
                                        "evaluable": True}]}
        risky = self.predict(payload=payload)[1]
        safe = self.predict(payload=self._saving(-1.0))[1]
        result = self.h.compare_capability_evolution(
            [risky.prediction_id, safe.prediction_id], horizon_tasks=10)
        self.assertEqual(result["recommendation"], "accept")
        self.assertEqual(result["selected_prediction_id"],
                         safe.prediction_id,
                         "a bigger saving that trades quality away must not "
                         "win by default")
        excluded = [c for c in result["incomparable"]
                    if c["prediction_id"] == risky.prediction_id]
        self.assertTrue(excluded)
        self.assertIn("QUALITY degradation", excluded[0]["reason"])

    def test_unquantified_direction_is_not_ranked(self):
        vague = self.predict(payload={
            "expected_changes": [{"metric": "resource_cost",
                                  "direction": "decrease",
                                  "beneficial_direction": "decrease"}],
            "verification_conditions": [{"condition": "cost falls",
                                          "evaluable": True}]})[1]
        result = self.h.compare_capability_evolution(
            [vague.prediction_id], horizon_tasks=10)
        self.assertEqual(result["recommendation"], "defer")
        self.assertIn("unquantified", result["incomparable"][0]["reason"])
        self.assertIn("cannot be ranked",
                      result["incomparable"][0]["reason"])

    def test_no_comparable_candidate_defers(self):
        quality_only = self.predict(payload={
            "expected_changes": [{"metric": "normalized_solution_quality",
                                  "unit": "1-gap", "direction": "increase",
                                  "value": 0.05,
                                  "beneficial_direction": "increase"}],
            "verification_conditions": [{"condition": "quality rises",
                                          "evaluable": True}]})[1]
        result = self.h.compare_capability_evolution(
            [quality_only.prediction_id], horizon_tasks=10)
        self.assertEqual(result["recommendation"], "defer")
        self.assertIn("no candidate predicts a cumulative",
                      result["basis"])
        self.assertTrue(any("legitimate decision outcome" in n
                            for n in result["notes"]))

    def test_invalid_predictions_are_incomparable(self):
        bad = self.predict(payload={})[1]
        result = self.h.compare_capability_evolution(
            [bad.prediction_id], horizon_tasks=10)
        self.assertEqual(result["recommendation"], "defer")
        self.assertIn("status 'contract_only'",
                      result["incomparable"][0]["reason"])

    def test_per_task_saving_is_never_extrapolated(self):
        prediction = self.predict(payload=self._saving(-3.0),
                                  horizon_tasks=None)[1]
        result = self.h.compare_capability_evolution(
            [prediction.prediction_id], horizon_tasks=None)
        # An undeclared window cannot amortize a ONE-TIME maintenance cost,
        # so the candidate is reported rather than ranked. The per-task
        # figure is still never extrapolated into a total.
        entry = result["incomparable"][0]
        self.assertEqual(entry["saving"]["per_task"], 3.0)
        self.assertIsNone(entry["saving"]["total"])
        self.assertIn("never extrapolated", entry["saving"]["note"])
        self.assertIn("no observation window was declared",
                      entry["reason"])
        self.assertIsNone(entry["net_saving"]["net_over_horizon"])

    def test_a_declared_window_amortizes_the_one_time_cost(self):
        """The maintenance cost is paid ONCE while the saving accrues per
        task, so the decision uses the CUMULATIVE saving over the window."""
        prediction = self.predict(payload=self._saving(-2.0),
                                  horizon_tasks=10)[1]
        result = self.h.compare_capability_evolution(
            [prediction.prediction_id], horizon_tasks=10)
        entry = result["comparisons"][0]
        net = entry["net_saving"]
        self.assertEqual(net["cumulative_saving"], 20.0)
        self.assertEqual(net["maintenance_cost"], 0.1)
        self.assertAlmostEqual(net["net_over_horizon"], 19.9, places=5)
        self.assertEqual(result["recommendation"], "accept")

    def test_relative_saving_converts_through_the_frozen_cost_baseline(self):
        """A relative saving becomes a unit count ONLY through a baseline the
        FRAMEWORK froze for that metric. The bundle's frozen mean
        solver_runtime_s (5.0 s) is that reference: 0.2 x 5.0 = 1.0 s."""
        payload = {"expected_changes": [
            {"metric": "resource_cost", "unit": "s", "direction": "decrease",
             "value": -0.2, "value_kind": "relative",
             "beneficial_direction": "decrease"}],
            "learning_cost": {"solver_runtime_s": 0.1},
            "verification_conditions": [{"condition": "cost falls",
                                          "evaluable": True}]}
        prediction = self.predict(payload=payload)[1]
        result = self.h.compare_capability_evolution(
            [prediction.prediction_id], horizon_tasks=10)
        self.assertEqual(result["recommendation"], "accept")
        saving = result["comparisons"][0]["saving"]
        self.assertEqual(saving["kind"], "quantified")
        self.assertEqual(saving["value_kind"], "relative")
        self.assertAlmostEqual(saving["per_task"], 1.0, places=5)

    def test_relative_saving_without_a_frozen_baseline_stays_a_ratio(self):
        """With no frozen reference for its metric, a ratio is reported and
        never converted: a percentage of an unknown quantity is unknown."""
        payload = {"expected_changes": [
            {"metric": "resource_cost", "unit": "s", "direction": "decrease",
             "value": -0.2, "value_kind": "relative",
             "beneficial_direction": "decrease"}],
            "learning_cost": {"solver_runtime_s": 0.1},
            "verification_conditions": [{"condition": "cost falls",
                                          "evaluable": True}]}
        prediction = self.predict(payload=payload, bundle={
            **_bundle(), "mean_cost": {}})[1]
        result = self.h.compare_capability_evolution(
            [prediction.prediction_id], horizon_tasks=10)
        self.assertEqual(result["recommendation"], "defer")
        self.assertIn("ratio", result["incomparable"][0]["reason"])
        self.assertIn("absolute", result["incomparable"][0]["reason"])
        saving = result["incomparable"][0]["saving"]
        self.assertEqual(saving["kind"], "relative")
        self.assertIn("per_task_ratio", saving)
        self.assertIn("never converted", saving["note"])

    def test_comparison_reports_its_own_call_cost_separately(self):
        prediction = self.predict()[1]
        result = self.h.compare_capability_evolution(
            [prediction.prediction_id], horizon_tasks=10)
        cost = result["comparison_cost"]
        self.assertEqual(cost["per_dim"]["llm_tokens"], 50.0)
        self.assertIn("separate from the predicted learning cost",
                      cost["note"])

    def test_different_evidence_versions_are_reported(self):
        first = self.predict()[1]
        # Change the evidence (a new execution) before the second call.
        self.h.bank.append(self.make_record(
            execution_id="later", task_id="t9", strategy_id="S04", gap=0.05,
            profile=self.make_profile(problem_id="t9")))
        second = self.predict()[1]
        result = self.h.compare_capability_evolution(
            [first.prediction_id, second.prediction_id], horizon_tasks=10)
        self.assertTrue(any("DIFFERENT" in n for n in result["notes"]))

    def test_unknown_prediction_id_is_refused(self):
        with self.assertRaises(StorageError):
            self.h.compare_capability_evolution(["hp_nope"])


# ---------------------------------------------------------------------------
# 5. only an explicit acceptance runs the operation
# ---------------------------------------------------------------------------


class TestExplicitAcceptance(M5Case):

    def _recommendation(self):
        prediction = self.predict()[1]
        return prediction, self.h.compare_capability_evolution(
            [prediction.prediction_id], horizon_tasks=10)

    def test_defer_cannot_be_accepted(self):
        quality_only = self.predict(payload={
            "expected_changes": [{"metric": "normalized_solution_quality",
                                  "direction": "increase", "value": 0.1,
                                  "beneficial_direction": "increase"}],
            "verification_conditions": [{"condition": "quality rises",
                                          "evaluable": True}]})[1]
        result = self.h.compare_capability_evolution(
            [quality_only.prediction_id], horizon_tasks=10)
        with self.assertRaises(ValueError) as ctx:
            self.h.accept_capability_operation(result)
        self.assertIn("only 'accept' names an operation",
                      str(ctx.exception))
        self.assertEqual(self.h.sbank.count(), 0)

    def test_rejection_touches_no_knowledge(self):
        prediction, recommendation = self._recommendation()
        before = self.h.sbank.count()
        result = self.h.reject_capability_operation(
            recommendation, reason="budget is tight this week")
        self.assertFalse(result["accepted"])
        self.assertFalse(result["strategic_bank_touched"])
        self.assertEqual(self.h.sbank.count(), before,
                         "declining must not modify knowledge")
        self.assertIsNone(self.h.capability_maintenance_binding(
            prediction.prediction_id))

    def test_comparison_alone_changes_nothing(self):
        before = self.h.sbank.count()
        prediction, recommendation = self._recommendation()
        self.assertEqual(recommendation["recommendation"], "accept")
        self.assertEqual(self.h.sbank.count(), before,
                         "a recommendation is not an execution")

    def test_acceptance_runs_on_the_prediction_own_scope(self):
        self.seed_executions()
        prediction, recommendation = self._recommendation()
        # A LATER execution outside the predicted scope exists: it must NOT
        # be silently consolidated.
        self.h.bank.append(self.make_record(
            execution_id="ex_late", task_id="t7", strategy_id="S04", gap=0.05,
            profile=self.make_profile(problem_id="t7")))
        result = self.h.accept_capability_operation(recommendation)
        self.assertEqual(sorted(result["execution_ids"]),
                         ["ex1", "ex2"])
        adoption = self.h.actions.get(result["adoption_action_id"])
        self.assertEqual(adoption.params["capability_prediction_id"],
                         prediction.prediction_id)

    def test_acceptance_without_a_scope_is_refused(self):
        h = ORHarness(home=self.home, world_model=self.provider,
                      embedding=self.backend)
        self.addCleanup(h.close)
        prediction = h.predict_capability_evolution(
            {"operation_type": "induce", "strategy_id": "S04"},
            horizon="next 10 tasks", horizon_tasks=10)
        recommendation = h.compare_capability_evolution(
            [prediction.prediction_id], horizon_tasks=10)
        # No comparable candidate -> defer; force an accept-shaped dict.
        with self.assertRaises(ValueError) as ctx:
            h.accept_capability_operation(
                {"recommendation": "accept",
                 "selected_prediction_id": prediction.prediction_id})
        self.assertIn("no experience scope", str(ctx.exception))

    def test_prediction_changes_the_recommendation(self):
        """The M5 requirement: under a FIXED decision rule, a different
        capability prediction really does change the advice."""
        small = self.predict(payload={"expected_changes": [
            {"metric": "resource_cost", "unit": "s", "direction": "decrease",
             "value": -0.5, "beneficial_direction": "decrease"}],
            "learning_cost": {"solver_runtime_s": 0.1},
            "verification_conditions": [{"condition": "cost falls",
                                          "evaluable": True}]})[1]
        large = self.predict(payload={"expected_changes": [
            {"metric": "resource_cost", "unit": "s", "direction": "decrease",
             "value": -8.0, "beneficial_direction": "decrease"}],
            "learning_cost": {"solver_runtime_s": 0.1},
            "verification_conditions": [{"condition": "cost falls",
                                          "evaluable": True}]})[1]
        first = self.h.compare_capability_evolution(
            [small.prediction_id], horizon_tasks=10)
        second = self.h.compare_capability_evolution(
            [large.prediction_id], horizon_tasks=10)
        self.assertEqual(first["recommendation"], "accept")
        self.assertEqual(second["recommendation"], "accept")
        # Same rule, different predicted magnitudes -> different entries.
        self.assertEqual(
            first["comparisons"][0]["saving"]["per_task"], 0.5)
        self.assertEqual(
            second["comparisons"][0]["saving"]["per_task"], 8.0)


# ---------------------------------------------------------------------------
# 6. facts are not effects
# ---------------------------------------------------------------------------


class TestTwoStageFeedback(M5Case):

    def _accepted(self, payload=None, *, prediction_id=None,
                  horizon_tasks=1):
        """Accept a candidate and run the real operation.

        When the automatic rule defers (a quality-only candidate has no
        shared yardstick with a cost saving), the outer agent may still
        choose it explicitly — that is exactly what the incomparable list
        is for. The recommendation dict models that explicit choice.

        ``horizon_tasks`` defaults to 1 so a test that closes ONE later
        episode can actually reach a verdict: a horizon the sample does not
        meet stays pending, which is the honest answer but not what an
        effect test is checking.
        """
        self.seed_executions()
        prediction = self.predict(payload=payload,
                                  horizon_tasks=horizon_tasks)[1]
        recommendation = self.h.compare_capability_evolution(
            [prediction.prediction_id], horizon_tasks=horizon_tasks)
        if recommendation["recommendation"] != "accept":
            recommendation = {
                "recommendation": "accept",
                "recommendation_id": recommendation["recommendation_id"],
                "selected_prediction_id": prediction.prediction_id,
                "basis": ("explicit agent choice among the incomparable "
                          "candidates"),
            }
        accepted = self.h.accept_capability_operation(
            recommendation, prediction_id=prediction_id)
        return prediction, accepted

    def test_binding_the_fact_never_verifies_the_effect(self):
        prediction, accepted = self._accepted()
        # ACCEPTING the recommendation already bound the real maintenance
        # fact automatically — the adoption action named the prediction, so
        # no separate call is required and nothing can fall through.
        bound = self.h.bind_capability_maintenance(
            prediction.prediction_id,
            adoption_action_id=accepted["adoption_action_id"])
        self.assertTrue(bound["already_bound"],
                        "accept-capability binds the fact itself")
        binding = bound["binding"]
        self.assertEqual(binding["stage"], "maintenance_fact")
        self.assertTrue(any("never sets effect_verified" in n
                            for n in binding["notes"]))
        evaluation = self.h.capability_effect_evaluation(
            prediction.prediction_id)
        self.assertIsNone(evaluation,
                          "binding a fact must not produce an effect verdict")

    def test_verified_knowledge_is_not_a_capability_improvement(self):
        prediction, accepted = self._accepted()
        bound = self.h.bind_capability_maintenance(
            prediction.prediction_id,
            adoption_action_id=accepted["adoption_action_id"])
        self.assertIn("VERIFIED ENTRY", bound["binding"]["verification"]["note"])

    def test_unadopted_prediction_has_no_fact_to_bind(self):
        prediction = self.predict()[1]
        result = self.h.bind_capability_maintenance(prediction.prediction_id)
        self.assertEqual(result["state"], "not_adopted")
        self.assertIsNone(result["binding"]["adoption_action_id"])
        self.assertIsNone(self.h.capability_maintenance_binding(
            prediction.prediction_id))

    def test_binding_is_idempotent(self):
        prediction, accepted = self._accepted()
        first = self.h.bind_capability_maintenance(
            prediction.prediction_id,
            adoption_action_id=accepted["adoption_action_id"])
        second = self.h.bind_capability_maintenance(prediction.prediction_id)
        self.assertTrue(second["already_bound"])
        self.assertEqual(first["binding"], second["binding"])

    def test_unmet_horizon_stays_pending_and_re_evaluable(self):
        prediction, accepted = self._accepted()
        self.h.bind_capability_maintenance(
            prediction.prediction_id,
            adoption_action_id=accepted["adoption_action_id"])
        result = self.h.evaluate_capability_effect(prediction.prediction_id)
        self.assertEqual(result["evaluation"]["state"], "pending")
        self.assertFalse(result["evaluation"]["effect_verified"])
        self.assertIn("re-evaluable", " ".join(
            result["evaluation"]["exclusion_reasons"]))
        # A pending evaluation is re-evaluated (never frozen).
        again = self.h.evaluate_capability_effect(prediction.prediction_id)
        self.assertFalse(again["already_evaluated"])

    def test_effect_without_a_paired_reference_is_inconclusive(self):
        """A before/after movement with no comparable control describes what
        happened, not what caused it."""
        prediction, accepted = self._accepted()
        self.h.bind_capability_maintenance(
            prediction.prediction_id,
            adoption_action_id=accepted["adoption_action_id"])
        # A LATER task with a real closed-episode evaluation.
        self._close_later_episode("later1", observed_quality=0.95)
        result = self.h.evaluate_capability_effect(prediction.prediction_id)
        evaluation = result["evaluation"]
        self.assertEqual(evaluation["state"], "inconclusive")
        self.assertFalse(evaluation["effect_verified"])
        self.assertIn("DESCRIPTIVE", " ".join(evaluation["exclusion_reasons"]))

    def test_inconclusive_is_not_frozen_and_a_later_reference_upgrades_it(self):
        """An inconclusive first look is RE-EVALUABLE: once the caller
        records the paired reference it needed, the same prediction reaches
        a real verdict. Freezing the first ``inconclusive`` would make a
        verdict unreachable forever."""
        prediction, accepted = self._accepted(payload={
            "expected_changes": [
                {"metric": "normalized_solution_quality", "unit": "1-gap",
                 "direction": "increase", "value": 0.2,
                 "beneficial_direction": "increase",
                 "baseline": {"kind": "conditional_stats", "value": 0.70}}],
            "learning_cost": {"llm_tokens": 100},
            "verification_conditions": [{"condition": "quality rises",
                                         "evaluable": True}]})
        self.h.bind_capability_maintenance(
            prediction.prediction_id,
            adoption_action_id=accepted["adoption_action_id"])
        self._close_later_episode("later1", observed_quality=0.95)
        first = self.h.evaluate_capability_effect(prediction.prediction_id)
        self.assertEqual(first["evaluation"]["state"], "inconclusive")
        self.assertFalse(first["already_evaluated"])
        # The missing reference is supplied; the SAME prediction must now
        # be re-evaluated rather than short-circuited.
        self.h.record_capability_paired_evaluation(
            prediction.prediction_id, metric="normalized_solution_quality",
            reference_value=0.70, treated_value=0.95, unit="1-gap",
            reference_task_ids=["t1", "t2"])
        second = self.h.evaluate_capability_effect(prediction.prediction_id)
        self.assertFalse(second["already_evaluated"],
                         "an inconclusive evaluation must not freeze the "
                         "prediction")
        self.assertEqual(second["evaluation"]["state"],
                         "observed_improvement")
        self.assertTrue(second["evaluation"]["effect_verified"])
        # Once the verdict is FINAL, a further look returns it unchanged.
        third = self.h.evaluate_capability_effect(prediction.prediction_id)
        self.assertTrue(third["already_evaluated"])
        self.assertEqual(third["evaluation"], second["evaluation"])

    def test_paired_reference_makes_the_change_attributable(self):
        prediction, accepted = self._accepted()
        self.h.bind_capability_maintenance(
            prediction.prediction_id,
            adoption_action_id=accepted["adoption_action_id"])
        self._close_later_episode("later1", observed_quality=0.95)
        paired = self.h.record_capability_paired_evaluation(
            prediction.prediction_id, metric="normalized_solution_quality",
            reference_value=0.70, treated_value=0.95, unit="1-gap",
            reference_task_ids=["t1", "t2"])
        self.assertEqual(paired["change"], 0.25)
        self.assertIn("paired", paired["discipline"])
        result = self.h.evaluate_capability_effect(prediction.prediction_id)
        evaluation = result["evaluation"]
        # With a reference recorded, the change is ATTRIBUTABLE: it is no
        # longer reported as a merely descriptive movement.
        self.assertNotIn("DESCRIPTIVE",
                         " ".join(evaluation["exclusion_reasons"]))

    def test_induction_tasks_are_excluded_from_the_validation_sample(self):
        prediction, accepted = self._accepted()
        self.h.bind_capability_maintenance(
            prediction.prediction_id,
            adoption_action_id=accepted["adoption_action_id"])
        # Close an episode of a task INSIDE the prediction's own scope.
        self._close_later_episode("t1", observed_quality=0.95)
        result = self.h.evaluate_capability_effect(prediction.prediction_id)
        evidence = result["evaluation"]["evidence"]
        self.assertEqual(evidence["n_later_evaluations"], 0,
                         "reusing the induction tasks checks consistency, "
                         "not transfer")
        self.assertIn("t1", evidence["induction_tasks_excluded"])

    def test_no_change_operation_is_recorded_honestly(self):
        """An operation that produces nothing is a recorded outcome, not a
        capability gain."""
        self.seed_executions()
        prediction = self.predict()[1]
        recommendation = self.h.compare_capability_evolution(
            [prediction.prediction_id], horizon_tasks=10)
        # Accept with an EMPTY scope: induce consolidates nothing.
        self.h.actions.report_action(
            "induce", "t1", "maint_empty",
            params={"capability_prediction_id": prediction.prediction_id,
                    "action": "accepted"},
            outcome={"accepted": True,
                     "capability_prediction_id": prediction.prediction_id,
                     "operation_result": {
                         "business_result": "no_candidate",
                         "knowledge_delta": {"entries_created": [],
                                             "entries_updated": []},
                         "execution_ids": ["ex1", "ex2"]}},
            status="completed")
        bound = self.h.bind_capability_maintenance(prediction.prediction_id)
        self.assertFalse(bound["binding"]["changed"])
        self.assertTrue(any("NO knowledge change" in n
                            for n in bound["binding"]["notes"]))
        evaluation = self.h.evaluate_capability_effect(
            prediction.prediction_id)
        self.assertEqual(evaluation["evaluation"]["state"], "no_change")

    def test_repeated_evaluation_does_not_double_the_sample(self):
        prediction, accepted = self._accepted(payload={
            "expected_changes": [
                {"metric": "normalized_solution_quality", "unit": "1-gap",
                 "direction": "increase", "value": 0.2,
                 "beneficial_direction": "increase",
                 "baseline": {"kind": "conditional_stats", "value": 0.70}}],
            "learning_cost": {"llm_tokens": 100},
            "verification_conditions": [{"condition": "quality rises",
                                         "evaluable": True}]})
        self.h.bind_capability_maintenance(
            prediction.prediction_id,
            adoption_action_id=accepted["adoption_action_id"])
        self._close_later_episode("later1", observed_quality=0.95)
        self.h.record_capability_paired_evaluation(
            prediction.prediction_id, metric="normalized_solution_quality",
            reference_value=0.70, treated_value=0.95, unit="1-gap",
            reference_task_ids=["later1"])
        first = self.h.evaluate_capability_effect(prediction.prediction_id)
        second = self.h.evaluate_capability_effect(prediction.prediction_id)
        self.assertTrue(second["already_evaluated"])
        self.assertEqual(first["evaluation"], second["evaluation"])

    def test_w_or_is_never_advanced_by_the_predictor_itself(self):
        prediction, accepted = self._accepted()
        self.h.bind_capability_maintenance(
            prediction.prediction_id,
            adoption_action_id=accepted["adoption_action_id"])
        self._close_later_episode("later1", observed_quality=0.95)
        self.h.record_capability_paired_evaluation(
            prediction.prediction_id, metric="normalized_solution_quality",
            reference_value=0.70, treated_value=0.95)
        self.h.evaluate_capability_effect(prediction.prediction_id)
        evidence = self.h.capability_evidence_with_effects()
        w_or = evidence.sources.get("w_or")
        self.assertNotEqual(
            w_or.status if w_or else None, "direct_evidence",
            "W_OR needs independent OR prediction-error evidence, not the "
            "capability predictor's own report")
        self.assertTrue(any("not advanced by a capability-effect"
                            in n for n in (w_or.notes if w_or else [])))

    def test_verified_effect_advances_only_the_sources_observed(self):
        # A QUALITY prediction with a value and its own baseline: only then
        # can the observed level be turned into a comparable change.
        prediction, accepted = self._accepted(payload={
            "expected_changes": [
                {"metric": "normalized_solution_quality", "unit": "1-gap",
                 "direction": "increase", "value": 0.2,
                 "beneficial_direction": "increase",
                 "baseline": {"kind": "conditional_stats", "value": 0.70}}],
            "learning_cost": {"llm_tokens": 100},
            "verification_conditions": [{"condition": "quality rises",
                                         "evaluable": True}]})
        self.h.bind_capability_maintenance(
            prediction.prediction_id,
            adoption_action_id=accepted["adoption_action_id"])
        self._close_later_episode("later1", observed_quality=0.95)
        self.h.record_capability_paired_evaluation(
            prediction.prediction_id, metric="normalized_solution_quality",
            reference_value=0.70, treated_value=0.95, unit="1-gap",
            reference_task_ids=["later1"])
        result = self.h.evaluate_capability_effect(prediction.prediction_id)
        evaluation = result["evaluation"]
        change = evaluation["changes"][0]
        # The change is read from the caller's PAIRED comparison (treated
        # minus reference), which is what makes it attributable: 0.95 - 0.70.
        self.assertAlmostEqual(change["observed_change"], 0.25, places=5)
        self.assertEqual(change["paired"], True)
        self.assertEqual(change["paired_reference"]["reference_value"], 0.70)
        self.assertEqual(change["paired_reference"]["treated_value"], 0.95)
        self.assertIn("paired comparison", change["change_basis"])
        self.assertEqual(change["agreement"], "confirmed")
        self.assertEqual(evaluation["state"], "observed_improvement")
        self.assertTrue(evaluation["effect_verified"])
        evidence = self.h.capability_evidence_with_effects()
        self.assertEqual(evidence.sources["m"].status, "direct_evidence")
        # The tool source had no cost observation: unchanged.
        self.assertNotEqual(evidence.sources["t"].status,
                            "direct_evidence")

    def test_feedback_summary_separates_fact_from_effect(self):
        prediction, accepted = self._accepted()
        summary = self.h.capability_feedback_summary()
        self.assertEqual(summary["n_predictions"], 1)
        # Acceptance bound the fact automatically: the operation really
        # ran, and that is a recorded fact.
        self.assertEqual(summary["n_fact_bound"], 1)
        self.assertEqual(summary["n_effect_verified"], 0,
                         "a bound fact is NOT a verified effect")
        # Re-binding is a no-op: the fact is not counted twice.
        again = self.h.bind_capability_maintenance(
            prediction.prediction_id,
            adoption_action_id=accepted["adoption_action_id"])
        self.assertTrue(again["already_bound"])
        self.assertEqual(
            self.h.capability_feedback_summary()["n_fact_bound"], 1)

    # -- helpers ----------------------------------------------------------

    def _close_later_episode(self, task_id, *, observed_quality):
        """Close a REAL episode whose bound prediction observed a quality."""
        task = _task(task_id)
        payload = {
            "benefit": {"kind": "solution_quality",
                        "metric": "normalized_objective_gap", "unit": "1-gap",
                        "value": observed_quality,
                        "baseline": {"kind": "conditional_stats",
                                     "value": 0.7}},
            "cost": {"llm_tokens": 1200, "solver_runtime_s": 3.0},
            "risk": {"events": []},
        }
        provider = StubProvider(payload=payload)
        h = ORHarness(home=self.home, world_model=provider,
                      embedding=self.backend)
        self.addCleanup(h.close)
        prediction = h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy", "strategy_id": "S04"},
            "ep1")
        from pathlib import Path
        work = Path(self.home) / f"ws_{task_id}"
        work.mkdir(parents=True, exist_ok=True)
        script = work / "solve.py"
        script.write_text(
            "import json\n"
            "with open('result.json', 'w') as fh:\n"
            "    json.dump({'status': 'optimal', 'objective_value': 1.0,"
            " 'objective_bound': 1.0, 'runtime_seconds': 0.01}, fh)\n",
            encoding="utf-8")
        record = h.execute(task, "S04", str(script), str(work),
                           solver="highs", episode_id="ep1")
        h.record(record)
        h.bind_strategy_outcome(prediction.prediction_id, record.action_id)
        h.close_episode(task_id, "ep1")


# ---------------------------------------------------------------------------
# 7. the review round: the seven defects found after the first delivery
# ---------------------------------------------------------------------------


class TestReviewRoundFixes(M5Case):
    """Regressions for the seven reviewed M5 defects.

    Each test reproduces the reported behaviour and asserts the corrected
    one, so a future refactor that reintroduces it fails loudly instead of
    quietly returning a flattering verdict.
    """

    # -- P1-1: a paired record must be READ, not merely present ----------

    def test_paired_reference_that_shows_a_collapse_refutes_the_claim(self):
        """A pair whose treated value is WORSE than its reference must be
        able to refute the prediction it was recorded for. Merely having a
        record is not attribution."""
        prediction, accepted = self._accepted(payload={
            "expected_changes": [
                {"metric": "normalized_solution_quality", "unit": "1-gap",
                 "direction": "increase", "value": 0.2,
                 "beneficial_direction": "increase"}],
            "learning_cost": {"llm_tokens": 100},
            "verification_conditions": [{"condition": "quality rises",
                                         "evaluable": True}]})
        self.h.bind_capability_maintenance(
            prediction.prediction_id,
            adoption_action_id=accepted["adoption_action_id"])
        self._close_later_episode("later1", observed_quality=0.95)
        # The paired window shows quality COLLAPSING 0.9 -> 0.1.
        self.h.record_capability_paired_evaluation(
            prediction.prediction_id, metric="normalized_solution_quality",
            reference_value=0.9, treated_value=0.1, unit="1-gap",
            reference_task_ids=["later1"])
        result = self.h.evaluate_capability_effect(prediction.prediction_id)
        evaluation = result["evaluation"]
        self.assertFalse(evaluation["effect_verified"],
                         "a pair that moved the WRONG way must not verify "
                         "an improvement")
        change = evaluation["changes"][0]
        self.assertAlmostEqual(change["observed_change"], -0.8, places=5)
        self.assertEqual(change["agreement"], "refuted")

    def test_paired_reference_on_another_metric_is_not_a_control(self):
        """A pair taken on a different metric covers none of the predicted
        changes, so the result stays DESCRIPTIVE."""
        prediction, accepted = self._accepted(payload={
            "expected_changes": [
                {"metric": "normalized_solution_quality", "unit": "1-gap",
                 "direction": "increase", "value": 0.2,
                 "beneficial_direction": "increase"}],
            "learning_cost": {"llm_tokens": 100},
            "verification_conditions": [{"condition": "quality rises",
                                         "evaluable": True}]})
        self.h.bind_capability_maintenance(
            prediction.prediction_id,
            adoption_action_id=accepted["adoption_action_id"])
        self._close_later_episode("later1", observed_quality=0.95)
        # A COST pair for a QUALITY claim.
        self.h.record_capability_paired_evaluation(
            prediction.prediction_id, metric="resource_cost",
            reference_value=5.0, treated_value=3.0, unit="s",
            reference_task_ids=["later1"])
        result = self.h.evaluate_capability_effect(prediction.prediction_id)
        evaluation = result["evaluation"]
        self.assertEqual(evaluation["state"], "inconclusive")
        self.assertFalse(evaluation["effect_verified"])
        self.assertTrue(evaluation["evidence"]["paired_reference_unusable"])

    def test_paired_reference_in_another_unit_is_not_a_control(self):
        prediction, accepted = self._accepted(payload={
            "expected_changes": [
                {"metric": "normalized_solution_quality", "unit": "1-gap",
                 "direction": "increase", "value": 0.2,
                 "beneficial_direction": "increase"}],
            "learning_cost": {"llm_tokens": 100},
            "verification_conditions": [{"condition": "quality rises",
                                         "evaluable": True}]})
        self.h.bind_capability_maintenance(
            prediction.prediction_id,
            adoption_action_id=accepted["adoption_action_id"])
        self._close_later_episode("later1", observed_quality=0.95)
        self.h.record_capability_paired_evaluation(
            prediction.prediction_id, metric="normalized_solution_quality",
            reference_value=70.0, treated_value=95.0, unit="percent",
            reference_task_ids=["later1"])
        result = self.h.evaluate_capability_effect(prediction.prediction_id)
        self.assertEqual(result["evaluation"]["state"], "inconclusive")

    def test_a_paired_record_citing_an_unrun_task_is_not_evidence(self):
        """A comparison between measurements nobody made is an assertion
        wearing a reference."""
        prediction, accepted = self._accepted(payload={
            "expected_changes": [
                {"metric": "normalized_solution_quality", "unit": "1-gap",
                 "direction": "increase", "value": 0.2,
                 "beneficial_direction": "increase"}],
            "learning_cost": {"llm_tokens": 100},
            "verification_conditions": [{"condition": "quality rises",
                                         "evaluable": True}]})
        self.h.bind_capability_maintenance(
            prediction.prediction_id,
            adoption_action_id=accepted["adoption_action_id"])
        self._close_later_episode("later1", observed_quality=0.95)
        # The cited task never ran at all.
        self.h.record_capability_paired_evaluation(
            prediction.prediction_id, metric="normalized_solution_quality",
            reference_value=0.70, treated_value=0.95, unit="1-gap",
            reference_task_ids=["never_ran"])
        result = self.h.evaluate_capability_effect(prediction.prediction_id)
        evaluation = result["evaluation"]
        self.assertEqual(evaluation["state"], "inconclusive")
        self.assertFalse(evaluation["effect_verified"])
        unusable = evaluation["evidence"]["paired_reference_unusable"]
        self.assertIn("no closed episode", unusable[0]["reason"])

    # -- P1-2: later + matching + horizon must really be enforced --------

    def test_a_task_closed_before_the_operation_is_not_evidence(self):
        """A pre-maintenance episode cannot validate a post-maintenance
        claim, however convenient its numbers are."""
        # Close the episode BEFORE any prediction/operation exists.
        self._close_later_episode("before1", observed_quality=0.95)
        prediction, accepted = self._accepted(payload={
            "expected_changes": [
                {"metric": "normalized_solution_quality", "unit": "1-gap",
                 "direction": "increase", "value": 0.2,
                 "beneficial_direction": "increase"}],
            "learning_cost": {"llm_tokens": 100},
            "verification_conditions": [{"condition": "quality rises",
                                         "evaluable": True}]})
        self.h.bind_capability_maintenance(
            prediction.prediction_id,
            adoption_action_id=accepted["adoption_action_id"])
        self.h.record_capability_paired_evaluation(
            prediction.prediction_id, metric="normalized_solution_quality",
            reference_value=0.70, treated_value=0.95, unit="1-gap",
            reference_task_ids=["before1"])
        result = self.h.evaluate_capability_effect(prediction.prediction_id)
        evaluation = result["evaluation"]
        self.assertEqual(evaluation["state"], "pending")
        evidence = evaluation["evidence"]
        self.assertEqual(evidence["n_later_evaluations"], 0)
        self.assertEqual([e["task_id"] for e in
                          evidence["excluded_not_later"]], ["before1"])

    def test_an_off_target_task_is_not_a_matching_task(self):
        """A task outside the prediction's frozen target cannot validate a
        claim about this population."""
        prediction, accepted = self._accepted(payload={
            "expected_changes": [
                {"metric": "normalized_solution_quality", "unit": "1-gap",
                 "direction": "increase", "value": 0.2,
                 "beneficial_direction": "increase"}],
            "learning_cost": {"llm_tokens": 100},
            "verification_conditions": [{"condition": "quality rises",
                                         "evaluable": True}]})
        self.h.bind_capability_maintenance(
            prediction.prediction_id,
            adoption_action_id=accepted["adoption_action_id"])
        # A task of ANOTHER family: structurally unrelated to the claim.
        self._close_later_episode("elsewhere", observed_quality=0.95,
                                  family="scheduling")
        result = self.h.evaluate_capability_effect(prediction.prediction_id)
        evidence = result["evaluation"]["evidence"]
        self.assertEqual(evidence["n_later_evaluations"], 0)
        # The task is rejected: it is outside the frozen target (and, having
        # run under the pre-operation knowledge, it is also not a later
        # task). Whichever filter catches it, it never becomes a sample.
        rejected = ([e["task_id"] for e in evidence["excluded_off_target"]]
                    + [e["task_id"] for e in
                       evidence["excluded_not_later"]])
        self.assertEqual(rejected, ["elsewhere"])

    def test_an_unmet_horizon_stays_pending(self):
        """A claim made over N tasks is not confirmed by fewer than N."""
        prediction, accepted = self._accepted(
            payload={
                "expected_changes": [
                    {"metric": "normalized_solution_quality",
                     "unit": "1-gap", "direction": "increase", "value": 0.2,
                     "beneficial_direction": "increase"}],
                "learning_cost": {"llm_tokens": 100},
                "verification_conditions": [{"condition": "quality rises",
                                             "evaluable": True}]},
            horizon_tasks=3)
        self.h.bind_capability_maintenance(
            prediction.prediction_id,
            adoption_action_id=accepted["adoption_action_id"])
        self._close_later_episode("later1", observed_quality=0.95)
        self.h.record_capability_paired_evaluation(
            prediction.prediction_id, metric="normalized_solution_quality",
            reference_value=0.70, treated_value=0.95, unit="1-gap",
            reference_task_ids=["later1"])
        result = self.h.evaluate_capability_effect(prediction.prediction_id)
        evaluation = result["evaluation"]
        self.assertEqual(evaluation["state"], "pending")
        self.assertFalse(evaluation["effect_verified"])
        self.assertEqual(evaluation["evidence"]["horizon_tasks"], 3)
        self.assertFalse(evaluation["evidence"]["horizon_met"])
        self.assertIn("INCOMPLETE", " ".join(evaluation["exclusion_reasons"]))

    # -- P1-3: the framework owns the baseline ---------------------------

    def test_a_model_supplied_baseline_is_recorded_and_replaced(self):
        """A model may cite a frozen reference, never set one: a baseline
        chosen after seeing the evidence grades the model's own work."""
        prediction = self.predict(payload={
            "expected_changes": [
                {"metric": "resource_cost", "unit": "s",
                 "direction": "decrease", "value": -6.0,
                 "beneficial_direction": "decrease",
                 "baseline": {"kind": "conditional_stats", "value": 0.0}}],
            "learning_cost": {"solver_runtime_s": 0.1},
            "verification_conditions": [{"condition": "cost falls",
                                         "evaluable": True}]})[1]
        self.assertEqual(prediction.status, "valid")
        change = prediction.expected_changes[0]
        # The bundle's frozen mean solver_runtime_s (5.0) wins.
        self.assertIsNotNone(change.baseline)
        self.assertAlmostEqual(change.baseline.value, 5.0, places=5)
        overrides = prediction.trace.model_info.get(
            "attempted_field_overrides") or {}
        self.assertIn("expected_changes[0].baseline", overrides)
        self.assertEqual(overrides["expected_changes[0].baseline"]["value"],
                         0.0)

    def test_the_frozen_baseline_is_published_per_metric(self):
        prediction = self.predict()[1]
        frozen = prediction.baselines_by_metric
        self.assertIn("normalized_solution_quality", frozen)
        # Cost references are keyed by DIMENSION: several dimensions share
        # the resource_cost metric but their units do not convert.
        self.assertIn("cost:solver_runtime_s", frozen)
        self.assertEqual(frozen["normalized_solution_quality"].metric,
                         "normalized_solution_quality")
        self.assertEqual(frozen["cost:solver_runtime_s"].unit, "s")
        # A reference is only handed to its OWN metric and unit.
        self.assertIsNone(prediction.frozen_baseline_for("unknown_metric"))
        self.assertEqual(
            prediction.frozen_baseline_for("resource_cost", unit="s").unit,
            "s")
        self.assertIsNone(
            prediction.frozen_baseline_for("resource_cost",
                                           unit="tokens"))

    # -- P1-4: comparable means the same unit AND a positive net ---------

    def test_savings_in_different_units_are_never_ranked_against_each_other(
            self):
        seconds = self.predict(payload={
            "expected_changes": [
                {"metric": "resource_cost", "unit": "s",
                 "direction": "decrease", "value": -10.0,
                 "beneficial_direction": "decrease"}],
            "learning_cost": {"solver_runtime_s": 1.0},
            "verification_conditions": [{"condition": "cost falls",
                                         "evaluable": True}]})[1]
        tokens = self.predict(payload={
            "expected_changes": [
                {"metric": "resource_cost", "unit": "tokens",
                 "direction": "decrease", "value": -100.0,
                 "beneficial_direction": "decrease"}],
            "learning_cost": {"llm_tokens": 10.0},
            "verification_conditions": [{"condition": "cost falls",
                                         "evaluable": True}]})[1]
        result = self.h.compare_capability_evolution(
            [seconds.prediction_id, tokens.prediction_id], horizon_tasks=10)
        self.assertEqual(result["recommendation"], "defer",
                         "10 seconds and 100 tokens are different "
                         "quantities: picking the larger NUMBER would be a "
                         "unit-blind ranking")
        self.assertIsNone(result["selected_prediction_id"])
        self.assertIn("different units", result["basis"])

    def test_a_maintenance_cost_larger_than_the_saving_defers(self):
        prediction = self.predict(payload={
            "expected_changes": [
                {"metric": "resource_cost", "unit": "s",
                 "direction": "decrease", "value": -100.0,
                 "beneficial_direction": "decrease"}],
            "learning_cost": {"solver_runtime_s": 10000.0},
            "verification_conditions": [{"condition": "cost falls",
                                         "evaluable": True}]})[1]
        result = self.h.compare_capability_evolution(
            [prediction.prediction_id], horizon_tasks=10)
        self.assertEqual(result["recommendation"], "defer")
        self.assertIn("does not pay for itself",
                      result["incomparable"][0]["reason"])
        net = result["incomparable"][0]["net_saving"]
        self.assertEqual(net["cumulative_saving"], 1000.0)
        self.assertEqual(net["maintenance_cost"], 10000.0)
        self.assertEqual(net["net_over_horizon"], -9000.0)

    def test_a_cost_that_only_pays_off_beyond_one_task_is_recommended(self):
        """A one-time cost larger than a SINGLE task's saving is still
        worth it when the declared window pays it back: netting the cost
        against one task would reject a profitable operation."""
        prediction = self.predict(payload={
            "expected_changes": [
                {"metric": "resource_cost", "unit": "s",
                 "direction": "decrease", "value": -2.0,
                 "beneficial_direction": "decrease"}],
            "learning_cost": {"solver_runtime_s": 5.0},
            "verification_conditions": [{"condition": "cost falls",
                                         "evaluable": True}]})[1]
        result = self.h.compare_capability_evolution(
            [prediction.prediction_id], horizon_tasks=10)
        self.assertEqual(result["recommendation"], "accept")
        net = result["comparisons"][0]["net_saving"]
        self.assertEqual(net["cumulative_saving"], 20.0)
        self.assertEqual(net["net_over_horizon"], 15.0)
        self.assertLess(net["net_per_task"], 0,
                        "the per-task figure alone would have deferred it")

    def test_a_learning_cost_in_another_currency_defers(self):
        prediction = self.predict(payload={
            "expected_changes": [
                {"metric": "resource_cost", "unit": "s",
                 "direction": "decrease", "value": -10.0,
                 "beneficial_direction": "decrease"}],
            "learning_cost": {"llm_tokens": 100.0},
            "verification_conditions": [{"condition": "cost falls",
                                         "evaluable": True}]})[1]
        result = self.h.compare_capability_evolution(
            [prediction.prediction_id], horizon_tasks=10)
        self.assertEqual(result["recommendation"], "defer")
        self.assertIn("not expressed in the saving's own unit",
                      result["incomparable"][0]["reason"])

    def test_the_net_saving_is_reported_per_candidate(self):
        prediction = self.predict(payload={
            "expected_changes": [
                {"metric": "resource_cost", "unit": "s",
                 "direction": "decrease", "value": -6.0,
                 "beneficial_direction": "decrease"}],
            "learning_cost": {"solver_runtime_s": 1.0},
            "verification_conditions": [{"condition": "cost falls",
                                         "evaluable": True}]})[1]
        result = self.h.compare_capability_evolution(
            [prediction.prediction_id], horizon_tasks=10)
        net = result["comparisons"][0]["net_saving"]
        self.assertEqual(net["per_task"], 6.0)
        self.assertEqual(net["maintenance_cost"], 1.0)
        self.assertEqual(net["net_per_task"], 5.0)
        self.assertEqual(net["net_over_horizon"], 59.0)

    # -- P1-5: the operation TYPE decides what runs ----------------------

    def test_an_operation_without_an_execution_path_is_refused(self):
        """A 'reverify' candidate has no execution entry: accepting it must
        REFUSE, never quietly run induce() under its name."""
        self.seed_executions()
        prediction = self.predict(
            payload={
                "expected_changes": [
                    {"metric": "resource_cost", "unit": "s",
                     "direction": "decrease", "value": -6.0,
                     "beneficial_direction": "decrease"}],
                "learning_cost": {"solver_runtime_s": 0.1},
                "verification_conditions": [{"condition": "cost falls",
                                             "evaluable": True}]},
            operation={"operation_type": "reverify",
                       "strategy_id": "S04"})[1]
        before = self.h.sbank.count()
        with self.assertRaises(ValueError) as ctx:
            self.h.accept_capability_operation({
                "recommendation": "accept",
                "selected_prediction_id": prediction.prediction_id})
        self.assertIn("no execution path", str(ctx.exception))
        self.assertEqual(self.h.sbank.count(), before,
                         "a refused operation must change nothing")

    def test_an_unsupported_operation_is_never_recommended(self):
        self.seed_executions()
        prediction = self.predict(
            payload={
                "expected_changes": [
                    {"metric": "resource_cost", "unit": "s",
                     "direction": "decrease", "value": -6.0,
                     "beneficial_direction": "decrease"}],
                "learning_cost": {"solver_runtime_s": 0.1},
                "verification_conditions": [{"condition": "cost falls",
                                             "evaluable": True}]},
            operation={"operation_type": "reverify",
                       "strategy_id": "S04"})[1]
        result = self.h.compare_capability_evolution(
            [prediction.prediction_id], horizon_tasks=10)
        self.assertEqual(result["recommendation"], "defer")
        self.assertIn("no execution path",
                      result["incomparable"][0]["reason"])

    def test_a_retire_operation_really_retires(self):
        """The dispatched operation matches the prediction: a 'retire'
        candidate removes an entry instead of creating one."""
        self.seed_executions()
        self.h.induce(strategy_id="S04")
        entries = self.h.sbank.list()
        self.assertTrue(entries, "the fixture needs a real entry to retire")
        entry_id = entries[0].entry_id
        prediction = self.predict(
            payload={
                "expected_changes": [
                    {"metric": "resource_cost", "unit": "s",
                     "direction": "decrease", "value": -6.0,
                     "beneficial_direction": "decrease"}],
                "learning_cost": {"solver_runtime_s": 0.1},
                "verification_conditions": [{"condition": "cost falls",
                                             "evaluable": True}]},
            operation={"operation_type": "retire", "strategy_id": "S04",
                       "config": {"target_entry_id": entry_id}})[1]
        result = self.h.accept_capability_operation({
            "recommendation": "accept",
            "selected_prediction_id": prediction.prediction_id})
        self.assertEqual(result["operation_type"], "retire")
        self.assertIsNone(self.h.sbank.get(entry_id),
                          "a retire must remove the entry, not create one")
        bound = self.h.bind_capability_maintenance(
            prediction.prediction_id,
            adoption_action_id=result["adoption_action_id"])
        self.assertEqual(
            bound["binding"]["knowledge_delta"]["entries_retired"],
            [entry_id])
        self.assertEqual(
            bound["binding"]["knowledge_delta"]["entries_created"], [])

    def test_a_retire_without_a_named_target_is_refused(self):
        self.seed_executions()
        prediction = self.predict(
            payload={
                "expected_changes": [
                    {"metric": "resource_cost", "unit": "s",
                     "direction": "decrease", "value": -6.0,
                     "beneficial_direction": "decrease"}],
                "learning_cost": {"solver_runtime_s": 0.1},
                "verification_conditions": [{"condition": "cost falls",
                                             "evaluable": True}]},
            operation={"operation_type": "retire",
                       "strategy_id": "S04"})[1]
        with self.assertRaises(ValueError) as ctx:
            self.h.accept_capability_operation({
                "recommendation": "accept",
                "selected_prediction_id": prediction.prediction_id})
        self.assertIn("target entry", str(ctx.exception))

    # -- P1-6: capability calls are real spend ---------------------------

    def test_capability_prediction_costs_enter_the_budget_ledger(self):
        """Three 50-token calls against a 40-token budget must EXCEED it,
        not report "ok" while the tokens burn."""
        h = ORHarness(home=self.home, world_model=StubProvider(
            usage={"prompt_tokens": 10, "completion_tokens": 50}),
            embedding=self.backend)
        self.addCleanup(h.close)
        h.declare_budget("t_only", {"llm_tokens": 40})
        for _ in range(3):
            h.predict_capability_evolution(
                {"operation_type": "induce", "strategy_id": "S04"},
                bundle=_bundle(), horizon="next 10 tasks",
                horizon_tasks=10, task_id="t_only")
        view = h.budget_view("t_only")
        self.assertEqual(view["status"], "exceeded")
        self.assertEqual(view["consumption"]["total_cost"]["llm_tokens"],
                         150.0)
        self.assertEqual(
            len(view["consumption"]["prediction_call_costs"]), 3)

    def test_a_failed_capability_call_is_still_charged(self):
        class FailingProvider(WorldModelProvider):
            name = "failing-m5"

            def predict(self, request, timeout_s=None):
                return {"payload": None, "error": "model unavailable",
                        "usage": {"completion_tokens": 30},
                        "latency_s": 0.01}

        h = ORHarness(home=self.home, world_model=FailingProvider(),
                      embedding=self.backend)
        self.addCleanup(h.close)
        h.declare_budget("t_fail", {"llm_tokens": 10})
        prediction = h.predict_capability_evolution(
            {"operation_type": "induce", "strategy_id": "S04"},
            bundle=_bundle(), horizon="next 10 tasks", horizon_tasks=10,
            task_id="t_fail")
        self.assertEqual(prediction.status, "invalid")
        view = h.budget_view("t_fail")
        self.assertEqual(view["status"], "exceeded",
                         "a call that really happened is real spend even "
                         "when it produced no usable prediction")
        self.assertEqual(view["consumption"]["total_cost"]["llm_tokens"],
                         30.0)

    # -- P2-7: verified effects condition the NEXT prediction ------------

    def test_verified_effects_reach_the_next_prediction(self):
        """The loop must close: a verified effect advances the evidence the
        NEXT capability prediction is built on."""
        prediction, accepted = self._accepted(payload={
            "expected_changes": [
                {"metric": "normalized_solution_quality", "unit": "1-gap",
                 "direction": "increase", "value": 0.2,
                 "beneficial_direction": "increase"}],
            "learning_cost": {"llm_tokens": 100},
            "verification_conditions": [{"condition": "quality rises",
                                         "evaluable": True}]})
        self.h.bind_capability_maintenance(
            prediction.prediction_id,
            adoption_action_id=accepted["adoption_action_id"])
        self._close_later_episode("later1", observed_quality=0.95)
        self.h.record_capability_paired_evaluation(
            prediction.prediction_id, metric="normalized_solution_quality",
            reference_value=0.70, treated_value=0.95, unit="1-gap",
            reference_task_ids=["later1"])
        evaluated = self.h.evaluate_capability_effect(
            prediction.prediction_id)
        self.assertTrue(evaluated["evaluation"]["effect_verified"])
        # A NEW prediction is built on evidence that includes the effect.
        later = self.predict()[1]
        self.assertEqual(later.current_evidence.sources["m"].status,
                         "direct_evidence")
        self.assertTrue(any("advanced by verified effect evaluation"
                            in n for n in
                            later.current_evidence.sources["m"].notes))
        # W_OR is never advanced by the predictor's own report.
        self.assertNotEqual(later.current_evidence.sources["w_or"].status,
                            "direct_evidence")

    def test_an_unverified_effect_does_not_reach_the_next_prediction(self):
        prediction, accepted = self._accepted(payload={
            "expected_changes": [
                {"metric": "normalized_solution_quality", "unit": "1-gap",
                 "direction": "increase", "value": 0.2,
                 "beneficial_direction": "increase"}],
            "learning_cost": {"llm_tokens": 100},
            "verification_conditions": [{"condition": "quality rises",
                                         "evaluable": True}]})
        self.h.bind_capability_maintenance(
            prediction.prediction_id,
            adoption_action_id=accepted["adoption_action_id"])
        # A bound fact alone is not an effect: the evidence stays as it was.
        later = self.predict()[1]
        self.assertNotEqual(later.current_evidence.sources["m"].status,
                            "direct_evidence")

    # -- helpers ----------------------------------------------------------

    def _close_later_episode(self, task_id, *, observed_quality,
                             family="routing"):
        """Close a REAL episode whose bound prediction observed a quality."""
        task = _task(task_id)
        if family != "routing":
            task["family"] = family
        payload = {
            "benefit": {"kind": "solution_quality",
                        "metric": "normalized_objective_gap", "unit": "1-gap",
                        "value": observed_quality,
                        "baseline": {"kind": "conditional_stats",
                                     "value": 0.7}},
            "cost": {"llm_tokens": 1200, "solver_runtime_s": 3.0},
            "risk": {"events": []},
        }
        provider = StubProvider(payload=payload)
        h = ORHarness(home=self.home, world_model=provider,
                      embedding=self.backend)
        self.addCleanup(h.close)
        prediction = h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy", "strategy_id": "S04"},
            "ep1")
        from pathlib import Path
        work = Path(self.home) / f"ws_{task_id}"
        work.mkdir(parents=True, exist_ok=True)
        script = work / "solve.py"
        script.write_text(
            "import json\n"
            "with open('result.json', 'w') as fh:\n"
            "    json.dump({'status': 'optimal', 'objective_value': 1.0,"
            " 'objective_bound': 1.0, 'runtime_seconds': 0.01}, fh)\n",
            encoding="utf-8")
        record = h.execute(task, "S04", str(script), str(work),
                           solver="highs", episode_id="ep1")
        h.record(record)
        h.bind_strategy_outcome(prediction.prediction_id, record.action_id)
        h.close_episode(task_id, "ep1")

    def _accepted(self, payload=None, *, prediction_id=None,
                  horizon_tasks=1, operation=None):
        """Accept a candidate and run the real operation."""
        self.seed_executions()
        prediction = self.predict(payload=payload, operation=operation,
                                  horizon_tasks=horizon_tasks)[1]
        recommendation = self.h.compare_capability_evolution(
            [prediction.prediction_id], horizon_tasks=horizon_tasks)
        if recommendation["recommendation"] != "accept":
            recommendation = {
                "recommendation": "accept",
                "recommendation_id": recommendation["recommendation_id"],
                "selected_prediction_id": prediction.prediction_id,
                "basis": ("explicit agent choice among the incomparable "
                          "candidates"),
            }
        accepted = self.h.accept_capability_operation(
            recommendation, prediction_id=prediction_id)
        return prediction, accepted


# ---------------------------------------------------------------------------
# 8. the second review round: six more defects
# ---------------------------------------------------------------------------


class TestReviewRoundTwo(M5Case):
    """Regressions for the second batch of reviewed M5 defects.

    Each made the framework accept evidence it should have refused: a
    sample that was not independent, a task that ran before the change, a
    yardstick taken on another dimension, a decision that netted a one-time
    cost against one task, spend that was never charged, and a real
    retirement reported as no change.
    """

    def _quality_payload(self, value=0.2):
        return {
            "expected_changes": [
                {"metric": "normalized_solution_quality", "unit": "1-gap",
                 "direction": "increase", "value": value,
                 "beneficial_direction": "increase"}],
            "learning_cost": {"llm_tokens": 100},
            "verification_conditions": [{"condition": "quality rises",
                                         "evaluable": True}]}

    def _accept(self, payload, *, horizon_tasks=1, operation=None):
        self.seed_executions()
        prediction = self.predict(payload=payload, operation=operation,
                                  horizon_tasks=horizon_tasks)[1]
        recommendation = self.h.compare_capability_evolution(
            [prediction.prediction_id], horizon_tasks=horizon_tasks)
        if recommendation["recommendation"] != "accept":
            recommendation = {
                "recommendation": "accept",
                "recommendation_id": recommendation["recommendation_id"],
                "selected_prediction_id": prediction.prediction_id,
                "basis": "explicit agent choice",
            }
        accepted = self.h.accept_capability_operation(recommendation)
        self.h.bind_capability_maintenance(
            prediction.prediction_id,
            adoption_action_id=accepted["adoption_action_id"])
        return prediction, accepted

    def _run_task(self, task_id, *, episode="ep1", observed_quality=0.95,
                  predictions_per_episode=1, at=None, close=True):
        """Run a real task (optionally with several bound predictions)."""
        payload = {
            "benefit": {"kind": "solution_quality",
                        "metric": "normalized_objective_gap",
                        "unit": "1-gap", "value": observed_quality,
                        "baseline": {"kind": "conditional_stats",
                                     "value": 0.7}},
            "cost": {"llm_tokens": 1200, "solver_runtime_s": 3.0},
            "risk": {"events": []},
        }
        orx = ORHarness(home=self.home, world_model=StubProvider(
            payload=payload), embedding=self.backend)
        self.addCleanup(orx.close)
        task = _task(task_id)
        predictions = [
            orx.predict_strategy_outcome(
                task, {"action_type": "execute_strategy",
                       "strategy_id": "S04"}, episode)
            for _ in range(predictions_per_episode)]
        from pathlib import Path
        work = Path(self.home) / f"ws_{task_id}_{episode}"
        work.mkdir(parents=True, exist_ok=True)
        script = work / "solve.py"
        script.write_text(
            "import json\n"
            "with open('result.json', 'w') as fh:\n"
            "    json.dump({'status': 'optimal', 'objective_value': 1.0,"
            " 'objective_bound': 1.0, 'runtime_seconds': 0.01}, fh)\n",
            encoding="utf-8")
        record = orx.execute(task, "S04", str(script), str(work),
                             solver="highs", episode_id=episode)
        orx.record(record)
        for prediction in predictions:
            orx.bind_strategy_outcome(prediction.prediction_id,
                                      record.action_id)
        if at is not None:
            with orx.store.transaction() as conn:
                row = conn.execute(
                    "SELECT payload FROM executions WHERE execution_id=?",
                    (record.execution_id,)).fetchone()
                data = orx.store.loads(row["payload"])
                data["created_at"] = at
                conn.execute(
                    "UPDATE executions SET payload=? WHERE execution_id=?",
                    (orx.store.dumps(data), record.execution_id))
        if close:
            orx.close_episode(task_id, episode)
        return record.execution_id

    # -- P1-A: the sample unit is the TASK-EPISODE -----------------------

    def test_one_episode_with_many_predictions_is_one_sample(self):
        """Ten predictions bound to ONE execution are ONE independent
        truth: re-planning a task many times cannot satisfy a ten-task
        horizon."""
        prediction, _ = self._accept(self._quality_payload(),
                                     horizon_tasks=10)
        self._run_task("later1", predictions_per_episode=10)
        self.h.record_capability_paired_evaluation(
            prediction.prediction_id, metric="normalized_solution_quality",
            reference_value=0.70, treated_value=0.95, unit="1-gap",
            reference_task_ids=["later1"])
        result = self.h.evaluate_capability_effect(prediction.prediction_id)
        evaluation = result["evaluation"]
        evidence = evaluation["evidence"]
        self.assertEqual(evidence["n_later_records"], 10)
        self.assertEqual(evidence["n_later_evaluations"], 1)
        self.assertEqual(evidence["distinct_episodes"], 1)
        self.assertFalse(evidence["horizon_met"])
        self.assertEqual(evaluation["state"], "pending")
        self.assertFalse(evaluation["effect_verified"])
        self.assertIn("TASK-EPISODES", evidence["note"])

    def test_two_episodes_of_one_task_are_two_samples(self):
        """Independence is the (task_id, episode_id) pair, so the same task
        re-run in a SECOND episode is genuinely new evidence."""
        prediction, _ = self._accept(self._quality_payload(),
                                     horizon_tasks=2)
        self._run_task("later1", episode="ep1")
        self._run_task("later1", episode="ep2")
        self.h.record_capability_paired_evaluation(
            prediction.prediction_id, metric="normalized_solution_quality",
            reference_value=0.70, treated_value=0.95, unit="1-gap",
            reference_task_ids=["later1"])
        result = self.h.evaluate_capability_effect(prediction.prediction_id)
        evidence = result["evaluation"]["evidence"]
        self.assertEqual(evidence["distinct_episodes"], 2)
        self.assertTrue(evidence["horizon_met"])

    # -- P1-B: the SOLVE time and knowledge decide, not the close time ---

    def test_a_task_solved_before_the_operation_is_not_evidence(self):
        """Closing an episode after the maintenance does not make the work
        later: the solve really ran against the earlier knowledge."""
        import time
        early = time.time() - 86400
        # The task is executed (and its episode left open) BEFORE the
        # prediction and the operation.
        self._run_task("solved_early", at=early, close=False)
        prediction, _ = self._accept(self._quality_payload())
        # Only the CLOSE happens after the maintenance.
        closer = ORHarness(home=self.home, world_model=StubProvider(),
                           embedding=self.backend)
        self.addCleanup(closer.close)
        closer.close_episode("solved_early", "ep1")
        self.h.record_capability_paired_evaluation(
            prediction.prediction_id, metric="normalized_solution_quality",
            reference_value=0.70, treated_value=0.95, unit="1-gap",
            reference_task_ids=["solved_early"])
        result = self.h.evaluate_capability_effect(prediction.prediction_id)
        evaluation = result["evaluation"]
        self.assertEqual(evaluation["evidence"]["n_later_evaluations"], 0)
        self.assertEqual(evaluation["state"], "pending")
        excluded = evaluation["evidence"]["excluded_not_later"]
        self.assertEqual([e["task_id"] for e in excluded],
                         ["solved_early"])
        self.assertIn("SOLVED before", excluded[0]["reason"])

    # -- P1-C: cost references are per DIMENSION -------------------------

    def test_cost_baselines_are_kept_per_dimension(self):
        """Several cost dimensions share the resource_cost metric but their
        units do not convert, so each keeps its OWN reference."""
        self.seed_executions()
        bundle = {**_bundle(), "mean_cost": {"solver_runtime_s": 5.0,
                                             "llm_tokens": 1000.0}}
        prediction = self.h.predict_capability_evolution(
            {"operation_type": "induce", "strategy_id": "S04"},
            bundle=bundle, horizon="next 10 tasks", horizon_tasks=10)
        frozen = prediction.baselines_by_metric
        self.assertEqual(frozen["cost:solver_runtime_s"].value, 5.0)
        self.assertEqual(frozen["cost:llm_tokens"].value, 1000.0)
        self.assertEqual(frozen["cost:solver_runtime_s"].unit, "s")
        self.assertEqual(frozen["cost:llm_tokens"].unit, "tokens")
        # A unit-qualified lookup returns THAT dimension's reference.
        self.assertEqual(
            prediction.frozen_baseline_for("resource_cost",
                                           unit="s").value, 5.0)
        self.assertEqual(
            prediction.frozen_baseline_for("resource_cost",
                                           unit="tokens").value, 1000.0)

    def test_a_relative_cost_saving_converts_through_its_own_dimension(self):
        """A 20% runtime saving is 20% of the RUNTIME reference, never of a
        token count that happened to be written into the same bundle."""
        self.seed_executions()
        bundle = {**_bundle(), "mean_cost": {"solver_runtime_s": 5.0,
                                             "llm_tokens": 1000.0}}
        prediction = self.h.predict_capability_evolution(
            {"operation_type": "induce", "strategy_id": "S04"},
            bundle=bundle, horizon="next 10 tasks", horizon_tasks=10)[1] \
            if False else self.predict(payload={
                "expected_changes": [
                    {"metric": "resource_cost", "unit": "s",
                     "direction": "decrease", "value": -0.2,
                     "value_kind": "relative",
                     "beneficial_direction": "decrease"}],
                "learning_cost": {"solver_runtime_s": 0.1},
                "verification_conditions": [{"condition": "cost falls",
                                             "evaluable": True}]},
                bundle=bundle, horizon_tasks=10)[1]
        result = self.h.compare_capability_evolution(
            [prediction.prediction_id], horizon_tasks=10)
        entry = result["comparisons"][0]
        # 0.2 x 5.0 s = 1.0 s, NOT 0.2 x 1000 tokens.
        self.assertAlmostEqual(entry["saving"]["per_task"], 1.0, places=5)
        self.assertIn("frozen reference", entry["saving"]["converted_from"])

    def test_a_cost_change_without_a_unit_gets_no_reference(self):
        """An ambiguous cost change is not silently handed one dimension's
        yardstick."""
        self.seed_executions()
        prediction = self.h.predict_capability_evolution(
            {"operation_type": "induce", "strategy_id": "S04"},
            bundle=_bundle(), horizon="next 10 tasks", horizon_tasks=10)
        self.assertIsNone(
            prediction.frozen_baseline_for("resource_cost"))

    # -- P1-D: the decision amortizes the one-time cost ------------------

    def test_a_one_time_cost_is_amortized_over_the_declared_window(self):
        """A cost larger than ONE task's saving is still profitable when the
        window pays it back."""
        self.seed_executions()
        prediction = self.h.predict_capability_evolution(
            {"operation_type": "induce", "strategy_id": "S04"},
            bundle=_bundle(), horizon="next 10 tasks", horizon_tasks=10,
            baselines_by_metric=None) if False else self.predict(payload={
                "expected_changes": [
                    {"metric": "resource_cost", "unit": "s",
                     "direction": "decrease", "value": -2.0,
                     "beneficial_direction": "decrease"}],
                "learning_cost": {"solver_runtime_s": 5.0},
                "verification_conditions": [{"condition": "cost falls",
                                             "evaluable": True}]},
                horizon_tasks=10)[1]
        result = self.h.compare_capability_evolution(
            [prediction.prediction_id], horizon_tasks=10)
        self.assertEqual(result["recommendation"], "accept")
        net = result["comparisons"][0]["net_saving"]
        self.assertEqual(net["cumulative_saving"], 20.0)
        self.assertEqual(net["maintenance_cost"], 5.0)
        self.assertEqual(net["net_over_horizon"], 15.0)
        self.assertLess(net["net_per_task"], 0)

    def test_candidates_are_ranked_by_net_saving_over_the_window(self):
        """Ranking uses the CUMULATIVE net, so a large saving with a large
        one-time cost can lose to a smaller saving that pays back sooner."""
        self.seed_executions()
        heavy = self.predict(payload={
            "expected_changes": [
                {"metric": "resource_cost", "unit": "s",
                 "direction": "decrease", "value": -20.0,
                 "beneficial_direction": "decrease"}],
            "learning_cost": {"solver_runtime_s": 10.0},
            "verification_conditions": [{"condition": "cost falls",
                                         "evaluable": True}]},
            horizon_tasks=10)[1]
        light = self.predict(payload={
            "expected_changes": [
                {"metric": "resource_cost", "unit": "s",
                 "direction": "decrease", "value": -20.0,
                 "beneficial_direction": "decrease"}],
            "learning_cost": {"solver_runtime_s": 0.1},
            "verification_conditions": [{"condition": "cost falls",
                                         "evaluable": True}]},
            horizon_tasks=10)[1]
        result = self.h.compare_capability_evolution(
            [heavy.prediction_id, light.prediction_id], horizon_tasks=10)
        self.assertEqual(result["selected_prediction_id"],
                         light.prediction_id)
        nets = {e["prediction_id"]: e["net_saving"]["net_over_horizon"]
                for e in result["comparisons"]}
        self.assertEqual(nets[light.prediction_id], 199.9)
        self.assertEqual(nets[heavy.prediction_id], 190.0)

    def test_an_undeclared_window_cannot_amortize_a_one_time_cost(self):
        self.seed_executions()
        prediction = self.predict(payload={
            "expected_changes": [
                {"metric": "resource_cost", "unit": "s",
                 "direction": "decrease", "value": -2.0,
                 "beneficial_direction": "decrease"}],
            "learning_cost": {"solver_runtime_s": 5.0},
            "verification_conditions": [{"condition": "cost falls",
                                         "evaluable": True}]},
            horizon_tasks=None)[1]
        result = self.h.compare_capability_evolution(
            [prediction.prediction_id], horizon_tasks=None)
        self.assertEqual(result["recommendation"], "defer")
        self.assertIn("no observation window was declared",
                      result["incomparable"][0]["reason"])

    # -- P1-E: a prediction's own episode is charged to that episode -----

    def test_a_capability_prediction_is_charged_to_its_own_episode(self):
        h = ORHarness(home=self.home, world_model=StubProvider(
            usage={"prompt_tokens": 10, "completion_tokens": 50}),
            embedding=self.backend)
        self.addCleanup(h.close)
        h.declare_budget("t_ep", {"llm_tokens": 40}, episode_id="ep1")
        for _ in range(3):
            h.predict_capability_evolution(
                {"operation_type": "induce", "strategy_id": "S04"},
                bundle=_bundle(), horizon="next 10 tasks", horizon_tasks=10,
                task_id="t_ep", episode_id="ep1")
        view = h.budget_view("t_ep", episode_id="ep1")
        consumption = view["consumption"]
        self.assertEqual(view["status"], "exceeded")
        self.assertEqual(len(consumption["prediction_call_costs"]), 3)
        self.assertIsNone(consumption["unattributed_prediction_costs"],
                          "an episode-attributed call is not unattributed")
        self.assertEqual(consumption["total_cost"]["llm_tokens"], 150.0)

    def test_a_call_with_no_episode_stays_unattributed(self):
        h = ORHarness(home=self.home, world_model=StubProvider(
            usage={"prompt_tokens": 10, "completion_tokens": 50}),
            embedding=self.backend)
        self.addCleanup(h.close)
        h.declare_budget("t_noep", {"llm_tokens": 40}, episode_id="ep1")
        h.predict_capability_evolution(
            {"operation_type": "induce", "strategy_id": "S04"},
            bundle=_bundle(), horizon="next 10 tasks", horizon_tasks=10,
            task_id="t_noep")
        view = h.budget_view("t_noep", episode_id="ep1")
        consumption = view["consumption"]
        self.assertEqual(len(consumption["prediction_call_costs"]), 0)
        self.assertEqual(len(consumption["unattributed_prediction_costs"]),
                         1)

    # -- P2-F: a retirement IS a knowledge change ------------------------

    def test_a_successful_retirement_counts_as_a_knowledge_change(self):
        self.seed_executions()
        self.h.induce(strategy_id="S04")
        entries = self.h.sbank.list()
        self.assertTrue(entries)
        entry_id = entries[0].entry_id
        prediction = self.predict(
            payload={
                "expected_changes": [
                    {"metric": "resource_cost", "unit": "s",
                     "direction": "decrease", "value": -6.0,
                     "beneficial_direction": "decrease"}],
                "learning_cost": {"solver_runtime_s": 0.1},
                "verification_conditions": [{"condition": "cost falls",
                                             "evaluable": True}]},
            operation={"operation_type": "retire", "strategy_id": "S04",
                       "config": {"target_entry_id": entry_id}})[1]
        accepted = self.h.accept_capability_operation({
            "recommendation": "accept",
            "selected_prediction_id": prediction.prediction_id})
        bound = self.h.bind_capability_maintenance(
            prediction.prediction_id,
            adoption_action_id=accepted["adoption_action_id"])
        binding = bound["binding"]
        self.assertTrue(binding["changed"],
                        "removing an entry IS a knowledge change")
        self.assertEqual(binding["retired_entry_ids"], [entry_id])
        self.assertEqual(binding["created_entry_ids"], [])
        # The effect path stays open rather than short-circuiting.
        result = self.h.evaluate_capability_effect(prediction.prediction_id)
        self.assertNotEqual(result["evaluation"]["state"], "no_change")


# ---------------------------------------------------------------------------
# 9. CLI coverage of the shortest complete chain
# ---------------------------------------------------------------------------


class TestCli(M5Case):

    def _run(self, argv):
        from or_harness import cli
        buffer = io.StringIO()
        old = sys.stdout
        sys.stdout = buffer
        try:
            code = cli.main(["--home", self.home] + argv)
        finally:
            sys.stdout = old
        return code, buffer.getvalue()

    def _configure(self):
        config = os.path.join(self.home, "config.yaml")
        with open(config, "w", encoding="utf-8") as handle:
            handle.write("world_model:\n  base_url: http://127.0.0.1:9/v1\n"
                         "  model: stub\n  api_key: k\n")
        return config

    def test_predict_capability_requires_an_operation(self):
        code, out = self._run([
            "predict-capability", "--operation", '{"strategy_id": "S04"}',
            "--horizon", "next 10 tasks"])
        self.assertEqual(code, 2)
        self.assertIn("operation_type", out)

    def test_compare_capability_requires_ids(self):
        code, out = self._run(["compare-capability", "--predictions", ""])
        self.assertEqual(code, 2)
        self.assertIn("comma-separated", out)

    def test_capability_feedback_is_readable_through_inspect(self):
        """The two-stage feedback view is one of `inspect`'s banks: the
        old standalone command is gone, the capability is not."""
        code, out = self._run(["inspect", "--bank", "capability"])
        self.assertEqual(code, 0)
        self.assertIn("0 capability prediction(s)", out)

    def test_bind_and_evaluate_report_unknown_predictions(self):
        code, out = self._run(["bind-capability", "--prediction", "hp_nope"])
        self.assertEqual(code, 2)
        code, out = self._run(
            ["evaluate-capability", "--prediction", "hp_nope"])
        self.assertEqual(code, 2)
        self.assertIn("unknown capability prediction", out)

    def test_parser_exposes_every_m5_flag(self):
        from or_harness.cli import build_parser
        parser = build_parser()
        expected = {
            "predict-capability": ["--operation", "--task", "--bundle",
                                   "--horizon", "--horizon-tasks",
                                   "--budget", "--task-id", "--episode",
                                   "--timeout"],
            "compare-capability": ["--predictions", "--horizon-tasks",
                                   "--allow-quality-loss"],
            "accept-capability": ["--recommendation", "--prediction",
                                  "--verify", "--note", "--force"],
            "reject-capability": ["--recommendation", "--prediction",
                                  "--reason"],
            "bind-capability": ["--prediction", "--adoption-action"],
            "evaluate-capability": ["--prediction", "--tasks", "--paired",
                                    "--allow-descriptive"],
        }
        for command, flags in expected.items():
            for action in parser._actions:
                if hasattr(action, "choices") and action.choices \
                        and command in action.choices:
                    actual = {opt for sub in
                              action.choices[command]._actions
                              for opt in sub.option_strings}
                    for flag in flags:
                        self.assertIn(flag, actual,
                                      f"{command} is missing {flag}")
                    break
            else:
                self.fail(f"command {command!r} is not in the parser")
        # The two-stage view moved into `inspect` (a query bank), so the
        # capability is still reachable through ONE entry point.
        for action in parser._actions:
            if hasattr(action, "choices") and action.choices \
                    and "inspect" in action.choices:
                inspect_parser = action.choices["inspect"]
                bank = next(sub for sub in inspect_parser._actions
                            if sub.dest == "bank")
                self.assertIn("capability", bank.choices)
                self.assertIn("evaluations", bank.choices)
                self.assertIn("retention", bank.choices)
                break
        else:
            self.fail("command 'inspect' is not in the parser")


# ---------------------------------------------------------------------------
# 8. persistence discipline
# ---------------------------------------------------------------------------


class TestPersistence(M5Case):

    def test_capability_predictions_survive_a_restart(self):
        _, prediction = self.predict()
        reopened = ORHarness(home=self.home, embedding=self.backend)
        self.addCleanup(reopened.close)
        restored = reopened.get_capability_evolution_prediction(
            prediction.prediction_id)
        self.assertIsNotNone(restored)
        self.assertEqual(restored.expected_changes[0].metric, "resource_cost")

    def test_predictions_are_queryable_by_task(self):
        h = ORHarness(home=self.home, world_model=self.provider,
                      embedding=self.backend)
        self.addCleanup(h.close)
        h.predict_capability_evolution(
            {"operation_type": "induce", "strategy_id": "S04"},
            bundle=_bundle(), horizon="next 10 tasks", task_id="t1")
        h.predict_capability_evolution(
            {"operation_type": "induce", "strategy_id": "S04"},
            bundle=_bundle(), horizon="next 10 tasks", task_id="t2")
        self.assertEqual(len(h.capability_evolution_predictions(
            task_id="t1")), 1)
        self.assertEqual(len(h.capability_evolution_predictions()), 2)

    def test_learning_material_marks_missing_executions(self):
        _, prediction = self.predict()
        request = self.provider.requests[0]
        material = request["learning_material"]
        # The bundle's executions were never recorded: the material says so
        # rather than inventing content.
        self.assertFalse(material["executions"][0]["available"])
        self.assertIn("not found", material["executions"][0]["reason"])


if __name__ == "__main__":
    unittest.main()
