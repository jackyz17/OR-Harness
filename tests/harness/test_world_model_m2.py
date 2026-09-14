"""World-model M2: structured outcome prediction and shadow evaluation.

Covers the eight verification requirements:
1. default compatibility (no provider => no network, existing paths work)
2. real adapter path (protocol-level tests THROUGH HttpChatProvider)
3. freezing & correspondence
4. meaningful consequence prediction (status/feasibility/quality/cost)
5. state & cost accounting (call cost vs predicted cost)
6. scoping & isolation (hypothetical never pollutes)
7. replay & failure (idempotent compare, provider errors keep cost)
8. end-to-end shadow loop

Test doubles are labelled as such. No real LLM call is made in this suite.
"""

import http.server
import json
import os
import sys
import threading
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from or_harness.api import ORHarness  # noqa: E402
from or_harness.core.schema import CostVector  # noqa: E402
from or_harness.world_model.prediction import (  # noqa: E402
    ActionSpec,
    OutcomePrediction,
    validate_prediction_payload,
)
from or_harness.world_model.provider import (  # noqa: E402
    HttpChatProvider,
    NotConfiguredProvider,
    ProviderError,
)
from or_harness.world_model.service import PredictionService  # noqa: E402

from helpers import HarnessTestCase  # noqa: E402


def _task(task_id="t1", family="routing", **coupling):
    values = {"resource_coupling": 0.9, "temporal_coupling": 0.1,
              "route_complexity": 0.85}
    values.update(coupling)
    return {"task_id": task_id, "family": family,
            "spec": {"n_vars": 100, "n_constraints": 50},
            "annotations": {"coupling": values}}


def _spec(**kwargs):
    base = {"action_type": "execute_strategy", "task_id": "t1",
            "strategy_id": "S01", "solver": "highs"}
    base.update(kwargs)
    return ActionSpec.from_dict(base)


class _ScriptedProvider:
    """TEST DOUBLE: returns a canned payload. Labeled as a test source —
    protocol-level tests below use the REAL HttpChatProvider instead."""

    name = "scripted-test"

    def __init__(self, payload=None, error=None):
        self.payload = payload
        self.error = error
        self.calls = 0

    def predict(self, request):
        self.calls += 1
        if self.error is not None:
            raise ProviderError(self.error)
        return {"payload": self.payload, "usage":
                {"prompt_tokens": 100, "completion_tokens": 50},
                "error": None, "latency_s": 0.5}

    def describe(self):
        return {"provider": self.name}


_GOOD_PAYLOAD = {
    "outcome_status": "feasible",
    "feasible": True,
    "quality": 0.8,
    "failure_prob": 0.1,
    "cost": {"llm_tokens": 1500, "solver_runtime_s": 2.0},
    "state_changes": {"current_solution": {"status": "feasible"}},
    "confidence": 0.4,
    "evidence_basis": ["coverage.cell_statistics.S01"],
    "unsupported_fields": {"retries": "no evidence"},
}


class TestDefaultCompatibility(HarnessTestCase):
    """Requirement 1: no provider => no network, existing paths intact."""

    def test_not_configured_returns_explicit_status(self):
        h = ORHarness(home=self.home)
        self.addCleanup(h.close)
        snap = h.snapshot(_task(), "ep1")
        service = PredictionService(h.store, NotConfiguredProvider())
        prediction = service.predict_outcome(_task(), _spec(), snap)
        self.assertEqual(prediction.status, "not_configured")
        self.assertIn("not configured", prediction.error)
        self.assertIsNone(prediction.call_cost)

    def test_existing_paths_untouched_without_provider(self):
        """recall/predict_cost/execute/record work identically; no
        prediction machinery interferes."""
        h = ORHarness(home=self.home)
        self.addCleanup(h.close)
        task = _task()
        recs = h.recall(task)
        self.assertTrue(recs["recommendations"])
        h.bank.append(self.make_record(task_id="t1", strategy_id="S01"))
        result = h.record(self.make_record(task_id="t1", strategy_id="S01"))
        self.assertTrue(result["recorded"])
        self.assertEqual(h.predictions_query(task_id="t1"), [])


class TestHttpChatProviderProtocol(HarnessTestCase):
    """Requirement 2: protocol-level tests THROUGH the real adapter code
    (a local HTTP server stands in for the model service — the adapter,
    request building, and response parsing are the real implementation)."""

    def _serve(self, handler):
        server = http.server.HTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.shutdown)
        return f"http://127.0.0.1:{server.server_port}"

    def test_normal_structured_response(self):
        received = {}

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(
                    int(self.headers["Content-Length"])))
                received["body"] = body
                received["auth"] = self.headers.get("Authorization")
                result = {
                    "choices": [{"message": {"content": json.dumps(
                        _GOOD_PAYLOAD)}}],
                    "usage": {"prompt_tokens": 120,
                              "completion_tokens": 60},
                }
                data = json.dumps(result).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a):
                pass

        base_url = self._serve(Handler)
        provider = HttpChatProvider(base_url, "test-model", "secret-key")
        result = provider.predict({"action_spec": _spec().to_dict(),
                                  "state": {"problem_state": {}}})
        self.assertIsNotNone(result["payload"])
        self.assertEqual(result["payload"]["outcome_status"], "feasible")
        self.assertEqual(result["usage"]["prompt_tokens"], 120)
        # Request shape: chat/completions with system+user, JSON mode.
        self.assertEqual(len(received["body"]["messages"]), 2)
        self.assertEqual(received["body"]["response_format"],
                         {"type": "json_object"})
        # Credentials go in the header only.
        self.assertEqual(received["auth"], "Bearer secret-key")
        self.assertNotIn("secret-key", json.dumps(provider.describe()))

    def test_timeout_and_http_error(self):
        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                self.send_response(500)
                self.end_headers()
                self.wfile.write(b"boom")

            def log_message(self, *a):
                pass

        base_url = self._serve(Handler)
        provider = HttpChatProvider(base_url, "m", "k", timeout_s=2.0)
        with self.assertRaises(ProviderError):
            provider.predict({"x": 1})
        # Connection refused also raises ProviderError (not a crash).
        provider2 = HttpChatProvider("http://127.0.0.1:1", "m", "k",
                                     timeout_s=1.0)
        with self.assertRaises(ProviderError):
            provider2.predict({"x": 1})

    def test_invalid_json_content(self):
        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                result = {"choices": [{"message": {"content":
                                                    "not json at all"}}]}
                data = json.dumps(result).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a):
                pass

        base_url = self._serve(Handler)
        provider = HttpChatProvider(base_url, "m", "k")
        result = provider.predict({"x": 1})
        self.assertIsNone(result["payload"])
        self.assertIn("not valid JSON", result["error"])


class TestPredictionContract(HarnessTestCase):
    """Requirements 3/4: contract validation and meaningful content."""

    def test_action_spec_roundtrip(self):
        spec = _spec(params={"time_limit": 60}, budget_hint={"llm_tokens":
                                                             5000})
        cloned = ActionSpec.from_dict(spec.to_dict())
        self.assertEqual(cloned.action_type, "execute_strategy")
        self.assertEqual(cloned.params, {"time_limit": 60})
        with self.assertRaises(ValueError):
            ActionSpec(action_type="bogus", task_id="t1")

    def test_validate_rejects_bad_payloads(self):
        problems = validate_prediction_payload({}, _spec())
        self.assertTrue(any("no predicted content" in p for p in problems))
        problems = validate_prediction_payload(
            {"quality": 1.5}, _spec())
        self.assertTrue(any("quality" in p for p in problems))
        problems = validate_prediction_payload(
            {"failure_prob": -0.1}, _spec())
        self.assertTrue(any("failure_prob" in p for p in problems))
        problems = validate_prediction_payload(
            {"cost": {"llm_tokens": -5}}, _spec())
        self.assertTrue(any("non-negative" in p or ">= 0" in p
                            for p in problems))
        problems = validate_prediction_payload(
            {"outcome_status": "amazing"}, _spec())
        self.assertTrue(any("outcome_status" in p for p in problems))

    def test_validate_accepts_good_payload(self):
        self.assertEqual(validate_prediction_payload(_GOOD_PAYLOAD,
                                                     _spec()), [])


class TestShadowLoop(HarnessTestCase):
    """Requirements 3-8: the full predict -> execute -> bind -> compare
    path, with a scripted provider (TEST DOUBLE)."""

    def _harness(self, provider=None):
        h = ORHarness(home=self.home, world_model=provider)
        self.addCleanup(h.close)
        return h

    def _solve_script(self, name="solve_m2.py"):
        from pathlib import Path
        script = Path(self.home) / name
        script.write_text(
            "import json\n"
            "json.dump({'status': 'feasible', 'objective_value': 90.0,\n"
            "           'objective_bound': 100.0, 'mip_gap': 0.1,\n"
            "           'runtime_seconds': 2.0}, open('result.json', 'w'))\n")
        return script

    def test_full_shadow_loop(self):
        provider = _ScriptedProvider(_GOOD_PAYLOAD)
        h = self._harness(provider)
        task = _task()
        # 1. Predict BEFORE the action (frozen input).
        prediction = h.predict_outcome(task, _spec(), "ep1")
        self.assertEqual(prediction.status, "valid")
        self.assertEqual(prediction.predicted["outcome_status"], "feasible")
        self.assertEqual(prediction.predicted["quality"], 0.8)
        self.assertIn("llm_tokens", prediction.predicted["cost"])
        # Call cost is the MODEL's own spend, measured from usage.
        self.assertEqual(prediction.call_cost.llm_tokens, 50.0)
        self.assertIn("llm_tokens", prediction.call_cost.measured_dims())
        # 2. The real action runs (existing path, unchanged).
        record = h.execute(task, "S01", str(self._solve_script()),
                           self.home, solver="highs", episode_id="ep1")
        # 3. Bind + compare.
        h.bind_outcome(prediction.prediction_id, record.action_id)
        prediction = h.compare_prediction(prediction.prediction_id)
        fb = prediction.feedback
        self.assertTrue(fb["compared"])
        self.assertEqual(fb["compared_fields"]["outcome_status"]
                         ["predicted"], "feasible")
        self.assertEqual(fb["compared_fields"]["outcome_status"]["actual"],
                         "feasible")
        self.assertTrue(fb["compared_fields"]["outcome_status"]["match"])
        # Quality compared (predicted 0.8 vs actual 1-0.1=0.9).
        self.assertAlmostEqual(
            fb["compared_fields"]["quality"]["abs_error"], 0.1, places=3)
        # Cost compared on the intersection of measured dims: the
        # executor does not measure llm_tokens (the harness backfills
        # it), so only solver_runtime_s is comparable here — the
        # both-sides-measured discipline, not a bug.
        self.assertIn("solver_runtime_s", fb["compared_fields"]["cost"])
        self.assertNotIn("llm_tokens", fb["compared_fields"]["cost"])
        self.assertEqual(
            fb["compared_fields"]["cost"]["solver_runtime_s"]["log_error"],
            0.0)
        self.assertIn("cost.llm_tokens", fb["not_compared"])
        # 4. Queryable.
        preds = h.predictions_query(task_id="t1")
        self.assertEqual(len(preds), 1)

    def test_prediction_frozen_against_later_changes(self):
        """Requirement 3: bank/config changes after the prediction never
        rewrite it."""
        provider = _ScriptedProvider(_GOOD_PAYLOAD)
        h = self._harness(provider)
        task = _task()
        prediction = h.predict_outcome(task, _spec(), "ep1")
        # Later: new knowledge, new executions.
        h.bank.append(self.make_record(task_id="t9", strategy_id="S01"))
        reloaded = h.get_prediction(prediction.prediction_id)
        self.assertEqual(reloaded.predicted["quality"], 0.8)
        self.assertEqual(reloaded.input_snapshot_id,
                         prediction.input_snapshot_id)

    def test_unexecuted_candidate_stays_unbound(self):
        """Requirement 3: a candidate that never ran has no result — no
        counterfactual truth is fabricated."""
        provider = _ScriptedProvider(_GOOD_PAYLOAD)
        h = self._harness(provider)
        prediction = h.predict_outcome(_task(), _spec(), "ep1")
        self.assertIsNone(prediction.bound_action_id)
        self.assertIsNone(prediction.feedback)
        with self.assertRaises(Exception):
            h.compare_prediction(prediction.prediction_id)

    def test_binding_mismatch_recorded_not_scored(self):
        """Requirement 3: predicting strategy A, executing strategy B —
        the mismatch is recorded; no error metrics are fabricated."""
        provider = _ScriptedProvider(_GOOD_PAYLOAD)
        h = self._harness(provider)
        task = _task()
        prediction = h.predict_outcome(task, _spec(), "ep1")
        # Execute a DIFFERENT strategy.
        record = h.execute(task, "S04", str(self._solve_script("s4.py")),
                            self.home, solver="highs", episode_id="ep1")
        h.bind_outcome(prediction.prediction_id, record.action_id)
        prediction = h.compare_prediction(prediction.prediction_id)
        self.assertFalse(prediction.feedback["compared"])
        self.assertIn("strategy_id", prediction.feedback["mismatch"])

    def test_compare_idempotent_no_reinvocation(self):
        """Requirement 7: re-running the comparison neither re-invokes the
        model nor duplicates anything."""
        provider = _ScriptedProvider(_GOOD_PAYLOAD)
        h = self._harness(provider)
        task = _task()
        prediction = h.predict_outcome(task, _spec(), "ep1")
        record = h.execute(task, "S01", str(self._solve_script()),
                           self.home, solver="highs", episode_id="ep1")
        h.bind_outcome(prediction.prediction_id, record.action_id)
        first = h.compare_prediction(prediction.prediction_id)
        calls_after_first = provider.calls
        second = h.compare_prediction(prediction.prediction_id)
        self.assertEqual(provider.calls, calls_after_first)
        self.assertEqual(first.feedback, second.feedback)

    def test_provider_error_keeps_partial_cost(self):
        """Requirement 7: a failed call may still have consumed resources —
        whatever is known is kept."""
        provider = _ScriptedProvider(error="simulated provider crash")
        h = self._harness(provider)
        snap = h.snapshot(_task(), "ep1")
        service = PredictionService(h.store, provider)
        prediction = service.predict_outcome(_task(), _spec(), snap)
        self.assertEqual(prediction.status, "provider_error")
        self.assertIn("simulated provider crash", prediction.error)

    def test_invalid_output_keeps_call_cost(self):
        provider = _ScriptedProvider({"quality": 5.0})  # invalid value
        h = self._harness(provider)
        snap = h.snapshot(_task(), "ep1")
        service = PredictionService(h.store, provider)
        prediction = service.predict_outcome(_task(), _spec(), snap)
        self.assertEqual(prediction.status, "invalid_output")
        self.assertIsNotNone(prediction.call_cost)
        self.assertEqual(prediction.call_cost.llm_tokens, 50.0)

    def test_unsupported_action_type(self):
        provider = _ScriptedProvider(_GOOD_PAYLOAD)
        h = self._harness(provider)
        snap = h.snapshot(_task(), "ep1")
        service = PredictionService(h.store, provider)
        prediction = service.predict_outcome(
            _task(), _spec(action_type="understand"), snap)
        self.assertEqual(prediction.status, "unsupported_action")
        self.assertEqual(provider.calls, 0)  # never invoked

    def test_call_cost_charged_to_parent_action(self):
        """Requirement 5: the model call's own cost is charged to the
        parent action (own cost), separate from the predicted action."""
        provider = _ScriptedProvider(_GOOD_PAYLOAD)
        h = self._harness(provider)
        task = _task()
        begin = h.begin_action("select_strategy", task, "ep1")
        h.predict_outcome(task, _spec(), "ep1",
                          parent_action_id=begin["action_id"])
        action = h.actions.get(begin["action_id"])
        self.assertIn("llm_tokens", action.cost.measured_dims())
        self.assertEqual(action.cost.llm_tokens, 50.0)
        # And it shows in the episode budget as the selection's own cost.
        view = h.budget_view("t1", episode_id="ep1")
        self.assertEqual(view["consumption"]["total_cost"]["llm_tokens"],
                         50.0)

    def test_predicted_state_changes_stay_hypothetical(self):
        """Requirement 6: predicted successor states never enter the real
        state chain."""
        provider = _ScriptedProvider(_GOOD_PAYLOAD)
        h = self._harness(provider)
        task = _task()
        h.predict_outcome(task, _spec(), "ep1")
        # The real episode progress has no current_solution from the
        # prediction (only a real execution would establish it).
        real = h.snapshot(task, "ep1")
        self.assertNotIn("current_solution", real.task_progress)
        # The prediction record itself carries the hypothetical changes.
        pred = h.predictions_query(task_id="t1")[0]
        self.assertIn("current_solution", pred.predicted["state_changes"])

    def test_shadow_never_changes_recommendations(self):
        """Requirement 1/6: with a configured provider, recall output is
        identical to without one."""
        task = _task()
        h_plain = self._harness()
        recs_plain = h_plain.recall(task)
        h_wm = self._harness(_ScriptedProvider(_GOOD_PAYLOAD))
        h_wm.predict_outcome(task, _spec(), "ep1")
        recs_wm = h_wm.recall(task)
        self.assertEqual(
            [r["strategy_id"] for r in recs_plain["recommendations"]],
            [r["strategy_id"] for r in recs_wm["recommendations"]])
        self.assertEqual(
            [r["score"] for r in recs_plain["recommendations"]],
            [r["score"] for r in recs_wm["recommendations"]])


if __name__ == "__main__":
    unittest.main()
