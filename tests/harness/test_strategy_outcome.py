"""World-model M3 tests: the strategy-outcome prediction service.

What this file asserts, one behaviour per test class:

1. the three pre-fixes hold: current prediction and planning really carry
   the retrieval evidence; an explicit-CIR context builds and is reused
   through the SAME CIR (with conflicts refused); the historical path
   refuses an external recall result and does not re-derive knowledge
   targets from today's bank;
2. one comparison gathers ONE context; candidates keep their own config
   and identity; later bank writes do not change the saved request;
3. the new protocol really reaches the provider (request shape, fixed
   identity, content not ids) and parses into the new contract; every
   failure state (not configured, empty payload, invalid JSON, NaN,
   out-of-range probability, wrong type) is honest;
4. an unexecuted strategy window can be validly predicted but not scored;
   unknown cost is never free; a business objective is not execution cost;
   a model self-report is never relabeled as measured;
5. predictions change the suggestion, only an explicit choice changes the
   selection; same strategy_id with different configs does not cross-bind;
   the fallback stays available and explains itself;
6. real call costs include failed calls and are counted once; budget and
   call limits really bind; one real execution binds; unexecuted
   candidates generate no facts.
"""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tests.harness.helpers import HarnessTestCase  # noqa: E402

from or_harness.api import ORHarness  # noqa: E402
from or_harness.strategy.embedding_index import (  # noqa: E402
    LocalHashEmbeddingBackend,
)
from or_harness.world_model.contracts import (  # noqa: E402
    validate_strategy_outcome,
)
from or_harness.world_model.prediction import ActionSpec  # noqa: E402
from or_harness.world_model.provider import WorldModelProvider  # noqa: E402

EMBEDDING_ENV_KEYS = ("OR_EMBEDDING_BACKEND", "OR_EMBEDDING_BASE_URL",
                      "OR_EMBEDDING_MODEL", "OR_EMBEDDING_API_KEY")

REQ_TEXT = ("A distribution centre must be loaded before the delivery window "
            "opens; demand 100 units may not be deferred.")


def _task(task_id="t1", **coupling):
    values = {"resource_coupling": 0.3, "temporal_coupling": 0.1,
              "route_complexity": 0.2}
    values.update(coupling)
    return {"task_id": task_id, "family": "routing",
            "description": REQ_TEXT,
            "spec": {"n_vars": 100, "n_constraints": 50, "n_int_vars": 100},
            "annotations": {"coupling": {**values, "semantic_coupling": 0.5}}}


def _cir():
    return {
        "entities": [{"name": "depot", "kind": "site"},
                     {"name": "route", "kind": "route"}],
        "decisions": [{"name": "load", "kind": "binary", "indexes": ["t"]}],
        "constraints": [{"id": "C1", "kind": "capacity",
                         "expr": "load[t] <= cap"}],
        "relations": [{"source": "load", "target": "depot",
                       "type": "uses_resource", "evidence": "declared"}],
        "coupling_groups": [], "issues": [],
    }


#: A well-formed wm-so/1 payload: benefit with baseline, cost on two dims,
#: two risk events, uncertainty as a self-report.
GOOD_PAYLOAD = {
    "benefit": {"kind": "solution_quality",
                "metric": "normalized_objective_gap", "unit": "1-gap",
                "value": 0.8,
                "baseline": {"kind": "conditional_stats", "value": 0.7}},
    "cost": {"llm_tokens": 1200, "solver_runtime_s": 3.0},
    "risk": {"events": [
        {"event": "model_invalid", "probability": 0.2,
         "severity": 1.0, "severity_unit": "attempts",
         "basis": "no verified knowledge for this cell"},
        {"event": "budget_exhausted", "probability": 0.05},
    ]},
    "uncertainty": {"execution_randomness": 0.3, "knowledge_gap": 0.6},
    "evidence_basis": ["retrieval_evidence.hits[0]"],
}


class StubProvider(WorldModelProvider):
    """A stub provider returning a fixed payload, recording requests."""

    name = "stub-strategy-outcome"

    def __init__(self, payload=GOOD_PAYLOAD, usage=None):
        self.payload = payload
        self.usage = usage if usage is not None else {
            "prompt_tokens": 100, "completion_tokens": 50}
        self.requests = []

    def predict(self, request, timeout_s=None):
        self.requests.append(request)
        return {"payload": self.payload, "usage": self.usage,
                "error": None, "latency_s": 0.02}


class StrategyCase(HarnessTestCase):
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

    def solve(self, task, strategy="S04", objective=100.0):
        from pathlib import Path
        work = Path(self.home) / f"ws_{task['task_id']}_{strategy}"
        work.mkdir(parents=True, exist_ok=True)
        script = work / "solve.py"
        script.write_text(
            "import json\n"
            "with open('result.json', 'w') as fh:\n"
            "    json.dump({'status': 'optimal', 'objective_value': "
            f"{objective}, 'objective_bound': {objective}, "
            "'runtime_seconds': 0.01}, fh)\n", encoding="utf-8")
        record = self.h.execute(task, strategy, str(script), str(work),
                                solver="highs")
        self.h.record(record)
        return record


# ---------------------------------------------------------------------------
# 1. the three pre-fixes hold end to end
# ---------------------------------------------------------------------------


class TestPreFixesHold(StrategyCase):

    def test_current_prediction_carries_real_retrieval_evidence(self):
        self.solve(_task("t1"))
        prediction = self.h.predict_strategy_outcome(
            _task("t1"),
            {"action_type": "execute_strategy", "strategy_id": "S01"},
            "ep1")
        request = self.provider.requests[0]
        block = request["prediction_context"]
        sem = (block["retrieval_evidence"]["semantic"])
        self.assertEqual(sem["status"], "ok",
                         "the semantic channel must really run")
        exec_hits = [x for x in block["retrieval_evidence"]["hits"]
                     if x["layer"] == "execution_evidence"]
        self.assertTrue(exec_hits,
                        "the recorded execution must reach the provider")
        self.assertEqual(prediction.status, "valid")

    def test_planning_carries_real_retrieval_evidence(self):
        self.solve(_task("t1"))
        self.h.plan_next(
            _task("t1"), "ep1",
            candidates=[ActionSpec("execute_strategy", "t1",
                                   strategy_id="S01")],
            limits={"horizon": 1}, protocol="strategy-outcome")
        request = self.provider.requests[0]
        block = request["prediction_context"]
        self.assertEqual(block["retrieval_evidence"]["semantic"]["status"],
                         "ok")
        exec_hits = [x for x in block["retrieval_evidence"]["hits"]
                     if x["layer"] == "execution_evidence"]
        self.assertTrue(exec_hits)

    def test_explicit_cir_context_builds_and_is_reused(self):
        task = _task("t1")
        ctx = self.h.build_prediction_context(task, "ep1", cir=_cir())
        self.assertEqual(ctx.joint.sources["cir"], "caller_supplied")
        # Reuse through the SAME explicit CIR: the effective inputs match.
        prediction = self.h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S01"}, "ep1",
            context=ctx, cir=_cir())
        self.assertEqual(prediction.status, "valid")
        self.assertEqual(prediction.trace.model_info["prediction_context_id"],
                         ctx.context_id)
        # One provider call, one context.
        self.assertEqual(len(self.provider.requests), 1)

    def test_explicit_cir_context_conflict_is_refused(self):
        task = _task("t1")
        ctx = self.h.build_prediction_context(task, "ep1", cir=_cir())
        # Reusing WITHOUT the CIR: the effective inputs differ -> refused.
        with self.assertRaises(ValueError) as caught:
            self.h.predict_strategy_outcome(
                task, {"action_type": "execute_strategy",
                       "strategy_id": "S01"}, "ep1", context=ctx)
        self.assertIn("effective problem input", str(caught.exception))
        self.assertEqual(self.provider.requests, [])

    def test_historical_path_refuses_external_recall(self):
        task = _task("t1")
        snap = self.h.snapshot(task, "ep1")
        self.solve(_task("t1"))
        current = self.h.recall(task)
        with self.assertRaises(ValueError) as caught:
            self.h.build_prediction_context(task, "ep1", snapshot=snap,
                                            recall_result=current)
        self.assertIn("cannot be proven to belong to the frozen moment",
                      str(caught.exception))

    def test_historical_path_does_not_rederive_knowledge_targets(self):
        task = _task("t1")
        snap = self.h.snapshot(task, "ep1")
        # Evidence arrives AFTER the snapshot; a re-derived proposal would
        # read it.
        self.solve(_task("t1"), strategy="S01")
        ctx = self.h.build_prediction_context(
            task, "ep1", snapshot=snap,
            candidates=[ActionSpec("execute_strategy", "t1",
                                   strategy_id="S01")])
        self.assertEqual(ctx.knowledge_targets, [],
                         "no target is re-derived from today's records")
        self.assertTrue(any("knowledge-target proposal was not saved"
                            in m for m in ctx.missing))


# ---------------------------------------------------------------------------
# 2. one comparison, one context; frozen inputs
# ---------------------------------------------------------------------------


class TestOneContextPerComparison(StrategyCase):

    def test_one_context_shared_by_all_candidates(self):
        plan = self.h.plan_next(
            _task("t1"), "ep1",
            candidates=[ActionSpec("execute_strategy", "t1",
                                   strategy_id="S01"),
                        ActionSpec("execute_strategy", "t1",
                                   strategy_id="S02")],
            limits={"horizon": 1}, protocol="strategy-outcome")
        context_ids = {r["prediction_context"]["context_id"]
                       for r in self.provider.requests}
        self.assertEqual(len(context_ids), 1)
        self.assertEqual(plan["prediction_context_id"],
                         next(iter(context_ids)))

    def test_candidates_keep_their_own_config_and_identity(self):
        specs = [ActionSpec("execute_strategy", "t1", strategy_id="S01",
                            params={"time_limit": 60, "seed": 42}),
                 ActionSpec("execute_strategy", "t1", strategy_id="S01",
                            params={"time_limit": 120})]
        for spec in specs:
            self.h.predict_strategy_outcome(_task("t1"), spec, "ep1")
        sent = [r["candidate"] for r in self.provider.requests]
        self.assertEqual(sent[0]["config"]["time_limit"], 60)
        self.assertEqual(sent[0]["config"]["seed"], 42)
        self.assertEqual(sent[1]["config"]["time_limit"], 120)
        # Same strategy_id, different config: different candidates.
        self.assertNotEqual(sent[0]["config"], sent[1]["config"])

    def test_later_bank_writes_do_not_change_the_saved_request(self):
        task = _task("t1")
        ctx = self.h.build_prediction_context(task, "ep1")
        frozen = json.dumps(ctx.provider_view(), sort_keys=True)
        self.solve(_task("t1"))          # the bank moves on
        again = json.dumps(ctx.provider_view(), sort_keys=True)
        self.assertEqual(frozen, again)


# ---------------------------------------------------------------------------
# 3. the protocol really reaches the provider; honest failure states
# ---------------------------------------------------------------------------


class TestProtocolReachesProvider(StrategyCase):

    def test_request_shape_and_fixed_identity(self):
        self.h.predict_strategy_outcome(
            _task("t1"),
            {"action_type": "execute_strategy", "strategy_id": "S01",
             "solver": "highs", "config": {"time_limit": 60}}, "ep1")
        request = self.provider.requests[0]
        self.assertEqual(request["prediction_protocol"], "wm-so/1")
        self.assertEqual(request["request_kind"],
                         "strategy_outcome_prediction")
        self.assertEqual(request["candidate"]["strategy_id"], "S01")
        self.assertEqual(request["candidate"]["config"]["time_limit"], 60)
        self.assertIn("prediction_context", request)
        self.assertIn("output_contract", request)
        self.assertIn("candidate", request["output_contract"]
                      ["fixed_by_framework"])

    def test_provider_receives_content_not_ids(self):
        task = dict(_task("t1"), coupling=_cir())
        self.solve(_task("t1"))
        self.h.predict_strategy_outcome(
            task,
            {"action_type": "execute_strategy", "strategy_id": "S01"},
            "ep1")
        block = self.provider.requests[0]["prediction_context"]
        joint = block["joint_problem"]
        self.assertIn("distribution centre", joint["text"])
        self.assertTrue(joint["cir"]["relations"])
        hits = block["retrieval_evidence"]["hits"]
        self.assertTrue(any(h["content"].get("task_text_excerpt")
                            for h in hits))
        self.assertIn("sources", block["harness_capability"])

    def test_payload_parses_into_the_new_contract(self):
        prediction = self.h.predict_strategy_outcome(
            _task("t1"),
            {"action_type": "execute_strategy", "strategy_id": "S01"},
            "ep1")
        self.assertEqual(prediction.status, "valid")
        self.assertEqual(prediction.prediction_type, "strategy_outcome")
        self.assertEqual(prediction.benefit.kind, "solution_quality")
        self.assertEqual(prediction.benefit.value, 0.8)
        self.assertIsNotNone(prediction.benefit.baseline)
        self.assertEqual(sorted(prediction.cost.expected.measured_dims()),
                         ["llm_tokens", "solver_runtime_s"])
        self.assertEqual(len(prediction.risk.events), 2)
        self.assertEqual(prediction.uncertainty.source, "model_self_report")
        self.assertEqual(validate_strategy_outcome(prediction), [])
        # Round-trips through storage.
        stored = self.h.get_strategy_outcome_prediction(
            prediction.prediction_id)
        self.assertEqual(stored.to_dict(), prediction.to_dict())

    def test_call_cost_is_recorded_on_the_prediction(self):
        prediction = self.h.predict_strategy_outcome(
            _task("t1"),
            {"action_type": "execute_strategy", "strategy_id": "S01"},
            "ep1")
        self.assertIsNotNone(prediction.trace.call_cost)
        self.assertIn("llm_tokens", prediction.trace.call_cost.measured_dims())

    def test_not_configured_is_honest(self):
        h = ORHarness(home=self.home, embedding=self.backend)
        self.addCleanup(h.close)
        prediction = h.predict_strategy_outcome(
            _task("t1"),
            {"action_type": "execute_strategy", "strategy_id": "S01"},
            "ep1")
        self.assertEqual(prediction.status, "contract_only")
        self.assertFalse(prediction.provider_configured)
        self.assertFalse(prediction.service_available)
        self.assertFalse(prediction.prediction_made)

    def test_empty_payload_is_a_recorded_non_prediction(self):
        provider = StubProvider(payload={})
        h = ORHarness(home=self.home, world_model=provider,
                      embedding=self.backend)
        self.addCleanup(h.close)
        prediction = h.predict_strategy_outcome(
            _task("t1"),
            {"action_type": "execute_strategy", "strategy_id": "S01"},
            "ep1")
        self.assertNotEqual(prediction.status, "valid")
        self.assertFalse(prediction.prediction_made)
        self.assertTrue(any("no predicted content" in n
                            for n in prediction.notes))

    def test_invalid_json_is_invalid_not_repaired(self):
        provider = StubProvider(payload="not-a-object")
        h = ORHarness(home=self.home, world_model=provider,
                      embedding=self.backend)
        self.addCleanup(h.close)
        prediction = h.predict_strategy_outcome(
            _task("t1"),
            {"action_type": "execute_strategy", "strategy_id": "S01"},
            "ep1")
        self.assertEqual(prediction.status, "invalid")
        self.assertTrue(any("not a JSON object" in n
                            for n in prediction.notes))

    def test_nan_and_out_of_range_values_are_rejected(self):
        payload = {
            "benefit": {"kind": "solution_quality", "metric": "q",
                        "value": 1.5,   # out of [0,1]
                        "baseline": {"kind": "declared", "value": 0.5}},
            "cost": {"llm_tokens": float("nan")},
            "risk": {"events": [{"event": "x", "probability": 1.5}]},
        }
        provider = StubProvider(payload=payload)
        h = ORHarness(home=self.home, world_model=provider,
                      embedding=self.backend)
        self.addCleanup(h.close)
        prediction = h.predict_strategy_outcome(
            _task("t1"),
            {"action_type": "execute_strategy", "strategy_id": "S01"},
            "ep1")
        self.assertEqual(prediction.status, "invalid")
        joined = " ".join(prediction.notes)
        self.assertIn("NORMALIZED", joined)
        self.assertIn("cost.llm_tokens", joined)
        self.assertIn("probability", joined)

    def test_provider_error_keeps_its_cost(self):
        class ExplodingProvider(WorldModelProvider):
            name = "exploding"

            def predict(self, request, timeout_s=None):
                return {"payload": None, "usage": {
                    "prompt_tokens": 10, "completion_tokens": 5},
                    "error": "boom", "latency_s": 0.01}

        h = ORHarness(home=self.home, world_model=ExplodingProvider(),
                      embedding=self.backend)
        self.addCleanup(h.close)
        prediction = h.predict_strategy_outcome(
            _task("t1"),
            {"action_type": "execute_strategy", "strategy_id": "S01"},
            "ep1")
        self.assertEqual(prediction.status, "invalid")
        self.assertIsNotNone(prediction.trace.call_cost,
                             "a failed call's spend is real and kept")


# ---------------------------------------------------------------------------
# 4. scope, cost and uncertainty honesty
# ---------------------------------------------------------------------------


class TestScopeAndHonesty(StrategyCase):

    def test_unexecuted_window_predicts_validly_but_not_comparably(self):
        prediction = self.h.predict_strategy_outcome(
            _task("t1"),
            {"action_type": "execute_strategy", "strategy_id": "S01",
             "scope": "strategy_window",
             "window_id": "win::t1::ep1::S01"}, "ep1")
        self.assertEqual(prediction.status, "valid")
        self.assertFalse(prediction.trace.comparable)
        self.assertTrue(prediction.trace.not_comparable_reasons)

    def test_unknown_cost_is_never_free_in_the_comparison(self):
        from or_harness.world_model.planner import (
            PlanLimits,
            score_strategy_outcome_predictions,
        )
        # Two valid predictions: one predicts cost, one does not.
        payload_full = dict(GOOD_PAYLOAD)
        payload_full["cost"] = {"llm_tokens": 1000}
        payload_none = dict(GOOD_PAYLOAD)
        payload_none.pop("cost")
        provider = StubProvider()
        h = ORHarness(home=self.home, world_model=provider,
                      embedding=self.backend)
        self.addCleanup(h.close)
        p_full = h.predict_strategy_outcome(
            _task("t1"), {"action_type": "execute_strategy",
                          "strategy_id": "S01"}, "ep1")
        provider.payload = payload_none
        p_none = h.predict_strategy_outcome(
            _task("t1"), {"action_type": "execute_strategy",
                          "strategy_id": "S02"}, "ep1")
        limits = PlanLimits(alpha=1.0, beta=1.0, gamma=1.0,
                            cost_weights={"llm_tokens": 1.0})
        scores = score_strategy_outcome_predictions([p_full, p_none], limits)
        self.assertIn("cost", scores[1].incomparable,
                      "the cost-less candidate is charged, not excused")
        # The silent candidate must not win on cost by saying nothing.
        self.assertLessEqual(scores[1].utility, scores[0].utility + 1e-9)

    def test_business_objective_is_not_execution_cost(self):
        prediction = self.h.predict_strategy_outcome(
            _task("t1"),
            {"action_type": "execute_strategy", "strategy_id": "S01"},
            "ep1")
        # The predicted cost dimensions are the RESOURCE dimensions only.
        if prediction.cost is not None:
            from or_harness.core.schema import COST_DIMENSIONS
            self.assertTrue(set(prediction.cost.expected.measured_dims())
                            <= set(COST_DIMENSIONS))

    def test_model_self_report_is_never_relabelled(self):
        prediction = self.h.predict_strategy_outcome(
            _task("t1"),
            {"action_type": "execute_strategy", "strategy_id": "S01"},
            "ep1")
        self.assertEqual(prediction.uncertainty.source, "model_self_report")
        self.assertTrue(any("UNCALIBRATED" in n
                            for n in prediction.uncertainty.notes))
        self.assertNotEqual(prediction.uncertainty.source, "measured")


# ---------------------------------------------------------------------------
# 5. predictions drive the suggestion; choices drive the selection
# ---------------------------------------------------------------------------


class TestPredictionDrivesDecision(StrategyCase):

    def _plan(self, payloads):
        """Plan with per-candidate payloads (S01 first, S02 second)."""
        provider = StubProvider()
        h = ORHarness(home=self.home, world_model=provider,
                      embedding=self.backend)
        self.addCleanup(h.close)
        original = provider.predict

        def per_candidate(request, timeout_s=None):
            sid = request["candidate"]["strategy_id"]
            provider.payload = payloads[sid]
            return original(request, timeout_s=timeout_s)
        provider.predict = per_candidate
        plan = h.plan_next(
            _task("t1"), "ep1",
            candidates=[ActionSpec("execute_strategy", "t1",
                                   strategy_id="S01"),
                        ActionSpec("execute_strategy", "t1",
                                   strategy_id="S02")],
            limits={"horizon": 1}, protocol="strategy-outcome")
        return h, provider, plan

    def test_prediction_changes_the_suggestion(self):
        # S02 predicts a much better benefit -> the suggestion must flip
        # from the catalog/Selector order to S02.
        better = json.loads(json.dumps(GOOD_PAYLOAD))
        better["benefit"]["value"] = 0.95
        worse = json.loads(json.dumps(GOOD_PAYLOAD))
        worse["benefit"]["value"] = 0.2
        _h, _provider, plan = self._plan({"S01": worse, "S02": better})
        self.assertEqual(plan["status"], "ok")
        self.assertEqual(plan["suggested"]["strategy_id"], "S02")
        self.assertIn("G(0.95)", plan["suggestion_basis"])

    def test_suggestion_is_not_a_selection(self):
        _h, _provider, plan = self._plan(
            {"S01": GOOD_PAYLOAD, "S02": GOOD_PAYLOAD})
        decision_id = plan["decision_action_id"]
        # The decision action lives in THIS plan's harness; read it there.
        action = _h.actions.get(decision_id)
        self.assertEqual(action.action_type, "select_strategy")
        # The decision action's outcome is an EVALUATION, not a selection.
        self.assertEqual(action.outcome["kind"], "plan_next_evaluation")
        # X.selected_plan is NOT written by planning.
        for snap in _h.snapshots(task_id="t1"):
            if snap.episode_id == "ep1":
                self.assertNotIn("selected_plan", snap.task_progress)

    def test_all_predictions_failed_reports_fallback(self):
        provider = StubProvider(payload=None)
        h = ORHarness(home=self.home, world_model=provider,
                      embedding=self.backend)
        self.addCleanup(h.close)
        plan = h.plan_next(
            _task("t1"), "ep1",
            candidates=[ActionSpec("execute_strategy", "t1",
                                   strategy_id="S01")],
            limits={"horizon": 1}, protocol="strategy-outcome")
        self.assertEqual(plan["status"], "no_valid_predictions")
        self.assertIn("Fall back to `recall`", plan["truncation_reason"])
        self.assertIsNone(plan["suggested"])

    def test_horizon_two_is_refused_for_the_new_protocol(self):
        with self.assertRaises(ValueError) as caught:
            self.h.plan_next(
                _task("t1"), "ep1",
                candidates=[ActionSpec("execute_strategy", "t1",
                                       strategy_id="S01")],
                limits={"horizon": 2}, protocol="strategy-outcome")
        self.assertIn("horizon=1 only", str(caught.exception))

    def test_budget_limit_stops_further_calls(self):
        h = ORHarness(home=self.home, world_model=StubProvider(),
                      embedding=self.backend)
        self.addCleanup(h.close)
        # A budget that real consumption has ALREADY exceeded stops the
        # decision before any model call.
        h.declare_budget("t1", {"llm_tokens": 10}, "ep1")
        report = h.report_action("model", _task("t1"), "ep1",
                                 outcome={"kind": "seed"},
                                 cost={"llm_tokens": 100})
        plan = h.plan_next(
            _task("t1"), "ep1",
            candidates=[ActionSpec("execute_strategy", "t1",
                                   strategy_id="S01")],
            limits={"horizon": 1}, protocol="strategy-outcome")
        self.assertEqual(plan["status"], "fallback")
        self.assertEqual(plan["model_calls_made"], 0)

    def test_max_calls_limit_binds(self):
        provider = StubProvider()
        h = ORHarness(home=self.home, world_model=provider,
                      embedding=self.backend)
        self.addCleanup(h.close)
        plan = h.plan_next(
            _task("t1"), "ep1",
            candidates=[ActionSpec("execute_strategy", "t1",
                                   strategy_id="S01"),
                        ActionSpec("execute_strategy", "t1",
                                   strategy_id="S02"),
                        ActionSpec("execute_strategy", "t1",
                                   strategy_id="S04")],
            limits={"horizon": 1, "max_model_calls": 2},
            protocol="strategy-outcome")
        self.assertEqual(plan["model_calls_made"], 2)
        self.assertEqual(plan["status"], "truncated")
        self.assertIn("model-call budget exhausted",
                      plan["truncation_reason"])


# ---------------------------------------------------------------------------
# 6. execution binding and cost accounting
# ---------------------------------------------------------------------------


class TestExecutionBinding(StrategyCase):

    def test_real_execution_binds_and_is_comparable(self):
        task = _task("t1")
        prediction = self.h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S04"}, "ep1")
        record = self.solve(task, strategy="S04")
        bound = self.h.bind_strategy_outcome(
            prediction.prediction_id, record.action_id)
        self.assertIsNone(bound.trace.model_info.get("binding_mismatch"))
        self.assertTrue(bound.trace.comparable)
        # Idempotent re-bind, no extra model call.
        calls = len(self.provider.requests)
        again = self.h.bind_strategy_outcome(
            prediction.prediction_id, record.action_id)
        self.assertEqual(len(self.provider.requests), calls)
        self.assertEqual(again.trace.comparable, bound.trace.comparable)

    def test_different_config_does_not_cross_bind(self):
        task = _task("t1")
        # Predict a candidate with a DIFFERENT solver than the one that ran.
        prediction = self.h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S04", "solver": "cbc"}, "ep1")
        record = self.solve(task, strategy="S04")   # ran with highs
        bound = self.h.bind_strategy_outcome(
            prediction.prediction_id, record.action_id)
        mismatch = bound.trace.model_info.get("binding_mismatch") or {}
        self.assertIn("solver", mismatch)
        self.assertFalse(bound.trace.comparable)

    def test_unexecuted_candidate_generates_no_fact(self):
        task = _task("t1")
        prediction = self.h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S01"}, "ep1")
        record = self.solve(task, strategy="S04")   # a DIFFERENT strategy ran
        executions_before = self.h.bank.count()
        bound = self.h.bind_strategy_outcome(
            prediction.prediction_id, record.action_id)
        mismatch = bound.trace.model_info.get("binding_mismatch") or {}
        self.assertIn("strategy_id", mismatch)
        self.assertFalse(bound.trace.comparable)
        self.assertEqual(self.h.bank.count(), executions_before,
                         "no fact is fabricated for the unexecuted candidate")

    def test_planning_cost_counts_failed_calls_once(self):
        provider = StubProvider(payload=None)   # every call fails
        h = ORHarness(home=self.home, world_model=provider,
                      embedding=self.backend)
        self.addCleanup(h.close)
        plan = h.plan_next(
            _task("t1"), "ep1",
            candidates=[ActionSpec("execute_strategy", "t1",
                                   strategy_id="S01"),
                        ActionSpec("execute_strategy", "t1",
                                   strategy_id="S02")],
            limits={"horizon": 1}, protocol="strategy-outcome")
        self.assertEqual(plan["model_calls_made"], 2)
        cost = plan["planning_cost"]
        self.assertTrue(cost, "failed calls' spend is real and recorded")
        self.assertEqual(cost["cost"]["llm_tokens"], 100.0,
                         "two failed calls of 50 tokens each, counted once")
        # And charged ONCE to the decision action.
        decision = h.actions.get(plan["decision_action_id"])
        self.assertEqual(decision.cost.llm_tokens, 100.0)


class TestCliStrategyCommands(StrategyCase):
    """CLI surface with a real local HTTP stub server (line protocol)."""

    def _run(self, argv):
        from or_harness.cli import main
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = main(["--home", self.home] + argv)
        return code, json.loads(buf.getvalue())

    def _with_http_stub(self, argv):
        """Run a CLI command against a local OpenAI-compatible stub."""
        import threading
        import urllib.request
        from http.server import BaseHTTPRequestHandler, HTTPServer

        seen: List[Dict[str, Any]] = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                seen.append(body)
                payload = {
                    "choices": [{"message": {"content": json.dumps(
                        GOOD_PAYLOAD)}}],
                    "usage": {"prompt_tokens": 100,
                              "completion_tokens": 50},
                }
                data = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):  # silence
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.shutdown)
        self.addCleanup(thread.join, timeout=2)
        url = f"http://127.0.0.1:{server.server_port}"
        code, payload = self._run(["--world-model", f"{url}::stub-model"]
                                  + argv)
        return code, payload, seen

    def test_predict_strategy_command(self):
        code, payload, seen = self._with_http_stub([
            "predict-strategy", "--task", json.dumps(_task("t1")),
            "--candidate", json.dumps({"action_type": "execute_strategy",
                                       "strategy_id": "S01"}),
            "--episode", "ep1"])
        self.assertEqual(code, 0)
        self.assertEqual(payload["result"]["prediction"]["status"], "valid")
        # The wire protocol really is the OpenAI chat shape, and the system
        # prompt selects the strategy-outcome protocol.
        self.assertEqual(seen[0]["model"], "stub-model")
        system = seen[0]["messages"][0]["content"]
        self.assertIn("wm-so/1", system)
        user = json.loads(seen[0]["messages"][1]["content"])
        self.assertEqual(user["prediction_protocol"], "wm-so/1")

    def test_predict_strategy_with_context_reuse(self):
        code, payload = self._run([
            "context", "--task", json.dumps(_task("t1")), "--episode",
            "ep1"])
        self.assertEqual(code, 0)
        ctx_id = payload["result"]["context_id"]
        code, payload, _seen = self._with_http_stub([
            "predict-strategy", "--task", json.dumps(_task("t1")),
            "--candidate", json.dumps({"action_type": "execute_strategy",
                                       "strategy_id": "S01"}),
            "--episode", "ep1", "--context", ctx_id])
        self.assertEqual(code, 0)
        self.assertEqual(payload["result"]["prediction"]["status"],
                         "valid")

    def test_bind_strategy_command(self):
        task = _task("t1")
        prediction = self.h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S04"}, "ep1")
        record = self.solve(task, strategy="S04")
        code, payload = self._run([
            "bind-strategy", "--prediction", prediction.prediction_id,
            "--action", record.action_id])
        self.assertEqual(code, 0)
        self.assertIn("COMPARABLE", payload["summary"])

    def test_plan_next_protocol_flag(self):
        code, payload, seen = self._with_http_stub([
            "plan-next", "--task", json.dumps(_task("t1")),
            "--episode", "ep1",
            "--candidates", json.dumps([
                {"action_type": "execute_strategy", "task_id": "t1",
                 "strategy_id": "S01"}]),
            "--protocol", "strategy-outcome"])
        self.assertEqual(code, 0)
        self.assertEqual(payload["result"]["protocol"], "strategy-outcome")
        self.assertTrue(seen, "the model was really called over HTTP")


if __name__ == "__main__":
    unittest.main()
