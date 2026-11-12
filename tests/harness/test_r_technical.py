"""Single-attempt technical R and common-term strategy comparison."""
import copy
import json
from pathlib import Path

from or_harness.api import ORHarness
from or_harness.execution.executor import SafePythonExecutor
from or_harness.strategy.embedding_index import LocalHashEmbeddingBackend
from or_harness.world_model.contracts import R_OBSERVATION_VERSION
from or_harness.world_model.prediction import ActionSpec
from tests.harness.g_fixtures import GCase, GProvider, G_PAYLOAD, COUPLED


def payload(probability=.2, *, cost=None, diagnostics=None):
    out = copy.deepcopy(G_PAYLOAD)
    out["benefit"]["value"] = .5
    out["cost"] = {"llm_tokens": 0} if cost is None else cost
    out["risk"] = {"events": ([] if probability is None else [
        {"event": "implementation_failure", "probability": probability}]) + (diagnostics or [])}
    return out


class CandidateProvider(GProvider):
    def __init__(self):
        super().__init__()
        self.by_candidate = {}

    def predict(self, request, timeout_s=None):
        if self.by_candidate:
            self.payload = copy.deepcopy(self.by_candidate[request["candidate"]["strategy_id"]])
        return super().predict(request, timeout_s)


class RCase(GCase):
    def setUp(self):
        super().setUp()
        self.provider = CandidateProvider()
        self.provider.payload = payload()
        self.h = ORHarness(home=self.home, world_model=self.provider,
            embedding=LocalHashEmbeddingBackend(), execution_context={
                "host": "test-runner", "model": "execution-model/v1", "agent_id": "agent/v1",
                "solver_interface": "highs", "phase": "initial_modelling"})
        self.addCleanup(self.h.close)

    def predict_r(self, *, episode="ep1", probability=.2, omit_r=False, scope="attempt"):
        self.provider.payload = payload(probability)
        if omit_r:
            self.provider.payload.pop("risk")
        candidate = {"action_type": "execute_strategy", "strategy_id": "S04", "solver": "highs", "method": COUPLED, "scope": scope}
        if scope == "strategy_window":
            candidate["window_id"] = "win::" + self.task["task_id"] + "::" + episode + "::S04"
        return self.h.predict_strategy_outcome(self.task, candidate, episode)

    def body(self, prediction=None, *, text=None, episode="ep1"):
        if prediction:
            episode = prediction.candidate.episode_id
        root = Path(self.home) / (prediction.prediction_id if prediction else "unpredicted-" + episode)
        root.mkdir(exist_ok=True)
        (root / "solve.py").write_text(text or "import json\njson.dump({'status':'optimal','objective_value':1},open('result.json','w'))\n")
        r = self.h.execute(self.task, "S04", str(root / "solve.py"), str(root), solver="highs", episode_id=episode,
                           prediction_id=prediction.prediction_id if prediction else None)
        self.h.record(r)
        return r

    def fact(self, evaluation):
        return evaluation["risk"]["implementation_observation"]


class TestTechnicalRLabels(RCase):
    def test_value_error_network_flow_is_code_failure(self):
        p = self.predict_r()
        r = self.body(p, text="raise ValueError('invalid network flow indices')\n")
        self.assertEqual(self.h.bank.get(r.execution_id).failures[0].error_class, "model")
        e = self.close_g()["evaluations"][0]
        self.assertEqual(self.fact(e)["label"], 1)
        self.assertEqual(self.fact(e)["exception_type"], "ValueError")
        scored = next(s for s in e["risk"]["scored"] if s["event"] == "implementation_failure")
        self.assertEqual(scored["observation_rule_version"], R_OBSERVATION_VERSION)
        self.assertAlmostEqual(scored["brier"], .64)

    def test_optimal_without_objective_is_protocol_failure(self):
        p = self.predict_r()
        r = self.body(p, text="import json\njson.dump({'status':'optimal'},open('result.json','w'))\n")
        e = self.close_g()["evaluations"][0]
        self.assertEqual(self.fact(e)["label"], 1)
        self.assertIn("feasible status without objective_value", self.fact(e)["protocol_problems"])
        self.assertEqual(r.quality["status"], "optimal")

    def test_malformed_result_and_syntax_are_technical_failures(self):
        for i, text in enumerate(("open('result.json','w').write('[]')\n", "if True print('bad')\n")):
            with self.subTest(text=text):
                p = self.predict_r(episode="malformed-" + str(i))
                self.body(p, text=text)
                e = self.close_g(episode=p.candidate.episode_id)["evaluations"][0]
                self.assertEqual(self.fact(e)["label"], 1)

    def test_legal_infeasible_is_technical_success(self):
        p = self.predict_r()
        self.body(p, text="import json\njson.dump({'status':'infeasible'},open('result.json','w'))\n")
        e = self.close_g()["evaluations"][0]
        self.assertEqual(self.fact(e)["label"], 0)
        self.assertEqual(e["risk"]["observed_units"]["solver_reported_infeasible"]["label"], "occurred")

    def test_valid_delivery_after_a_caught_exception_is_technical_success(self):
        p = self.predict_r()
        self.body(p, text="import json, sys\ntry:\n    raise ValueError('bad network flow indices')\nexcept ValueError as exc:\n    print('ValueError: ' + str(exc), file=sys.stderr)\njson.dump({'status':'optimal','objective_value':1},open('result.json','w'))\n")
        e = self.close_g()["evaluations"][0]
        self.assertEqual(self.fact(e)["label"], 0)
        self.assertEqual(self.fact(e)["exit_code"], 0)

    def test_environment_blocked_is_unknown_even_with_probability(self):
        p = self.predict_r()
        self.body(p, text="import os\nos.system('echo blocked')\n")
        e = self.close_g()["evaluations"][0]
        self.assertIsNone(self.fact(e)["label"])
        c = self.h.calibration_summary()["technical_r"]["coverage"]
        self.assertEqual(c, {"n_actual_attempts": 1, "n_known_labels": 0,
                             "n_reported_probabilities": 1, "n_scorable": 0})

    def test_normal_timeout_is_unknown(self):
        self.h.executor = SafePythonExecutor(timeout_seconds=1)
        p = self.predict_r()
        self.body(p, text="import time\ntime.sleep(2)\n")
        e = self.close_g()["evaluations"][0]
        self.assertIsNone(self.fact(e)["label"])
        self.assertEqual(e["risk"]["observed_units"]["timeout"]["label"], "occurred")

    def test_only_a_sourced_code_timeout_assessment_makes_a_positive(self):
        self.h.executor = SafePythonExecutor(timeout_seconds=1)
        p = self.predict_r()
        r = self.body(p, text="import sys\nprint('unexpected busy loop',file=sys.stderr,flush=True)\nwhile True: pass\n")
        self.h.bank.annotate_features(r.execution_id, {"technical_failure_evidence": {
            "kind": "code_caused_timeout", "source": "test source/trace inspection",
            "assessment": "The captured submitted script enters an unconditional busy loop after the captured trace.",
            "evidence_refs": ["artifacts.submitted_script", "artifacts.stderr_clip"]}})
        self.assertEqual(self.fact(self.close_g()["evaluations"][0])["label"], 1)

    def test_failure_and_repair_have_separate_labels_even_for_window_predictions(self):
        a = self.predict_r(scope="strategy_window")
        first = self.body(a, text="raise TypeError('bad API argument')\n")
        b = self.predict_r(scope="strategy_window")
        second = self.body(b)
        result = self.close_g()
        by_id = {self.fact(e)["execution_id"]: self.fact(e)["label"] for e in result["evaluations"]}
        self.assertEqual(by_id, {first.execution_id: 1, second.execution_id: 0})
        coverage = self.h.calibration_summary()["technical_r"]["coverage"]
        self.assertEqual(coverage, {"n_actual_attempts": 2, "n_known_labels": 2,
                                    "n_reported_probabilities": 2, "n_scorable": 2})
        self.close_g()
        self.assertEqual(self.h.calibration_summary()["technical_r"]["coverage"], coverage)

    def test_business_failure_keeps_semantic_evidence_and_technical_success(self):
        p = self.predict_r()
        r = self.run_g(p, split=True)
        self.check_g(r, fidelity="deviated", failure="implementation")
        e = self.close_g()["evaluations"][0]
        self.assertEqual(self.fact(e)["label"], 0)
        self.assertNotEqual(e["benefit"].get("label_eligibility"), "attributable")

    def test_legacy_prediction_errors_keep_the_legacy_rule(self):
        p = self.predict_r()
        p.trace.model_info.pop("risk_convention")
        self.h.strategy_predictions._save(p)  # simulate an archived pre-rule prediction
        self.body(p, text="import json\njson.dump({'status':'optimal'},open('result.json','w'))\n")
        e = self.close_g()["evaluations"][0]
        self.assertNotIn("implementation_observation", e["risk"])
        old = next(s for s in e["risk"]["scored"] if s["event"] == "implementation_failure")
        self.assertEqual(old["label"], "not_occurred")
        self.assertIsNone(old["observation_rule_version"])
        self.assertEqual(self.h.calibration_summary()["technical_r"]["coverage"]["n_scorable"], 0)


class TestRFeedback(RCase):
    def test_execution_labels_deduplicate_and_raw_probabilities_are_retained(self):
        a = self.predict_r(probability=.123456789)
        b = self.predict_r(probability=.2)
        r = self.body(a)
        self.h.bind_strategy_outcome(b.prediction_id, r.action_id)
        result = self.close_g()
        e = next(e for e in result["evaluations"] if e["prediction_id"] == a.prediction_id)
        scored = next(s for s in e["risk"]["scored"] if s["event"] == "implementation_failure")
        self.assertEqual(scored["predicted_probability"], .123456789)
        stats = self.h.calibration_summary()["technical_r"]
        self.assertEqual(stats["coverage"], {"n_actual_attempts": 1, "n_known_labels": 1,
            "n_reported_probabilities": 2, "n_scorable": 2})
        self.assertEqual(stats["groups"][0]["n_scored_distinct_episodes"], 1)
        self.assertAlmostEqual(stats["groups"][0]["mean_brier"], (.123456789 ** 2 + .2 ** 2) / 2)

    def test_later_offline_execution_is_outside_the_published_episode(self):
        self.body(episode="past")
        self.close_g("past")
        self.body(episode="past", text="raise ValueError('later network flow failure')\n")
        stats = self.h.calibration_summary()["technical_r"]
        self.assertEqual(stats["coverage"]["n_actual_attempts"], 1)
        self.assertEqual(stats["groups"][0]["technical_successes"], 1)

    def test_frozen_context_keeps_only_history_visible_at_its_decision(self):
        self.body(episode="past")
        self.close_g("past")
        ctx = self.h.build_prediction_context(self.task, "frozen")
        self.body(episode="later", text="raise ValueError('network flow failure')\n")
        self.close_g("later")
        self.h.predict_strategy_outcome(self.task, {"action_type": "execute_strategy", "strategy_id": "S04",
            "solver": "highs", "method": COUPLED}, "frozen", context=ctx)
        group = self.provider.requests[-1]["prediction_context"]["implementation_r_evidence"]["groups"][0]
        self.assertEqual((group["n_attempts"], group["technical_failures"], group["technical_successes"]), (1, 0, 1))

    def test_unpredicted_execution_supplies_first_request_history_and_reusable_script(self):
        r = self.body(episode="unpredicted")
        self.close_g("unpredicted")
        stats = self.h.calibration_summary()["technical_r"]
        self.assertEqual(stats["coverage"], {"n_actual_attempts": 1, "n_known_labels": 1,
            "n_reported_probabilities": 0, "n_scorable": 0})
        self.task = dict(self.task, task_id="new-task")
        self.predict_r(episode="first")
        evidence = self.provider.requests[-1]["prediction_context"]["implementation_r_evidence"]
        self.assertEqual(len(evidence["groups"]), 1)
        g = evidence["groups"][0]
        self.assertEqual(g["technical_successes"], 1)
        self.assertEqual(g["reusable_scripts"][0]["execution_id"], r.execution_id)
        self.assertTrue(g["reusable_scripts"][0]["artifact"]["sha256"])
        self.assertNotIn("technical_r", self.provider.requests[-1]["prediction_context"]["strategy_outcome_calibration"])

    def test_omitted_r_still_counts_labels_and_future_open_attempts_do_not_leak(self):
        p = self.predict_r(omit_r=True)
        self.body(p)
        self.close_g()
        p2 = self.predict_r(episode="open")
        self.body(p2, text="raise ValueError('bad network flow')\n")
        self.predict_r(episode="next")
        g = self.provider.requests[-1]["prediction_context"]["implementation_r_evidence"]["groups"][0]
        self.assertEqual((g["n_attempts"], g["technical_failures"], g["technical_successes"]), (1, 0, 1))
        self.assertNotIn("mean_brier", g)
        self.assertEqual(self.h.calibration_summary()["technical_r"]["coverage"]["n_reported_probabilities"], 0)

    def test_other_agent_interface_and_missing_identity_have_no_prior(self):
        self.body(episode="history")
        self.close_g("history")
        for index, change in enumerate(({"model": "other-model"}, {"agent_id": "other-agent"})):
            original = copy.deepcopy(self.h.execution_context)
            self.h.execution_context.update(change)
            self.predict_r(episode="other" + str(index))
            e = self.provider.requests[-1]["prediction_context"]["implementation_r_evidence"]
            self.assertFalse(e["groups"])
            self.h.execution_context = original
        self.provider.payload = payload()
        self.h.predict_strategy_outcome(self.task, {"action_type": "execute_strategy", "strategy_id": "S04",
            "solver": "pulp", "method": COUPLED}, "other-interface")
        self.assertFalse(self.provider.requests[-1]["prediction_context"]["implementation_r_evidence"]["groups"])
        self.h.execution_context = {}
        self.predict_r(episode="missing")
        self.assertFalse(self.provider.requests[-1]["prediction_context"]["implementation_r_evidence"]["groups"])


class TestCommonCRScoring(RCase):
    def plan(self, a, b, **limits):
        self.provider.by_candidate = {"S04": a, "S06": b}
        result = self.h.plan_next(self.task, "ep1", candidates=[
            ActionSpec("execute_strategy", self.task["task_id"], strategy_id=s, solver="highs", method=COUPLED)
            for s in ("S04", "S06")], limits={"horizon": 1, "alpha": 1, "beta": 1, "gamma": 1,
                "cost_weights": {"llm_tokens": 1, "remaining_latency_s": 1, "solver_runtime_s": 1, "latency_s": 1}, **limits})
        return [row["score"] for row in result["candidates"]]

    def test_only_implementation_failure_is_r_despite_timeout_point_eight(self):
        scores = self.plan(payload(.2, diagnostics=[{"event": "timeout", "probability": .8}]), payload(.2))
        self.assertEqual([s["risk_effective"] for s in scores], [.2, .2])
        self.assertEqual([s["utility"] for s in scores], [.3, .3])

    def test_missing_r_is_omitted_for_everyone(self):
        scores = self.plan(payload(.2), payload(None))
        self.assertEqual([s["utility"] for s in scores], [.5, .5])
        self.assertEqual([s["risk_probability"] for s in scores], [.2, None])
        self.assertTrue(all(s["risk_effective"] is None and "R" not in s["scoring_basis"]["terms"] for s in scores))

    def test_common_tokens_do_not_charge_unknown_time(self):
        a = payload(None, cost={"llm_tokens": 500000, "remaining_latency_s": 120})
        b = payload(None, cost={"llm_tokens": 700000})
        scores = self.plan(a, b)
        self.assertTrue(all(s["scoring_basis"]["cost_dimensions"] == ["llm_tokens"] for s in scores))
        self.assertAlmostEqual(scores[0]["cost_normalized"], 5 / 7, places=6)
        self.assertEqual(scores[1]["cost_normalized"], 1)
        self.assertIsNone(scores[1]["cost_reported"]["values"].get("remaining_latency_s"))

    def test_explicit_zero_in_all_candidates_is_valid(self):
        scores = self.plan(payload(0), payload(0))
        self.assertEqual([s["utility"] for s in scores], [.5, .5])
        self.assertTrue(all(s["scoring_basis"]["terms"] == ["G", "C", "R"] for s in scores))
        self.assertEqual([s["risk_effective"] for s in scores], [0, 0])

    def test_no_common_cost_dimension_is_explicitly_incomparable(self):
        scores = self.plan(payload(None, cost={"latency_s": 2}), payload(None, cost={"remaining_latency_s": 120}))
        self.assertTrue(all(s["cost_normalized"] is None and "cost" in s["incomparable"] for s in scores))
        self.assertEqual([s["utility"] for s in scores], [.5, .5])
        self.assertTrue(all(s["scoring_basis"]["terms"] == ["G"] for s in scores))

    def test_component_deduplication_occurs_after_intersection(self):
        scores = self.plan(payload(None, cost={"remaining_latency_s": 120, "solver_runtime_s": 2}),
                           payload(None, cost={"solver_runtime_s": 3}))
        self.assertTrue(all(s["scoring_basis"]["cost_dimensions"] == ["solver_runtime_s"] for s in scores))
        self.assertAlmostEqual(scores[0]["cost_normalized"], 2 / 3, places=6)

    def test_all_valid_data_uses_common_container_and_keeps_components_visible(self):
        scores = self.plan(payload(.2, cost={"llm_tokens": 50, "remaining_latency_s": 120, "solver_runtime_s": 2}),
                           payload(.3, cost={"llm_tokens": 70, "remaining_latency_s": 240, "solver_runtime_s": 3}))
        self.assertTrue(all(s["scoring_basis"]["cost_dimensions"] == ["llm_tokens", "remaining_latency_s"] for s in scores))
        self.assertTrue(all(s["scoring_basis"]["terms"] == ["G", "C", "R"] for s in scores))
        self.assertTrue(all(not s["scoring_basis"]["partial"] for s in scores))
        self.assertEqual(scores[0]["cost_reported"]["values"]["solver_runtime_s"], 2)

    def test_off_yardstick_candidate_does_not_change_the_common_basis(self):
        a, b = payload(.2, cost={"llm_tokens": 10}), payload(None, cost={"llm_tokens": 1000000})
        b["benefit"].update(kind="solution_quality", metric="normalized_objective_gap", unit="1-gap")
        scores = self.plan(a, b)
        self.assertIsNone(scores[1]["utility"])
        self.assertEqual(scores[0]["scoring_basis"]["cost_scales"]["llm_tokens"], 10)
        self.assertEqual(scores[0]["scoring_basis"]["terms"], ["G", "C", "R"])
        self.assertEqual(len(scores[0]["scoring_basis"]["participant_prediction_ids"]), 1)

    def test_explicit_legacy_cost_scope_is_not_relabelled_as_remaining(self):
        from or_harness.world_model.planner import score_strategy_outcome_predictions, PlanLimits
        a, b = self.predict_r(), self.predict_r(episode="second")
        a.cost.scope = "attempt"  # replay a legacy explicit narrow declaration
        scores = score_strategy_outcome_predictions([a, b], PlanLimits())
        self.assertTrue(all(s.cost_normalized is None for s in scores))
        self.assertTrue(all(s.scoring_basis["omitted_cost_dimensions"]["llm_tokens"]["reason"] ==
                            "different or unknown measurement scopes" for s in scores))

    def test_non_json_r_values_become_unknown_without_losing_g_or_c(self):
        for index, invalid in enumerate((True, "0.2", -1, 1.2, float("nan"), float("inf"), 10 ** 400)):
            with self.subTest(invalid=invalid):
                p = self.predict_r(episode="invalid" + str(index), probability=invalid)
                self.assertEqual(p.status, "valid")
                self.assertIsNone(p.risk.events[0].probability)
                self.assertIsNotNone(p.benefit.value)
                self.assertIsNotNone(p.cost)
                self.assertIn("risk.implementation_failure", p.trace.unsupported_fields)
