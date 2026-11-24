"""Stage 1: provider-only history-feedback compression and request budget.

These tests are about the DELIVERED request view, never about changing the
stored evidence:

- the compact pair must carry the SAME original prediction, real
  observation, error, eligibility and a reference that restores every
  omitted field;
- a missing attribution is never fabricated and an UNKNOWN label never
  becomes a failure or G=0;
- the history block is reduced per-row before per-case, and the full
  request refuses to send when the protected core is over budget;
- the token measurement is a LABELLED conservative estimate that counts
  CJK/JSON-dense text more heavily than the legacy chars/4 default.
"""
from __future__ import annotations

import copy
import json
import os
import unittest

from tests.harness.test_episode_closeout import M4Case, _task
from tests.harness.decision_fixtures import (
    bind_legacy, planned_prediction,
)


def _pair(index, *, task_id="t1", strategy="S04", observed=1.0,
          predicted=0.8, crash=False, status="optimal", attribution=None,
          unknown_label=False, big=False, cjk=False):
    """One FULL stored pair row (the shape ``build_paired_feedback`` stores)."""
    benefit = {
        "metric": "normalized_objective_gap",
        "observation_rule_version": "wm-g-faithful/1",
        "label_eligibility": "attributable",
        "predicted": predicted,
        "observed": observed,
        "signed_error": round(observed - predicted, 6),
        "eligibility": "evaluable",
        "reason": ("the task check ran on real values" if not crash
                   else "no task check is on record for this execution"),
        "acceptance_rule": {
            "checker": "objective-equality/1",
            "fingerprint": "fp-" + str(index),
            "spec": {"big": list(range(80))}},
        "evidence_refs": [f"artifacts/{index}/result", f"artifacts/{index}/log"],
    }
    if attribution is not None:
        benefit["attribution"] = attribution
    if big:
        benefit["notes"] = [
            ("原题要求两个工厂必须同厂且交付窗口不得推迟；候选方法把同厂约束"
             "显式保留，并复用了上一条执行脚本中的路线选择。") if cjk
            else ("the candidate keeps the same-factory constraint and "
                  "reuses the route selection from the previous attempt "
                  "such that the acceptance check can be recomputed."),
            "second prediction note that is not part of the numeric forecast",
            "third prediction note used to test the reference on truncation",
        ]
    row = {
        "evaluation_id": f"ev_{index}",
        "prediction_id": f"sp_{index}",
        "episode_id": f"ep{index}",
        "task_id": task_id,
        "scope": "attempt",
        "action_id": f"ac_{index}",
        "execution_ids": [f"ex_{index}"],
        "conditions": {"task_id": task_id, "strategy_id": strategy,
                       "family": "routing",
                       "cell": "class=mixed|rc[0.25,0.50]|tc[0.00,0.25]"},
        "problem_identity_digest": "digest" + str(index),
        "method_planned": {
            "name": f"coupled method {index}",
            "n_steps": 4,
            "steps": [f"Build the model variant {index}.",
                      f"Add the resource coupling {index}.",
                      f"Solve with the recorded solver {index}.",
                      f"Check the acceptance rule {index}."],
            "why": "this variant respects the same-factory condition",
            "fallback": "fall back to the split formulation",
        },
        "method_actual": {"name": f"coupled method {index}", "n_steps": 4,
                          "steps": [f"Build the model variant {index}.",
                                    f"Add the resource coupling {index}.",
                                    f"Solve with the recorded solver {index}.",
                                    f"Check the acceptance rule {index}."]},
        "benefit": benefit,
        "cost_log_ratio": {"solver_runtime_s": -2.3},
        "cost_measured": {"solver_runtime_s": {"predicted": 3.0,
                                               "actual": 0.01,
                                               "abs_error": 2.99}},
        "cost_observed_fact": {
            "values": {"solver_runtime_s": 0.01, "latency_s": 0.02},
            "dimension_scopes": {"solver_runtime_s":
                                 "recorded_solver_receipts"},
            "lower_bounds": {"remaining_latency_s": {
                "value": 0.5, "scope": "remaining_to_task_end",
                "source_refs": [f"ex_{index}"]}},
            "boundary_exclusions": ["no complete window"],
            "execution_ids": [f"ex_{index}"],
            "conditions": {"family": "routing",
                           "solver_conditions": {"solver": "highs"}},
            "observation_rule_version": "wm-c-window/1",
            "window": {"anchor": 1}, "work_facts": {"host_rounds": 3},
        },
        "risk_predicted": [
            {"event": "timeout", "probability": 0.2,
             "probability_ref": "risk_predicted[timeout].probability"},
            {"event": "implementation_failure", "probability": 0.1,
             "probability_ref":
                 "risk_predicted[implementation_failure].probability"},
        ],
        "risk_actual": [
            {"event": "timeout", "label": "not_occurred",
             "label_basis": "no_timeout_interruption", "predicted": True,
             "probability_ref": "risk_predicted[timeout].probability",
             "brier": 0.04, "observation_rule_version": "wm-events/4"},
            {"event": "implementation_failure",
             "label": "occurred" if not crash else None,
             "label_basis": "class_not_recorded", "predicted": True,
             "probability_ref":
                 "risk_predicted[implementation_failure].probability",
             "brier": 0.81,
             "observation_rule_version": "wm-events/4"},
            {"event": "environment_failure", "label": "not_occurred",
             "label_basis": "unpredicted_no_brier", "predicted": False,
             "observation_rule_version": "wm-events/4"},
        ],
        "execution_status": "error" if crash else status,
        "task_check": {"state": "failed" if (observed < 1 and not crash)
                       else "passed" if not crash else None,
                       "action_id": f"ac_check_{index}",
                       "scope": {"basis": ["semantic_probe"]}},
        "execution_conditions": {
            "comparable": True,
            "execution_id": f"ex_{index}",
            "comparison_key": {"solver": "highs"},
            "config": {"solver": "highs"}},
        "implementation_observation": {
            "event": "implementation_failure",
            "label": 1 if crash else 0,
            "comparable": True,
            "executed": True,
            "exit_code": 1 if crash else 0,
            "reason": ("process exited with a protocol error"
                       if crash else "protocol completed normally"),
            "result_received": not crash,
            "scope": "attempt", "task_id": task_id,
            "solver_interface": "highs",
            "observation_rule_version": "wm-r-technical/1",
            "reusable_script": {"artifact": {"path": "artifacts/script"}},
        },
        "selected_reason": ["a recalled hit names one of this pair's "
                            "executions", "same task as the current problem"],
        "relevance": 4,
        "background": False,
    }
    if unknown_label:
        row["risk_actual"][1]["label"] = None
    if crash:
        row["benefit"].pop("observed", None)
        row["benefit"].pop("signed_error", None)
        row["benefit"]["observed"] = None
        row["benefit"].pop("observed")
    return row


def _selection_block(rows, *, omitted=0):
    from or_harness.world_model.episode_closeout import RISK_REASON_LEGEND
    return {
        "feedback_version": "wm-pairs/1",
        "model_identity": "m@1",
        "observation_rule_version": "wm-obs/4",
        "n_pairs_total": len(rows),
        "n_pairs_available": len(rows),
        "n_pairs_readable": len(rows),
        "n_pairs_included": len(rows),
        "n_pairs_omitted": omitted,
        "selection": {"limit": 12, "limit_basis": "display ceiling",
                      "signature": "sig", "basis": "relevance"},
        "selection_signature": "sig",
        "risk_reason_legend": dict(RISK_REASON_LEGEND),
        "same_task_comparisons": [],
        "counts_note": ("n_pairs_total is the calibration window; "
                        "n_pairs_included is what was selected for display"),
        "ordering": "labeled cases first, execution errors last",
        "note": "labels rank before relevance",
        "basis": "derived from closed episodes",
        "pairs": rows,
    }


class TestProviderFeedbackProjection(unittest.TestCase):
    """Compression keeps the original facts and restores every omission."""

    def test_prediction_observation_error_and_reference_correspond(self):
        from or_harness.strategy.evidence_views import provider_feedback_pair
        row = _pair(1, observed=1.0, predicted=0.8)
        compact = provider_feedback_pair(row)
        self.assertEqual(compact["benefit"]["predicted"], 0.8)
        self.assertEqual(compact["benefit"]["observed"], 1.0)
        self.assertEqual(compact["benefit"]["signed_error"], 0.2)
        self.assertEqual(compact["benefit"]["metric"],
                         row["benefit"]["metric"])
        self.assertEqual(compact["benefit"]["observation_rule_version"],
                         "wm-g-faithful/1")
        self.assertEqual(compact["task_check"]["state"], "passed")
        self.assertEqual(compact["detail_ref"],
                         "orx inspect --bank evaluations --evaluation ev_1")
        self.assertEqual(compact["conditions"], row["conditions"])
        # Evidence references and the measured cost correspond exactly.
        self.assertEqual(compact["benefit"]["evidence_refs"],
                         row["benefit"]["evidence_refs"])
        self.assertEqual(compact["cost_measured"],
                         row["cost_measured"])
        self.assertEqual(compact["task_check"]["action_id"], "ac_check_1")
        # The same input twice is byte-identical (deterministic projection).
        self.assertEqual(json.dumps(compact, ensure_ascii=False, sort_keys=True),
                         json.dumps(provider_feedback_pair(row),
                                    ensure_ascii=False, sort_keys=True))

    def test_omitted_detail_is_restored_by_a_valid_reference(self):
        from or_harness.strategy.evidence_views import provider_feedback_pair
        row = _pair(2, big=True)
        compact = provider_feedback_pair(row)
        # The plan and the performed method are IDENTICAL, so the performed
        # steps are carried once and the plan points at them (a fact, never
        # an inference).
        self.assertEqual(compact["method_planned"]["steps_ref"],
                         "method_actual.steps")
        self.assertIn("steps", compact["method_actual"])
        # The acceptance spec, the measured cost conditions and the long
        # notes are referenced, never silently lost.
        self.assertNotIn("spec",
                         compact["benefit"]["acceptance_rule"])
        self.assertIn("report_ref",
                      compact["benefit"]["acceptance_rule"])
        self.assertEqual(
            compact["cost_observed_fact"]["conditions_ref"],
            "orx inspect --bank evaluations --evaluation ev_2 "
            "#cost_observed_fact.conditions")
        self.assertEqual(len(compact["benefit"]["notes"]), 2)
        self.assertIn("notes_ref", compact["benefit"])
        # No carried method step exceeds the compact window.
        for step in compact["method_actual"]["steps"]:
            self.assertLessEqual(len(step), 160)

    def test_unknown_label_is_not_a_failure_and_keeps_its_unknown(self):
        from or_harness.strategy.evidence_views import provider_feedback_pair
        row = _pair(3, unknown_label=True, observed=1.0)
        compact = provider_feedback_pair(row)
        implementation = next(
            event for event in compact["risk_actual"]
            if event["event"] == "implementation_failure")
        self.assertIsNone(implementation["label"])
        # Label evidence stays a fact, and no G=0/crash was invented.
        self.assertEqual(compact["benefit"]["observed"], 1.0)
        self.assertEqual(compact["execution_status"], "optimal")

    def test_crash_is_unobserved_for_g_not_a_zero(self):
        from or_harness.strategy.evidence_views import provider_feedback_pair
        compact = provider_feedback_pair(_pair(4, crash=True))
        # A crash carries NO observed G label (never 0) and keeps the
        # execution status as a SEPARATE fact.
        self.assertNotIn("observed", compact["benefit"])
        self.assertEqual(compact["execution_status"], "error")
        self.assertEqual(
            compact["implementation_observation"]["label"], 1)
        self.assertFalse(
            compact["implementation_observation"]["result_received"])

    def test_missing_attribution_is_not_fabricated(self):
        from or_harness.strategy.evidence_views import provider_feedback_pair
        compact = provider_feedback_pair(_pair(5))
        self.assertNotIn("attribution", compact["benefit"])
        # A RECORDED attribution travels; it is copied, never inferred.
        recorded = {"method_fidelity": "deviated",
                    "failure_source": "implementation",
                    "source": "auditor", "assessment": "deviation recorded"}
        with_attr = provider_feedback_pair(_pair(6, attribution=recorded))
        self.assertEqual(with_attr["benefit"]["attribution"]["method_fidelity"],
                         "deviated")
        self.assertEqual(with_attr["benefit"]["attribution"]["source"],
                         "auditor")

    def test_eligibility_and_comparability_survive(self):
        from or_harness.strategy.evidence_views import provider_feedback_pair
        row = _pair(7)
        row["benefit"]["label_eligibility"] = "unknown"
        row["benefit"]["eligibility"] = "unobserved"
        row["execution_conditions"]["comparable"] = False
        row["execution_conditions"]["unknown"] = ["time_limit"]
        compact = provider_feedback_pair(row)
        self.assertEqual(compact["benefit"]["label_eligibility"], "unknown")
        self.assertEqual(compact["benefit"]["eligibility"], "unobserved")
        self.assertFalse(compact["execution_conditions"]["comparable"])
        self.assertEqual(compact["execution_conditions"]["unknown"],
                         ["time_limit"])


class TestHistoryBlockBudgets(unittest.TestCase):
    """The history view is bounded per row before any case is dropped."""

    def _block(self, n):
        return _selection_block([_pair(index, big=True) for index in range(n)])

    def test_no_history_is_an_explicit_empty_view(self):
        from or_harness.strategy.evidence_views import provider_paired_feedback
        out = provider_paired_feedback(_selection_block([]))
        self.assertEqual(out["pairs"], [])
        self.assertEqual(out["n_pairs_included"], 0)
        self.assertEqual(out["provider_view_version"],
                         "wm-pairs-provider/1")
        self.assertIn("provider_compaction", out)

    def test_one_pair_is_kept_whole(self):
        from or_harness.strategy.evidence_views import provider_paired_feedback
        out = provider_paired_feedback(self._block(1))
        self.assertEqual(len(out["pairs"]), 1)
        self.assertEqual(out["n_pairs_included"], 1)
        self.assertEqual(out["pairs"][0]["benefit"]["observed"], 1.0)

    def test_twelve_pairs_stay_within_the_history_budget(self):
        from or_harness.strategy.evidence_views import (
            history_feedback_budget_chars, provider_paired_feedback,
        )
        out = provider_paired_feedback(
            _selection_block([_pair(index) for index in range(12)]),
            budget_chars=40000)
        self.assertEqual(len(out["pairs"]), 12)
        self.assertEqual(out["n_pairs_included"], 12)
        self.assertLessEqual(len(json.dumps(out, ensure_ascii=False)),
                             40000)
        # The same 12 pairs under the DEFAULT budget stay bounded and keep
        # the counts consistent even when some rows do not fit.
        default = provider_paired_feedback(
            _selection_block([_pair(index) for index in range(12)]))
        self.assertLessEqual(len(json.dumps(default, ensure_ascii=False)),
                             history_feedback_budget_chars())
        self.assertEqual(default["n_pairs_included"],
                         len(default["pairs"]))

    def test_twelve_rich_pairs_reduce_rows_before_dropping(self):
        from or_harness.strategy.evidence_views import provider_paired_feedback
        out = provider_paired_feedback(self._block(12))
        self.assertLessEqual(len(out["pairs"]), 12)
        self.assertEqual(out["n_pairs_included"], len(out["pairs"]))
        self.assertEqual(
            out["n_pairs_provider_dropped"], 12 - len(out["pairs"]))
        self.assertTrue(out["provider_compaction"]["rows_reduced"])

    def test_many_pairs_reduce_rows_then_drop_cases_with_references(self):
        from or_harness.strategy.evidence_views import provider_paired_feedback
        block = self._block(40)
        out = provider_paired_feedback(block, budget_chars=8000)
        self.assertLess(len(out["pairs"]), 40)
        self.assertGreater(len(out["pairs"]), 0)
        self.assertEqual(out["n_pairs_included"], len(out["pairs"]))
        self.assertEqual(out["n_pairs_provider_dropped"], 40 - len(out["pairs"]))
        self.assertEqual(out["n_pairs_omitted"], 40 - len(out["pairs"]))
        self.assertTrue(out.get("dropped_pair_refs"))
        for ref in out["dropped_pair_refs"]:
            self.assertIn("orx inspect --bank evaluations", ref)
        self.assertIn("omission_note", out)

    def test_before_after_is_measured_on_a_fixed_input(self):
        from or_harness.strategy.evidence_views import provider_paired_feedback
        from or_harness.strategy.evidence_views import chars as view_chars
        block = _selection_block([_pair(index) for index in range(12)])
        stored_copy = copy.deepcopy(block)
        before_chars = view_chars(block)
        out = provider_paired_feedback(block, budget_chars=40000)
        # The projection never mutates the STORED selection/data.
        self.assertEqual(block, stored_copy)
        after_chars = view_chars(out)
        compaction = out["provider_compaction"]
        self.assertEqual(compaction["before_chars"], before_chars)
        self.assertEqual(compaction["after_chars"], after_chars)
        self.assertLess(after_chars, before_chars)
        # The fixed input keeps every case; prose/audit scaffolding is what
        # was reduced, and the reduction is named.
        self.assertEqual(out["n_pairs_included"], 12)
        self.assertTrue(compaction["rows_reduced"])
        self.assertTrue(compaction["fields_reduced"])
        self.assertGreaterEqual(
            len(compaction["fields_reduced"]), 5)
        # The same fixed input gives the same output size (no randomness).
        from or_harness.strategy.evidence_views import chars as view_chars2
        self.assertEqual(
            after_chars,
            view_chars2(provider_paired_feedback(block, budget_chars=40000)))

    def test_lean_keeps_the_recorded_failure_attribution(self):
        """Compression keeps WHY the case failed before audit scaffolding.

        The auditor's short assessment is the highest-value part of a failed
        case; it must survive the per-row reduction, and only whole-case
        removal may lose it (with a restore reference).
        """
        from or_harness.strategy.evidence_views import (
            _lean_provider_pair, provider_feedback_pair,
            provider_paired_feedback,
        )
        recorded = {
            "method_fidelity": "faithful",
            "failure_source": "candidate_method",
            "source": "auditor",
            "assessment": ("The task required integer quantities, but the "
                           "method used continuous variables; the small "
                           "instance hid the violation."),
        }
        row = _pair(9, big=True, attribution=recorded)
        compact = provider_feedback_pair(row)
        self.assertEqual(compact["benefit"]["attribution"]["assessment"],
                         recorded["assessment"])
        lean, dropped = _lean_provider_pair(compact)
        # The recorded cause survives the LEAN reduction ...
        self.assertEqual(lean["benefit"]["attribution"]["assessment"],
                         recorded["assessment"])
        self.assertNotIn("benefit.attribution.assessment", dropped)
        # ... while audit scaffolding is what actually went.
        self.assertNotIn("executed",
                         lean.get("implementation_observation") or {})
        block = _selection_block([_pair(index, big=True,
                                        attribution=recorded)
                                  for index in range(12)])
        out = provider_paired_feedback(block, budget_chars=30000)
        self.assertTrue(out["pairs"])
        for pair in out["pairs"]:
            self.assertEqual(pair["benefit"]["attribution"]["assessment"],
                             recorded["assessment"])
        self.assertNotIn("benefit.attribution.assessment",
                         out["provider_compaction"]["fields_reduced"])

    def test_legends_explain_only_codes_that_are_displayed(self):
        from or_harness.strategy.evidence_views import provider_paired_feedback
        out = provider_paired_feedback(self._block(2))
        used = {event["label_basis"] for row in out["pairs"]
                for event in row.get("risk_actual") or []}
        self.assertEqual(set(out["risk_reason_legend"]), used)


class TestTokenMeasurement(unittest.TestCase):
    """CJK/JSON token density is counted conservatively, never as chars/4."""

    def test_cjk_density_is_higher_than_the_legacy_default(self):
        from or_harness.world_model.request_budget import estimate_tokens
        text = '{"说明": "原题要求两个工厂必须同厂，交付窗口不得推迟。"}' * 20
        measured = estimate_tokens(text)
        self.assertGreater(measured["tokens"], len(text) / 4.0)
        self.assertGreater(measured["cjk_chars"], 100)
        self.assertEqual(measured["basis"], "labeled_conservative_estimate")
        self.assertIn("CONSERVATIVE", measured["estimator"])

    def test_json_density_is_higher_than_plain_chars_per_token(self):
        from or_harness.world_model.request_budget import estimate_tokens
        dense = json.dumps({"a": [1, 2, 3], "b": {"c": "d"}}) * 20
        self.assertGreater(estimate_tokens(dense)["tokens"],
                           len(dense) / 4.0)

    def test_provider_tokenizer_is_used_when_exposed(self):
        from or_harness.world_model.request_budget import measure_text

        class Tokenizer:
            def count_tokens(self, text):
                return 1234

        measured = measure_text("anything", provider=Tokenizer())
        self.assertEqual(measured["tokens"], 1234)
        self.assertEqual(measured["basis"], "provider_tokenizer")

    def test_a_provider_without_a_tokenizer_is_an_estimate(self):
        from or_harness.world_model.provider import WorldModelProvider
        from or_harness.world_model.request_budget import measure_text
        measured = measure_text("plain text", provider=WorldModelProvider())
        self.assertEqual(measured["basis"], "labeled_conservative_estimate")


class TestFullRequestBudget(unittest.TestCase):
    """The WHOLE request is measured; history shrinks before the task."""

    def _request(self, pairs=12):
        rows = [_pair(index, big=True) for index in range(pairs)]
        context = {
            "task_id": "t1",
            "joint_problem": {"text": "The CURRENT TASK BODY " * 60,
                              "profile": {"family": "routing"}},
            "execution_constraints": {"declared_budget": None,
                                      "executor": {"timeout_seconds": 120}},
            "solving_context": {"budget_state": {"status": "ok"}},
            "prediction_execution_pairs": _selection_block(rows),
            "strategy_outcome_calibration": {"conditional_g": [
                {"conditions": {"family": "routing"}}]},
            "retrieval_evidence": {"hits": [
                {"identity": f"execution_evidence:ex_{index}",
                 "content": {"text": "evidence " * 40}}
                for index in range(12)]},
            "harness_capability": {"sources": {
                name: {"status": "observed",
                       "evidence": [{"note": "evidence " * 30}] * 5}
                for name in ("m", "w_or", "pi", "r", "t")}},
            "cost_evidence": {"cases": [{"note": "cost " * 40}] * 6},
            "implementation_r_evidence": {"groups": [
                {"events": {"timeout": {"occurred": 0}},
                 "reusable_scripts": [{"artifact": {"path": "a"}}] * 4}]},
        }
        return {
            "prediction_context": context,
            "candidate": {"action_type": "execute_strategy",
                          "strategy_id": "S04",
                          "method": {"name": "plan",
                                     "steps": ["step one", "step two"]}},
            "risk_convention": {"events": ["implementation_failure",
                                           "timeout",
                                           "environment_failure"]},
            "benefit_convention": {"kind": "solution_quality",
                                   "metric": "normalized_objective_gap"},
            "cost_scope": {"anchor": "decision"},
            "output_contract": {"allowed_fields": ["benefit", "cost"]},
        }

    def _with_budget(self, tokens, fn):
        prior = os.environ.get("OR_HARNESS_INPUT_BUDGET_TOKENS")
        os.environ["OR_HARNESS_INPUT_BUDGET_TOKENS"] = str(tokens)
        try:
            return fn()
        finally:
            if prior is None:
                os.environ.pop("OR_HARNESS_INPUT_BUDGET_TOKENS", None)
            else:
                os.environ["OR_HARNESS_INPUT_BUDGET_TOKENS"] = prior

    def test_history_is_reduced_first_and_the_task_is_kept(self):
        from or_harness.world_model.request_budget import (
            enforce_full_request_budget,
        )
        request = self._request(12)
        task_before = copy.deepcopy(
            request["prediction_context"]["joint_problem"])
        candidate_before = copy.deepcopy(request["candidate"])

        def run():
            return enforce_full_request_budget(
                request, "system prompt " * 800)

        size = self._with_budget(24000, run)
        self.assertTrue(size["sendable"])
        self.assertTrue(size["trimming"])
        self.assertEqual(size["trimming"][0]["stage"], "history_feedback")
        # The current task and the candidate are BYTE-identical.
        self.assertEqual(request["prediction_context"]["joint_problem"],
                         task_before)
        self.assertEqual(request["candidate"], candidate_before)
        # The history is smaller than the input block.
        history = request["prediction_context"]["prediction_execution_pairs"]
        self.assertLessEqual(len(history.get("pairs") or []), 12)
        self.assertLessEqual(size["tokens"], size["bound_tokens"])

    def test_protected_core_over_budget_refuses_to_send(self):
        from or_harness.world_model.request_budget import (
            enforce_full_request_budget,
        )
        request = self._request(12)
        candidate_before = copy.deepcopy(request["candidate"])
        task_before = copy.deepcopy(
            request["prediction_context"]["joint_problem"])
        size = self._with_budget(
            600, lambda: enforce_full_request_budget(
                request, "system prompt " * 500))
        self.assertFalse(size["sendable"])
        self.assertFalse(size["within_bound"])
        self.assertIn("over_budget", size)
        self.assertEqual(size["over_budget"]["core_chars"],
                         size["core"]["core_chars"])
        # The core was NOT truncated to buy space.
        self.assertEqual(request["candidate"], candidate_before)
        self.assertEqual(request["prediction_context"]["joint_problem"],
                         task_before)

    def test_wire_request_excludes_local_diagnostics(self):
        """The budget describes the SENT object, not the local diagnostics.

        ``request_size`` is attached after enforcement and never serialized
        to the provider, so attaching it cannot push an in-bound request over
        the wire bound.
        """
        from or_harness.world_model.request_budget import (
            enforce_full_request_budget, measure_request,
            strip_request_diagnostics,
        )
        request = self._request(12)
        prompt = "system prompt " * 800
        size = self._with_budget(
            24000, lambda: enforce_full_request_budget(request, prompt))
        self.assertTrue(size["sendable"])
        wire = strip_request_diagnostics({**request, "request_size": size})
        self.assertNotIn("request_size", wire)
        measured = measure_request(wire, prompt)
        # The measured object and the reported size are the SAME wire body.
        self.assertEqual(measured["chars"], size["chars"])
        self.assertEqual(measured["tokens"], size["tokens"])
        self.assertLessEqual(measured["tokens"], size["bound_tokens"])
        self.assertEqual(size["wire"]["excluded_keys"], ["request_size",
                                                         "request_diagnostics"])
        # Adding the local diagnostics does change the serialized dict ...
        self.assertGreater(
            len(json.dumps({**request, "request_size": size},
                           ensure_ascii=False)),
            len(json.dumps(wire, ensure_ascii=False)))
        # ... which is exactly why it is excluded from the wire and budget.

    def test_cjk_request_is_measured_conservatively(self):
        from or_harness.world_model.request_budget import (
            enforce_full_request_budget,
        )
        request = self._request(2)
        request["prediction_context"]["joint_problem"]["text"] = (
            "原题要求两个工厂必须同厂，交付窗口不得推迟。" * 300)
        size = self._with_budget(
            40000, lambda: enforce_full_request_budget(
                request, "系统提示 " * 500))
        self.assertEqual(size["token_basis"],
                         "labeled_conservative_estimate")
        self.assertIn("CONSERVATIVE", size["estimator"].upper())
        self.assertGreater(size["tokens"],
                           size["before"]["chars"] / 4.0)

    def test_provider_tokenizer_makes_the_bound_real(self):
        from or_harness.world_model.request_budget import (
            enforce_full_request_budget,
        )

        class HugeTokenizer:
            def count_tokens(self, text):
                return 10 ** 6

        request = self._request(2)
        size = self._with_budget(
            100000,
            lambda: enforce_full_request_budget(
                request, "prompt", provider=HugeTokenizer()))
        self.assertEqual(size["token_basis"], "provider_tokenizer")
        self.assertFalse(size["sendable"])


class TestEndToEndPriority(M4Case):
    """Through the real prediction service: history shrinks, core is kept."""

    def _history(self, pairs=2):
        conv = {"kind": "solution_quality",
                "metric": "normalized_objective_gap", "unit": "1-gap"}
        task = _task("t1")
        for index in range(pairs):
            candidate = {"action_type": "execute_strategy",
                         "strategy_id": f"S{index:02d}",
                         "solver": "highs",
                         "method": {"name": f"method {index}",
                                    "steps": [f"build {index}",
                                              f"solve {index}"]}}
            prediction = planned_prediction(self.h, task, candidate,
                                            f"ep{index}",
                                            benefit_convention=conv)
            record = self.solve(task, strategy=f"S{index:02d}",
                                episode_id=f"ep{index}",
                                tag=f"e2e_{index}")
            bind_legacy(self.h, prediction.prediction_id,
                        record.action_id)
            self.h.close_episode("t1", f"ep{index}")
        return task, conv

    def test_history_reduced_first_and_current_task_kept(self):
        from or_harness.world_model.context import _input_budget_chars
        task, conv = self._history(2)
        self.provider.requests.clear()
        prior = os.environ.get("OR_HARNESS_INPUT_BUDGET_TOKENS")
        # A budget that passes the legacy chars/4 check but is too small for
        # the conservative token estimate: the history must be reduced. The
        # figure is a fixture calibration, not a minimum deployment budget:
        # it stays above the protected core (which includes the system
        # prompt) while forcing history-first trimming.
        os.environ["OR_HARNESS_INPUT_BUDGET_TOKENS"] = "28000"
        try:
            prediction = planned_prediction(
                self.h, task, {"action_type": "execute_strategy",
                               "strategy_id": "SNEW", "solver": "highs",
                               "method": {"name": "new",
                                          "steps": ["a", "b"]}},
                "epnew", benefit_convention=conv)
        finally:
            if prior is None:
                os.environ.pop("OR_HARNESS_INPUT_BUDGET_TOKENS", None)
            else:
                os.environ["OR_HARNESS_INPUT_BUDGET_TOKENS"] = prior
        self.assertTrue(self.provider.requests,
                        "the request was within the bound and was sent")
        request = self.provider.requests[-1]
        # The provider receives the request AS SENT: the local budget
        # diagnostics are kept on the prediction record, never on the wire.
        self.assertNotIn("request_size", request)
        context = request["prediction_context"]
        # The current task body and the candidate steps survive untouched.
        self.assertIn("text", context["joint_problem"])
        self.assertIn("description",
                      context["joint_problem"]["task_payload"])
        self.assertEqual(request["candidate"]["method"]["steps"],
                         ["a", "b"])
        self.assertIn("execution_constraints", context)
        self.assertIn("risk_convention", request)
        self.assertIn("benefit_convention", request)
        self.assertIn("cost_scope", request)
        self.assertIn("output_contract", request)
        size = prediction.trace.model_info["request_size"]
        self.assertTrue(size["sendable"])
        self.assertLessEqual(size["tokens"], size["bound_tokens"])
        if size["trimming"]:
            self.assertEqual(size["trimming"][0]["stage"],
                             "history_feedback")

    def test_over_budget_core_is_refused_not_sent(self):
        task, conv = self._history(1)
        self.provider.requests.clear()
        prior = os.environ.get("OR_HARNESS_INPUT_BUDGET_TOKENS")
        os.environ["OR_HARNESS_INPUT_BUDGET_TOKENS"] = "600"
        try:
            plan = self.h.plan_next(
                task, "epnew",
                candidates=[{"action_type": "execute_strategy",
                             "strategy_id": "SNEW", "solver": "highs",
                             "method": {"name": "new",
                                        "steps": ["a", "b"]}}])
        finally:
            if prior is None:
                os.environ.pop("OR_HARNESS_INPUT_BUDGET_TOKENS", None)
            else:
                os.environ["OR_HARNESS_INPUT_BUDGET_TOKENS"] = prior
        self.assertEqual(self.provider.requests, [],
                         "an over-budget request must NOT be sent")
        rows = plan.get("candidates") or []
        prediction_ids = [row.get("prediction_id") for row in rows
                          if row.get("prediction_id")]
        self.assertTrue(prediction_ids)
        prediction = self.h.strategy_predictions.get(prediction_ids[0])
        self.assertEqual(prediction.status, "invalid")
        failure = prediction.trace.model_info.get("failure") or {}
        self.assertEqual(failure.get("kind"), "request_over_budget")
        self.assertIn("input request exceeds",
                      prediction.trace.model_info.get("error", ""))
        self.assertFalse(
            (prediction.trace.model_info.get("request_size") or {}).get(
                "sendable"))


if __name__ == "__main__":
    unittest.main()
