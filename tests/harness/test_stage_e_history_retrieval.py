"""Stage E/1: unified historical retrieval, local cases and global statistics.

The tests are deterministic and hermetic. They cover:

- the candidate-aware multi-view embedding retrieval (existing index, several
  query views, one merged pool);
- the per-view similarity floors, the no-padding rule, the 10-case ceiling
  and the task-diversity rule;
- G's no-keyword-method-match rule and the explicit UNKNOWN label;
- C's removal of the phase classification and the four remaining-task cost
  dimensions (missing dimensions stay missing);
- the GLOBAL G/C/R statistics and the rule that they use pre-decision closed
  pairs only;
- the provider block shapes, the budget trim and coverage refresh.

They also keep the active four-dimensional C prediction contract assertions
that used to live in ``test_stage_d_prediction_evidence``.
"""
import copy
import unittest
from types import SimpleNamespace

from g_fixtures import COUPLED, SPLIT, GCase

from or_harness.core.schema import ExecutionRecord, ProblemProfile
from or_harness.world_model.contracts import (
    FAITHFUL_G_METRIC,
    FAITHFUL_G_OBSERVATION_VERSION,
    R_CONVENTION,
    R_OBSERVATION_VERSION,
    RETRIES_FUTURE_VERSION,
)
from or_harness.world_model.history_retrieval import (
    _attach_evaluation_facts,
    _diversify,
    _method_digest,
    _passes_admission,
    _pool_row as _resolve_pool_row,
    _evaluation_fact,
    build_history_retrieval,
    build_query_views,
    decision_cost_scales,
    global_c_statistics,
    global_g_statistics,
    global_r_statistics,
    history_config,
    refresh_c_history_coverage,
    select_c_history_evidence,
    select_g_history_evidence,
    select_r_history_evidence,
)
from or_harness.world_model.r_signal import (
    global_event_base_rates, technical_r_statistics,
)
from or_harness.strategy.embedding_index import (
    document_digest, document_execution,
)


# ---------------------------------------------------------------------------
# synthetic history / candidate helpers
# ---------------------------------------------------------------------------

def _candidate(method=None, *, solver="highs", strategy_id="S04",
               config=None, family="assignment", scope="attempt"):
    return SimpleNamespace(
        method=method or {"name": "joint selection",
                          "steps": ["select the same factory"]},
        solver=solver, strategy_id=strategy_id, config=config or {},
        scope=scope, task_id="t", action_type="execute_strategy")


def _history(pool, *, candidate=None, config=None, g=None, c=None, r=None,
             degraded=None, missing=None):
    candidate = candidate or _candidate()
    config = {
        "version": "wm-history/1",
        "max_local_cases": 10,
        "pool_size": 30,
        "max_task_attempts": 2,
        "similarity_thresholds": {
            "problem": 0.62, "candidate_method": 0.60, "conditions": 0.30},
        "method_similarity_threshold": 0.62,
        "threshold_verified": True,
        "threshold_sources": {},
        "score_definition": "fixture",
        **(config or {}),
    }
    return {
        "version": "wm-history/1",
        "status": "ok" if not degraded else "degraded",
        "config": config,
        "candidate_views": [{
            "view": "candidate:0",
            "strategy_id": candidate.strategy_id,
            "method_digest": _method_digest(candidate.method)}],
        "queries": [{"name": "problem"}, {"name": "candidate:0"},
                    {"name": "conditions"}],
        "pool": copy.deepcopy(pool),
        "accounting": {"pool_size": len(pool),
                       "distinct_tasks": len({row.get("task_id")
                                              for row in pool}),
                       "above_threshold": len(pool),
                       "below_threshold": 0,
                       "index_items": len(pool),
                       "scanned": len(pool)},
        "global": {"g": g or {}, "c": c or {}, "r": r or {}},
        "degraded": degraded or [],
        "missing": missing or [],
        "notes": [],
    }


def _pool_row(execution_id="ex1", task_id="t1", *,
              problem=0.80, method=0.70, conditions=0.30,
              evaluation=None, status="optimal", solver="highs",
              structural="different_cell", failure_classes=None,
              r_events=None, task_excerpt="a comparable task"):
    relative = {
        "problem": problem / 0.62,
        "candidate:0": method / 0.60,
        "conditions": conditions / 0.30,
    }
    row = {
        "execution_id": execution_id,
        "task_id": task_id,
        "strategy_id": "S04",
        "task_excerpt": task_excerpt,
        "similarity": {
            "overall": round(max(relative.values()), 6),
            "by_view": {"problem": problem, "candidate:0": method,
                        "conditions": conditions},
            "relative_by_view": {key: round(value, 6)
                                 for key, value in relative.items()},
        },
        "structural_match": structural,
        "profile_cell": "cell",
        "method": {
            "basis": "performed",
            "planned": {"name": "historical method",
                        "steps": ["do the historical work"]},
            "actual": {"name": "historical method",
                       "steps": ["do the historical work"]},
        },
        "status": status,
        "feasible": status == "optimal",
        "measurement_scope": "attempt",
        "failure_classes": list(failure_classes or []),
        "r_events": r_events or {
            "implementation_failure": {"label": 0, "basis": "no failure"},
            "timeout": {"label": 0, "basis": "finished"},
            "environment_failure": {"label": 0, "basis": "no environment"},
        },
        "solver_interface": solver,
        "created_at": 1.0,
    }
    if evaluation is not None:
        row["evaluation"] = copy.deepcopy(evaluation)
    return row


def _g_evaluation(evaluation_id="ev1", *, observed=1, predicted=0.7,
                  metric=FAITHFUL_G_METRIC,
                  rule=FAITHFUL_G_OBSERVATION_VERSION,
                  attribution=None):
    benefit = {"observed": observed, "predicted": predicted,
               "signed_error": (predicted - observed
                                if observed is not None else None),
               "metric": metric, "observation_rule_version": rule}
    if attribution:
        benefit["attribution"] = attribution
    return {"evaluation_id": evaluation_id, "g": benefit,
            "task_check": {"state": "passed" if observed == 1 else "failed"}}


def _c_evaluation(evaluation_id="ev-c1", *, values=None, scope=None,
                  window="complete", family="assignment",
                  structure=None, solver="highs",
                  predicted_actual=None, retry_rule=RETRIES_FUTURE_VERSION):
    return {
        "evaluation_id": evaluation_id,
        "cost": {
            "values": dict(values or {}),
            "scope": scope or "remaining_to_task_end",
            "window_coverage": window,
            "family": family,
            "structure": structure if structure is not None
            else {"problem_class": "milp"},
            "solver_conditions": {"constraints": {"executor": None,
                                                  "declared_budget": None}},
            "actual_solving_conditions": [{"solver": solver, "config": {}}],
            "reusable_artifacts": False,
            "source_refs": ["ex-c"],
        },
        "cost_predicted_actual": copy.deepcopy(predicted_actual or {}),
    }


def _r_evaluation(evaluation_id="ev-r1", *, events=None, predicted=None):
    return {
        "evaluation_id": evaluation_id,
        "risk_actual": [{"event": name, "label": label,
                         "predicted": (predicted or {}).get(name),
                         "brier": ((predicted or {}).get(name, 0.5) - label) ** 2
                         if label in (0, 1) else None,
                         "label_basis": "fixture"}
                        for name, label in (events or {}).items()],
    }


def _pair(evaluation_id="ev1", *, task_id="t1", episode_id="e1",
          observed=1, predicted=0.7, cost_values=None,
          cost_predicted_actual=None, risk=None, retry_rule=None,
          metric=FAITHFUL_G_METRIC, rule=FAITHFUL_G_OBSERVATION_VERSION,
          window="complete", scope="remaining_to_task_end"):
    pair = {
        "evaluation_id": evaluation_id,
        "task_id": task_id,
        "episode_id": episode_id,
        "benefit": {"observed": observed, "predicted": predicted,
                    "signed_error": (predicted - observed
                                     if observed is not None else None),
                    "metric": metric, "observation_rule_version": rule},
    }
    if cost_values is not None or cost_predicted_actual is not None:
        pair["cost_observed_fact"] = {
            "values": dict(cost_values or {}),
            "conditions": {
                "scope": scope, "window_coverage": window,
                "family": "assignment",
                "structure": {"problem_class": "milp"},
                "solver_conditions": {"constraints": {}},
                "actual_solving_conditions": [{"solver": "highs",
                                               "config": {}}],
                "retry_observation_rule_version": retry_rule
                or RETRIES_FUTURE_VERSION,
            },
            "lower_bounds": {},
            "source_refs": ["ex"],
        }
        if cost_predicted_actual:
            pair["cost_measured"] = copy.deepcopy(cost_predicted_actual)
    if risk:
        predicted_risk = {name: probability for name, (probability, _label)
                          in risk.items()}
        pair["risk_predicted"] = [
            {"event": name, "probability": probability}
            for name, probability in predicted_risk.items()]
        pair["risk_actual"] = [
            {"event": name, "label": label,
             "predicted": probability,
             "brier": (probability - label) ** 2 if label in (0, 1) else None,
             "label_basis": "fixture"}
            for name, (probability, label) in risk.items()]
    return pair


def _context(*, family="assignment", problem_class="milp", artifacts=None,
             history=None, constraints=None):
    progress = {}
    if artifacts is not None:
        progress["model_artifact"] = {"value": artifacts, "source": "fixture"}
    return SimpleNamespace(
        joint=SimpleNamespace(profile={"family": family,
                                       "problem_class": problem_class,
                                       "problem_id": "t"}),
        solving_context={"task_progress": progress},
        execution_constraints=constraints or {},
        created_at=1.0,
        task_id="t",
        history_retrieval=history or {},
    )


# ---------------------------------------------------------------------------
# retrieval: query views, threshold, diversity, no padding, degradation
# ---------------------------------------------------------------------------

class TestUnifiedRetrieval(unittest.TestCase):
    def test_query_views_cover_problem_method_and_conditions(self):
        views = build_query_views(
            {"task_id": "t", "text": "assign the same factory"},
            {"family": "assignment", "problem_class": "milp"},
            [_candidate({"name": "joint selection",
                         "steps": ["select one factory"]})])
        names = [view["name"] for view in views]
        self.assertEqual(names, ["problem", "candidate:0", "conditions"])
        method_view = next(view for view in views
                           if view["kind"] == "candidate_method")
        self.assertIn("joint selection", method_view["text"])
        # The conditions view carries NO problem prose: shared generic words
        # must not masquerade as a structural match.
        conditions_view = next(view for view in views
                               if view["kind"] == "conditions")
        self.assertIn("family assignment", conditions_view["text"])
        self.assertNotIn("same factory", conditions_view["text"])

    def test_any_view_reaching_its_own_floor_admits_a_row(self):
        floors = {"problem": 0.62, "candidate_method": 0.60,
                  "conditions": 0.30}
        self.assertTrue(_passes_admission({"problem": 0.70}, floors))
        self.assertTrue(_passes_admission({"candidate:0": 0.61}, floors))
        self.assertFalse(_passes_admission({"problem": 0.61,
                                            "candidate:0": 0.59,
                                            "conditions": 0.29}, floors))

    def test_case_ceiling_is_ten_and_no_padding(self):
        candidate = _candidate()
        rows = [_pool_row(f"ex{i}", task_id=f"task{i}") for i in range(14)]
        history = _history(rows, candidate=candidate)
        block = select_g_history_evidence(history, candidate=candidate,
                                          task_id="t")
        self.assertEqual(len(block["cases"]), 10)
        self.assertEqual(block["selection"]["n_included"], 10)
        # A pool of 3 relevant rows returns 3, not 10 padded rows.
        short = select_g_history_evidence(
            _history(rows[:3], candidate=candidate), candidate=candidate,
            task_id="t")
        self.assertEqual(len(short["cases"]), 3)
        self.assertEqual(short["selection"]["n_included"], 3)
        # An empty pool yields an EMPTY list plus a reason.
        empty = select_g_history_evidence(
            _history([], candidate=candidate), candidate=candidate,
            task_id="t")
        self.assertEqual(empty["cases"], [])
        self.assertTrue(empty["missing"])

    def test_repeat_attempts_of_one_task_are_not_independent_tasks(self):
        rows = [_pool_row(f"ex{i}", task_id="same") for i in range(6)]
        selected, report = _diversify(rows, pool_size=30,
                                      max_task_attempts=2)
        self.assertEqual(len(selected), 2)
        self.assertEqual(report["n_distinct_tasks"], 1)
        self.assertEqual(report["same_task_extra_attempts"], 1)

    def test_task_representative_selection_prefers_failure_repair(self):
        rows = [
            _pool_row("ex-ok", task_id="same", status="optimal"),
            _pool_row("ex-fail", task_id="same", status="error"),
            _pool_row("ex-other", task_id="other"),
        ]
        selected, _report = _diversify(rows, pool_size=30,
                                       max_task_attempts=2)
        self.assertEqual({row["execution_id"] for row in selected},
                         {"ex-ok", "ex-other", "ex-fail"})

    def test_degraded_embedding_backend_is_explicit_not_keyword_matching(self):
        class _NoIndex:
            embedding_index = None
        block = build_history_retrieval(
            _NoIndex(), task={"task_id": "t", "text": "task"},
            profile={"family": "assignment"}, candidates=[])
        self.assertEqual(block["status"], "degraded")
        self.assertTrue(block["degraded"])
        self.assertIn("keyword", block["degraded"][0]["reason"])
        self.assertEqual(block["pool"], [])
        # Global statistics do NOT depend on the embedding channel.
        self.assertIn("g", block["global"])

    def test_config_records_thresholds_backend_and_definition(self):
        config = history_config(None)
        self.assertEqual(config["max_local_cases"], 10)
        self.assertIn("problem", config["similarity_thresholds"])
        self.assertIn("candidate_method", config["similarity_thresholds"])
        self.assertIn("conditions", config["similarity_thresholds"])
        self.assertTrue(config["score_definition"])
        self.assertEqual(config["admission_rule"],
                         "a recalled execution enters the pool when at "
                         "least one query view reaches ITS floor")


# ---------------------------------------------------------------------------
# G: candidate-method relevance, no keyword claim, unknown label kept
# ---------------------------------------------------------------------------

class TestGHistoryEvidence(unittest.TestCase):
    def test_cross_family_method_similar_case_is_not_excluded(self):
        candidate = _candidate()
        row = _pool_row("ex1", task_id="other-task", structural="different_cell",
                        method=0.70,
                        evaluation=_g_evaluation(observed=0))
        block = select_g_history_evidence(
            _history([row], candidate=candidate), candidate=candidate,
            task_id="t")
        self.assertEqual(len(block["cases"]), 1)
        self.assertEqual(block["cases"][0]["relevance"], "similar_method")
        self.assertEqual(block["cases"][0]["g"]["observed"], 0)

    def test_generic_word_overlap_is_not_a_method_match(self):
        candidate = _candidate()
        # Admitted through the problem view only: the candidate-method view is
        # below its own floor, so this must NOT be called a method match.
        row = _pool_row("ex1", problem=0.90, method=0.40)
        block = select_g_history_evidence(
            _history([row], candidate=candidate), candidate=candidate,
            task_id="t")
        self.assertEqual(len(block["cases"]), 1)
        self.assertNotEqual(block["cases"][0]["relevance"], "similar_method")
        self.assertEqual(block["cases"][0]["relevance"], "task_similar")
        self.assertFalse(block["cases"][0].get("shared_method_terms"))

    def test_unknown_g_label_is_carried_as_unknown_not_failure(self):
        candidate = _candidate()
        execution_error = _g_evaluation(observed=None)
        execution_error["g"] = {"observed": None,
                                "reason": "execution error: no G label"}
        block = select_g_history_evidence(
            _history([_pool_row("ex1", evaluation=execution_error)],
                     candidate=candidate),
            candidate=candidate, task_id="t")
        case = block["cases"][0]
        self.assertIsNone(case["g"]["observed"])
        self.assertIn("reason", case["g"])
        self.assertNotIn("recorded_cause", case)

    def test_recorded_cause_is_copied_only_when_recorded(self):
        candidate = _candidate()
        attribution = {"method_fidelity": "faithful",
                       "failure_source": "modelling",
                       "source": "auditor", "assessment": "wrong premise"}
        block = select_g_history_evidence(
            _history([_pool_row(
                "ex1", evaluation=_g_evaluation(observed=0,
                                                attribution=attribution))],
                candidate=candidate),
            candidate=candidate, task_id="t")
        self.assertEqual(block["cases"][0]["recorded_cause"]
                         ["failure_source"], "modelling")

    def test_global_g_is_diagnostic_and_has_counts_and_error(self):
        pairs = [
            _pair("ev1", observed=1, predicted=0.9),
            _pair("ev2", task_id="t2", episode_id="e2", observed=0,
                  predicted=0.7),
            _pair("ev3", task_id="t3", episode_id="e3", observed=1,
                  predicted=0.5),
        ]
        stats = global_g_statistics(pairs)
        self.assertEqual(stats["n_observed"], 3)
        self.assertEqual(stats["n_distinct_tasks"], 3)
        self.assertEqual(stats["n_distinct_episodes"], 3)
        self.assertAlmostEqual(stats["mae"], (0.1 + 0.7 + 0.5) / 3, places=6)
        self.assertAlmostEqual(stats["brier"],
                               (0.01 + 0.49 + 0.25) / 3, places=6)
        self.assertIn(stats["support"], ("measured",
                                         "insufficient_independent_tasks"))
        self.assertNotIn("candidate_probability", stats)
        self.assertIn("diagnostic", stats["note"])

    def test_global_g_unknown_is_counted_with_a_reason(self):
        pair = _pair("ev1", observed=None)
        pair["benefit"] = {"metric": FAITHFUL_G_METRIC,
                           "observation_rule_version":
                               FAITHFUL_G_OBSERVATION_VERSION,
                           "reason": "no attributable label"}
        stats = global_g_statistics([pair])
        self.assertEqual(stats["n_observed"], 0)
        self.assertEqual(stats["unknown"]["n"], 1)


# ---------------------------------------------------------------------------
# C: no phase, four dimensions, missing stays missing, grade separation
# ---------------------------------------------------------------------------

class TestCHistoryEvidence(unittest.TestCase):
    def test_current_conditions_carry_no_phase_but_keep_artifacts(self):
        from or_harness.world_model.history_retrieval import (
            _current_cost_conditions,
        )
        context = _context(artifacts={"status": "available",
                                      "path": "ws/model.py"})
        current = _current_cost_conditions(context, _candidate())
        self.assertNotIn("phase", current)
        self.assertNotIn("phase_basis", current)
        self.assertTrue(current["reusable_artifacts"])

    def test_cost_conditions_helper_has_no_phase_and_keeps_artifact(self):
        from or_harness.world_model.cost_signal import cost_conditions
        from or_harness.world_model.context import PredictionContext
        from or_harness.world_model.context import (
            JointProblemRepresentation,
        )
        context = PredictionContext(
            context_id="ctx", task_id="t", task_digest="d",
            joint=JointProblemRepresentation(
                "t", "text", profile={"family": "assignment",
                                      "problem_id": "t"}),
            solving_context={"task_progress": {"model_artifact": {
                "value": {"status": "available", "path": "m.py"}}}}
        )
        conditions = cost_conditions(
            context, SimpleNamespace(solver="highs", config={}))
        self.assertNotIn("phase", conditions)
        self.assertNotIn("phase_basis", conditions)
        self.assertEqual(conditions["scope"], "remaining_to_task_end")
        self.assertTrue(conditions["reusable_artifacts"])

    def test_local_cases_use_four_dimensions_and_missing_is_not_zero(self):
        candidate = _candidate()
        case = _c_evaluation(
            values={"tool_calls": 12, "output_tokens": 4000},
            predicted_actual={
                "tool_calls": {"predicted": 10, "actual": 12},
                "retries": {"predicted": 1, "actual": None}})
        block = select_c_history_evidence(
            _history([_pool_row("ex1", evaluation=case)],
                     candidate=candidate),
            context=_context(), candidate=candidate)
        self.assertEqual(len(block["cases"]), 1)
        item = block["cases"][0]
        self.assertEqual(item["cost"], {"tool_calls": 12,
                                        "output_tokens": 4000})
        self.assertEqual(set(item["missing_dimensions"]),
                         {"remaining_latency_s", "retries"})
        for dim in ("remaining_latency_s", "retries"):
            self.assertNotIn(dim, item["cost"])
        self.assertNotIn(0, item["cost"].values())
        self.assertEqual(item["grade"], "strict")

    def test_scope_mismatch_is_unavailable_and_not_relabelled(self):
        candidate = _candidate()
        attempt_scope = _c_evaluation(values={"tool_calls": 3},
                                      scope="attempt")
        block = select_c_history_evidence(
            _history([_pool_row("ex1", evaluation=attempt_scope)],
                     candidate=candidate),
            context=_context(), candidate=candidate)
        self.assertEqual(block["cases"], [])
        self.assertEqual(block["reference_coverage"]["unavailable"], 1)
        self.assertEqual(block["evidence_status"], "insufficient_evidence")

    def test_broader_reference_is_separated_from_strict(self):
        candidate = _candidate()
        strict_case = _c_evaluation("ev-strict", values={"tool_calls": 4})
        broad_case = _c_evaluation("ev-broad", values={"tool_calls": 9},
                                   family="routing")
        block = select_c_history_evidence(
            _history([_pool_row("ex1", evaluation=strict_case),
                      _pool_row("ex2", task_id="t2",
                                evaluation=broad_case)],
                     candidate=candidate),
            context=_context(), candidate=candidate)
        grades = {case["grade"] for case in block["cases"]}
        self.assertEqual(grades, {"strict", "broader_reference"})
        coverage = block["reference_coverage"]
        self.assertEqual(coverage["strict"]["n_cases_included"], 1)
        self.assertEqual(coverage["broader_reference"]["n_cases_included"], 1)
        self.assertEqual(block["evidence_status"], "strict_and_broader")

    def test_global_cost_uses_actuals_and_prediction_pairs(self):
        pairs = [
            _pair("ev1", cost_values={"tool_calls": 10,
                                      "remaining_latency_s": 100.0},
                  cost_predicted_actual={
                      "tool_calls": {"predicted": 8, "actual": 10},
                      "remaining_latency_s": {"predicted": 80.0,
                                              "actual": 100.0}}),
            _pair("ev2", task_id="t2", episode_id="e2",
                  cost_values={"tool_calls": 20,
                               "remaining_latency_s": 200.0},
                  cost_predicted_actual={
                      "tool_calls": {"predicted": 30, "actual": 20}}),
            _pair("ev3", task_id="t3", episode_id="e3",
                  cost_values={"tool_calls": 30}),
        ]
        stats = global_c_statistics(pairs)
        tool = stats["dimensions"]["tool_calls"]
        self.assertEqual(tool["n"], 3)
        self.assertEqual(tool["p50"], 20)
        self.assertIsNotNone(tool["p90"])
        self.assertEqual(tool["n_predicted_actual"], 2)
        # predicted 8 vs actual 10 and 30 vs 20 -> the median direction is
        # not fabricated into a single multiplier; both values are carried.
        self.assertIn(tool["error_direction"],
                      ("actual_higher", "predicted_higher", "none"))
        latency = stats["dimensions"]["remaining_latency_s"]
        self.assertEqual(latency["n"], 2)
        self.assertEqual(latency["n_predicted_actual"], 1)
        # A dimension with NO complete observation stays visible as an
        # incomplete count, never as a fabricated value.
        output = stats["dimensions"]["output_tokens"]
        self.assertEqual(output["n"], 0)
        self.assertEqual(output["n_incomplete"], 3)
        self.assertNotIn("p50", output)

    def test_incomplete_observation_is_counted_not_zero_filled(self):
        pair = _pair("ev1", cost_values={"tool_calls": 5},
                     retry_rule="wm-c-retries-legacy/1")
        stats = global_c_statistics([pair])
        tool = stats["dimensions"]["tool_calls"]
        self.assertEqual(tool["n"], 1)
        self.assertEqual(tool["n_incomplete"], 0)
        self.assertEqual(stats["dimensions"]["remaining_latency_s"]["n"], 0)
        self.assertEqual(
            stats["dimensions"]["remaining_latency_s"]["n_incomplete"], 1)
        retries = stats["dimensions"]["retries"]
        self.assertEqual(retries["n"], 0)
        self.assertEqual(retries["n_incomplete"], 1)
        self.assertIn("legacy_retry_observation_rule",
                      retries["incomplete_reasons"])

    def test_decision_cost_scales_use_global_p90_and_configured_wins(self):
        history = _history([], c={"dimensions": {
            "tool_calls": {"n": 4, "p50": 10, "p90": 25}}})
        context = _context(history=history)
        scales = decision_cost_scales(context, [_candidate()])
        self.assertEqual(scales["tool_calls"]["peak"], 25)
        self.assertEqual(scales["tool_calls"]["source"],
                         "pre_decision_measured_history")
        self.assertEqual(scales["tool_calls"]["n"], 4)
        configured = decision_cost_scales(
            context, [_candidate()], {"tool_calls": 99})
        self.assertEqual(configured["tool_calls"]["peak"], 99)
        self.assertEqual(configured["tool_calls"]["source"],
                         "explicit_plan_limits")


# ---------------------------------------------------------------------------
# R: technical cases, global rates, no joint probability
# ---------------------------------------------------------------------------

class TestRHistoryEvidence(unittest.TestCase):
    def test_local_technical_case_carries_three_events_and_labels(self):
        candidate = _candidate()
        row = _pool_row(
            "ex1", method=0.70,
            evaluation=_r_evaluation(events={
                "implementation_failure": 1,
                "timeout": 0, "environment_failure": None},
                predicted={"implementation_failure": 0.2}))
        block = select_r_history_evidence(
            _history([row], candidate=candidate,
                     r={"events": {"implementation_failure": {
                         "occurred": 1, "not_occurred": 2,
                         "n_known_labels": 3, "occurrence_rate": 0.333}}}),
            context=_context(), candidate=candidate)
        self.assertEqual(len(block["cases"]), 1)
        events = block["cases"][0]["events"]
        self.assertEqual(events["implementation_failure"]["label"], 1)
        self.assertEqual(events["implementation_failure"]["predicted"], 0.2)
        self.assertIsNone(events["environment_failure"]["label"])
        self.assertEqual(block["support"],
                         "local_cases_with_global_base_rates")
        self.assertNotIn("joint", block)

    def test_global_rates_stay_separate_from_local_cases(self):
        candidate = _candidate()
        global_r = {"version": "wm-r-global/1",
                    "events": {"implementation_failure": {
                        "occurred": 2, "not_occurred": 3,
                        "n_known_labels": 5, "occurrence_rate": 0.4}},
                    "scope": "closed window"}
        block = select_r_history_evidence(
            _history([_pool_row("ex1")], candidate=candidate, r=global_r),
            context=_context(), candidate=candidate)
        self.assertEqual(block["global_event_base_rates"]
                         ["implementation_failure"]["occurred"], 2)
        self.assertTrue(block["cases"])
        self.assertNotIn("joint_event_probability", block)

    def test_global_r_keeps_unknown_out_of_the_denominator(self):
        summary = {"technical_r": {
            "observation_rule_version": R_OBSERVATION_VERSION,
            "coverage": {"n_actual_attempts": 4},
            "events": {
                "implementation_failure": {
                    "occurred": 2, "not_occurred": 3, "unknown": 7,
                    "n_known_labels": 5, "occurrence_rate": 0.4,
                    "n_scorable": 4, "mean_brier": 0.2,
                    "n_scored_distinct_episodes": 2,
                    "discrimination": 0.3},
            }}}
        stats = global_r_statistics(summary)
        entry = stats["events"]["implementation_failure"]
        self.assertEqual(entry["n_known_labels"], 5)
        self.assertEqual(entry["occurrence_rate"], 0.4)
        self.assertEqual(entry["mean_brier"], 0.2)
        self.assertNotIn("rate_with_unknown_as_negative", entry)
        self.assertIn("independent", stats["note"])

    def test_global_r_without_labels_is_insufficient_not_zero_risk(self):
        summary = {"technical_r": {"coverage": {"n_actual_attempts": 1},
                                   "events": {}}}
        stats = global_r_statistics(summary)
        self.assertEqual(stats["support"], "insufficient_evidence")
        self.assertIn("not zero risk", stats["note"])


# ---------------------------------------------------------------------------
# provider budget: caps, coverage refresh, globals preserved
# ---------------------------------------------------------------------------

class TestProviderBudget(unittest.TestCase):
    def _c_block(self, count=12):
        cases = [{"execution_id": f"ex{i}", "grade": "strict",
                  "task_id": f"t{i}", "cost": {"tool_calls": i + 1},
                  "missing_dimensions": []} for i in range(count)]
        return {"version": "wm-c-history/1", "cases": cases,
                "reference_coverage": {"strict": {"n_cases_available": count,
                                                  "n_cases_included": count},
                                       "broader_reference": {
                                           "n_cases_available": 0,
                                           "n_cases_included": 0}},
                "global_cost": {"dimensions": {"tool_calls": {
                    "n": 4, "p50": 3, "p90": 8}}},
                "evidence_status": "strict",
                "missing": []}

    def test_cost_trim_keeps_global_values_and_refreshes_coverage(self):
        from or_harness.world_model.request_budget import trim_evidence_block
        trimmed, info = trim_evidence_block(self._c_block(12), 1200,
                                            kind="cost_evidence")
        self.assertLessEqual(len(trimmed["cases"]), 6)
        self.assertIn("budget_trim", trimmed)
        coverage = trimmed["reference_coverage"]
        self.assertEqual(coverage["strict"]["n_cases_included"],
                         len(trimmed["cases"]))
        self.assertEqual(coverage["strict"]["n_cases_omitted"], 12 -
                         len(trimmed["cases"]))
        self.assertEqual(trimmed["global_cost"]["dimensions"]
                         ["tool_calls"]["p90"], 8)
        self.assertTrue(info["removed"])

    def test_gcr_summary_keeps_global_and_drops_case_bodies(self):
        from or_harness.world_model.request_budget import (
            _gcr_evidence_summary,
        )
        summary = _gcr_evidence_summary("cost_evidence", self._c_block(3))
        self.assertEqual(summary["cases"], [])
        self.assertEqual(summary["global_cost"]["dimensions"]
                         ["tool_calls"]["p50"], 3)
        self.assertTrue(summary["omitted_for_size"])

    def test_refresh_never_claims_evidence_after_empty_trim(self):
        block = self._c_block(2)
        block["cases"] = []
        refresh_c_history_coverage(block)
        self.assertEqual(block["evidence_status"], "insufficient_evidence")
        self.assertEqual(block["reference_coverage"]["strict"]
                         ["n_cases_included"], 0)


# ---------------------------------------------------------------------------
# end-to-end: existing embedding index -> frozen context -> provider view
# ---------------------------------------------------------------------------

class TestEndToEndHistory(GCase):
    def test_local_and_global_history_reach_the_next_request(self):
        first = self.predict_g(COUPLED, episode="ep1")
        execution = self.run_g(first)
        self.check_g(execution)
        self.close_g("ep1")
        # A DIFFERENT method on the same task: the problem view must still
        # surface the past attempt (similarity is not a method-identity gate).
        self.predict_g(SPLIT, episode="ep2")
        context = self.provider.requests[-1]["prediction_context"]
        history = context["history_retrieval"]
        self.assertEqual(history["status"], "ok")
        self.assertGreaterEqual(history["accounting"]["pool_size"], 1)
        self.assertEqual(history["accounting"]["distinct_tasks"], 1)
        self.assertIn("candidate:0", history["queries"])
        # G and R carry LOCAL cases + GLOBAL background.
        g_block = context["g_candidate_evidence"]
        self.assertTrue(g_block["cases"])
        self.assertTrue(g_block["global_history"])
        self.assertEqual(g_block["cases"][0]["g"]["observed"], 1)
        r_block = context["implementation_r_evidence"]
        self.assertTrue(r_block["cases"])
        self.assertTrue(r_block["global_event_base_rates"])
        # C reports its state honestly: no fabricated cost case is invented
        # when the closed window carries no complete remaining-cost fact.
        c_block = context["cost_evidence"]
        self.assertIn("global_cost", c_block)
        self.assertEqual(c_block["current_conditions"]["phase_used"], False)

    def test_prediction_time_cutoff_excludes_later_executions(self):
        from or_harness.world_model.prediction import ActionSpec
        first = self.predict_g(COUPLED, episode="ep1")
        execution = self.run_g(first)
        self.check_g(execution)
        self.close_g("ep1")
        spec = ActionSpec(action_type="execute_strategy",
                          task_id=self.task["task_id"], episode_id="ep2",
                          strategy_id="S04", solver="highs",
                          params={}, method=SPLIT)
        profile = self.h.profile(self.task)
        before = build_history_retrieval(
            self.h, task=self.task, profile=profile, candidates=[spec])
        self.assertGreaterEqual(before["accounting"]["pool_size"], 1)
        blocked = build_history_retrieval(
            self.h, task=self.task, profile=profile, candidates=[spec],
            cutoff=float(execution.created_at) - 1.0)
        self.assertEqual(blocked["pool"], [])
        self.assertEqual(blocked["accounting"]["excluded_after_cutoff"], 1)


# ---------------------------------------------------------------------------
# P0-1: GLOBAL G calibration is computed on ONE aligned scored sample
# ---------------------------------------------------------------------------

class TestGlobalGAlignment(unittest.TestCase):
    @staticmethod
    def _pair(evaluation_id, *, observed, predicted=None, task_id="t1",
              episode_id="e1"):
        benefit = {"observed": observed,
                   "metric": FAITHFUL_G_METRIC,
                   "observation_rule_version":
                       FAITHFUL_G_OBSERVATION_VERSION}
        if predicted is not None:
            benefit["predicted"] = predicted
            benefit["signed_error"] = predicted - observed
        return {"evaluation_id": evaluation_id, "task_id": task_id,
                "episode_id": episode_id, "benefit": benefit}

    def test_bias_direction_positive_means_over_predicted(self):
        # predicted 0.9, observed 0 -> the model predicted HIGHER.
        stats = global_g_statistics([self._pair("ev1", observed=0,
                                                predicted=0.9)])
        self.assertEqual(stats["bias_direction"], "over_predicted")
        self.assertGreater(stats["signed_error_mean"], 0)
        # predicted 0.3, observed 1 -> the model predicted LOWER.
        stats = global_g_statistics([self._pair("ev2", observed=1,
                                                predicted=0.3)])
        self.assertEqual(stats["bias_direction"], "under_predicted")
        self.assertLess(stats["signed_error_mean"], 0)

    def test_predicted_and_observed_means_use_the_same_scored_sample(self):
        pairs = [
            self._pair("ev0", observed=1),              # label, no prediction
            self._pair("ev1", observed=0, predicted=0.9),
            self._pair("ev2", observed=1, predicted=0.3),
        ]
        stats = global_g_statistics(pairs)
        self.assertEqual(stats["n_observed"], 3)
        self.assertEqual(stats["n_scored"], 2)
        self.assertEqual(stats["n_observed_without_prediction"], 1)
        # Both means come from the SAME two scored pairs, not from 3 vs 2.
        self.assertEqual(stats["observed_mean"], 0.5)
        self.assertEqual(stats["predicted_mean"], 0.6)
        self.assertEqual(stats["observed_without_prediction_mean"], 1.0)
        self.assertEqual(stats["mae"], (0.9 + 0.7) / 2)

    def test_reliability_bins_never_pair_by_array_index(self):
        pairs = [self._pair("ev-observed-only", observed=0)]
        # Four high-probability pairs, all observed failures; four
        # low-probability pairs, all observed successes.
        for index in range(4):
            pairs.append(self._pair(f"hi{index}", observed=0,
                                    predicted=0.95))
        for index in range(4):
            pairs.append(self._pair(f"lo{index}", observed=1,
                                    predicted=0.05))
        stats = global_g_statistics(pairs)
        bins = {round(entry["lo"], 2): entry
                for entry in stats["reliability_bins"]}
        high = bins[0.8]
        low = bins[0.0]
        self.assertEqual((high["n"], high["observed_rate"]), (4, 0.0))
        self.assertEqual((low["n"], low["observed_rate"]), (4, 1.0))
        self.assertEqual(stats["brier"], 0.95 ** 2)


# ---------------------------------------------------------------------------
# P0-2: only the measurement window is a hard exclusion
# ---------------------------------------------------------------------------

class TestCostComparabilityDeductions(unittest.TestCase):
    def _case(self, evaluation_id, *, solver="highs", values=None,
              family="assignment", structure=None, config=None):
        case = _c_evaluation(evaluation_id,
                             values=values or {"tool_calls": 8},
                             family=family, structure=structure)
        case["cost"]["actual_solving_conditions"] = [{
            "solver": solver, "config": dict(config or {})}]
        return case

    def test_different_solver_is_kept_as_weak_reference(self):
        candidate = _candidate(solver="highs")
        # Historical case solved with CBC, same remaining-task window.
        cbc = self._case("ev-cbc", solver="cbc")
        block = select_c_history_evidence(
            _history([_pool_row("ex1", evaluation=cbc)],
                     candidate=candidate),
            context=_context(), candidate=candidate)
        self.assertEqual(len(block["cases"]), 1)
        case = block["cases"][0]
        self.assertEqual(case["grade"], "broader_reference")
        self.assertIn("solver highs vs cbc", case["differences"])
        self.assertLess(case["relevance_weight"], 1.0)
        self.assertEqual(case["cost"]["tool_calls"], 8)
        self.assertEqual(block["reference_coverage"]["unavailable"], 0)

    def test_different_structure_and_config_are_differences_not_exclusions(self):
        candidate = _candidate(config={"time_limit": 60})
        history_case = self._case(
            "ev-other-structure", values={"tool_calls": 20},
            structure={"problem_class": "scheduling"},
            config={"time_limit": 180})
        block = select_c_history_evidence(
            _history([_pool_row("ex1", evaluation=history_case)],
                     candidate=candidate),
            context=_context(), candidate=candidate)
        case = block["cases"][0]
        self.assertEqual(case["grade"], "broader_reference")
        self.assertIn("structure.problem_class", case["differences"])
        self.assertIn("solving config time_limit", case["differences"])
        # Strict cases rank first; a difference lowers relevance but keeps
        # the observed value available.
        self.assertEqual(case["cost"]["tool_calls"], 20)

    def test_scope_mismatch_stays_the_only_hard_exclusion(self):
        candidate = _candidate()
        attempt = _c_evaluation("ev-attempt", values={"tool_calls": 3},
                                scope="attempt")
        block = select_c_history_evidence(
            _history([_pool_row("ex1", evaluation=attempt)],
                     candidate=candidate),
            context=_context(), candidate=candidate)
        self.assertEqual(block["cases"], [])
        self.assertEqual(block["reference_coverage"]["unavailable"], 1)


# ---------------------------------------------------------------------------
# P0-3: the task-vector path re-applies the vector-recall validity rules
# ---------------------------------------------------------------------------

class _FakeBank:
    def __init__(self, records):
        self.records = records

    def get(self, execution_id):
        return self.records.get(execution_id)


class _FakeStore:
    def __init__(self, text):
        self.text = text

    def get_task_text(self, task_id, digest):
        return self.text


class _FakeHarness:
    def __init__(self, record, text):
        self.bank = _FakeBank({record.execution_id: record})
        self.store = _FakeStore(text)


def _record(execution_id="ex1", *, source="executed", created_at=100.0):
    return ExecutionRecord(
        execution_id=execution_id, task_id="t", strategy_id="S04",
        source=source, created_at=created_at,
        task_text_digest="text-digest",
        profile_snapshot=ProblemProfile(problem_id="t", family="assignment"),
        quality={"status": "optimal", "feasible": True},
        solver={"name": "highs"},
        method_planned={"name": "m", "steps": ["s"]},
    )


class TestPoolValidity(unittest.TestCase):
    TEXT = "Solve the assignment task."

    def _deepcopy_record(self):
        import copy as _copy
        record = _record()
        return record, _copy.deepcopy(record)

    def _call(self, record, *, expected=None, cutoff=None, source_record=None):
        harness = _FakeHarness(source_record or record, self.TEXT)
        recalled = {"execution_id": record.execution_id,
                    "task_id": "t", "strategy_id": "S04",
                    "task_text_excerpt": self.TEXT, "method": {},
                    "profile_cell": None}
        return _resolve_pool_row(harness, recalled,
                                 view_scores={"problem": 0.9},
                         floors={"problem": 0.62},
                         cutoff=cutoff, query_cell=None,
                         expected_doc_digest=expected)

    def test_stale_index_document_is_excluded(self):
        record = _record()
        row, reason = self._call(record, expected="not-the-current-digest")
        self.assertIsNone(row)
        self.assertEqual(reason, "stale_index")

    def test_matching_document_digest_is_admitted(self):
        record = _record()
        expected = document_digest(document_execution(record, self.TEXT))
        row, reason = self._call(record, expected=expected)
        self.assertIsNotNone(row)
        self.assertIsNone(reason)
        self.assertEqual(row["execution_id"], "ex1")

    def test_non_executed_source_is_excluded(self):
        record = _record(source="agent_reported")
        row, reason = self._call(record)
        self.assertIsNone(row)
        self.assertEqual(reason, "invalid_source")

    def test_record_after_cutoff_is_excluded(self):
        record = _record(created_at=100.0)
        row, reason = self._call(record, cutoff=50.0)
        self.assertIsNone(row)
        self.assertEqual(reason, "after_cutoff")


# ---------------------------------------------------------------------------
# P1: one evaluation is attributed to the BOUND execution / one C carrier
# ---------------------------------------------------------------------------

class TestEvidenceAttribution(unittest.TestCase):
    def _pair(self):
        return {
            "evaluation_id": "ev1",
            "task_id": "t1",
            "episode_id": "e1",
            "execution_ids": ["ex1", "ex2"],
            "execution_conditions": {"execution_id": "ex1",
                                     "comparable": True},
            "benefit": {"observed": 1, "predicted": 0.7,
                        "metric": FAITHFUL_G_METRIC,
                        "observation_rule_version":
                            FAITHFUL_G_OBSERVATION_VERSION},
            "cost_observed_fact": {
                "values": {"tool_calls": 10},
                "conditions": {
                    "scope": "remaining_to_task_end",
                    "window_coverage": "complete",
                    "family": "assignment",
                    "structure": {"problem_class": "milp"},
                    "solver_conditions": {"constraints": {}},
                    "actual_solving_conditions": [{"solver": "highs",
                                                   "config": {}}],
                    "retry_observation_rule_version":
                        RETRIES_FUTURE_VERSION,
                },
                "source_refs": ["ex1", "ex2"],
            },
            "risk_actual": [{"event": "implementation_failure", "label": 1,
                             "brier": 0.09, "label_basis": "fixture"}],
        }

    def _history_with_pair(self):
        bound = _pool_row("ex1", method=0.8)
        other = _pool_row("ex2", method=0.8)
        pair = self._pair()
        attached = _attach_evaluation_facts([bound, other], [pair])
        self.assertEqual(attached["bound_rows"], 1)
        self.assertEqual(attached["cost_rows"], 1)
        candidate = _candidate()
        return _history([bound, other], candidate=candidate), candidate

    def test_g_label_is_attributed_only_to_the_bound_execution(self):
        history, candidate = self._history_with_pair()
        block = select_g_history_evidence(history, candidate=candidate,
                                          task_id="t")
        labelled = [case for case in block["cases"]
                    if (case.get("g") or {}).get("observed") in (0, 1)]
        self.assertEqual(len(labelled), 1)
        self.assertEqual(labelled[0]["execution_id"], "ex1")
        self.assertEqual(labelled[0]["bound_execution_id"], "ex1")
        self.assertEqual(labelled[0]["evaluation_id"], "ev1")
        # The other execution in the window is still a method case, but its
        # G label is UNKNOWN and it carries no evaluation id.
        other = next(case for case in block["cases"]
                     if case["execution_id"] == "ex2")
        self.assertIsNone(other["g"]["observed"])
        self.assertNotIn("evaluation_id", other)

    def test_cost_window_is_one_reference_with_covers_ids(self):
        history, candidate = self._history_with_pair()
        block = select_c_history_evidence(history, context=_context(),
                                          candidate=candidate)
        self.assertEqual(len(block["cases"]), 1)
        case = block["cases"][0]
        self.assertEqual(case["execution_id"], "ex1")
        self.assertEqual(case["covers_execution_ids"], ["ex1", "ex2"])
        self.assertEqual(case["cost"]["tool_calls"], 10)

    def test_r_risk_label_is_attributed_only_to_the_bound_execution(self):
        history, candidate = self._history_with_pair()
        block = select_r_history_evidence(history, context=_context(),
                                          candidate=candidate)
        bound = next(case for case in block["cases"]
                     if case["execution_id"] == "ex1")
        other = next(case for case in block["cases"]
                     if case["execution_id"] == "ex2")
        self.assertEqual(bound["evidence_status"], "closed_pair")
        self.assertEqual(bound["evaluation_id"], "ev1")
        self.assertEqual(bound["events"]["implementation_failure"]["label"], 1)
        # The non-bound execution keeps ITS record-derived event label.
        self.assertEqual(other["evidence_status"], "execution_only")
        self.assertNotIn("evaluation_id", other)


# ---------------------------------------------------------------------------
# compatibility: the active four-dimensional C prediction contract
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
            for dim in ("remaining_latency_s", "output_tokens", "retries"):
                self.assertNotIn(dim,
                                 prediction.cost.expected.measured_dims())
        finally:
            case.doCleanups()

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
        self.assertEqual(validate_strategy_outcome(restored), [])


if __name__ == "__main__":
    unittest.main()
