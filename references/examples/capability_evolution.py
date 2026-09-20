"""Runnable example: the M5 capability-evolution loop.

Runs with NO real model, NO network, NO API key and NO embedding service
(a fixed-output stub provider plus the deterministic offline hashing
backend). It walks the WHOLE slow-time-scale loop in order, and it is the
demonstration the milestone asks for:

1. **frozen input + real candidate** — closed task experience is turned
   into a REAL induction candidate bundle (the existing C1-C6 scan), the
   capability evidence, the task targeting and the baseline are frozen, and
   a capability prediction is requested under ``wm-ce/1``;
2. **the protocol really reaches the provider** — a LOCAL HTTP STUB SERVER
   receives the request, confirms the capability system prompt and schema
   were selected, and answers; the parsed prediction is saved and restored;
3. **prediction changes the advice** — TWO candidates are predicted (one
   large saving, one small) and the bounded comparison rule recommends the
   larger NET saving, i.e. its saving minus its own predicted maintenance
   cost IN THE SAME UNIT. A third candidate that trades quality away is NOT
   auto-ranked; a candidate whose maintenance cost is quoted in another
   currency, or that costs more than it saves, is reported rather than
   ranked; and a candidate with no comparable yardstick yields ``defer``;
4. **explicit choice, then the real operation** — the recommendation is
   accepted EXPLICITLY and the existing induction runs on the prediction's
   OWN frozen scope; a second candidate is REJECTED and changes nothing;
5. **facts are not effects** — the real maintenance fact is bound (what
   changed, what it really cost), and the capability effect is still
   PENDING;
6. **the effect is judged on REAL later results** — a later task's closed
   episode is evaluated, a pre-arranged paired comparison makes the change
   attributable, and the effect becomes VERIFIED — while W_OR stays
   unadvanced and a repeat evaluation does not double the sample. The
   evaluation only counts tasks that finished AFTER the operation, fall
   inside the prediction's FROZEN target, and are not part of its own
   experience scope; and it only reaches a verdict once the declared
   horizon is met;
7. **honest branches** — a candidate whose operation changes nothing is
   recorded as ``no_change``; a prediction the real evidence CONTRADICTS is
   recorded as refuted; a candidate that was never accepted produces no
   fabricated result.

This fixture proves the PROCESS is wired end to end. It does NOT prove the
prediction model is accurate, and it is NOT evidence that the harness got
stronger: the numbers are a stub's, and the milestone's point is that the
framework keeps prediction, fact and effect apart.

    PYTHONPATH=src python3 references/examples/capability_evolution.py
"""
from __future__ import annotations

import json
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from or_harness.api import ORHarness  # noqa: E402
from or_harness.strategy.embedding_index import (  # noqa: E402
    LocalHashEmbeddingBackend,
)
from or_harness.world_model.capability_evolution import (  # noqa: E402
    CAPABILITY_EVOLUTION_PROTOCOL_VERSION,
    CAPABILITY_EVOLUTION_SYSTEM_PROMPT,
)
from or_harness.world_model.contracts import (  # noqa: E402
    BaselineStatement,
    TaskTargeting,
)
from or_harness.world_model.provider import (  # noqa: E402
    HttpChatProvider,
    WorldModelProvider,
)

TASK_TEXT = ("A distribution centre must be loaded before the delivery "
             "window opens; demand of 100 units may not be deferred.")

#: The stub's capability payload: one quantified resource saving whose
#: quality does NOT degrade (quality is expected to improve modestly), a
#: learning cost, a degradation risk and a verification condition. It
#: carries NO claim that anything was verified.
#:
#: The learning cost is expressed in the SAVING's own unit (seconds): a
#: comparison only nets quantities in the same currency, so a candidate
#: whose maintenance cost is quoted in tokens is reported as incomparable
#: rather than ranked against a saving in seconds.
PAYLOAD = {
    "expected_changes": [
        {"metric": "resource_cost", "unit": "s", "direction": "decrease",
         "value": -6.0, "value_kind": "absolute",
         "beneficial_direction": "decrease"},
        {"metric": "normalized_solution_quality", "unit": "1-gap",
         "direction": "increase", "beneficial_direction": "increase"},
    ],
    "learning_cost": {"solver_runtime_s": 1.0},
    "degradation_risk": {"events": [
        {"event": "overgeneralized_entry", "probability": 0.15,
         "basis": "the evidence spans two structural cells"},
        {"event": "revised_performance_regression"},
    ]},
    "uncertainty": {"knowledge_gap": 0.55, "execution_randomness": 0.2},
    "verification_conditions": [
        {"condition": ("later matching tasks need less solver time while "
                       "quality holds"),
         "check_basis": "solver_runtime_s and normalized quality on the "
                        "next 5 unseen tasks",
         "evaluable": True},
    ],
    "evidence_basis": ["capability_evidence.sources.m",
                       "learning_material.executions[0]"],
}


class StubProvider(WorldModelProvider):
    """A fixed-output provider that records what it was asked."""

    name = "example-m5-stub"

    def __init__(self, payload=PAYLOAD):
        self.payload = payload
        self.requests = []

    def predict(self, request, timeout_s=None):
        self.requests.append(request)
        return {"payload": self.payload,
                "usage": {"prompt_tokens": 120, "completion_tokens": 60},
                "error": None, "latency_s": 0.02}


def _task(task_id):
    return {
        "task_id": task_id,
        "family": "routing",
        "description": TASK_TEXT,
        "spec": {"n_vars": 400, "n_constraints": 250, "n_int_vars": 400},
        "annotations": {"coupling": {"resource_coupling": 0.3,
                                     "temporal_coupling": 0.1,
                                     "route_complexity": 0.2,
                                     "semantic_coupling": 0.5}},
    }


def _seed_evidence(h, tasks, strategy="S04"):
    """Real executed records: the evidence a candidate can be built on."""
    from or_harness.core.schema import CostVector, ExecutionRecord, \
        ProblemProfile

    ids = []
    for index, task_id in enumerate(tasks):
        profile = ProblemProfile(
            problem_id=task_id, family="routing",
            scale_features={"n_vars": 400.0, "n_constraints": 250.0,
                            "n_int_vars": 400.0, "density": 0.01},
            semantic_coupling=0.5, resource_coupling=0.3,
            temporal_coupling=0.1, route_complexity=0.2)
        record = ExecutionRecord(
            execution_id=f"ex_{task_id}_{strategy}",
            task_id=task_id, strategy_id=strategy,
            profile_snapshot=profile,
            quality={"feasible": True, "objective": 100.0, "gap": 0.05,
                     "status": "optimal"},
            cost=CostVector(llm_tokens=900, tool_calls=3,
                            solver_runtime_s=5.0, retries=0, latency_s=1.2,
                            measured={"llm_tokens", "tool_calls",
                                      "solver_runtime_s", "retries",
                                      "latency_s"}),
            solver={"name": "highs", "family": "milp"},
            source="executed")
        h.bank.append(record)
        ids.append(record.execution_id)
    return ids


def _close_later_episode(h, task_id, *, quality_value):
    """Run one REAL later task and close its episode.

    The bound strategy prediction's observed quality is what the M4
    close-out records; the M5 effect evaluation reads exactly that.
    """
    from or_harness.world_model.provider import WorldModelProvider

    class QualityProvider(WorldModelProvider):
        name = "example-m5-quality"

        def predict(self, request, timeout_s=None):
            return {"payload": {
                "benefit": {"kind": "solution_quality",
                            "metric": "normalized_objective_gap",
                            "unit": "1-gap", "value": quality_value,
                            "baseline": {"kind": "conditional_stats",
                                         "value": 0.7}},
                "cost": {"solver_runtime_s": 4.0},
                "risk": {"events": []}},
                "usage": {"prompt_tokens": 20, "completion_tokens": 20},
                "error": None, "latency_s": 0.01}

    task = _task(task_id)
    provider = QualityProvider()
    # A SECOND harness on the same home: the prediction and the execution
    # are ordinary OR-side records, which is what the effect evaluation
    # reads. (The capability prediction lives in the same store.)
    from or_harness.api import ORHarness as H
    orx = H(home=h.home, world_model=provider,
            embedding=LocalHashEmbeddingBackend())
    prediction = orx.predict_strategy_outcome(
        task, {"action_type": "execute_strategy", "strategy_id": "S04"},
        "ep1")
    work = Path(h.home) / f"ws_later_{task_id}"
    work.mkdir(parents=True, exist_ok=True)
    (work / "solve.py").write_text(
        "import json\n"
        "with open('result.json', 'w') as fh:\n"
        "    json.dump({'status': 'optimal', 'objective_value': 1.0,"
        " 'objective_bound': 1.0, 'runtime_seconds': 0.01}, fh)\n",
        encoding="utf-8")
    record = orx.execute(task, "S04", str(work / "solve.py"), str(work),
                         solver="highs", episode_id="ep1")
    orx.record(record)
    orx.bind_strategy_outcome(prediction.prediction_id, record.action_id)
    result = orx.close_episode(task_id, "ep1")
    orx.close()
    return result


def _local_http_check(home):
    """One REAL HTTP round trip through a local stub server.

    Confirms the capability protocol selected the capability prompt and
    schema on the wire — with no 30B model, no remote service and no key.
    """
    captured = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            captured["system"] = body["messages"][0]["content"]
            captured["request"] = json.loads(body["messages"][1]["content"])
            reply = json.dumps({
                "choices": [{"message": {"content": json.dumps(PAYLOAD)}}],
                "usage": {"prompt_tokens": 30, "completion_tokens": 40},
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
    try:
        provider = HttpChatProvider(
            base_url=f"http://127.0.0.1:{server.server_port}",
            model="local-stub", api_key="not-a-real-key")
        h = ORHarness(home=home, world_model=provider,
                      embedding=LocalHashEmbeddingBackend())
        prediction = h.predict_capability_evolution(
            {"operation_type": "induce", "strategy_id": "S04"},
            task=_task("http_check"), horizon="the next matching task",
            horizon_tasks=1,
            # A fresh home has no candidate bundle, so the framework's
            # identity (target + baseline) is supplied explicitly here. The
            # framework still OWNS these fields: the model may not restate
            # them, and the request records them as fixed.
            task_targeting=TaskTargeting(
                description="routing tasks matching the S04 cell",
                family="routing"),
            baseline=BaselineStatement(
                kind="conditional_stats", value=5.0,
                note="the observed solver runtime before the operation"))
        h.close()
    finally:
        server.shutdown()
    return captured, prediction


def main() -> int:
    tmp = tempfile.TemporaryDirectory()
    home = tmp.name
    backend = LocalHashEmbeddingBackend()
    provider = StubProvider()
    h = ORHarness(home=home, world_model=provider, embedding=backend)

    # ------------------------------------------------------------------
    print("=" * 72)
    print("1. Real closed experience -> a REAL candidate bundle")
    print("=" * 72)
    tasks = ["m5_a", "m5_b", "m5_c"]
    _seed_evidence(h, tasks)
    bundles = h.induction_candidates()
    print(f"candidates built       : {len(bundles)}")
    assert bundles, ("the seeded evidence must produce a real candidate "
                     "bundle through the existing C1-C6 scan")
    bundle = bundles[0]
    print(f"bundle                 : {bundle['bundle_id']} "
          f"({bundle['kind']}, strategy {bundle['strategy_id']})")
    print(f"scope                  : {len(bundle['execution_ids'])} "
          f"execution(s) over {len(bundle['tasks'])} task(s)")
    print(f"frozen baseline        : mean_quality="
          f"{bundle['mean_quality']}")

    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print("2. The capability protocol really reaches the provider")
    print("=" * 72)
    http_home = tempfile.TemporaryDirectory()
    captured, http_prediction = _local_http_check(http_home.name)
    print(f"wire protocol          : "
          f"{captured['request']['prediction_protocol']}")
    print(f"capability prompt      : "
          f"{'wm-ce/1' in captured['system']}")
    print(f"request kind           : {captured['request']['request_kind']}")
    print(f"fixed by framework     : "
          f"{captured['request']['output_contract']['fixed_by_framework']}")
    assert captured["request"]["prediction_protocol"] == \
        CAPABILITY_EVOLUTION_PROTOCOL_VERSION
    assert captured["system"] == CAPABILITY_EVOLUTION_SYSTEM_PROMPT
    assert http_prediction.status == "valid"

    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print("3. Predictions under a FIXED rule, and the honest branches")
    print("=" * 72)
    big = h.predict_capability_evolution(
        {"operation_type": "induce", "strategy_id": "S04"},
        task=_task("m5_a"), bundle=bundle,
        horizon="the next matching task", horizon_tasks=1)
    print(f"prediction             : {big.prediction_id} "
          f"(status {big.status})")
    for change in big.expected_changes:
        flag = ("" if change.is_improvement is None
                else " improvement" if change.is_improvement
                else " DEGRADATION")
        print(f"  expected change      : {change.metric} "
              f"{change.direction} {change.value} {change.unit}{flag}")
    print(f"  learning cost        : "
          f"{big.learning_cost.expected.to_dict()} (PREDICTED, not real "
          "spend; kept in the SAVING's own unit so the two are nettable)")
    print(f"  degradation risk     : "
          f"{[e.event for e in big.degradation_risk.events]}")
    print(f"  uncertainty source   : {big.uncertainty.source} "
          f"(self-reported, uncalibrated)")
    condition = big.verification_conditions[0]
    print(f"  verification         : prediction_made="
          f"{condition.prediction_made}, fact_bound={condition.fact_bound}, "
          f"effect_verified={condition.effect_verified}")
    assert big.status == "valid"
    assert condition.effect_verified is False

    # A smaller saving, so the comparison has two comparable candidates.
    small_provider = StubProvider(payload={
        "expected_changes": [
            {"metric": "resource_cost", "unit": "s",
             "direction": "decrease", "value": -1.5,
             "beneficial_direction": "decrease"}],
        "learning_cost": {"solver_runtime_s": 0.5},
        "verification_conditions": [{"condition": "cost falls",
                                     "evaluable": True}]})
    small_h = ORHarness(home=home, world_model=small_provider,
                        embedding=backend)
    small = small_h.predict_capability_evolution(
        {"operation_type": "induce", "strategy_id": "S04"},
        task=_task("m5_a"), bundle=bundle,
        horizon="the next matching task", horizon_tasks=1)
    small_h.close()

    recommendation = h.compare_capability_evolution(
        [big.prediction_id, small.prediction_id], horizon_tasks=1)
    print(f"recommendation         : {recommendation['recommendation']} -> "
          f"{recommendation['selected_prediction_id']}")
    print(f"rule                   : "
          f"{recommendation['rule']['name']}")
    print(f"basis                  : {recommendation['basis']}")
    for entry in recommendation["comparisons"]:
        print(f"  compared             : {entry['prediction_id']} "
              f"saving/task={entry['saving']['per_task']} "
              f"total={entry['saving']['total']}")
    assert recommendation["recommendation"] == "accept"
    assert recommendation["selected_prediction_id"] == big.prediction_id
    assert h.sbank.count() == 0, ("comparing must not touch the Strategic "
                                  "Bank")

    # -- a candidate that trades quality away is NOT auto-ranked ---------
    risky_provider = StubProvider(payload={
        "expected_changes": [
            {"metric": "resource_cost", "unit": "s",
             "direction": "decrease", "value": -20.0,
             "beneficial_direction": "decrease"},
            {"metric": "normalized_solution_quality", "unit": "1-gap",
             "direction": "decrease", "beneficial_direction": "increase"}],
        "learning_cost": {"solver_runtime_s": 1.0},
        "verification_conditions": [{"condition": "cost falls",
                                     "evaluable": True}]})
    risky_h = ORHarness(home=home, world_model=risky_provider,
                        embedding=backend)
    risky = risky_h.predict_capability_evolution(
        {"operation_type": "induce", "strategy_id": "S04"},
        task=_task("m5_a"), bundle=bundle,
        horizon="the next matching task", horizon_tasks=1)
    risky_h.close()
    quality_result = h.compare_capability_evolution(
        [risky.prediction_id, small.prediction_id], horizon_tasks=1)
    print(f"quality-loss candidate : NOT auto-ranked "
          f"({quality_result['incomparable'][0]['reason'][:60]}...)")
    assert quality_result["selected_prediction_id"] == small.prediction_id

    # -- a candidate with no comparable yardstick -> defer ----------------
    quality_provider = StubProvider(payload={
        "expected_changes": [
            {"metric": "normalized_solution_quality", "unit": "1-gap",
             "direction": "increase", "value": 0.1,
             "beneficial_direction": "increase"}],
        "learning_cost": {"llm_tokens": 400},
        "verification_conditions": [{"condition": "quality rises",
                                     "evaluable": True}]})
    quality_h = ORHarness(home=home, world_model=quality_provider,
                          embedding=backend)
    quality_only = quality_h.predict_capability_evolution(
        {"operation_type": "induce", "strategy_id": "S04"},
        task=_task("m5_a"), bundle=bundle,
        horizon="the next matching task", horizon_tasks=1)
    quality_h.close()
    defer_result = h.compare_capability_evolution(
        [quality_only.prediction_id], horizon_tasks=1)
    print(f"no comparable yardstick: {defer_result['recommendation']} — "
          "defer is a legitimate outcome, not a failure")
    assert defer_result["recommendation"] == "defer"
    assert defer_result["incomparable"]

    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print("4. Explicit choice -> the REAL operation (scope never widened)")
    print("=" * 72)
    declined = h.reject_capability_operation(
        defer_result, prediction_id=quality_only.prediction_id,
        reason="quality-only gain has no comparable yardstick this round")
    print(f"declined               : {declined['rejection_action_id']} "
          f"(strategic_bank_touched={declined['strategic_bank_touched']})")
    assert declined["strategic_bank_touched"] is False
    assert h.sbank.count() == 0, "declining must change nothing"

    accepted = h.accept_capability_operation(recommendation)
    print(f"accepted               : {accepted['adoption_action_id']}")
    print(f"scope used             : {accepted['execution_ids']}")
    assert sorted(accepted["execution_ids"]) == sorted(
        bundle["execution_ids"]), ("the operation must run on the "
                                  "prediction's OWN frozen scope")
    print(f"knowledge after        : {h.sbank.count()} entr(y/ies)")
    assert h.sbank.count() >= 1, "the accepted operation really changed " \
                                 "knowledge"

    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print("5. Stage 1 — bind the FACT (never the effect)")
    print("=" * 72)
    bound = h.bind_capability_maintenance(big.prediction_id)
    binding = bound["binding"]
    print(f"stage                  : {binding['stage']}")
    print(f"changed knowledge      : {binding['changed']} "
          f"(created={len(binding['knowledge_delta']['entries_created'])})")
    print(f"scope consistent       : {binding['scope_consistent']} "
          f"(problems={binding['scope_problems']})")
    print(f"real cost              : "
          f"{(binding['real_cost'] or {}).get('per_dim')}")
    print(f"verification           : {binding['verification']['note'][:60]}"
          "...")
    effect = h.capability_effect_evaluation(big.prediction_id)
    print(f"effect evaluation      : {effect} "
          "(a bound fact produces NO effect verdict)")
    assert effect is None
    assert binding["changed"] is True

    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print("6. Stage 2 — the EFFECT, judged on REAL later results")
    print("=" * 72)
    pending = h.evaluate_capability_effect(big.prediction_id)
    print(f"before later tasks     : {pending['evaluation']['state']} — "
          "the horizon is unmet, and pending stays re-evaluable")
    assert pending["evaluation"]["state"] == "pending"

    # A later task, OUTSIDE the prediction's own experience scope.
    # The declared horizon is ONE matching task, which is what this fixture
    # can really close: a claim over ten tasks that only ever sees one is
    # INCOMPLETE, and the evaluation says so instead of deciding early.
    _close_later_episode(h, "m5_later", quality_value=0.95)
    descriptive = h.evaluate_capability_effect(big.prediction_id)
    print(f"without a reference    : "
          f"{descriptive['evaluation']['state']} — a before/after movement "
          "with no comparable control is DESCRIPTIVE, not causal")
    assert descriptive["evaluation"]["state"] == "inconclusive"
    assert descriptive["evaluation"]["effect_verified"] is False

    paired = h.record_capability_paired_evaluation(
        big.prediction_id, metric="normalized_solution_quality",
        reference_value=0.70, treated_value=0.95, unit="1-gap",
        reference_task_ids=["m5_later"],
        note="a paired window the caller really arranged")
    print(f"paired reference       : change={paired['change']} "
          f"{paired['unit']} (source={paired['source']})")

    verified = h.evaluate_capability_effect(big.prediction_id)
    evaluation = verified["evaluation"]
    print(f"effect state           : {evaluation['state']} "
          f"(effect_verified={evaluation['effect_verified']})")
    for change in evaluation["changes"]:
        print(f"  channel              : {change['channel']} "
              f"predicted={change['predicted_direction']} "
              f"observed_change={change.get('observed_change')} "
              f"agreement={change.get('agreement')}")
    assert evaluation["effect_verified"] is True
    assert evaluation["source_evidence"]["m"]["status"] == "direct_evidence"
    assert evaluation["source_evidence"]["w_or"]["status"] != \
        "direct_evidence", ("W_OR is never advanced by a capability "
                            "prediction's own report")

    repeat = h.evaluate_capability_effect(big.prediction_id)
    print(f"repeat evaluation      : already_evaluated="
          f"{repeat['already_evaluated']} (the sample is not counted twice)")
    assert repeat["already_evaluated"] is True

    evidence = h.capability_evidence_with_effects()
    print(f"capability evidence    : m="
          f"{evidence.sources['m'].status}, w_or="
          f"{evidence.sources['w_or'].status}, t="
          f"{evidence.sources['t'].status}")

    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print("7. Honest branches: no-change, refuted, and never-accepted")
    print("=" * 72)
    # A candidate whose operation produces NOTHING.
    empty_provider = StubProvider()
    empty_h = ORHarness(home=home, world_model=empty_provider,
                        embedding=backend)
    empty = empty_h.predict_capability_evolution(
        {"operation_type": "induce", "strategy_id": "S04"},
        task=_task("m5_a"), bundle=bundle,
        horizon="the next matching task", horizon_tasks=1)
    empty_h.close()
    h.actions.report_action(
        "induce", "m5_a", "m5_empty_adopt",
        params={"capability_prediction_id": empty.prediction_id,
                "action": "accepted"},
        outcome={"accepted": True,
                 "capability_prediction_id": empty.prediction_id,
                 "operation_result": {
                     "business_result": "no_candidate",
                     "knowledge_delta": {"entries_created": [],
                                         "entry_changes": []},
                     "execution_ids": list(bundle["execution_ids"])}},
        status="completed")
    empty_bound = h.bind_capability_maintenance(empty.prediction_id)
    print(f"no-change operation    : changed="
          f"{empty_bound['binding']['changed']} — recorded, not a gain")
    assert empty_bound["binding"]["changed"] is False
    empty_effect = h.evaluate_capability_effect(empty.prediction_id)
    print(f"  effect               : {empty_effect['evaluation']['state']}")
    assert empty_effect["evaluation"]["state"] == "no_change"

    # A prediction the REAL evidence contradicts: the candidate expects a
    # COST decrease, but the paired window moved the other way.
    contradicted_provider = StubProvider(payload={
        "expected_changes": [
            {"metric": "resource_cost", "unit": "s", "direction": "decrease",
             "value": -6.0, "beneficial_direction": "decrease"}],
        "learning_cost": {"solver_runtime_s": 1.0},
        "verification_conditions": [{"condition": "cost falls",
                                     "evaluable": True}]})
    contradicted_h = ORHarness(home=home, world_model=contradicted_provider,
                               embedding=backend)
    contradicted = contradicted_h.predict_capability_evolution(
        {"operation_type": "induce", "strategy_id": "S04"},
        task=_task("m5_a"), bundle=bundle,
        horizon="the next matching task", horizon_tasks=1)
    contradicted_h.close()
    recommendation2 = h.compare_capability_evolution(
        [contradicted.prediction_id], horizon_tasks=1)
    if recommendation2["recommendation"] != "accept":
        recommendation2 = {
            "recommendation": "accept",
            "recommendation_id": recommendation2["recommendation_id"],
            "selected_prediction_id": contradicted.prediction_id,
            "basis": "explicit agent choice"}
    h.accept_capability_operation(recommendation2)
    h.bind_capability_maintenance(contradicted.prediction_id)
    h.record_capability_paired_evaluation(
        contradicted.prediction_id, metric="resource_cost",
        reference_value=5.0, treated_value=9.0, unit="s",
        reference_task_ids=["m5_later"],
        note="the paired window moved the other way")
    refuted = h.evaluate_capability_effect(contradicted.prediction_id)
    print(f"refuted prediction     : {refuted['evaluation']['state']} — a "
          "negative result is kept, never dropped")
    assert refuted["evaluation"]["state"] in ("no_change",
                                              "observed_degradation")

    # A candidate that was NEVER accepted: no fact, no fabricated result.
    never = h.predict_capability_evolution(
        {"operation_type": "induce", "strategy_id": "S04"},
        task=_task("m5_a"), bundle=bundle,
        horizon="the next matching task", horizon_tasks=1)
    unbound = h.bind_capability_maintenance(never.prediction_id)
    print(f"never-accepted         : {unbound['state']} — nothing is "
          "recorded as if it ran")
    assert unbound["state"] == "not_adopted"
    never_effect = h.evaluate_capability_effect(never.prediction_id)
    print(f"  effect               : "
          f"{never_effect['evaluation']['state']}")
    assert never_effect["evaluation"]["state"] == "not_evaluable"

    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print("8. The two-stage feedback state, and what was NOT run")
    print("=" * 72)
    summary = h.capability_feedback_summary()
    print(f"predictions            : {summary['n_predictions']}")
    print(f"facts bound            : {summary['n_fact_bound']}")
    print(f"effects VERIFIED       : {summary['n_effect_verified']} "
          "(only a real later-task observation under a comparable setup)")
    for entry in summary["predictions"]:
        print(f"  {entry['prediction_id']}: fact_bound="
              f"{entry['fact_bound']}, changed={entry['changed']}, "
              f"effect={entry['effect_state']}, paired="
              f"{entry['paired_reference']}")
    print()
    print("NOT RUN in this example (and not claimed):")
    print("  * no real 30B model, no remote service, no API key — the")
    print("    numbers come from a stub, so nothing here is evidence that")
    print("    the prediction model is ACCURATE")
    print("  * no counterfactual replay: the paired reference is a record")
    print("    the caller supplies, not something the framework fabricated")
    print("  * no claim that the harness got stronger: a bound fact says")
    print("    the operation happened, and an effect verdict says real")
    print("    later performance moved under a comparable setup")
    print()
    print("M5 loop OK: predict -> compare -> explicit choice -> real")
    print("operation -> bind the fact -> verify the effect on real results.")

    h.close()
    http_home.cleanup()
    tmp.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
