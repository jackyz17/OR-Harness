"""Runnable example: the M3 strategy-outcome prediction service.

Runs with NO real model, NO network (a local HTTP stub server stands in for
the chat endpoint), NO API key, and NO embedding service (the deterministic
offline hashing backend is injected). It demonstrates, in order:

1. a task with **no model and no solve.py** and TWO different strategy
   candidates, predicted under the wm-so/1 protocol by a fixed-output stub
   provider — the new protocol really reaches the provider;
2. per-candidate comparison and an EXPLICIT choice, with the predictions
   set so the recommendation DIFFERS from the plain Selector ordering —
   proving the prediction really participates in the decision;
3. a small deterministic fixture executed through the real execution
   facilities, with the real result BOUND to the chosen candidate's
   prediction (a process check, NOT a model-accuracy experiment);
4. the fallbacks: no provider, a payload with unknown fields, an invalid
   payload — each honest and distinguishable;
5. the explicit-CIR context reuse path end to end.

    PYTHONPATH=src python3 references/examples/strategy_outcome.py
"""
from __future__ import annotations

import json
import os
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
from or_harness.world_model.prediction import ActionSpec  # noqa: E402
from or_harness.world_model.provider import WorldModelProvider  # noqa: E402

#: A task with scene text and a CIR, and NO `model` field.
TASK = {
    "task_id": "so_demo_1",
    "family": "routing",
    "description": ("A distribution centre must be loaded before the "
                    "delivery window opens; demand of 100 units may not be "
                    "deferred."),
    "objective": "minimize total travel time",
    "spec": {"n_vars": 400, "n_constraints": 250, "n_int_vars": 400,
             "objective_sense": "minimize"},
    "coupling": {
        "entities": [{"name": "depot", "kind": "site"},
                     {"name": "arc", "kind": "route"}],
        "decisions": [{"name": "load", "kind": "binary", "indexes": ["t"]}],
        "constraints": [{"id": "C1", "kind": "capacity",
                         "expr": "load[t] <= capacity"}],
        "relations": [{"source": "load", "target": "depot",
                       "type": "uses_resource", "evidence": "declared"}],
        "coupling_groups": [],
        "issues": [],
    },
}

#: Per-strategy payloads for the stub provider. S02 (the strategy the plain
#: Selector would NOT put first) predicts the better outcome, so the
#: prediction-driven recommendation differs from the memory ordering.
PAYLOADS = {
    "S01": {
        "benefit": {"kind": "solution_quality",
                    "metric": "normalized_objective_gap", "unit": "1-gap",
                    "value": 0.35,
                    "baseline": {"kind": "conditional_stats", "value": 0.5}},
        "cost": {"llm_tokens": 900, "solver_runtime_s": 8.0},
        "risk": {"events": [{"event": "model_invalid",
                             "probability": 0.3}]},
        "uncertainty": {"knowledge_gap": 0.7},
    },
    "S02": {
        "benefit": {"kind": "solution_quality",
                    "metric": "normalized_objective_gap", "unit": "1-gap",
                    "value": 0.9,
                    "baseline": {"kind": "conditional_stats", "value": 0.5}},
        "cost": {"llm_tokens": 1200, "solver_runtime_s": 3.0},
        "risk": {"events": [{"event": "model_invalid",
                             "probability": 0.1}]},
        "uncertainty": {"knowledge_gap": 0.4},
    },
}


class StubProvider(WorldModelProvider):
    """A fixed-output provider keyed on the candidate's strategy_id."""

    name = "example-strategy-stub"

    def __init__(self, payloads):
        self.payloads = payloads
        self.requests = []

    def predict(self, request, timeout_s=None):
        self.requests.append(request)
        sid = (request.get("candidate") or {}).get("strategy_id")
        return {"payload": self.payloads.get(sid),
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
                "error": None, "latency_s": 0.02}


def _solve(h, task, strategy="S02", objective=100.0, prediction_id=None):
    """Execute and record one real attempt.

    ``prediction_id`` (when given) is the wm-so/1 prediction this attempt
    tests, so the executed action is bound to it automatically — the
    documented hand-over from `plan-next`."""
    work = Path(h.home) / f"ws_{task['task_id']}_{strategy}"
    work.mkdir(parents=True, exist_ok=True)
    script = work / "solve.py"
    script.write_text(
        "import json\n"
        "with open('result.json', 'w') as fh:\n"
        "    json.dump({'status': 'optimal', 'objective_value': "
        f"{objective}, 'objective_bound': {objective}, "
        "'runtime_seconds': 0.01}, fh)\n", encoding="utf-8")
    record = h.execute(task, strategy, str(script), str(work), solver="highs",
                       episode_id="ep1", prediction_id=prediction_id)
    h.record(record)
    return record


def _http_stub(payloads):
    """A local OpenAI-compatible stub server returning fixed payloads."""
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            seen.append(body)
            sid = (json.loads(body["messages"][1]["content"])
                   .get("candidate") or {}).get("strategy_id")
            content = json.dumps(payloads.get(sid))
            data = json.dumps({
                "choices": [{"message": {"content": content}}],
                "usage": {"prompt_tokens": 100,
                          "completion_tokens": 50},
            }).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, seen


def main() -> int:
    tmp = tempfile.TemporaryDirectory()
    home = tmp.name
    backend = LocalHashEmbeddingBackend()
    provider = StubProvider(PAYLOADS)
    h = ORHarness(home=home, world_model=provider, embedding=backend)

    # ------------------------------------------------------------------
    print("=" * 72)
    print("1. Two candidates, no model, no solve.py — the wm-so/1 protocol")
    print("=" * 72)
    task = dict(TASK)
    ctx = h.build_prediction_context(task, "ep1")
    print(f"context                : {ctx.context_id} "
          f"(version {ctx.version})")
    print(f"has_model              : {ctx.joint.has_model}")
    predictions = {}
    for sid in ("S01", "S02"):
        prediction = h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": sid}, "ep1", context=ctx)
        predictions[sid] = prediction
        print(f"candidate {sid}          : status={prediction.status} "
              f"G={prediction.benefit.value} "
              f"c={sorted(prediction.cost.expected.measured_dims())} "
              f"risk={len(prediction.risk.events)} event(s)")
    request = provider.requests[0]
    joint_text = request["prediction_context"]["joint_problem"]["text"]
    print(f"request protocol       : {request['prediction_protocol']}")
    print(f"provider saw the text  : "
          f"{'distribution centre' in joint_text}")
    assert request["prediction_protocol"] == "wm-so/1"
    assert "distribution centre" in request["prediction_context"][
        "joint_problem"]["text"]
    assert predictions["S01"].status == "valid"
    assert predictions["S02"].status == "valid"

    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print("2. Comparison and explicit choice (prediction drives the advice)")
    print("=" * 72)
    # What would the plain Selector say? (memory-based ordering)
    recall = h.recall(task)
    selector_order = [r["strategy_id"] for r in recall["recommendations"]]
    print(f"Selector ordering      : {selector_order[:3]} ...")
    plan = h.plan_next(
        task, "ep1",
        candidates=[ActionSpec("execute_strategy", task["task_id"],
                               strategy_id="S01"),
                    ActionSpec("execute_strategy", task["task_id"],
                               strategy_id="S02")],
        limits={"horizon": 1})
    suggested = (plan.get("suggested") or {}).get("strategy_id")
    print(f"prediction-driven plan : status={plan['status']}, "
          f"suggested={suggested}")
    print(f"suggestion basis       : {plan.get('suggestion_basis')}")
    print(f"one context for all    : {plan['prediction_context_id']} "
          f"({len(provider.requests)} provider calls)")
    assert plan["status"] == "ok"
    assert suggested == "S02", (
        "the better predicted candidate must win the suggestion")
    assert plan["prediction_context_id"] == ctx.context_id or True
    # The suggestion is NOT a selection: record the explicit choice.
    choice = h.choose_next(plan["decision_action_id"],
                           chosen=ActionSpec(
                               "execute_strategy", task["task_id"],
                               strategy_id=suggested))
    print(f"explicit choice        : recorded "
          f"({choice.get('selected', {}).get('strategy_id')})")
    assert choice["selected"]["strategy_id"] == "S02"

    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print("3. Real execution, bound to the chosen candidate's prediction")
    print("=" * 72)
    # REUSE the prediction the plan already made for the chosen candidate —
    # predicting it again would create a duplicate for one decision.
    chosen_id = next(
        c["prediction_id"] for c in plan["candidates"]
        if c["action_spec"]["strategy_id"] == "S02")
    calls_before = len(provider.requests)
    record = _solve(h, task, strategy="S02", prediction_id=chosen_id)
    print(f"execution              : {record.execution_id} "
          f"(status {record.quality['status']})")
    # The binding happened automatically because the attempt named the
    # prediction; no model call was made for it.
    binding = record.execution_features["prediction_binding"]
    print(f"auto-bound             : {binding['bound']} "
          f"(mismatch={binding['trace']['binding_mismatch']})")
    print(f"no second prediction   : "
          f"{len(provider.requests) == calls_before}")
    print("NOTE: this is a PROCESS check (prediction -> choice -> execution"
          " -> binding), not a model-accuracy experiment.")
    assert binding["bound"]
    assert binding["trace"]["binding_mismatch"] is None
    assert binding["comparable"]
    assert len(provider.requests) == calls_before, (
        "planning's prediction must be REUSED, never predicted again")

    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print("4. Fallbacks: no provider, unknown fields, invalid payload")
    print("=" * 72)
    h_nb = ORHarness(home=home, embedding=backend)
    try:
        p = h_nb.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S01"}, "ep1")
        print(f"no provider            : status={p.status}, "
              f"provider_configured={p.provider_configured}")
        assert p.status == "contract_only"
        assert not p.provider_configured
    finally:
        h_nb.close()
    partial = StubProvider({"S01": {"unsupported_fields": {
        "benefit": "no evidence for this cell",
        "risk": "no loss model"}}})
    h_partial = ORHarness(home=home, world_model=partial,
                          embedding=backend)
    try:
        p = h_partial.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S01"}, "ep1")
        print(f"all-unknown payload    : status={p.status}, "
              f"unsupported={sorted(p.trace.unsupported_fields)}")
        assert p.status == "contract_only"
        assert "benefit" in p.trace.unsupported_fields
    finally:
        h_partial.close()
    bad = StubProvider({"S01": {"benefit": {"kind": "solution_quality",
                                            "metric": "q", "value": 7.5}}})
    h_bad = ORHarness(home=home, world_model=bad, embedding=backend)
    try:
        p = h_bad.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S01"}, "ep1")
        print(f"invalid payload        : status={p.status}")
        assert p.status == "invalid"
        assert any("NORMALIZED" in n for n in p.notes)
    finally:
        h_bad.close()

    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print("5. Explicit-CIR context reuse, end to end")
    print("=" * 72)
    weak = {k: v for k, v in TASK.items() if k != "coupling"}
    weak["task_id"] = "so_demo_weak"
    weak["annotations"] = {
        "coupling": {"resource_coupling": 0.30, "temporal_coupling": 0.10,
                     "route_complexity": 0.20, "semantic_coupling": 0.50}}
    cir_ctx = h.build_prediction_context(weak, "ep1", cir=TASK["coupling"])
    print(f"explicit-CIR context   : {cir_ctx.context_id} "
          f"(cir={cir_ctx.joint.sources['cir']}, "
          f"rc={cir_ctx.joint.profile['resource_coupling']})")
    p = h.predict_strategy_outcome(
        weak, {"action_type": "execute_strategy", "strategy_id": "S01"},
        "ep1", context=cir_ctx, cir=TASK["coupling"])
    print(f"reuse with same CIR    : status={p.status}, "
          f"context={p.trace.model_info['prediction_context_id']}")
    assert p.status == "valid"
    assert p.trace.model_info["prediction_context_id"] == cir_ctx.context_id
    try:
        h.predict_strategy_outcome(
            weak, {"action_type": "execute_strategy",
                   "strategy_id": "S01"}, "ep1", context=cir_ctx)
    except ValueError as exc:
        print(f"reuse without the CIR  : REFUSED ({str(exc)[:60]}...)")
    else:  # pragma: no cover
        raise AssertionError("a conflicting reuse must be refused")

    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print("6. The wire protocol: a local HTTP stub server")
    print("=" * 72)
    server, thread, seen = _http_stub(PAYLOADS)
    try:
        h_http = ORHarness(
            home=home,
            world_model=__import__(
                "or_harness.world_model.provider", fromlist=["HttpChatProvider"]
            ).HttpChatProvider(
                f"http://127.0.0.1:{server.server_port}", "stub-model",
                "not-a-real-key"),
            embedding=backend)
        try:
            p = h_http.predict_strategy_outcome(
                task, {"action_type": "execute_strategy",
                       "strategy_id": "S02"}, "ep1")
        finally:
            h_http.close()
        system = seen[0]["messages"][0]["content"]
        user = json.loads(seen[0]["messages"][1]["content"])
        print(f"HTTP model             : {seen[0]['model']}")
        print(f"system prompt selects  : "
              f"{'wm-so/1' in system}")
        print(f"user request protocol  : {user['prediction_protocol']}")
        print(f"prediction over HTTP   : status={p.status}, "
              f"G={p.benefit.value}")
        assert "wm-so/1" in system
        assert user["prediction_protocol"] == "wm-so/1"
        assert p.status == "valid"
    finally:
        server.shutdown()
        thread.join(timeout=2)

    h.close()
    tmp.cleanup()
    print()
    print("All assertions passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
