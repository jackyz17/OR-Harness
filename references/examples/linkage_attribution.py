"""Runnable example: a missing execution receipt no longer discards a real
attempt (bounded attribution).

The gap this example demonstrates: `orx predict-strategy` proposed a
configuration and the run really happened, but the solve script did not read
some of those settings back into `result.json`. The old close-out treated
"the action could not CONFIRM this config key" the same as "the action KNOWN
to differ from the prediction": it marked the whole sample
`identity_mismatch` and dropped it from calibration. One missing receipt, one
discarded observation.

This example shows the corrected rule, ONE field at a time:

1. **an unreported effort knob is a caveat** — the run never reported
   `time_limit`/`seed`. Nothing is blocked: the measured `solver_runtime_s`
   is compared and the benefit is scored. The missing receipt is REPORTED,
   not punished;
2. **an unreported approach knob blocks the benefit, keeps the cost** — the
   run never reported `relaxation`, so the answer may not be the predicted
   approach's answer. The benefit is not scored; the real spend still is;
3. **a condition deviation keeps the cost** — the run really reported a
   DIFFERENT value for a key the prediction fixed (`time_limit=30` vs
   predicted 60) or used a different solver. The quality is not the original
   condition's performance, but the attempt really cost what it cost, so the
   cost sample survives;
4. **the genuine boundaries still refuse** — a wrong strategy is not this
   prediction's attempt at all, and the whole sample is excluded. An
   unexecuted candidate gets no counterfactual outcome.

It runs with NO real model (a fixed-output stub provider), NO network, NO
API key and NO embedding service (the deterministic offline hashing backend
is injected).

    PYTHONPATH=src python3 references/examples/linkage_attribution.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from or_harness.api import ORHarness  # noqa: E402
from or_harness.strategy.embedding_index import (  # noqa: E402
    LocalHashEmbeddingBackend,
)
from or_harness.world_model.provider import WorldModelProvider  # noqa: E402

TASK = {
    "task_id": "linkage_demo",
    "family": "production_planning",
    "description": ("A workshop maximises profit under machining and "
                    "finishing limits; every quantity is a whole number."),
    "spec": {"n_vars": 2, "n_constraints": 2, "n_int_vars": 2},
    "annotations": {"coupling": {"resource_coupling": 0.3,
                                 "temporal_coupling": 0.1,
                                 "route_complexity": 0.2,
                                 "semantic_coupling": 0.5}},
}

PAYLOAD = {
    "benefit": {"kind": "solution_quality",
                "metric": "normalized_objective_gap", "unit": "1-gap",
                "value": 0.8,
                "baseline": {"kind": "conditional_stats", "value": 0.7}},
    "cost": {"solver_runtime_s": 3.0},
}


class StubProvider(WorldModelProvider):
    name = "example-linkage-stub"

    def predict(self, request, timeout_s=None):
        return {"payload": PAYLOAD,
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
                "error": None, "latency_s": 0.02}


def _script(h, tag, *, config=None, method=None, status="optimal"):
    """A solve script that stamps any receipt it writes with OR_ACTION_ID."""
    work = Path(h.home) / f"ws_{tag}"
    work.mkdir(parents=True, exist_ok=True)
    extra = ""
    if config is not None:
        cfg = dict(config)
        cfg["action_id"] = "__STAMP__"
        extra += (f"cfg = {json.dumps(cfg)}\n"
                  "cfg['action_id'] = os.environ.get('OR_ACTION_ID')\n")
    if method is not None:
        m = dict(method)
        m["action_id"] = "__STAMP__"
        extra += (f"method = {json.dumps(m)}\n"
                  "method['action_id'] = os.environ.get('OR_ACTION_ID')\n")
    body = ("import json, os\n" + extra +
            f"payload = {{'status': '{status}', 'objective_value': 100.0,\n"
            "           'objective_bound': 100.0, 'runtime_seconds': 0.01}\n"
            "if 'cfg' in dir(): payload['config'] = cfg\n"
            "if 'method' in dir(): payload['method_performed'] = method\n"
            "with open('result.json', 'w') as fh:\n"
            "    json.dump(payload, fh)\n")
    script = work / "solve.py"
    script.write_text(body, encoding="utf-8")
    return script, work


def _evaluate(h, prediction_id, script, work, *, solver="highs",
              run_strategy="S01"):
    """Run, record, bind explicitly, close, and return the evaluation.

    The bind is explicit so the run may legitimately differ from the
    prediction (a deviation); the auto-binding path refuses a contradiction
    at entry by design.
    """
    record = h.execute(TASK, run_strategy, str(script), str(work),
                       solver=solver, episode_id="ep1")
    h.record(record)
    h.bind_strategy_outcome(prediction_id, record.action_id)
    result = h.close_episode(TASK["task_id"], "ep1")
    return result["evaluations"][0]


def main() -> int:
    tmp = tempfile.TemporaryDirectory()
    h = ORHarness(home=tmp.name, world_model=StubProvider(),
                  embedding=LocalHashEmbeddingBackend())

    # ------------------------------------------------------------------
    print("=" * 72)
    print("1. An unreported effort knob (time_limit, seed) is a CAVEAT")
    print("=" * 72)
    prediction = h.predict_strategy_outcome(
        TASK, {"action_type": "execute_strategy", "strategy_id": "S01",
               "solver": "highs", "config": {"time_limit": 60, "seed": 42}},
        "ep1")
    # The script reports NO config at all: nothing to confirm the knobs with.
    script, work = _script(h, "caveat")
    evaluation = _evaluate(h, prediction.prediction_id, script, work)
    print(f"benefit eligibility    : "
          f"{evaluation['benefit']['eligibility']}")
    print(f"cost eligibility       : {evaluation['cost']['eligibility']}")
    print(f"cost dimensions scored : {sorted(evaluation['cost']['per_dim'])}")
    print(f"state                  : {evaluation['state']}")
    print(f"attribution (blocked)  : {evaluation.get('attribution') or '{}'}")
    # The whole sample survives: nothing is blocked by a missing receipt.
    assert evaluation["state"] == "evaluated"
    assert evaluation["benefit"]["eligibility"] == "evaluable"
    assert "solver_runtime_s" in evaluation["cost"]["per_dim"]
    assert not evaluation.get("attribution")

    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print("2. An unreported APPROACH knob (relaxation) blocks the benefit")
    print("=" * 72)
    prediction2 = h.predict_strategy_outcome(
        TASK, {"action_type": "execute_strategy", "strategy_id": "S02",
               "solver": "highs", "config": {"relaxation": "lp"}},
        "ep2")
    script2, work2 = _script(h, "approach")
    record2 = h.execute(TASK, "S02", str(script2), str(work2),
                        solver="highs", episode_id="ep2")
    h.record(record2)
    h.bind_strategy_outcome(prediction2.prediction_id, record2.action_id)
    result2 = h.close_episode(TASK["task_id"], "ep2")
    evaluation2 = result2["evaluations"][0]
    print(f"benefit eligibility    : "
          f"{evaluation2['benefit']['eligibility']}")
    print(f"  blocked by           : "
          f"{evaluation2['benefit'].get('identity_fields')}")
    print(f"cost eligibility       : {evaluation2['cost']['eligibility']}")
    print(f"state                  : {evaluation2['state']}")
    # The approach could not be confirmed: the benefit is not scored, but
    # the real spend is preserved.
    assert evaluation2["benefit"]["eligibility"] == "identity_unknown"
    assert "config.relaxation" in evaluation2["benefit"]["identity_fields"]
    assert evaluation2["cost"]["eligibility"] == "evaluable"
    assert evaluation2["state"] == "evaluated"

    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print("3. A DEVIATION (solver changed) keeps the cost, drops the outcome")
    print("=" * 72)
    prediction3 = h.predict_strategy_outcome(
        TASK, {"action_type": "execute_strategy", "strategy_id": "S03",
               "solver": "glpk"}, "ep3")
    script3, work3 = _script(h, "deviation")
    # The run really used highs, not the predicted glpk.
    record3 = h.execute(TASK, "S03", str(script3), str(work3),
                        solver="highs", episode_id="ep3")
    h.record(record3)
    bound3 = h.bind_strategy_outcome(prediction3.prediction_id,
                                     record3.action_id)
    print(f"binding mismatch       : "
          f"{bound3.trace.model_info.get('binding_mismatch')}")
    result3 = h.close_episode(TASK["task_id"], "ep3")
    evaluation3 = result3["evaluations"][0]
    print(f"benefit eligibility    : "
          f"{evaluation3['benefit']['eligibility']}")
    print(f"cost eligibility       : {evaluation3['cost']['eligibility']}")
    print(f"state                  : {evaluation3['state']}")
    # A different solver is a deviation: the outcome goes, the spend stays.
    assert evaluation3["benefit"]["eligibility"] == "identity_mismatch"
    assert evaluation3["cost"]["eligibility"] == "evaluable"
    assert evaluation3["state"] == "evaluated"

    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print("4. Genuine boundaries still refuse")
    print("=" * 72)
    prediction4 = h.predict_strategy_outcome(
        TASK, {"action_type": "execute_strategy", "strategy_id": "S04"},
        "ep4")
    script4, work4 = _script(h, "wrong_strategy")
    # A DIFFERENT strategy really ran.
    record4 = h.execute(TASK, "S09", str(script4), str(work4),
                        solver="highs", episode_id="ep4")
    h.record(record4)
    h.bind_strategy_outcome(prediction4.prediction_id, record4.action_id)
    result4 = h.close_episode(TASK["task_id"], "ep4")
    evaluation4 = result4["evaluations"][0]
    print(f"state                  : {evaluation4['state']}")
    print(f"exclusion reasons      : {evaluation4['exclusion_reasons']}")
    # A wrong strategy is not this prediction's attempt at all.
    assert evaluation4["state"] == "excluded"
    assert not evaluation4["calibratable"]

    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print("5. Entry-point discipline: a METHOD in `config` is re-homed")
    print("=" * 72)
    prediction5 = h.predict_strategy_outcome(
        TASK, {"action_type": "execute_strategy", "strategy_id": "S05",
               "config": {"method": "benders", "time_limit": 60}}, "ep5")
    print(f"candidate.config       : {prediction5.candidate.config}")
    print(f"candidate.method       : {prediction5.candidate.method}")
    print(f"recorded normalization : "
          f"{prediction5.trace.model_info.get('config_normalized')}")
    assert prediction5.candidate.config == {"time_limit": 60}
    assert prediction5.candidate.method == "benders"
    # A key that names no execution parameter is refused BEFORE an attempt.
    try:
        h.predict_strategy_outcome(
            TASK, {"action_type": "execute_strategy", "strategy_id": "S06",
                   "config": {"task_id": "other"}}, "ep5")
        raise AssertionError("a non-execution config key should be refused")
    except ValueError as exc:
        print(f"refused                : {exc}")
        assert "not an execution parameter" in str(exc)

    h.close()
    tmp.cleanup()
    print()
    print("All assertions passed.")
    print("NOTE: a missing receipt is a CAVEAT, not a discard. The close-out "
          "blocks only the comparison a problem really invalidates — an "
          "unreported time_limit blocks nothing, a different solver blocks "
          "the outcome and keeps the cost — while a wrong task, a wrong "
          "strategy or a post-hoc prediction still excludes the sample.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
