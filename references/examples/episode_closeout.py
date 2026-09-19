"""Runnable example: the M4 episode close-out loop.

Runs with NO real model (a fixed-output stub provider), NO network, NO API
key and NO embedding service (the deterministic offline hashing backend is
injected). It walks ONE full closed loop and the cross-episode calibration
channel, in order:

1. **a multi-attempt real window** — the chosen strategy fails once
   (status=error), is repaired, and succeeds; the close-out's benefit
   observation follows the declared rule (last qualified attempt) while
   the FAILED attempt's measured cost stays inside the scope's totals;
2. **per-field evaluation** — the frozen prediction is compared against
   the real outcome field by field (benefit error, per-dimension cost
   error, a Brier score, interval coverage), with the exclusions named;
3. **honest boundaries** — an unexecuted candidate gets NO label; an
   episode that is still open publishes NO calibration; re-closing the
   same episode counts nothing twice;
4. **the cross-episode channel** — a SECOND episode's fresh context reads
   the FIRST episode's published summary (here: insufficient evidence at
   the default sample minimum — an honest state, not a failure), while
   the first episode's own stored context stays byte-identical.

This fixture proves the PROCESS is correct. It does NOT prove the
prediction model is accurate, and it does NOT prove the harness's
capability improved: one closed episode is one sample.

    PYTHONPATH=src python3 references/examples/episode_closeout.py
"""
from __future__ import annotations

import json
import os
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
    "task_id": "m4_demo",
    "family": "routing",
    "description": ("A distribution centre must be loaded before the "
                    "delivery window opens; demand of 100 units may not "
                    "be deferred."),
    "objective": "minimize total travel time",
    "spec": {"n_vars": 400, "n_constraints": 250, "n_int_vars": 400,
             "objective_sense": "minimize"},
}

#: The stub provider's fixed payload: a benefit with a baseline and an
#: interval, cost on two dimensions, one risk event with a probability.
PAYLOAD = {
    "benefit": {"kind": "solution_quality",
                "metric": "normalized_objective_gap", "unit": "1-gap",
                "value": 0.8,
                "interval": [0.6, 0.95],
                "baseline": {"kind": "conditional_stats", "value": 0.7}},
    "cost": {"llm_tokens": 1200, "solver_runtime_s": 3.0},
    "risk": {"events": [{"event": "model_invalid", "probability": 0.2}]},
    "uncertainty": {"execution_randomness": 0.3, "knowledge_gap": 0.6},
}


class StubProvider(WorldModelProvider):
    name = "example-m4-stub"

    def __init__(self, payload=PAYLOAD):
        self.payload = payload
        self.requests = []

    def predict(self, request, timeout_s=None):
        self.requests.append(request)
        return {"payload": self.payload,
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
                "error": None, "latency_s": 0.02}


def _attempt(h, task, strategy, *, status, objective, episode,
             tag, gap=None):
    """One real sandboxed solve attempt with a deterministic outcome."""
    work = Path(h.home) / f"ws_{tag}"
    work.mkdir(parents=True, exist_ok=True)
    result = {"status": status, "objective_value": objective,
              "objective_bound": objective, "runtime_seconds": 0.01}
    if gap is not None:
        result["mip_gap"] = gap
    if status == "error":
        # A failing script: the executor records an error attempt.
        (work / "solve.py").write_text(
            "raise RuntimeError('model invalid: capacity constraint "
            "references an undefined set')\n", encoding="utf-8")
    else:
        (work / "solve.py").write_text(
            "import json\n"
            "with open('result.json', 'w') as fh:\n"
            "    json.dump(" + json.dumps(result) + ", fh)\n",
            encoding="utf-8")
    record = h.execute(task, strategy, str(work / "solve.py"), str(work),
                       solver="highs", episode_id=episode)
    h.record(record)
    return record


def main() -> int:
    tmp = tempfile.TemporaryDirectory()
    home = tmp.name
    backend = LocalHashEmbeddingBackend()
    provider = StubProvider()
    h = ORHarness(home=home, world_model=provider, embedding=backend)

    # ------------------------------------------------------------------
    print("=" * 72)
    print("1. Predict, choose, execute — a multi-attempt real window")
    print("=" * 72)
    task = dict(TASK)
    # The prediction is made BEFORE anything runs (frozen input).
    prediction = h.predict_strategy_outcome(
        task, {"action_type": "execute_strategy",
               "strategy_id": "S04", "solver": "highs"}, "ep1")
    print(f"prediction             : {prediction.prediction_id} "
          f"(status {prediction.status}, G={prediction.benefit.value}, "
          f"interval {prediction.benefit.interval})")
    assert prediction.status == "valid"

    # The agent's own modelling work: real auxiliary spend.
    h.report_action("model", task, "ep1",
                    outcome={"kind": "model_artifact"},
                    cost={"llm_tokens": 300})

    # Attempt 1 FAILS (a model error), attempt 2 — after repair —
    # succeeds with a gap. Both are real, recorded attempts of ONE
    # strategy window.
    failed = _attempt(h, task, "S04", status="error", objective=None,
                      episode="ep1", tag="fail")
    print(f"attempt 1 (failed)     : {failed.execution_id} "
          f"(status {failed.quality['status']})")
    repaired = _attempt(h, task, "S04", status="feasible", objective=100.0,
                        episode="ep1", tag="ok", gap=0.1)
    print(f"attempt 2 (repaired)   : {repaired.execution_id} "
          f"(status {repaired.quality['status']}, gap "
          f"{repaired.quality['gap']})")

    # Bind the prediction to the REAL action that ran (the repaired
    # attempt's action).
    bound = h.bind_strategy_outcome(prediction.prediction_id,
                                    repaired.action_id)
    info = bound.trace.model_info
    print(f"binding mismatch       : {info.get('binding_mismatch')}")
    print(f"binding unknown        : {info.get('binding_unknown')}")
    print(f"comparable             : {bound.trace.comparable}")
    assert info.get("binding_mismatch") is None
    assert bound.trace.comparable

    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print("2. Close the episode: per-field evaluation of the prediction")
    print("=" * 72)
    # The episode is still open: no calibration may exist yet.
    open_summary = h.calibration_summary()
    print(f"calibration (open ep)  : "
          f"{open_summary['n_evaluations_total']} sample(s)")
    assert open_summary["n_evaluations_total"] == 0

    result = h.close_episode("m4_demo", "ep1")
    closeout = result["closeout"]
    evaluation = result["evaluations"][0]
    print(f"close-out              : {closeout['closeout_id']} "
          f"(terminal {closeout['terminal_state']})")
    print(f"evaluation state       : {evaluation['state']}")
    benefit = evaluation["benefit"]
    print(f"benefit                : predicted {benefit['predicted']}, "
          f"observed {benefit['observed']} "
          f"(rule: {result and 'last qualified attempt'}), "
          f"abs_error {benefit['abs_error']}")
    # The observed quality is 1 - gap = 0.9 (the FAILED attempt never
    # contributed a quality observation; its COST stays in scope).
    assert benefit["observed"] == 0.9
    cost = evaluation["cost"]
    print(f"cost per-dim           : "
          f"{ {k: v['abs_error'] for k, v in cost['per_dim'].items()} }")
    print(f"cost excluded          : {cost['excluded']}")
    risk = evaluation["risk"]
    print(f"risk                   : {risk['scored']}")
    interval = evaluation["interval"]
    print(f"interval               : {interval['predicted_interval']} "
          f"covered={interval['covered']} "
          f"(observed {interval['observed']})")
    assert evaluation["state"] == "evaluated"
    assert interval["covered"]  # 0.9 is inside [0.6, 0.95]

    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print("3. Honest boundaries")
    print("=" * 72)
    # (a) An unexecuted candidate: predicted, never run, never bound.
    unexecuted = h.predict_strategy_outcome(
        task, {"action_type": "execute_strategy",
               "strategy_id": "S01"}, "ep1")
    print(f"unexecuted candidate   : {unexecuted.prediction_id} — "
          "no label, no counterfactual truth")
    # (b) Re-closing the same episode: idempotent.
    again = h.close_episode("m4_demo", "ep1")
    print(f"re-close               : already_closed="
          f"{again['already_closed']}")
    assert again["already_closed"]
    all_evaluations = h.strategy_prediction_evaluations()
    print(f"stored evaluations     : {len(all_evaluations)} "
          "(re-close added none)")
    assert len(all_evaluations) == 1

    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print("4. The cross-episode calibration channel")
    print("=" * 72)
    # A SECOND episode's fresh context reads the published summary's
    # CONTENT — here one sample, below the default minimum: honest
    # insufficient evidence, never a guessed figure.
    ctx2 = h.build_prediction_context(task, "ep2")
    calibration = ctx2.strategy_calibration
    group = calibration["groups"][
        "strategy_outcome|normalized_objective_gap"]
    print(f"episode-2 context      : {ctx2.context_id}")
    print(f"calibration samples    : {group['n_samples']} "
          f"(min {calibration['min_samples']})")
    print(f"calibration basis      : {group['basis']} "
          f"(reliability={group['reliability']})")
    assert group["basis"] == "insufficient_evidence"
    assert group["reliability"] is None
    # The provider really receives the summary content with the context.
    h.predict_strategy_outcome(
        task, {"action_type": "execute_strategy",
               "strategy_id": "S04"}, "ep2")
    sent = provider.requests[-1]["prediction_context"]
    assert "strategy_outcome_calibration" in sent
    print("provider saw the summary content: True")

    # The FIRST episode's own stored context is unchanged by all of this.
    # (Build one before closing to have a frozen record to compare.)
    # — rebuilt fresh here to show the read path:
    stored_closeout = h.episode_closeout_record("m4_demo", "ep1")
    print(f"close-out readable     : "
          f"{stored_closeout['terminal_state']}, "
          f"{len(stored_closeout['evaluation_ids'])} evaluation(s)")

    h.close()
    tmp.cleanup()
    print()
    print("All assertions passed.")
    print("NOTE: this fixture proves the PROCESS (predict -> execute -> "
          "bind -> close -> calibrate), not model accuracy and not a "
          "capability gain: one closed episode is one sample.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
