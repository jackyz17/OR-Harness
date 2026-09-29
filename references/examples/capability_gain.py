"""Runnable example: joint online prediction of potential capability gain (H+).

The gap this example demonstrates: a candidate was predicted only on the
CURRENT task's benefit/cost/risk, so nothing in the online loop recorded what
the candidate might TEACH. This shows the H+ block — predicted in the SAME
wm-so/1 call, surfaced read-only, and never confused with a verified gain.

It runs with NO real model (a fixed-output stub provider), NO network and NO
API key. The five assertions:

1. **one call, one answer** — H+ arrives with the benefit/cost/risk; no extra
   model round is spent per candidate;
2. **each change keeps its own metric/unit** — a runtime decrease in seconds
   is not compressed into a 0-1 score;
3. **H+ is not the utility** — two candidates with equal benefit/cost/risk rank
   equally even when only one claims a gain;
4. **absence is honest** — a prediction with no gain block still scores, and an
   empty block reads as "no gain claimed", never a positive default;
5. **a predicted gain is never verified** — nothing is written into the
   capability evidence, and the block carries no verified state.

    PYTHONPATH=src python3 references/examples/capability_gain.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from or_harness.api import ORHarness  # noqa: E402
from or_harness.strategy.embedding_index import (  # noqa: E402
    LocalHashEmbeddingBackend,
)
from or_harness.world_model.prediction import ActionSpec  # noqa: E402
from or_harness.world_model.provider import WorldModelProvider  # noqa: E402

TASK = {"task_id": "hplus_demo", "family": "routing",
        "description": "A small routing instance.",
        "spec": {"n_vars": 4, "n_constraints": 3, "n_int_vars": 4}}

BASE = {
    "benefit": {"kind": "solution_quality",
                "metric": "normalized_objective_gap", "unit": "1-gap",
                "value": 0.8,
                "baseline": {"kind": "conditional_stats", "value": 0.7}},
    "cost": {"solver_runtime_s": 3.0},
}

GAIN = {
    "claim": "builds a reusable warm-start structure",
    "applies_to": ["routing with high resource_coupling"],
    "expected_changes": [
        {"metric": "solver_runtime_s", "direction": "decrease",
         "value": 20, "unit": "seconds", "value_kind": "relative",
         "beneficial_direction": "decrease",
         "baseline": {"kind": "conditional_stats", "value": 30}}],
    "evidence_required": ["reuse on a similar task and solve at least as well"],
    "verification_conditions": [{"condition": "a later task shows a >=10% "
                                            "runtime drop"}],
    "degradation_risk": {"events": [{"event": "timeout", "probability": 0.1}]},
    "uncertainty": ["may not generalize to other structures"],
}


class StubProvider(WorldModelProvider):
    name = "example-hplus-stub"

    def __init__(self, gain_on="S01"):
        self.gain_on = gain_on

    def predict(self, request, timeout_s=None):
        sid = (request.get("candidate") or {}).get("strategy_id")
        payload = dict(BASE)
        if sid == self.gain_on:
            payload["capability_gain"] = GAIN
        return {"payload": payload,
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
                "error": None, "latency_s": 0.01}


def main() -> int:
    tmp = tempfile.TemporaryDirectory()
    h = ORHarness(home=tmp.name, world_model=StubProvider(),
                  embedding=LocalHashEmbeddingBackend())

    # ------------------------------------------------------------------
    print("=" * 72)
    print("1. H+ rides the SAME call as benefit/cost/risk")
    print("=" * 72)
    plan = h.plan_next(
        TASK, "ep1",
        candidates=[ActionSpec("execute_strategy", "hplus_demo",
                               strategy_id="S01", solver="highs"),
                    ActionSpec("execute_strategy", "hplus_demo",
                               strategy_id="S02", solver="highs")],
        limits={"horizon": 1})
    print(f"model calls            : {plan['model_calls_made']} "
          "(one per candidate — H+ added no round)")
    assert plan["model_calls_made"] == 2
    by_sid = {c["action_spec"]["strategy_id"]: c for c in plan["candidates"]}
    gain = by_sid["S01"]["capability_gain"]
    print(f"S01 capability_gain    : {gain['kind']}")
    print(f"  claim                : {gain['claim']}")
    print(f"  applies_to           : {gain['applies_to']}")
    assert gain is not None and gain["kind"] == "predicted_capability_gain"

    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print("2. Each change keeps its own metric and unit (no 0-1 score)")
    print("=" * 72)
    change = gain["expected_changes"][0]
    print(f"expected change        : {change['metric']} "
          f"{change['direction']} by {change['value']}{change['unit']} "
          f"(baseline {change['baseline']['value']})")
    print(f"evidence required      : {gain['evidence_required']}")
    print(f"verification condition : "
          f"{gain['verification_conditions'][0]['condition']}")
    print(f"degradation risk       : "
          f"{[e['event'] for e in gain['degradation_risk']['events']]}")
    assert change["metric"] == "solver_runtime_s"
    assert change["unit"] == "seconds"

    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print("3. H+ is NOT the utility: equal G/c/L -> equal ranking")
    print("=" * 72)
    utilities = {sid: c["score"]["utility"] for sid, c in by_sid.items()}
    print(f"utilities              : {utilities}")
    print(f"S01 claims a gain      : {by_sid['S01']['capability_gain'] is not None}")
    print(f"S02 claims a gain      : {by_sid['S02']['capability_gain'] is not None}")
    assert utilities["S01"] == utilities["S02"], \
        "a capability gain must not change the utility"
    assert by_sid["S02"]["capability_gain"] is None

    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print("4. Absence is honest: no block, or an empty block")
    print("=" * 72)
    h2 = ORHarness(home=tempfile.mkdtemp(),
                   world_model=StubProvider(gain_on=None),
                   embedding=LocalHashEmbeddingBackend())
    self_gain_none = h2.predict_strategy_outcome(
        TASK, {"action_type": "execute_strategy", "strategy_id": "S01"}, "ep1")
    print(f"no gain block          : status={self_gain_none.status}, "
          f"claims={self_gain_none.claims_capability_gain}")
    assert self_gain_none.status == "valid"
    assert not self_gain_none.claims_capability_gain
    h2.close()

    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print("5. A predicted gain is never a VERIFIED capability")
    print("=" * 72)
    evidence_before = h.capability_evidence_with_effects().to_dict()
    h.predict_strategy_outcome(
        TASK, {"action_type": "execute_strategy", "strategy_id": "S01"}, "ep1")
    evidence_after = h.capability_evidence_with_effects().to_dict()

    def _source_states(evidence):
        return {name: source.get("status")
                for name, source in (evidence.get("sources") or {}).items()}

    print(f"capability sources     : {_source_states(evidence_before)}")
    print(f"unchanged by predicting: "
          f"{_source_states(evidence_before) == _source_states(evidence_after)}")
    print(f"gain carries verified? : {'effect_verified' in gain}")
    assert _source_states(evidence_before) == _source_states(evidence_after), \
        "a predicted gain must not change capability evidence"
    assert "effect_verified" not in gain

    h.close()
    tmp.cleanup()
    print()
    print("All assertions passed.")
    print("NOTE: H+ is a PREDICTION about what a candidate might teach. It "
          "rides the same call, never enters the utility, and becomes "
          "'verified' only through the offline capability path on later "
          "tasks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
