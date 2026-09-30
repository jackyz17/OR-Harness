"""Runnable example: an online capability-gain claim, followed up for real.

Three chains meet in this example, end to end and with NO real model (a
fixed-output stub provider), NO network and NO API key:

1. **A token number has to EARN its standing.** A declared
   ``agent_estimate`` is shown but never becomes a calibration truth; a
   provider/host report does. The same figure means different things
   depending on where it came from.
2. **A real failure is not erased by a retry.** An interrupted attempt is
   staged as its own honest failure fact (with the wall-clock it really
   consumed), and the retry is a separate attempt.
3. **An online capability gain is followed up without a second
   prediction.** The H+ claimed with the candidate is archived as a trace;
   the real execution binds to it; ``inspect --bank capability`` reports it;
   and ``evaluate-capability`` judges the ORIGINAL claim against later real
   evidence — the framework never re-predicts after seeing the result.

What is verified where:
  * here (stub provider + local sandbox): the accounting, the failure
    preservation, the trace, the binding and the evaluation states;
  * NOT verified: a real host's usage collection (no OpenClaw/Hermes is
    installed in this environment — the host report below is a fixture in
    the canonical shape), and a real model's H+ content.
"""
from __future__ import annotations

import json
import sys
import tempfile
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from or_harness.api import ORHarness  # noqa: E402
from or_harness.core.schema import CostVector  # noqa: E402
from or_harness.strategy.embedding_index import (  # noqa: E402
    LocalHashEmbeddingBackend,
)
from or_harness.world_model.provider import WorldModelProvider  # noqa: E402

TASK = {"task_id": "followup_demo", "family": "routing",
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
    "assessment": "expected",
    "claim": "builds a reusable warm-start structure",
    "applies_to": ["routing with high resource_coupling"],
    "expected_changes": [
        {"metric": "solver_runtime_s", "direction": "decrease",
         "value": 20, "unit": "seconds", "value_kind": "relative",
         "beneficial_direction": "decrease",
         "baseline": {"kind": "conditional_stats", "value": 30}}],
    "evidence_required": ["reuse on a similar task and solve at least as "
                          "well with less runtime"],
    "verification_conditions": [{"condition": "a later task shows a >=10% "
                                            "runtime drop"}],
    "uncertainty": ["may not generalize to other structures"],
}


class StubProvider(WorldModelProvider):
    """A labelled stub: declared answers, never a hidden law."""

    name = "example-stub"

    def predict(self, request, timeout_s=None):
        return {"payload": dict(BASE, capability_gain=dict(GAIN)),
                "usage": {"prompt_tokens": 900, "completion_tokens": 100},
                "error": None, "latency_s": 0.01}


SOLVE = """
    import json, os
    result = {"status": "optimal", "objective_value": 1.0,
              "objective_bound": 1.0, "runtime_seconds": 0.01}
    with open("result.json", "w") as fh:
        json.dump(result, fh)
"""


def main() -> int:
    home = tempfile.mkdtemp(prefix="orx_followup_")
    harness = ORHarness(home=home, world_model=StubProvider(),
                        embedding=LocalHashEmbeddingBackend())
    workspace = Path(home) / "ws"
    workspace.mkdir(parents=True, exist_ok=True)
    script = workspace / "solve.py"
    script.write_text(textwrap.dedent(SOLVE), encoding="utf-8")

    # 1. Predict a candidate WITH an online capability-gain claim.
    prediction = harness.predict_strategy_outcome(
        TASK, {"action_type": "execute_strategy", "strategy_id": "S01",
               "solver": "highs", "episode_id": "ep1"}, "ep1")
    print(f"1. prediction {prediction.prediction_id}: status "
          f"{prediction.status}, claims_gain="
          f"{prediction.claims_capability_gain}, assessment="
          f"{prediction.capability_gain.assessment}")
    assert prediction.claims_capability_gain
    assert prediction.capability_gain.assessment == "expected"
    print(f"   gain block is read-only: view="
          f"{harness.inspect(bank='capability', prediction_id=prediction.prediction_id)['prediction_source']}")
    assert harness.online_capability_gains()["n_online_gains"] == 1

    # 2. Execute it for real: the trace advances to `bound`.
    record = harness.execute(TASK, None, str(script), str(workspace),
                             solver=None, episode_id="ep1",
                             prediction_id=prediction.prediction_id)

    # 3. Record with a HOST usage report (a fixture in the canonical shape;
    #    no host is installed here). A host report is a real observation.
    host_report = {"prompt_tokens": 700, "completion_tokens": 300,
                   "source": "fixture-host"}
    outcome = harness.record(record, host_usage=host_report)
    stored = harness.bank.get(outcome["execution_id"])
    print(f"2. recorded {stored.execution_id}: llm_tokens="
          f"{stored.cost.llm_tokens} from "
          f"{stored.execution_features['cost_provenance']['llm_tokens']['source']}")
    assert stored.cost.llm_tokens == 1000.0

    # 4. A DECLARED estimate never stands as a measured truth: it is dropped
    #    from the counted total rather than scored as an observation.
    from or_harness.world_model.episode_closeout import _aggregate_costs
    declared = CostVector(llm_tokens=20000.0, measured={"llm_tokens"})
    stored.cost = declared
    stored.execution_features["cost_provenance"] = {
        "llm_tokens": {"source": "agent_estimate"}}
    aggregate = _aggregate_costs([declared], [stored])
    block = aggregate["llm_tokens"]
    print(f"3. declared 20000 tokens: counted total={block['total']}, "
          f"n_measured={block['n_measured']}, excluded={block.get('excluded')}")
    assert block["total"] is None and block["excluded"] == 1

    # 5. The online gain is followed up along the REAL path, with no second
    #    prediction: bind the claim, then evaluate it.
    bound = harness.bind_capability_maintenance(prediction.prediction_id)
    print(f"4. bind: state={bound['state']}, source="
          f"{bound.get('prediction_source')}, execution="
          f"{bound['binding']['actual_execution_ids']}")
    assert bound["state"] == "bound"

    evaluation = harness.evaluate_capability_effect(prediction.prediction_id)
    result = evaluation["evaluation"]
    print(f"5. evaluate: state={result['state']}, effect_verified="
          f"{result['effect_verified']}, source="
          f"{evaluation['prediction_source']}")
    # No knowledge change happened here, so no gain can be "verified": the
    # honest verdict is NOT an improvement claim.
    assert evaluation["prediction_source"] == "online_trace"
    assert result["effect_verified"] is False

    summary = harness.inspect(bank="capability")
    print(f"6. capability bank: {summary['n_predictions']} offline "
          f"prediction(s), {summary['online_gains']['n_online_gains']} "
          f"online gain(s) "
          f"(expected={summary['online_gains']['n_expected']}, "
          f"none={summary['online_gains']['n_none']}, "
          f"unassessed={summary['online_gains']['n_unassessed']}), "
          f"{summary['online_gains']['n_effect_verified']} verified")
    assert summary["online_gains"]["n_online_gains"] == 1
    assert summary["online_gains"]["n_expected"] == 1

    # 7. An interrupted attempt is a real failure, preserved as its own fact
    #    with the wall-clock it really consumed.
    prediction2 = harness.predict_strategy_outcome(
        TASK, {"action_type": "execute_strategy", "strategy_id": "S02",
               "solver": "highs", "episode_id": "ep2"}, "ep2")

    class Boom:
        def execute(self, *args, **kwargs):
            raise RuntimeError("interrupted before the sandbox")

    real_executor = harness.executor
    harness.executor = Boom()
    try:
        harness.execute(TASK, "S02", str(script), str(workspace),
                        solver="highs", episode_id="ep2",
                        prediction_id=prediction2.prediction_id)
    except RuntimeError:
        pass
    finally:
        harness.executor = real_executor
    staged = harness.bank.pending(task_id=TASK["task_id"])
    failure = [f for f in staged if f.quality.get("status") == "error"][-1]
    print(f"7. interrupted attempt: status={failure.quality['status']}, "
          f"latency_s={failure.cost.latency_s} "
          f"(measured={sorted(failure.cost.measured_dims())}), "
          f"predicted_for={failure.execution_features.get('predicted_for')}")
    assert "latency_s" in failure.cost.measured_dims()
    assert failure.execution_features.get("predicted_for") == \
        prediction2.prediction_id

    harness.close()
    print("\nAll assertions held. Host usage collection itself is NOT "
          "verified here: no OpenClaw/Hermes is installed, so the report "
          "above is a fixture in the canonical shape.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
