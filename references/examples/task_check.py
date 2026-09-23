"""Runnable example: task-result check -> reflection -> repair -> close-out.

The gap this example demonstrates: `orx execute` answers "did the solver
solve the MODEL it was given" (a legal status, a finite objective, a gap).
It cannot answer "is this a valid answer to the TASK". A relaxed LP answered
with fractional values reports ``optimal`` with ``gap=0``, and without a
task-level check it would enter recall, the conditional statistics, the
world-model feedback and offline induction as a SUCCESS sample.

It runs with NO real model (a fixed-output stub provider), NO network, NO
API key and NO embedding service (the deterministic offline hashing backend
is injected). The scenario is the acceptance case:

1. **the relaxed answer** — an optimal LP solution with FRACTIONAL units
   matches the reference objective exactly, and the task check still FAILS
   it: the integer-domain basis is what catches it. The objective matching
   the reference does not make the answer valid;
2. **the framework does not diagnose** — it returns the reflection MATERIAL
   (task text, artifacts, the check report, prior attempts) and the outer
   agent decides what was wrong. Nothing is rebuilt, and the task is never
   relaxed to match a reference value;
3. **the repair** — whole units only, re-solved in the SAME episode. It
   passes. BOTH attempts stay recorded with their real cost; only the first
   stops counting as a success sample;
4. **the through-line** — the failed attempt's quality is 0.0 everywhere it
   is read (statistics, world-model feedback), its cost still counts in the
   task total, and recall carries the restriction explicitly;
5. **the close-out** — the episode closes, the bound prediction is evaluated
   against the TASK's real outcome (0.0, not the solver's 1.0), and the
   close-out reports its task-check coverage: closing an episode is not
   certifying the answer.

This fixture proves the PROCESS is correct. It does NOT prove the model
represents the task (only a declared check does that, and only within its
declared scope).

    PYTHONPATH=src python3 references/examples/task_check.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from or_harness.api import ORHarness  # noqa: E402
from or_harness.core.schema import task_check_state  # noqa: E402
from or_harness.strategy.embedding_index import (  # noqa: E402
    LocalHashEmbeddingBackend,
)
from or_harness.strategy.stats import quality_score  # noqa: E402
from or_harness.world_model.provider import WorldModelProvider  # noqa: E402

#: The task states the INTEGER requirement itself, so the check basis is read
#: off the task — never reverse-engineered from the reference value.
TASK = {
    "task_id": "check_demo",
    "family": "production_planning",
    "description": (
        "A workshop produces two products. Product 1 needs 3 hours of "
        "machining and 2 hours of finishing; product 2 needs 1 hour of "
        "machining and 4 hours of finishing. Machining has 9 hours "
        "available and finishing has 12. Every production quantity MUST BE "
        "A WHOLE NUMBER of units — fractional units cannot be shipped. "
        "Maximise profit (7 per unit of product 1, 3 per unit of product 2)."),
    "spec": {"n_vars": 2, "n_constraints": 2, "n_int_vars": 2},
    "annotations": {"coupling": {"resource_coupling": 0.3,
                                 "temporal_coupling": 0.1,
                                 "route_complexity": 0.2,
                                 "semantic_coupling": 0.5}},
}

#: A fixed prediction payload: a normalized quality with a baseline, so the
#: strategy-outcome channel has something real to compare at close-out.
PAYLOAD = {
    "benefit": {"kind": "solution_quality",
                "metric": "normalized_objective_gap", "unit": "1-gap",
                "value": 0.8,
                "baseline": {"kind": "conditional_stats", "value": 0.7}},
    "cost": {"llm_tokens": 1200, "solver_runtime_s": 3.0},
    "risk": {"events": [{"event": "model_invalid", "probability": 0.2}]},
    "uncertainty": {"execution_randomness": 0.3, "knowledge_gap": 0.6},
}


class StubProvider(WorldModelProvider):
    name = "example-task-check-stub"

    def __init__(self, payload=PAYLOAD):
        self.payload = payload

    def predict(self, request, timeout_s=None):
        return {"payload": self.payload,
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
                "error": None, "latency_s": 0.02}


def _solve(h, *, variables, objective, tag):
    """One real sandboxed attempt reporting its solution vector."""
    work = Path(h.home) / f"ws_{tag}"
    work.mkdir(parents=True, exist_ok=True)
    payload = {"status": "optimal", "objective_value": objective,
               "objective_bound": objective, "mip_gap": 0.0,
               "runtime_seconds": 0.01, "variables": variables}
    (work / "solve.py").write_text(
        "import json\n"
        "with open('result.json', 'w') as fh:\n"
        "    json.dump(" + json.dumps(payload) + ", fh)\n",
        encoding="utf-8")
    record = h.execute(TASK, "S01", str(work / "solve.py"), str(work),
                       solver="highs", episode_id="ep1")
    h.record(record)
    return record


def main() -> int:
    tmp = tempfile.TemporaryDirectory()
    home = tmp.name
    h = ORHarness(home=home, world_model=StubProvider(),
                  embedding=LocalHashEmbeddingBackend())

    # ------------------------------------------------------------------
    print("=" * 72)
    print("1. The relaxed answer: optimal, matches the reference, still WRONG")
    print("=" * 72)
    # The frozen prediction is made BEFORE anything runs.
    prediction = h.predict_strategy_outcome(
        TASK, {"action_type": "execute_strategy", "strategy_id": "S01"},
        "ep1")
    print(f"prediction             : {prediction.prediction_id} "
          f"(G={prediction.benefit.value})")

    relaxed = _solve(h, variables={"x1": 2.5, "x2": 1.5}, objective=10750.0,
                     tag="relaxed")
    print(f"relaxed attempt        : {relaxed.execution_id} "
          f"(status {relaxed.quality['status']}, "
          f"objective {relaxed.quality['objective']}, "
          f"gap {relaxed.quality['gap']})")
    print(f"  solution             : x1=2.5, x2=1.5  "
          "<- fractional units, cannot be shipped")
    print(f"  quality before check : {quality_score(relaxed)} "
          "(the solver's own model was solved optimally)")

    # The check basis is read off the TASK's integer requirement.
    check = {"reference_objective": 10750.0,
             "integer": {"variables": ["x1", "x2"]}}
    verdict = h.check_task_result(relaxed.execution_id, check)
    report = verdict["report"]
    print(f"task check             : {report['state'].upper()}")
    for item in report["checks"]:
        print(f"  {item['check']:22s} ok={item.get('ok')}")
    for diff in report["diffs"]:
        print(f"  diff: {diff['reason']}")
    # The objective MATCHED; the domain check is what caught it.
    assert report["state"] == "failed"
    assert report["scope"]["basis"] == ["reference_objective", "integer"]
    assert any("is not an integer" in d["reason"] for d in report["diffs"])
    assert "not integral" in report["conclusion"]
    print(f"  scope checked        : {report['scope']['basis']}")
    print(f"  scope unchecked      : {len(report['scope']['unchecked'])} "
          "item(s) — the model's fidelity to the task is NOT established "
          "by a matching objective")

    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print("2. The framework returns MATERIAL, not a diagnosis")
    print("=" * 72)
    material = verdict["reflection_material"]
    print(f"task text (excerpt)    : "
          f"{material['task_text'][:60]}...")
    print(f"code hash              : {material['code_hash']}")
    print(f"solution variables     : {material['solution_variables']}")
    print(f"prior attempts         : "
          f"{[a['execution_id'] for a in material['prior_attempts']]}")
    print(f"guidance               : {material['guidance'][:80]}...")
    assert "WHOLE NUMBER" in material["task_text"]
    # The framework did not rebuild anything and did not touch the task:
    # the agent decides, and the task is never relaxed to fit a reference.
    assert "not classify" in material["guidance"]

    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print("3. Repair in the SAME episode: whole units only")
    print("=" * 72)
    # The agent's stated basis for the change (recorded as its own action).
    h.report_action("model", TASK, "ep1",
                    outcome={"kind": "model_artifact",
                             "revised_from": relaxed.execution_id,
                             "basis": ("the task requires whole units; the "
                                       "relaxation dropped the integrality "
                                       "domain — the model, not the task, "
                                       "was wrong")},
                    cost={"llm_tokens": 220})
    repaired = _solve(h, variables={"x1": 2, "x2": 3}, objective=10755.0,
                      tag="repaired")
    second = h.check_task_result(repaired.execution_id,
                                 {"reference_objective": 10755.0,
                                  "integer": {"variables": ["x1", "x2"]}})
    print(f"repaired attempt       : {repaired.execution_id} "
          f"(objective {repaired.quality['objective']})")
    print(f"  solution             : x1=2, x2=3  <- whole units")
    print(f"task check             : {second['state'].upper()}")
    assert second["state"] == "passed"

    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print("4. Both attempts survive; only one is a success sample")
    print("=" * 72)
    records = h.bank.query(task_id="check_demo")
    for record in records:
        print(f"{record.execution_id}  status={record.quality['status']:8s} "
              f"objective={record.quality['objective']}  "
              f"task_check={task_check_state(record)}  "
              f"quality_contributed={quality_score(record)}")
    assert len(records) == 2
    assert quality_score(h.bank.get(relaxed.execution_id)) == 0.0
    assert quality_score(h.bank.get(repaired.execution_id)) == 1.0

    # The cost of BOTH attempts still counts: a disqualified answer does
    # not refund the money spent producing it.
    totals = h.task_cost_summary("check_demo")
    print(f"task cost              : {totals['n_attempts']} attempt(s), "
          f"solver_runtime_s total "
          f"{totals['total_cost']['solver_runtime_s']}")
    assert totals["n_attempts"] == 2

    # Recall carries the restriction instead of hiding it.
    recall = h.recall(TASK, top=3)
    hits = (recall.get("vector_recall") or {}).get("execution_evidence") or []
    if hits:
        for hit in hits:
            print(f"recall hit             : {hit['execution_id']} "
                  f"task_check={hit['task_check']} "
                  f"{hit['task_check_limitations']}")
    else:
        print("recall hit             : (text channel off in this run)")

    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print("5. Close-out: the prediction is judged against the TASK")
    print("=" * 72)
    h.bind_strategy_outcome(prediction.prediction_id, repaired.action_id)
    result = h.close_episode("check_demo", "ep1")
    evaluation = result["evaluations"][0]
    print(f"evaluation state       : {evaluation['state']}")
    print(f"benefit                : predicted "
          f"{evaluation['benefit']['predicted']}, observed "
          f"{evaluation['benefit']['observed']}")
    print(f"close-out task checks  : {result['task_checks']}")
    # The repaired answer really is a good one here, so the observed value
    # is the honest 1.0 — the gate only bites on the disqualified attempt.
    assert evaluation["benefit"]["observed"] == 1.0
    assert result["task_checks"]["verdicts"] == {"failed": 1, "passed": 1}

    # A late correction on an ALREADY-closed episode changes later USE, not
    # history: the stored evaluation is byte-identical, but the sample
    # leaves the calibration means and is counted separately.
    stored_before = h.get_strategy_evaluation(evaluation["evaluation_id"])
    import time
    h.bank.set_task_check(repaired.execution_id, {
        "state": "failed", "execution_id": repaired.execution_id,
        "task_id": "check_demo", "episode_id": "ep1",
        "diffs": [{"basis": "integer_domains", "variable": "x1"}],
        "checked_at": time.time() + 10,
    })
    summary = h.calibration_summary()
    print(f"late correction        : "
          f"exclusions={summary['exclusions']}")
    print(f"stored evaluation kept : "
          f"{h.get_strategy_evaluation(evaluation['evaluation_id']) == stored_before}")
    assert summary["exclusions"].get("validity_corrected") == 1
    assert h.get_strategy_evaluation(
        evaluation["evaluation_id"]) == stored_before
    assert summary["validity_corrections"]

    h.close()
    tmp.cleanup()
    print()
    print("All assertions passed.")
    print("NOTE: this fixture proves the PROCESS (check -> diagnose -> "
          "repair -> record -> close-out). A `passed` verdict covers only "
          "the DECLARED bases: it is not proof that the model represents "
          "the task.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
