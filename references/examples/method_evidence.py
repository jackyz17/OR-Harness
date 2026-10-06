"""Runnable example: method evidence -> material -> new strategy -> recall.

The gap this example demonstrates: two layers of memory need different kinds
of content. Execution Evidence must record what a single attempt ACTUALLY
did, and Strategic Knowledge must be a technique with conditions, a method, a
consequence and a boundary — not a mean with a strategy name attached.

The chain it walks:

1. **record methods, not just outcomes** — each solve reports the method it
   performed (a ``method_performed`` receipt stamped with the action id), and
   an earlier failed attempt reports a DIFFERENT one. The plan and the
   performed method are kept apart: the record never promotes a plan to fact.
2. **the material names the method content** — ``induction_material`` shows
   each attempt's `method.basis` (`performed` / `planned_only` / `none`) and
   its outcomes; the outer agent (here, a scripted stand-in) reads it and
   writes the "condition -> how -> consequence -> boundary" strategy.
3. **the write is checked and auditable** — the submission records a
   maintenance action with a knowledge delta, its declared `check` is
   evaluated, and the published strategy reaches recall from an INDEPENDENT
   task.
4. **an honest refusal to invent** — a second batch whose evidence reports NO
   method is readable (`method.basis == "none"`), and submitting a strategy
   from it carries a `material` warning: the framework will not turn a mean
   into a technique.

It runs with no model, no network and no API key — the deterministic offline
hashing backend is injected.

    PYTHONPATH=src python3 references/examples/method_evidence.py
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
from or_harness.strategy.embedding_index import (  # noqa: E402
    LocalHashEmbeddingBackend,
)

#: The condition the strategy is about: high temporal coupling needs the
#: cross-period state carried across the decomposition.
TASK_BASE = {
    "task_id": "schedule",
    "family": "scheduling",
    "description": "scheduling with cross-period carry-over state",
    # Scalar coupling values belong in annotations.coupling; a CIR would be
    # an entities/decisions/constraints structure.
    "annotations": {"coupling": {"temporal_coupling": 0.7}},
}

#: The method that FAILED: solve the horizon in one shot, dropping the
#: cross-period state (a plain MIP that ignores the carry-over).
FAILED_METHOD = {
    "name": "single-shot MIP",
    "steps": ["build the full-horizon model",
              "solve without carry-over state"],
}

#: The method that WORKED: decompose across periods, carrying the state.
FIXED_METHOD = {
    "name": "rolling-horizon decomposition",
    "steps": ["relax the coupling constraint",
              "solve each period",
              "carry the cross-period state forward",
              "recombine"],
}

#: A solve script that reports the method it performed, stamped with this
#: attempt's action id so a leftover receipt cannot be read as this run's.
SOLVE_TEMPLATE = textwrap.dedent("""
    import json, os
    method = {method!r}
    result = {{"status": "optimal", "objective_value": 42.0,
              "objective_bound": 42.0, "runtime_seconds": 0.01,
              "solver": "highs",
              "method_performed": dict(
                  method, action_id=os.environ.get("OR_ACTION_ID"))}}
    if {feasible!r}:
        pass
    else:
        result["status"] = "error"
    with open("result.json", "w") as fh:
        json.dump(result, fh)
""")


def _run(h, task, method, feasible, slot):
    ws = Path(h.home) / f"ws_{slot}"
    ws.mkdir(parents=True, exist_ok=True)
    script = ws / "solve.py"
    script.write_text(SOLVE_TEMPLATE.format(method=method, feasible=feasible),
                      encoding="utf-8")
    record = h.execute(task, "S-decompose", str(script), str(ws),
                       solver="highs", episode_id=f"ep_{slot}")
    h.record(record)
    return record


def main() -> int:
    with tempfile.TemporaryDirectory() as home:
        h = ORHarness(home=home)
        h.embedding_index = None  # keep the example hermetic (text off)
        try:
            print("=" * 72)
            print("1. record two attempts: a failed one and a same-solver fix")
            print("=" * 72)
            # Task T1: the single-shot method fails, then the SAME task is
            # retried with a DIFFERENT method under the SAME solver.
            t1 = dict(TASK_BASE, task_id="T1")
            failed = _run(h, t1, FAILED_METHOD, feasible=False, slot="f")
            fixed = _run(h, t1, FIXED_METHOD, feasible=True, slot="s")
            # A second independent task so the strategy can be published
            # (the publication gate needs >=2 distinct tasks).
            t2 = dict(TASK_BASE, task_id="T2")
            fixed2 = _run(h, t2, FIXED_METHOD, feasible=True, slot="t")
            print(f"failed attempt {failed.execution_id}: "
                  f"method_actual={failed.method_actual['name']}")
            print(f"fixed attempt  {fixed.execution_id}: "
                  f"method_actual={fixed.method_actual['name']}")
            print("The plan/actual split holds: what RAN came from the "
                  "script's receipt, never from the plan.\n")

            print("=" * 72)
            print("2. read the material, then form the strategy")
            print("=" * 72)
            material = h.induction_material(strategy_id="S-decompose")
            assert material["count"], "expected the batch of attempts"
            print(f"batch spans {material['n_distinct_tasks']} task(s)")
            for m in material["material"]:
                label = (m["method"]["actual"] or {}).get("name") or \
                    "(no method)"
                print(f"  {m['execution_id']}: {m['method']['basis']} / "
                      f"{label}")
            # The strategy: conditions -> method -> consequence -> boundary.
            # The evidence spans TWO independent tasks (the failed attempt on
            # T1 and the fixed method on T2), which is what the publication
            # gate needs; the framework DERIVES the identity from the facts.
            strategy = {
                "subject": "principle:cross_period_state",
                "claim": ("when temporal coupling is high, the cross-period "
                          "state must be carried across the decomposition; "
                          "dropping it (a single-shot MIP) yields an error, "
                          "while carrying it yields a feasible optimum"),
                "conditions": {"predicates": {
                    "family": "scheduling", "temporal_coupling": [0.5, 1.0]}},
                "evidence": [{"execution_id": failed.execution_id,
                              "role": "dropped"},
                             {"execution_id": fixed.execution_id,
                              "role": "preserved"},
                             {"execution_id": fixed2.execution_id,
                              "role": "preserved"}],
                "kind": "intervention_recovery",
            }
            # The declared check: the preserved role finishes optimal, and
            # quality is materially higher than the dropped role. The
            # framework evaluates these over the cited facts.
            verify = {
                "check": {"assertions": [
                    {"kind": "status", "roles": ["preserved"],
                     "status": "optimal"},
                    {"kind": "comparison", "metric": "quality",
                     "roles_a": ["preserved"], "roles_b": ["dropped"],
                     "direction": "higher", "min_gap": 0.2,
                     "mode": "group", "aggregation": "mean"},
                ]},
            }
            result = h.induce(relations=[strategy], verify=verify)
            outcome = result["relations"][0]
            print(f"saved     : {outcome.get('saved')}")
            print(f"publication: {outcome['publication']}")
            print(f"material  : {outcome.get('material') or 'sufficient'}")
            delta = result["action"]["knowledge_delta"]
            print(f"action    : {result['action']['action_id']} "
                  f"({result['action']['business_result']})")
            print(f"delta     : created={delta['entries_created']}\n")

            print("=" * 72)
            print("3. a batch with NO method is readable but not invented")
            print("=" * 72)
            # A separate, method-less batch: two tasks, records with no method.
            from or_harness.core.schema import ExecutionRecord
            for i in range(2):
                rec = ExecutionRecord(
                    execution_id=f"plain{i}",
                    task_id=f"P{i}",
                    strategy_id="S-plain",
                    profile_snapshot=h.profile(dict(
                        TASK_BASE, task_id=f"P{i}",
                        annotations={"coupling": {"temporal_coupling": 0.1}},
                        family="routing")),
                    quality={"feasible": True, "objective": 10.0,
                             "gap": 0.0, "status": "optimal"},
                    solver={"name": "highs"},
                )
                h.record(rec)
            bare = h.induction_material(strategy_id="S-plain")
            if bare["count"]:
                print(f"method.basis: {bare['material'][0]['method']['basis']}")
                # Submitting a strategy from it still saves, with a warning.
                ids = [m["execution_id"] for m in bare["material"]]
                warned = h.induce(relations=[{
                    "subject": "principle:no_method",
                    "claim": "S-plain works well on routing",
                    "evidence": [{"execution_id": e, "role": "evidence"}
                                 for e in ids]}])
                warning = warned["relations"][0].get("material")
                print(f"strategy saved: "
                      f"{bool(warned['relations'][0].get('saved'))}")
                print(f"warning       : {warning['reason'][:110]}...")
            print()

            print("=" * 72)
            print("4. the verified strategy reaches recall from a NEW task")
            print("=" * 72)
            recall = h.recall(dict(TASK_BASE, task_id="T4"))
            hits = [r for r in recall.get("recommendations", [])
                    if (r.get("knowledge") or {}).get("claim")]
            print(f"recommendations carrying a strategy: {len(hits)}")
            for item in hits:
                knowledge = item["knowledge"]
                claim = knowledge.get("claim") or {}
                print(f"  entry     : {knowledge.get('entry_id')}")
                print(f"  strategy  : {str(claim.get('text'))[:66]}...")
                print(f"  state     : {knowledge.get('verification_state')}")
                print(f"  tasks     : {claim.get('tasks')}")
                print(f"  evidence  : "
                      f"{[e['execution_id'] for e in claim.get('evidence', [])]}")
            print("\nThe strategy is quoted from accumulated evidence on OTHER "
                  "tasks, with the entry's own verification state and a "
                  "citable evidence list — not from a mean.")
            return 0
        finally:
            h.close()


if __name__ == "__main__":
    raise SystemExit(main())
