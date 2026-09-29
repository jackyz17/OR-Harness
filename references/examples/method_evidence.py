"""Runnable example: method evidence -> material -> relation -> recall.

The gap this example demonstrates: two layers of memory need different kinds
of content. Execution Evidence must record what a single attempt ACTUALLY
did, and Strategic Knowledge must be a technique with conditions, a method, a
consequence and a boundary — not a mean with a strategy name attached.

The chain it walks:

1. **record methods, not just outcomes** — each solve reports the method it
   performed (a ``method_performed`` receipt stamped with the action id), and
   an earlier failed attempt reports a DIFFERENT one. The plan and the
   performed method are kept apart: the record never promotes a plan to fact.
2. **the same-solver fix is detected** — the failed attempt and the
   successful one used the SAME solver, but their methods differ, so
   ``intervention_recovery`` fires. A plain retry with no change evidence
   would not.
3. **material, then a claim** — ``induction_material`` shows the methods, the
   change and what followed; the outer agent (here, a scripted stand-in)
   reads it and writes the "condition -> how -> consequence -> boundary"
   claim, submitting it as a structured relation.
4. **an honest refusal to invent** — a second candidate whose evidence
   reports NO method is reported ``insufficient``, and submitting a claim
   from it carries a `material` warning: the framework will not turn a mean
   into a technique.
5. **the write is auditable and reachable** — the relation submission records
   a maintenance action with a knowledge delta, and the published relation
   reaches recall from an INDEPENDENT task.

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

#: The condition the claim is about: high temporal coupling needs the
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
            # A second independent task so the relation can be published
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
            print("2. the same-solver fix is DETECTED (not a plain retry)")
            print("=" * 72)
            hints = h.bank.get(fixed.execution_id).execution_features.get(
                "induction_hints") or []
            recovery = [x for x in hints
                        if x["pattern"] == "intervention_recovery"]
            assert recovery, "expected a same-solver intervention hint"
            change = recovery[0]["evidence"]["change"]
            print(f"pattern   : {recovery[0]['pattern']}")
            print(f"change    : {change['kind']} "
                  f"({change['from_method']['name']} -> "
                  f"{change['to_method']['name']})")
            print("A retry that reports the SAME method would not fire: the "
                  "change must be visible.\n")

            print("=" * 72)
            print("3. read the material, then write the claim")
            print("=" * 72)
            material = h.induction_material(pattern="intervention_recovery")
            assert material["count"], "expected a recovery candidate"
            item = material["material"][0]
            print(f"candidate {item['bundle_id']}")
            print(f"material  : {item['material_state']['state']}")
            for m in item["methods"]:
                label = (m.get("actual") or {}).get("name") or "(no method)"
                print(f"  {m['execution_id']}: {label}")
            # The claim: conditions -> method -> consequence -> boundary.
            # The evidence spans TWO independent tasks (the failed attempt on
            # T1 and the fixed method on T2), which is what the publication
            # gate needs; the framework DERIVES the identity from the facts.
            relation = {
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
            # Check what the claim declares: the preserved role finishes
            # optimal, and quality is materially higher than the dropped
            # role. The framework evaluates these over the cited facts.
            verify = {
                "purpose": "relation",
                "check": {"assertions": [
                    {"kind": "status", "roles": ["preserved"],
                     "status": "optimal"},
                    {"kind": "comparison", "metric": "quality",
                     "roles_a": ["preserved"], "roles_b": ["dropped"],
                     "direction": "higher", "min_gap": 0.2,
                     "mode": "group", "aggregation": "mean"},
                ]},
            }
            result = h.induce(relations=[relation], verify=verify)
            outcome = result["relations"][0]
            print(f"saved     : {outcome.get('saved')}")
            print(f"publication: {outcome['publication']}")
            print(f"material  : {outcome.get('material') or 'sufficient'}")
            delta = result["action"]["knowledge_delta"]
            print(f"action    : {result['action']['action_id']} "
                  f"({result['action']['business_result']})")
            print(f"delta     : created={delta['entries_created']}\n")

            print("=" * 72)
            print("4. a candidate with NO method is refused, not invented")
            print("=" * 72)
            # A separate, method-less cell: two tasks, records with no method.
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
                )
                h.record(rec)
            bare = h.induction_material(strategy_id="S-plain")
            if bare["count"]:
                state = bare["material"][0]["material_state"]
                print(f"material_state: {state['state']}")
                print(f"reason        : {state['reason']}")
                # Submitting a claim from it still saves, with a warning.
                ids = [m["execution_id"] for m in bare["material"][0]["methods"]]
                warned = h.induce(relations=[{
                    "subject": "principle:no_method",
                    "claim": "S-plain works well on routing",
                    "evidence": [{"execution_id": e, "role": "evidence"}
                                 for e in ids]}])
                warning = warned["relations"][0].get("material")
                print(f"claim saved   : "
                      f"{bool(warned['relations'][0].get('saved'))}")
                print(f"warning       : {warning['reason'][:110]}...")
            print()

            print("=" * 72)
            print("5. the verified relation reaches recall from a NEW task")
            print("=" * 72)
            recall = h.recall(dict(TASK_BASE, task_id="T4"))
            knowledge = recall.get("knowledge") or []
            print(f"knowledge entries returned: {len(knowledge)}")
            for item in knowledge:
                print(f"  entry     : {item.get('entry_id')} "
                      f"(relation_only={item.get('relation_only')})")
                print(f"  claim     : {str(item.get('claim'))[:66]}...")
                print(f"  state     : {item.get('verification_state')}  "
                      f"|  published: {item.get('published')}")
                print(f"  tasks     : {item.get('tasks')}")
                print(f"  evidence  : "
                      f"{[e['execution_id'] for e in item.get('evidence', [])]}")
            print("\nThe claim is quoted from accumulated evidence on OTHER "
                  "tasks, with its own verification state and a citable "
                  "evidence list — not from a mean.")
            return 0
        finally:
            h.close()


if __name__ == "__main__":
    raise SystemExit(main())
