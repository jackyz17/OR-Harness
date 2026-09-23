"""Empty memory bank: the outer agent proposes a method and the framework
records what happened.

Run from the repo root:

    PYTHONPATH=src python3 references/examples/no_catalog.py

The walkthrough answers one question: **what happens when the framework has
no built-in strategy directory at all?** The sequence is the whole flow the
Skill describes, with nothing supplied from a menu:

    recall (nothing)  ->  the agent proposes a method  ->  predict (unknown)
      ->  execute  ->  check the answer  ->  record  ->  recall (the memory)

Two things to watch:

* ``recall`` returns an EMPTY list with a reason. It does not offer a menu,
  and it does not report ``no_memory`` rows at a score of ``-inf``.
* The method's name is one the framework has never seen. Neither ``predict``
  nor ``execute`` refuses it: an unknown cost basis is reported as unknown,
  and the execution runs. Afterwards it is a candidate — because it really
  ran, not because a directory listed it.
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

#: A method id that appears in NO built-in list anywhere. The point of the
#: example is that this changes nothing.
METHOD = "custom:two-phase-milp"

TASK = {
    "task_id": "demo_empty",
    "family": "routing",
    "text": ("Route the fleet to minimise total distance. Every vehicle may "
             "serve at most two depots."),
    "annotations": {"coupling": {"resource_coupling": 0.9,
                                 "temporal_coupling": 0.1,
                                 "route_complexity": 0.85,
                                 "semantic_coupling": 0.8}},
}

#: The solve script the OUTER AGENT writes. The framework never generates
#: code: it runs what it is given and records what came back.
SOLVE = textwrap.dedent("""
    import json
    # A deliberately simple two-phase sketch: build an LP, round, repair.
    objective = 10755.0
    with open("result.json", "w") as fh:
        json.dump({
            "status": "optimal",
            "objective_value": objective,
            "objective_bound": objective,
            "runtime_seconds": 0.02,
            # The solution vector is what a task-level check reads. Without
            # it, `orx check-task` can only report `insufficient`.
            "variables": {"x1": 2, "x2": 3, "x3": 1},
        }, fh)
""")


def step(number: int, title: str) -> None:
    print(f"\n{'=' * 72}\n{number}. {title}\n{'=' * 72}")


def main() -> int:
    work = Path(tempfile.mkdtemp(prefix="orx_nocatalog_"))
    task_path = work / "task.json"
    task_path.write_text(json.dumps(TASK), encoding="utf-8")
    solve_path = work / "solve.py"
    solve_path.write_text(SOLVE, encoding="utf-8")

    h = ORHarness(home=str(work / "home"))
    try:
        # ------------------------------------------------------------------
        step(1, "recall on an EMPTY bank: an empty answer, with a reason")
        # ------------------------------------------------------------------
        empty = h.recall(TASK)
        print(f"recommendations: {empty['recommendations']}")
        print(f"reason: {empty['recommendations_basis']['reason']}")
        assert empty["recommendations"] == [], "no menu may be fabricated"
        assert "NO MEMORY" in empty["recommendations_basis"]["reason"]
        # The old shape must be gone entirely.
        blob = json.dumps(empty["recommendations"])
        assert "no_memory" not in blob and "-inf" not in blob
        print("-> no candidate menu, no placeholder row, no -inf score")

        # ------------------------------------------------------------------
        step(2, "the outer agent proposes a method (and its configuration)")
        # ------------------------------------------------------------------
        print(f"proposed method: {METHOD}")
        print("proposed config: {'time_limit': 120, 'rounding': 'deterministic'}")
        proposal = [METHOD]
        # The framework reports the ones it has NO memory about, rather than
        # inventing evidence for them.
        proposed = h.recall(TASK, candidates=proposal)
        print("candidates_without_evidence: "
              f"{proposed['candidates_without_evidence']}")
        assert proposed["candidates_without_evidence"] == [METHOD]
        print("-> the proposal is a FILTER over real memory, never a source "
              "of it")

        # ------------------------------------------------------------------
        step(3, "predict the candidate: UNKNOWN, not a refusal")
        # ------------------------------------------------------------------
        snapshot = h.predict_cost(TASK, METHOD)
        print(f"source: {snapshot.source}")
        print(f"expected_cost: {snapshot.expected_cost}")
        print(f"note: {snapshot.note}")
        assert snapshot.source == "unknown"
        assert snapshot.expected_cost is None, "unknown is not zero"
        print("-> an unknown cost basis never blocks a legitimate execution")

        # ------------------------------------------------------------------
        step(4, "execute the proposed method in the sandbox")
        # ------------------------------------------------------------------
        record = h.execute(TASK, METHOD, str(solve_path), str(work),
                           solver="highs", episode_id="ep1")
        print(f"execution_id: {record.execution_id}")
        print(f"strategy_id: {record.strategy_id}   (the framework's own "
              "record of what RAN)")
        print(f"solver status: {record.quality['status']}  "
              f"objective={record.quality.get('objective')}")
        assert record.strategy_id == METHOD
        print("-> no directory membership was required to run it")

        # ------------------------------------------------------------------
        step(5, "check the ANSWER against the task, not just the solver")
        # ------------------------------------------------------------------
        # `execute` answered "did the solver solve its own model?". This
        # answers "is that a valid answer to the task?" — a different
        # question, and the one a later recall must not confuse.
        checked = h.check_task_result(
            record.execution_id,
            {"reference_objective": 10755.0,
             "integer": {"variables": ["x1", "x2", "x3"]}},
            episode_id="ep1")
        report = checked["report"]
        print(f"task verdict: {report['state']}")
        for check in report["checks"]:
            print(f"  - {check['check']}: {'OK' if check.get('ok') else 'no'}")
        print(f"unchecked: {report['scope']['unchecked']}")
        assert report["state"] == "passed"
        print("-> a passed task check is still not 'knowledge verified'; the "
              "report names what it did NOT check")

        # ------------------------------------------------------------------
        step(6, "record the fact (with the harness's own cost backfill)")
        # ------------------------------------------------------------------
        outcome = h.record(record, override={"llm_tokens": 1840,
                                            "tool_calls": 9})
        print(f"recorded: {outcome['recorded']}")
        print(f"induction hints: "
              f"{[hint['pattern'] for hint in outcome['induction_hints']]}")
        print("-> recording accumulates EVIDENCE; it changes no knowledge")

        # ------------------------------------------------------------------
        step(7, "recall again: the method is now a candidate, because it ran")
        # ------------------------------------------------------------------
        recalled = h.recall(TASK, candidates=proposal)
        rec = recalled["recommendations"][0]
        print(f"strategy_id: {rec['strategy_id']}")
        print(f"evidence: {rec['evidence']}")
        print(f"evidence_refs: {rec['evidence_refs']}")
        print(f"observed quality: {rec['expected']['quality']}")
        print(f"observed cost: {rec['expected']['cost']}")
        assert rec["strategy_id"] == METHOD
        assert rec["evidence"] == "conditional_stats"
        assert rec["evidence_refs"] == [record.execution_id]
        assert "name" not in rec, "no directory content may be attached"
        print("-> the memory about it came from the execution, not from a "
              "directory")

        # ------------------------------------------------------------------
        step(8, "the content that travels is the ENTRY's own, never a menu's")
        # ------------------------------------------------------------------
        # A claim needs evidence from >=2 INDEPENDENT tasks: repeating one
        # task is repetition, not reproduction. So the agent runs the method
        # once more on a second task of the same structural cell.
        second_task = dict(TASK, task_id="demo_empty_2")
        second = h.execute(second_task, METHOD, str(solve_path), str(work),
                           solver="highs", episode_id="ep2")
        h.record(second, override={"llm_tokens": 1900, "tool_calls": 9})
        induced = h.induce(strategy_id=METHOD)
        created = induced["results"][0].get("created")
        print(f"induction result: created={created} "
              f"skipped={induced['results'][0].get('skipped')!r}")
        entries = [h.sbank.get(created)]
        print(f"entries for {METHOD}: {[e.entry_id for e in entries]}")
        for entry in entries:
            print(f"  strategy_type={entry.strategy_type!r} "
                  f"actions={entry.actions} "
                  f"fallback={entry.fallback_strategy_id!r}")
            assert entry.strategy_type is None and entry.actions == []
        print("-> the framework recorded that it does NOT know the method's "
              "type/actions; it did not copy them from anywhere")
        # Publishing needs an ADMISSION check the framework computes from real
        # executions — a claim about future tasks needs independent evidence,
        # which is what the second task provided.
        verify = {
            "purpose": "rule",
            "claim": f"{METHOD} reaches the reference objective in this cell",
            "check": {"reference_objective": 10755.0},
            "executions": [h.bank.get(record.execution_id).to_dict()],
            "supporting": [h.bank.get(second.execution_id).to_dict()],
        }
        verified = h.induce(strategy_id=METHOD, verify=verify)["results"][0]
        print(f"admission verdict: "
              f"{(verified.get('verification') or {}).get('state')}")
        assert (verified.get("verification") or {}).get("state") == "verified"
        # A harness that DOES know can say so, and that is what travels.
        entry = h.sbank.get(verified.get("created") or verified.get("updated"))
        entry.strategy_type = "decomposition"
        entry.actions = ["solve the LP relaxation",
                         "round and repair deterministically"]
        h.sbank.update(entry)
        rec = h.recall(TASK)["recommendations"][0]
        knowledge = rec["knowledge"]
        print(f"evidence kind now: {rec['evidence']}")
        print(f"after the harness declares it: "
              f"{knowledge['strategy_type']} / {knowledge['actions']}")
        assert rec["evidence"] == "strategic_entry"
        assert knowledge["strategy_type"] == "decomposition"
        print("-> the entry's content is the entry's; a directory contributes "
              "nothing to it")

        # ------------------------------------------------------------------
        step(9, "no candidates supplied -> no menu is invented")
        # ------------------------------------------------------------------
        plan = h.plan_next(TASK, "ep2")
        print(f"plan status: {plan['status']}")
        print(f"reason: {plan['truncation_reason']}")
        assert plan["status"] == "no_candidates"
        print("-> the planner requires the caller's candidates; it has no "
              "default set")

        # ------------------------------------------------------------------
        step(10, "history is readable, and nothing was back-filled")
        # ------------------------------------------------------------------
        print(f"experience bank count: {h.bank.count()}")
        print(f"strategic bank count:  {h.sbank.count()}")
        print("-> the whole loop above worked with no built-in strategy "
              "directory in the codebase")
        print(f"\nAll assertions passed. Working directory: {work}")
        return 0
    finally:
        h.close()


if __name__ == "__main__":
    raise SystemExit(main())
