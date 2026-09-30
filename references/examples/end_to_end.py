"""Runnable example: ONE task, cold start to close-out, including a failure.

This is the whole loop the Skill describes, with no prior memory and no
real model — a fixed-output stub provider is injected, so what it shows is
the PROCESS, never model accuracy. Read it when you want to see, in order:

    profile -> recall (empty) -> propose 2 candidates -> predict both
      -> execute the chosen one -> check-task FAILS -> record the failure
      -> fix the model -> execute again -> check-task PASSES -> record
      -> close-episode -> read the calibration summary

The failure/repair path is deliberate: a model written wrong and solved to
a legal optimum is the most common real case, so the example shows the
check catching it, the failed attempt STAYING on record with its real cost,
and the retry needing its OWN prediction.

Run from the repo root:

    PYTHONPATH=src python3 references/examples/end_to_end.py

What is real here:
  * the CLI/API calls, the sandbox run of solve.py, the checks, the
    staging/recording, the close-out and the calibration summary;
  * the token figure comes from a labelled host usage report (the
    canonical shape), NOT from a real provider.

What is simulated:
  * the world-model answers (benefit/cost/risk) — a stub, so treat the
    numbers as fixtures, not forecasts.

The CLI spellings are the SAME ones the Skill's tables use; the Python API
is shown once at the end, so an agent working in either mode sees the
mapping.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"

TASK = {
    "task_id": "e2e_demo",
    "family": "routing",
    "text": "Ship 100 units from the depot to the store; whole pallets only.",
    "spec": {"n_vars": 12, "n_constraints": 6, "n_int_vars": 12},
}

# Two candidates, each with a described METHOD (an undescribed candidate
# makes the prediction request unanswerable).
CANDIDATE_A = {
    "action_type": "execute_strategy", "task_id": "e2e_demo",
    "episode_id": "ep1", "strategy_id": "direct-milp", "solver": "highs",
    "method": {"name": "direct MILP",
               "steps": ["write the MILP", "solve it in one shot"]},
    "config": {"time_limit": 60},
}
CANDIDATE_B = {
    "action_type": "execute_strategy", "task_id": "e2e_demo",
    "episode_id": "ep1", "strategy_id": "relax-and-fix", "solver": "highs",
    "method": {"name": "relax-and-fix",
               "steps": ["LP relaxation", "fix integer decisions",
                         "re-solve"]},
}

# A WRONG model first: it answers the relaxation with FRACTIONAL pallets.
# The solver is happy (optimal, gap 0); the TASK check is not.
SOLVE_WRONG = """
    import json, os
    result = {"status": "optimal", "objective_value": 42.0,
              "objective_bound": 42.0, "runtime_seconds": 0.01,
              "variables": {"x_depot": 42.5, "x_store": 57.5}}
    aid = os.environ.get("OR_ACTION_ID")
    if aid:
        result["method_performed"] = {
            "action_id": aid, "name": "direct MILP",
            "steps": ["write the MILP", "solve it in one shot"]}
    with open("result.json", "w") as fh:
        json.dump(result, fh)
"""

# The repair: whole pallets only. Same method, corrected model.
SOLVE_RIGHT = """
    import json, os
    result = {"status": "optimal", "objective_value": 42.0,
              "objective_bound": 42.0, "runtime_seconds": 0.01,
              "variables": {"x_depot": 43.0, "x_store": 57.0}}
    aid = os.environ.get("OR_ACTION_ID")
    if aid:
        result["method_performed"] = {
            "action_id": aid, "name": "direct MILP",
            "steps": ["write the MILP", "solve it in one shot"]}
    with open("result.json", "w") as fh:
        json.dump(result, fh)
"""

# The host's usage report for the attempt (a fixture in the canonical
# versioned shape; no OpenClaw/Hermes is installed in this environment).
# One report carries BOTH dimensions: llm_tokens (the full口径 total,
# provider_usage) and tool_calls (the whole-scope count, agent_observed).
HOST_USAGE = {
    "schema": "or-host-usage/1",
    "host": "example-fixture",
    "model": "stub-model",
    "tokens": {"prompt_tokens": 700, "completion_tokens": 300},
    "tool_calls": 12,
    "tool_calls_lower_bound": 1,
    "measured": ["prompt_tokens", "completion_tokens", "tool_calls"],
    "provenance": {"llm_tokens": "provider_usage",
                   "tool_calls": "agent_observed"},
}

# The check that catches the fractional answer: the integer domain.
CHECK = {"integer": {"variables": ["x_depot", "x_store"]}}


def _orx(home: Path, *args: str) -> dict:
    """Run one `orx` command and return its `result` object."""
    proc = subprocess.run(
        [sys.executable, "-m", "or_harness.cli", "--home", str(home), *args],
        capture_output=True, text=True, cwd=str(ROOT),
        env={**__import__("os").environ, "PYTHONPATH": str(SRC)})
    if not proc.stdout.strip():
        raise RuntimeError(f"orx {' '.join(args)} produced no output: "
                           f"{proc.stderr[:400]}")
    payload = json.loads(proc.stdout.splitlines()[-1])
    if proc.returncode == 2:
        raise RuntimeError(f"orx {' '.join(args)} exited 2: "
                           f"{payload.get('summary')}")
    return payload["result"]


def main() -> int:
    home = Path(tempfile.mkdtemp(prefix="orx_e2e_"))
    workspace = home / "ws"
    workspace.mkdir(parents=True, exist_ok=True)
    task_path = home / "task.json"
    task_path.write_text(json.dumps(TASK), encoding="utf-8")
    cand_path = home / "cand.json"
    cand_path.write_text(json.dumps(CANDIDATE_A), encoding="utf-8")

    print(f"home: {home}\n")

    # ---- 1. understand (profile) and recall -----------------------------
    profiled = _orx(home, "profile", "--task", str(task_path))
    print(f"1. profile: family={profiled['profile'].get('family')} "
          f"coupling keys={sorted((profiled.get('coupling') or {}).keys())}")
    recalled = _orx(home, "recall", "--task", str(task_path), "--top", "3")
    print(f"   recall: {len(recalled.get('recommendations') or [])} "
          f"recommendation(s); basis="
          f"{(recalled.get('recommendations_basis') or {}).get('reason', '')[:60]}")
    # Cold start: no memory, no text channel configured -> no vector_recall.
    assert not recalled.get("recommendations")
    assert "vector_recall" not in recalled, (
        "the text channel is off, so the key is ABSENT (a different fact "
        "from an empty result)")

    # ---- 2. execute the CHOSEN candidate --------------------------------
    # The example drives the run through the Python API so the stub provider
    # can be injected; every call is the same one the CLI makes.
    sys.path.insert(0, str(SRC))
    from or_harness.api import ORHarness
    from or_harness.strategy.embedding_index import (  # noqa: E402
        LocalHashEmbeddingBackend,
    )
    from or_harness.world_model.provider import WorldModelProvider  # noqa: E402

    class StubProvider(WorldModelProvider):
        """Declared synthetic answers -- a labelled stub, not a real model."""

        name = "e2e-stub"

        def predict(self, request, timeout_s=None):
            return {
                "payload": {
                    "benefit": {"kind": "solution_quality",
                                "metric": "normalized_objective_gap",
                                "unit": "1-gap", "value": 0.8,
                                "baseline": {"kind": "conditional_stats",
                                             "value": 0.7}},
                    "cost": {"solver_runtime_s": 3.0}},
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
                "error": None, "latency_s": 0.01}

    harness = ORHarness(home=str(home), world_model=StubProvider(),
                        embedding=LocalHashEmbeddingBackend())

    prediction = harness.predict_strategy_outcome(
        TASK, CANDIDATE_A, "ep1")
    # A PREDICTION is an expectation, never a fact: no execution exists yet.
    assert prediction.status == "valid", prediction.notes
    print(f"2. predicted {CANDIDATE_A['strategy_id']}: "
          f"prediction_id={prediction.prediction_id} status={prediction.status} "
          f"(this is an EXPECTATION, not a measurement)")

    # ---- 3. run the WRONG model, and let the check catch it -------------
    wrong = workspace / "solve_wrong.py"
    wrong.write_text(textwrap.dedent(SOLVE_WRONG), encoding="utf-8")
    first = harness.execute(TASK, None, str(wrong), str(workspace),
                            solver=None, episode_id="ep1",
                            prediction_id=prediction.prediction_id)
    checked = harness.check_task_result(first.execution_id, check=CHECK)
    print(f"3. attempt 1: solver status={first.quality.get('status')} "
          f"(its OWN model), task check state={checked['state']}")
    print(f"   report.conclusion={checked['report']['conclusion'][:70]}")
    print(f"   report.scope.unchecked={checked['report']['scope']['unchecked'][:2]}")
    # The solver was happy; the TASK check was not. These are different
    # questions and the second one is why the check exists.
    assert checked["state"] == "failed"

    # ---- 4. record the FAILED attempt with its real cost ----------------
    # The documented order is check (step 6) THEN record (step 7), and the
    # verdict survives it: a check written to the staged execution is
    # carried onto the recorded fact. No usage report was kept for THIS
    # call, so the token count is hand-declared: a bare
    # `--override llm_tokens=` is an `agent_estimate` — kept and shown, but
    # never used as a calibration truth. (Do not pass --usage-file AND
    # --override for the SAME dimension: the host report is a real
    # observation, so the override is refused rather than silently
    # overwriting it.)
    recorded = harness.record(first, override={"llm_tokens": 1000})
    stored = harness.bank.get(recorded["execution_id"])
    provenance = stored.execution_features["cost_provenance"]["llm_tokens"]
    print(f"4. recorded the failure: llm_tokens={stored.cost.llm_tokens} "
          f"source={provenance['source']} (a DECLARATION, not a measurement)")
    # The failed attempt is evidence and is NEVER dropped, whatever its
    # cost standing, and its FAILED verdict is still on the fact.
    assert provenance["source"] == "agent_estimate"
    assert stored.execution_features.get("task_check", {}).get("state") \
        == "failed"

    # ---- 5. fix the model, and RE-PREDICT the corrected attempt ---------
    # A retry of a CONSUMED attempt needs its OWN prediction: the first one
    # was tested by attempt 1.
    corrected = harness.predict_strategy_outcome(TASK, CANDIDATE_B, "ep1")
    right = workspace / "solve_right.py"
    right.write_text(textwrap.dedent(SOLVE_RIGHT), encoding="utf-8")
    second = harness.execute(TASK, None, str(right), str(workspace),
                             solver=None, episode_id="ep1",
                             prediction_id=corrected.prediction_id)
    checked2 = harness.check_task_result(second.execution_id, check=CHECK)
    print(f"5. attempt 2 ({CANDIDATE_B['strategy_id']}): "
          f"task check state={checked2['state']}")
    assert checked2["state"] == "passed"
    # This attempt DID keep the host's report, so BOTH its token figure and
    # its tool-call count stand as real observations and can enter
    # calibration — no hand-typing, one report.
    harness.record(second, host_usage=HOST_USAGE)
    stored2 = harness.bank.get(second.execution_id)
    source2 = stored2.execution_features["cost_provenance"]["llm_tokens"]
    tool_source = stored2.execution_features["cost_provenance"]["tool_calls"]
    print(f"   recorded with the host report: llm_tokens="
          f"{stored2.cost.llm_tokens} source={source2['source']} "
          f"basis={source2.get('basis')}")
    print(f"   tool_calls={stored2.cost.tool_calls} "
          f"source={tool_source['source']} (the whole-scope count)")
    assert source2["source"] == "provider_usage"
    assert source2["basis"] == "provider_total"
    assert tool_source["source"] == "agent_observed"
    assert {"llm_tokens", "tool_calls"} <= stored2.cost.measured_dims()

    # ---- 6. close the episode and read the calibration ------------------
    closed = harness.close_episode("e2e_demo", "ep1",
                                   terminal_state="completed")
    summary = closed["calibration_summary"]
    print(f"6. close-episode: terminal="
          f"{closed['closeout']['terminal_state']}, "
          f"task_checks={closed['task_checks']}")
    print(f"   evaluations={len(closed['evaluations'])} "
          f"(one per bound prediction), "
          f"n_evaluated={summary.get('n_evaluated')}")
    # The FIRST attempt's tool_calls was never backfilled (its llm_tokens
    # was hand-declared), so the close-out WARNS about it: unknown is never
    # zero, and the gap is visible at the moment it can still be repaired.
    cost_warnings = closed.get("cost_completeness_warnings")
    if cost_warnings:
        unknown = sorted(cost_warnings["unknown_dimensions"])
        print(f"   cost warnings        : {unknown} still UNKNOWN on "
              f"{cost_warnings['unknown_dimensions'][unknown[0]]}")
    assert "cost_completeness_warnings" in closed
    assert "tool_calls" in closed["cost_completeness_warnings"][
        "unknown_dimensions"]
    # The failed attempt is still visible in the episode's occurrence
    # count: the retry did not erase it.
    occurrence = (summary.get("occurrence") or {})
    print(f"   occurrence events="
          f"{sorted(occurrence.keys()) if isinstance(occurrence, dict) else occurrence}")
    assert closed["task_checks"]["verdicts"].get("failed") == 1
    assert closed["task_checks"]["verdicts"].get("passed") == 1

    harness.close()

    # ---- 7. the CLI equivalent of the same decisions --------------------
    # Shown once so the mapping between the two modes is explicit. These
    # need a configured provider, so they are printed, not run here.
    print("\n7. the same decisions through the CLI (provider required):")
    for line in [
        f"orx --home {home} --world-model $URL::MODEL profile --task {task_path}",
        f"orx --home {home} --world-model $URL::MODEL recall --task {task_path} --top 3",
        f"orx --home {home} --world-model $URL::MODEL predict-strategy "
        f"--task {task_path} --candidate {cand_path} --episode ep1",
        f"orx --home {home} execute --task {task_path} --episode ep1 "
        "--prediction <sp_...> --code solve.py --workspace ws",
        f"orx --home {home} check-task <execution_id> --check '{json.dumps(CHECK)}'",
        f"orx --home {home} record --from-staged <execution_id> --usage-host openclaw",
        f"orx --home {home} close-episode --task e2e_demo --episode ep1 "
        "--terminal completed",
    ]:
        print(f"   {line}")

    print("\nAll assertions held: cold start -> failed check -> repair -> "
          "close-out. The world-model numbers are STUB fixtures; the "
          "sandbox run, checks, costs and calibration are real.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
