"""Runnable example: real token accounting with provenance.

The gap this example demonstrates: the framework used to record only a
model call's COMPLETION tokens, so a call that billed 9000 prompt + 100
completion tokens was stored as ``llm_tokens=100`` — five orders of
magnitude wrong — and a hand-typed estimate was indistinguishable from a
real measurement in the stored fact.

It shows the corrected accounting, with NO real model, NO network and NO
API key:

1. **the full口径 total** — a provider that reports prompt + completion is
   recorded as the sum; reasoning/cached tokens stay sub-facts and are
   never added again;
2. **a host report is the real spend** — an OpenClaw-style and a
   Hermes-style report (including a per-call breakdown) both normalize to
   the same attempt-level shape and become the record's cost;
3. **provenance travels** — an amended dimension says WHERE its number came
   from, and a declaration is never confused with a measurement;
4. **a real observation is protected** — a framework-measured dimension
   cannot be silently overwritten;
5. **口径 never mix silently** — a legacy completion-only figure and a full
   total are reported as different units rather than summed as one;
6. **a versioned HostUsageReport fills BOTH dimensions in one command** —
   the host's or-host-usage/1 report (an openclaw-shaped fixture) writes
   llm_tokens (provider_usage) AND tool_calls (agent_observed) into
   cost_measured with no hand-typing, and the whitelist governs what may
   be measured.

    PYTHONPATH=src python3 references/examples/usage_accounting.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from or_harness.api import ORHarness  # noqa: E402
from or_harness.core.schema import (  # noqa: E402
    CostVector,
    ExecutionRecord,
    ProblemProfile,
)
from or_harness.core.storage import StorageError  # noqa: E402
from or_harness.strategy.embedding_index import (  # noqa: E402
    LocalHashEmbeddingBackend,
)
from or_harness.world_model.usage import (  # noqa: E402
    host_report_dimensions,
    host_usage_adapter,
    host_usage_cost_vector,
    token_basis_of,
    token_bases_mixed,
    token_breakdown,
)

TASK = {"task_id": "usage_demo", "family": "routing",
        "description": "A small routing instance.",
        "spec": {"n_vars": 4, "n_constraints": 3, "n_int_vars": 4}}

#: A versioned HostUsageReport in the exact shape the host contract
#: specifies (an openclaw-shaped FIXTURE — no OpenClaw is installed here;
#: the real report is produced by the host itself).
HOST_REPORT = {
    "schema": "or-host-usage/1",
    "host": "openclaw",
    "model": "paratera/DeepSeek-V4.1-Flash",
    "scope": {"task_id": "usage_demo", "episode_id": "ep1",
               "attempt_id": "ex_report"},
    "tokens": {"prompt_tokens": 12345, "completion_tokens": 678,
               "reasoning_tokens": 291, "cached_tokens": 0, "calls": 3},
    "tool_calls": 38,
    "tool_calls_lower_bound": 1,
    "measured": ["prompt_tokens", "completion_tokens", "tool_calls"],
    "provenance": {"llm_tokens": "provider_usage",
                   "tool_calls": "agent_observed"},
    "notes": "fixture in the canonical host shape",
}


def main() -> int:
    tmp = tempfile.TemporaryDirectory()
    home = tmp.name
    h = ORHarness(home=home, embedding=LocalHashEmbeddingBackend())

    # ------------------------------------------------------------------
    print("=" * 72)
    print("1. The FULL口径: prompt + completion, sub-facts not added again")
    print("=" * 72)
    usage = {"prompt_tokens": 9000, "completion_tokens": 325,
             "completion_tokens_details": {"reasoning_tokens": 291}}
    breakdown = token_breakdown(usage)
    print(f"prompt_tokens          : {breakdown['prompt_tokens']}")
    print(f"completion_tokens      : {breakdown['completion_tokens']}")
    print(f"reasoning_tokens (sub) : {breakdown['reasoning_tokens']}")
    print(f"llm_tokens (total)     : {breakdown['llm_tokens']}")
    print(f"basis                  : {breakdown['basis']}")
    # 9000 + 325 = 9325. The 291 reasoning tokens are INSIDE the 325.
    assert breakdown["llm_tokens"] == 9325.0
    assert breakdown["reasoning_tokens"] == 291.0

    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print("2. A host usage report is the REAL attempt spend")
    print("=" * 72)
    # OpenClaw's llm_output-style report.
    openclaw = {"source": "openclaw", "model": "MiniMax-M2.7",
                "prompt_tokens": 1200, "completion_tokens": 340}
    vector, info = host_usage_cost_vector(openclaw)
    print(f"openclaw report        : llm_tokens={vector.llm_tokens}, "
          f"host={info['host']}, scope={info['scope']}")
    assert vector.llm_tokens == 1540.0
    # Hermes' post_api_request-style report with a per-call breakdown.
    hermes = {"source": "hermes", "calls": [
        {"usage": {"prompt_tokens": 100, "completion_tokens": 20}},
        {"usage": {"prompt_tokens": 200, "completion_tokens": 30}}]}
    vector2, info2 = host_usage_cost_vector(hermes)
    print(f"hermes report (2 calls): llm_tokens={vector2.llm_tokens} "
          f"(summed from the calls)")
    assert vector2.llm_tokens == 350.0

    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print("3. End to end: the report becomes the record's real cost")
    print("=" * 72)
    work = Path(home) / "ws"
    work.mkdir(parents=True, exist_ok=True)
    (work / "solve.py").write_text(
        "import json\n"
        "with open('result.json', 'w') as fh:\n"
        "    json.dump({'status': 'optimal', 'objective_value': 1.0,\n"
        "               'objective_bound': 1.0, 'runtime_seconds': 0.01}, fh)\n",
        encoding="utf-8")
    record = h.execute(TASK, "S01", str(work / "solve.py"), str(work),
                       solver="highs", episode_id="ep1")
    h.record(record, host_usage=openclaw)
    stored = h.bank.get(record.execution_id)
    print(f"record llm_tokens      : {stored.cost.llm_tokens}")
    print(f"provenance source      : "
          f"{stored.execution_features['cost_provenance']['llm_tokens']}")
    print(f"recorded basis         : "
          f"{stored.execution_features['cost_provenance']['llm_tokens']['basis']}")
    print(f"token_basis_of(record) : {token_basis_of(stored)}")
    assert stored.cost.llm_tokens == 1540.0
    assert stored.execution_features["cost_provenance"]["llm_tokens"][
        "source"] == "provider_usage"

    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print("4. A real observation is protected from a silent overwrite")
    print("=" * 72)
    real_latency = stored.cost.latency_s
    print(f"measured latency_s     : {real_latency}")
    try:
        h.bank.update_cost(record.execution_id, latency_s=0.001)
        raise AssertionError("a framework-measured dimension should be refused")
    except StorageError as exc:
        print(f"silent overwrite       : REFUSED ({str(exc)[:60]}...)")
    # With an explicit force it is allowed (a deliberate correction).
    forced = h.bank.update_cost(record.execution_id, force=True, latency_s=9.0)
    print(f"with force             : latency now {forced.cost.latency_s}")

    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print("5. Token口径 never mix silently")
    print("=" * 72)
    legacy = ExecutionRecord(
        execution_id="ex_legacy", task_id="usage_demo", strategy_id="S01",
        profile_snapshot=ProblemProfile(problem_id="p", family="routing"),
        quality={"status": "optimal", "feasible": True},
        cost=CostVector(llm_tokens=100.0, measured={"llm_tokens"}))
    h.bank.append(legacy)
    print(f"legacy basis           : {token_basis_of(legacy)} "
          "(a completion-only number, no provenance marker)")
    print(f"modern basis           : {token_basis_of(stored)}")
    # Pooling a completion-only figure with a full total is a silent unit
    # error; the framework makes the mismatch DETECTABLE.
    print(f"bases mixed in a pool  : "
          f"{token_bases_mixed([legacy, stored])}")
    assert token_bases_mixed([legacy, stored]) is True
    # The budget reports the口径 of the records that ARE in this episode.
    view = h.budget_view("usage_demo", episode_id="ep1")
    print(f"budget token_basis     : {view['consumption']['token_basis']}")
    assert view["consumption"]["token_basis"] == ["provider_total"]

    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print("6. A versioned HostUsageReport fills BOTH dimensions at once")
    print("=" * 72)
    # The host's or-host-usage/1 report: llm_tokens is the full口径 total,
    # tool_calls is the whole-scope count, and the 'measured' whitelist
    # governs which dimensions may enter cost_measured.
    dimensions, info = host_report_dimensions(HOST_REPORT)
    print(f"dimensions             : {dimensions}")
    print(f"provenance             : {info['provenance']}")
    print(f"basis                  : {info['basis']} "
          f"(reasoning={info['reasoning_tokens']} is a sub-fact)")
    assert dimensions["llm_tokens"] == 12345 + 678
    assert dimensions["tool_calls"] == 38.0
    assert info["provenance"]["llm_tokens"] == "provider_usage"
    assert info["provenance"]["tool_calls"] == "agent_observed"

    # The named adapter LOCATES the report the host wrote (here: a fixture
    # under <home>/host_usage/), so `record --usage-host openclaw` needs no
    # hand-typed numbers at all.
    import json as _json
    report_dir = Path(home) / "host_usage"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "openclaw-usage.ex_report.json"
    report_path.write_text(_json.dumps(HOST_REPORT), encoding="utf-8")
    adapter = host_usage_adapter("openclaw")
    located = adapter.locate({"home": home, "execution_id": "ex_report"})
    print(f"adapter located        : {Path(located).name}")
    assert located is not None
    collected = adapter.collect({"home": home, "execution_id": "ex_report"})
    assert collected is not None and collected[0] == dimensions

    # End to end: one record call writes BOTH dimensions with their own
    # provenance — llm_tokens becomes CALIBRATABLE (provider_usage).
    record2 = h.execute(TASK, "S02", str(work / "solve.py"), str(work),
                        solver="highs", episode_id="ep2")
    h.record(record2, host_usage=dict(HOST_REPORT))
    stored2 = h.bank.get(record2.execution_id)
    prov2 = stored2.execution_features["cost_provenance"]
    print(f"record llm_tokens      : {stored2.cost.llm_tokens} "
          f"({prov2['llm_tokens']['source']})")
    print(f"record tool_calls      : {stored2.cost.tool_calls} "
          f"({prov2['tool_calls']['source']})")
    print(f"cost_measured          : "
          f"{sorted(stored2.cost.measured_dims())}")
    assert stored2.cost.llm_tokens == 12345 + 678
    assert stored2.cost.tool_calls == 38.0
    assert prov2["llm_tokens"]["source"] == "provider_usage"
    assert prov2["tool_calls"]["source"] == "agent_observed"
    assert {"llm_tokens", "tool_calls"} <= stored2.cost.measured_dims()

    # A dimension OUTSIDE the whitelist stays unknown even when the report
    # carries its number: unknown is never zero.
    narrow = dict(HOST_REPORT, measured=["prompt_tokens",
                                         "completion_tokens"])
    dims_narrow, _ = host_report_dimensions(narrow)
    print(f"whitelist narrowed     : {sorted(dims_narrow)} "
          "(tool_calls stays unknown)")
    assert "tool_calls" not in dims_narrow

    h.close()
    tmp.cleanup()
    print()
    print("All assertions passed.")
    print("NOTE: the token figure is the provider's TOTAL (prompt + "
          "completion); reasoning/cached are sub-facts and are never added "
          "again. A host report is a real observation, a hand-typed value is "
          "a declaration, and the two are recorded as such. A versioned "
          "or-host-usage/1 report fills llm_tokens AND tool_calls in one "
          "command — the host produces it, the harness consumes it, and a "
          "missing report leaves the dimensions unknown (never zero).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
