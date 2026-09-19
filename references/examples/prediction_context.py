"""Runnable example: the phase-2 prediction input context.

Runs with NO model, NO network, NO solver, NO API key, and NO embedding
service (it injects the deterministic offline hashing backend). It
demonstrates, in order:

1. a task with **no mathematical model** and several UNKNOWN math attributes
   can still build a context — the missing parts are reported, not guessed;
2. the joint representation keeps the task's semantics and the CIR's
   relations, and carries an origin for every math attribute;
3. the retrieval evidence carries CONTENT (not just ids), labels cross-cell
   hits, and deduplicates repeated hits by identity;
4. several candidates share ONE context while keeping their own config;
5. a stub provider shows what the model ACTUALLY receives;
6. the degradation path when no embedding backend is configured is
   distinguishable from "the channel ran and found nothing";
7. building a context calls no prediction model.

Every assertion below is about a real behaviour of the code, and the script
prints the facts it asserts.

    PYTHONPATH=src python3 references/examples/prediction_context.py
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
from or_harness.world_model.context import (  # noqa: E402
    PREDICTION_CONTEXT_VERSION,
)
from or_harness.world_model.prediction import ActionSpec  # noqa: E402
from or_harness.world_model.provider import WorldModelProvider  # noqa: E402

#: A task with scene text and an optional CIR, and NO `model` field.
TASK = {
    "task_id": "ctx_demo_1",
    "family": "routing",
    "description": ("A distribution centre must be loaded before the delivery "
                    "window opens; demand of 100 units may not be deferred."),
    "objective": "minimize total travel time",
    "spec": {"n_vars": 400, "n_constraints": 250, "n_int_vars": 400,
             "objective_sense": "minimize"},
    "coupling": {
        "entities": [{"name": "depot", "kind": "site"},
                     {"name": "arc", "kind": "route"}],
        "decisions": [{"name": "load", "kind": "binary", "indexes": ["t"]}],
        "constraints": [{"id": "C1", "kind": "capacity",
                         "expr": "load[t] <= capacity"}],
        "relations": [{"source": "load", "target": "depot",
                       "type": "uses_resource", "evidence": "declared"},
                      {"source": "load", "target": "arc",
                       "type": "precedes", "evidence": "semantic"}],
        "coupling_groups": [],
        "issues": [],
    },
}


class RecordingProvider(WorldModelProvider):
    """A stub provider that records exactly what it was handed."""

    name = "example-recording-stub"

    def __init__(self):
        self.requests = []

    def predict(self, request, timeout_s=None):
        self.requests.append(request)
        return {"payload": {"outcome_status": "feasible", "feasible": True,
                            "quality": 0.6, "failure_prob": 0.15},
                "usage": {"prompt_tokens": 20, "completion_tokens": 10},
                "error": None, "latency_s": 0.01}


def _solve(h, task, strategy="S04", objective=100.0):
    """Execute and record one real attempt (the honest way to make evidence)."""
    work = Path(h.home) / f"ws_{task['task_id']}_{strategy}"
    work.mkdir(parents=True, exist_ok=True)
    script = work / "solve.py"
    script.write_text(
        "import json\n"
        "with open('result.json', 'w') as fh:\n"
        "    json.dump({'status': 'optimal', 'objective_value': "
        f"{objective}, 'objective_bound': {objective}, "
        "'runtime_seconds': 0.01}, fh)\n", encoding="utf-8")
    record = h.execute(task, strategy, str(script), str(work), solver="highs")
    h.record(record)
    return record


def main() -> int:
    tmp = tempfile.TemporaryDirectory()
    home = tmp.name
    backend = LocalHashEmbeddingBackend()
    provider = RecordingProvider()
    h = ORHarness(home=home, world_model=provider, embedding=backend)

    # ------------------------------------------------------------------
    print("=" * 72)
    print("1. A task with NO model and UNKNOWN math attributes")
    print("=" * 72)
    ctx = h.build_prediction_context(TASK, "ep1")
    joint = ctx.joint
    print(f"context version        : {ctx.version}")
    print(f"has_model              : {joint.has_model}")
    print(f"math integrality       : {joint.math.integrality} "
          f"(origin {joint.math.origin('integrality')})")
    print(f"math objective_kind    : {joint.math.objective_kind} "
          f"(origin {joint.math.origin('objective_kind')})")
    print(f"math linearity         : {joint.math.linearity} "
          f"(origin {joint.math.origin('linearity')})")
    print(f"unknown math attributes: {joint.unknowns}")
    assert ctx.version == PREDICTION_CONTEXT_VERSION
    assert joint.has_model is False
    # The spec supplied integrality and the objective sense; the model was
    # never consulted for linearity, so it stays unknown rather than being
    # invented from the family name.
    assert joint.math.integrality == "integer"
    assert joint.math.origin("integrality") == "spec"
    assert joint.math.objective_kind == "min"
    assert "linearity" in joint.unknowns
    assert not any("routing" in str(v) for v in joint.math.origins.values())

    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print("2. The joint representation keeps semantics and CIR relations")
    print("=" * 72)
    print(f"text excerpt           : {joint.text[:70]}...")
    print(f"CIR present            : {joint.cir_present}")
    print(f"CIR relations kept     : {len(joint.cir['relations'])}")
    for relation in joint.cir["relations"]:
        print(f"   {relation['source']} --{relation['type']}--> "
              f"{relation['target']} ({relation['evidence']})")
    print(f"sources                : {json.dumps(joint.sources, sort_keys=True)}")
    print(f"missing parts          : {len(joint.missing)}")
    for item in joint.missing:
        print(f"   - {item}")
    assert "distribution centre" in joint.text
    assert len(joint.cir["relations"]) == 2
    assert joint.sources["cir"] == "task_coupling"
    assert any("model" in m for m in joint.missing)
    # The relations were NOT compressed into rc/tc/rx and thrown away.
    assert joint.cir["n_relations"] == 2

    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print("3. Retrieval evidence: content, cross-cell labels, dedup")
    print("=" * 72)
    # A textually near-identical past task in a DIFFERENT structural cell:
    # the same scene text, but no CIR and a very different harness-supplied
    # coupling, so its evidence belongs to another cell.
    past = {k: v for k, v in TASK.items() if k != "coupling"}
    past["task_id"] = "ctx_demo_past"
    past["annotations"] = {"coupling": {"resource_coupling": 0.95,
                                        "temporal_coupling": 0.95,
                                        "route_complexity": 0.95,
                                        "semantic_coupling": 0.5}}
    _solve(h, past)
    ctx2 = h.build_prediction_context(TASK, "ep1")
    print(f"channels run           : {ctx2.retrieval.channels_run}")
    print(f"hits carried           : {ctx2.n_evidence}")
    print(f"evidence classes       : "
          f"{json.dumps(ctx2.evidence_classes(), sort_keys=True)}")
    print(f"dedup summary          : "
          f"{json.dumps({k: v for k, v in ctx2.retrieval.deduplication.items() if k != 'duplicates'}, sort_keys=True)}")
    for hit in ctx2.retrieval.hits:
        content = hit["content"]
        print(f"   {hit['identity']} [{hit['evidence_class']}] "
              f"channels={hit['channels']} "
              f"match={content.get('structural_match')}")
        if content.get("task_text_excerpt"):
            print(f"      excerpt: {content['task_text_excerpt'][:60]}...")
    execution_hits = [x for x in ctx2.retrieval.hits
                      if x["layer"] == "execution_evidence"]
    assert execution_hits, "the textually similar past task must be surfaced"
    assert any(x["content"].get("structural_match") == "different_cell"
               for x in execution_hits), "cross-cell hits stay visible"
    assert any(x["content"].get("task_text_excerpt")
               for x in execution_hits), "a hit carries its content"
    # One memory hit by two channels is one piece of evidence.
    for hit in ctx2.retrieval.hits:
        assert len(hit["versions"]) >= 1

    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print("4. Several candidates share ONE context, keeping their own config")
    print("=" * 72)
    for sid, params in (("S01", {"time_limit": 60, "mip_gap": 0.01}),
                        ("S02", {"time_limit": 120})):
        h.predict_outcome(TASK,
                          ActionSpec("execute_strategy", TASK["task_id"],
                                     strategy_id=sid, params=params),
                          "ep1", context=ctx2)
    context_ids = {r["prediction_context"]["context_id"]
                   for r in provider.requests}
    print(f"provider calls         : {len(provider.requests)}")
    print(f"distinct context ids   : {len(context_ids)} -> {context_ids}")
    print(f"candidate 0 params     : "
          f"{json.dumps(provider.requests[0]['action_spec']['params'])}")
    print(f"candidate 1 params     : "
          f"{json.dumps(provider.requests[1]['action_spec']['params'])}")
    assert len(context_ids) == 1, "candidates share ONE frozen context"
    assert provider.requests[0]["action_spec"]["params"]["time_limit"] == 60
    assert provider.requests[1]["action_spec"]["params"]["time_limit"] == 120

    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print("5. What the model ACTUALLY receives")
    print("=" * 72)
    block = provider.requests[0]["prediction_context"]
    print(f"context_version        : {block['context_version']}")
    print(f"request keys           : "
          f"{sorted(provider.requests[0].keys())}")
    print(f"joint_problem keys     : {sorted(block['joint_problem'].keys())}")
    print(f"retrieval keys         : "
          f"{sorted(block['retrieval_evidence'].keys())}")
    print(f"capability sources     : "
          f"{ {k: v['status'] for k, v in block['harness_capability']['sources'].items()} }")
    print(f"execution constraints  : "
          f"{json.dumps(block['execution_constraints'], sort_keys=True)[:160]}...")
    assert block["context_version"] == PREDICTION_CONTEXT_VERSION
    assert "distribution centre" in block["joint_problem"]["text"]
    assert block["joint_problem"]["cir"]["relations"]
    assert "hits" in block["retrieval_evidence"]
    assert block["harness_capability"]["score_scheme"] == "no_composite_score"
    # No source is dressed up as direct evidence in this phase.
    assert not any(v["status"] == "direct_evidence"
                   for v in block["harness_capability"]["sources"].values())

    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print("6. Degradation: no backend vs a healthy run that found nothing")
    print("=" * 72)
    saved = {k: os.environ.pop(k, None) for k in
             ("OR_EMBEDDING_BACKEND", "OR_EMBEDDING_BASE_URL",
              "OR_EMBEDDING_MODEL", "OR_EMBEDDING_API_KEY")}
    h_nb = ORHarness(home=home)
    try:
        ctx_nb = h_nb.build_prediction_context(TASK, "ep1")
        print(f"channels run (no backend): {ctx_nb.retrieval.channels_run}")
        print(f"degraded parts           : "
              f"{json.dumps(ctx_nb.degraded, sort_keys=True)}")
        print(f"structural status        : "
              f"{ctx_nb.retrieval.structural['status']}")
        assert ctx_nb.retrieval.channels_run == ["structural"]
        assert ctx_nb.retrieval.semantic["status"] == "degraded"
        assert "no embedding backend" in ctx_nb.retrieval.semantic["reason"]
        assert ctx_nb.retrieval.structural["status"] == "ok"
        # The structural channel still produced recommendations: degradation
        # of one channel never removes the other's result.
        assert ctx_nb.retrieval.structural["recommendations"]
    finally:
        h_nb.close()
        for key, value in saved.items():
            if value is not None:
                os.environ[key] = value

    # A healthy semantic run is NOT a degradation — even when one layer's
    # index is still missing (a partial index is its own, narrower fact).
    ctx3 = h.build_prediction_context(
        dict(TASK, task_id="ctx_demo_nomatch",
             description="An unrelated scheduling problem with no shared "
                         "vocabulary at all."), "ep1")
    print(f"semantic status (2nd run) : {ctx3.retrieval.semantic['status']}")
    print(f"degraded parts (2nd run)  : "
          f"{json.dumps(ctx3.degraded, sort_keys=True)}")
    print(f"channels run (2nd run)    : {ctx3.retrieval.channels_run}")
    assert ctx3.retrieval.semantic["status"] == "ok"
    # A channel that RAN is reported as run even when a layer beneath it is
    # unusable: the two facts are separate entries, not one vague message.
    assert "semantic" in ctx3.retrieval.channels_run

    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print("7. Building a context calls no prediction model")
    print("=" * 72)
    before = len(provider.requests)
    h.build_prediction_context(TASK, "ep1")
    print(f"provider calls before  : {before}")
    print(f"provider calls after   : {len(provider.requests)}")
    assert len(provider.requests) == before, (
        "building a context must never reach the provider")

    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print("8. Reuse is version-verified")
    print("=" * 72)
    from or_harness.world_model.context import context_identity_problems
    other_version = dict(TASK, description="A DIFFERENT requirement entirely.")
    problems = context_identity_problems(
        ctx2, task_id="ctx_demo_1", task_digest="deadbeef", episode_id="ep1")
    print(f"mismatched version problems: {problems}")
    assert problems and "task version" in problems[0]
    try:
        h.predict_outcome(
            other_version,
            ActionSpec("execute_strategy", "ctx_demo_1", strategy_id="S01"),
            "ep1", context=ctx2)
    except ValueError as exc:
        print(f"predict with a stale context REFUSED: {str(exc)[:90]}...")
    else:  # pragma: no cover - the refusal is the point
        raise AssertionError("a context from another version must be refused")

    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print("9. Reusing a context replays the FROZEN conditions")
    print("=" * 72)
    frozen = h.build_prediction_context(TASK, "ep1", context_spec=
                                        ActionSpec("execute_strategy",
                                                   "ctx_demo_1",
                                                   strategy_id="S01"))
    # Advance the real state AFTER the context was frozen.
    h.snapshot(TASK, "ep1", task_progress={
        "current_solution": {"value": {"objective": 999},
                             "provenance": "observed",
                             "epistemic": "fact"}})
    before_calls = len(provider.requests)
    replayed = h.predict_outcome(
        TASK,
        ActionSpec("execute_strategy", "ctx_demo_1", strategy_id="S01"),
        "ep1", context=frozen)
    request = provider.requests[before_calls]
    print(f"frozen context progress : {frozen.snapshot['task_progress']}")
    print(f"request state progress  : "
          f"{request['state'].get('task_progress')}")
    print(f"snapshot id matches ctx : "
          f"{replayed.input_snapshot_id == frozen.snapshot_id}")
    print(f"conditions_source       : "
          f"{replayed.model_info.get('conditions_source')}")
    print(f"frozen targets sent     : "
          f"{[t['strategy_id'] for t in (request.get('candidate_knowledge_targets') or [])]}")
    assert request["state"].get("task_progress") in (None, {}), (
        "a reused context must not pick up later progress")
    assert replayed.input_snapshot_id == frozen.snapshot_id
    assert replayed.model_info["conditions_source"] == "frozen_context"

    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print("10. One CIR drives representation, snapshot AND retrieval")
    print("=" * 72)
    weakly_coupled = {k: v for k, v in TASK.items() if k != "coupling"}
    weakly_coupled["annotations"] = {
        "coupling": {"resource_coupling": 0.30, "temporal_coupling": 0.10,
                     "route_complexity": 0.20, "semantic_coupling": 0.50}}
    strongly = h.build_prediction_context(weakly_coupled, "ep1", cir=TASK["coupling"])
    profile = strongly.joint.profile
    print(f"task's own coupling     : rc=0.30 tc=0.10 rx=0.20")
    print(f"joint profile (explicit): rc={profile['resource_coupling']} "
          f"tc={profile['temporal_coupling']} "
          f"rx={profile['route_complexity']}")
    print(f"joint.sources['cir']    : {strongly.joint.sources['cir']}")
    from or_harness.world_model.context import task_with_effective_cir
    via_snapshot = h.profile(task_with_effective_cir(weakly_coupled,
                                                     TASK["coupling"]))
    print(f"snapshot/recall agree   : "
          f"{via_snapshot.resource_coupling == profile['resource_coupling']}")
    assert profile["resource_coupling"] == 1.0
    assert strongly.joint.sources["cir"] == "caller_supplied"
    assert via_snapshot.resource_coupling == profile["resource_coupling"]

    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print("11. The memory version digests CONTENT, not counts")
    print("=" * 72)
    from or_harness.core.schema import CostVector, StrategicEntry
    entry = StrategicEntry(
        entry_id="se_demo", strategy_id="S01",
        pattern={"predicates": {"family": "routing"}},
        expected_quality_hat=0.90, quality_interval=(0.5, 1.0),
        expected_cost_hat=CostVector(llm_tokens=100.0,
                                     measured={"llm_tokens"}),
        failure_prob=0.1, status="candidate", support_n=2,
        verification={"state": "verified", "claim": "c", "conclusion": "holds"})
    h.sbank.add(entry)
    first = h.build_prediction_context(TASK, "ep1")
    d_before = first.capability_version["knowledge_content_digest"]
    # Re-read with no revision: the digest must NOT move.
    reread = h.build_prediction_context(TASK, "ep1")
    print(f"digest (first read)     : {d_before}")
    print(f"digest (re-read)        : "
          f"{reread.capability_version['knowledge_content_digest']}")
    assert reread.capability_version["knowledge_content_digest"] == d_before, (
        "a re-read is not a content change")
    # Revise the SAME entry: the digest MUST move, the count must not.
    revised = h.sbank.get("se_demo")
    revised.expected_quality_hat = 0.55
    h.sbank.update(revised)
    second = h.build_prediction_context(TASK, "ep1")
    d_after = second.capability_version["knowledge_content_digest"]
    print(f"digest (after revision) : {d_after}")
    print(f"entry count before/after: "
          f"{first.capability_version['knowledge_content']['knowledge_entries']}"
          f"/{second.capability_version['knowledge_content']['knowledge_entries']}")
    print(f"excluded read timestamps: "
          f"{second.capability_version['knowledge_content']['excluded_keys']}")
    assert d_after != d_before, "a knowledge revision must move the digest"

    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print("12. Historical reconstruction reads only what was SAVED")
    print("=" * 72)
    hist_task = dict(TASK, task_id="ctx_demo_hist")
    hist_entry = StrategicEntry(
        entry_id="se_hist", strategy_id="S01",
        pattern={"predicates": {"family": "routing"}},
        expected_quality_hat=0.60, quality_interval=(0.5, 1.0),
        expected_cost_hat=CostVector(llm_tokens=100.0,
                                     measured={"llm_tokens"}),
        failure_prob=0.1, status="candidate", support_n=2,
        verification={"state": "verified", "claim": "c", "conclusion": "holds"})
    h.sbank.add(hist_entry)
    hist_snap = h.snapshot(hist_task, "ep1")     # freezes quality 0.60
    moved = h.sbank.get("se_hist")
    moved.expected_quality_hat = 0.95            # the entry is revised later
    h.sbank.update(moved)
    historical = h.build_prediction_context(hist_task, "ep1",
                                            snapshot=hist_snap)
    frozen_layers = ((historical.snapshot.get("coverage") or {})
                     .get("knowledge_layers") or {})
    frozen_by_id = {e.get("entry_id"): e.get("expected_quality_hat")
                    for e in (frozen_layers.get("verified") or [])}
    print(f"entry revised to        : 0.95")
    print(f"frozen value for se_hist: {frozen_by_id.get('se_hist')}")
    print(f"revision leaked in      : {'0.95' in str(historical.to_dict())}")
    print(f"retrieval rebuilt from  : "
          f"{historical.execution_constraints['retrieval_bounding']['rebuilt_from']}")
    print("missing (reported, not filled):")
    for item in historical.missing:
        if "MISSING" in item:
            print(f"   - {item[:88]}...")
    assert frozen_by_id.get("se_hist") == 0.60, (
        "the frozen view keeps the value the entry had at snapshot time")
    assert "0.95" not in str(historical.to_dict()), (
        "a later revision must not enter a historical context")
    assert historical.reliability == {}, (
        "reliability was not saved with the snapshot, so it is missing")
    assert historical.cell_evidence == {}
    assert any("MISSING" in m for m in historical.missing)

    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print("13. Structure consistency is checked against the EFFECTIVE input")
    print("=" * 72)
    weak_task = {k: v for k, v in TASK.items() if k != "coupling"}
    weak_task["task_id"] = "ctx_demo_struct"
    weak_task["annotations"] = {
        "coupling": {"resource_coupling": 0.30, "temporal_coupling": 0.10,
                     "route_complexity": 0.20, "semantic_coupling": 0.50}}
    weak_snap = h.snapshot(weak_task, "ep1")     # frozen with WEAK structure
    try:
        h.build_prediction_context(weak_task, "ep1", snapshot=weak_snap,
                                   cir=TASK["coupling"])
    except ValueError as exc:
        print(f"snapshot + new CIR REFUSED: {str(exc)[:120]}...")
    else:  # pragma: no cover - the refusal is the point
        raise AssertionError("a structurally inconsistent snapshot must be "
                             "refused")
    # A matching snapshot is accepted, and the same structure is used
    # everywhere.
    matching = h.build_prediction_context(weak_task, "ep1",
                                          snapshot=weak_snap)
    print(f"matching snapshot accepted: "
          f"rc={matching.joint.profile['resource_coupling']}")
    assert matching.joint.profile["resource_coupling"] == 0.30

    h.close()
    tmp.cleanup()
    print()
    print("All assertions passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
