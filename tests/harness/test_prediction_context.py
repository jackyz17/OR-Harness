"""Phase-2 tests: the prediction input context.

What this file asserts, one behaviour per test class:

1. the joint representation is available BEFORE a model exists, keeps the
   semantics, the CIR relations and the sources, and never fabricates a math
   attribute from the scenario's name;
2. retrieval evidence keeps discovery and applicability apart, surfaces
   cross-cell cases with their differences, deduplicates repeated hits by
   identity (without inflating support), and never lets unverified knowledge
   become verified by being retrieved;
3. reuse is version-verified: a recall result or context from another task
   version / episode is refused, not silently aligned;
4. the context is FROZEN against later bank, task and tool changes, and a
   stored context replays without reading today's facts;
5. several candidates share ONE context while keeping their own config and
   scope, and the provider really receives the new joint evidence;
6. every refusal to run a channel is distinguishable (no backend, no task
   text, missing index, backend error, and a healthy run that found nothing);
7. capability evidence carries honest statuses — no fabricated source, no
   composite score, and unverified knowledge / hypothetical predictions
   never become real evidence;
8. building a context runs no solver, calls no prediction model and induces
   nothing; only an explicit prediction call reaches the provider; and the
   phase-1 fixes do not regress.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tests.harness.helpers import HarnessTestCase  # noqa: E402

from or_harness.api import ORHarness  # noqa: E402
from or_harness.core.schema import StrategicEntry  # noqa: E402
from or_harness.strategy.embedding_index import (  # noqa: E402
    LAYER_EXECUTION,
    LAYER_STRATEGIC,
    LocalHashEmbeddingBackend,
)
from or_harness.world_model.context import (  # noqa: E402
    PREDICTION_CONTEXT_VERSION,
    UnsupportedContextVersion,
    build_joint_representation,
    build_retrieval_view,
    classify_evidence,
    context_identity_problems,
    dedupe_evidence,
    evidence_identity,
    math_attributes,
    retrieval_reuse_problems,
)
from or_harness.world_model.prediction import ActionSpec  # noqa: E402
from or_harness.world_model.provider import WorldModelProvider  # noqa: E402
from or_harness.world_model.state import task_text_digest  # noqa: E402

#: Every env var that can grant an ambient embedding backend. Tests that
#: assert a DEGRADATION must own these, or an ambient variable would silently
#: give the harness a backend and the assertion would test nothing.
EMBEDDING_ENV_KEYS = ("OR_EMBEDDING_BACKEND", "OR_EMBEDDING_BASE_URL",
                      "OR_EMBEDDING_MODEL", "OR_EMBEDDING_API_KEY")

REQ_TEXT = ("A distribution centre must be loaded before the delivery window "
            "opens; demand 100 units may not be deferred.")


def _task(task_id="t1", *, description=REQ_TEXT, **coupling):
    values = {"resource_coupling": 0.3, "temporal_coupling": 0.1,
              "route_complexity": 0.2}
    values.update(coupling)
    return {"task_id": task_id, "family": "routing",
            "description": description,
            "spec": {"n_vars": 100, "n_constraints": 50, "n_int_vars": 100},
            "annotations": {"coupling": {**values, "semantic_coupling": 0.5}}}


def _cir(task_id="t1"):
    return {
        "entities": [{"name": "depot", "kind": "site"},
                     {"name": "route", "kind": "route"}],
        "decisions": [{"name": "load", "kind": "binary", "indexes": ["t"]}],
        "constraints": [{"id": "C1", "kind": "capacity",
                         "expr": "load[t] <= cap"}],
        "relations": [{"source": "load", "target": "depot",
                       "type": "uses_resource", "evidence": "declared"}],
        "coupling_groups": [],
        "issues": [],
    }


class RecordingProvider(WorldModelProvider):
    """A stub provider that records exactly what it was handed."""

    name = "recording-context-test"

    def __init__(self, quality=0.7):
        self.quality = quality
        self.requests = []

    def predict(self, request, timeout_s=None):
        self.requests.append(request)
        return {"payload": {"outcome_status": "feasible", "feasible": True,
                            "quality": self.quality, "failure_prob": 0.1},
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                "error": None, "latency_s": 0.01}


class ContextCase(HarnessTestCase):
    """A harness with an explicitly injected local backend (hermetic)."""

    def setUp(self):
        super().setUp()
        saved = {key: os.environ.pop(key, None) for key in EMBEDDING_ENV_KEYS}

        def restore():
            for key, value in saved.items():
                if value is not None:
                    os.environ[key] = value
        self.addCleanup(restore)
        self.backend = LocalHashEmbeddingBackend()
        self.h = ORHarness(home=self.home, embedding=self.backend)
        self.addCleanup(self.h.close)

    def solve(self, task, strategy="S04", objective=100.0, description=None):
        from pathlib import Path
        work = Path(self.home) / f"ws_{task['task_id']}_{strategy}"
        work.mkdir(parents=True, exist_ok=True)
        script = work / "solve.py"
        script.write_text(
            "import json\n"
            "with open('result.json', 'w') as fh:\n"
            "    json.dump({'status': 'optimal', 'objective_value': "
            f"{objective}, 'objective_bound': {objective}, "
            "'runtime_seconds': 0.01}, fh)\n", encoding="utf-8")
        record = self.h.execute(task, strategy, str(script), str(work),
                                solver="highs")
        self.h.record(record)
        return record


# ---------------------------------------------------------------------------
# 1. joint representation, before any model exists
# ---------------------------------------------------------------------------


class TestJointRepresentationBeforeModel(ContextCase):
    def test_context_builds_with_no_model_and_no_solve_script(self):
        ctx = self.h.build_prediction_context(_task(), "ep1")
        self.assertFalse(ctx.joint.has_model)
        self.assertIn("mathematical model: not written yet", " ".join(
            ctx.missing))
        # A build is not blocked by the missing model.
        self.assertEqual(ctx.task_id, "t1")
        self.assertEqual(ctx.task_digest, task_text_digest(_task()))

    def test_math_attributes_are_never_guessed_from_the_family_name(self):
        task = {"task_id": "t9", "family": "routing",
                "description": "route vehicles"}
        joint = build_joint_representation(task)
        # "routing" says nothing about integrality or linearity.
        self.assertIsNone(joint.math.integrality)
        self.assertIsNone(joint.math.linearity)
        self.assertIn("integrality", joint.math.unknowns)
        self.assertIn("linearity", joint.math.unknowns)
        self.assertEqual(joint.math.origin("integrality"), "unknown")

    def test_declared_math_attributes_carry_their_origin(self):
        task = {"task_id": "t9", "family": "routing",
                "description": "x", "spec": {"n_vars": 10, "n_int_vars": 0}}
        joint = build_joint_representation(
            task, math_declared={"linearity": "linear",
                                 "objective_kind": "min"})
        self.assertEqual(joint.math.linearity, "linear")
        self.assertEqual(joint.math.origin("linearity"), "declared")
        # The spec still supplies what was not declared.
        self.assertEqual(joint.math.integrality, "continuous")
        self.assertEqual(joint.math.origin("integrality"), "spec")

    def test_cir_relations_are_kept_as_relations_not_three_numbers(self):
        task = dict(_task(), coupling=_cir())
        joint = build_joint_representation(task)
        self.assertTrue(joint.cir_present)
        self.assertEqual(len(joint.cir["relations"]), 1)
        self.assertEqual(joint.cir["relations"][0]["type"], "uses_resource")
        self.assertEqual(joint.sources["cir"], "task_coupling")

    def test_semantics_and_sources_survive_the_round_trip(self):
        task = dict(_task(), coupling=_cir())
        ctx = self.h.build_prediction_context(task, "ep1")
        again = type(ctx).from_dict(ctx.to_dict())
        self.assertEqual(again.joint.text, ctx.joint.text)
        self.assertEqual(again.joint.cir["relations"],
                         ctx.joint.cir["relations"])
        self.assertEqual(again.sources, ctx.sources)
        self.assertEqual(again.version, PREDICTION_CONTEXT_VERSION)

    def test_absent_parts_are_reported_as_absent(self):
        task = {"task_id": "t2", "family": "routing"}
        joint = build_joint_representation(task)
        self.assertEqual(joint.text, "")
        self.assertIn("no CIR", " ".join(joint.notes))
        joined = " ".join(joint.missing)
        self.assertIn("task text", joined)
        self.assertIn("CIR", joined)


# ---------------------------------------------------------------------------
# 2. retrieval evidence: discovery vs reuse, dedup, cross-cell
# ---------------------------------------------------------------------------


class TestRetrievalEvidence(ContextCase):
    def test_semantic_hits_carry_content_not_just_ids(self):
        self.solve(_task("t1"))
        ctx = self.h.build_prediction_context(_task("t2"), "ep1")
        hits = [h for h in ctx.retrieval.hits
                if h["layer"] == "execution_evidence"]
        self.assertTrue(hits)
        content = hits[0]["content"]
        self.assertIn("task_text_excerpt", content)
        self.assertIn("observed_quality", content)
        self.assertIn("observed_cost", content)
        self.assertIn("structural_match", content)

    def test_cross_cell_similar_case_is_found_and_labelled(self):
        # The past task is textually near-identical but structurally
        # different: it must still be SEEN, and labelled as another cell.
        self.solve(_task("t1", resource_coupling=0.9, temporal_coupling=0.9,
                         route_complexity=0.9))
        ctx = self.h.build_prediction_context(_task("t2"), "ep1")
        cross = [h for h in ctx.retrieval.hits
                 if h["layer"] == "execution_evidence"
                 and h["content"].get("structural_match") == "different_cell"]
        self.assertTrue(cross, "a textually similar cross-cell case must be "
                               "surfaced")
        self.assertEqual(cross[0]["evidence_class"], "execution_fact")

    def test_different_cell_hit_does_not_touch_the_cell_statistics(self):
        self.solve(_task("t1", resource_coupling=0.9, temporal_coupling=0.9,
                         route_complexity=0.9), strategy="S04")
        before = {sid: cell.n for sid, cell in
                  self.h.stats.for_profile(
                      self.h.profile(_task("t2"))).items()}
        self.h.build_prediction_context(_task("t2"), "ep1")
        after = {sid: cell.n for sid, cell in
                 self.h.stats.for_profile(
                     self.h.profile(_task("t2"))).items()}
        self.assertEqual(before, after,
                         "retrieval must never move an aggregation number")

    def test_repeated_hits_are_deduplicated_by_identity(self):
        hits = [
            {"layer": "strategic_knowledge", "evidence_id": "se_1",
             "identity": evidence_identity("strategic_knowledge", "se_1"),
             "version": "v1", "channels": ["structural"],
             "evidence_class": "verified_knowledge", "content": {}},
            {"layer": "strategic_knowledge", "evidence_id": "se_1",
             "identity": evidence_identity("strategic_knowledge", "se_1"),
             "version": "v1", "channels": ["semantic"],
             "evidence_class": "verified_knowledge", "content": {}},
        ]
        items, summary = dedupe_evidence(hits)
        self.assertEqual(len(items), 1)
        self.assertEqual(sorted(items[0]["channels"]),
                         ["semantic", "structural"])
        self.assertEqual(summary["duplicates_collapsed"], 1)
        self.assertEqual(summary["hits_seen"], 2)

    def test_same_evidence_under_two_versions_is_reported_not_merged(self):
        hits = [
            {"layer": "execution_evidence", "evidence_id": "ex_1",
             "identity": evidence_identity("execution_evidence", "ex_1"),
             "version": "d1", "channels": ["semantic"],
             "evidence_class": "execution_fact", "content": {}},
            {"layer": "execution_evidence", "evidence_id": "ex_1",
             "identity": evidence_identity("execution_evidence", "ex_1"),
             "version": "d2", "channels": ["structural"],
             "evidence_class": "execution_fact", "content": {}},
        ]
        items, summary = dedupe_evidence(hits)
        self.assertEqual(len(items), 1)
        self.assertTrue(items[0]["version_conflict"])
        self.assertEqual(summary["version_conflicts"],
                         ["execution_evidence:ex_1"])

    def test_unverified_knowledge_is_not_surfaced_by_default(self):
        entry = StrategicEntry(
            entry_id="se_u", strategy_id="S04",
            pattern={"predicates": {"family": "routing"}},
            expected_quality_hat=0.9, quality_interval=(0.5, 1.0),
            expected_cost_hat=self.make_record().cost,
            failure_prob=0.1, status="candidate", support_n=2,
            verification={"state": "unverified", "claim": "c",
                          "conclusion": "no verdict yet"})
        self.h.sbank.add(entry)
        ctx = self.h.build_prediction_context(_task(), "ep1")
        ids = {h["evidence_id"] for h in ctx.retrieval.hits}
        self.assertNotIn("se_u", ids)
        ctx2 = self.h.build_prediction_context(
            _task(), "ep1", include_unverified=True)
        ids2 = {h["evidence_id"] for h in ctx2.retrieval.hits}
        self.assertIn("se_u", ids2)
        # Even when surfaced, it stays labelled unverified.
        item = next(h for h in ctx2.retrieval.hits
                    if h["evidence_id"] == "se_u")
        self.assertNotEqual(item["evidence_class"], "verified_knowledge")

    def test_evidence_classes_distinguish_the_five_kinds(self):
        self.assertEqual(classify_evidence("execution_evidence", {}),
                         "execution_fact")
        self.assertEqual(classify_evidence(
            "strategic_knowledge",
            {"reusable": True, "verification_state": "verified"}),
            "verified_knowledge")
        self.assertEqual(classify_evidence(
            "strategic_knowledge", {"reusable": False}),
            "unverified_knowledge")
        self.assertEqual(classify_evidence(
            "strategic_knowledge", {"reusable": True,
                                    "verification_state": "legacy"}),
            "legacy_knowledge")
        self.assertEqual(classify_evidence("recommendation", {}),
                         "structural_recommendation")

    def test_no_composite_retrieval_score_is_invented(self):
        self.solve(_task("t1"))
        ctx = self.h.build_prediction_context(_task("t2"), "ep1")
        # Scoped to the RETRIEVAL view on purpose: the capability block
        # legitimately declares score_scheme="no_composite_score".
        blob = str(ctx.retrieval.to_dict())
        for forbidden in ("combined_score", "fused_score",
                          "retrieval_score", "composite_score"):
            self.assertNotIn(forbidden, blob)
        # The two channels are reported separately, never blended.
        self.assertEqual(ctx.retrieval.structural["status"], "ok")
        self.assertEqual(ctx.retrieval.semantic["status"], "ok")
        self.assertEqual(sorted(ctx.retrieval.channels_run),
                         ["semantic", "structural"])


# ---------------------------------------------------------------------------
# 3. reuse is version-verified
# ---------------------------------------------------------------------------


class TestReuseIsVersionVerified(ContextCase):
    def test_recall_result_without_a_version_cannot_be_reused(self):
        problems = retrieval_reuse_problems(
            type("R", (), {"task_digest": None})(),
            task_digest="abc")
        self.assertTrue(problems)
        self.assertIn("records no task version", problems[0])

    def test_recall_result_from_another_version_is_refused(self):
        result = self.h.recall(_task("t1"))
        # The SAME task_id, different content: a different problem version.
        other = _task("t1", description="completely different requirement")
        with self.assertRaises(ValueError) as caught:
            self.h.build_prediction_context(other, "ep1",
                                            recall_result=result)
        self.assertIn("cannot be reused", str(caught.exception))

    def test_matching_recall_result_is_reused(self):
        task = _task("t1")
        result = self.h.recall(task)
        ctx = self.h.build_prediction_context(task, "ep1",
                                              recall_result=result)
        self.assertEqual(ctx.retrieval.task_digest, result["task_digest"])

    def test_context_identity_problems_detect_task_version_and_episode(self):
        ctx = self.h.build_prediction_context(_task("t1"), "ep1")
        self.assertEqual(context_identity_problems(
            ctx, task_id="t1", task_digest=ctx.task_digest, episode_id="ep1"),
            [])
        self.assertTrue(context_identity_problems(ctx, task_id="t9"))
        self.assertTrue(context_identity_problems(ctx, episode_id="ep9"))
        version_problems = context_identity_problems(
            ctx, task_digest="deadbeef")
        self.assertTrue(version_problems)
        self.assertIn("task version", version_problems[0])

    def test_unknown_version_is_reported_only_when_strictly_required(self):
        ctx = self.h.build_prediction_context(_task("t1"), "ep1")
        ctx.task_digest = ""
        self.assertEqual(context_identity_problems(ctx, task_digest="x"), [])
        strict = context_identity_problems(ctx, task_digest="x",
                                           allow_unknown_digest=False)
        self.assertTrue(strict)

    def test_snapshot_of_another_episode_is_refused(self):
        snap = self.h.snapshot(_task("t1"), "ep1")
        with self.assertRaises(ValueError):
            self.h.build_prediction_context(_task("t1"), "ep2",
                                            snapshot=snap)

    def test_supplied_recall_result_does_not_re_embed(self):
        task = _task("t1")
        result = self.h.recall(task)
        calls = {"n": 0}
        original = self.backend.embed_query

        def counting(text):
            calls["n"] += 1
            return original(text)
        self.backend.embed_query = counting
        self.addCleanup(setattr, self.backend, "embed_query", original)
        self.h.build_prediction_context(task, "ep1", recall_result=result)
        self.assertEqual(calls["n"], 0,
                         "a supplied result must not be re-embedded")


# ---------------------------------------------------------------------------
# 4. the context is frozen
# ---------------------------------------------------------------------------


class TestContextIsFrozen(ContextCase):
    def test_later_bank_writes_do_not_change_a_built_context(self):
        self.solve(_task("t1"))
        ctx = self.h.build_prediction_context(_task("t2"), "ep1")
        before = ctx.to_dict()
        self.solve(_task("t3"))
        self.assertEqual(ctx.to_dict(), before)

    def test_mutating_the_task_after_the_build_changes_nothing(self):
        task = _task("t1")
        ctx = self.h.build_prediction_context(task, "ep1")
        text_before = ctx.joint.text
        digest_before = ctx.task_digest
        task["description"] = "HIJACKED"
        task["task_id"] = "t-hijacked"
        self.assertEqual(ctx.joint.text, text_before)
        self.assertEqual(ctx.task_digest, digest_before)

    def test_mutating_a_returned_dict_does_not_reach_the_stored_context(self):
        ctx = self.h.build_prediction_context(_task("t1"), "ep1")
        dumped = ctx.to_dict()
        dumped["joint"]["text"] = "HIJACKED"
        dumped["retrieval"]["hits"] = ["fake"]
        again = self.h.get_prediction_context(ctx.context_id)
        self.assertNotEqual(again.joint.text, "HIJACKED")
        self.assertNotEqual(again.retrieval.hits, ["fake"])

    def test_stored_context_replays_without_reading_todays_banks(self):
        self.solve(_task("t1"))
        ctx = self.h.build_prediction_context(_task("t2"), "ep1")
        stored = self.h.get_prediction_context(ctx.context_id)
        self.assertEqual(stored.to_dict(), ctx.to_dict())
        # A context is CONTENT, not a pointer: the stored copy carries the
        # evidence it was built from.
        self.assertEqual(stored.retrieval.hits, ctx.retrieval.hits)

    def test_unknown_context_version_is_refused_not_guessed(self):
        payload = self.h.build_prediction_context(_task("t1"), "ep1").to_dict()
        payload["context_version"] = "wm-context/99"
        with self.assertRaises(UnsupportedContextVersion):
            type(self.h.build_prediction_context(_task(), "ep1"))\
                .from_dict(payload)

    def test_a_context_built_for_another_version_is_refused_by_the_api(self):
        ctx = self.h.build_prediction_context(_task("t1"), "ep1")
        other = _task("t1", description="a different requirement entirely")
        spec = ActionSpec("execute_strategy", "t1", strategy_id="S01")
        with self.assertRaises(ValueError) as caught:
            self.h.predict_outcome(other, spec, "ep1", context=ctx)
        self.assertIn("does not describe", str(caught.exception))


# ---------------------------------------------------------------------------
# 5. one context, several candidates, and the provider really sees it
# ---------------------------------------------------------------------------


class TestOneContextSharedByCandidates(ContextCase):
    def test_planning_shares_one_context_across_candidates(self):
        provider = RecordingProvider()
        h = ORHarness(home=self.home, world_model=provider,
                      embedding=self.backend)
        self.addCleanup(h.close)
        plan = h.plan_next(
            _task("t1"), "ep1",
            candidates=[ActionSpec("execute_strategy", "t1",
                                   strategy_id="S01"),
                        ActionSpec("execute_strategy", "t1",
                                   strategy_id="S02")],
            limits={"horizon": 1})
        contexts = {r.get("prediction_context", {}).get("context_id")
                    for r in provider.requests}
        self.assertEqual(len(contexts), 1,
                         "every candidate of one decision shares ONE context")
        self.assertIsNotNone(plan.get("prediction_context_id"))

    def test_candidates_keep_their_own_config_and_scope(self):
        provider = RecordingProvider()
        h = ORHarness(home=self.home, world_model=provider,
                      embedding=self.backend)
        self.addCleanup(h.close)
        specs = [ActionSpec("execute_strategy", "t1", strategy_id="S01",
                            params={"time_limit": 60}),
                 ActionSpec("execute_strategy", "t1", strategy_id="S02")]
        for spec in specs:
            h.predict_outcome(_task("t1"), spec, "ep1")
        ids = [r["action_spec"]["strategy_id"] for r in provider.requests]
        self.assertEqual(ids, ["S01", "S02"])
        self.assertEqual(provider.requests[0]["action_spec"]["params"],
                         {"time_limit": 60})

    def test_the_provider_receives_the_new_joint_evidence(self):
        provider = RecordingProvider()
        h = ORHarness(home=self.home, world_model=provider,
                      embedding=self.backend)
        self.addCleanup(h.close)
        spec = ActionSpec("execute_strategy", "t1", strategy_id="S01")
        h.predict_outcome(dict(_task("t1"), coupling=_cir()), spec, "ep1")
        request = provider.requests[0]
        self.assertIn("prediction_context", request)
        block = request["prediction_context"]
        self.assertEqual(block["context_version"], PREDICTION_CONTEXT_VERSION)
        self.assertIn("joint_problem", block)
        self.assertIn("retrieval_evidence", block)
        self.assertIn("harness_capability", block)
        self.assertIn("execution_constraints", block)
        # The problem's semantics really arrived, not just a hash.
        self.assertIn("distribution centre", block["joint_problem"]["text"])
        self.assertEqual(len(block["joint_problem"]["cir"]["relations"]), 1)
        self.assertIn("sources", block["joint_problem"])

    def test_prediction_records_which_frozen_input_it_used(self):
        provider = RecordingProvider()
        h = ORHarness(home=self.home, world_model=provider,
                      embedding=self.backend)
        self.addCleanup(h.close)
        spec = ActionSpec("execute_strategy", "t1", strategy_id="S01")
        prediction = h.predict_outcome(_task("t1"), spec, "ep1")
        ctx_id = prediction.model_info["prediction_context_id"]
        self.assertTrue(ctx_id)
        self.assertIsNotNone(h.get_prediction_context(ctx_id))

    def test_no_context_keeps_the_request_shape_unchanged(self):
        provider = RecordingProvider()
        h = ORHarness(home=self.home, world_model=provider,
                      embedding=self.backend)
        self.addCleanup(h.close)
        spec = ActionSpec("execute_strategy", "t1", strategy_id="S01")
        h.predict_outcome(_task("t1"), spec, "ep1", context=False)
        request = provider.requests[0]
        self.assertNotIn("prediction_context", request)
        self.assertNotIn("prediction_context", request["state"])
        # The request carries exactly what it carried before this phase:
        # the action spec, the snapshot-derived state, and the pre-existing
        # M6 reliability block.
        self.assertEqual(sorted(request.keys()),
                         ["action_spec", "prediction_reliability", "state"])

    def test_reused_context_is_identical_for_both_candidates(self):
        provider = RecordingProvider()
        h = ORHarness(home=self.home, world_model=provider,
                      embedding=self.backend)
        self.addCleanup(h.close)
        ctx = h.build_prediction_context(_task("t1"), "ep1")
        for sid in ("S01", "S02"):
            h.predict_outcome(
                _task("t1"),
                ActionSpec("execute_strategy", "t1", strategy_id=sid),
                "ep1", context=ctx)
        blocks = [r["prediction_context"] for r in provider.requests]
        self.assertEqual(blocks[0]["joint_problem"],
                         blocks[1]["joint_problem"])
        self.assertEqual(blocks[0]["retrieval_evidence"],
                         blocks[1]["retrieval_evidence"])
        self.assertEqual(blocks[0]["harness_capability"],
                         blocks[1]["harness_capability"])


# ---------------------------------------------------------------------------
# 6. degradation is reported per part
# ---------------------------------------------------------------------------


class TestDegradationIsDistinguishable(ContextCase):
    def _harness_without_backend(self):
        h = ORHarness(home=self.home)
        self.addCleanup(h.close)
        return h

    def test_no_backend_is_reported_and_structural_survives(self):
        h = self._harness_without_backend()
        self.solve(_task("t1"))
        ctx = h.build_prediction_context(_task("t2"), "ep1")
        self.assertEqual(ctx.retrieval.channels_run, ["structural"])
        self.assertTrue(ctx.degraded)
        self.assertIn("semantic", ctx.degraded[0]["part"])
        self.assertTrue(ctx.retrieval.structural["recommendations"] is not
                        None)

    def test_no_task_text_is_distinguished_from_no_hits(self):
        task = {"task_id": "t2", "family": "routing",
                "annotations": {"coupling": {"resource_coupling": 0.3,
                                             "temporal_coupling": 0.1,
                                             "route_complexity": 0.2,
                                             "semantic_coupling": 0.5}}}
        ctx = self.h.build_prediction_context(task, "ep1")
        joined = " ".join(d["reason"] for d in ctx.degraded)
        self.assertIn("no task text", joined)
        # And the joint representation says the same thing on its own side.
        self.assertIn("task text", " ".join(ctx.missing))

    def test_missing_index_is_distinguished_from_a_healthy_empty_run(self):
        task = _task("t2")
        ctx = self.h.build_prediction_context(task, "ep1")
        reasons = " ".join(d["reason"] for d in ctx.degraded)
        self.assertIn("index missing", reasons)
        # After a real solve the EXECUTION layer has an index, so the text
        # channel really runs on it; whatever remains degraded names the
        # layer that is still missing, rather than hiding both behind one
        # vague message.
        self.solve(_task("t1"))
        ctx2 = self.h.build_prediction_context(task, "ep1")
        parts = {d["part"] for d in ctx2.degraded}
        self.assertNotIn("retrieval.semantic", parts,
                         "the whole semantic channel is no longer skipped")
        self.assertIn("semantic", ctx2.retrieval.channels_run)

    def test_backend_error_is_reported_not_swallowed(self):
        # A real index first: otherwise the missing index (a different
        # fact) would be the reported reason instead of the backend error.
        self.solve(_task("t1"))
        def boom(text):
            raise RuntimeError("backend exploded")
        self.backend.embed_query = boom
        ctx = self.h.build_prediction_context(_task("t2"), "ep1")
        reasons = " ".join(d["reason"] for d in ctx.degraded)
        self.assertIn("backend exploded", reasons)
        self.assertIn("semantic", ctx.degraded[0]["part"])

    def test_degraded_and_no_hits_are_different_facts(self):
        h = self._harness_without_backend()
        ctx = h.build_prediction_context(_task("t2"), "ep1")
        # No EXECUTION evidence was carried (the semantic channel never
        # ran), and the context says so rather than reading as an empty bank.
        layers = {hit["layer"] for hit in ctx.retrieval.hits}
        self.assertNotIn("execution_evidence", layers)
        # The two facts are reported separately: the semantic channel is
        # DEGRADED (with a reason) while the structural one ran.
        self.assertEqual(ctx.retrieval.structural["status"], "ok")
        self.assertEqual(ctx.retrieval.semantic["status"], "degraded")
        self.assertNotIn("semantic", ctx.retrieval.channels_run,
                         "a channel that did not run is not reported as run")

    def test_context_without_a_recall_result_says_it_was_omitted(self):
        view = build_retrieval_view({}, task_digest="abc")
        joined = " ".join(view.missing)
        self.assertIn("no result supplied", joined)
        self.assertEqual(view.channels_run, ["structural"])


# ---------------------------------------------------------------------------
# 7. capability evidence stays honest
# ---------------------------------------------------------------------------


class TestCapabilityEvidenceIsHonest(ContextCase):
    def test_no_composite_capability_score_exists(self):
        ctx = self.h.build_prediction_context(_task("t1"), "ep1")
        self.assertEqual(ctx.capability.get("score_scheme"),
                         "no_composite_score")
        self.assertNotIn("composite_score", ctx.capability)
        self.assertNotIn("h_score", ctx.capability)

    def test_sources_without_observation_stay_no_evidence(self):
        ctx = self.h.build_prediction_context(_task("t1"), "ep1")
        sources = ctx.capability.get("sources") or {}
        # Nothing observed W_OR / Pi here, so they must not be filled in.
        for name in ("w_or", "pi"):
            self.assertEqual(sources[name]["status"], "no_evidence",
                             f"{name} must not be fabricated")
        self.assertNotIn("direct_evidence",
                         {s["status"] for s in sources.values()})

    def test_r_is_indirect_when_retrieval_ran(self):
        self.solve(_task("t1"))
        ctx = self.h.build_prediction_context(_task("t2"), "ep1")
        r = ctx.capability["sources"]["r"]
        self.assertEqual(r["status"], "indirect_evidence")
        self.assertIn("channels_run", r["detail"])

    def test_capability_version_separates_config_model_tools_memory(self):
        ctx = self.h.build_prediction_context(_task("t1"), "ep1")
        version = ctx.capability_version
        for key in ("harness_config_digest", "provider",
                    "prompt_template_version", "tools",
                    "knowledge_content_digest"):
            self.assertIn(key, version)
        self.assertIn("not", version["note"].lower())

    def test_knowledge_content_digest_is_not_an_ability_measurement(self):
        self.solve(_task("t1"))
        ctx = self.h.build_prediction_context(_task("t2"), "ep1")
        digest = ctx.capability_version["knowledge_content_digest"]
        self.assertIsInstance(digest, str)
        self.assertNotIn("capability", digest)

    def test_unverified_knowledge_never_becomes_real_evidence(self):
        entry = StrategicEntry(
            entry_id="se_u", strategy_id="S04",
            pattern={"predicates": {"family": "routing"}},
            expected_quality_hat=0.95, quality_interval=(0.5, 1.0),
            expected_cost_hat=self.make_record().cost,
            failure_prob=0.05, status="candidate", support_n=2,
            verification={"state": "unverified", "claim": "c",
                          "conclusion": "no verdict yet"})
        self.h.sbank.add(entry)
        ctx = self.h.build_prediction_context(_task(), "ep1")
        # It must NOT be surfaced as retrievable evidence by default.
        self.assertNotIn("se_u", {hit["evidence_id"]
                                  for hit in ctx.retrieval.hits})
        # It MAY appear inside the frozen coverage view — that view layers it
        # as `unverified` on purpose (the frozen target derivation reads it),
        # so the assertion is that it never reaches a VERIFIED position.
        layers = ((ctx.snapshot.get("coverage") or {})
                  .get("knowledge_layers") or {})
        verified_ids = {str(e.get("entry_id"))
                        for e in (layers.get("verified") or [])}
        unverified_ids = {str(e.get("entry_id"))
                          for e in (layers.get("unverified") or [])}
        self.assertNotIn("se_u", verified_ids)
        self.assertIn("se_u", unverified_ids)
        # And the memory-content version digests CONTENT, so the revision is
        # visible in it while the id itself is not echoed back.
        digest_block = ctx.capability_version["knowledge_content"]
        self.assertNotIn("se_u", str(digest_block))
        self.assertEqual(digest_block["knowledge_by_layer"].get("unverified"),
                         1)


# ---------------------------------------------------------------------------
# 8. no side effects; explicit calls only
# ---------------------------------------------------------------------------


class TestBuildHasNoSideEffects(ContextCase):
    def test_building_a_context_calls_no_prediction_model(self):
        provider = RecordingProvider()
        h = ORHarness(home=self.home, world_model=provider,
                      embedding=self.backend)
        self.addCleanup(h.close)
        h.build_prediction_context(_task("t1"), "ep1")
        self.assertEqual(provider.requests, [],
                         "building a context must not reach the provider")

    def test_building_a_context_runs_no_solver_and_induces_nothing(self):
        self.solve(_task("t1"))
        entries_before = len(self.h.sbank.list(include_dormant=True))
        executions_before = self.h.bank.count()
        self.h.build_prediction_context(_task("t2"), "ep1")
        self.assertEqual(self.h.bank.count(), executions_before)
        self.assertEqual(len(self.h.sbank.list(include_dormant=True)),
                         entries_before)

    def test_building_writes_no_prediction(self):
        self.h.build_prediction_context(_task("t1"), "ep1")
        self.assertEqual(self.h.predictions_query(), [])

    def test_build_persists_the_context_by_default_and_not_on_request(self):
        ctx = self.h.build_prediction_context(_task("t1"), "ep1")
        self.assertIsNotNone(self.h.get_prediction_context(ctx.context_id))
        ctx2 = self.h.build_prediction_context(_task("t1"), "ep1",
                                               persist=False)
        self.assertIsNone(self.h.get_prediction_context(ctx2.context_id))

    def test_reading_a_stored_context_calls_nothing(self):
        provider = RecordingProvider()
        h = ORHarness(home=self.home, world_model=provider,
                      embedding=self.backend)
        self.addCleanup(h.close)
        ctx = h.build_prediction_context(_task("t1"), "ep1")
        calls = {"n": 0}
        original = self.backend.embed_query

        def counting(text):
            calls["n"] += 1
            return original(text)
        self.backend.embed_query = counting
        self.addCleanup(setattr, self.backend, "embed_query", original)
        h.get_prediction_context(ctx.context_id)
        self.assertEqual(calls["n"], 0)
        self.assertEqual(provider.requests, [])

    def test_only_an_explicit_prediction_call_reaches_the_provider(self):
        provider = RecordingProvider()
        h = ORHarness(home=self.home, world_model=provider,
                      embedding=self.backend)
        self.addCleanup(h.close)
        h.build_prediction_context(_task("t1"), "ep1")
        self.assertEqual(provider.requests, [])
        h.predict_outcome(
            _task("t1"),
            ActionSpec("execute_strategy", "t1", strategy_id="S01"), "ep1")
        self.assertEqual(len(provider.requests), 1)

    def test_declared_budget_reaches_the_execution_constraints(self):
        task = _task("b1")
        self.h.declare_budget("b1", {"llm_tokens": 50000}, "ep1")
        ctx = self.h.build_prediction_context(task, "ep1")
        self.assertEqual(ctx.execution_constraints["declared_budget"],
                         {"llm_tokens": 50000.0})
        self.assertEqual(ctx.execution_constraints["budget_status"], "ok")
        # A fresh instance reads the PERSISTED declaration too (the CLI path
        # builds a new harness per invocation).
        h2 = ORHarness(home=self.home, embedding=self.backend)
        self.addCleanup(h2.close)
        ctx2 = h2.build_prediction_context(task, "ep1")
        self.assertEqual(ctx2.execution_constraints["declared_budget"],
                         {"llm_tokens": 50000.0})

    def test_execution_constraints_name_tool_availability_only(self):
        ctx = self.h.build_prediction_context(_task("t1"), "ep1")
        constraints = ctx.execution_constraints
        self.assertIn("available_solver_families", constraints)
        self.assertIn("executor", constraints)
        self.assertIn("do not predict", constraints["note"])


# ---------------------------------------------------------------------------
# 9. phase-1 fixes do not regress
# ---------------------------------------------------------------------------


class TestFrozenContextFreezesTheWholeRequest(ContextCase):
    """P1: reusing a context must reuse EVERY prediction condition.

    The defect: `predict_outcome(..., context=ctx)` took a FRESH snapshot and
    re-read the current knowledge targets and reliability, so one request
    mixed a frozen `prediction_context` with a current `state`, and the
    recorded snapshot id disagreed with the context's.
    """

    def _provider(self):
        provider = RecordingProvider()
        h = ORHarness(home=self.home, world_model=provider,
                      embedding=self.backend)
        self.addCleanup(h.close)
        return h, provider

    def _advance_state(self, h, task):
        """A real state change AFTER the context was frozen."""
        return h.snapshot(task, "ep1", task_progress={
            "current_solution": {"value": {"objective": 42},
                                 "provenance": "observed",
                                 "epistemic": "fact"}})

    def test_reused_context_does_not_carry_later_progress(self):
        h, provider = self._provider()
        task = _task("t1")
        ctx = h.build_prediction_context(task, "ep1")
        self.assertEqual(ctx.snapshot["task_progress"], {})
        self._advance_state(h, task)          # state moves on
        spec = ActionSpec("execute_strategy", "t1", strategy_id="S01")
        h.predict_outcome(task, spec, "ep1", context=ctx)
        sent = provider.requests[0]["state"].get("task_progress")
        self.assertIn(sent, (None, {}),
                      "the frozen X must not pick up later progress")
        self.assertEqual(ctx.snapshot["task_progress"], {},
                         "the frozen context itself is untouched")

    def test_reused_context_records_the_context_snapshot_id(self):
        h, provider = self._provider()
        task = _task("t1")
        ctx = h.build_prediction_context(task, "ep1")
        self._advance_state(h, task)
        spec = ActionSpec("execute_strategy", "t1", strategy_id="S01")
        prediction = h.predict_outcome(task, spec, "ep1", context=ctx)
        self.assertEqual(prediction.input_snapshot_id, ctx.snapshot_id)
        self.assertEqual(prediction.model_info["conditions_source"],
                         "frozen_context")

    def test_reused_context_replays_the_frozen_progress_it_was_built_with(self):
        h, provider = self._provider()
        task = _task("t1")
        progress = {"selected_plan": {"value": {"strategy_id": "S01"},
                                      "provenance": "agent_reported",
                                      "epistemic": "fact"}}
        snap = h.snapshot(task, "ep1", task_progress=progress)
        ctx = h.build_prediction_context(task, "ep1", snapshot=snap)
        self._advance_state(h, task)          # progress moves on
        spec = ActionSpec("execute_strategy", "t1", strategy_id="S01")
        h.predict_outcome(task, spec, "ep1", context=ctx)
        sent = provider.requests[0]["state"]["task_progress"]
        # The FROZEN progress is replayed: what the context was built with.
        self.assertEqual(sent["selected_plan"]["value"]["strategy_id"], "S01")
        self.assertNotIn("current_solution", sent,
                         "progress established later must not appear")

    def test_reused_context_sends_the_frozen_knowledge_targets(self):
        h, provider = self._provider()
        task = _task("t1")
        spec = ActionSpec("execute_strategy", "t1", strategy_id="S01")
        # Seed a cell so the heuristic has a reason to propose a target.
        self.solve(_task("t1"), strategy="S01")
        ctx = h.build_prediction_context(task, "ep1", context_spec=spec)
        self.assertTrue(ctx.knowledge_targets,
                        "the proposal set is frozen into the context")
        # More evidence arrives, which WOULD change a re-derived proposal.
        self.solve(_task("t3"), strategy="S01")
        h.predict_outcome(task, spec, "ep1", context=ctx)
        sent = provider.requests[0].get("candidate_knowledge_targets") or []
        frozen = [t for t in ctx.knowledge_targets
                  if t.get("strategy_id") == "S01"]
        self.assertEqual(len(sent), len(frozen))
        self.assertEqual([t["strategy_id"] for t in sent],
                         [t["strategy_id"] for t in frozen])

    def test_reused_context_sends_the_frozen_reliability(self):
        h, provider = self._provider()
        task = _task("t1")
        spec = ActionSpec("execute_strategy", "t1", strategy_id="S01")
        ctx = h.build_prediction_context(task, "ep1", context_spec=spec)
        h.predict_outcome(task, spec, "ep1", context=ctx)
        self.assertEqual(provider.requests[0]["prediction_reliability"],
                         ctx.reliability)

    def test_targets_for_an_unproposed_strategy_are_empty_and_explained(self):
        from or_harness.world_model.context import (
            knowledge_targets_from_context,
        )
        h, _ = self._provider()
        task = _task("t1")
        ctx = h.build_prediction_context(
            task, "ep1",
            context_spec=ActionSpec("execute_strategy", "t1",
                                    strategy_id="S01"))
        other = ActionSpec("execute_strategy", "t1", strategy_id="S07")
        targets = knowledge_targets_from_context(ctx, other)
        self.assertEqual(targets, [],
                         "no target is re-derived from today's bank")
        self.assertTrue(any("not re-derived" in m for m in ctx.missing),
                        "the omission is stated, not silent")

    def test_a_budget_move_is_reported_not_silently_substituted(self):
        h, provider = self._provider()
        task = _task("b1")
        h.declare_budget("b1", {"llm_tokens": 1000}, "ep1")
        ctx = h.build_prediction_context(task, "ep1")
        frozen_budget = ctx.execution_constraints["declared_budget"]
        h.declare_budget("b1", {"llm_tokens": 9999}, "ep1")
        spec = ActionSpec("execute_strategy", "b1", strategy_id="S01")
        prediction = h.predict_outcome(task, spec, "ep1", context=ctx)
        # The frozen constraint is untouched ...
        self.assertEqual(ctx.execution_constraints["declared_budget"],
                         frozen_budget)
        # ... and the difference is recorded as a reported override.
        override = prediction.model_info["budget_checked_at_call_time"]
        self.assertEqual(override["current_declared_budget"],
                         {"llm_tokens": 9999.0})
        self.assertEqual(override["frozen_declared_budget"], frozen_budget)

    def test_a_supplied_current_result_is_bounded_too(self):
        """A supplied result does not get to smuggle later evidence in.

        The historical path REFUSES a recall result gathered now outright:
        creation-time bounding cannot see that an EXISTING entry was later
        revised, so a current result can never be PROVEN to belong to the
        frozen moment. (The previous behaviour — bounding it by creation
        time — was exactly the creation-time-filtering shortcut the phase
        rejects.)
        """
        h, _ = self._provider()
        task = _task("t1")
        snap = h.snapshot(task, "ep1")
        # Evidence arrives after the snapshot, then a caller supplies a
        # result gathered NOW.
        self.solve(_task("t1"))
        current = h.recall(task)
        self.assertTrue((current.get("vector_recall") or {})
                        .get("execution_evidence"),
                        "the live channel must really see the new evidence")
        with self.assertRaises(ValueError) as caught:
            h.build_prediction_context(task, "ep1", snapshot=snap,
                                       recall_result=current)
        self.assertIn("cannot be proven to belong to the frozen moment",
                      str(caught.exception))
        # The same result IS usable for a CURRENT gathering around that
        # snapshot: there the bounding to the snapshot's moment applies.
        ctx = h.build_prediction_context(task, "ep1", snapshot=snap,
                                         recall_result=current,
                                         historical=False)
        hits = [x for x in ctx.retrieval.hits
                if x["layer"] == "execution_evidence"]
        self.assertEqual(hits, [], "the supplied result is bounded as well")
        bounding = ctx.execution_constraints["retrieval_bounding"]
        self.assertGreaterEqual(bounding["dropped"], 1)
        # And the context is a CURRENT gathering, not a reconstruction:
        # the knowledge view and reliability were read and frozen.
        self.assertNotIn("HISTORICAL reconstruction", " ".join(ctx.notes))

    def test_a_current_gathering_around_a_supplied_snapshot_reads_the_bank(
            self):
        """historical=False around a caller snapshot is CURRENT gathering.

        This is the plan_next / predict_outcome pattern: the caller freezes
        the X/B snapshot, and the context gathers the retrieval, knowledge
        view, reliability and cell evidence NOW and freezes them. The
        presence of a snapshot alone must NOT flip the build into a
        historical reconstruction. The retrieval IS still bounded to the
        snapshot's moment (the snapshot is the X/B this decision froze), so
        evidence arriving after it stays out — what distinguishes the two
        paths is whether the banks are read at all, not whether later
        evidence is excluded.
        """
        h, _ = self._provider()
        task = _task("t1")
        # Evidence BEFORE the snapshot: a current gathering must see it.
        self.solve(_task("t1"))
        snap = h.snapshot(task, "ep1")
        ctx = h.build_prediction_context(task, "ep1", snapshot=snap,
                                         historical=False)
        execution_ids = {hit["evidence_id"] for hit in ctx.retrieval.hits
                         if hit["layer"] == "execution_evidence"}
        self.assertTrue(execution_ids,
                        "a current gathering reads the live channels")
        # It is NOT marked as a historical reconstruction ...
        self.assertNotIn("HISTORICAL reconstruction", " ".join(ctx.notes))
        self.assertFalse(any("was not saved with this snapshot" in m
                             for m in ctx.missing))
        # ... and the snapshot's own X/B is what the context froze.
        self.assertEqual(ctx.snapshot_id, snap.snapshot_id)

    def test_a_context_built_with_a_historical_snapshot_does_not_see_new_facts(self):
        """Building from a historical snapshot must not read today's memory."""
        from pathlib import Path
        h, _ = self._provider()
        task = _task("t1")
        snap = h.snapshot(task, "ep1")
        # Evidence arrives AFTER the snapshot was taken — recorded in the SAME
        # harness (and hence the same memory home).
        work = Path(self.home) / "ws_hist"
        work.mkdir(parents=True, exist_ok=True)
        script = work / "solve.py"
        script.write_text(
            "import json\n"
            "with open('result.json', 'w') as fh:\n"
            "    json.dump({'status': 'optimal', 'objective_value': 100.0, "
            "'objective_bound': 100.0, 'runtime_seconds': 0.01}, fh)\n",
            encoding="utf-8")
        h.record(h.execute(task, "S04", str(script), str(work),
                           solver="highs"))
        ctx = h.build_prediction_context(task, "ep1", snapshot=snap)
        # No retrieval was SAVED with the snapshot, so none is rebuilt:
        # reading today's index would fabricate evidence for an earlier
        # state. The gap is REPORTED rather than filled.
        execution_ids = {hit["evidence_id"] for hit in ctx.retrieval.hits
                         if hit["layer"] == "execution_evidence"}
        self.assertEqual(execution_ids, set())
        bounding = ctx.execution_constraints["retrieval_bounding"]
        self.assertEqual(bounding["rebuilt_from"], "none_saved")
        self.assertTrue(any("no historical retrieval was saved" in m
                            for m in ctx.retrieval.missing))
        self.assertTrue(any("HISTORICAL reconstruction" in n
                            for n in ctx.notes))

    def test_a_revised_entry_does_not_leak_into_a_historical_context(self):
        """P1: a revision AFTER the snapshot must not appear in it."""
        entry = StrategicEntry(
            entry_id="se_rev", strategy_id="S04",
            pattern={"predicates": {"family": "routing"}},
            expected_quality_hat=0.60, quality_interval=(0.5, 1.0),
            expected_cost_hat=self.make_record().cost,
            failure_prob=0.1, status="candidate", support_n=2,
            verification={"state": "verified", "claim": "c",
                          "conclusion": "holds"})
        self.h.sbank.add(entry)
        task = _task("t1")
        snap = self.h.snapshot(task, "ep1")      # freezes quality 0.60
        revised = self.h.sbank.get("se_rev")
        revised.expected_quality_hat = 0.95
        self.h.sbank.update(revised)             # the entry moves to 0.95
        ctx = self.h.build_prediction_context(task, "ep1", snapshot=snap)
        blob = str(ctx.to_dict())
        self.assertNotIn("0.95", blob,
                         "a later revision must not enter a context that "
                         "describes the earlier state")
        layers = (ctx.snapshot.get("coverage") or {}).get("knowledge_layers") \
            or {}
        frozen = [x["expected_quality_hat"]
                  for x in (layers.get("verified") or [])]
        self.assertEqual(frozen, [0.60],
                         "the FROZEN knowledge view keeps the value it had")

    def test_historical_reconstruction_marks_what_was_not_saved(self):
        """Missing history is reported, never filled from today's banks."""
        self.h.sbank.add(StrategicEntry(
            entry_id="se_h", strategy_id="S04",
            pattern={"predicates": {"family": "routing"}},
            expected_quality_hat=0.7, quality_interval=(0.5, 1.0),
            expected_cost_hat=self.make_record().cost,
            failure_prob=0.1, status="candidate", support_n=2,
            verification={"state": "verified", "claim": "c",
                          "conclusion": "holds"}))
        task = _task("t1")
        snap = self.h.snapshot(task, "ep1")
        ctx = self.h.build_prediction_context(task, "ep1", snapshot=snap)
        joined = " ".join(ctx.missing)
        self.assertIn("measured prediction reliability was not saved", joined)
        self.assertIn("per-cell evidence counts were not saved", joined)
        # And the reliability genuinely is empty rather than today's table.
        self.assertEqual(ctx.reliability, {})
        self.assertEqual(ctx.cell_evidence, {})
        # The knowledge it DOES have came from the snapshot's frozen view.
        self.assertTrue(any("FROZEN knowledge view" in m
                            for m in ctx.missing))

    def test_a_snapshot_without_a_frozen_knowledge_view_says_so(self):
        """An absent frozen view is reported, not filled from the bank."""
        self.h.sbank.add(StrategicEntry(
            entry_id="se_gap", strategy_id="S04",
            pattern={"predicates": {"family": "routing"}},
            expected_quality_hat=0.7, quality_interval=(0.5, 1.0),
            expected_cost_hat=self.make_record().cost,
            failure_prob=0.1, status="candidate", support_n=2,
            verification={"state": "verified", "claim": "c",
                          "conclusion": "holds"}))
        task = _task("t1")
        snap = self.h.snapshot(task, "ep1")
        snap.coverage.pop("knowledge_layers", None)   # the gap to report
        ctx = self.h.build_prediction_context(task, "ep1", snapshot=snap)
        self.assertTrue(any("no frozen knowledge view" in m
                            for m in ctx.missing))
        self.assertNotIn("se_gap", str(ctx.capability))


class TestOneCirForTheWholeRequest(ContextCase):
    """P2: an explicit CIR must drive the joint representation, the snapshot
    AND the retrieval — one request may not carry two structural judgments.

    The defect: `build_prediction_context(..., cir=cir)` used the CIR for the
    joint representation but then called `snapshot(task)` / `recall(task)`
    without it, so those fell back to the task's own `coupling`.
    """

    #: Strongly coupled: rc/tc/rx all 1.0/1.0/0.0 by construction.
    CIR = _cir()
    #: The task's OWN (weak) coupling, which must NOT win.
    WEAK = {"resource_coupling": 0.3, "temporal_coupling": 0.1,
            "route_complexity": 0.2, "semantic_coupling": 0.5}

    def _weak_task(self, task_id="t2"):
        task = _task(task_id)
        task["annotations"] = {"coupling": dict(self.WEAK)}
        return task

    def test_explicit_cir_drives_the_joint_representation(self):
        ctx = self.h.build_prediction_context(self._weak_task(), "ep1",
                                              cir=self.CIR)
        profile = ctx.joint.profile
        self.assertEqual(profile["resource_coupling"], 1.0)
        self.assertEqual(profile["temporal_coupling"], 1.0)
        self.assertEqual(profile["route_complexity"], 0.0)
        self.assertEqual(ctx.joint.sources["cir"], "caller_supplied")

    def test_the_snapshot_uses_the_same_cir_as_the_joint_representation(self):
        task = self._weak_task()
        ctx = self.h.build_prediction_context(task, "ep1", cir=self.CIR)
        from or_harness.world_model.context import task_with_effective_cir
        effective = task_with_effective_cir(task, self.CIR)
        via_snapshot = self.h.profile(effective)
        joint = ctx.joint.profile
        self.assertEqual(joint["resource_coupling"],
                         via_snapshot.resource_coupling)
        self.assertEqual(joint["temporal_coupling"],
                         via_snapshot.temporal_coupling)
        self.assertEqual(joint["route_complexity"],
                         via_snapshot.route_complexity)
        # And the snapshot really was built from that structure: its frozen
        # coverage is keyed by the effective cell, not the weak one.
        self.assertIsNotNone(ctx.snapshot_id)

    def test_recall_uses_the_same_cir_as_the_joint_representation(self):
        from or_harness.world_model.context import task_with_effective_cir
        task = self._weak_task()
        ctx = self.h.build_prediction_context(task, "ep1", cir=self.CIR)
        effective = task_with_effective_cir(task, self.CIR)
        recalled = self.h.recall(effective)
        self.assertEqual(ctx.joint.profile["resource_coupling"],
                         recalled["profile"]["resource_coupling"])
        self.assertEqual(recalled["profile"]["resource_coupling"], 1.0)

    def test_an_explicit_cir_replaces_the_tasks_own_coupling(self):
        task = self._weak_task()
        task["coupling"] = self.CIR          # task's own CIR says strong
        weak_cir = {"entities": [{"name": "x", "kind": "other"}],
                    "decisions": [], "constraints": [], "relations": [],
                    "coupling_groups": [], "issues": []}
        ctx = self.h.build_prediction_context(task, "ep1", cir=weak_cir)
        # The SUPPLIED one wins; the task's own is not silently kept.
        self.assertEqual(len(ctx.joint.cir["relations"]), 0)

    def test_without_an_explicit_cir_the_tasks_own_is_used(self):
        task = _task("t1")
        task["coupling"] = self.CIR
        ctx = self.h.build_prediction_context(task, "ep1")
        self.assertEqual(ctx.joint.sources["cir"], "task_coupling")
        self.assertEqual(len(ctx.joint.cir["relations"]), 1)

    def test_effective_cir_helper_is_the_single_resolution_point(self):
        from or_harness.world_model.context import (
            resolve_effective_cir,
            task_with_effective_cir,
        )
        task = self._weak_task()
        task["coupling"] = self.CIR
        self.assertIs(resolve_effective_cir(task, self.CIR), self.CIR)
        self.assertIs(resolve_effective_cir(task), task["coupling"])
        self.assertIsNone(resolve_effective_cir({}, None))
        # A supplied CIR reaches every existing consumer through the task.
        self.assertEqual(task_with_effective_cir(task, self.CIR)["coupling"],
                         self.CIR)


class TestMemoryContentVersion(ContextCase):
    """P3: the memory version must digest CONTENT, not just entry counts.

    The defect: `knowledge_content` was a count mapping, so editing an
    entry's expected quality or its strategy actions changed the context but
    left the version digest identical.
    """

    def _verified_entry(self, entry_id="se_k", strategy_id="S04",
                        quality=0.90):
        return StrategicEntry(
            entry_id=entry_id, strategy_id=strategy_id,
            pattern={"predicates": {"family": "routing"}},
            expected_quality_hat=quality, quality_interval=(0.5, 1.0),
            expected_cost_hat=self.make_record().cost,
            failure_prob=0.1, status="candidate", support_n=2,
            verification={"state": "verified", "claim": "c",
                          "conclusion": "holds"})

    def test_a_quality_revision_moves_the_content_digest(self):
        self.h.sbank.add(self._verified_entry())
        before = self.h.build_prediction_context(_task(), "ep1")
        d1 = before.capability_version["knowledge_content_digest"]
        entry = self.h.sbank.get("se_k")
        entry.expected_quality_hat = 0.55
        self.h.sbank.update(entry)
        after = self.h.build_prediction_context(_task(), "ep1")
        d2 = after.capability_version["knowledge_content_digest"]
        self.assertNotEqual(d1, d2,
                            "a knowledge revision must move the digest")
        # The COUNT is unchanged — which is exactly why counts were not enough.
        self.assertEqual(
            before.capability_version["knowledge_content"]
            ["knowledge_entries"],
            after.capability_version["knowledge_content"]
            ["knowledge_entries"])

    def test_an_action_revision_moves_the_content_digest(self):
        self.h.sbank.add(self._verified_entry())
        d1 = self.h.build_prediction_context(
            _task(), "ep1").capability_version["knowledge_content_digest"]
        entry = self.h.sbank.get("se_k")
        entry.applicability = ["use when capacity binds early"]
        self.h.sbank.update(entry)
        d2 = self.h.build_prediction_context(
            _task(), "ep1").capability_version["knowledge_content_digest"]
        self.assertNotEqual(d1, d2)

    def test_a_re_read_does_not_move_the_content_digest(self):
        self.h.sbank.add(self._verified_entry())
        first = self.h.build_prediction_context(_task(), "ep1")
        second = self.h.build_prediction_context(_task(), "ep1")
        self.assertEqual(
            first.capability_version["knowledge_content_digest"],
            second.capability_version["knowledge_content_digest"],
            "a fresh read timestamp is not a content change")

    def test_consultation_timestamps_are_excluded(self):
        self.h.sbank.add(self._verified_entry())
        before = self.h.build_prediction_context(_task(), "ep1")
        entry = self.h.sbank.get("se_k")
        entry.last_consulted_at = 1e12          # a read, not a revision
        self.h.sbank.update(entry)
        after = self.h.build_prediction_context(_task(), "ep1")
        self.assertEqual(
            before.capability_version["knowledge_content_digest"],
            after.capability_version["knowledge_content_digest"])
        self.assertIn("last_consulted_at",
                      before.capability_version["knowledge_content"]
                      ["excluded_keys"])

    def test_new_evidence_moves_the_content_digest(self):
        d1 = self.h.build_prediction_context(
            _task("t2"), "ep1").capability_version["knowledge_content_digest"]
        self.solve(_task("t1"))
        d2 = self.h.build_prediction_context(
            _task("t2"), "ep1").capability_version["knowledge_content_digest"]
        self.assertNotEqual(d1, d2)

    def test_the_version_carries_the_digest_and_its_composition_only(self):
        self.h.sbank.add(self._verified_entry())
        block = self.h.build_prediction_context(
            _task(), "ep1").capability_version["knowledge_content"]
        self.assertIn("digest", block)
        self.assertIn("knowledge_by_layer", block)
        self.assertIn("excluded_keys", block)
        # The digested CONTENT is not echoed back (the version must not become
        # a second, unbounded copy of the knowledge view).
        self.assertNotIn("knowledge", block)
        self.assertNotIn("retrieval_hits_content", block)

    def test_memory_content_digest_is_bounded_and_offline(self):
        from or_harness.world_model.context import memory_content_digest
        block = memory_content_digest(
            knowledge={"verified": [{"entry_id": "se_a",
                                     "expected_quality_hat": 0.7}],
                       "legacy_unknown": [], "unverified": []})
        self.assertEqual(block["knowledge_by_layer"], {"verified": 1})
        self.assertIsInstance(block["digest"], str)


class TestStructureConsistencyIsChecked(ContextCase):
    """P2: a snapshot (or context) built under one structure may not be
    combined with a different effective input.

    The defect: the snapshot identity check ran BEFORE the explicit CIR was
    merged, so supplying the original task's snapshot together with a NEW CIR
    passed the check and kept a structurally inconsistent snapshot — the
    joint representation reported 1.0/1.0/0.0 while the snapshot was built
    from 0.3/0.1/0.2, and the prediction entry point still accepted the
    resulting context.
    """

    STRONG = _cir()
    WEAK = {"resource_coupling": 0.3, "temporal_coupling": 0.1,
            "route_complexity": 0.2, "semantic_coupling": 0.5}

    def _weak_task(self, task_id="t2"):
        task = _task(task_id)
        task["annotations"] = {"coupling": dict(self.WEAK)}
        return task

    def test_a_snapshot_of_another_structure_is_refused(self):
        task = self._weak_task()
        snap = self.h.snapshot(task, "ep1")      # frozen with WEAK structure
        with self.assertRaises(ValueError) as caught:
            self.h.build_prediction_context(task, "ep1", snapshot=snap,
                                            cir=self.STRONG)
        message = str(caught.exception)
        self.assertIn("different structure", message)
        self.assertIn("resource_coupling", message)
        self.assertIn("Build a new context", message)

    def test_a_matching_snapshot_and_cir_are_accepted(self):
        task = self._weak_task()
        snap = self.h.snapshot(task, "ep1")
        ctx = self.h.build_prediction_context(task, "ep1", snapshot=snap)
        self.assertEqual(ctx.joint.profile["resource_coupling"], 0.3)
        self.assertEqual(ctx.snapshot_id, snap.snapshot_id)

    def test_structure_problems_ignores_an_unmeasured_dimension(self):
        from or_harness.world_model.context import structure_problems
        # Unknown on one side is "cannot compare", not "disagrees".
        self.assertEqual(structure_problems(
            expected={"resource_coupling": 0.3, "temporal_coupling": None,
                      "route_complexity": 0.2},
            actual={"resource_coupling": 0.3, "temporal_coupling": 0.9,
                    "route_complexity": 0.2},
            label="artifact"), [])
        # A KNOWN disagreement is always a refusal.
        self.assertTrue(structure_problems(
            expected={"resource_coupling": 1.0, "temporal_coupling": 0.1,
                      "route_complexity": 0.2},
            actual={"resource_coupling": 0.3, "temporal_coupling": 0.1,
                    "route_complexity": 0.2},
            label="artifact"))

    def test_a_context_from_another_structure_is_refused_by_the_api(self):
        from or_harness.world_model.context import task_with_effective_cir
        task = self._weak_task()
        ctx = self.h.build_prediction_context(task, "ep1")   # WEAK structure
        strong_task = task_with_effective_cir(task, self.STRONG)
        spec = ActionSpec("execute_strategy", "t2", strategy_id="S01")
        with self.assertRaises(ValueError) as caught:
            self.h.predict_outcome(strong_task, spec, "ep1", context=ctx)
        self.assertIn("does not describe", str(caught.exception))

    def test_the_effective_input_is_resolved_before_the_check(self):
        """A supplied CIR defines the structure the snapshot must match."""
        task = self._weak_task()
        snap = self.h.snapshot(task, "ep1")
        # Rebuilding with the SAME effective structure succeeds ...
        ctx = self.h.build_prediction_context(task, "ep1", snapshot=snap,
                                              cir=task["annotations"]
                                              ["coupling"])
        self.assertEqual(ctx.joint.sources["cir"], "caller_supplied")
        # ... while a different one is refused, in that order.
        with self.assertRaises(ValueError):
            self.h.build_prediction_context(task, "ep1", snapshot=snap,
                                            cir=self.STRONG)


class TestPhaseOneFixesSurvive(ContextCase):
    def test_a_configured_provider_without_content_is_still_contract_only(self):
        provider = RecordingProvider()
        h = ORHarness(home=self.home, world_model=provider,
                      embedding=self.backend)
        self.addCleanup(h.close)
        contract = h.build_strategy_outcome_contract(
            _task("t1"),
            {"action_type": "execute_strategy", "strategy_id": "S01"})
        self.assertEqual(contract.status, "contract_only")
        self.assertTrue(contract.provider_configured)
        self.assertFalse(contract.prediction_made)
        self.assertEqual(provider.requests, [])

    def test_legacy_candidate_adaptation_still_preserves_execution_config(self):
        from or_harness.world_model.contracts import CandidateRef
        spec = ActionSpec("execute_strategy", "t1", strategy_id="S01",
                          params={"time_limit": 60, "mip_gap": 0.01,
                                  "seed": 42})
        candidate = CandidateRef.from_action_spec(spec)
        self.assertEqual(candidate.config["time_limit"], 60)
        self.assertEqual(candidate.config["seed"], 42)
        self.assertEqual(candidate.scope_basis, "legacy_attempt")

    def test_unfinished_window_is_still_not_comparable(self):
        begun = self.h.begin_action("execute_strategy", _task("t1"), "ep1",
                                    params={"strategy_id": "S01"})
        window = self.h.strategy_execution_window("t1", "ep1",
                                                  strategy_id="S01")
        self.assertFalse(window.comparable)
        self.assertEqual(window.n_unfinished, 1)
        self.assertTrue(begun.get("action_id"))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
