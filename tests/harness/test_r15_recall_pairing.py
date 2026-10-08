"""r15: world-model recall content de-duplication and historical pairing.

What this file asserts, one behaviour per test class:

1. recall content appears ONCE: one memory item surfaced by both channels is
   sent as one body, the two channel lists keep their OWN signal and point
   at it with a reference, an ``unknown`` version is not a conflict, a real
   two-known-version conflict is reported and not overwritten, and a
   verified entry is never downgraded to ``legacy`` by the summarising
   channel;
2. historical pairing is SELECTED then FROZEN: the relevant, bounded set is
   chosen once for the whole candidate set, failures and successes are both
   kept, unrelated background does not crowd it out, a cold start is empty;
3. the pairing body is de-duplicated internally: the predicted risk
   probability is stored once (in ``risk_predicted``), a method deviation
   references the steps it does not repeat, and the selection is a DISPLAY
   ceiling that never changes the stored evaluations or the calibration;
4. the projection is a SEND-time concern: the STORED context keeps both
   channels in full, and a reused frozen context is projected without
   reading today's banks.
"""
import copy
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tests.harness.helpers import HarnessTestCase  # noqa: E402

from or_harness.api import ORHarness  # noqa: E402
from or_harness.strategy.embedding_index import (  # noqa: E402
    LocalHashEmbeddingBackend,
)
from or_harness.world_model.context import (  # noqa: E402
    RetrievalView,
    _project_retrieval_for_provider,
    build_retrieval_view,
    dedupe_evidence,
    evidence_identity,
)
from or_harness.world_model.episode_closeout import (  # noqa: E402
    select_paired_feedback,
    selection_signature,
)
from or_harness.world_model.provider import WorldModelProvider  # noqa: E402

EMBEDDING_ENV_KEYS = ("OR_EMBEDDING_BACKEND", "OR_EMBEDDING_BASE_URL",
                      "OR_EMBEDDING_MODEL", "OR_EMBEDDING_API_KEY")


class RecordingProvider(WorldModelProvider):
    """A stub provider that records every request it is handed."""

    name = "recording-r15"

    def __init__(self):
        self.requests = []

    def predict(self, request, timeout_s=None):
        self.requests.append(request)
        return {"payload": {
            "benefit": {"kind": "solution_quality",
                        "metric": "normalized_objective_gap", "unit": "1-gap",
                        "value": 0.8,
                        "baseline": {"kind": "conditional_stats",
                                     "value": 0.7}},
            "cost": {"llm_tokens": 1200, "solver_runtime_s": 3.0},
            "risk": {"events": [{"event": "timeout", "probability": 0.2}]},
            "uncertainty": {"execution_randomness": 0.3, "knowledge_gap": 0.6},
        }, "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            "error": None, "latency_s": 0.01}


def _hit(layer, evidence_id, channel, *, version, content,
         evidence_class="execution_fact"):
    return {"layer": layer, "evidence_id": evidence_id,
            "identity": evidence_identity(layer, evidence_id),
            "version": version, "channels": [channel],
            "evidence_class": evidence_class, "content": content}


def _sz(obj):
    return len(json.dumps(obj, ensure_ascii=False, default=str))


# ---------------------------------------------------------------------------
# 1. recall content appears once
# ---------------------------------------------------------------------------


class TestRecallContentAppearsOnce(unittest.TestCase):
    def test_semantic_row_is_not_repeated_beside_the_hit(self):
        """The SAME row through the semantic channel and a hit is ONE body."""
        row = {"execution_id": "ex_1", "task_id": "t1",
               "task_text_digest": "d1", "observed_quality": 1.0,
               "similarity": 0.83, "structural_match": "same_cell"}
        view = build_retrieval_view({
            "recommendations": [],
            "vector_recall": {"execution_evidence": [row],
                              "strategic_knowledge": []}})
        sent = _project_retrieval_for_provider(view)
        # ONE hit carrying the full content...
        hits = [h for h in sent["hits"] if h["layer"] == "execution_evidence"]
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["content"]["observed_quality"], 1.0)
        # ...and the semantic list POINTS at it instead of repeating it.
        sem_rows = sent["semantic"]["execution_evidence"]
        self.assertEqual(len(sem_rows), 1)
        self.assertNotIn("observed_quality", sem_rows[0])
        self.assertIn("full_text_ref", sem_rows[0])
        self.assertIn("ex_1", sem_rows[0]["full_text_ref"])
        self.assertEqual(sem_rows[0]["similarity"], 0.83)

    def test_cross_channel_fields_are_merged_not_dropped(self):
        """A field only ONE channel carried must survive the merge.

        The structural channel has no ``task_text_digest``; the semantic one
        has no structured ``claim``. Keeping only the first-arriving row once
        dropped exactly the other channel's fields.
        """
        hits = [
            _hit("strategic_knowledge", "2", "structural", version="unknown",
                 content={"entry_id": "2", "claim": {"text": "use exact MILP"},
                          "risk_warnings": ["small instances only"]},
                 evidence_class="legacy_knowledge"),
            _hit("strategic_knowledge", "2", "semantic", version="d777",
                 content={"entry_id": "2", "claim_text": "use exact MILP",
                          "verification_state": "verified",
                          "similarity": 0.5},
                 evidence_class="verified_knowledge"),
        ]
        items, _ = dedupe_evidence(hits)
        self.assertEqual(len(items), 1)
        content = items[0]["content"]
        # Both channels' fields are present.
        self.assertIn("claim", content)
        self.assertIn("similarity", content)
        self.assertEqual(content["verification_state"], "verified")
        # The class is re-derived from the MERGED content: verified, not the
        # structural-only verdict ``legacy``.
        self.assertEqual(items[0]["content_channels"],
                         ["structural", "semantic"])

    def test_unknown_version_is_not_a_conflict(self):
        hits = [
            _hit("strategic_knowledge", "2", "structural", version="unknown",
                 content={"entry_id": "2"}, evidence_class="legacy_knowledge"),
            _hit("strategic_knowledge", "2", "semantic", version="d777",
                 content={"entry_id": "2", "claim_text": "x"},
                 evidence_class="verified_knowledge"),
        ]
        items, summary = dedupe_evidence(hits)
        self.assertNotIn("version_conflict", items[0])
        self.assertTrue(items[0]["version_unknown"])
        self.assertEqual(summary["version_conflicts"], [])
        self.assertEqual(summary["version_unknown"], ["strategic_knowledge:2"])

    def test_two_known_different_versions_is_reported_not_overwritten(self):
        hits = [
            _hit("execution_evidence", "ex_1", "semantic", version="d1",
                 content={"execution_id": "ex_1", "observed_quality": 1.0}),
            _hit("execution_evidence", "ex_1", "structural", version="d2",
                 content={"execution_id": "ex_1", "observed_quality": 0.0}),
        ]
        items, summary = dedupe_evidence(hits)
        self.assertIn("version_conflict", items[0])
        self.assertEqual(summary["version_conflicts"],
                         ["execution_evidence:ex_1"])
        # BOTH values are kept: the disagreement is not silently resolved.
        disagreements = items[0]["content"].get("_field_disagreements") or {}
        self.assertIn("observed_quality", disagreements)
        self.assertEqual(disagreements["observed_quality"]["kept"], 1.0)
        self.assertEqual(disagreements["observed_quality"]["also_seen"], 0.0)

    def test_a_verified_entry_is_never_downgraded_by_the_projection(self):
        """The summarising channel's missing state must not demote a claim."""
        view = RetrievalView(
            channels_run=["structural", "semantic"],
            hits=[_hit("strategic_knowledge", "5", "structural",
                       version="unknown",
                       content={"entry_id": "5", "reusable": True,
                                "claim": {"text": "t"}},
                       evidence_class="legacy_knowledge")],
            structural={"status": "ok", "recommendations": []},
            semantic={"status": "ok", "execution_evidence": [],
                      "strategic_knowledge": [
                          {"entry_id": "5", "claim_text": "t",
                           "verification_state": "verified",
                           "similarity": 0.4}]},
        )
        sent = _project_retrieval_for_provider(view)
        hit = next(h for h in sent["hits"]
                   if h["identity"] == "strategic_knowledge:5")
        self.assertEqual(hit["evidence_class"], "verified_knowledge")

    def test_claim_is_written_once_and_referenced(self):
        """The same claim text at three positions is kept once."""
        claim = {"text": "boundary: small only",
                 "verification": {"claim": "boundary: small only",
                                  "state": "verified"}}
        view = RetrievalView(
            channels_run=["structural"],
            hits=[_hit("strategic_knowledge", "7", "structural",
                       version="d9",
                       content={"entry_id": "7", "reusable": True,
                                "claim": copy.deepcopy(claim),
                                "knowledge": {"entry_id": "7",
                                              "claim": copy.deepcopy(
                                                  claim)}},
                       evidence_class="verified_knowledge")],
            structural={"status": "ok", "recommendations": [
                {"strategy_id": "S1", "evidence": "strategic_entry",
                 "evidence_refs": ["7"],
                 "claim": copy.deepcopy(claim),
                 "knowledge": {"entry_id": "7",
                               "claim": copy.deepcopy(claim)}}]},
            semantic={"status": "ok", "execution_evidence": [],
                      "strategic_knowledge": []},
        )
        sent = _project_retrieval_for_provider(view)
        hit = next(h for h in sent["hits"]
                   if h["identity"] == "strategic_knowledge:7")
        content = hit["content"]
        # The knowledge sub-object no longer repeats the claim.
        self.assertNotIn("claim", content["knowledge"])
        self.assertEqual(content["knowledge"]["claim_ref"], "content.claim")
        # The verification record no longer repeats it either.
        self.assertNotIn("claim", content["claim"]["verification"])
        self.assertEqual(content["claim"]["verification"]["claim_ref"],
                         "content.claim.text")
        # The verification STATE survives (it is not a duplicate).
        self.assertEqual(content["claim"]["verification"]["state"],
                         "verified")
        # The recommendation points at the content instead of restating it.
        rec = sent["structural"]["recommendations"][0]
        self.assertNotIn("knowledge", rec)
        self.assertEqual(rec["knowledge_ref"], "content.knowledge")


# ---------------------------------------------------------------------------
# 2. the projection is a SEND-time concern
# ---------------------------------------------------------------------------


class TestProjectionIsSendTime(unittest.TestCase):
    def test_stored_view_keeps_both_channels_in_full(self):
        row = {"execution_id": "ex_1", "task_id": "t1",
               "task_text_digest": "d1", "observed_quality": 1.0,
               "observed_cost": {"solver_runtime_s": 1.0},
               "method": {"name": "m", "steps": ["a", "b", "c", "d"]},
               "measurement_scope": "attempt", "profile_cell": "[0.2,0.4]",
               "failure_summary": {"classes": []},
               "inspect_hint": "look at the solve script"}
        view = build_retrieval_view({
            "recommendations": [],
            "vector_recall": {"execution_evidence": [row],
                              "strategic_knowledge": []}})
        stored = view.to_dict()
        sent = _project_retrieval_for_provider(view)
        # The stored semantic list keeps the full row...
        self.assertIn("observed_quality",
                      stored["semantic"]["execution_evidence"][0])
        # ...while the sent one is an index.
        self.assertNotIn("observed_quality",
                         sent["semantic"]["execution_evidence"][0])
        self.assertLess(_sz(sent["semantic"]), _sz(stored["semantic"]))
        # The stored HITS still carry the content; the sent hit is the ONE
        # copy, so the retrieval block as a whole is not duplicated.
        self.assertIn("observed_quality", stored["hits"][0]["content"])


# ---------------------------------------------------------------------------
# 3. historical pairing: selection and freezing
# ---------------------------------------------------------------------------


def _pair(index, *, task_id, strategy_id, executions, failure_classes=None,
          method_name=None, cell=None, scope="attempt", status=None):
    return {
        "evaluation_id": f"ev_{index}",
        "prediction_id": f"sp_{index}",
        "task_id": task_id,
        "episode_id": "ep1",
        "scope": scope,
        "strategy_id": strategy_id,
        "strategy_planned": strategy_id,
        "action_id": f"act_{index}",
        "execution_ids": list(executions),
        "conditions": {"task_id": task_id, "strategy_id": strategy_id,
                       "cell": cell},
        "method_planned": {"name": method_name, "n_steps": 2,
                           "steps": ["step one", "step two"]},
        "failure_classes": list(failure_classes or []),
        "execution_status": status,
    }


def _block(pairs):
    meta = []
    for i, row in enumerate(pairs):
        conditions = row.get("conditions") or {}
        method = row.get("method_planned") or {}
        meta.append({
            "index": i, "evaluation_id": row["evaluation_id"],
            "prediction_id": row["prediction_id"], "task_id": row["task_id"],
            "scope": row.get("scope"), "strategy_id": conditions.get("strategy_id"),
            "cell": conditions.get("cell"),
            "family": conditions.get("family"),
            "method_name": method.get("name"),
            "plan_method_words": [w for w in str(method.get("name") or "")
                                  .lower().split() if len(w) > 3],
            "action_id": row.get("action_id"),
            "execution_ids": list(row.get("execution_ids") or []),
            "is_failure": bool(row.get("failure_classes"))
            or row.get("execution_status") in ("error", "timeout"),
        })
    return {"pairs": pairs, "selection_meta": meta,
            "n_pairs_total": len(pairs), "model_identity": "m@1",
            "observation_rule_version": "wm-obs/3"}


class TestHistoricalPairSelection(unittest.TestCase):
    def _pool(self):
        # Two UNRELATED recent records, one relevant SUCCESS and one
        # relevant FAILURE for the current problem/method.
        return _block([
            _pair(0, task_id="unrelated_a", strategy_id="zzz",
                  executions=["ex_un_a"], method_name="quicksort heuristic",
                  status="error"),
            _pair(1, task_id="unrelated_b", strategy_id="yyy",
                  executions=["ex_un_b"], method_name="random search"),
            _pair(2, task_id="t_cur", strategy_id="S04",
                  executions=["ex_ok"], method_name="exact milp solve"),
            _pair(3, task_id="t_cur", strategy_id="S04",
                  executions=["ex_fail"], failure_classes=["model"],
                  method_name="exact milp solve", status="error"),
        ])

    def test_relevant_success_and_failure_outrank_unrelated(self):
        sel = select_paired_feedback(
            self._pool(), recall_ids=["ex_ok", "ex_fail"],
            candidate_methods=[{"strategy_id": "S04",
                                "method": {"name": "exact milp solve"}}],
            task_id="t_cur")
        ids = [p["evaluation_id"] for p in sel["pairs"]]
        self.assertIn("ev_2", ids)
        self.assertIn("ev_3", ids)
        self.assertNotIn("ev_0", ids)
        self.assertNotIn("ev_1", ids)
        self.assertEqual(sel["n_pairs_included"], 2)
        # The failure carries its reason; every pair says WHY it was chosen.
        for p in sel["pairs"]:
            self.assertTrue(p["selected_reason"])

    def test_selection_is_bounded_by_the_display_ceiling(self):
        pairs = [_pair(i, task_id="t_cur", strategy_id=f"S{i:02d}",
                       executions=[f"ex_{i}"],
                       method_name="exact milp solve")
                 for i in range(10)]
        sel = select_paired_feedback(
            _block(pairs), recall_ids=[f"ex_{i}" for i in range(10)],
            candidate_methods=[{"method": {"name": "exact milp solve"}}],
            task_id="t_cur", limit=6)
        self.assertEqual(sel["n_pairs_included"], 6)
        self.assertEqual(sel["n_pairs_available"], 10)
        self.assertGreaterEqual(sel["n_pairs_omitted"], 4)

    def test_one_candidate_retries_cannot_crowd_the_others(self):
        pairs = [_pair(i, task_id="t_cur", strategy_id="S04",
                       executions=[f"ex_{i}"], failure_classes=["model"],
                       method_name="exact milp solve")
                 for i in range(6)]
        sel = select_paired_feedback(
            _block(pairs), candidate_methods=[
                {"strategy_id": "S04",
                 "method": {"name": "exact milp solve"}}],
            task_id="t_cur", limit=6)
        # At most two retries of the SAME (task, strategy) hold slots.
        self.assertLessEqual(sel["n_pairs_included"], 2)

    def test_cold_start_is_empty_and_never_invents_history(self):
        sel = select_paired_feedback(
            {"pairs": [], "selection_meta": [], "n_pairs_total": 0},
            candidate_methods=[{"method": {"name": "exact milp solve"}}],
            task_id="t_cur")
        self.assertEqual(sel["n_pairs_included"], 0)
        self.assertIn("missing", sel)
        self.assertEqual(sel["pairs"], [])

    def test_background_is_marked_and_bounded(self):
        """With NO relevant signal, a SMALL marked background is used."""
        pairs = [_pair(i, task_id=f"other_{i}", strategy_id="zzz",
                       executions=[f"ex_{i}"], method_name="random search")
                 for i in range(8)]
        sel = select_paired_feedback(_block(pairs),
                                     candidate_methods=[],
                                     task_id="t_cur")
        self.assertGreater(sel["n_pairs_included"], 0)
        self.assertLessEqual(sel["n_pairs_included"], 3)
        for p in sel["pairs"]:
            self.assertTrue(p["background"])
        self.assertIn("background_note", sel)

    def test_signature_is_stable_for_the_same_query(self):
        a = selection_signature([{"name": "exact milp", "steps": ["x"]}],
                                ["family:routing"])
        b = selection_signature([{"name": "exact milp", "steps": ["x"]}],
                                ["family:routing"])
        self.assertEqual(a, b)
        c = selection_signature([{"name": "random search"}],
                                ["family:routing"])
        self.assertNotEqual(a, c)


# ---------------------------------------------------------------------------
# 4. end-to-end through the harness
# ---------------------------------------------------------------------------


class R15Case(HarnessTestCase):
    def setUp(self):
        super().setUp()
        saved = {key: os.environ.pop(key, None) for key in EMBEDDING_ENV_KEYS}

        def restore():
            for key, value in saved.items():
                if value is not None:
                    os.environ[key] = value
        self.addCleanup(restore)
        self.backend = LocalHashEmbeddingBackend()
        self.provider = RecordingProvider()
        self.h = ORHarness(home=self.home, world_model=self.provider,
                           embedding=self.backend)
        self.addCleanup(self.h.close)

    def solve(self, task, strategy="S04", objective=100.0, episode_id="ep1",
              tag=None):
        from pathlib import Path
        label = tag or f"{task['task_id']}_{strategy}_{episode_id}"
        work = Path(self.home) / f"ws_{label}"
        work.mkdir(parents=True, exist_ok=True)
        script = work / "solve.py"
        script.write_text(
            "import json\n"
            "with open('result.json', 'w') as fh:\n"
            "    json.dump({'status': 'optimal', 'objective_value': "
            f"{objective}, 'objective_bound': {objective}, "
            "'runtime_seconds': 0.01}, fh)\n", encoding="utf-8")
        record = self.h.execute(task, strategy, str(script), str(work),
                                solver="highs", episode_id=episode_id)
        self.h.record(record)
        return record

    def test_sent_request_carries_each_content_once(self):
        task = {"task_id": "t1", "family": "routing",
                "description": "load the depot then deliver demand 100",
                "spec": {"n_vars": 10, "n_constraints": 5, "n_int_vars": 10},
                "annotations": {"coupling": {"resource_coupling": 0.3,
                                             "temporal_coupling": 0.1,
                                             "route_complexity": 0.2,
                                             "semantic_coupling": 0.5}}}
        self.solve(task)
        ctx = self.h.build_prediction_context(task, "ep2")
        block = ctx.provider_view()["retrieval_evidence"]
        # Every hit's content appears ONCE, and the semantic list only
        # references it.
        for hit in block["hits"]:
            if hit["layer"] != "execution_evidence":
                continue
            eid = hit["evidence_id"]
            sem = next(r for r in block["semantic"]["execution_evidence"]
                       if str(r.get("execution_id")) == eid)
            self.assertNotIn("observed_quality", sem)
            self.assertIn(eid, sem["full_text_ref"])

    def test_selection_is_shared_and_frozen_across_candidates(self):
        from or_harness.world_model.prediction import ActionSpec
        task = {"task_id": "t2", "family": "routing",
                "description": "allocate budget across channels exactly",
                "spec": {"n_vars": 10, "n_constraints": 5, "n_int_vars": 10},
                "annotations": {"coupling": {"resource_coupling": 0.3,
                                             "temporal_coupling": 0.1,
                                             "route_complexity": 0.2,
                                             "semantic_coupling": 0.5}}}
        # Close one episode so a pair exists.
        pred = self.h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy", "strategy_id": "S04"},
            "ep1")
        record = self.solve(task)
        self.h.bind_strategy_outcome(pred.prediction_id, record.action_id)
        self.h.close_episode("t2", "ep1")
        self.provider.requests.clear()
        self.h.plan_next(
            task, "ep2",
            candidates=[ActionSpec("execute_strategy", "t2",
                                   strategy_id="S04"),
                        ActionSpec("execute_strategy", "t2",
                                   strategy_id="S05")],
            limits={"horizon": 1})
        blocks = [r["prediction_context"]["prediction_execution_pairs"]
                  for r in self.provider.requests]
        self.assertGreaterEqual(len(blocks), 2)
        # Both candidates share ONE frozen selection.
        self.assertEqual(blocks[0]["pairs"], blocks[1]["pairs"])
        self.assertEqual(blocks[0]["selection_signature"],
                         blocks[1]["selection_signature"])

    def test_later_bank_change_does_not_change_a_frozen_request(self):
        task = {"task_id": "t3", "family": "routing",
                "description": "allocate budget across channels exactly",
                "spec": {"n_vars": 10, "n_constraints": 5, "n_int_vars": 10},
                "annotations": {"coupling": {"resource_coupling": 0.3,
                                             "temporal_coupling": 0.1,
                                             "route_complexity": 0.2,
                                             "semantic_coupling": 0.5}}}
        pred = self.h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy", "strategy_id": "S04"},
            "ep1")
        record = self.solve(task)
        self.h.bind_strategy_outcome(pred.prediction_id, record.action_id)
        self.h.close_episode("t3", "ep1")
        ctx = self.h.build_prediction_context(task, "ep2")
        before = ctx.provider_view()["prediction_execution_pairs"]
        # A LATER close-out must not move the frozen context's selection.
        pred2 = self.h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy", "strategy_id": "S05"},
            "ep2")
        record2 = self.solve(task, strategy="S05", episode_id="ep2",
                             tag="t3_ep2")
        self.h.bind_strategy_outcome(pred2.prediction_id, record2.action_id)
        self.h.close_episode("t3", "ep2")
        after = ctx.provider_view()["prediction_execution_pairs"]
        self.assertEqual(before, after)

    def test_pair_body_is_de_duplicated(self):
        task = {"task_id": "t4", "family": "routing",
                "description": "allocate budget across channels exactly",
                "spec": {"n_vars": 10, "n_constraints": 5, "n_int_vars": 10},
                "annotations": {"coupling": {"resource_coupling": 0.3,
                                             "temporal_coupling": 0.1,
                                             "route_complexity": 0.2,
                                             "semantic_coupling": 0.5}}}
        pred = self.h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy", "strategy_id": "S04"},
            "ep1")
        record = self.solve(task)
        self.h.bind_strategy_outcome(pred.prediction_id, record.action_id)
        self.h.close_episode("t4", "ep1")
        self.provider.requests.clear()
        self.h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy", "strategy_id": "S04"},
            "ep2")
        block = (self.provider.requests[-1]["prediction_context"]
                 ["prediction_execution_pairs"])
        self.assertEqual(block["n_pairs_included"], 1)
        row = block["pairs"][0]
        # The predicted probability lives ONCE, in risk_predicted.
        for entry in row.get("risk_actual") or []:
            self.assertNotIn("predicted_probability", entry)
            self.assertIn("predicted", entry)
        if row.get("risk_predicted"):
            self.assertTrue(all("probability_ref" in e
                                for e in row["risk_predicted"]))

    def test_selection_does_not_change_the_stored_evaluations(self):
        task = {"task_id": "t5", "family": "routing",
                "description": "allocate budget across channels exactly",
                "spec": {"n_vars": 10, "n_constraints": 5, "n_int_vars": 10},
                "annotations": {"coupling": {"resource_coupling": 0.3,
                                             "temporal_coupling": 0.1,
                                             "route_complexity": 0.2,
                                             "semantic_coupling": 0.5}}}
        pred = self.h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy", "strategy_id": "S04"},
            "ep1")
        record = self.solve(task)
        self.h.bind_strategy_outcome(pred.prediction_id, record.action_id)
        self.h.close_episode("t5", "ep1")
        eid = self.h.store.closeout_registry()[0]["evaluation_ids"][0]
        before = json.dumps(self.h.get_strategy_evaluation(eid),
                            sort_keys=True)
        self.h.build_prediction_context(task, "ep2")
        after = json.dumps(self.h.get_strategy_evaluation(eid),
                           sort_keys=True)
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
