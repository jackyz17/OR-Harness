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
from decision_fixtures import bind_legacy, execute_scored, import_legacy_execution, planned_prediction


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
    DEFAULT_PAIRED_FEEDBACK_CHARS,
    DEFAULT_PAIRED_FEEDBACK_MAX_PAIRS,
    _paired_feedback_budget,
    _paired_feedback_max_pairs,
    _paired_feedback_order_key,
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
            "cost": {"output_tokens": 1200, "remaining_latency_s": 3.0, "tool_calls": 4},
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
          method_name=None, cell=None, scope="attempt", status=None,
          observed=None, eligibility=None, g_observed=None):
    row = {
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
    if observed is not None or eligibility is not None:
        benefit = {"metric": "task_check_passed_given_faithful_execution"}
        if observed is not None:
            benefit["observed"] = observed
        if eligibility is not None:
            benefit["eligibility"] = eligibility
        row["benefit"] = benefit
    if g_observed is not None:
        # A legacy row keeps the SAME label under g_observed_fact.
        row["g_observed_fact"] = {
            "metric": "task_check_passed_given_faithful_execution",
            "observed": g_observed}
    return row


def _row_is_execution_error(row):
    return row.get("execution_status") in ("error", "timeout")


def _row_has_label(row):
    benefit = row.get("benefit") or {}
    fact = row.get("g_observed_fact") or {}
    return ((isinstance(benefit, dict) and benefit.get("observed") is not None)
            or (isinstance(fact, dict)
                and fact.get("observed") is not None))


def _row_is_g_failure(row):
    if _row_is_execution_error(row):
        # A crash is unobserved for G: not a labeled failure, not a zero.
        return False
    if row.get("failure_classes"):
        return True
    return any(e.get("label") == "occurred"
               for e in row.get("risk_actual") or [])


def _block(pairs):
    meta = []
    for i, row in enumerate(pairs):
        conditions = row.get("conditions") or {}
        method = row.get("method_planned") or {}
        meta.append({
            "index": i, "pair_ref": row["evaluation_id"],
            "evaluation_id": row["evaluation_id"],
            "prediction_id": row["prediction_id"], "task_id": row["task_id"],
            "scope": row.get("scope"), "strategy_id": conditions.get("strategy_id"),
            "cell": conditions.get("cell"),
            "family": conditions.get("family"),
            "method_name": method.get("name"),
            "plan_method_words": [w for w in str(method.get("name") or "")
                                  .lower().split() if len(w) > 3],
            "action_id": row.get("action_id"),
            "execution_ids": list(row.get("execution_ids") or []),
            "has_label": _row_has_label(row),
            "is_failure": _row_is_g_failure(row),
            "is_execution_error": _row_is_execution_error(row),
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


class TestLabelFirstPairedDisplay(unittest.TestCase):
    """G calibration material shows LABELED cases first, crashes last.

    A crash has ``benefit.observed = null`` (unobserved, never 0). It must
    not outrank a labeled failure/success, and may hold only two display
    slots; the label semantics and the counts are unchanged.
    """

    _CANDIDATE = [{"method": {"name": "exact milp solve",
                              "steps": ["step one", "step two"]}}]

    def _mixed_pool(self):
        # Directory order deliberately puts the two CRASHES first and the
        # labeled failures LAST: position must not decide the display.
        return _block([
            _pair(0, task_id="t_cur", strategy_id="S00", executions=["ex_e0"],
                  failure_classes=["implementation"],
                  method_name="exact milp solve", status="error",
                  eligibility="unobserved"),
            _pair(1, task_id="t_cur", strategy_id="S01", executions=["ex_e1"],
                  failure_classes=["implementation"],
                  method_name="exact milp solve", status="error",
                  eligibility="unobserved"),
            _pair(2, task_id="t_cur", strategy_id="S02", executions=["ex_ok0"],
                  method_name="exact milp solve", observed=1.0,
                  eligibility="evaluable"),
            _pair(3, task_id="t_cur", strategy_id="S03", executions=["ex_ok1"],
                  method_name="exact milp solve", observed=1.0,
                  eligibility="evaluable"),
            _pair(4, task_id="t_cur", strategy_id="S04", executions=["ex_f0"],
                  failure_classes=["model"],
                  method_name="exact milp solve", observed=0.0,
                  eligibility="evaluable"),
            _pair(5, task_id="t_cur", strategy_id="S05", executions=["ex_f1"],
                  failure_classes=["model"],
                  method_name="exact milp solve", observed=0.0,
                  eligibility="evaluable"),
        ])

    # T1 ------------------------------------------------------------------
    def test_labeled_failures_and_successes_come_before_crashes(self):
        sel = select_paired_feedback(
            self._mixed_pool(), candidate_methods=self._CANDIDATE,
            task_id="t_cur")
        ids = [p["evaluation_id"] for p in sel["pairs"]]
        # Four labeled rows first (failures before successes), then crashes.
        self.assertEqual(ids, ["ev_4", "ev_5", "ev_2", "ev_3",
                               "ev_0", "ev_1"])
        self.assertEqual([_row_has_label(p) for p in sel["pairs"]],
                         [True, True, True, True, False, False])
        self.assertEqual([p["benefit"]["observed"] for p in sel["pairs"][:4]],
                         [0.0, 0.0, 1.0, 1.0])
        self.assertTrue(all(p.get("execution_status") == "error"
                            for p in sel["pairs"][4:]))
        self.assertEqual(sel["n_pairs_available"], 6)
        self.assertEqual(sel["n_pairs_execution_error"], 2)
        self.assertEqual(sel["n_pairs_included"] + sel["n_pairs_omitted"],
                         sel["n_pairs_available"])
        # The crash rows keep observed=None: they are never turned into 0.
        for p in sel["pairs"][4:]:
            self.assertIsNone(p.get("benefit", {}).get("observed"))

    # T2 ------------------------------------------------------------------
    def test_a_pool_of_only_crashes_is_small_but_not_empty(self):
        pairs = [_pair(i, task_id="t_cur", strategy_id=f"S{i:02d}",
                       executions=[f"ex_{i}"],
                       failure_classes=["implementation"],
                       method_name="exact milp solve", status="error",
                       eligibility="unobserved")
                 for i in range(6)]
        sel = select_paired_feedback(_block(pairs),
                                     candidate_methods=self._CANDIDATE,
                                     task_id="t_cur")
        self.assertGreater(sel["n_pairs_included"], 0)
        # The execution-error quota: two slots, rest counted as omitted.
        self.assertEqual(sel["n_pairs_included"], 2)
        self.assertEqual(sel["n_pairs_omitted"], 4)
        self.assertEqual([p["evaluation_id"] for p in sel["pairs"]],
                         ["ev_0", "ev_1"])
        self.assertEqual(sel["n_pairs_execution_error"], 6)

    # T3 ------------------------------------------------------------------
    def test_legacy_directory_without_label_flags_is_readable(self):
        block = self._mixed_pool()
        for entry in block["selection_meta"]:
            entry.pop("has_label", None)
            entry.pop("is_execution_error", None)
        sel = select_paired_feedback(block,
                                     candidate_methods=self._CANDIDATE,
                                     task_id="t_cur")
        # Missing flags default to False: no exception, and no crash is
        # counted as an execution error by the new quota.
        self.assertEqual(sel["n_pairs_execution_error"], 0)
        self.assertEqual(sel["n_pairs_included"], 6)
        self.assertEqual(sel["n_pairs_included"] + sel["n_pairs_omitted"],
                         sel["n_pairs_available"])

    # T4 ------------------------------------------------------------------
    def test_display_ceiling_and_defaults(self):
        self.assertEqual(DEFAULT_PAIRED_FEEDBACK_MAX_PAIRS, 12)
        self.assertEqual(DEFAULT_PAIRED_FEEDBACK_CHARS, 112000)
        pairs = [_pair(i, task_id="t_cur", strategy_id=f"S{i:02d}",
                       executions=[f"ex_{i}"],
                       method_name="exact milp solve", observed=1.0,
                       eligibility="evaluable")
                 for i in range(20)]
        sel = select_paired_feedback(_block(pairs),
                                     candidate_methods=self._CANDIDATE,
                                     task_id="t_cur", limit=12)
        self.assertEqual(sel["n_pairs_included"], 12)
        self.assertLessEqual(sel["n_pairs_included"],
                             sel["n_pairs_available"])
        self.assertEqual(sel["n_pairs_available"], 20)
        self.assertEqual(sel["n_pairs_omitted"], 8)
        # The environment overrides keep working.
        old_max = os.environ.get("OR_HARNESS_PAIRED_FEEDBACK_MAX")
        old_chars = os.environ.get("OR_HARNESS_PAIRED_FEEDBACK_CHARS")
        try:
            os.environ["OR_HARNESS_PAIRED_FEEDBACK_MAX"] = "7"
            os.environ["OR_HARNESS_PAIRED_FEEDBACK_CHARS"] = "9000"
            self.assertEqual(_paired_feedback_max_pairs(), 7)
            self.assertEqual(_paired_feedback_budget(), 9000)
        finally:
            if old_max is None:
                os.environ.pop("OR_HARNESS_PAIRED_FEEDBACK_MAX", None)
            else:
                os.environ["OR_HARNESS_PAIRED_FEEDBACK_MAX"] = old_max
            if old_chars is None:
                os.environ.pop("OR_HARNESS_PAIRED_FEEDBACK_CHARS", None)
            else:
                os.environ["OR_HARNESS_PAIRED_FEEDBACK_CHARS"] = old_chars

    # T5 ------------------------------------------------------------------
    def test_execution_errors_are_capped_at_two_display_slots(self):
        pairs = [
            _pair(0, task_id="t_cur", strategy_id="S00",
                  executions=["ex_f0"], failure_classes=["model"],
                  method_name="exact milp solve", observed=0.0,
                  eligibility="evaluable"),
            _pair(1, task_id="t_cur", strategy_id="S01",
                  executions=["ex_f1"], failure_classes=["model"],
                  method_name="exact milp solve", observed=0.0,
                  eligibility="evaluable"),
            _pair(2, task_id="t_cur", strategy_id="S02",
                  executions=["ex_ok0"], method_name="exact milp solve",
                  observed=1.0, eligibility="evaluable"),
            _pair(3, task_id="t_cur", strategy_id="S03",
                  executions=["ex_ok1"], method_name="exact milp solve",
                  observed=1.0, eligibility="evaluable"),
        ] + [
            _pair(10 + i, task_id="t_cur", strategy_id=f"E{i:02d}",
                  executions=[f"ex_e{i}"], failure_classes=["implementation"],
                  method_name="exact milp solve", status="error",
                  eligibility="unobserved")
            for i in range(5)
        ]
        sel = select_paired_feedback(_block(pairs),
                                     candidate_methods=self._CANDIDATE,
                                     task_id="t_cur")
        crashes = [p for p in sel["pairs"]
                   if p.get("execution_status") == "error"]
        self.assertEqual(len(crashes), 2)
        self.assertEqual(sel["n_pairs_available"], 9)
        self.assertEqual(sel["n_pairs_included"], 6)
        self.assertEqual(sel["n_pairs_omitted"], 3)
        self.assertEqual(sel["n_pairs_execution_error"], 5)
        self.assertEqual(sel["n_pairs_included"] + sel["n_pairs_omitted"],
                         sel["n_pairs_available"])

    # T6 ------------------------------------------------------------------
    def test_the_per_candidate_retry_cap_still_applies(self):
        same = [_pair(i, task_id="t_cur", strategy_id="SAME",
                      executions=[f"ex_{i}"], method_name="exact milp solve",
                      observed=1.0, eligibility="evaluable")
                for i in range(5)]
        others = [_pair(10 + i, task_id="t_cur", strategy_id=f"O{i:02d}",
                        executions=[f"ex_o{i}"],
                        method_name="exact milp solve", observed=1.0,
                        eligibility="evaluable")
                  for i in range(3)]
        sel = select_paired_feedback(_block(same + others),
                                     candidate_methods=self._CANDIDATE,
                                     task_id="t_cur")
        same_bucket = [p for p in sel["pairs"]
                       if (p.get("task_id"), p.get("strategy_id"))
                       == ("t_cur", "SAME")]
        self.assertLessEqual(len(same_bucket), 2)
        self.assertEqual(sel["n_pairs_included"] + sel["n_pairs_omitted"],
                         sel["n_pairs_available"])

    # A1: publish-time ordering -------------------------------------------
    def test_publish_order_is_labels_first_and_crashes_last(self):
        rows = [
            ("ev_err_new", 100.0, _pair(0, task_id="t_cur",
                                        strategy_id="E0",
                                        executions=["ex_e0"],
                                        status="error",
                                        eligibility="unobserved")),
            ("ev_err_old", 90.0, _pair(1, task_id="t_cur",
                                       strategy_id="E1",
                                       executions=["ex_e1"],
                                       status="error",
                                       eligibility="unobserved")),
            ("ev_success", 5.0, _pair(2, task_id="t_cur",
                                      strategy_id="S0",
                                      executions=["ex_ok"],
                                      observed=1.0,
                                      eligibility="evaluable")),
            ("ev_failure", 10.0, _pair(3, task_id="t_cur",
                                       strategy_id="F0",
                                       executions=["ex_f"],
                                       failure_classes=["model"],
                                       observed=0.0,
                                       eligibility="evaluable")),
        ]
        ordered = sorted(
            rows, key=lambda t: _paired_feedback_order_key(t[2], t[1],
                                                           rows.index(t)))
        self.assertEqual([row[0] for row in ordered],
                         ["ev_failure", "ev_success",
                          "ev_err_new", "ev_err_old"])

    def test_a_legacy_g_observed_fact_is_a_label(self):
        # An old row keeps the same label under g_observed_fact.
        legacy_failure = _pair(0, task_id="t_cur", strategy_id="S00",
                               executions=["ex_l"], failure_classes=["model"],
                               method_name="exact milp solve",
                               g_observed=0.0)
        modern_success = _pair(1, task_id="t_cur", strategy_id="S01",
                               executions=["ex_m"],
                               method_name="exact milp solve",
                               observed=1.0, eligibility="evaluable")
        crash = _pair(2, task_id="t_cur", strategy_id="S02",
                      executions=["ex_c"], status="error",
                      eligibility="unobserved")
        sel = select_paired_feedback(
            _block([crash, legacy_failure, modern_success]),
            candidate_methods=self._CANDIDATE, task_id="t_cur")
        self.assertEqual([p["evaluation_id"] for p in sel["pairs"]],
                         ["ev_0", "ev_1", "ev_2"])
        self.assertTrue(_row_has_label(sel["pairs"][0]))
        self.assertFalse(_row_has_label(sel["pairs"][2]))


class TestPairDirectoryAlwaysNamesItsBody(unittest.TestCase):
    """The directory (``selection_meta``) and the bodies (``pairs``) must
    describe the SAME case after sorting, budget trimming and filtering."""

    def test_stale_position_is_ignored_and_the_body_is_still_found(self):
        # The published block is relevance-ordered and budget-trimmed, but
        # its legacy directory kept the ORIGINAL positions (9, 4), which are
        # outside this two-pair block. A positional lookup would return an
        # empty object; identity lookup must return the right body.
        block = _block([
            _pair(0, task_id="t_cur", strategy_id="S04",
                  executions=["ex_a"], method_name="exact milp solve"),
            _pair(1, task_id="t_cur", strategy_id="S05",
                  executions=["ex_b"], method_name="exact lp solve"),
        ])
        block["selection_meta"][0]["index"] = 9
        block["selection_meta"][1]["index"] = 4
        block["n_pairs_total"] = 43
        block["n_pairs_included"] = 2
        block["n_pairs_omitted"] = 41
        sel = select_paired_feedback(
            block, recall_ids=["ex_a", "ex_b"],
            candidate_methods=[{"method": {"name": "exact milp solve"}}],
            task_id="t_cur")
        self.assertEqual([p["evaluation_id"] for p in sel["pairs"]],
                         ["ev_0", "ev_1"])
        for pair in sel["pairs"]:
            # A real body, never a directory-only shell.
            self.assertIn("conditions", pair)
            self.assertIn("method_planned", pair)
            self.assertNotIn(None, [pair.get("evaluation_id")])

    def test_directory_entry_without_a_body_is_not_a_case(self):
        real = _pair(0, task_id="t_cur", strategy_id="S04",
                     executions=["ex_real"], method_name="exact milp solve")
        block = _block([real])
        # A directory row for a case whose body was trimmed away, plus a
        # shell row (the old failure mode: selection fields only).
        block["selection_meta"].append({
            "index": 0, "pair_ref": "ev_missing", "evaluation_id": "ev_missing",
            "prediction_id": "sp_missing", "task_id": "t_cur",
            "strategy_id": "S04", "execution_ids": []})
        block["pairs"].append({"selected_reason": ["old selection"],
                               "relevance": 4, "background": False})
        sel = select_paired_feedback(
            block, recall_ids=["ex_real"],
            candidate_methods=[{"method": {"name": "exact milp solve"}}],
            task_id="t_cur")
        self.assertEqual([p["evaluation_id"] for p in sel["pairs"]], ["ev_0"])
        self.assertEqual(sel["n_pairs_unreadable"], 1)
        self.assertEqual(sel["n_pairs_index_without_body"], 1)
        self.assertIn("integrity_note", sel)
        self.assertEqual(sel["n_pairs_included"], 1)
        self.assertEqual(sel["n_pairs_available"], 1)

    def test_window_budget_and_display_counts_stay_separate(self):
        block = _block([
            _pair(0, task_id="t_cur", strategy_id="S04",
                  executions=["ex_a"], method_name="exact milp solve"),
        ])
        block.update({"n_pairs_total": 43, "n_pairs_included": 1,
                      "n_pairs_omitted": 42})
        sel = select_paired_feedback(
            block, task_id="other", candidate_methods=[])
        # Window: 43. Published block carried 1 and omitted 42 for size.
        self.assertEqual(sel["n_pairs_total"], 43)
        self.assertEqual(sel["n_pairs_publish_included"], 1)
        self.assertEqual(sel["n_pairs_publish_omitted"], 42)
        # This block can READ one body and DISPLAY one (background pass).
        self.assertEqual(sel["n_pairs_readable"], 1)
        self.assertEqual(sel["n_pairs_available"], 1)
        self.assertEqual(sel["n_pairs_included"], 1)
        self.assertIn("counts_note", sel)

    def test_model_filter_keeps_directory_and_bodies_aligned(self):
        from or_harness.world_model.episode_closeout import (
            _filter_pairs_by_model,
        )
        a = _pair(0, task_id="t_cur", strategy_id="S04",
                  executions=["ex_a"], method_name="exact milp solve")
        b = _pair(1, task_id="t_cur", strategy_id="S05",
                  executions=["ex_b"], method_name="exact lp solve")
        a["model_identity"] = b["model_identity"] = "model-a@1"
        block = _block([a, b])
        block["model_identity"] = "model-a@1"
        filtered = _filter_pairs_by_model(block, "model-a@1")
        self.assertEqual(len(filtered["pairs"]), 2)
        self.assertEqual(
            [m["pair_ref"] for m in filtered["selection_meta"]],
            [p["evaluation_id"] for p in filtered["pairs"]])
        withheld = _filter_pairs_by_model(block, "model-b@9")
        self.assertEqual(withheld["pairs"], [])
        self.assertEqual(withheld["selection_meta"], [])
        self.assertEqual(withheld["n_pairs_withheld"], 2)

    def test_legacy_block_without_directory_is_derived_from_bodies(self):
        block = _block([_pair(0, task_id="t_cur", strategy_id="S04",
                              executions=["ex_a"],
                              method_name="exact milp solve")])
        block.pop("selection_meta")
        sel = select_paired_feedback(
            block, recall_ids=["ex_a"],
            candidate_methods=[{"method": {"name": "exact milp solve"}}],
            task_id="t_cur")
        self.assertEqual([p["evaluation_id"] for p in sel["pairs"]], ["ev_0"])
        self.assertEqual(sel["n_pairs_unreadable"], 0)
        self.assertNotIn("integrity_note", sel)


class TestProviderMaterialIsCompacted(unittest.TestCase):
    """The sent material keeps method/prediction/outcome and stops inlining
    the check specs and repeated explanations the stored record holds."""

    def _view(self, content):
        return RetrievalView(
            channels_run=["structural"],
            hits=[{"identity": "strategic_knowledge:7",
                   "layer": "strategic_knowledge", "evidence_id": "7",
                   "version": "unknown", "channels": ["structural"],
                   "evidence_class": "legacy_knowledge",
                   "content": content}],
        )

    def test_full_check_spec_is_replaced_by_the_exact_reference(self):
        content = {
            "entry_id": "7", "strategy_id": "S04",
            "knowledge": {"entry_id": "7", "rule": {
                "entry_id": "7", "strategy_id": "S04",
                "detail_ref": "orx inspect --bank strategic --entry 7",
                "body": {"when": "W", "do": "D", "expect": "E"},
                "verification_scope": {
                    "state": "fact_checked", "authority": "agent",
                    "rule_version": 1, "n_checks": 2,
                    "publication": {"state": "published", "scope": "small"},
                    "seed_checks": [{
                        "execution_id": "ex_1", "state": "passed",
                        "validity": {"valid": True, "revision": 0},
                        "acceptance_rule": {"checker": "x",
                                            "spec": {"a": list(range(200))}},
                        "scope": {"basis": ["x"], "unchecked": ["y"]},
                    }],
                }}}}
        sent = _project_retrieval_for_provider(self._view(content))
        rule = sent["hits"][0]["content"]["knowledge"]["rule"]
        # Method and verdict stay readable.
        self.assertEqual(rule["body"]["when"], "W")
        self.assertEqual(rule["verification_scope"]["state"], "fact_checked")
        self.assertEqual(rule["verification_scope"]["publication"]["scope"],
                         "small")
        check = rule["verification_scope"]["seed_checks"][0]
        self.assertEqual(check["execution_id"], "ex_1")
        self.assertEqual(check["state"], "passed")
        # The spec is NOT inlined a second time; the exact field is named.
        self.assertNotIn("acceptance_rule", check)
        self.assertNotIn("scope", check)
        self.assertIn("acceptance_rule", check["check_ref"])
        self.assertIn("relative to detail_ref", check["check_ref"])
        self.assertEqual(rule["verification_scope"]["detail_ref"],
                         "orx inspect --bank strategic --entry 7")
        # Exact duplicate identity fields are stated once.
        self.assertNotIn("entry_id", rule)
        self.assertEqual(rule["entry_ref"], "content.entry_id")
        self.assertNotIn("strategy_id", rule)
        self.assertEqual(rule["strategy_id_ref"], "content.strategy_id")

    def test_large_solution_variable_dump_becomes_a_bounded_preview(self):
        variables = {f"x_{i}": float(i) for i in range(120)}
        content = {"execution_id": "ex_1", "method": {"name": "m"},
                   "failure_summary": [{"error_class": "model"}],
                   "task_check": "failed",
                   "solution_variables": variables,
                   "inspect_hint": "orx inspect --bank experience --id ex_1"}
        sent = _project_retrieval_for_provider(
            RetrievalView(channels_run=["semantic"], hits=[{
                "identity": "execution_evidence:ex_1",
                "layer": "execution_evidence", "evidence_id": "ex_1",
                "version": "v", "channels": ["semantic"],
                "evidence_class": "execution_fact", "content": content}]))
        body = sent["hits"][0]["content"]
        preview = body["solution_variables"]
        self.assertEqual(preview["n_variables"], 120)
        self.assertLessEqual(len(preview["preview_nonzero"]), 8)
        self.assertIn("solution_variables", preview["detail_ref"])
        # Method, failure and task check survive untouched.
        self.assertEqual(body["method"]["name"], "m")
        self.assertEqual(body["failure_summary"], [{"error_class": "model"}])
        self.assertEqual(body["task_check"], "failed")

    def test_identical_warning_and_version_note_are_stated_once(self):
        def hit(identity, content):
            return {"identity": identity, "layer": "strategic_knowledge",
                    "evidence_id": identity.split(":")[1], "version": "unknown",
                    "channels": ["structural"],
                    "evidence_class": "legacy_knowledge",
                    "version_unknown_note": "same explanation",
                    "content": content}
        warning = "cost not comparable across candidates"
        view = RetrievalView(channels_run=["structural"], hits=[
            hit("strategic_knowledge:1",
                {"entry_id": "1", "risk_warnings": [warning, "own-1"]}),
            hit("strategic_knowledge:2",
                {"entry_id": "2", "risk_warnings": [warning, "own-2"]}),
        ])
        sent = _project_retrieval_for_provider(view)
        self.assertEqual(sent["common_risk_warnings"], [warning])
        self.assertNotIn(warning, sent["hits"][0]["content"]["risk_warnings"])
        self.assertIn("own-1", sent["hits"][0]["content"]["risk_warnings"])
        self.assertEqual(sent["version_unknown_note"], "same explanation")
        self.assertNotIn("version_unknown_note", sent["hits"][0])


class TestPairBodyStaysReadableAfterCompaction(unittest.TestCase):
    """The pair projection removes scaffolding, never the method, the
    prediction, the outcome or a failure difference."""

    def _pair(self):
        return {
            "evaluation_id": "ev_1", "prediction_id": "sp_1",
            "task_id": "t1", "scope": "attempt",
            "method_planned": {"name": "m", "steps": ["a", "b", "c"]},
            "method_actual": {"name": "m", "steps": ["a", "b2", "c"]},
            "benefit": {"predicted": 0.8, "observed": 1.0,
                        "acceptance_rule": {"checker": "c",
                                            "fingerprint": "f",
                                            "spec": {"big": list(range(50))}}},
            "risk_actual": [
                {"event": "implementation_failure", "label": "not_occurred",
                 "predicted": True, "label_basis": "ok"},
                {"event": "timeout", "label": "not_occurred",
                 "predicted": False, "label_basis": "unpredicted_no_brier"},
                {"event": "task_check_failed", "label": "occurred",
                 "predicted": False, "label_basis": "unpredicted_no_brier"},
            ],
            "cost_measured": {"tool_calls": {"predicted": 3, "actual": 4}},
            "cost_observed_fact": {
                "values": {"tool_calls": 4, "latency_s": 1.0},
                "dimension_scopes": {"tool_calls": "attempt"},
                "boundary_exclusions": ["no complete window"],
                "execution_ids": ["ex_1"],
                "conditions": {"family": "routing",
                               "solver_conditions": {"solver": "python"}},
                "window": {"anchor": 1}, "work_facts": {"host_rounds": 3},
                "source_refs": ["ac_1"]},
            "task_check": {"state": "failed", "action_id": "ac_2",
                           "scope": {"basis": ["x"], "unchecked": ["y"]}},
            "execution_conditions": {
                "comparable": True, "execution_id": "ex_1",
                "comparison_key": {"solver": "python"},
                "config": {"solver": "python"}, "sources": {"x": "y"}},
        }

    def test_method_difference_survives_and_plan_does_not_repeat_steps(self):
        from or_harness.strategy.evidence_views import compact_feedback_pair
        value = compact_feedback_pair(self._pair())
        self.assertEqual(value["method_actual"]["steps"], ["a", "b2", "c"])
        self.assertNotIn("steps", value["method_planned"])
        self.assertEqual(value["method_planned"]["steps_ref"],
                         "method_actual.steps")
        self.assertEqual(
            value["method_planned"]["step_differences"],
            [{"step": 2, "planned": "b", "actual": "b2"}])
        self.assertIn("method_deviation", value)

    def test_unpredicted_non_events_are_counted_and_occurrences_stay(self):
        from or_harness.strategy.evidence_views import compact_feedback_pair
        value = compact_feedback_pair(self._pair())
        events = [e["event"] for e in value["risk_actual"]]
        self.assertIn("implementation_failure", events)  # predicted
        self.assertIn("task_check_failed", events)       # occurred
        self.assertNotIn("timeout", events)              # neither
        self.assertEqual(value["risk_actual_omitted"]["n"], 1)
        self.assertIn("detail_ref", value["risk_actual_omitted"])

    def test_cost_fact_keeps_values_scopes_exclusions_and_ref(self):
        from or_harness.strategy.evidence_views import compact_feedback_pair
        value = compact_feedback_pair(self._pair())
        fact = value["cost_observed_fact"]
        self.assertEqual(fact["dimension_scopes"], {"tool_calls": "attempt"})
        self.assertEqual(fact["boundary_exclusions"],
                         ["no complete window"])
        self.assertEqual(fact["execution_ids"], ["ex_1"])
        self.assertNotIn("window", fact)
        self.assertNotIn("work_facts", fact)
        self.assertNotIn("source_refs", fact)
        self.assertIn("detail_ref", fact)
        # The measured actuals are not stated twice.
        self.assertNotIn("values", fact)
        self.assertEqual(fact["values_not_scored"], {"latency_s": 1.0})
        self.assertEqual(value["cost_measured"]["tool_calls"]["actual"], 4)
        # The check scope is a spec; the verdict stays.
        self.assertEqual(value["task_check"]["state"], "failed")
        self.assertNotIn("scope", value["task_check"])
        self.assertIn("scope_ref", value["task_check"])


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
        record = import_legacy_execution(self.h,task, strategy, str(script), str(work),
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
        from or_harness.world_model.contracts import ActionSpec
        task = {"task_id": "t2", "family": "routing",
                "description": "allocate budget across channels exactly",
                "spec": {"n_vars": 10, "n_constraints": 5, "n_int_vars": 10},
                "annotations": {"coupling": {"resource_coupling": 0.3,
                                             "temporal_coupling": 0.1,
                                             "route_complexity": 0.2,
                                             "semantic_coupling": 0.5}}}
        # Close one episode so a pair exists.
        pred = planned_prediction(self.h,
            task, {"action_type": "execute_strategy", "strategy_id": "S04"},
            "ep1")
        record = self.solve(task)
        bind_legacy(self.h,pred.prediction_id, record.action_id)
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
        pred = planned_prediction(self.h,
            task, {"action_type": "execute_strategy", "strategy_id": "S04"},
            "ep1")
        record = self.solve(task)
        bind_legacy(self.h,pred.prediction_id, record.action_id)
        self.h.close_episode("t3", "ep1")
        ctx = self.h.build_prediction_context(task, "ep2")
        before = ctx.provider_view()["prediction_execution_pairs"]
        # A LATER close-out must not move the frozen context's selection.
        pred2 = planned_prediction(self.h,
            task, {"action_type": "execute_strategy", "strategy_id": "S05"},
            "ep2")
        record2 = self.solve(task, strategy="S05", episode_id="ep2",
                             tag="t3_ep2")
        bind_legacy(self.h,pred2.prediction_id, record2.action_id)
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
        pred = planned_prediction(self.h,
            task, {"action_type": "execute_strategy", "strategy_id": "S04"},
            "ep1")
        record = self.solve(task)
        bind_legacy(self.h,pred.prediction_id, record.action_id)
        self.h.close_episode("t4", "ep1")
        self.provider.requests.clear()
        planned_prediction(self.h,
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
        pred = planned_prediction(self.h,
            task, {"action_type": "execute_strategy", "strategy_id": "S04"},
            "ep1")
        record = self.solve(task)
        bind_legacy(self.h,pred.prediction_id, record.action_id)
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
