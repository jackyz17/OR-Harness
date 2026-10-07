"""The online capability-gain claim, followed along the REAL path.

An H+ predicted with a strategy-outcome call used to be a dead end: it was
printed, it could be read back from the prediction, and nothing else. This
file pins the minimum bridge that connects it to the EXISTING offline
machinery:

1. **A vanished H+ now says why.** An absent block records "not predicted
   by the model" (never a silent null); a malformed block is dropped with
   its reason and the valid G/C/R forecast is KEPT.
2. **An online gain is visible to the capability bank.** ``inspect
   --bank capability`` reports it (pending) instead of 0.
3. **It can be bound and evaluated without a second prediction.** The
   offline stage-1/-2 API accepts the ``sp_`` id, materializing the
   prediction from the trace's frozen claim — no ``predict-capability``
   round is required.
4. **The claim is never rewritten.** The trace keeps what was predicted
   (task, changes, conditions) even after the real result arrives.
5. **An unexecuted or unlearned gain stays pending.** No knowledge change
   and no later evidence leave the verdict pending / no_change, never
   "verified".
6. **The model cannot declare its own verification flags.** Whatever the
   payload says about fact_bound/effect_verified is reset.
"""
import json
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tests.harness.helpers import HarnessTestCase  # noqa: E402

from or_harness.api import ORHarness  # noqa: E402
from or_harness.strategy.embedding_index import (  # noqa: E402
    LocalHashEmbeddingBackend,
)
from or_harness.world_model.provider import WorldModelProvider  # noqa: E402
from or_harness.world_model.trace_archive import (  # noqa: E402
    archive_capability_gain,
    effect_prediction_of,
    get_capability_trace,
)

EMBEDDING_ENV_KEYS = ("OR_EMBEDDING_BACKEND", "OR_EMBEDDING_BASE_URL",
                      "OR_EMBEDDING_MODEL", "OR_EMBEDDING_API_KEY")

TASK = {"task_id": "t1", "family": "routing",
        "description": "A small routing instance.",
        "spec": {"n_vars": 4, "n_constraints": 3, "n_int_vars": 4}}

GAIN = {
    "claim": "builds a reusable warm-start structure",
    "applies_to": ["routing with high resource_coupling"],
    "expected_changes": [
        {"metric": "solver_runtime_s", "direction": "decrease",
         "value": 20, "unit": "seconds", "value_kind": "relative",
         "beneficial_direction": "decrease",
         "baseline": {"kind": "conditional_stats", "value": 30}}],
    "verification_conditions": [{"condition": "a later task shows a >=10% "
                                            "runtime drop"}],
    "uncertainty": ["may not generalize"],
}

BASE = {
    "benefit": {"kind": "solution_quality",
                "metric": "normalized_objective_gap", "unit": "1-gap",
                "value": 0.8,
                "baseline": {"kind": "conditional_stats", "value": 0.7}},
    "cost": {"solver_runtime_s": 3.0},
}


class StubProvider(WorldModelProvider):
    name = "stub-trace"

    def __init__(self, payload=None):
        self.payload = payload or dict(BASE, capability_gain=dict(GAIN))
        self.requests = []

    def predict(self, request, timeout_s=None):
        self.requests.append(request)
        return {"payload": self.payload,
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                "error": None, "latency_s": 0.01}


class TraceCase(HarnessTestCase):
    def setUp(self):
        super().setUp()
        saved = {key: os.environ.pop(key, None) for key in EMBEDDING_ENV_KEYS}

        def restore():
            for key, value in saved.items():
                if value is not None:
                    os.environ[key] = value
        self.addCleanup(restore)
        self.backend = LocalHashEmbeddingBackend()
        self.provider = StubProvider()
        self.h = ORHarness(home=self.home, world_model=self.provider,
                           embedding=self.backend)
        self.addCleanup(self.h.close)

    def _spec(self, strategy_id="S01", episode_id="ep1"):
        return {"action_type": "execute_strategy", "strategy_id": strategy_id,
                "solver": "highs", "episode_id": episode_id, "task_id": "t1"}

    def _predict(self, **kwargs):
        return self.h.predict_strategy_outcome(self._task(), self._spec(),
                                               "ep1", **kwargs)

    def _task(self):
        return dict(TASK)


# ---------------------------------------------------------------------------
# 1. the trace is written for a claimed gain, and never rewritten
# ---------------------------------------------------------------------------


class TestTraceIsArchived(TraceCase):

    def test_a_claimed_gain_writes_a_trace(self):
        prediction = self._predict()
        self.assertTrue(prediction.claims_capability_gain)
        trace = get_capability_trace(self.h, prediction.prediction_id)
        self.assertIsNotNone(trace)
        self.assertEqual(trace.prediction_id, prediction.prediction_id)
        self.assertEqual(trace.task_id, "t1")
        self.assertEqual(trace.claim, GAIN["claim"])
        self.assertEqual(trace.state, "pending")

    def test_a_gain_less_prediction_writes_no_trace(self):
        self.provider.payload = dict(BASE)
        prediction = self._predict()
        self.assertFalse(prediction.claims_capability_gain)
        self.assertIsNone(
            get_capability_trace(self.h, prediction.prediction_id))

    def test_the_frozen_claim_survives_the_real_result(self):
        prediction = self._predict()
        before = get_capability_trace(self.h, prediction.prediction_id)
        # Execute the candidate for real.
        work = Path(self.home) / "ws"
        work.mkdir(parents=True, exist_ok=True)
        script = work / "solve.py"
        script.write_text(
            "import json\n"
            "with open('result.json', 'w') as fh:\n"
            "    json.dump({'status': 'optimal', 'objective_value': 1.0,"
            " 'objective_bound': 1.0, 'runtime_seconds': 0.01}, fh)\n",
            encoding="utf-8")
        record = self.h.execute(self._task(), None, str(script), str(work),
                                solver=None, episode_id="ep1",
                                prediction_id=prediction.prediction_id)
        self.h.record(record)
        after = get_capability_trace(self.h, prediction.prediction_id)
        # The CLAIM is untouched; only the live state moved.
        self.assertEqual(after.claim, before.claim)
        self.assertEqual(after.expected_changes, before.expected_changes)
        self.assertEqual(after.verification_conditions,
                         before.verification_conditions)
        self.assertEqual(after.state, "bound")
        self.assertEqual(after.bound_execution_ids, [record.execution_id])


# ---------------------------------------------------------------------------
# 2. the online gain is visible to the capability bank
# ---------------------------------------------------------------------------


class TestOnlineGainIsVisible(TraceCase):

    def test_the_capability_summary_reports_the_online_gain(self):
        prediction = self._predict()
        summary = self.h.online_capability_gains()
        self.assertEqual(summary["n_online_gains"], 1)
        self.assertEqual(summary["n_pending"], 1)
        self.assertEqual(summary["n_bound"], 0)
        self.assertFalse(summary["n_effect_verified"])
        entry = summary["online_gains"][0]
        self.assertEqual(entry["prediction_id"], prediction.prediction_id)
        self.assertEqual(entry["n_expected_changes"], 1)

    def test_inspect_capability_includes_the_online_view(self):
        self._predict()
        result = self.h.inspect(bank="capability")
        self.assertEqual(result["n_predictions"], 0)
        self.assertEqual(result["online_gains"]["n_online_gains"], 1)


# ---------------------------------------------------------------------------
# 3. the offline API accepts the sp_ id, materialized from the trace
# ---------------------------------------------------------------------------


class TestMaterializedPrediction(TraceCase):

    def test_the_effect_prediction_carries_the_original_claim(self):
        prediction = self._predict()
        materialized = effect_prediction_of(self.h, prediction.prediction_id)
        self.assertIsNotNone(materialized)
        self.assertEqual(materialized.prediction_id, prediction.prediction_id)
        self.assertEqual(len(materialized.expected_changes), 1)
        self.assertEqual(materialized.expected_changes[0].metric,
                         "solver_runtime_s")
        self.assertFalse(materialized.service_available,
                         "materialized from a trace: no model call, no "
                         "billing")
        info = materialized.trace.model_info
        self.assertEqual(info["online_prediction_id"], prediction.prediction_id)

    def test_inspect_capability_reads_the_sp_id(self):
        prediction = self._predict()
        result = self.h.inspect(bank="capability",
                                prediction_id=prediction.prediction_id)
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["prediction_source"], "online_trace")
        self.assertEqual(
            result["prediction"]["prediction_id"], prediction.prediction_id)

    def test_an_unknown_id_is_still_refused(self):
        with self.assertRaises(ValueError):
            self.h.inspect(bank="capability", prediction_id="sp_nope")


# ---------------------------------------------------------------------------
# 4. binding and evaluating the online claim needs no extra prediction
# ---------------------------------------------------------------------------


class TestFollowUpAlongTheRealPath(TraceCase):

    def _run(self, prediction_id=None):
        work = Path(self.home) / "ws_run"
        work.mkdir(parents=True, exist_ok=True)
        script = work / "solve.py"
        script.write_text(
            "import json\n"
            "with open('result.json', 'w') as fh:\n"
            "    json.dump({'status': 'optimal', 'objective_value': 1.0,"
            " 'objective_bound': 1.0, 'runtime_seconds': 0.01}, fh)\n",
            encoding="utf-8")
        record = self.h.execute(self._task(), None, str(script), str(work),
                                solver=None, episode_id="ep1",
                                prediction_id=prediction_id)
        self.h.record(record)
        return record

    def test_binding_the_sp_id_binds_the_real_execution(self):
        prediction = self._predict()
        record = self._run(prediction.prediction_id)
        result = self.h.bind_capability_maintenance(prediction.prediction_id)
        self.assertEqual(result["state"], "bound")
        self.assertEqual(result["prediction_source"], "online_trace")
        binding = result["binding"]
        self.assertIsNotNone(binding["adoption_action_id"])
        self.assertIn(record.execution_id, binding["actual_execution_ids"])

    def test_evaluating_the_sp_id_reports_a_real_state(self):
        prediction = self._predict()
        self._run(prediction.prediction_id)
        self.h.bind_capability_maintenance(prediction.prediction_id)
        result = self.h.evaluate_capability_effect(prediction.prediction_id)
        self.assertEqual(result["prediction_source"], "online_trace")
        evaluation = result["evaluation"]
        # An execution that produced NO knowledge change cannot "verify" a
        # capability gain: the honest verdict is no_change, never verified.
        self.assertIn(evaluation["state"],
                      ("no_change", "pending", "insufficient_evidence",
                       "inconclusive", "not_evaluable"))
        self.assertFalse(evaluation["effect_verified"])
        # And the trace reflects it.
        trace = get_capability_trace(self.h, prediction.prediction_id)
        self.assertIsNotNone(trace.effect_state)
        self.assertFalse(trace.effect_verified)

    def test_an_unexecuted_gain_stays_pending(self):
        prediction = self._predict()
        result = self.h.bind_capability_maintenance(prediction.prediction_id)
        self.assertEqual(result["state"], "not_adopted")
        trace = get_capability_trace(self.h, prediction.prediction_id)
        self.assertEqual(trace.state, "pending")
        self.assertFalse(trace.effect_verified)


# ---------------------------------------------------------------------------
# 5. the model may not declare its own verification
# ---------------------------------------------------------------------------


class TestModelCannotSelfVerify(TraceCase):

    def test_self_declared_verification_flags_are_reset(self):
        payload = dict(BASE, capability_gain=dict(
            GAIN,
            verification_conditions=[{
                "condition": "a later task shows a runtime drop",
                "fact_bound": True, "effect_verified": True}]))
        self.provider.payload = payload
        prediction = self._predict()
        condition = prediction.capability_gain.verification_conditions[0]
        self.assertTrue(condition.prediction_made)
        self.assertFalse(condition.fact_bound,
                         "the model may not declare its prediction bound")
        self.assertFalse(condition.effect_verified,
                         "the model may not declare its effect verified")


# ---------------------------------------------------------------------------
# 6. the knowledge delta is read in EITHER shape (the r10 defect)
# ---------------------------------------------------------------------------


class TestKnowledgeDeltaIsReadInBothShapes(TraceCase):
    """A DIRECT induction writes ``outcome.knowledge_delta`` at the top
    level; a WRAPPED operation nests it under ``operation_result``. Reading
    only the wrapped shape read a real 1-entry creation as 0."""

    def test_knowledge_result_of_accepts_both_shapes(self):
        from or_harness.world_model.trace_archive import knowledge_result_of
        direct = {"business_result": "relation_created",
                  "knowledge_delta": {"entries_created": ["se_1"]},
                  "execution_ids": ["ex_a"]}
        out = knowledge_result_of(outcome=direct, params={})
        self.assertEqual(out["knowledge_delta"]["entries_created"], ["se_1"])
        self.assertEqual(out["execution_ids"], ["ex_a"])
        wrapped = {"operation_result": {
            "business_result": "created",
            "knowledge_delta": {"entries_created": ["se_2"]},
            "execution_ids": ["ex_b"]}}
        out = knowledge_result_of(outcome=wrapped, params={})
        self.assertEqual(out["knowledge_delta"]["entries_created"], ["se_2"])
        self.assertEqual(out["execution_ids"], ["ex_b"])

    def test_direct_induction_is_read_as_one_entry_not_zero(self):
        """An ``induce`` action's own outcome feeds the H+ reader."""
        from or_harness.world_model.trace_archive import _operation_entries
        action = self.h.actions.begin_action(
            "induce", "__maintenance__", "maint_1", params={})
        self.h.actions.end_action(action.action_id, status="completed", outcome={
            "business_result": "relation_created",
            "knowledge_delta": {"entries_created": ["se_direct_1"],
                                "entries_removed": [],
                                "entry_changes": []},
            "execution_ids": ["ex_direct"],
        })
        produced = _operation_entries(self.h, action.action_id)
        self.assertEqual(produced["created_entry_ids"], ["se_direct_1"])
        self.assertEqual(produced["actual_execution_ids"], ["ex_direct"])


# ---------------------------------------------------------------------------
# 7. a 'none' stance still gets a product comparison
# ---------------------------------------------------------------------------


class TestNoneStanceIsFollowable(TraceCase):

    def test_none_stance_reports_the_real_product(self):
        """A claim that named no expected change (the r10 'none' shape) is
        followed up with the operation's ACTUAL knowledge product — never a
        silent 'not evaluable', and never a false 'verified'."""
        self.provider.payload = dict(BASE, capability_gain={
            "assessment": "none",
            "claim": "",
            "basis": ["using an existing tool adds no new capability"],
        })
        prediction = self._predict()
        # A pure 'none' stance claims no content, but it IS a stated stance,
        # so it is archived for follow-up (never dropped as silence).
        self.assertFalse(prediction.claims_capability_gain)
        trace = get_capability_trace(self.h, prediction.prediction_id)
        self.assertIsNotNone(trace)
        self.assertEqual(trace.assessment, "none")
        # Bind a REAL maintenance fact with a knowledge product, through the
        # public stage-1 API (which persists the binding for stage 2).
        adoption = self.h.actions.begin_action(
            "induce", "__maintenance__", "maint_none", params={
                "capability_prediction_id": prediction.prediction_id})
        self.h.actions.end_action(adoption.action_id, status="completed",
                                  outcome={
            "knowledge_delta": {"entries_created": ["se_new"],
                                "entry_changes": [], "entries_removed": []},
            "business_result": "relation_created",
        })
        bound = self.h.bind_capability_maintenance(
            prediction.prediction_id,
            adoption_action_id=adoption.action_id)
        self.assertTrue(bound["binding"]["changed"])
        result = self.h.evaluate_capability_effect(prediction.prediction_id)
        evaluation = result["evaluation"]
        # The `none` stance gets a PRODUCT comparison, not nothing.
        self.assertEqual(evaluation["state"], "product_compared")
        self.assertEqual(
            evaluation["evidence"]["knowledge_delta"]["n_created"], 1)
        # A new entry is a knowledge change, NOT a capability improvement.
        self.assertFalse(evaluation["effect_verified"])


if __name__ == "__main__":
    unittest.main()
