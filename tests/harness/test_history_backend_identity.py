"""Synthetic backend/threshold fixtures, NOT GLM calibration evidence.

No real embedding endpoint, experiment bank or existing index is used. The
fixed vectors test configuration and admission mechanics only; they say
nothing about the similarity distribution of any real embedding model.
"""
from __future__ import annotations

import copy
import math
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from or_harness.api import ORHarness
from or_harness.core.schema import ExecutionRecord, ProblemProfile
from or_harness.strategy.embedding_index import (
    EmbeddingIndex,
    LocalHashEmbeddingBackend,
    OpenAICompatibleEmbeddingBackend,
    describe_backend,
    document_digest,
    document_execution,
)
from or_harness.world_model.history_retrieval import (
    _passes_admission,
    build_history_retrieval,
    build_query_views,
    history_config,
)
from or_harness.world_model.contracts import ActionSpec
from g_fixtures import COUPLED, GProvider


LOCAL = "local-hashing-embedding-v1"
REMOTE_FLOORS = {"problem": 0.45, "candidate_method": 0.45, "conditions": 0.40}
LOCAL_FLOORS = {"problem": 0.62, "candidate_method": 0.60, "conditions": 0.30}
OVERRIDES = {
    "problem": "OR_HARNESS_HISTORY_THRESHOLD",
    "candidate_method": "OR_HARNESS_HISTORY_METHOD_THRESHOLD",
    "conditions": "OR_HARNESS_HISTORY_CONDITIONS_THRESHOLD",
    "method_relevance": "OR_HARNESS_HISTORY_METHOD_RELEVANCE_THRESHOLD",
}
SENT_CONFIG_KEYS = (
    "backend_model_id", "max_local_cases", "pool_size", "similarity_thresholds",
    "method_similarity_threshold", "threshold_verified", "threshold_sources",
)


def synthetic_remote(model_id="GLM-Embedding-3"):
    backend = OpenAICompatibleEmbeddingBackend(
        "https://synthetic.invalid/v1", "synthetic-key", model_id)
    # Same width as local hashing deliberately does NOT establish identity.
    backend.dimension = 384
    return backend


class SyntheticInjectedBackend:
    model_id = "synthetic-injected/v1"
    dimension = 384


class IdentityCase(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {name: "" for name in OVERRIDES.values()})
        env.start()
        self.addCleanup(env.stop)
        network = patch("urllib.request.urlopen", side_effect=AssertionError(
            "Synthetic identity tests must never call a remote service"))
        self.network = network.start()
        self.addCleanup(network.stop)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)

    def wrapped(self, backend):
        return EmbeddingIndex(self.temporary.name, backend)


class TestBackendThresholdIdentity(IdentityCase):
    def test_local_backend_without_describe_and_index_have_identical_verified_config(self):
        backend = LocalHashEmbeddingBackend()
        self.assertFalse(hasattr(backend, "describe"))
        direct = history_config(backend)
        wrapped = history_config(self.wrapped(backend))
        self.assertEqual(direct, wrapped)
        self.assertEqual(wrapped["backend_model_id"], LOCAL)
        self.assertEqual(wrapped["backend_source"], "local-hashing")
        self.assertEqual(wrapped["similarity_thresholds"], LOCAL_FLOORS)
        self.assertEqual(wrapped["method_similarity_threshold"], 0.62)
        self.assertTrue(wrapped["threshold_verified"])
        self.assertEqual(wrapped["threshold_sources"],
                         {key: "profile:" + LOCAL for key in OVERRIDES})

    def test_wrapped_glm_and_direct_remote_use_the_same_unverified_profile(self):
        backend = synthetic_remote()
        direct = history_config(backend)
        wrapped = history_config(self.wrapped(backend))
        self.assertEqual(direct, wrapped)
        self.assertEqual(wrapped["backend_model_id"], "GLM-Embedding-3")
        self.assertEqual(wrapped["backend_source"], "remote")
        self.assertEqual(wrapped["threshold_profile"], "openai-compatible")
        self.assertEqual(wrapped["similarity_thresholds"], REMOTE_FLOORS)
        self.assertEqual(wrapped["method_similarity_threshold"], 0.45)
        self.assertIs(wrapped["threshold_verified"], False)
        self.assertIn("GLM-Embedding-3", wrapped["threshold_basis"])
        self.assertIn("PROVISIONAL", wrapped["threshold_basis"])
        self.assertEqual(wrapped["threshold_sources"],
                         {key: "profile:openai-compatible" for key in OVERRIDES})
        self.network.assert_not_called()

    def test_other_remote_model_is_not_verified_by_matching_the_local_dimension(self):
        backend = synthetic_remote("synthetic-remote/v2")
        self.assertEqual(backend.dimension, LocalHashEmbeddingBackend().dimension)
        config = history_config(self.wrapped(backend))
        self.assertEqual(config["similarity_thresholds"], REMOTE_FLOORS)
        self.assertEqual(config["backend_model_id"], "synthetic-remote/v2")
        self.assertFalse(config["threshold_verified"])

    def test_remote_model_named_like_hashing_does_not_inherit_local_calibration(self):
        config = history_config(synthetic_remote(LOCAL))
        self.assertEqual(config["backend_source"], "remote")
        self.assertEqual(config["similarity_thresholds"], REMOTE_FLOORS)
        self.assertFalse(config["threshold_verified"])

    def test_unknown_injected_backend_is_an_explicit_unverified_fallback(self):
        direct = history_config(SyntheticInjectedBackend())
        wrapped = history_config(self.wrapped(SyntheticInjectedBackend()))
        self.assertEqual(direct, wrapped)
        self.assertEqual(wrapped["backend_model_id"], "synthetic-injected/v1")
        self.assertEqual(wrapped["backend_source"], "injected")
        self.assertFalse(wrapped["threshold_verified"])
        self.assertTrue(wrapped["threshold_diagnostics"])
        self.assertIn("unverified", wrapped["threshold_basis"].lower())
        self.assertTrue(all(source.startswith("fallback:")
                            for source in wrapped["threshold_sources"].values()))

    def test_unidentified_backend_never_invents_a_local_model_id(self):
        config = history_config(SimpleNamespace(dimension=384))
        self.assertIsNone(config["backend_model_id"])
        self.assertFalse(config["threshold_verified"])
        self.assertTrue(config["threshold_diagnostics"])

    def test_describe_backend_handles_the_index_and_completes_partial_descriptions(self):
        class PartialDescription(SyntheticInjectedBackend):
            def describe(self):
                return {"source": "remote"}

        backend = PartialDescription()
        identity = describe_backend(self.wrapped(backend))
        self.assertEqual(identity["source"], "remote")
        self.assertEqual(identity["model_id"], backend.model_id)
        self.assertEqual(identity["dimension"], backend.dimension)

    def test_index_model_id_wins_over_a_conflicting_description_label(self):
        class ConflictingDescription(SyntheticInjectedBackend):
            def describe(self):
                return {"source": "remote", "model_id": LOCAL}

        config = history_config(self.wrapped(ConflictingDescription()))
        self.assertEqual(config["backend_model_id"], "synthetic-injected/v1")
        self.assertEqual(config["similarity_thresholds"], REMOTE_FLOORS)
        self.assertFalse(config["threshold_verified"])

    def test_injected_model_name_alone_does_not_prove_local_calibration(self):
        backend = SyntheticInjectedBackend()
        backend.model_id = LOCAL
        config = history_config(self.wrapped(backend))
        self.assertEqual(config["backend_source"], "injected")
        self.assertEqual(config["similarity_thresholds"], REMOTE_FLOORS)
        self.assertFalse(config["threshold_verified"])

    def test_each_override_has_its_own_source_and_invalidates_verification(self):
        for name, env_name in OVERRIDES.items():
            for value in (0.0, 0.71):
                with self.subTest(name=name, value=value), patch.dict(
                        os.environ, {env_name: str(value)}):
                    config = history_config(self.wrapped(LocalHashEmbeddingBackend()))
                    expected = {**LOCAL_FLOORS, "method_relevance": 0.62, name: value}
                    actual = {**config["similarity_thresholds"],
                              "method_relevance": config["method_similarity_threshold"]}
                    self.assertEqual(actual, expected)
                    for other in OVERRIDES:
                        self.assertEqual(config["threshold_sources"][other],
                                         "env:" + env_name if other == name
                                         else "profile:" + LOCAL)
                    self.assertFalse(config["threshold_verified"])
                    self.assertIn("override", config["threshold_basis"].lower())

    def test_all_remote_overrides_remain_independent_and_unverified(self):
        values = dict(zip(OVERRIDES, (0.11, 0.22, 0.33, 0.44)))
        with patch.dict(os.environ, {OVERRIDES[name]: str(value)
                                   for name, value in values.items()}):
            config = history_config(self.wrapped(synthetic_remote()))
        self.assertEqual(config["similarity_thresholds"],
                         {key: values[key] for key in REMOTE_FLOORS})
        self.assertEqual(config["method_similarity_threshold"], values["method_relevance"])
        self.assertFalse(config["threshold_verified"])
        self.assertEqual(config["threshold_sources"],
                         {name: "env:" + env for name, env in OVERRIDES.items()})

    def test_override_equal_to_a_local_default_is_still_not_validation_evidence(self):
        with patch.dict(os.environ, {OVERRIDES["problem"]: "0.62"}):
            config = history_config(LocalHashEmbeddingBackend())
        self.assertEqual(config["similarity_thresholds"], LOCAL_FLOORS)
        self.assertFalse(config["threshold_verified"])
        self.assertEqual(config["threshold_sources"]["problem"],
                         "env:" + OVERRIDES["problem"])

    def test_invalid_overrides_are_diagnosed_and_do_not_produce_verified_thresholds(self):
        for value in ("not-a-number", "nan", "inf"):
            with self.subTest(value=value), patch.dict(os.environ, {
                    OVERRIDES["problem"]: value}):
                config = history_config(LocalHashEmbeddingBackend())
                self.assertEqual(config["similarity_thresholds"], LOCAL_FLOORS)
                self.assertFalse(config["threshold_verified"])
                self.assertTrue(config["threshold_diagnostics"])
                self.assertEqual(config["threshold_sources"]["problem"], "profile:" + LOCAL)

    def test_missing_backend_preserves_zero_overrides_without_a_local_identity(self):
        with patch.dict(os.environ, {name: "0" for name in OVERRIDES.values()}):
            config = history_config(None)
        self.assertIsNone(config["backend_model_id"])
        self.assertFalse(config["threshold_verified"])
        self.assertEqual(config["similarity_thresholds"],
                         {name: 0.0 for name in REMOTE_FLOORS})
        self.assertEqual(config["method_similarity_threshold"], 0.0)
        self.assertEqual(set(config["threshold_sources"]), set(OVERRIDES))

    def test_missing_backend_or_index_keeps_existing_explicit_degradation(self):
        for index in (None, self.wrapped(synthetic_remote())):
            with self.subTest(index=index):
                block = build_history_retrieval(
                    SimpleNamespace(embedding_index=index),
                    task={"task_id": "synthetic", "text": "Synthetic task"},
                    profile=ProblemProfile(problem_id="synthetic", family="synthetic"))
                self.assertEqual(block["status"], "degraded")
                self.assertEqual(block["pool"], [])
                self.assertTrue(block["degraded"])
                self.assertIn("g", block["global"])
                self.assertFalse(block["config"]["threshold_verified"])


class SyntheticIndex(EmbeddingIndex):
    """Read-only in-memory fixture; never builds or modifies an index."""
    def __init__(self, backend, items):
        super().__init__("/tmp/synthetic-history-index-never-read", backend)
        self.fixture_items = copy.deepcopy(items)

    def status(self, layer):
        return {"usable": True, "count": len(self.fixture_items)}

    def items(self, layer):
        return copy.deepcopy(self.fixture_items)


class TestSyntheticBackendAdmission(IdentityCase):
    TEXT = "Synthetic historical assignment task."

    def history(self, backend, scores, *, candidates=()):
        record = ExecutionRecord(
            execution_id="synthetic-execution", task_id="synthetic-task",
            strategy_id="synthetic-method", created_at=100.0,
            task_text_digest="synthetic-text",
            profile_snapshot=ProblemProfile(problem_id="synthetic-task", family="synthetic"),
            quality={"status": "optimal", "feasible": True}, solver={"name": "highs"})
        query_vector = [1.0] + [0.0] * 383
        problem = scores.get("problem", 0.0)
        task_vector = [problem, math.sqrt(1 - problem ** 2)] + [0.0] * 382
        items = [{"id": record.execution_id,
                  "doc_digest": document_digest(document_execution(record, self.TEXT)),
                  "vector": task_vector, "task_vector": task_vector}]
        index = SyntheticIndex(backend, items)
        harness = SimpleNamespace(
            embedding_index=index,
            bank=SimpleNamespace(get=lambda execution_id: record),
            store=SimpleNamespace(get_task_text=lambda task_id, digest: self.TEXT))
        task = {"task_id": "synthetic-task", "text": self.TEXT}
        profile = record.profile_snapshot
        views = {view["text"]: view["name"] for view in
                 build_query_views(task, profile, candidates)}

        def retrieve(_harness, text, **kwargs):
            score = scores.get(views[text], 0.0)
            return [{"execution_id": record.execution_id, "task_id": record.task_id,
                     "strategy_id": record.strategy_id, "similarity": score}], None

        before = copy.deepcopy(record.to_dict())
        with patch.object(backend, "embed_query", return_value=query_vector), patch(
                "or_harness.world_model.history_retrieval._retrieve_view", side_effect=retrieve):
            block = build_history_retrieval(harness, task=task, profile=profile,
                                            candidates=candidates, cutoff=200.0)
        self.assertEqual(record.to_dict(), before)
        self.assertEqual(index.fixture_items, items)
        return block

    def test_same_synthetic_vectors_change_admission_only_with_backend_profile(self):
        # 0.50 is a synthetic cosine, not a real GLM measurement.
        local = self.history(LocalHashEmbeddingBackend(), {"problem": 0.50})
        remote = self.history(synthetic_remote(), {"problem": 0.50})
        self.assertEqual(local["pool"], [])
        self.assertEqual(len(remote["pool"]), 1)
        self.assertEqual(remote["backend"]["model_id"], remote["config"]["backend_model_id"])
        self.assertEqual(remote["config"]["similarity_thresholds"], REMOTE_FLOORS)
        self.assertFalse(remote["config"]["threshold_verified"])
        self.assertEqual(remote["accounting"]["above_threshold"], 1)

    def test_any_view_crossing_its_remote_floor_is_sufficient(self):
        candidate = ActionSpec("execute_strategy", "synthetic-task",
                               strategy_id="synthetic-method", method=COUPLED)
        for scores, admitted in (
                ({"problem": 0.46, "candidate:0": 0.44, "conditions": 0.39}, True),
                ({"problem": 0.44, "candidate:0": 0.46, "conditions": 0.39}, True),
                ({"problem": 0.44, "candidate:0": 0.44, "conditions": 0.41}, True),
                ({"problem": 0.44, "candidate:0": 0.44, "conditions": 0.39}, False)):
            with self.subTest(scores=scores):
                block = self.history(synthetic_remote(), scores, candidates=[candidate])
                self.assertEqual(bool(block["pool"]), admitted)
        self.assertTrue(_passes_admission({"candidate:0": 0.45}, REMOTE_FLOORS))


class TestFrozenHistoryConfigAudit(IdentityCase):
    def test_trace_config_copies_sent_fields_and_never_recomputes_live_overrides(self):
        provider = GProvider()
        harness = ORHarness(home=self.temporary.name, world_model=provider,
                            embedding=synthetic_remote())
        self.addCleanup(harness.close)
        task = {"task_id": "synthetic-audit", "family": "synthetic",
                "text": "Synthetic prediction-context audit fixture."}
        candidate = ActionSpec("execute_strategy", task["task_id"],
                               episode_id="synthetic-episode", method=COUPLED)
        with patch.dict(os.environ, {OVERRIDES["problem"]: "0.51"}):
            context = harness.build_prediction_context(
                task, "synthetic-episode", candidates=[candidate])
        frozen = copy.deepcopy(context.history_retrieval["config"])
        restored = harness.get_prediction_context(context.context_id)
        self.assertEqual(restored.history_retrieval["config"], frozen)
        with patch.dict(os.environ, {OVERRIDES["problem"]: "0.99"}), patch(
                "or_harness.world_model.history_retrieval.history_config",
                side_effect=AssertionError("audit must not recompute thresholds")):
            prediction = harness.predict_strategy_outcome(
                task, candidate, "synthetic-episode", context=context)
        sent = provider.requests[-1]["prediction_context"]["history_retrieval"]
        audit = prediction.trace.model_info["input_selection"]["history_selection"]["retrieval"]
        self.assertEqual(audit["config"], {key: sent[key] for key in SENT_CONFIG_KEYS})
        for key in SENT_CONFIG_KEYS:
            self.assertEqual(audit["config"][key], frozen[key])
        self.assertEqual(audit["config"]["similarity_thresholds"]["problem"], 0.51)
        self.assertEqual(audit["status"], sent["status"])
        self.assertEqual(audit["accounting"], sent["accounting"])
        self.assertNotIn("pool", audit)
        self.assertNotIn("cases", audit["config"])
        saved = harness.strategy_predictions.get(prediction.prediction_id)
        self.assertEqual(saved.trace.model_info["input_selection"]["history_selection"]
                         ["retrieval"], audit)


if __name__ == "__main__":
    unittest.main()
