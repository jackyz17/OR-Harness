"""Embedding-first retrieval: discovery vs reuse, and the degradation matrix.

The design contract asserted here:

1. Text similarity decides WHICH memories are SEEN (cross-cell recall works);
   profile/applicability decides whether they may be REUSED.
2. Cross-cell recall never contaminates the target cell's statistics —
   ``for_profile`` / ``evidence_predicates`` / the score formula are untouched.
3. Unverified candidates are not surfaced by default.
4. Unindexed / legacy memories never silently disappear.
5. Read paths write nothing; ``rebuild-index --dry-run`` writes nothing.
6. Every honest refusal to run the text channel is REPORTED (``degraded``),
   never silently rendered as "found nothing".
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from helpers import HarnessTestCase

from or_harness.api import ORHarness
from or_harness.core.schema import ProblemProfile, StrategicEntry
from or_harness.strategy.embedding_index import (
    LAYER_EXECUTION,
    LAYER_STRATEGIC,
    LocalHashEmbeddingBackend,
    create_embedding_backend,
    document_digest,
    document_entry,
    document_execution,
    index_dir_for,
    readable_predicates,
)
from or_harness.strategy.vector_recall import (
    VectorRecallUnavailable,
    classify_applicability,
    recall_vectors,
)
from or_harness.world_model.state import (
    task_text,
    task_text_digest,
    task_text_from_payload,
)

REPO = Path(__file__).resolve().parents[2]

#: Environment keys that select a retrieval backend. Every test that asserts
#: a DEGRADATION must own these, otherwise an ambient variable in the
#: developer's (or CI's) shell would silently give the harness a backend and
#: the assertion would be testing the wrong thing.
EMBEDDING_ENV_KEYS = ("OR_EMBEDDING_BACKEND", "OR_EMBEDDING_BASE_URL",
                      "OR_EMBEDDING_MODEL", "OR_EMBEDDING_API_KEY")

REQ_A = ("A distribution centre must be loaded before the delivery window "
         "opens; demand 100 units may not be deferred.")
REQ_B = ("A distribution centre must be loaded before the delivery window "
         "opens; demand 200 units may be deferred by one period.")


def _task(task_id="t1", *, description, **coupling):
    values = {"resource_coupling": 0.3, "temporal_coupling": 0.1,
              "route_complexity": 0.2}
    values.update(coupling)
    return {"task_id": task_id, "family": "routing", "description": description,
            "annotations": {"coupling": {**values, "semantic_coupling": 0.5}}}


def _entry(entry_id="se_x", strategy_id="S04", predicates=None, *,
           status="candidate", verified=True, quality=0.9, support_n=3,
           applicability=None) -> StrategicEntry:
    """A published entry by default (the default LEGACY path — no stale
    marker), so tests exercise the recall gate rather than the admission
    machinery."""
    from or_harness.core.schema import CostVector
    verification: dict = {}
    if not verified:
        verification = {"state": "unverified", "claim": "c",
                        "conclusion": "no verdict yet"}
    return StrategicEntry(
        entry_id=entry_id, strategy_id=strategy_id,
        pattern={"predicates": predicates if predicates is not None else
                 {"family": "routing", "resource_coupling": [0.75, 1.0]}},
        expected_quality_hat=quality, quality_interval=(0.5, 1.0),
        expected_cost_hat=CostVector(llm_tokens=100.0,
                                     measured={"llm_tokens"}),
        failure_prob=0.1, status=status, support_n=support_n,
        applicability=list(applicability or ["use when capacity binds early"]),
        verification=verification)


class VectorCase(HarnessTestCase):
    """A harness with an EXPLICITLY injected local backend (hermetic).

    The embedding environment is cleared for the duration of the test so
    "no backend" is a real state here, not a coincidence.
    """

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

    def solve(self, task, strategy="S04", objective=100.0):
        """Write a solve script, execute and record it — the honest path."""
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


class TestDiscoveryVsReuse(VectorCase):
    """Cases 1, 2, 7: what is seen, and how it is labelled."""

    def test_cross_cell_similar_task_is_recalled_and_labelled(self):
        """Case 1: text-similar tasks in DIFFERENT structural cells see each
        other. A(rc=0.3) and B(rc=0.8) are nearly identical in text, so a
        recall for B must surface A's execution — labelled different_cell."""
        a = self.solve(_task("t_a", description=REQ_A, resource_coupling=0.3))
        self.solve(_task("t_b", description=REQ_B, resource_coupling=0.8))
        query_task = _task("t_q", description=REQ_B, resource_coupling=0.8)
        query_cell = self.h.profile(query_task) and \
            __import__("or_harness.core.schema", fromlist=["x"]).group_key(
                self.h.profile(query_task))
        self.assertNotEqual(a.group_l1, query_cell,
                            "the two tasks must occupy different cells")
        out = self.h.recall(query_task)
        hits = out["vector_recall"]["execution_evidence"]
        self.assertTrue(hits, "a text-similar execution must be discovered")
        cross = [hit for hit in hits
                 if hit["execution_id"] == a.execution_id]
        self.assertTrue(cross, "the cross-cell execution must be surfaced")
        self.assertEqual(cross[0]["structural_match"], "different_cell")
        self.assertEqual(cross[0]["profile_cell"], a.group_l1)

    def test_failure_evidence_is_visible(self):
        """Case 7: a failed execution of a similar task appears with its
        failure count — similarity never hides bad news."""
        work = Path(self.home) / "ws_fail"
        work.mkdir(parents=True, exist_ok=True)
        script = work / "solve.py"
        script.write_text("raise RuntimeError('boom')\n", encoding="utf-8")
        record = self.h.execute(_task("t_f", description=REQ_A), "S04",
                                str(script), str(work), solver="highs")
        record.quality = {"status": "error", "feasible": False, "objective": None,
                          "gap": None}
        self.h.record(record)
        out = self.h.recall(_task("t_q", description=REQ_A))
        hit = next(h for h in out["vector_recall"]["execution_evidence"]
                   if h["execution_id"] == record.execution_id)
        self.assertGreater(hit["failures"], 0)
        self.assertEqual(hit["status"], "error")
        self.assertFalse(hit["observed_quality"]["feasible"])

    def test_applicability_applies_conflicts_and_unknown(self):
        """Case 2: three distinguishable verdicts; missing information is
        never treated as applicability."""
        profile = self.h.profile(_task("t_q", description=REQ_A,
                                       resource_coupling=0.9))
        applies, reusable, reason = classify_applicability(profile, {
            "family": "routing", "resource_coupling": [0.75, 1.0]})
        self.assertEqual(applies, "applies")
        self.assertTrue(reusable)
        self.assertIsNone(reason)
        conflicts, reusable2, reason2 = classify_applicability(profile, {
            "family": "vrp", "resource_coupling": [0.75, 1.0]})
        self.assertEqual(conflicts, "conflicts")
        self.assertFalse(reusable2)
        self.assertIn("family mismatch", reason2)
        # A family-free claim with an unmeasured dimension on the task side:
        # nothing is contradicted, but nothing decides it either.
        bare = ProblemProfile(problem_id="p", family="routing")
        unknown, reusable3, reason3 = classify_applicability(
            bare, {"resource_coupling": [0.75, 1.0]})
        self.assertEqual(unknown, "unknown")
        self.assertFalse(reusable3, "unknown must not default to reusable")
        self.assertIn("unmeasured", reason3)

    def test_knowledge_hit_carries_claim_text_and_expected_values(self):
        """Case 6 (return payload): the surfaced knowledge carries its text,
        its expected (promised) values, and the applicability verdict."""
        self.solve(_task("t_a", description=REQ_A))  # populate so index exists
        predicates = {"family": "routing", "resource_coupling": [0.25, 0.50]}
        entry = _entry("se_match", predicates=predicates, quality=0.83)
        self.h.sbank.add(entry)
        self.h.index_sync.sync_entries(["se_match"])
        out = self.h.recall(_task("t_q", description=REQ_A,
                                  resource_coupling=0.3))
        hit = next(k for k in out["vector_recall"]["strategic_knowledge"]
                   if k["entry_id"] == "se_match")
        self.assertEqual(hit["expected_quality_hat"], 0.83)
        self.assertEqual(hit["structural_match"], "applies")
        self.assertTrue(hit["reusable"])
        self.assertIn("capacity binds early", hit["claim_text"])
        self.assertIn("resource coupling in [0.25, 0.50]", hit["claim_text"])

    def test_conflicting_knowledge_is_flagged_not_hidden(self):
        self.solve(_task("t_a", description=REQ_A))
        self.h.sbank.add(_entry(
            "se_other_family", strategy_id="S01",
            predicates={"family": "scheduling",
                        "resource_coupling": [0.25, 0.50]}))
        self.h.index_sync.sync_entries(["se_other_family"])
        out = self.h.recall(_task("t_q", description=REQ_A))
        hit = next(k for k in out["vector_recall"]["strategic_knowledge"]
                   if k["entry_id"] == "se_other_family")
        self.assertEqual(hit["structural_match"], "conflicts")
        self.assertFalse(hit["reusable"])
        self.assertIn("family mismatch", hit["reason"])


class TestStatisticsNotContaminated(VectorCase):
    """Case 3: discovery must not move a single aggregated number."""

    def test_recall_does_not_change_cell_statistics(self):
        self.solve(_task("t_a", description=REQ_A, resource_coupling=0.3))
        self.solve(_task("t_b", description=REQ_B, resource_coupling=0.8))
        profile_b = self.h.profile(_task("t_q", description=REQ_B,
                                         resource_coupling=0.8))
        before = {sid: (cell.n, round(cell.mean_quality, 6),
                        round(cell.fail_rate, 6))
                  for sid, cell in self.h.stats.for_profile(profile_b).items()}
        out = self.h.recall(_task("t_q", description=REQ_B,
                                  resource_coupling=0.8))
        self.assertTrue(out["vector_recall"]["execution_evidence"])
        after = {sid: (cell.n, round(cell.mean_quality, 6),
                       round(cell.fail_rate, 6))
                 for sid, cell in self.h.stats.for_profile(profile_b).items()}
        self.assertEqual(before, after)
        # And the cross-cell execution is NOT part of cell B's evidence.
        cell = self.h.stats.for_profile(profile_b).get("S04")
        if cell is not None:
            self.assertNotIn(
                next(h["execution_id"] for h
                     in out["vector_recall"]["execution_evidence"]
                     if h["structural_match"] == "different_cell"),
                list(cell.execution_ids))


class TestUnverifiedKnowledge(VectorCase):
    """Case 4: a candidate is not knowledge, in EITHER channel."""

    def setUp(self):
        super().setUp()
        self.solve(_task("t_a", description=REQ_A))
        self.h.sbank.add(_entry("se_unverified", verified=False,
                                predicates={"family": "routing",
                                            "resource_coupling": [0.25, 0.50]}))
        self.h.index_sync.sync_entries(["se_unverified"])

    def test_unverified_absent_by_default_present_on_request(self):
        out = self.h.recall(_task("t_q", description=REQ_A,
                                  resource_coupling=0.3))
        ids = [k["entry_id"]
               for k in out["vector_recall"]["strategic_knowledge"]]
        self.assertNotIn("se_unverified", ids)
        rec_refs = [ref for r in out["recommendations"]
                    for ref in r["evidence_refs"]]
        self.assertNotIn("se_unverified", rec_refs)
    def test_include_unverified_reveals_it(self):
        out = self.h.recall(_task("t_q", description=REQ_A,
                                  resource_coupling=0.3),
                            include_unverified=True)
        ids = [k["entry_id"]
               for k in out["vector_recall"]["strategic_knowledge"]]
        self.assertIn("se_unverified", ids)

    def test_offline_view_labels_the_candidate(self):
        """Surfacing an unpublished candidate must not present it as
        knowledge — the recommendation carries an explicit warning."""
        out = self.h.recall(_task("t_q", description=REQ_A,
                                  resource_coupling=0.3),
                            include_unverified=True)
        rec = next(r for r in out["recommendations"]
                   if r["strategy_id"] == "S04")
        self.assertIn("se_unverified", rec["evidence_refs"])
        self.assertTrue(any("UNPUBLISHED" in w for w in rec["risk_warnings"]))
        knowledge = next(k for k in out["vector_recall"]["strategic_knowledge"]
                         if k["entry_id"] == "se_unverified")
        self.assertEqual(knowledge["verification_state"], "unverified")


class TestLegacyRecords(VectorCase):
    """Case 5: no vector is a LABEL, never a disappearance."""

    def test_legacy_record_is_unindexed_but_queryable(self):
        # A record appended straight to the bank, with no task text anywhere.
        # (First build an index from a real execution, so the channel RUNS
        # and the legacy record is genuinely visible as an unindexed item 
        # rather than hidden behind a degraded response.)
        self.solve(_task("t_indexed", description=REQ_A))
        legacy = self.make_record(task_id="t_legacy", strategy_id="S01")
        self.h.bank.append(legacy)
        out = self.h.recall(_task("t_q", description=REQ_A))
        hits = out["vector_recall"]["execution_evidence"]
        self.assertNotIn(legacy.execution_id, [h["execution_id"] for h in hits])
        self.assertGreaterEqual(
            out["vector_recall"]["unindexed"]["execution_evidence"], 1)
        note = out["vector_recall"]["unindexed"]["note"]
        self.assertIn("orx inspect", note)
        # Still reachable through the documented entry points.
        found = self.h.inspect(bank="experience", task_id="t_legacy")
        self.assertEqual(found["count"], 1)

    def test_snapshot_recovers_text_for_records_without_execute(self):
        """A record appended directly still links to text when the task's
        real belief snapshot exists — recovered, never invented."""
        task = _task("t_snap", description=REQ_A, resource_coupling=0.9)
        self.h.snapshot(task, "ep1")
        record = self.h.execute(task, "S04",
                                self._script(task, "ws_snap"), 
                                str(Path(self.home) / "ws_snap"),
                                solver="highs")
        # Simulate the agent path: drop the digest the write path assigned.
        record.task_text_digest = None
        out = self.h.record(record)
        stored = self.h.bank.get(record.execution_id)
        self.assertEqual(stored.task_text_digest, task_text_digest(task))
        self.assertEqual(out["index_sync"]["state"], "synced")
        texts = self.h.inspect(bank="texts", task_id="t_snap")
        self.assertEqual(texts["count"], 1)
        self.assertIn("demand 100", texts["task_texts"][0]["text"])

    def test_no_snapshot_and_no_digest_stays_unindexed(self):
        record = self.make_record(task_id="t_orphan", strategy_id="S01")
        out = self.h.record(record)
        self.assertIsNone(self.h.bank.get(record.execution_id).task_text_digest)
        self.assertEqual(out["index_sync"]["state"], "skipped")
        self.assertIn("unavailable", out["index_sync"]["reason"])

    def _script(self, task, name):
        work = Path(self.home) / name
        work.mkdir(parents=True, exist_ok=True)
        script = work / "solve.py"
        script.write_text("import json\n"
                          "with open('result.json', 'w') as fh:\n"
                          "    json.dump({'status': 'optimal', 'objective_value':"
                          " 1.0, 'objective_bound': 1.0, 'runtime_seconds': 0.01}"
                          ", fh)\n", encoding="utf-8")
        return str(script)


class TestTextVersions(VectorCase):
    """Case 10: one task_id, two contents, two links — never one shared row."""

    def test_two_executions_of_one_task_link_to_their_own_text(self):
        task_a = _task("t1", description=REQ_A)
        task_b = _task("t1", description=REQ_B)
        first = self.solve(task_a)
        second = self.solve(task_b)
        self.assertNotEqual(first.task_text_digest, second.task_text_digest)
        texts = self.h.inspect(bank="texts", task_id="t1")["task_texts"]
        self.assertEqual(len(texts), 2)
        self.assertEqual(
            self.h.store.get_task_text("t1", first.task_text_digest),
            task_text(task_a))
        self.assertEqual(
            self.h.store.get_task_text("t1", second.task_text_digest),
            task_text(task_b))
        # Idempotent: re-capturing the same version stores nothing new.
        self.h.capture_task_text(task_a)
        self.assertEqual(
            self.h.inspect(bank="texts", task_id="t1")["count"], 2)


class TestNewMemoryIsSearchable(VectorCase):
    """Case 11: a fresh record is discoverable without a rebuild."""

    def test_record_then_recall_hits_without_rebuild(self):
        record = self.solve(_task("t_new", description=REQ_A))
        out = self.h.recall(_task("t_q", description=REQ_A))
        ids = [h["execution_id"] for h in out["vector_recall"]["execution_evidence"]]
        self.assertIn(record.execution_id, ids)
        self.assertEqual(out["vector_recall"]["stale_indexed"]
                         [LAYER_EXECUTION], 0)

    def test_excerpt_is_returned_with_the_hit(self):
        self.solve(_task("t_new", description=REQ_A))
        out = self.h.recall(_task("t_q", description=REQ_A))
        hit = out["vector_recall"]["execution_evidence"][0]
        self.assertIn("distribution centre", hit["task_text_excerpt"])


class TestStaleAndLifecycle(VectorCase):
    """Case 12: a changed/retired memory never returns its old advice."""

    def test_retired_entry_leaves_the_index(self):
        self.solve(_task("t_a", description=REQ_A))
        self.h.sbank.add(_entry("se_gone", predicates={
            "family": "routing", "resource_coupling": [0.25, 0.50]}))
        self.h.index_sync.sync_entries(["se_gone"])
        out = self.h.recall(_task("t_q", description=REQ_A,
                                  resource_coupling=0.3))
        self.assertIn("se_gone",
                      [k["entry_id"]
                       for k in out["vector_recall"]["strategic_knowledge"]])
        self.h.retire("se_gone", reason="stopped working")
        out2 = self.h.recall(_task("t_q", description=REQ_A,
                                   resource_coupling=0.3))
        self.assertNotIn("se_gone",
                         [k["entry_id"]
                          for k in out2["vector_recall"]["strategic_knowledge"]])

    def test_dormant_entry_is_not_surfaced(self):
        self.solve(_task("t_a", description=REQ_A))
        entry = _entry("se_sleep", predicates={
            "family": "routing", "resource_coupling": [0.25, 0.50]})
        self.h.sbank.add(entry)
        self.h.index_sync.sync_entries(["se_sleep"])
        entry.status = "dormant"
        self.h.sbank.update(entry)
        out = self.h.recall(_task("t_q", description=REQ_A,
                                  resource_coupling=0.3))
        self.assertNotIn("se_sleep",
                         [k["entry_id"]
                          for k in out["vector_recall"]["strategic_knowledge"]])

    def test_changed_claim_text_is_reported_stale(self):
        self.solve(_task("t_a", description=REQ_A))
        entry = _entry("se_changing", predicates={
            "family": "routing", "resource_coupling": [0.25, 0.50]})
        self.h.sbank.add(entry)
        self.h.index_sync.sync_entries(["se_changing"])
        # The claim text changes WITHOUT a re-index (simulating a drift).
        entry.applicability = ["a completely different set of notes now"]
        self.h.sbank.update(entry)
        out = self.h.recall(_task("t_q", description=REQ_A,
                                  resource_coupling=0.3))
        self.assertNotIn("se_changing",
                         [k["entry_id"]
                          for k in out["vector_recall"]["strategic_knowledge"]])
        self.assertEqual(out["vector_recall"]["stale_indexed"][LAYER_STRATEGIC],
                         1)


class TestDegradation(VectorCase):
    """Cases 6, 8, 9 and the whole degradation matrix."""

    def test_no_backend_degrades_to_profile_only(self):
        h = ORHarness(home=self.home)  # no injected backend, no env
        self.addCleanup(h.close)
        out = h.recall(_task("t_q", description=REQ_A))
        self.assertNotIn("vector_recall", out)
        self.assertEqual(out["degraded"]["path"], "profile_only")
        self.assertIn("no embedding backend", out["degraded"]["reason"])
        # The structural result is intact.
        self.assertIn("recommendations", out)

    def test_task_without_text_degrades(self):
        h = ORHarness(home=self.home, embedding=self.backend)
        self.addCleanup(h.close)
        out = h.recall({"task_id": "t_q", "family": "routing",
                        "annotations": {"coupling": {"resource_coupling": 0.3}}})
        self.assertEqual(out["degraded"]["path"], "profile_only")
        self.assertIn("no task text", out["degraded"]["reason"])
        self.assertNotIn("vector_recall", out)

    def test_missing_index_degrades_with_a_hint(self):
        out = self.h.recall(_task("t_q", description=REQ_A))
        self.assertIn("rebuild-index", out["degraded"]["reason"])

    def test_partial_index_runs_and_reports_the_missing_layer(self):
        """One layer indexed, the other absent: the channel RUNS on the layer
        that exists and names the missing one — a partial index is not a
        reason to fall back entirely."""
        self.solve(_task("t_a", description=REQ_A))
        self.h.rebuild_index(layer="execution")
        self.h.embedding_index.remove(LAYER_STRATEGIC, [])
        out = self.h.recall(_task("t_q", description=REQ_A))
        vector = out["vector_recall"]
        self.assertEqual(len(vector["execution_evidence"]), 1)
        self.assertEqual(vector["strategic_knowledge"], [])
        self.assertIn(LAYER_STRATEGIC, vector["degraded_layers"])
        self.assertNotIn("degraded", out)

    def test_model_change_degrades_without_mixing_spaces(self):
        """Case 6: an index built by ANOTHER model is refused wholesale —
        recall degrades to the profile path instead of comparing vectors
        from two different spaces."""
        self.solve(_task("t_a", description=REQ_A))
        path = index_dir_for(self.home) / f"{LAYER_EXECUTION}.embedding.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["model_id"], self.backend.model_id)
        payload["model_id"] = "some-other-model-v9"
        path.write_text(json.dumps(payload), encoding="utf-8")
        status = self.h.embedding_index.status(LAYER_EXECUTION)
        self.assertFalse(status["usable"])
        self.assertIn("embedding model changed", status["reason"])
        self.assertEqual(self.h.embedding_index.load(LAYER_EXECUTION), {})
        out = self.h.recall(_task("t_q", description=REQ_A))
        self.assertNotIn("vector_recall", out)
        self.assertIn("embedding model changed", out["degraded"]["reason"])

    def test_backend_error_degrades(self):
        class Exploding(LocalHashEmbeddingBackend):
            def embed_query(self, text):
                raise RuntimeError("endpoint down")

        h = ORHarness(home=self.home, embedding=Exploding())
        self.addCleanup(h.close)
        self.solve(_task("t_a", description=REQ_A))  # uses the good backend
        self.h.close()
        out = h.recall(_task("t_q", description=REQ_A))
        self.assertEqual(out["degraded"]["path"], "profile_only")
        self.assertIn("endpoint down", out["degraded"]["reason"])

    def test_recall_writes_nothing(self):
        """Case 8: read-only discipline — no new text rows, no index mtime
        change, no new index items."""
        self.solve(_task("t_a", description=REQ_A))
        index_path = index_dir_for(self.home) / f"{LAYER_EXECUTION}.embedding.json"
        texts_before = self.h.store.count_task_texts()
        mtime_before = index_path.stat().st_mtime_ns
        items_before = len(self.h.embedding_index.items(LAYER_EXECUTION))
        self.h.recall(_task("t_q", description=REQ_A))
        self.h.recall(_task("t_q", description=REQ_A))
        self.assertEqual(self.h.store.count_task_texts(), texts_before)
        self.assertEqual(index_path.stat().st_mtime_ns, mtime_before)
        self.assertEqual(len(self.h.embedding_index.items(LAYER_EXECUTION)),
                         items_before)

    def test_rebuild_dry_run_writes_nothing(self):
        """Case 9: --dry-run produces no index file at all."""
        isolated = tempfile.TemporaryDirectory()
        self.addCleanup(isolated.cleanup)
        h = ORHarness(home=isolated.name, embedding=self.backend)
        self.addCleanup(h.close)
        h.bank.append(self.make_record(task_id="t1", strategy_id="S01"))
        result = h.rebuild_index(dry_run=True)
        self.assertEqual(result["layers"][LAYER_EXECUTION]["would_index"], 0)
        self.assertFalse((Path(isolated.name) / "index").exists(),
                         "a dry run must not even create the index directory")

    def test_rebuild_populates_and_reports(self):
        self.solve(_task("t_a", description=REQ_A))
        result = self.h.rebuild_index()
        self.assertEqual(result["layers"][LAYER_EXECUTION]["items"], 1)
        self.assertEqual(result["backend"], self.backend.model_id)

    def test_rebuild_without_backend_refuses(self):
        h = ORHarness(home=self.home)
        self.addCleanup(h.close)
        with self.assertRaises(ValueError):
            h.rebuild_index()

    def test_doctor_reports_index_health_read_only(self):
        self.solve(_task("t_a", description=REQ_A))
        out = self.h.doctor()
        retrieval = out["retrieval_index"]
        self.assertTrue(retrieval["configured"])
        layer = retrieval["layers"][LAYER_EXECUTION]
        self.assertEqual(layer["count"], 1)
        self.assertEqual(layer["documents"], 1)
        self.assertEqual(layer["stale"], 0)


class TestIndexSyncDiscipline(VectorCase):
    """Case 11 + the write path: sync failure never blocks the fact."""

    def test_embedding_failure_defers_but_saves_the_fact(self):
        class Exploding(LocalHashEmbeddingBackend):
            def embed_documents(self, texts):
                raise RuntimeError("rate limited")

        h = ORHarness(home=self.home, embedding=Exploding())
        self.addCleanup(h.close)
        task = _task("t_sync", description=REQ_A)
        work = Path(self.home) / "ws_sync"
        work.mkdir(parents=True, exist_ok=True)
        script = work / "solve.py"
        script.write_text("import json\n"
                          "with open('result.json','w') as fh:\n"
                          "    json.dump({'status':'optimal','objective_value':1.0,"
                          "'objective_bound':1.0,'runtime_seconds':0.01}, fh)\n",
                          encoding="utf-8")
        record = h.execute(task, "S04", str(script), str(work), solver="highs")
        out = h.record(record)
        self.assertTrue(out["recorded"], "the fact must be saved regardless")
        self.assertEqual(out["index_sync"]["state"], "deferred")
        self.assertIn("rate limited", out["index_sync"]["reason"])
        self.assertIsNotNone(h.bank.get(record.execution_id))

    def test_upsert_refuses_a_foreign_model_index(self):
        self.solve(_task("t_a", description=REQ_A))
        path = index_dir_for(self.home) / f"{LAYER_EXECUTION}.embedding.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["model_id"] = "some-other-model-v9"
        path.write_text(json.dumps(payload), encoding="utf-8")
        out = self.h.index_sync.sync_execution(
            self.h.bank.all()[0])
        self.assertEqual(out["state"], "deferred")
        self.assertIn("rebuild-index", out["reason"])

    def test_induce_syncs_entries_and_dry_run_does_not(self):
        for task_id in ("t1", "t2"):
            self.solve(_task(task_id, description=REQ_A))
        result = self.h.induce(all_=True, verify={
            "purpose": "rule", "claim": "construction works here",
            "check": {"reference_status": "optimal"}})
        self.assertIn("index_sync", result)
        self.assertEqual(result["index_sync"]["state"], "synced")
        before = len(self.h.embedding_index.items(LAYER_STRATEGIC))
        dry = self.h.induce(all_=True, dry_run=True)
        self.assertNotIn("index_sync", dry)
        self.assertEqual(len(self.h.embedding_index.items(LAYER_STRATEGIC)),
                         before)


class TestDocumentBuilders(HarnessTestCase):
    """The two document builders use existing fields only."""

    def test_execution_document_leads_with_task_text(self):
        record = self.make_record(task_id="t1", strategy_id="S04")
        text = "solve the routing problem with two vehicles"
        doc = document_execution(record, text)
        self.assertTrue(doc.startswith(text))
        self.assertIn("strategy S04", doc)
        self.assertIn("outcome optimal", doc)

    def test_entry_document_is_built_from_existing_fields(self):
        entry = _entry(applicability=["notes here"])
        entry.actions = ["split along the resource axis"]
        doc = document_entry(entry)
        self.assertIn("notes here", doc)
        self.assertIn("strategy S04", doc)
        self.assertIn("family routing", doc)
        self.assertIn("split along the resource axis", doc)

    def test_entry_document_carries_no_directory_content(self):
        """The document is composed of text the ENTRY already has.

        Nothing is pulled from a built-in directory: two entries that share a
        strategy id but record different content produce different documents,
        and an entry that records nothing produces a document without any
        method description at all.
        """
        bare = _entry(entry_id="se_bare")
        doc = document_entry(bare)
        self.assertNotIn("monolithic", doc)
        self.assertNotIn("MILP in one pass", doc)
        # Same id, different recorded content -> different document.
        other = _entry(entry_id="se_other", applicability=["a different note"])
        self.assertNotEqual(document_digest(doc),
                            document_digest(document_entry(other)))

    def test_entry_document_carries_relation_claims(self):
        """A relation's claim is part of what the knowledge says, so a text
        search must be able to find it — and editing it must move the digest
        (which is what invalidates the old vector)."""
        entry = _entry()
        entry.relations = [{
            "relation_id": "rel_1",
            "claim": "keep the cross-period state",
            "evidence": [{"execution_id": "ex_1", "role": "preserved"}],
            "verification": {"state": "verified"},
        }]
        doc = document_entry(entry)
        self.assertIn("keep the cross-period state", doc)
        edited = document_entry(entry)
        self.assertEqual(document_digest(doc), document_digest(edited))
        entry.relations[0]["claim"] = "a revised claim"
        revised = document_entry(entry)
        self.assertNotEqual(document_digest(doc), document_digest(revised))

    def test_readable_predicates_covers_the_unknown_bucket(self):
        text = readable_predicates({"family": "vrp", "route_complexity": "unknown",
                                    "temporal_coupling": [0.5, 0.75]})
        self.assertIn("family vrp", text)
        self.assertIn("route complexity unknown", text)
        self.assertIn("temporal coupling in [0.50, 0.75]", text)

    def test_document_digest_is_stable(self):
        self.assertEqual(document_digest("abc"), document_digest("abc"))
        self.assertNotEqual(document_digest("abc"), document_digest("abd"))


class TestBackendSelection(unittest.TestCase):
    """`auto` must never silently substitute a lexical hash for semantics."""

    def setUp(self):
        import os
        self._saved = {k: os.environ.get(k) for k in EMBEDDING_ENV_KEYS}
        for key in self._saved:
            os.environ.pop(key, None)
        self.addCleanup(self._restore)

    def _restore(self):
        import os
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_auto_without_config_returns_none(self):
        self.assertIsNone(create_embedding_backend())
        self.assertIsNone(create_embedding_backend("auto"))

    def test_auto_uses_remote_when_fully_configured(self):
        import os
        os.environ.update({"OR_EMBEDDING_BASE_URL": "https://example.test/v1",
                           "OR_EMBEDDING_MODEL": "embed-1",
                           "OR_EMBEDDING_API_KEY": "k"})
        backend = create_embedding_backend()
        self.assertIsNotNone(backend)
        self.assertEqual(backend.model_id, "embed-1")
        described = backend.describe()
        self.assertEqual(described["source"], "remote")
        self.assertNotIn("k", json.dumps(described))

    def test_explicit_local_hashing_still_available(self):
        self.assertIsInstance(create_embedding_backend("local-hashing"),
                              LocalHashEmbeddingBackend)

    def test_unknown_name_is_rejected(self):
        with self.assertRaises(ValueError):
            create_embedding_backend("nonsense")


class TestVectorRecallInternals(VectorCase):
    def test_recall_vectors_raises_when_no_text(self):
        with self.assertRaises(VectorRecallUnavailable):
            recall_vectors(self.h, "   ")

    def test_recall_vectors_raises_without_backend(self):
        h = ORHarness(home=self.home)
        self.addCleanup(h.close)
        with self.assertRaises(VectorRecallUnavailable):
            recall_vectors(h, "some task text")

    def test_task_text_reader_is_stable_across_paths(self):
        """The capture path and the snapshot-recovery path must produce the
        SAME text for the same task (one reader, two callers)."""
        from or_harness.world_model.state import _task_payload
        task = _task("t1", description=REQ_A, resource_coupling=0.9)
        self.assertEqual(task_text(task),
                         task_text_from_payload(_task_payload(task)))


def run_orx(home, *argv):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO / "src")
    env["OR_EMBEDDING_BACKEND"] = "local-hashing"
    for key in ("OR_EMBEDDING_BASE_URL", "OR_EMBEDDING_MODEL",
                "OR_EMBEDDING_API_KEY"):
        env.pop(key, None)
    return subprocess.run(
        [sys.executable, "-m", "or_harness.cli", "--home", home, *argv],
        capture_output=True, text=True, env=env, cwd=str(REPO))


def run_orx_without_backend(home, *argv):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO / "src")
    for key in EMBEDDING_ENV_KEYS:
        env.pop(key, None)
    return subprocess.run(
        [sys.executable, "-m", "or_harness.cli", "--home", home, *argv],
        capture_output=True, text=True, env=env, cwd=str(REPO))


class TestVectorRecallCLI(HarnessTestCase):
    """The CLI surface: recall transparency, rebuild-index, inspect, doctor."""

    def setUp(self):
        super().setUp()
        self.work = Path(self.home) / "ws"
        self.work.mkdir(parents=True, exist_ok=True)
        self.task = _task("t1", description=REQ_A)
        self.task_path = self.work / "task.json"
        self.task_path.write_text(json.dumps(self.task), encoding="utf-8")
        (self.work / "solve.py").write_text(
            "import json\n"
            "with open('result.json','w') as fh:\n"
            "    json.dump({'status':'optimal','objective_value':1.0,"
            "'objective_bound':1.0,'runtime_seconds':0.01}, fh)\n",
            encoding="utf-8")

    def _execute_and_record(self):
        proc = run_orx(self.home, "execute", "--task", str(self.task_path),
                       "--strategy", "S04", "--code",
                       str(self.work / "solve.py"), "--workspace",
                       str(self.work), "--solver", "highs")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        execution = json.loads(proc.stdout)["result"]["execution"]
        payload = self.work / "execution.json"
        payload.write_text(json.dumps(execution), encoding="utf-8")
        proc = run_orx(self.home, "record", "--execution", str(payload))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        return json.loads(proc.stdout)["result"]

    def test_recall_reports_the_text_channel(self):
        recorded = self._execute_and_record()
        self.assertEqual(recorded["index_sync"]["state"], "synced")
        proc = run_orx(self.home, "recall", "--task", str(self.task_path))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        out = json.loads(proc.stdout)
        self.assertIn("vector_recall", out["result"])
        hits = out["result"]["vector_recall"]["execution_evidence"]
        self.assertEqual([h["execution_id"] for h in hits],
                         [recorded["execution_id"]])
        self.assertIn("Text similarity", out["summary"])

    def test_recall_degrades_without_a_backend(self):
        proc = run_orx_without_backend(self.home, "recall", "--task",
                                       str(self.task_path))
        out = json.loads(proc.stdout)
        self.assertNotIn("vector_recall", out["result"])
        self.assertEqual(out["result"]["degraded"]["path"], "profile_only")
        self.assertIn("no embedding backend", out["result"]["degraded"]["reason"])

    def test_rebuild_index_dry_run_then_real(self):
        self._execute_and_record()
        index_path = index_dir_for(self.home) / f"{LAYER_EXECUTION}.embedding.json"
        before = index_path.stat().st_mtime_ns
        proc = run_orx(self.home, "rebuild-index", "--dry-run")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        out = json.loads(proc.stdout)
        self.assertTrue(out["result"]["dry_run"])
        self.assertEqual(index_path.stat().st_mtime_ns, before)
        proc = run_orx(self.home, "rebuild-index", "--layer", "execution")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        out = json.loads(proc.stdout)
        self.assertEqual(out["result"]["layers"][LAYER_EXECUTION]["items"], 1)

    def test_inspect_texts_bank(self):
        self._execute_and_record()
        proc = run_orx(self.home, "inspect", "--bank", "texts",
                       "--task", "t1")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        out = json.loads(proc.stdout)
        self.assertEqual(out["result"]["count"], 1)
        self.assertIn("distribution centre",
                      out["result"]["task_texts"][0]["text"])

    def test_doctor_reports_retrieval_index(self):
        self._execute_and_record()
        proc = run_orx(self.home, "doctor")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        out = json.loads(proc.stdout)
        retrieval = out["result"]["retrieval_index"]
        self.assertTrue(retrieval["configured"])
        self.assertEqual(
            retrieval["layers"][LAYER_EXECUTION]["count"], 1)
        self.assertIn("Text retrieval", out["summary"])

    def test_rebuild_index_without_backend_fails_honestly(self):
        proc = run_orx_without_backend(self.home, "rebuild-index")
        self.assertEqual(proc.returncode, 2)
        self.assertIn("no embedding backend configured", proc.stdout)