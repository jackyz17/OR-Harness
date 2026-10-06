"""The Execution Evidence Bank is a BOUNDED window of complete episodes.

What this file asserts, one behaviour per test class:

1. **Bounded** — after the bound is exceeded the bank stays bounded; the
   oldest COMPLETE episodes leave, and the whole episode leaves at once so a
   contrast/repair chain is never split.
2. **Protected** — episodes inside the calibration window, episodes awaiting
   a late task verdict within the grace period, and young UNCLOSED episodes
   are never evicted.
3. **Idempotent and clean** — a repeated pass removes nothing more, and the
   derived vectors / unreferenced task-text versions go with the rows.
4. **Dry run writes nothing** — no row, no vector, no text.
5. **Eviction is not withdrawal** — a window-evicted source does NOT revoke
   the stored evaluation (a live re-derivation keeps it FINAL), while an
   ``exclude`` still does.
6. **Knowledge survives** — a verified entry learned from now-evicted
   evidence is still recalled and can still be revised from current evidence.
7. **Surface** — the API and CLI entry points reach the same pass.
"""
import io
import json
import os
import sys
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tests.harness.helpers import HarnessTestCase  # noqa: E402

from or_harness.api import ORHarness  # noqa: E402
from or_harness.strategy.embedding_index import (  # noqa: E402
    LAYER_EXECUTION, LocalHashEmbeddingBackend,
)
from or_harness.world_model.episode_closeout import (  # noqa: E402
    CalibrationPolicy, EvidenceWindowPolicy, enforce_evidence_window,
)
from or_harness.world_model.provider import WorldModelProvider  # noqa: E402

EMBEDDING_ENV_KEYS = ("OR_EMBEDDING_BACKEND", "OR_EMBEDDING_BASE_URL",
                      "OR_EMBEDDING_MODEL", "OR_EMBEDDING_API_KEY")

REQ_TEXT = ("A distribution centre must be loaded before the delivery window "
            "opens; demand 100 units may not be deferred.")


def _task(task_id="t1", **coupling):
    values = {"resource_coupling": 0.3, "temporal_coupling": 0.1,
              "route_complexity": 0.2}
    values.update(coupling)
    return {"task_id": task_id, "family": "routing",
            "description": REQ_TEXT,
            "spec": {"n_vars": 100, "n_constraints": 50, "n_int_vars": 100},
            "annotations": {"coupling": {**values, "semantic_coupling": 0.5}}}


PAYLOAD = {
    "benefit": {"kind": "solution_quality",
                "metric": "normalized_objective_gap", "unit": "1-gap",
                "value": 0.8,
                "baseline": {"kind": "conditional_stats", "value": 0.7}},
    "cost": {"llm_tokens": 1200, "solver_runtime_s": 3.0},
    "risk": {"events": [{"event": "timeout", "probability": 0.2}]},
}


class StubProvider(WorldModelProvider):
    """A stub provider returning a fixed payload."""

    name = "stub-window"

    def predict(self, request, timeout_s=None):
        return {"payload": PAYLOAD,
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
                "error": None, "latency_s": 0.02}


class EvidenceWindowCase(HarnessTestCase):

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

    def enforce(self, episodes, *, open_grace_days=30.0, dry_run=False,
                calibration_window=0):
        """Run the evidence pass with BOTH scopes explicit, so a test states
        exactly what protects what."""
        return enforce_evidence_window(
            self.h,
            policy=EvidenceWindowPolicy(window_episodes=episodes,
                                        open_grace_days=open_grace_days),
            calibration_policy=CalibrationPolicy(
                window=calibration_window, late_check_grace_days=0.0),
            dry_run=dry_run)

    def solve(self, task, strategy="S04", episode_id="ep1",
              objective=100.0, tag=None):
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

    def close_with_evaluation(self, task_id, episode_id, strategy="S04"):
        """Solve, predict, bind and close ONE episode (a real sample)."""
        task = _task(task_id)
        prediction = self.h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": strategy}, episode_id)
        record = self.solve(task, strategy=strategy, episode_id=episode_id)
        self.h.bind_strategy_outcome(prediction.prediction_id,
                                     record.action_id)
        return self.h.close_episode(task_id, episode_id), record

    def make_episodes(self, count, task_prefix="t"):
        """Close ``count`` independent episodes across distinct tasks."""
        records = []
        for i in range(count):
            _, record = self.close_with_evaluation(f"{task_prefix}{i}", "ep1")
            records.append(record)
        return records


# ---------------------------------------------------------------------------
# 1. bounded
# ---------------------------------------------------------------------------


class TestBounded(EvidenceWindowCase):

    def test_exceeding_the_bound_evicts_oldest_and_stays_bounded(self):
        self.make_episodes(5)
        self.assertEqual(self.h.bank.count(), 5)
        result = self.enforce(2)
        self.assertEqual(result["n_evicted"], 3)
        self.assertEqual(self.h.bank.count(), 2,
                         "the bank must stay bounded after the pass")
        # The two survivors are the newest two episodes.
        self.assertEqual({r.task_id for r in self.h.bank.all()},
                         {"t3", "t4"})
        # The eviction set is reported oldest-first.
        self.assertEqual([e["task_id"] for e in result["evicted"]],
                         ["t0", "t1", "t2"])

    def test_whole_episode_leaves_at_once(self):
        """A contrast/repair chain is never split: both attempts of one
        episode leave together, in the same pass."""
        task = _task("t0")
        first = self.solve(task, strategy="S04", episode_id="ep1",
                           objective=999.0, tag="t0a")
        second = self.solve(task, strategy="S04", episode_id="ep1",
                            objective=100.0, tag="t0b")
        self.assertNotEqual(first.execution_id, second.execution_id)
        self.h.close_episode("t0", "ep1")
        self.close_with_evaluation("u0", "ep1")
        result = self.enforce(1)
        evicted_ids = set()
        for entry in result["evicted"]:
            evicted_ids.update(entry["execution_ids"])
        # BOTH t0 attempts left; NEITHER survived alone.
        self.assertIn(first.execution_id, evicted_ids)
        self.assertIn(second.execution_id, evicted_ids)
        self.assertIsNone(self.h.bank.get(first.execution_id))
        self.assertIsNone(self.h.bank.get(second.execution_id))

    def test_bound_holds_across_repeated_passes(self):
        self.make_episodes(6)
        self.enforce(3)
        self.assertEqual(self.h.bank.count(), 3)
        again = self.enforce(3)
        self.assertEqual(again["n_evicted"], 0)
        self.assertEqual(self.h.bank.count(), 3)


# ---------------------------------------------------------------------------
# 2. protected
# ---------------------------------------------------------------------------


class TestProtected(EvidenceWindowCase):

    def test_calibration_window_episodes_are_never_evicted(self):
        self.make_episodes(5)
        # A calibration window of 5 protects every episode even though the
        # evidence window is 1.
        result = self.enforce(1, calibration_window=5)
        self.assertEqual(result["n_evicted"], 0)
        self.assertEqual(self.h.bank.count(), 5)
        self.assertGreaterEqual(result["n_protected"], 5)

    def test_unclosed_episode_is_not_evicted(self):
        # An open episode's executions must survive the count bound.
        self.solve(_task("open1"), episode_id="ep1", tag="open1")
        self.make_episodes(4, task_prefix="z")
        result = self.enforce(1)
        self.assertIn("open1", {r.task_id for r in self.h.bank.all()})
        self.assertEqual(result["n_evicted_unclosed"], 0)

    def test_aged_unclosed_episode_is_evicted_and_reported(self):
        record = self.solve(_task("aged"), episode_id="ep1", tag="aged")
        # Backdate the lone execution far beyond the open grace period.
        raw = self.store.loads(
            self.store.conn.execute(
                "SELECT payload FROM executions WHERE execution_id=?",
                (record.execution_id,)).fetchone()["payload"])
        raw["created_at"] = time.time() - 400 * 86400.0
        with self.store.transaction() as conn:
            conn.execute("UPDATE executions SET payload=? WHERE execution_id=?",
                         (self.store.dumps(raw), record.execution_id))
        result = self.enforce(800, open_grace_days=30.0)
        self.assertEqual(result["n_evicted_unclosed"], 1)
        self.assertIsNone(self.h.bank.get(record.execution_id))


# ---------------------------------------------------------------------------
# 3. idempotent and clean
# ---------------------------------------------------------------------------


class TestCleanup(EvidenceWindowCase):

    def test_evicted_vectors_and_task_texts_are_cleaned(self):
        records = self.make_episodes(4)
        doomed = records[0]
        indexed_before = {str(i.get("id")) for i in
                          self.h.embedding_index.items(LAYER_EXECUTION)}
        self.assertIn(doomed.execution_id, indexed_before)
        self.assertGreater(self.store.count_task_texts(), 0)
        self.enforce(1)
        indexed_after = {str(i.get("id")) for i in
                         self.h.embedding_index.items(LAYER_EXECUTION)}
        self.assertNotIn(doomed.execution_id, indexed_after)
        # Only task texts still referenced by a surviving fact remain.
        surviving = {(r.task_id, r.task_text_digest)
                     for r in self.h.bank.all() if r.task_text_digest}
        for key in self.store.all_task_text_keys():
            self.assertIn((key["task_id"], key["text_digest"]), surviving)

    def test_repeated_pass_is_idempotent(self):
        self.make_episodes(4)
        first = self.enforce(1)
        second = self.enforce(1)
        self.assertGreater(first["n_evicted"], 0)
        self.assertEqual(second["n_evicted"], 0)
        self.assertEqual(second["removed_executions"], 0)

    def test_source_files_are_never_touched(self):
        self.make_episodes(3)
        script = Path(self.home) / "ws_t0_S04_ep1" / "solve.py"
        self.assertTrue(script.exists())
        self.enforce(1)
        self.assertTrue(script.exists(),
                        "the window pass must never delete solve sources")


# ---------------------------------------------------------------------------
# 4. dry run
# ---------------------------------------------------------------------------


class TestDryRun(EvidenceWindowCase):

    def test_dry_run_writes_nothing(self):
        self.make_episodes(4)
        bank_before = self.h.bank.count()
        texts_before = self.store.count_task_texts()
        indexed_before = {str(i.get("id")) for i in
                          self.h.embedding_index.items(LAYER_EXECUTION)}
        result = self.enforce(1, dry_run=True)
        self.assertTrue(result["dry_run"])
        self.assertEqual(result["n_evicted"], 3)
        self.assertEqual(self.h.bank.count(), bank_before)
        self.assertEqual(self.store.count_task_texts(), texts_before)
        self.assertEqual(
            {str(i.get("id")) for i in self.h.embedding_index.items(LAYER_EXECUTION)},
            indexed_before)


# ---------------------------------------------------------------------------
# 5. eviction is not withdrawal
# ---------------------------------------------------------------------------


class TestEvictionIsNotWithdrawal(EvidenceWindowCase):

    def test_evicted_source_does_not_revoke_the_stored_evaluation(self):
        task = _task("t1")
        prediction = self.h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S04"}, "ep1")
        record = self.solve(task, strategy="S04")
        self.h.bind_strategy_outcome(prediction.prediction_id,
                                     record.action_id)
        first = self.h.close_episode("t1", "ep1")
        stored_id = first["evaluations"][0]["evaluation_id"]
        # Evict the episode's executions out from under the evaluation.
        self.h.bank.delete_episode_executions([record.execution_id])
        self.assertIsNone(self.h.bank.get(record.execution_id))
        summary = self.h.calibration_summary(rebuild=True)
        # The stored evaluation is NOT reported as withdrawn — the window is
        # a cut-off, not a retraction.
        corrections = [c for c in (summary.get("validity_corrections") or [])
                       if c.get("evaluation_id") == stored_id]
        self.assertEqual(corrections, [])
        self.assertNotIn("validity_corrected", summary.get("exclusions", {}))

    def test_excluded_source_IS_a_withdrawal(self):
        task = _task("t1")
        prediction = self.h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S04"}, "ep1")
        record = self.solve(task, strategy="S04")
        self.h.bind_strategy_outcome(prediction.prediction_id,
                                     record.action_id)
        self.h.close_episode("t1", "ep1")
        # An EXCLUDE is the explicit correction: the fact is withdrawn.
        self.h.exclude_execution(record.execution_id, "wrong answer")
        summary = self.h.calibration_summary(rebuild=True)
        self.assertEqual(summary["exclusions"].get("validity_corrected"), 1)
        withdrawn = [c for c in (summary.get("validity_corrections") or [])]
        self.assertTrue(withdrawn)
        self.assertTrue(any(c.get("evaluated_execution_ids") for c in withdrawn))


# ---------------------------------------------------------------------------
# 6. knowledge survives
# ---------------------------------------------------------------------------


class TestKnowledgeSurvives(EvidenceWindowCase):

    def test_verified_knowledge_is_recalled_and_revisable_after_eviction(self):
        # Two independent tasks -> a verified method claim.
        records = [self.solve(_task(f"k{i}"), strategy="S04",
                              episode_id="ep1", tag=f"k{i}")
                   for i in range(2)]
        claim = {
            "subject": "S04",
            "claim": "S04 solves routing tasks with optimal status",
            "evidence": [{"execution_id": r.execution_id, "role": "e"}
                         for r in records]}
        self.h.induce(
            relations=[claim],
            verify={"claim": claim["claim"], "check": {"assertions": [
                {"kind": "status", "roles": ["e"], "status": "optimal"}]}})
        entry = next(e for e in self.h.sbank.list()
                     if e.strategy_id == "S04")
        self.assertEqual(entry.verification.get("state"), "verified")
        # Evict everything the entry was learned from: no registry rows, so
        # only the open-episode path applies (grace 0 evicts the aged ones).
        self.enforce(0, open_grace_days=0.0)
        self.assertEqual(self.h.bank.count(), 0)
        # The entry is still there (self-contained) and still publishable.
        survived = self.h.sbank.get(entry.entry_id)
        self.assertIsNotNone(survived)
        self.assertEqual(survived.verification.get("state"), "verified")
        recall = self.h.recall(_task("k3"))
        entry_ids = {item.get("evidence_refs", [None])[0]
                     for item in recall.get("recommendations", [])}
        self.assertIn(entry.entry_id, entry_ids)
        # A NEW piece of evidence can still revise it.
        new = self.solve(_task("k9"), strategy="S04", episode_id="ep1",
                         tag="k9")
        revised = self.h.induce(relations=[{
            "subject": "S04",
            "claim": "S04 solves routing tasks with optimal status",
            "evidence": [{"execution_id": new.execution_id, "role": "e"}]}])
        self.assertTrue(revised["relations"][0]["saved"])


# ---------------------------------------------------------------------------
# 7. surface
# ---------------------------------------------------------------------------


class TestSurface(EvidenceWindowCase):

    def test_api_method_reaches_the_pass(self):
        self.make_episodes(3)
        # The API method uses the CALIBRATION policy as protection, so with
        # the default calibration window the episodes are protected.
        result = self.h.enforce_window(window_episodes=0, dry_run=True)
        self.assertTrue(result["dry_run"])
        self.assertEqual(result["n_evicted"], 0)
        self.assertGreaterEqual(result["n_protected"], 3)

    def test_cli_command_runs(self):
        from or_harness import cli
        self.make_episodes(3)
        with mock.patch.object(sys, "argv",
                               ["orx", "--home", self.home,
                                "enforce-window", "--window-episodes", "0",
                                "--dry-run"]):
            buffer = io.StringIO()
            with mock.patch("sys.stdout", buffer):
                code = cli.main()
        self.assertEqual(code, 0)
        payload = json.loads(buffer.getvalue())
        self.assertTrue(payload["result"]["dry_run"])


if __name__ == "__main__":
    unittest.main()
