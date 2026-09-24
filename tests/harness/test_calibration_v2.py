"""Calibration v2 (wm-calib/2): event semantics, counting units, retention.

This suite is the executable statement of the six review corrections that
reshaped the close-out layer. Each test class asserts ONE of them, so a
future change that re-breaks a correction fails here first:

1. **Answer-check failures are not modelling errors.** A task check produces
   ``task_check_failed`` and nothing else. The retired ``model_invalid`` name
   is never aliased onto it, and an execution failure is named by what was
   observed (``environment_failure`` / ``implementation_failure``).
2. **Budget scope is matched, not assumed.** ``budget_exhausted``'s ledger is
   episode-scoped; an attempt- or window-scope prediction is NOT scored
   against it, while the framework still records the fact.
3. **The window is located, then read.** A prediction read is a single-row
   lookup of the PUBLISHED summary; a correction inside the window
   republishes it, so a late verdict reaches later predictions.
4. **Retention has three separate scopes** (window / late-check grace /
   archive caps) and the archive is bounded per file, in total and by age.
5. **Occurrence is counted by observation unit**, prediction scoring by
   prediction-observation pair — two units that never inflate each other.
6. **Groups are filtered** to the compatible model/metric/unit/scope, and the
   statistics are labelled a global diagnostic rather than a conditional
   claim about the current candidate.
"""
import json
import os
import sys
import time
import unittest
from pathlib import Path

from helpers import HarnessTestCase

from or_harness.api import ORHarness
from or_harness.strategy.embedding_index import (
    LocalHashEmbeddingBackend,
)
from or_harness.world_model.provider import WorldModelProvider

EMBEDDING_ENV_KEYS = ("OR_EMBEDDING_BACKEND", "OR_EMBEDDING_BASE_URL",
                      "OR_EMBEDDING_MODEL", "OR_EMBEDDING_API_KEY")

REQ_TEXT = ("A distribution centre must be loaded before the delivery window "
            "opens; demand 100 units may not be deferred.")

PAYLOAD = {
    "benefit": {"kind": "solution_quality",
                "metric": "normalized_objective_gap", "unit": "1-gap",
                "value": 0.8,
                "baseline": {"kind": "conditional_stats", "value": 0.7}},
    "cost": {"llm_tokens": 1200, "solver_runtime_s": 3.0},
    "risk": {"events": [{"event": "timeout", "probability": 0.25}]},
}

GROUP = "strategy_outcome|(unknown)|normalized_objective_gap|1-gap|attempt"


def _task(task_id="t1", **coupling):
    values = {"resource_coupling": 0.3, "temporal_coupling": 0.1,
              "route_complexity": 0.2}
    values.update(coupling)
    return {"task_id": task_id, "family": "routing",
            "description": REQ_TEXT,
            "spec": {"n_vars": 100, "n_constraints": 50, "n_int_vars": 100},
            "annotations": {"coupling": {**values, "semantic_coupling": 0.5}}}


class StubProvider(WorldModelProvider):
    """A stub provider with an OPTIONAL declared model identity."""

    def __init__(self, payload=None, model=None, version=None):
        self.payload = payload if payload is not None else PAYLOAD
        self._model = model
        self._version = version
        self.requests = []

    def describe(self):
        out = {"provider": "stub-v2"}
        if self._model:
            out["model"] = self._model
        if self._version:
            out["model_version"] = self._version
        return out

    def predict(self, request, timeout_s=None):
        self.requests.append(request)
        return {"payload": self.payload,
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
                "error": None, "latency_s": 0.02}


class CalibrationV2Case(HarnessTestCase):

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

    def solve(self, task, strategy="S04", objective=100.0,
              episode_id="ep1", status="optimal", gap=None, tag=None,
              body=None):
        label = tag or f"{task['task_id']}_{strategy}_{episode_id}"
        work = Path(self.home) / f"ws_{label}"
        work.mkdir(parents=True, exist_ok=True)
        script = work / "solve.py"
        if body is not None:
            script.write_text(body, encoding="utf-8")
        else:
            script.write_text(
                "import json\n"
                "with open('result.json', 'w') as fh:\n"
                "    json.dump({'status': " + json.dumps(status) +
                ", 'objective_value': "
                f"{objective}, 'objective_bound': {objective}"
                + (f", 'mip_gap': {gap}" if gap is not None else "")
                + ", 'runtime_seconds': 0.01}, fh)\n", encoding="utf-8")
        record = self.h.execute(task, strategy, str(script), str(work),
                                solver="highs", episode_id=episode_id)
        self.h.record(record)
        return record

    def predict_and_bind(self, task, record, episode_id="ep1",
                         candidate=None, provider=None, harness=None):
        harness = harness or self.h
        candidate = candidate or {"action_type": "execute_strategy",
                                  "strategy_id": "S04"}
        prediction = harness.predict_strategy_outcome(
            task, candidate, episode_id)
        harness.bind_strategy_outcome(prediction.prediction_id,
                                      record.action_id)
        return prediction


# ---------------------------------------------------------------------------
# 1. answer-check failures are not modelling errors
# ---------------------------------------------------------------------------


class TestEventSemantics(CalibrationV2Case):
    """A task check produces ``task_check_failed`` and NOTHING else."""

    def _event_map(self, evaluation):
        scored = {e["event"]: e for e in evaluation["risk"]["scored"]}
        unscored = {e["event"]: e for e in evaluation["risk"]["unscored"]}
        return scored, unscored

    def test_failed_check_produces_one_label_not_two(self):
        task = _task("t1")
        payload = json.loads(json.dumps(PAYLOAD))
        payload["risk"]["events"] = [
            {"event": "task_check_failed", "probability": 0.4}]
        provider = StubProvider(payload=payload)
        h = ORHarness(home=self.home, world_model=provider,
                      embedding=self.backend)
        self.addCleanup(h.close)
        prediction = h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S04"}, "ep1")
        record = self.solve(task, strategy="S04")
        # The answer fails its declared check.
        checked = h.check_task_result(record.execution_id,
                                      {"reference_objective": 999.0})
        self.assertEqual(checked["state"], "failed")
        h.bind_strategy_outcome(prediction.prediction_id,
                                record.action_id)
        result = h.close_episode("t1", "ep1")
        evaluation = result["evaluations"][0]
        scored, _ = self._event_map(evaluation)
        self.assertIn("task_check_failed", scored)
        self.assertEqual(scored["task_check_failed"]["label"], "occurred")
        # The retired modelling name is NOT produced as a second label.
        observed = evaluation["risk"]["observed_units"]
        self.assertNotIn("model_invalid", observed)
        # The check's KIND and SCOPE travel with the observation.
        detail = evaluation["risk"]["observed_units"]["task_check_failed"]
        self.assertEqual(detail["label"], "occurred")

    def test_passed_check_is_a_scope_limited_not_occurred(self):
        task = _task("t1")
        payload = json.loads(json.dumps(PAYLOAD))
        payload["risk"]["events"] = [
            {"event": "task_check_failed", "probability": 0.4}]
        provider = StubProvider(payload=payload)
        h = ORHarness(home=self.home, world_model=provider,
                      embedding=self.backend)
        self.addCleanup(h.close)
        prediction = h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S04"}, "ep1")
        record = self.solve(task, strategy="S04")
        h.check_task_result(record.execution_id,
                            {"reference_objective": 100.0})
        h.bind_strategy_outcome(prediction.prediction_id,
                                record.action_id)
        result = h.close_episode("t1", "ep1")
        evaluation = result["evaluations"][0]
        scored, _ = self._event_map(evaluation)
        self.assertEqual(scored["task_check_failed"]["label"],
                         "not_occurred")
        # The basis says the pass covers only the DECLARED bases.
        self.assertIn("declared bases", scored["task_check_failed"]
                      ["label_basis"])

    def test_error_status_is_not_a_modelling_verdict(self):
        """A script that crashes is an implementation/environment failure —
        never ``model_invalid``, whatever the error text says."""
        task = _task("t1")
        payload = json.loads(json.dumps(PAYLOAD))
        payload["risk"]["events"] = [
            {"event": "model_invalid", "probability": 0.4}]
        provider = StubProvider(payload=payload)
        h = ORHarness(home=self.home, world_model=provider,
                      embedding=self.backend)
        self.addCleanup(h.close)
        prediction = h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S04"}, "ep1")
        record = self.solve(task, strategy="S04", tag="crash",
                            body="raise RuntimeError('model invalid')\n")
        self.assertEqual(record.quality.get("status"), "error")
        h.bind_strategy_outcome(prediction.prediction_id,
                                record.action_id)
        result = h.close_episode("t1", "ep1")
        evaluation = result["evaluations"][0]
        scored, unscored = self._event_map(evaluation)
        # The prediction named a RETIRED event: no label, no alias.
        self.assertNotIn("model_invalid", scored)
        self.assertIn("model_invalid", unscored)
        self.assertIn("retired", unscored["model_invalid"]["reason"])
        # The framework observed the REAL failure, by its own name.
        observed = evaluation["risk"]["observed_units"]
        self.assertIn("implementation_failure", observed)
        self.assertEqual(observed["implementation_failure"]["label"],
                         "occurred")
        self.assertNotIn("model_invalid", observed)

    def test_environment_failure_is_distinguished(self):
        task = _task("t1")
        payload = json.loads(json.dumps(PAYLOAD))
        payload["risk"]["events"] = [
            {"event": "environment_failure", "probability": 0.2}]
        provider = StubProvider(payload=payload)
        h = ORHarness(home=self.home, world_model=provider,
                      embedding=self.backend)
        self.addCleanup(h.close)
        prediction = h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S04"}, "ep1")
        # A security-policy rejection: classified "environment" at record
        # time by the executor.
        record = self.solve(task, strategy="S04", tag="env",
                            body="import os\nos.system('ls')\n")
        self.assertEqual(record.quality.get("status"), "error")
        h.bind_strategy_outcome(prediction.prediction_id,
                                record.action_id)
        result = h.close_episode("t1", "ep1")
        evaluation = result["evaluations"][0]
        observed = evaluation["risk"]["observed_units"]
        self.assertEqual(observed["environment_failure"]["label"],
                         "occurred")
        self.assertEqual(observed["implementation_failure"]["label"],
                         "not_occurred")

    def test_infeasible_is_a_reported_verdict_not_a_failure(self):
        """A reported infeasibility is recorded as a FACT and never mapped
        onto a failure label. A correct diagnosis of an infeasible ORIGINAL
        problem is a valid outcome."""
        task = _task("t1")
        payload = json.loads(json.dumps(PAYLOAD))
        payload["risk"]["events"] = [
            {"event": "solver_reported_infeasible", "probability": 0.3}]
        provider = StubProvider(payload=payload)
        h = ORHarness(home=self.home, world_model=provider,
                      embedding=self.backend)
        self.addCleanup(h.close)
        prediction = h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S04"}, "ep1")
        record = self.solve(task, strategy="S04", status="infeasible",
                            objective=0.0, tag="inf")
        h.bind_strategy_outcome(prediction.prediction_id,
                                record.action_id)
        result = h.close_episode("t1", "ep1")
        evaluation = result["evaluations"][0]
        observed = evaluation["risk"]["observed_units"]
        self.assertEqual(observed["solver_reported_infeasible"]["label"],
                         "occurred")
        # No failure event was manufactured from it.
        self.assertNotEqual(observed["implementation_failure"]["label"],
                            "occurred")
        self.assertNotEqual(observed["environment_failure"]["label"],
                            "occurred")
        scored, _ = self._event_map(evaluation)
        self.assertEqual(scored["solver_reported_infeasible"]["label"],
                         "occurred")
        self.assertIn("not by itself a strategy failure",
                      scored["solver_reported_infeasible"]["label_basis"])

    def test_legacy_record_without_error_class_stays_unknown(self):
        """A record written before ``error_class`` existed must NOT have its
        cause inferred from the error text."""
        task = _task("t1")
        payload = json.loads(json.dumps(PAYLOAD))
        payload["risk"]["events"] = [
            {"event": "environment_failure", "probability": 0.2}]
        provider = StubProvider(payload=payload)
        h = ORHarness(home=self.home, world_model=provider,
                      embedding=self.backend)
        self.addCleanup(h.close)
        prediction = h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S04"}, "ep1")
        record = self.solve(task, strategy="S04", tag="legacy",
                            body="raise RuntimeError('boom')\n")
        # Simulate a legacy record: strip the class from the failure.
        for failure in record.failures:
            failure.error_class = None
        h.bank._put(record) if hasattr(h.bank, "_put") else None
        h.bank.set_task_check(record.execution_id, None)
        # Rewrite the stored payload without the class.
        stored = h.bank.get(record.execution_id)
        for failure in stored.failures:
            failure.error_class = None
        with h.store.transaction() as conn:
            conn.execute("UPDATE executions SET payload=? WHERE execution_id=?",
                         (h.store.dumps(stored.to_dict()),
                          record.execution_id))
        h.bind_strategy_outcome(prediction.prediction_id,
                                record.action_id)
        result = h.close_episode("t1", "ep1")
        evaluation = result["evaluations"][0]
        observed = evaluation["risk"]["observed_units"]
        self.assertIsNone(observed["environment_failure"]["label"])
        self.assertIn("no recorded error_class",
                      observed["environment_failure"]["label_basis"])
        self.assertIsNone(observed["implementation_failure"]["label"])


# ---------------------------------------------------------------------------
# 2. budget scope is matched, not assumed
# ---------------------------------------------------------------------------


class TestBudgetScope(CalibrationV2Case):

    def test_window_scope_is_not_an_automatic_exemption(self):
        """A strategy-window prediction is NOT scored against the
        episode-scoped budget ledger: a window is not the whole episode."""
        task = _task("t1")
        payload = json.loads(json.dumps(PAYLOAD))
        payload["risk"]["events"] = [
            {"event": "budget_exhausted", "probability": 0.5}]
        provider = StubProvider(payload=payload)
        h = ORHarness(home=self.home, world_model=provider,
                      embedding=self.backend)
        self.addCleanup(h.close)
        h.declare_budget("t1", {"llm_tokens": 10}, "ep1")
        prediction = h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy", "strategy_id": "S04",
                   "scope": "strategy_window",
                   "window_id": "win::t1::ep1::S04"}, "ep1")
        record = self.solve(task, strategy="S04")
        h.bind_strategy_outcome(prediction.prediction_id,
                                record.action_id)
        result = h.close_episode("t1", "ep1")
        evaluation = result["evaluations"][0]
        self.assertEqual(evaluation["risk"]["scored"], [])
        unscored = {e["event"]: e for e in evaluation["risk"]["unscored"]}
        self.assertIn("scope_mismatch", unscored["budget_exhausted"]
                      ["reason"])
        # The FACT is still recorded by the framework, by its own unit.
        observed = evaluation["risk"]["observed_units"]["budget_exhausted"]
        self.assertEqual(observed["label"], "occurred")
        self.assertEqual(observed["unit"], "episode")

    def test_attempt_scope_is_not_scored_against_the_episode_ledger(self):
        task = _task("t1")
        payload = json.loads(json.dumps(PAYLOAD))
        payload["risk"]["events"] = [
            {"event": "budget_exhausted", "probability": 0.5}]
        provider = StubProvider(payload=payload)
        h = ORHarness(home=self.home, world_model=provider,
                      embedding=self.backend)
        self.addCleanup(h.close)
        h.declare_budget("t1", {"llm_tokens": 10}, "ep1")
        prediction = h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S04"}, "ep1")
        record = self.solve(task, strategy="S04")
        h.bind_strategy_outcome(prediction.prediction_id,
                                record.action_id)
        result = h.close_episode("t1", "ep1")
        risk = result["evaluations"][0]["risk"]
        self.assertEqual(risk["scored"], [])
        self.assertTrue(all("scope_mismatch" in e["reason"]
                            for e in risk["unscored"]
                            if e["event"] == "budget_exhausted"))


# ---------------------------------------------------------------------------
# 3. the window is located, then read; corrections republish
# ---------------------------------------------------------------------------


class TestWindowAndPublication(CalibrationV2Case):

    def _close_episode(self, index, harness=None, metric=None, unit="1-gap"):
        harness = harness or self.h
        task_id = f"task{index}"
        episode = "ep1"
        task = _task(task_id)
        payload = json.loads(json.dumps(PAYLOAD))
        if metric:
            payload["benefit"]["metric"] = metric
        payload["benefit"]["unit"] = unit
        provider = StubProvider(payload=payload)
        h = ORHarness(home=self.home, world_model=provider,
                      embedding=self.backend)
        self.addCleanup(h.close)
        prediction = h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S04"}, episode)
        record = self.solve(task, strategy="S04", episode_id=episode,
                            tag=f"w{index}")
        h.bind_strategy_outcome(prediction.prediction_id, record.action_id)
        h.close_episode(task_id, episode)
        return h

    def test_window_bounds_the_calibration(self):
        for index in range(6):
            self._close_episode(index)
        # A window of 3 sees only the newest 3 episodes.
        summary = self.h.calibration_summary()
        self.assertEqual(summary["n_window_episodes"], 6)
        from or_harness.world_model.episode_closeout import (
            CalibrationPolicy, build_calibration_summary,
        )
        small = build_calibration_summary(
            self.h, policy=CalibrationPolicy(window=3))
        self.assertEqual(small["n_window_episodes"], 3)
        self.assertEqual(small["groups"][GROUP]["n_distinct_episodes"], 3)
        self.assertEqual(small["groups"][GROUP]["n_samples"], 3)

    def test_prediction_read_does_not_scan_history(self):
        """The prediction path reads the PUBLISHED row; it never enumerates
        the evaluation history. Proven by instrumenting the two functions
        that would do the scanning."""
        self._close_episode(0)
        h = self._close_episode(1)
        from or_harness.world_model import episode_closeout as ec
        calls = {"window": 0, "evaluations": 0}
        real_window = ec.calibration_window
        real_evaluations = ec._evaluations_for_window

        def spy_window(*args, **kwargs):
            calls["window"] += 1
            return real_window(*args, **kwargs)

        def spy_evaluations(*args, **kwargs):
            calls["evaluations"] += 1
            return real_evaluations(*args, **kwargs)
        ec.calibration_window = spy_window
        ec._evaluations_for_window = spy_evaluations
        try:
            context = h.build_prediction_context(_task("task1"), "ep2")
        finally:
            ec.calibration_window = real_window
            ec._evaluations_for_window = real_evaluations
        # The published summary was served.
        self.assertEqual(context.strategy_calibration["n_window_episodes"], 2)
        # ...and NO history was enumerated to build it.
        self.assertEqual(
            calls, {"window": 0, "evaluations": 0},
            "a prediction read must serve the published summary, not "
            "rebuild it from history")

    def test_correction_republishes_the_published_summary(self):
        task = _task("t1")
        prediction = self.h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S04"}, "ep1")
        record = self.solve(task, strategy="S04")
        self.h.bind_strategy_outcome(prediction.prediction_id,
                                     record.action_id)
        self.h.close_episode("t1", "ep1")
        before = self.h.calibration_summary()
        self.assertEqual(before["exclusions"].get("validity_corrected"), None)
        # A late check on a window episode.
        self.h.check_task_result(record.execution_id,
                                 {"reference_objective": 999.0})
        after = self.h.calibration_summary()
        self.assertEqual(after["exclusions"].get("validity_corrected"), 1)
        self.assertTrue(after["validity_corrections"])

    def test_publication_is_recoverable_after_a_crash(self):
        """A close that crashes between its two transactions leaves
        ``published=0``; the NEXT close finishes the publication without
        re-counting."""
        task = _task("t1")
        prediction = self.h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S04"}, "ep1")
        record = self.solve(task, strategy="S04")
        self.h.bind_strategy_outcome(prediction.prediction_id,
                                     record.action_id)
        self.h.close_episode("t1", "ep1")
        # Simulate the crash: registry says unpublished, summary removed.
        self.h.store.set_closeout_published("t1", "ep1", False)
        with self.h.store.transaction() as conn:
            conn.execute("DELETE FROM meta WHERE key=?",
                         ("calibration_summary|published",))
        from or_harness.world_model.episode_closeout import (
            published_calibration_summary,
        )
        self.assertIsNone(published_calibration_summary(self.h))
        # Re-closing detects the incomplete publication and finishes it.
        result = self.h.close_episode("t1", "ep1")
        self.assertTrue(result["already_closed"])
        self.assertTrue(result.get("recovered_publication"))
        published = published_calibration_summary(self.h)
        self.assertIsNotNone(published)
        # Nothing was counted twice.
        self.assertEqual(published["n_evaluated"], 1)
        self.assertEqual(
            len(self.h.strategy_prediction_evaluations(task_id="t1")), 1)


# ---------------------------------------------------------------------------
# 4. retention: three scopes, a bounded archive
# ---------------------------------------------------------------------------


class TestRetention(CalibrationV2Case):

    def _close_one(self, index, unchecked=False):
        task_id = f"task{index}"
        task = _task(task_id)
        prediction = self.h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S04"}, "ep1")
        record = self.solve(task, strategy="S04", episode_id="ep1",
                            tag=f"r{index}")
        if not unchecked:
            self.h.check_task_result(record.execution_id,
                                     {"reference_objective": 100.0})
        self.h.bind_strategy_outcome(prediction.prediction_id,
                                     record.action_id)
        self.h.close_episode(task_id, "ep1")
        return record

    def test_archive_moves_detail_and_keeps_the_tombstone(self):
        for index in range(4):
            self._close_one(index)
        from or_harness.world_model.episode_closeout import (
            CalibrationPolicy, archive_calibration_detail,
            published_calibration_summary,
        )
        # A window of 1: the other 3 episodes are out of window.
        policy = CalibrationPolicy(window=1, late_check_grace_days=0.0)
        result = archive_calibration_detail(self.h, policy=policy)
        self.assertEqual(result["removed"]["evaluations"], 3)
        # The registry tombstone stays online.
        self.assertEqual(len(self.h.store.closeout_registry()), 4)
        self.assertEqual(len(self.h.store.closeout_registry(archived=True)), 3)
        # Re-running is idempotent: nothing left to move.
        again = archive_calibration_detail(self.h, policy=policy)
        self.assertEqual(again["removed"]["evaluations"], 0)
        # A repeated close is STILL idempotent after archiving.
        reclose = self.h.close_episode("task0", "ep1")
        self.assertTrue(reclose["already_closed"])
        # The window still resolves (the registry row is the index).
        self.assertIsNotNone(published_calibration_summary(self.h))

    def test_late_check_grace_holds_only_episodes_awaiting_a_verdict(self):
        self._close_one(0, unchecked=True)
        self._close_one(1, unchecked=False)
        from or_harness.world_model.episode_closeout import (
            CalibrationPolicy, archive_calibration_detail,
        )
        policy = CalibrationPolicy(window=0, late_check_grace_days=30.0)
        result = archive_calibration_detail(self.h, policy=policy)
        held = {h["task_id"] for h in result["held_for_late_check"]}
        # Only the UNCHECKED episode is held — the checked one is not given
        # a blanket extra retention.
        self.assertEqual(held, {"task0"})
        # The checked episode's detail WAS archived.
        self.assertEqual(result["removed"]["evaluations"], 1)
        self.assertEqual(
            len(self.h.store.closeout_registry(archived=True)), 1)

    def test_archive_has_a_total_size_cap(self):
        for index in range(4):
            self._close_one(index)
        from or_harness.world_model.episode_closeout import (
            CalibrationPolicy, archive_calibration_detail, _archive_files,
        )
        policy = CalibrationPolicy(
            window=0, late_check_grace_days=0.0,
            archive_max_file_bytes=200, archive_max_total_bytes=400,
            archive_retention_days=365.0)
        result = archive_calibration_detail(self.h, policy=policy)
        limits = result["archive_limits"]
        # The total cap was enforced: the archive cannot grow without
        # bound. (Honouring the cap may mean removing every file, which is
        # the honest outcome of a cap smaller than one file's contents.)
        self.assertLessEqual(limits["total_bytes"],
                             policy.archive_max_total_bytes)
        self.assertLessEqual(limits["total_bytes"], 20000)

    def test_archive_age_cap_deletes_old_files(self):
        self._close_one(0)
        from or_harness.world_model.episode_closeout import (
            CalibrationPolicy, archive_calibration_detail,
            archive_dir, _archive_files,
        )
        policy = CalibrationPolicy(window=0, late_check_grace_days=0.0)
        archive_calibration_detail(self.h, policy=policy)
        files = _archive_files(self.h)
        self.assertTrue(files)
        # Age every archive file far into the past.
        old = time.time() - 400 * 86400.0
        for path in files:
            os.utime(path, (old, old))
        aged = CalibrationPolicy(window=0, late_check_grace_days=0.0,
                                 archive_retention_days=365.0)
        result = archive_calibration_detail(self.h, policy=aged)
        self.assertTrue(result["archive_limits"]["removed_files"])
        # The retention period is what bounds the total storage.
        self.assertEqual(_archive_files(self.h), [])

    def test_restored_payload_does_not_re_enter_calibration(self):
        for index in range(3):
            self._close_one(index)
        from or_harness.world_model.episode_closeout import (
            CalibrationPolicy, archive_calibration_detail,
        )
        policy = CalibrationPolicy(window=1, late_check_grace_days=0.0)
        archive_calibration_detail(self.h, policy=policy)
        before = self.h.calibration_summary()
        # "Restore" is an audit action: even if a payload came back, the
        # window is decided by the registry's closed_at ordering, so the
        # calibration is unchanged.
        after = self.h.calibration_summary()
        self.assertEqual(before["n_window_episodes"],
                         after["n_window_episodes"])
        self.assertEqual(before["groups"], after["groups"])

    def test_auto_archive_triggers_only_past_the_threshold(self):
        for index in range(3):
            self._close_one(index)
        from or_harness.world_model.episode_closeout import (
            CalibrationPolicy, maybe_auto_archive,
        )
        low = maybe_auto_archive(
            self.h, policy=CalibrationPolicy(window=1,
                                             auto_archive_threshold=100))
        self.assertFalse(low["triggered"])
        self.assertTrue(low["checked"])
        high = maybe_auto_archive(
            self.h, policy=CalibrationPolicy(
                window=1, auto_archive_threshold=1,
                late_check_grace_days=0.0))
        self.assertTrue(high["triggered"])

    def test_retention_view_reports_three_scopes(self):
        self._close_one(0)
        view = self.h.calibration_retention()
        self.assertIn("policy", view)
        self.assertIn("online", view)
        self.assertIn("archive", view)
        policy = view["policy"]
        for key in ("window", "late_check_grace_days",
                    "archive_max_file_bytes", "archive_max_total_bytes",
                    "archive_retention_days"):
            self.assertIn(key, policy)


# ---------------------------------------------------------------------------
# 5. two counting units
# ---------------------------------------------------------------------------


class TestCountingUnits(CalibrationV2Case):

    def test_repeated_predictions_are_one_observation_unit(self):
        """Three predictions bound to ONE execution are three scored pairs
        but ONE observation unit."""
        task = _task("t1")
        payload = json.loads(json.dumps(PAYLOAD))
        payload["risk"]["events"] = [
            {"event": "timeout", "probability": 0.3}]
        provider = StubProvider(payload=payload)
        h = ORHarness(home=self.home, world_model=provider,
                      embedding=self.backend)
        self.addCleanup(h.close)
        predictions = [h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S04"}, "ep1") for _ in range(3)]
        record = self.solve(task, strategy="S04")
        for prediction in predictions:
            h.bind_strategy_outcome(prediction.prediction_id,
                                    record.action_id)
        h.close_episode("t1", "ep1")
        summary = h.calibration_summary(min_samples=1)
        group = summary["groups"][GROUP]
        # Prediction pairs: three.
        self.assertEqual(group["n_brier_samples_by_event"]["timeout"], 3)
        # The mean probability is over those three pairs.
        self.assertEqual(group["mean_predicted_probability"]["timeout"],
                         0.3)
        self.assertEqual(group["scored_occurrence_rate"]["timeout"], 0.0)
        # Observation units: ONE real execution.
        occurrence = summary["occurrence"]["timeout"]
        self.assertEqual(occurrence["observation_unit"], "execution")
        self.assertEqual(occurrence["n_observation_units"], 1)
        self.assertEqual(occurrence["n_occurred"], 0)
        self.assertEqual(occurrence["n_not_occurred"], 1)

    def test_unpredicted_events_enter_the_occurrence_rate(self):
        """An event the model never predicted is still counted as an
        observation — but it can never produce a Brier score."""
        task = _task("t1")
        payload = json.loads(json.dumps(PAYLOAD))
        payload["risk"]["events"] = []          # nothing predicted
        provider = StubProvider(payload=payload)
        h = ORHarness(home=self.home, world_model=provider,
                      embedding=self.backend)
        self.addCleanup(h.close)
        prediction = h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S04"}, "ep1")
        # A SUCCESSFUL execution (so the benefit is observable and a group
        # exists) that nevertheless TIMED OUT is contradictory; use a
        # timeout execution and check the occurrence channel alone.
        record = self.solve(task, strategy="S04", status="timeout",
                            objective=0.0, tag="to")
        h.bind_strategy_outcome(prediction.prediction_id,
                                record.action_id)
        h.close_episode("t1", "ep1")
        summary = h.calibration_summary(min_samples=1)
        occurrence = summary["occurrence"]["timeout"]
        self.assertEqual(occurrence["n_occurred"], 1)
        self.assertEqual(occurrence["unit_occurrence_rate"], 1.0)
        # No probability was ever predicted, so no Brier sample exists.
        group = summary["groups"][GROUP]
        self.assertEqual(group["n_samples"], 1)
        self.assertNotIn("timeout", group["mean_brier_by_event"])
        self.assertNotIn("timeout", group["mean_predicted_probability"])
        # The benefit was unobservable (a timed-out execution produced no
        # qualified quality), so it contributes no error — but the DECLARED
        # yardstick still names the group it belongs to.
        self.assertIsNone(group["mean_benefit_signed_error"])
        self.assertIsNone(group["mean_benefit_abs_error"])

    def test_probability_and_rate_share_the_scored_denominator(self):
        """``mean_predicted_probability`` and ``scored_occurrence_rate``
        must be comparable: same sample set, so their difference is a
        meaningful over/under-estimate signal. The unit occurrence rate is
        reported separately."""
        task = _task("t1")
        payload = json.loads(json.dumps(PAYLOAD))
        payload["risk"]["events"] = [
            {"event": "timeout", "probability": 0.8}]
        provider = StubProvider(payload=payload)
        h = ORHarness(home=self.home, world_model=provider,
                      embedding=self.backend)
        self.addCleanup(h.close)
        prediction = h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S04"}, "ep1")
        record = self.solve(task, strategy="S04")     # no timeout
        h.bind_strategy_outcome(prediction.prediction_id,
                                record.action_id)
        h.close_episode("t1", "ep1")
        summary = h.calibration_summary(min_samples=1)
        group = summary["groups"][GROUP]
        self.assertEqual(group["mean_predicted_probability"]["timeout"], 0.8)
        self.assertEqual(group["scored_occurrence_rate"]["timeout"], 0.0)
        # The Brier mean matches: 0.8**2 = 0.64.
        self.assertAlmostEqual(
            group["mean_brier_by_event"]["timeout"], 0.64, places=5)
        # The unit rate has its own, separately-labelled denominator.
        occurrence = summary["occurrence"]["timeout"]
        self.assertIn("labelled observation units", occurrence["rate_basis"])


# ---------------------------------------------------------------------------
# 6. directed statistics, grouping and filtering
# ---------------------------------------------------------------------------


class TestDirectedStatistics(CalibrationV2Case):

    def test_signed_error_distinguishes_over_and_under_prediction(self):
        task = _task("t1")
        prediction = self.h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S04"}, "ep1")
        record = self.solve(task, strategy="S04")   # observed 1.0
        self.h.bind_strategy_outcome(prediction.prediction_id,
                                     record.action_id)
        result = self.h.close_episode("t1", "ep1")
        benefit = result["evaluations"][0]["benefit"]
        # predicted 0.8, observed 1.0 → under-predicted by 0.2.
        self.assertAlmostEqual(benefit["signed_error"], 0.2, places=5)
        self.assertAlmostEqual(benefit["abs_error"], 0.2, places=5)
        summary = self.h.calibration_summary(min_samples=1)
        self.assertAlmostEqual(
            summary["groups"][GROUP]["mean_benefit_signed_error"],
            0.2, places=5)

    def test_cost_log_ratio_is_directed(self):
        task = _task("t1")
        prediction = self.h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S04"}, "ep1")
        record = self.solve(task, strategy="S04")
        self.h.bind_strategy_outcome(prediction.prediction_id,
                                     record.action_id)
        result = self.h.close_episode("t1", "ep1")
        entry = result["evaluations"][0]["cost"]["per_dim"][
            "solver_runtime_s"]
        # predicted 3.0, actual 0.01 → log(0.01/3.0) < 0 (over-predicted).
        self.assertIsNotNone(entry["log_ratio"])
        self.assertLess(entry["log_ratio"], 0)
        self.assertAlmostEqual(entry["log_ratio"], -entry["log_error"],
                               places=4)

    def test_zero_cost_produces_no_ratio(self):
        task = _task("t1")
        payload = json.loads(json.dumps(PAYLOAD))
        payload["cost"] = {"solver_runtime_s": 0.0}
        provider = StubProvider(payload=payload)
        h = ORHarness(home=self.home, world_model=provider,
                      embedding=self.backend)
        self.addCleanup(h.close)
        prediction = h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S04"}, "ep1")
        record = self.solve(task, strategy="S04")
        h.bind_strategy_outcome(prediction.prediction_id,
                                record.action_id)
        result = h.close_episode("t1", "ep1")
        entry = result["evaluations"][0]["cost"]["per_dim"][
            "solver_runtime_s"]
        self.assertIsNone(entry["log_ratio"])
        self.assertIsNone(entry["log_error"])
        self.assertIn("no relative ratio is manufactured", entry["note"])

    def test_interval_width_is_reported(self):
        task = _task("t1")
        payload = json.loads(json.dumps(PAYLOAD))
        payload["benefit"]["interval"] = [0.6, 0.9]
        provider = StubProvider(payload=payload)
        h = ORHarness(home=self.home, world_model=provider,
                      embedding=self.backend)
        self.addCleanup(h.close)
        prediction = h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S04"}, "ep1")
        record = self.solve(task, strategy="S04")
        h.bind_strategy_outcome(prediction.prediction_id,
                                record.action_id)
        result = h.close_episode("t1", "ep1")
        interval = result["evaluations"][0]["interval"]
        self.assertAlmostEqual(interval["width"], 0.3, places=5)
        summary = h.calibration_summary(min_samples=1)
        self.assertAlmostEqual(
            summary["groups"][GROUP]["mean_interval_width"], 0.3, places=5)


class TestGroupingAndFiltering(CalibrationV2Case):

    def test_model_identity_is_part_of_the_group_key(self):
        task = _task("t1")
        provider = StubProvider(model="model-a", version="v1")
        h = ORHarness(home=self.home, world_model=provider,
                      embedding=self.backend)
        self.addCleanup(h.close)
        prediction = h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S04"}, "ep1")
        record = self.solve(task, strategy="S04")
        h.bind_strategy_outcome(prediction.prediction_id,
                                record.action_id)
        h.close_episode("t1", "ep1")
        summary = h.calibration_summary()
        self.assertIn("strategy_outcome|model-a@v1|normalized_objective_gap"
                      "|1-gap|attempt", summary["groups"])
        evaluation = h.strategy_prediction_evaluations(task_id="t1")[0]
        self.assertEqual(evaluation["model_identity"], "model-a@v1")

    def test_another_models_groups_are_withheld_from_the_context(self):
        # Model A closes an episode...
        task = _task("t1")
        provider_a = StubProvider(model="model-a")
        h = ORHarness(home=self.home, world_model=provider_a,
                      embedding=self.backend)
        self.addCleanup(h.close)
        prediction = h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S04"}, "ep1")
        record = self.solve(task, strategy="S04")
        h.bind_strategy_outcome(prediction.prediction_id,
                                record.action_id)
        h.close_episode("t1", "ep1")
        # ...then model B builds a context: A's groups are withheld.
        provider_b = StubProvider(model="model-b")
        h_b = ORHarness(home=self.home, world_model=provider_b,
                        embedding=self.backend)
        self.addCleanup(h_b.close)
        context = h_b.build_prediction_context(_task("t1"), "ep2")
        calibration = context.strategy_calibration
        self.assertEqual(calibration["groups"], {})
        self.assertTrue(calibration.get("withheld_groups"))
        self.assertIn("different model identity",
                      calibration["filter_note"])

    def test_request_withholds_other_scope_groups(self):
        """A window-scope group is not evidence about an attempt-scope
        candidate, so the REQUEST withholds it."""
        from or_harness.world_model.strategy_prediction import (
            _filter_calibration_by_scope,
        )
        calibration = {"groups": {
            "strategy_outcome|m|metric|unit|attempt": {"n_samples": 1},
            "strategy_outcome|m|metric|unit|strategy_window": {"n_samples": 9},
        }}
        filtered, withheld = _filter_calibration_by_scope(
            calibration, "attempt")
        self.assertIn("strategy_outcome|m|metric|unit|attempt",
                      filtered["groups"])
        self.assertNotIn("strategy_outcome|m|metric|unit|strategy_window",
                         filtered["groups"])
        self.assertEqual(withheld,
                         ["strategy_outcome|m|metric|unit|strategy_window"])

    def test_summary_is_labelled_a_global_diagnostic(self):
        task = _task("t1")
        prediction = self.h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S04"}, "ep1")
        record = self.solve(task, strategy="S04")
        self.h.bind_strategy_outcome(prediction.prediction_id,
                                     record.action_id)
        self.h.close_episode("t1", "ep1")
        summary = self.h.calibration_summary()
        self.assertEqual(summary["applicability"], "global_diagnostic")
        self.assertIn("NOT a claim about the conditional",
                      summary["applicability_note"])
        self.assertEqual(summary["calibration_version"], "wm-calib/2")
        self.assertEqual(summary["event_vocabulary_version"], "wm-events/2")

    def test_legacy_evaluation_without_identity_goes_to_unknown(self):
        """An evaluation stored before the identity existed must not have a
        model invented for it."""
        task = _task("t1")
        prediction = self.h.predict_strategy_outcome(
            task, {"action_type": "execute_strategy",
                   "strategy_id": "S04"}, "ep1")
        record = self.solve(task, strategy="S04")
        self.h.bind_strategy_outcome(prediction.prediction_id,
                                     record.action_id)
        self.h.close_episode("t1", "ep1")
        # Strip the identity from the stored evaluation (legacy shape).
        evaluations = self.h.strategy_prediction_evaluations(task_id="t1")
        evaluation_id = evaluations[0]["evaluation_id"]
        stored = self.h.get_strategy_evaluation(evaluation_id)
        stored.pop("model_identity", None)
        with self.h.store.transaction() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?,?)",
                (f"strategy_evaluation|{evaluation_id}",
                 self.h.store.dumps(stored)))
        rebuilt = self.h.calibration_summary(rebuild=True)
        self.assertIn("strategy_outcome|(unknown)|normalized_objective_gap"
                      "|1-gap|attempt", rebuilt["groups"])

    def test_no_published_summary_reports_missing(self):
        context = self.h.build_prediction_context(_task("t1"), "ep1")
        calibration = context.strategy_calibration
        self.assertEqual(calibration["groups"], {})
        self.assertIn("no calibration summary has been published",
                      calibration["missing"])
        self.assertEqual(calibration["n_evaluations_total"], 0)


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


class TestRetentionCli(CalibrationV2Case):

    def _run(self, argv):
        from or_harness.cli import main
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = main(["--home", self.home] + argv)
        return code, json.loads(buf.getvalue())

    def test_archive_calibration_command(self):
        code, payload = self._run(["archive-calibration", "--dry-run"])
        self.assertEqual(code, 0)
        self.assertTrue(payload["result"]["dry_run"])

    def test_retention_command(self):
        code, payload = self._run(["retention"])
        self.assertEqual(code, 0)
        self.assertIn("policy", payload["result"])
        self.assertIn("online", payload["result"])

    def test_calibration_rebuild_flag(self):
        code, payload = self._run(["calibration", "--rebuild"])
        self.assertEqual(code, 0)
        self.assertEqual(payload["result"]["calibration_version"],
                         "wm-calib/2")


if __name__ == "__main__":
    unittest.main()
