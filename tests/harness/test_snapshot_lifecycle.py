"""Behavioral acceptance tests for disposable prediction inputs and old banks."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from or_harness.api import ORHarness
from or_harness.core.schema import CostVector, ExecutionRecord, StrategicEntry
from or_harness.core.storage import StorageError
from or_harness.strategy.embedding_index import LocalHashEmbeddingBackend
from or_harness.world_model.conditions import (
    ContextExpired, get_value, read_snapshot, release_context,
)
from or_harness.world_model.context import PredictionContext
from or_harness.world_model.maintenance_decision import _snapshot_knowledge_entry_ids
from or_harness.world_model.prediction import ActionSpec
from or_harness.world_model.provider import WorldModelProvider
from or_harness.world_model.snapshot_maintenance import fork_memory, maintain_snapshots
from or_harness.world_model.state import BeliefSnapshot, task_text_digest


TASK = {"task_id": "t", "family": "routing", "description": "Minimize cost with capacity 10.",
        "spec": {"n_vars": 10, "n_constraints": 5, "n_int_vars": 10},
        "annotations": {"coupling": {"resource_coupling": 0.3,
            "temporal_coupling": 0.1, "route_complexity": 0.2}}}
CANDIDATE = {"action_type": "execute_strategy", "strategy_id": "candidate",
             "solver": "highs", "scope": "attempt"}


class Stub(WorldModelProvider):
    name = "snapshot-fixture"

    def __init__(self, fail=False):
        self.requests = []
        self.fail = fail

    def predict(self, request, timeout_s=None):
        self.requests.append(copy.deepcopy(request))
        return {"payload": None if self.fail else {
                    "benefit": {"kind": "solution_quality", "metric": "normalized_objective_gap",
                                "unit": "1-gap", "value": 0.8,
                                "baseline": {"kind": "conditional_stats", "value": 0.7}},
                    "cost": {"llm_tokens": 10},
                    "risk": {"events": [{"event": "timeout", "probability": 0.1}]}},
                "error": "fixture failure" if self.fail else None,
                "usage": {"prompt_tokens": 7, "completion_tokens": 3}, "latency_s": 0.01}


class LifecycleCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.home = str(Path(self.tmp.name) / "home")
        self.provider = Stub()
        self.h = ORHarness(home=self.home, world_model=self.provider,
                           embedding=LocalHashEmbeddingBackend())
        self.addCleanup(self.h.close)

    def count(self, table):
        return self.h.store.conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]

    def context(self, **kwargs):
        return self.h.build_prediction_context(TASK, "e", candidates=[
            ActionSpec("execute_strategy", "t", strategy_id="candidate")], **kwargs)

    def legacy_snapshot(self, episode="closed"):
        snap = self.h.snapshot(TASK, episode, persist=False)
        with self.h.store.transaction() as conn:
            conn.execute("INSERT INTO belief_snapshots VALUES (?,?,?,?,?)",
                         (snap.snapshot_id, snap.task_id, snap.episode_id,
                          snap.created_at, self.h.store.dumps(snap.to_dict())))
        return snap

    def test_no_full_snapshots_and_no_duplicate_x_b(self):
        for i in range(10):
            self.h.report_action("model", TASK, "e", outcome={"model": i})
            self.context()
        self.assertEqual(self.count("belief_snapshots"), 0)
        ctx = self.context()
        self.assertNotIn("snapshot", ctx.to_dict())
        self.assertNotIn("task_progress", ctx.to_dict()["conditions"])
        self.assertNotIn("budget_state", ctx.to_dict()["conditions"])
        receipt = read_snapshot(self.h.store, ctx.snapshot_id)
        self.assertTrue(receipt.coverage["conditions_only"])
        self.assertNotIn("cell_statistics", receipt.coverage)
        self.assertEqual(self.h.store.conn.execute(
            "SELECT count(*) FROM meta WHERE key LIKE 'knowledge_version|%'"
        ).fetchone()[0], 0)

    def test_no_persist_does_not_write_conditions_or_runtime(self):
        before = [self.count(table) for table in (
            "frozen_conditions", "condition_values", "prediction_contexts", "runtime_inputs")]
        self.context(persist=False)
        after = [self.count(table) for table in (
            "frozen_conditions", "condition_values", "prediction_contexts", "runtime_inputs")]
        self.assertEqual(before, after)

    def test_fixed_candidates_and_hits_do_not_store_unrelated_statistics(self):
        recall = {"task_digest": task_text_digest(TASK), "recommendations": [],
                  "degraded": {"part": "semantic", "reason": "fixed fixture"}}
        sizes = []
        for total in (1, 200):
            for i in range(self.count("executions"), total):
                self.h.bank.append(ExecutionRecord(
                    execution_id=f"ex_noise_{i}", task_id=f"noise_{i}",
                    strategy_id="candidate" if i == 0 else f"unrelated_{i}",
                    profile_snapshot=self.h.profile(TASK),
                    quality={"feasible": True, "gap": 0, "status": "optimal"},
                    cost=CostVector(llm_tokens=10, measured={"llm_tokens"})))
            ctx = self.context(recall_result=recall)
            header = self.h.store.get_prediction_context(ctx.context_id)
            raw = self.h.store.conn.execute("SELECT payload FROM runtime_inputs WHERE context_id=?",
                                            (ctx.context_id,)).fetchone()[0]
            self.assertNotIn("unrelated_", raw)
            self.assertTrue(all("execution_ids" not in cell for cell in
                                ctx.snapshot["coverage"]["cell_statistics"].values()))
            sizes.append(len(header) + len(raw))
        self.assertLess(sizes[1] - sizes[0], 1000)

    def test_large_problem_is_shared_across_active_decisions(self):
        task = dict(TASK, description="large problem " * 10000)
        for _ in range(5):
            self.h.build_prediction_context(task, "e")
        rows = self.h.store.conn.execute(
            "SELECT payload FROM runtime_inputs").fetchall()
        self.assertTrue(all("large problem" not in row[0] for row in rows))
        self.assertEqual(self.h.store.conn.execute(
            "SELECT count(*) FROM condition_values WHERE kind='problem_input'").fetchone()[0], 1)

    def test_multi_candidate_frozen_budget_survives_input_release(self):
        self.h.declare_budget("t", {"llm_tokens": 1000}, "e")
        ctx = self.context()
        p1 = self.h.predict_strategy_outcome(TASK, CANDIDATE, "e", context=ctx)
        self.h.declare_budget("t", {"llm_tokens": 2000}, "e")
        p2 = self.h.predict_strategy_outcome(TASK, dict(CANDIDATE, strategy_id="other"), "e", context=ctx)
        self.assertEqual(p1.trace.input_snapshot_id, p2.trace.input_snapshot_id)
        self.assertEqual(p2.trace.model_info["frozen_budget_state"]["budget"], {"llm_tokens": 1000.0})
        release_context(self.h.store, ctx.context_id)
        from or_harness.world_model.episode_closeout import _frozen_declared_budget
        self.assertEqual(_frozen_declared_budget(self.h, p1), {"llm_tokens": 1000.0})
        with self.assertRaises(ContextExpired):
            self.h.predict_strategy_outcome(TASK, CANDIDATE, "e", context=ctx)

    def test_failed_prediction_first_save_has_frozen_identity_and_usage(self):
        self.provider.fail = True
        saved = []
        original = self.h.strategy_predictions._save
        def observe(prediction):
            saved.append(copy.deepcopy(prediction.to_dict()))
            original(prediction)
        with patch.object(self.h.strategy_predictions, "_save", side_effect=observe):
            prediction = self.h.predict_strategy_outcome(TASK, CANDIDATE, "e")
        self.assertEqual(prediction.status, "invalid")
        self.assertEqual(prediction.trace.call_cost.llm_tokens, 10)
        info = saved[0]["trace"]["model_info"]
        self.assertTrue(info["problem_identity_digest"])
        self.assertTrue(info["prediction_context_id"])
        self.assertTrue(info["request_digest"])
        self.assertIsNotNone(read_snapshot(self.h.store, info["conditions_id"]))

    def test_hypothetical_prediction_cannot_bind_real_action(self):
        snapshot = self.h.snapshot(TASK, "e", hypothetical=True)
        ctx = self.context(snapshot=snapshot, historical=False)
        prediction = self.h.predict_strategy_outcome(TASK, CANDIDATE, "e", context=ctx)
        real = self.h.snapshot(TASK, "e")
        action = self.h.actions.begin_action("execute_strategy", "t", "e", pre_snapshot=real,
                                             params={"strategy_id": "candidate", "solver": "highs"})
        self.h.actions.end_action(action.action_id, outcome={})
        bound = self.h.bind_strategy_outcome(prediction.prediction_id, action.action_id)
        self.assertIn("input_state", bound.trace.model_info["binding_mismatch"])
        self.assertFalse(bound.trace.comparable)

    def test_missing_legacy_source_remains_unknown_instead_of_becoming_real(self):
        prediction = self.h.predict_strategy_outcome(TASK, CANDIDATE, "e")
        prediction.trace.model_info.pop("hypothetical")
        prediction.trace.model_info.pop("conditions_id")
        prediction.trace.model_info.pop("prediction_context_id")
        prediction.trace.input_snapshot_id = "bs_missing_historical_source"
        self.h.strategy_predictions._save(prediction)
        action = self.h.actions.begin_action("execute_strategy", "t", "e",
            pre_snapshot=self.h.snapshot(TASK, "e"), params={"strategy_id": "candidate", "solver": "highs"})
        self.h.actions.end_action(action.action_id, outcome={})
        bound = self.h.bind_strategy_outcome(prediction.prediction_id, action.action_id)
        self.assertIn("input_state", bound.trace.model_info["binding_unknown"])
        from or_harness.world_model.attribution import binding_attribution
        attributed = binding_attribution({}, bound.trace.model_info["binding_unknown"])
        self.assertTrue(all(attributed["blocked"][dim] for dim in attributed["blocked"]))

    def test_absent_budget_declaration_is_distinct_from_missing_budget_history(self):
        prediction = self.h.predict_strategy_outcome(TASK, CANDIDATE, "e")
        from or_harness.world_model.episode_closeout import _frozen_declared_budget
        self.assertEqual(_frozen_declared_budget(self.h, prediction), {})
        self.h.declare_budget("t", {"llm_tokens": 1000}, "e")
        self.assertEqual(_frozen_declared_budget(self.h, prediction), {})
        prediction.trace.model_info.pop("frozen_budget_state")
        prediction.trace.model_info.pop("conditions_id")
        prediction.trace.model_info.pop("prediction_context_id")
        prediction.trace.input_snapshot_id = "bs_missing"
        self.assertIsNone(_frozen_declared_budget(self.h, prediction))

    def test_new_budget_after_forecast_does_not_relabel_its_condition(self):
        response = Stub().predict({})
        response["payload"]["risk"]["events"] = [{"event": "budget_exhausted", "probability": 0.5}]
        with patch.object(self.provider, "predict", return_value=response):
            prediction = self.h.predict_strategy_outcome(TASK, CANDIDATE, "e")
        self.h.declare_budget("t", {"llm_tokens": 1000}, "e")
        action = self.h.actions.begin_action("execute_strategy", "t", "e",
            pre_snapshot=self.h.snapshot(TASK, "e"), params={"strategy_id": "candidate", "solver": "highs"})
        record = ExecutionRecord(execution_id="overrun", task_id="t", strategy_id="candidate",
            profile_snapshot=self.h.profile(TASK), quality={"status": "optimal", "feasible": True, "gap": 0},
            cost=CostVector(llm_tokens=5000, measured={"llm_tokens"}))
        self.h.bank.append(record)
        self.h.actions.end_action(action.action_id, outcome={}, linked_execution_id=record.execution_id)
        self.h.bind_strategy_outcome(prediction.prediction_id, action.action_id)
        risk = self.h.close_episode("t", "e")["evaluations"][0]["risk"]
        self.assertNotIn("budget_exhausted", {row["event"] for row in risk["scored"]})
        # A real overrun remains an execution fact even though its new
        # budget cannot retroactively become the prediction's condition.
        self.assertEqual(self.h.bank.get("overrun").cost.llm_tokens, 5000)

    def test_missing_old_budget_excludes_probability_but_preserves_real_cost(self):
        response = Stub().predict({})
        response["payload"]["risk"]["events"] = [{"event": "budget_exhausted", "probability": 0.5}]
        with patch.object(self.provider, "predict", return_value=response):
            prediction = self.h.predict_strategy_outcome(TASK, CANDIDATE, "e")
        for key in ("frozen_budget_state", "conditions_id", "prediction_context_id"):
            prediction.trace.model_info.pop(key)
        with self.h.store.transaction() as conn:
            raw = self.h.store.loads(conn.execute("SELECT payload FROM frozen_conditions WHERE snapshot_id=?",
                (prediction.trace.input_snapshot_id,)).fetchone()[0])
            raw["budget_state"] = {}  # legacy fixture: budget history was never saved
            conn.execute("UPDATE frozen_conditions SET payload=? WHERE snapshot_id=?",
                         (self.h.store.dumps(raw), prediction.trace.input_snapshot_id))
        self.h.strategy_predictions._save(prediction)
        self.h.declare_budget("t", {"llm_tokens": 1000}, "e")
        action = self.h.actions.begin_action("execute_strategy", "t", "e",
            pre_snapshot=self.h.snapshot(TASK, "e"), params={"strategy_id": "candidate", "solver": "highs"})
        record = ExecutionRecord(execution_id="missing-budget", task_id="t", strategy_id="candidate",
            profile_snapshot=self.h.profile(TASK), quality={"status": "optimal", "feasible": True, "gap": 0},
            cost=CostVector(llm_tokens=5000, measured={"llm_tokens"}))
        self.h.bank.append(record)
        self.h.actions.end_action(action.action_id, outcome={}, linked_execution_id=record.execution_id)
        self.h.bind_strategy_outcome(prediction.prediction_id, action.action_id)
        risk = self.h.close_episode("t", "e")["evaluations"][0]["risk"]
        self.assertNotIn("budget_exhausted", {row["event"] for row in risk["scored"]})
        self.assertEqual(self.h.bank.get(record.execution_id).cost.measured_dims(), {"llm_tokens"})
        self.assertEqual(self.h.bank.get(record.execution_id).cost.llm_tokens, 5000)

    def test_capacity_and_clock_expiry_are_explicit_and_bounded(self):
        with patch.dict(os.environ, {"OR_HARNESS_ACTIVE_CONTEXTS": "2"}):
            oldest = self.context()
            self.context()
            self.context()
        self.assertEqual(self.count("runtime_inputs"), 2)
        with self.assertRaises(ContextExpired):
            self.h.get_prediction_context(oldest.context_id)
        current = self.context()
        with self.h.store.transaction() as conn:
            conn.execute("UPDATE runtime_inputs SET expires_at=0 WHERE context_id=?", (current.context_id,))
        with self.assertRaises(ContextExpired):
            self.h.get_prediction_context(current.context_id)

    def test_problem_model_write_keeps_progress_real_change_resets_it(self):
        self.h.snapshot(TASK, "e", task_progress={"selected_plan": {"value": "candidate"}})
        after_model = self.h.snapshot(dict(TASK, model={"variables": []}), "e")
        self.assertIn("selected_plan", after_model.task_progress)
        changed = self.h.snapshot(dict(TASK, description="different problem"), "e")
        self.assertNotIn("selected_plan", changed.task_progress)

    def test_calibration_detail_not_sent_aggregate_survives(self):
        ctx = self.context()
        ctx.strategy_calibration = {"groups": {"g": {"n": 7}}, "validity_corrections": [
            {"kind": "live_rederivation", "fields": ["benefit"],
             "detail": {"stored": "secret full audit", "derived": "new"}}]}
        view = ctx.provider_view()["strategy_outcome_calibration"]
        self.assertEqual(view["groups"], {"g": {"n": 7}})
        self.assertNotIn("validity_corrections", view)
        self.assertNotIn("secret full audit", json.dumps(view))
        self.assertEqual(view["validity_correction_summary"]["count"], 1)

    def test_old_context_and_snapshot_read_without_payload_migration(self):
        entry = StrategicEntry(entry_id="", strategy_id="candidate", expected_quality_hat=0.4,
            pattern={"predicates": {"family": "routing"}}, verification={"state": "verified"})
        self.h.sbank.add(entry)
        snap = self.legacy_snapshot(episode="e")
        entry.expected_quality_hat = 0.9
        self.h.sbank.update(entry)
        ctx = self.context(snapshot=snap, historical=True, persist=False)
        data = ctx.to_dict()
        data["context_version"] = "wm-context/3"
        data["snapshot"] = snap.to_dict()
        data.pop("conditions")
        self.h.store.put_prediction_context(ctx.context_id, "t", "closed", self.h.store.dumps(data))
        stored_before = self.h.store.get_prediction_context(ctx.context_id)
        self.assertEqual(self.h.get_prediction_context(ctx.context_id).version, "wm-context/3")
        self.assertEqual(self.h.get_snapshot(snap.snapshot_id).to_dict(), snap.to_dict())
        self.assertEqual(self.h.store.get_prediction_context(ctx.context_id), stored_before)
        data["context_version"] = "wm-context/future"
        from or_harness.world_model.context import UnsupportedContextVersion
        with self.assertRaises(UnsupportedContextVersion):
            PredictionContext.from_dict(data)

    def test_availability_as_of_execution_distinct_from_adoption(self):
        entry = StrategicEntry(entry_id="", strategy_id="candidate", pattern={"predicates": {"family": "routing"}})
        key = self.h.sbank.add(entry)
        before = self.h.snapshot(TASK, "e")
        entry.pattern = {"predicates": {"family": "other"}}
        self.h.sbank.update(entry)
        after = self.h.snapshot(TASK, "e")
        self.assertIn(key, _snapshot_knowledge_entry_ids(self.h.get_snapshot(before.snapshot_id), store=self.h.store))
        self.assertNotIn(key, _snapshot_knowledge_entry_ids(self.h.get_snapshot(after.snapshot_id), store=self.h.store))

    def test_nested_transaction_failure_leaves_no_dangling_reference(self):
        count = self.count("frozen_conditions")
        with self.assertRaises(RuntimeError):
            with self.h.store.transaction():
                self.context()
                raise RuntimeError("interrupt before commit")
        self.assertEqual(self.count("frozen_conditions"), count)
        self.assertEqual(self.count("runtime_inputs"), 0)
        self.assertEqual(self.count("prediction_contexts"), 0)

    def test_interruption_keeps_unknown_call_and_durable_conditions(self):
        with patch.object(self.provider, "predict", side_effect=KeyboardInterrupt("interrupted call")):
            with self.assertRaises(KeyboardInterrupt):
                self.h.predict_strategy_outcome(TASK, CANDIDATE, "e")
        predictions = self.h.strategy_outcome_predictions(task_id="t")
        self.assertEqual(len(predictions), 1)
        pending = predictions[0]
        self.assertEqual(pending.status, "invalid")
        self.assertEqual(pending.trace.model_info["call_state"], "in_flight")
        self.assertIsNone(pending.trace.call_cost)
        self.assertIsNotNone(read_snapshot(self.h.store, pending.trace.input_snapshot_id))

    def test_capacity_cannot_discard_an_inflight_provider_input(self):
        original = self.provider.predict
        def while_calling(request, timeout_s=None):
            with self.assertRaises(ContextExpired):
                self.context()
            return original(request, timeout_s=timeout_s)
        with patch.dict(os.environ, {"OR_HARNESS_ACTIVE_CONTEXTS": "1"}), \
                patch.object(self.provider, "predict", side_effect=while_calling):
            prediction = self.h.predict_strategy_outcome(TASK, CANDIDATE, "e")
        self.assertEqual(prediction.status, "valid")
        self.assertEqual(self.count("runtime_inputs"), 1)
        self.assertEqual(self.h.store.conn.execute("SELECT in_use FROM runtime_inputs").fetchone()[0], 0)

    def test_abandoned_call_lease_expires_without_pinning_input_forever(self):
        ctx = self.context()
        with self.h.store.transaction() as conn:
            conn.execute("UPDATE runtime_inputs SET in_use=1,lease_until=0 WHERE context_id=?", (ctx.context_id,))
        self.context()
        with self.assertRaises(ContextExpired):
            self.h.get_prediction_context(ctx.context_id)

    def test_future_task_text_does_not_fill_an_unknown_old_execution(self):
        self.h.snapshot(TASK, "e")
        record = ExecutionRecord(execution_id="old", task_id="t", strategy_id="candidate",
                                 profile_snapshot=self.h.profile(TASK), created_at=1)
        self.h.record(record)
        self.assertIsNone(self.h.bank.get("old").task_text_digest)

    def test_missing_legacy_work_time_does_not_become_today_for_hplus(self):
        raw = ExecutionRecord(execution_id="untimed", task_id="t", strategy_id="candidate",
                              profile_snapshot=self.h.profile(TASK)).to_dict()
        raw.pop("created_at")
        record = ExecutionRecord.from_dict(raw)
        self.h.bank.append(record)
        from or_harness.world_model.maintenance_decision import _episode_execution_span
        span = _episode_execution_span(self.h, "t", None)
        self.assertTrue(span["timing_unknown"])
        self.assertIsNone(span["finished_at"])
        self.assertTrue(self.h.bank.get("untimed").execution_features["created_at_unknown"])

    def test_adopted_knowledge_ref_is_the_value_before_the_executor(self):
        entry = StrategicEntry(entry_id="", strategy_id="candidate", expected_quality_hat=0.4,
                               pattern={"predicates": {"family": "routing"}})
        key = self.h.sbank.add(entry)
        work = Path(self.tmp.name) / "work"
        work.mkdir()
        script = work / "solve.py"
        script.write_text("pass\n")
        def executor(*args, **kwargs):
            entry.expected_quality_hat = 0.9
            self.h.sbank.update(entry)
            return ExecutionRecord(execution_id="ex_adopted", task_id="t", strategy_id="candidate",
                profile_snapshot=self.h.profile(TASK), quality={"status": "optimal", "feasible": True, "gap": 0})
        with patch.object(self.h, "_run_executor", side_effect=executor):
            record = self.h.execute(TASK, "candidate", str(script), str(work), solver="highs",
                                    episode_id="e", used_entry_ids=[key])
        ref = record.execution_features["used_entries"]["versions"][key]
        self.assertEqual(get_value(self.h.store, ref)["expected"]["quality_hat"], 0.4)
        self.assertIsNone(record.method_actual)

    def test_artifact_io_error_does_not_turn_an_observed_success_into_failure(self):
        work = Path(self.tmp.name) / "work"
        work.mkdir()
        script = work / "solve.py"
        script.write_text("import json\nprint('observed log')\njson.dump({'status':'optimal','objective_value':1,'objective_bound':1},open('result.json','w'))\n")
        from or_harness.core.artifacts import capture
        calls = []
        def fail_output(*args, **kwargs):
            calls.append(kwargs["kind"])
            if len(calls) > 1:
                raise OSError("artifact disk unavailable")
            return capture(*args, **kwargs)
        with patch("or_harness.core.artifacts.capture", side_effect=fail_output):
            record = self.h.execute(TASK, "candidate", str(script), str(work), solver="highs", episode_id="e")
        self.assertEqual(record.quality["status"], "optimal")
        self.assertIn("artifact_capture_missing", record.execution_features)
        self.assertEqual(self.h.bank.get_pending(record.execution_id).quality["status"], "optimal")

    def test_cli_active_context_roundtrip_across_processes(self):
        env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2] / "src"),
                   PYTHONDONTWRITEBYTECODE="1", OR_EMBEDDING_BACKEND="none")
        command = [sys.executable, "-m", "or_harness.cli", "--home", self.home, "context"]
        built = json.loads(subprocess.check_output(command + ["--task", json.dumps(TASK), "--episode", "e"], env=env))["result"]
        loaded = json.loads(subprocess.check_output(command + ["--context-id", built["context_id"]], env=env))["result"]
        self.assertEqual(built, loaded)
        release_context(self.h.store, built["context_id"])
        result = subprocess.run(command + ["--context-id", built["context_id"]], env=env,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("expired", json.loads(result.stdout)["result"]["error"])
        audit = json.loads(subprocess.check_output(command + ["--context-id", built["context_id"], "--audit"], env=env))["result"]
        self.assertEqual(audit["lifecycle"], "consumed")
        self.assertNotIn("joint", audit)

    def test_plan_choose_execute_across_cli_processes(self):
        from http.server import BaseHTTPRequestHandler, HTTPServer
        import threading
        payload = Stub().predict({})["payload"]
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                data = json.dumps({"choices": [{"message": {"content": json.dumps(payload)}}],
                                   "usage": {"prompt_tokens": 7, "completion_tokens": 3}}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            def log_message(self, *args):
                pass
        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        work = Path(self.tmp.name) / "cli-work"
        work.mkdir()
        task_file, candidates_file = work / "task.json", work / "candidates.json"
        task_file.write_text(json.dumps(TASK))
        candidates_file.write_text(json.dumps([CANDIDATE]))
        script = work / "solve.py"
        script.write_text("import json\njson.dump({'status':'optimal','objective_value':1,'objective_bound':1},open('result.json','w'))\n")
        env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2] / "src"),
                   PYTHONDONTWRITEBYTECODE="1", OR_EMBEDDING_BACKEND="none", OR_WM_API_KEY="local-fixture")
        base = [sys.executable, "-m", "or_harness.cli", "--home", self.home]
        def cli(arguments):
            result = subprocess.run(base + arguments, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            return json.loads(result.stdout)["result"]
        try:
            plan = cli(["--world-model", f"http://127.0.0.1:{server.server_port}::fixture", "plan-next",
                        "--task", str(task_file), "--candidates", str(candidates_file), "--episode", "e"])["plan"]
            prediction_id = plan["candidates"][0]["prediction_id"]
            cli(["choose-next", "--decision", plan["decision_action_id"], "--prediction", prediction_id])
            with self.assertRaises(ContextExpired):
                self.h.get_prediction_context(plan["prediction_context_id"])
            executed = cli(["execute", "--task", str(task_file), "--prediction", prediction_id,
                            "--code", str(script), "--workspace", str(work), "--episode", "e"])
            self.assertTrue(executed.get("execution_id"))
            self.assertEqual(self.count("belief_snapshots"), 0)
            self.assertEqual(self.count("runtime_inputs"), 0)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_cleanup_preview_backup_compaction_and_idempotence(self):
        entry = StrategicEntry(entry_id="", strategy_id="candidate", pattern={"predicates": {"family": "routing"}})
        key = self.h.sbank.add(entry)
        snap = self.legacy_snapshot()
        self.h.store.put_closeout_registry("t", "closed", closed_at=1, terminal_state="completed",
            evaluation_ids=[], published=False)
        before = self.h.store.conn.execute("SELECT payload FROM belief_snapshots").fetchone()[0]
        preview = maintain_snapshots(self.home)
        self.assertEqual(preview["candidate_rows"], 1)
        self.assertEqual(before, self.h.store.conn.execute("SELECT payload FROM belief_snapshots").fetchone()[0])
        backup = str(Path(self.tmp.name) / "backup.db")
        result = maintain_snapshots(self.home, apply=True, backup=backup)
        self.assertEqual(result["compacted_rows"], 1)
        self.assertEqual(self.count("belief_snapshots"), 0)
        receipt = self.h.get_snapshot(snap.snapshot_id)
        self.assertEqual(receipt.problem_state, snap.problem_state)
        self.assertIn(key, _snapshot_knowledge_entry_ids(receipt, store=self.h.store))
        with sqlite3.connect(backup) as original:
            self.assertEqual(original.execute("SELECT payload FROM belief_snapshots").fetchone()[0], before)
        again = maintain_snapshots(self.home, apply=True, backup=str(Path(self.tmp.name) / "backup2.db"))
        self.assertEqual(again["compacted_rows"], 0)

    def test_cleanup_protects_unfinished_and_unresolvable_history(self):
        snap = self.legacy_snapshot(episode="open")
        self.assertIn(snap.snapshot_id, maintain_snapshots(self.home)["protected"])
        self.h.store.put_closeout_registry("t", "open", closed_at=1, terminal_state="completed",
            evaluation_ids=[], published=False)
        with self.h.store.transaction() as conn:
            conn.execute("INSERT INTO prediction_contexts VALUES (?,?,?,?,?)", (
                "future", "t", "open", 1, json.dumps({"context_id": "future", "context_version": "future"})))
        result = maintain_snapshots(self.home)
        self.assertIn("unsupported context version", result["protected"][snap.snapshot_id])

    def test_cleanup_protects_unresolved_old_meta_reference(self):
        snap = self.legacy_snapshot()
        self.h.store.put_closeout_registry("t", "closed", closed_at=1, terminal_state="completed",
            evaluation_ids=[], published=False)
        with self.h.store.transaction() as conn:
            conn.execute("INSERT INTO meta VALUES (?,?)", ("unknown_consumer", json.dumps({"ref": snap.snapshot_id})))
        result = maintain_snapshots(self.home)
        self.assertIn("unresolved meta reference: unknown_consumer", result["protected"][snap.snapshot_id])

    def test_explicit_inactive_view_expiry_preserves_pending_attempt(self):
        snap = self.legacy_snapshot(episode="open")
        raw = snap.to_dict()
        raw["created_at"] = 1
        with self.h.store.transaction() as conn:
            conn.execute("UPDATE belief_snapshots SET created_at=1,payload=? WHERE snapshot_id=?",
                         (self.h.store.dumps(raw), snap.snapshot_id))
        record = ExecutionRecord(execution_id="pending", task_id="t", strategy_id="candidate",
                                 profile_snapshot=self.h.profile(TASK), created_at=1)
        self.h.bank.stage_pending(record)
        self.assertEqual(maintain_snapshots(self.home)["candidate_rows"], 0)
        preview = maintain_snapshots(self.home, expire_open_days=30)
        self.assertEqual(preview["candidate_rows"], 1)
        backup = str(Path(self.tmp.name) / "expire-backup.db")
        maintain_snapshots(self.home, apply=True, backup=backup, expire_open_days=30)
        self.assertIsNotNone(self.h.bank.get_pending("pending"))
        self.assertEqual(self.h.get_snapshot(snap.snapshot_id).coverage["view_lifecycle"], "expired_maintenance")

    def test_fixed_base_question_a_does_not_change_b_memory(self):
        seed = ExecutionRecord(execution_id="ex_base", task_id="t", strategy_id="candidate",
            profile_snapshot=self.h.profile(TASK), quality={"feasible": True, "gap": 0, "status": "optimal"},
            task_text_digest=self.h.capture_task_text(TASK))
        self.h.record(seed)
        self.h.sbank.add(StrategicEntry(entry_id="", strategy_id="candidate",
            pattern={"predicates": {"family": "routing"}}))
        self.h.index_sync.sync_entries()
        self.h.close_episode("t")
        self.context()
        published = self.h.store.conn.execute("SELECT value FROM meta WHERE key='calibration_summary|published'").fetchone()[0]
        self.h.close()
        base = Path(self.home) / "or_harness.db"
        before = hashlib.sha256(base.read_bytes()).hexdigest()
        a_home, b_home = str(Path(self.tmp.name) / "a"), str(Path(self.tmp.name) / "b")
        fork_memory(self.home, a_home)
        a = ORHarness(home=a_home, world_model=Stub(), embedding=LocalHashEmbeddingBackend())
        try:
            record = ExecutionRecord(execution_id="ex_a", task_id="a", strategy_id="candidate",
                profile_snapshot=a.profile(TASK), quality={"feasible": True, "gap": 0},
                task_text_digest=a.capture_task_text(dict(TASK, task_id="a")))
            a.record(record)
            a.snapshot(TASK, "retry", task_progress={"current_solution": {"value": 42}})
            self.assertEqual(a.snapshot(TASK, "retry").task_progress["current_solution"]["value"], 42)
            a.close_episode("a")
        finally:
            a.close()
        fork_memory(self.home, b_home)
        b = ORHarness(home=b_home, world_model=Stub(), embedding=LocalHashEmbeddingBackend())
        try:
            self.assertEqual(b.bank.count(), 1)
            self.assertEqual(b.stats.for_profile(b.profile(TASK))["candidate"].n, 1)
            self.assertEqual(b.sbank.count(), 1)
            self.assertEqual(b.store.conn.execute("SELECT value FROM meta WHERE key='calibration_summary|published'").fetchone()[0], published)
            from or_harness.strategy.embedding_index import INDEX_SUBDIR
            for original in (Path(self.home) / INDEX_SUBDIR).glob("*.json"):
                self.assertEqual((Path(b_home) / INDEX_SUBDIR / original.name).read_bytes(), original.read_bytes())
        finally:
            b.close()
        self.assertEqual(hashlib.sha256(base.read_bytes()).hexdigest(), before)


if __name__ == "__main__":
    unittest.main()
