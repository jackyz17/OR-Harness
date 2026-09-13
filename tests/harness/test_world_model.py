"""World-model M1 substrate: state, actions, budget, knowledge view.

Covers the eight verification requirements from the implementation plan:
snapshot freeze isolation, recommendation/selection/execution separation,
the seven action classes, real/hypothetical isolation, cost dedup and
idempotency, legacy compatibility, episode scoping, and the verified-
knowledge view (WM-K1).
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from or_harness.adapters.solver import available_families  # noqa: E402
from or_harness.api import ORHarness  # noqa: E402
from or_harness.core.schema import (  # noqa: E402
    COST_DIMENSIONS,
    CostVector,
    ExecutionRecord,
    StrategicEntry,
    empty_verification,
)
from or_harness.core.storage import Store  # noqa: E402
from or_harness.strategy.selector import is_publishable  # noqa: E402
from or_harness.world_model.actions import (  # noqa: E402
    ACTION_TYPES,
    ActionLog,
    ActionRecord,
)
from or_harness.world_model.budget import BudgetLedger  # noqa: E402
from or_harness.world_model.state import (  # noqa: E402
    MAINTENANCE_TASK_ID,
    BeliefSnapshot,
    verified_knowledge_view,
)

from helpers import HarnessTestCase  # noqa: E402


def _task(task_id="t1", family="routing", **coupling):
    values = {"resource_coupling": 0.9, "temporal_coupling": 0.1,
              "route_complexity": 0.85}
    values.update(coupling)
    return {"task_id": task_id, "family": family,
            "spec": {"n_vars": 100, "n_constraints": 50},
            "annotations": {"coupling": values}}


class TestBeliefSnapshot(HarnessTestCase):
    """Requirement 1: freeze isolation at all three boundaries."""

    def _harness(self):
        h = ORHarness(home=self.home)
        self.addCleanup(h.close)
        return h

    def test_snapshot_isolated_from_caller_originals(self):
        h = self._harness()
        progress = {"selected_plan": {"value": {"strategy_id": "S01"},
                                      "provenance": "agent_reported",
                                      "epistemic": "fact"}}
        task = _task()
        snap = h.snapshot(task, "ep1", task_progress=progress)
        # Mutate the caller's originals AFTER the snapshot was taken.
        progress["selected_plan"]["value"]["strategy_id"] = "S09"
        task["task_id"] = "t-hijacked"
        reloaded = h.get_snapshot(snap.snapshot_id)
        self.assertEqual(reloaded.task_progress["selected_plan"]["value"]
                         ["strategy_id"], "S01")
        self.assertEqual(reloaded.task_id, "t1")

    def test_snapshot_isolated_from_to_dict_mutation(self):
        h = self._harness()
        snap = h.snapshot(_task(), "ep1")
        dumped = snap.to_dict()
        dumped["problem_state"]["profile"]["family"] = "hijacked"
        dumped["harness_state"]["experience"]["total_executions"] = 999
        reloaded = h.get_snapshot(snap.snapshot_id)
        self.assertEqual(
            reloaded.problem_state["profile"]["family"], "routing")
        self.assertNotEqual(
            reloaded.harness_state["experience"]["total_executions"], 999)

    def test_snapshot_isolated_from_loaded_mutation(self):
        h = self._harness()
        snap = h.snapshot(_task(), "ep1")
        loaded = h.get_snapshot(snap.snapshot_id)
        loaded.coverage["knowledge_layers"] = {"verified": ["fake"]}
        loaded.problem_state["task_digest"] = "deadbeef"
        again = h.get_snapshot(snap.snapshot_id)
        self.assertEqual(again.coverage.get("knowledge_layers", {})
                         .get("verified", []), [])
        self.assertNotEqual(again.problem_state["task_digest"], "deadbeef")

    def test_snapshot_not_changed_by_later_bank_writes(self):
        h = self._harness()
        task = _task()
        snap = h.snapshot(task, "ep1")
        # Later writes: a new execution fact + a new strategic entry.
        h.bank.append(self.make_record(task_id="t1", strategy_id="S01"))
        entry = StrategicEntry(
            entry_id=StrategicEntry.new_id(), strategy_id="S01",
            pattern={"predicates": {"family": "routing"}},
            verification={"state": "verified", "claim": "x"})
        h.sbank.add(entry)
        reloaded = h.get_snapshot(snap.snapshot_id)
        self.assertEqual(
            reloaded.harness_state["experience"]["total_executions"], 0)
        self.assertEqual(reloaded.coverage["knowledge_layers"]["verified"], [])

    def test_same_profile_different_task_digest(self):
        """P is more than the profile: two tasks with identical coupling
        annotations get different digests."""
        h = self._harness()
        t1 = _task("t1")
        t2 = _task("t2")  # same structure, different task
        t2["spec"] = {"n_vars": 200, "n_constraints": 50}
        s1 = h.snapshot(t1, "ep1")
        s2 = h.snapshot(t2, "ep1")
        self.assertNotEqual(s1.problem_state["task_digest"],
                            s2.problem_state["task_digest"])

    def test_origin_labels_are_orthogonal(self):
        # Build an UNFROZEN snapshot to exercise the labelled setters.
        snap = BeliefSnapshot(
            snapshot_id=BeliefSnapshot.new_id(), task_id="t1",
            episode_id="ep1", created_at=0.0)
        snap.set_progress("understanding", {"cir": True},
                           provenance="agent_reported", epistemic="inferred")
        field = snap.task_progress["understanding"]
        self.assertEqual(field["provenance"], "agent_reported")
        self.assertEqual(field["epistemic"], "inferred")
        with self.assertRaises(ValueError):
            snap.set_progress("x", 1, provenance="bogus")
        with self.assertRaises(ValueError):
            snap.set_progress("x", 1, epistemic="bogus")
        snap.freeze()
        with self.assertRaises(RuntimeError):
            snap.set_progress("late", 1)


class TestActionLog(HarnessTestCase):
    """Requirements 2/3/5: seven action classes, lifecycle, idempotency."""

    def setUp(self):
        super().setUp()
        self.log = ActionLog(self.store)

    def test_all_seven_action_types_recordable(self):
        for action_type in ACTION_TYPES:
            record = self.log.begin_action(action_type, "t1", "ep1")
            self.assertEqual(record.status, "running")
            ended = self.log.end_action(record.action_id,
                                        status="completed")
            self.assertEqual(ended.status, "completed")

    def test_running_action_visible_after_interrupt(self):
        record = self.log.begin_action("understand", "t1", "ep1")
        running = self.log.running(task_id="t1")
        self.assertEqual([r.action_id for r in running],
                         [record.action_id])

    def test_end_action_idempotent_replay_same_content(self):
        record = self.log.begin_action("model", "t1", "ep1")
        cost = CostVector(llm_tokens=100, measured={"llm_tokens"})
        first = self.log.end_action(record.action_id, status="completed",
                                    outcome={"digest": "abc"}, cost=cost)
        again = self.log.end_action(record.action_id, status="completed",
                                    outcome={"digest": "abc"}, cost=cost)
        self.assertEqual(first.action_id, again.action_id)
        self.assertEqual(self.log.query(task_id="t1").__len__(), 1)

    def test_end_action_conflict_on_different_content(self):
        record = self.log.begin_action("model", "t1", "ep1")
        self.log.end_action(record.action_id, status="completed",
                            outcome={"digest": "abc"})
        from or_harness.core.storage import StorageError
        with self.assertRaises(StorageError):
            self.log.end_action(record.action_id, status="failed",
                                outcome={"digest": "abc"})

    def test_duplicate_action_id_rejected(self):
        record = self.log.begin_action("verify", "t1", "ep1")
        from or_harness.core.storage import StorageError
        with self.assertRaises(StorageError):
            self.log._insert(record)

    def test_report_action_marks_missing_pre_snapshot(self):
        record = self.log.report_action("select_strategy", "t1", "ep1",
                                        outcome={"strategy_id": "S04"})
        self.assertTrue(record.outcome.get("pre_snapshot_missing"))
        self.assertEqual(record.source, "agent_reported")

    def test_report_action_rejects_execute_strategy(self):
        with self.assertRaises(ValueError):
            self.log.report_action("execute_strategy", "t1", "ep1")

    def test_amend_action_cost_idempotent(self):
        record = self.log.begin_action("select_strategy", "t1", "ep1")
        self.log.end_action(record.action_id, status="completed")
        self.log.amend_action_cost(record.action_id, llm_tokens=500)
        again = self.log.amend_action_cost(record.action_id, llm_tokens=500)
        self.assertEqual(again.cost.llm_tokens, 500.0)
        self.assertIn("llm_tokens", again.cost.measured_dims())

    def test_failure_statuses_recordable(self):
        for status in ("failed", "cancelled", "timeout", "no_valid_entry"):
            record = self.log.begin_action("induce", "t1", "ep1")
            ended = self.log.end_action(record.action_id, status=status)
            self.assertEqual(ended.status, status)


class TestBudgetLedger(HarnessTestCase):
    """Requirement 5: cost dedup, staged inclusion, honest statuses."""

    def setUp(self):
        super().setUp()
        self.h = ORHarness(home=self.home)
        self.addCleanup(self.h.close)
        self.ledger = BudgetLedger(self.h.bank, self.h.actions)

    def test_staged_execution_counts_until_recorded(self):
        rec = self.make_record(task_id="t1", strategy_id="S01")
        self.h.bank.stage_pending(rec)
        view = self.ledger.view("t1")
        self.assertEqual(view["consumption"]["n_staged"], 1)
        self.assertEqual(view["consumption"]["n_attempts"], 1)
        # Record it: the same execution moves from staged to recorded —
        # still counted exactly once.
        self.h.bank.append(rec)
        self.h.bank.clear_pending(rec.execution_id)
        view = self.ledger.view("t1")
        self.assertEqual(view["consumption"]["n_staged"], 0)
        self.assertEqual(view["consumption"]["n_recorded"], 1)
        self.assertEqual(view["consumption"]["n_attempts"], 1)

    def test_budget_statuses(self):
        rec = self.make_record(task_id="t1", strategy_id="S01",
                               cost=CostVector(llm_tokens=100,
                                               solver_runtime_s=10.0,
                                               measured={"llm_tokens",
                                                         "solver_runtime_s"}))
        self.h.bank.append(rec)
        # No budget declared.
        self.assertEqual(self.ledger.view("t1")["status"],
                         "no_budget_declared")
        # Unknown dims + within limits => unconfirmed, NOT ok.
        view = self.ledger.view("t1", budget={"llm_tokens": 1000,
                                              "solver_runtime_s": 100,
                                              "tool_calls": 5,
                                              "retries": 1,
                                              "latency_s": 10})
        self.assertEqual(view["status"], "unconfirmed")
        # Measured over limit => exceeded.
        view = self.ledger.view("t1", budget={"llm_tokens": 50})
        self.assertEqual(view["status"], "exceeded")
        # All dims measured and within => ok.
        rec2 = self.make_record(task_id="t1", strategy_id="S01")
        self.h.bank.append(rec2)
        view = self.ledger.view("t1", budget={
            d: 1e9 for d in COST_DIMENSIONS})
        # rec2 is fully metered but rec measured only llm_tokens and
        # solver_runtime_s — those two stay unknown overall, so the honest
        # status is still unconfirmed. Replace rec with a fully metered
        # one to reach ok.
        self.h.bank.append(self.make_record(task_id="t1", strategy_id="S01",
                                            cost=None))
        # (the default cost in make_record is fully metered; rec above is
        # the partial one — remove its effect by checking a fresh task.)
        view = self.ledger.view("t1", budget={
            d: 1e9 for d in COST_DIMENSIONS})
        self.assertEqual(view["status"], "unconfirmed")

    def test_fully_measured_budget_ok(self):
        for _ in range(2):
            self.h.bank.append(self.make_record(task_id="tb",
                                                strategy_id="S01"))
        view = self.ledger.view("tb", budget={d: 1e9
                                              for d in COST_DIMENSIONS})
        self.assertEqual(view["status"], "ok")

    def test_own_cost_actions_counted_macro_reference_not(self):
        # An own-cost select action adds to consumption.
        action = self.h.actions.begin_action("select_strategy", "t1", "ep1")
        self.h.actions.end_action(
            action.action_id, status="completed",
            cost=CostVector(llm_tokens=200, measured={"llm_tokens"}))
        # A macro execute action with rollup=reference adds nothing.
        macro = self.h.actions.begin_action("execute_strategy", "t1", "ep1")
        self.h.actions.end_action(
            macro.action_id, status="completed",
            cost=CostVector(llm_tokens=999, measured={"llm_tokens"}),
            rollup="reference")
        view = self.ledger.view("t1")
        self.assertEqual(view["consumption"]["total_cost"]["llm_tokens"],
                         200.0)

    def test_hypothetical_excluded_from_budget(self):
        action = self.h.actions.begin_action("select_strategy", "t1", "ep1",
                                             source="hypothetical")
        self.h.actions.end_action(
            action.action_id, status="completed",
            cost=CostVector(llm_tokens=5000, measured={"llm_tokens"}))
        view = self.ledger.view("t1")
        self.assertIsNone(view["consumption"]["total_cost"]["llm_tokens"])


class TestVerifiedKnowledgeView(HarnessTestCase):
    """Requirement 4/WM-K1: admission gating in the knowledge view."""

    def _entry(self, verification):
        return StrategicEntry(
            entry_id=StrategicEntry.new_id(), strategy_id="S01",
            pattern={"predicates": {"family": "routing"}},
            verification=verification)

    def test_layering_verified_legacy_unverified(self):
        h = ORHarness(home=self.home)
        self.addCleanup(h.close)
        verified = self._entry({"state": "verified", "claim": "c1"})
        legacy = self._entry({})  # no block at all
        unverified = self._entry({"state": "unverified"})
        refuted = self._entry({"state": "refuted"})
        for e in (verified, legacy, unverified, refuted):
            h.sbank.add(e)
        profile = self.make_profile()
        layers = verified_knowledge_view(profile, h.sbank)
        self.assertEqual([k["entry_id"] for k in layers["verified"]],
                         [verified.entry_id])
        self.assertEqual([k["entry_id"] for k in layers["legacy_unknown"]],
                         [legacy.entry_id])
        self.assertEqual(
            sorted(k["entry_id"] for k in layers["unverified"]),
            sorted([unverified.entry_id, refuted.entry_id]))

    def test_stale_after_revision_not_publishable(self):
        entry = self._entry({"state": "verified", "claim": "c1"})
        entry.verification["stale_after_revision"] = True
        self.assertFalse(is_publishable(entry))
        # Without the stale flag it would be publishable.
        clean = self._entry({"state": "verified"})
        self.assertTrue(is_publishable(clean))

    def test_substantive_revision_marks_stale(self):
        """An induced entry whose claim substantively changes without a
        fresh verdict loses publishability until re-verified."""
        h = ORHarness(home=self.home)
        self.addCleanup(h.close)
        # Two tasks, same cell, strategy S01.
        for task_id in ("t1", "t2"):
            h.bank.append(self.make_record(
                task_id=task_id, strategy_id="S01",
                profile=self.make_profile(problem_id=task_id)))
        # First induction WITH a verification verdict.
        result = h.induce(strategy_id="S01", verify={
            "purpose": "rule", "claim": "S01 solves routing",
            "check": {"reference_status": "optimal"},
            "executions": [r.to_dict() for r in h.bank.query()],
        })
        entry_id = result["results"][0]["created"]
        self.assertEqual(h.sbank.get(entry_id).verification_state,
                         "verified")
        self.assertTrue(is_publishable(h.sbank.get(entry_id)))
        # New evidence that MOVES the claim (a failure: quality drops).
        rec = self.make_record(task_id="t3", strategy_id="S01",
                               profile=self.make_profile(problem_id="t3"),
                               feasible=False, status="error")
        h.bank.append(rec)
        h.induce(strategy_id="S01")
        entry = h.sbank.get(entry_id)
        self.assertTrue(entry.verification.get("stale_after_revision"))
        self.assertFalse(is_publishable(entry))
        # Re-verify (on the executions that can support the claim — the
        # failed t3 execution honestly yields insufficient_evidence for an
        # optimal-status check): publishable again.
        h.induce(strategy_id="S01", verify={
            "purpose": "rule", "claim": "S01 solves routing",
            "check": {"reference_status": "optimal"},
            "executions": [r.to_dict() for r in h.bank.query()
                           if r.quality.get("status") == "optimal"],
        })
        entry = h.sbank.get(entry_id)
        self.assertFalse(entry.verification.get("stale_after_revision"))
        self.assertTrue(is_publishable(entry))

    def test_support_growth_alone_does_not_stale(self):
        """More evidence for the SAME claim only sharpens it."""
        h = ORHarness(home=self.home)
        self.addCleanup(h.close)
        for task_id in ("t1", "t2"):
            h.bank.append(self.make_record(
                task_id=task_id, strategy_id="S01",
                profile=self.make_profile(problem_id=task_id)))
        result = h.induce(strategy_id="S01", verify={
            "purpose": "rule", "claim": "c",
            "check": {"reference_status": "optimal"},
            "executions": [r.to_dict() for r in h.bank.query()],
        })
        entry_id = result["results"][0]["created"]
        # A third task with the SAME quality profile (no claim change).
        h.bank.append(self.make_record(
            task_id="t3", strategy_id="S01",
            profile=self.make_profile(problem_id="t3")))
        h.induce(strategy_id="S01")
        entry = h.sbank.get(entry_id)
        self.assertFalse(entry.verification.get("stale_after_revision"))
        self.assertTrue(is_publishable(entry))


class TestHarnessIntegration(HarnessTestCase):
    """Requirements 2/3/7/8: API integration, episodes, end-to-end."""

    def _harness(self):
        h = ORHarness(home=self.home)
        self.addCleanup(h.close)
        return h

    def test_execute_records_macro_action_with_snapshots(self):
        h = self._harness()
        task = _task()
        # A trivially succeeding solve script.
        from pathlib import Path
        script = Path(self.home) / "solve_ok.py"
        script.write_text(
            "import json\n"
            "json.dump({'status': 'optimal', 'objective_value': 10.0,\n"
            "           'objective_bound': 10.0, 'mip_gap': 0.0,\n"
            "           'runtime_seconds': 0.1}, open('result.json', 'w'))\n")
        record = h.execute(task, "S01", str(script), self.home,
                           solver="highs", episode_id="ep1")
        self.assertIsNotNone(record.action_id)
        action = h.actions.get(record.action_id)
        self.assertEqual(action.action_type, "execute_strategy")
        self.assertEqual(action.linked_execution_id, record.execution_id)
        self.assertEqual(action.rollup, "reference")
        self.assertIsNotNone(action.pre_snapshot_id)
        self.assertIsNotNone(action.post_snapshot_id)
        # Post snapshot carries the solution state.
        post = h.get_snapshot(action.post_snapshot_id)
        self.assertIn("current_solution", post.task_progress)
        # The action is NOT the record: recording is still separate.
        self.assertEqual(h.bank.count(), 0)
        self.assertEqual(len(h.bank.pending(task_id="t1")), 1)

    def test_recommendation_not_auto_interpreted_as_selection(self):
        """recall returns suggestions; no select action appears unless the
        agent reports one."""
        h = self._harness()
        h.bank.append(self.make_record(task_id="t1", strategy_id="S01"))
        result = h.recall(_task())
        self.assertTrue(result["recommendations"])
        # No select_strategy action was created by recall.
        self.assertEqual(
            h.actions.query(action_type="select_strategy"), [])

    def test_report_action_via_api(self):
        h = self._harness()
        out = h.report_action("select_strategy", _task(), "ep1",
                              outcome={"strategy_id": "S04",
                                       "basis": "structural fit"})
        action = h.actions.get(out["action_id"])
        self.assertEqual(action.source, "agent_reported")
        self.assertEqual(action.outcome["strategy_id"], "S04")

    def test_induce_action_in_maintenance_scope(self):
        h = self._harness()
        for task_id in ("t1", "t2"):
            h.bank.append(self.make_record(
                task_id=task_id, strategy_id="S01",
                profile=self.make_profile(problem_id=task_id)))
        result = h.induce(strategy_id="S01")
        action_info = result["action"]
        action = h.actions.get(action_info["action_id"])
        self.assertEqual(action.task_id, MAINTENANCE_TASK_ID)
        self.assertTrue(action.episode_id.startswith("maint_"))
        self.assertIn(action_info["business_result"],
                      ("created", "updated", "revised", "refused",
                       "unchanged"))
        # Refused induction (single task) is no_valid_entry, not failed.
        h2 = ORHarness(home=self.home + "_2")
        self.addCleanup(h2.close)
        h2.bank.append(self.make_record(task_id="only", strategy_id="S01"))
        refused = h2.induce(strategy_id="S01")
        self.assertEqual(refused["action"]["business_result"], "refused")
        self.assertEqual(
            h2.actions.get(refused["action"]["action_id"]).status,
            "no_valid_entry")

    def test_induce_dry_run_persists_nothing(self):
        h = self._harness()
        for task_id in ("t1", "t2"):
            h.bank.append(self.make_record(
                task_id=task_id, strategy_id="S01",
                profile=self.make_profile(problem_id=task_id)))
        h.induce(strategy_id="S01", dry_run=True)
        self.assertEqual(h.actions.query(action_type="induce"), [])
        self.assertEqual(h.snapshots(), [])

    def test_episode_isolation_and_harness_persistence(self):
        """Requirement 7: independent tasks reset task context; harness
        experience survives."""
        h = self._harness()
        h.bank.append(self.make_record(task_id="t1", strategy_id="S01"))
        # New task, new episode: budget scoped, experience kept.
        # Declare a MULTI-dimension budget: llm_tokens known and within,
        # solver_runtime_s declared but unmeasured => unconfirmed (NOT ok).
        h.declare_budget("t2", {"llm_tokens": 1000,
                                "solver_runtime_s": 100}, episode_id="ep-a")
        rec = self.make_record(
            task_id="t2", strategy_id="S01",
            cost=CostVector(llm_tokens=100.0,
                            measured={"llm_tokens"}))
        h.bank.append(rec)
        view = h.budget_view("t2", episode_id="ep-a")
        self.assertEqual(view["status"], "unconfirmed")
        # Harness experience is RETAINED across tasks: t1's execution is
        # still visible to t2's snapshot (total = t1 + t2's own).
        snap = h.snapshot(_task("t2"), "ep-a")
        self.assertEqual(
            snap.harness_state["experience"]["total_executions"], 2)
        # Task-scoped count: only t2's own execution (t1's is retained in
        # the TOTAL, not in this task's context).
        self.assertEqual(
            snap.harness_state["experience"]["task_execution_count"], 1)

    def test_end_to_end_flow(self):
        """Requirement 8: snapshot -> selection -> execution -> record ->
        post-state/budget query."""
        h = self._harness()
        task = _task()
        # 1. Initial snapshot.
        snap1 = h.snapshot(task, "ep1")
        # 2. Explicit selection (agent action).
        sel = h.report_action("select_strategy", task, "ep1",
                              outcome={"strategy_id": "S01",
                                       "solver": "highs"})
        # 3. Real execution + record.
        from pathlib import Path
        script = Path(self.home) / "solve_e2e.py"
        script.write_text(
            "import json\n"
            "json.dump({'status': 'optimal', 'objective_value': 42.0,\n"
            "           'objective_bound': 42.0, 'mip_gap': 0.0,\n"
            "           'runtime_seconds': 0.2}, open('result.json', 'w'))\n")
        record = h.execute(task, "S01", str(script), self.home,
                           solver="highs", episode_id="ep1")
        h.record(record, override={"llm_tokens": 1500})
        # 4. Post-state / budget queries.
        action = h.actions.get(record.action_id)
        post = h.get_snapshot(action.post_snapshot_id)
        self.assertEqual(post.task_progress["current_solution"]["value"]
                         ["status"], "optimal")
        budget = h.budget_view("t1", episode_id="ep1")
        self.assertEqual(budget["consumption"]["n_recorded"], 1)
        self.assertEqual(budget["consumption"]["total_cost"]["llm_tokens"],
                         1500.0)
        # Selection is distinct from execution.
        sel_action = h.actions.get(sel["action_id"])
        self.assertEqual(sel_action.action_type, "select_strategy")
        self.assertIsNone(sel_action.linked_execution_id)


class TestLegacyCompatibility(HarnessTestCase):
    """Requirement 6: old databases and old payloads."""

    def test_old_db_gains_new_tables_idempotently(self):
        # A store created and closed twice: schema init is idempotent.
        Store(self.home).close()
        store2 = Store(self.home)
        self.addCleanup(store2.close)
        row = store2.conn.execute(
            "SELECT value FROM meta WHERE key='schema_version'").fetchone()
        self.assertEqual(row["value"], "wm1")
        # Tables exist and are queryable.
        store2.conn.execute("SELECT COUNT(*) FROM belief_snapshots")
        store2.conn.execute("SELECT COUNT(*) FROM action_records")

    def test_action_record_roundtrip(self):
        log = ActionLog(self.store)
        record = log.begin_action("verify", "t1", "ep1",
                                  params={"level": "basic"})
        log.end_action(record.action_id, status="completed",
                       outcome={"passed": True},
                       cost=CostVector(tool_calls=1, measured={"tool_calls"}))
        loaded = log.get(record.action_id)
        self.assertEqual(loaded.params, {"level": "basic"})
        self.assertEqual(loaded.outcome, {"passed": True})
        self.assertIn("tool_calls", loaded.cost.measured_dims())


if __name__ == "__main__":
    unittest.main()
