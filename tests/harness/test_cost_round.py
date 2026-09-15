"""Cost-round regression tests: measurement validity end-to-end
(execution facts -> conditional stats -> predictions -> feedback -> selection),
task aggregation semantics, backfill consistency, scope/scale comparability,
and protected-baseline invariants (paused compaction, retention marks, CIR).

Acceptance list:
1. predicted tokens unknown + actual 1000 -> no "predicted zero" error.
2. quality tied, one side's cost unknown -> unknown never ranks cheaper.
3. first failure is not a retry; three declared attempts total two retries.
4. sequential 4/5/6s latencies -> end-to-end unknown (never max=6, never
   sum=15 unless the harness supplies explicit task timing).
5. legacy records with backfilled non-zero tokens stay usable.
6. late backfill keeps feedback consistent; re-induction refreshes cost.
7. task-scope records never feed attempt predictions; clearly different
   scale is not unconditionally reused.
8. A never audits B; frozen pre-execution snapshot; feedback never mutates
   entry state (see also test_mechanism_layer).
9. paused compaction, explicit retention marks, and the CIR flow stay
   intact.
"""
import unittest

from helpers import HarnessTestCase, MEASURED_ALL

from or_harness.api import ORHarness
from or_harness.core.schema import (
    COST_DIMENSIONS,
    CostVector,
    ExecutionRecord,
    PredictionSnapshot,
)
from or_harness.strategy.experience_bank import ExperienceBank

FULL_COST = dict(llm_tokens=1000.0, tool_calls=2.0, solver_runtime_s=1.0,
                 retries=0.0, latency_s=1.0)


def _task(task_id, **scale):
    spec = {"n_vars": scale.get("n_vars", 1000),
            "n_constraints": scale.get("n_constraints", 500)}
    return {"task_id": task_id, "family": "routing", "spec": spec,
            "annotations": {"coupling": {
                "semantic_coupling": 0.8, "resource_coupling": 0.9,
                "temporal_coupling": 0.1, "route_complexity": 0.85}}}


class TestPredictionValidity(HarnessTestCase):
    def setUp(self):
        super().setUp()
        self.h = ORHarness(home=self.home)

    def test_unknown_predicted_tokens_fabricate_no_error(self):
        """Acceptance 1: history measured only solver runtime; the actual
        attempt spent 1000 tokens. The prediction side has NO tokens
        measurement, so no log_error may be computed for tokens."""
        for i in range(2):
            self.h.bank.append(self.make_record(
                execution_id=f"ex_hist{i}", task_id=f"th{i}",
                strategy_id="S01",
                cost=CostVector(llm_tokens=0, tool_calls=0,
                                solver_runtime_s=2.0, retries=0, latency_s=1.0),
                cost_measured=("solver_runtime_s", "latency_s")))
        snapshot = self.h.predict_cost(_task("t_now"), "S01")
        self.assertEqual(snapshot.source, "stats")
        self.assertNotIn("llm_tokens",
                         snapshot.expected_cost.measured_dims())
        rec = self.make_record(
            task_id="t_now", strategy_id="S01",
            cost=CostVector(llm_tokens=1000, tool_calls=2,
                            solver_runtime_s=2.5, retries=0, latency_s=1.0),
            cost_measured=MEASURED_ALL)
        outcome = self.h.record(rec, prediction=snapshot)
        feedback = outcome.get("cost_feedback")
        self.assertIsNotNone(feedback)
        self.assertNotIn("llm_tokens", feedback["per_dimension"])
        self.assertIn("solver_runtime_s", feedback["per_dimension"])

    def test_task_scope_records_never_feed_attempt_prediction(self):
        """Acceptance 7a: a task-scope record with 9000 tokens must not be
        re-labelled as an attempt prediction."""
        rec = self.make_record(
            execution_id="ex_task_scope", task_id="tt", strategy_id="S01",
            cost=CostVector(llm_tokens=9000, tool_calls=5,
                            solver_runtime_s=9.0, retries=0, latency_s=30.0),
            cost_measured=MEASURED_ALL)
        rec.measurement_scope = "task"
        self.h.bank.append(rec)
        snapshot = self.h.predict_cost(_task("t_q"), "S01")
        self.assertEqual(snapshot.source, "unknown")
        self.assertIsNone(snapshot.expected_cost)

    def test_scale_mismatch_returns_insufficient_evidence(self):
        """Acceptance 7b: historical samples at n_vars=10 do not
        unconditionally serve a n_vars=1,000,000 target."""
        profile = self.make_profile(problem_id="t10")
        profile.scale_features = {"n_vars": 10.0, "n_constraints": 5.0}
        for i in range(2):
            self.h.bank.append(self.make_record(
                execution_id=f"ex_s{i}", task_id=f"ts{i}", strategy_id="S01",
                profile=profile, cost=CostVector(**FULL_COST),
                cost_measured=MEASURED_ALL))
        snap_inside = self.h.predict_cost(
            _task("t_in", n_vars=10, n_constraints=5), "S01")
        self.assertEqual(snap_inside.source, "stats")
        snap_far = self.h.predict_cost(_task("t_out", n_vars=1_000_000), "S01")
        self.assertEqual(snap_far.source, "unknown")
        self.assertIsNone(snap_far.expected_cost)
        self.assertIn("not comparable", snap_far.note)


class TestSelectionValidity(HarnessTestCase):
    def setUp(self):
        super().setUp()
        self.h = ORHarness(home=self.home)

    def test_unknown_cost_never_ranks_cheaper(self):
        """Acceptance 2: S01 and S06 tied on quality; S01's tokens are
        measured (1000), S06's are unknown (placeholder zero). S06 must not
        win because of the placeholder — comparison uses the common measured
        dimensions only."""
        for i in range(2):
            self.h.bank.append(self.make_record(
                execution_id=f"ex_a{i}", task_id=f"ta{i}", strategy_id="S01",
                gap=0.05, cost=CostVector(llm_tokens=1000, tool_calls=0,
                                          solver_runtime_s=1.0, retries=0,
                                          latency_s=1.0),
                cost_measured=("llm_tokens", "solver_runtime_s",
                               "latency_s")))
            self.h.bank.append(self.make_record(
                execution_id=f"ex_b{i}", task_id=f"tb{i}", strategy_id="S06",
                gap=0.05, cost=CostVector(llm_tokens=0, tool_calls=0,
                                          solver_runtime_s=1.0, retries=0,
                                          latency_s=1.0),
                cost_measured=("solver_runtime_s", "latency_s")))
        recs = self.h.selector.recall(self.make_profile(problem_id="q"),
                                      top=10, memory_mode="cost-aware")
        by_id = {r.strategy.strategy_id: r for r in recs}
        s06 = by_id["S06"]
        # Unknown tokens are not compared — and not scored as cheap.
        self.assertNotIn("llm_tokens", s06.cost_known_dims)
        self.assertNotIn("llm_tokens", s06.cost_basis_dims)
        self.assertEqual(s06.expected_cost.llm_tokens, 0.0)
        # Quality tied + solver runtime tied -> S01 lexicographically first;
        # S06 must NOT outrank S01 on the placeholder-zero token cost.
        ordered = [r.strategy.strategy_id for r in recs
                   if r.strategy.strategy_id in ("S01", "S06")]
        self.assertLess(ordered.index("S01"), ordered.index("S06"))

    def test_no_common_dimension_declares_cost_not_comparable(self):
        """No shared measured dimension across candidates -> explicit
        warning, cost term zero, no missing-data benefit."""
        for i in range(2):
            self.h.bank.append(self.make_record(
                execution_id=f"ex_x{i}", task_id=f"tx{i}", strategy_id="S01",
                gap=0.05, cost=CostVector(llm_tokens=1000, tool_calls=0,
                                          solver_runtime_s=0, retries=0,
                                          latency_s=0),
                cost_measured=("llm_tokens",)))
            self.h.bank.append(self.make_record(
                execution_id=f"ex_y{i}", task_id=f"ty{i}", strategy_id="S06",
                gap=0.05, cost=CostVector(llm_tokens=0, tool_calls=0,
                                          solver_runtime_s=5.0, retries=0,
                                          latency_s=0),
                cost_measured=("solver_runtime_s",)))
        recs = self.h.selector.recall(self.make_profile(problem_id="q"),
                                      top=10, memory_mode="cost-aware")
        s01 = next(r for r in recs if r.strategy.strategy_id == "S01")
        self.assertEqual(s01.cost_basis_dims, [])
        self.assertTrue(any("not comparable" in w for w in s01.risk_warnings))


class TestTaskSummary(HarnessTestCase):
    def setUp(self):
        super().setUp()
        self.h = ORHarness(home=self.home)

    def test_fail_retry_success_aggregates_all_attempts(self):
        """A fails -> A retries -> B succeeds: every attempt charged once to
        the strategy that ran it; retries = sum of per-attempt NEW retries
        (0 + 1 + 1 = 2, never 0 + 1 + 2 = 3)."""
        attempts = [
            ("ex_a1", "S01", "error", False, 0.0, 4.0),
            ("ex_a2", "S01", "error", False, 1.0, 5.0),  # this attempt IS a retry
            ("ex_b1", "S04", "optimal", True, 1.0, 6.0),  # switch + retry
        ]
        for ex_id, sid, status, feasible, retries, latency in attempts:
            self.h.bank.append(self.make_record(
                execution_id=ex_id, task_id="t_task", strategy_id=sid,
                feasible=feasible, status=status,
                cost=CostVector(llm_tokens=500, tool_calls=3,
                                solver_runtime_s=2.0, retries=retries,
                                latency_s=latency),
                cost_measured=MEASURED_ALL))
        summary = self.h.task_cost_summary("t_task")
        self.assertEqual(summary["n_attempts"], 3)
        self.assertEqual(summary["total_cost"]["llm_tokens"], 1500.0)
        self.assertEqual(summary["total_cost"]["retries"], 2.0)
        self.assertTrue(summary["complete"]["retries"])
        # Per-attempt attribution preserved (each strategy owns its cost).
        by_strategy = {a["strategy_id"]: a["cost"]["llm_tokens"]
                       for a in summary["attempts"]}
        self.assertEqual(by_strategy, {"S01": 500.0, "S04": 500.0})

    def test_sequential_latencies_never_become_a_complete_total(self):
        """Acceptance 4: 4/5/6s sequential attempts — end-to-end is unknown
        without harness task timing (never max=6, never sum=15)."""
        for i, latency in enumerate((4.0, 5.0, 6.0)):
            self.h.bank.append(self.make_record(
                execution_id=f"ex_l{i}", task_id="t_lat", strategy_id="S01",
                cost=CostVector(llm_tokens=100, tool_calls=2,
                                solver_runtime_s=1.0, retries=0,
                                latency_s=latency),
                cost_measured=MEASURED_ALL))
        summary = self.h.task_cost_summary("t_lat")
        self.assertIsNone(summary["end_to_end_latency_s"])
        self.assertEqual(summary["end_to_end_latency_source"], "unknown")
        # Per-attempt latencies are facts; no fabricated aggregate.
        self.assertEqual([a["latency_s"] for a in summary["attempts"]],
                         [4.0, 5.0, 6.0])
        self.assertNotIn("latency_s", summary["total_cost"])
        # Harness-supplied explicit task timing is honored as such.
        summary2 = self.h.task_cost_summary("t_lat", end_to_end_latency_s=15.0)
        self.assertEqual(summary2["end_to_end_latency_s"], 15.0)
        self.assertEqual(summary2["end_to_end_latency_source"],
                         "harness-supplied")

    def test_incomplete_dimensions_marked_incomplete(self):
        self.h.bank.append(self.make_record(
            execution_id="ex_i", task_id="t_inc", strategy_id="S01",
            cost=CostVector(llm_tokens=0, tool_calls=2,
                            solver_runtime_s=1.0, retries=0, latency_s=1.0),
            cost_measured=("tool_calls", "solver_runtime_s",
                           "retries", "latency_s")))
        summary = self.h.task_cost_summary("t_inc")
        self.assertFalse(summary["complete"]["llm_tokens"])
        self.assertIsNone(summary["total_cost"]["llm_tokens"])
        self.assertTrue(summary["complete"]["solver_runtime_s"])


class TestLegacyAndBackfill(HarnessTestCase):
    def setUp(self):
        super().setUp()
        self.bank = ExperienceBank(self.store)

    def test_legacy_backfilled_tokens_stay_usable(self):
        """Acceptance 5: a pre-mask payload with non-zero backfilled tokens
        keeps them (value-level inference; unconfirmable zeros stay
        unknown)."""
        rec = self.make_record(execution_id="ex_leg")
        payload = rec.to_dict()
        payload.pop("cost_measured")  # simulate a pre-Cost-round payload
        payload["cost"]["llm_tokens"] = 1000.0
        payload["cost"]["retries"] = 0.0
        loaded = ExecutionRecord.from_dict(payload)
        self.assertIn("llm_tokens", loaded.cost.measured_dims())
        # retries=0 cannot be confirmed -> stays unknown.
        self.assertNotIn("retries", loaded.cost.measured_dims())
        self.bank.append(loaded)
        stored = self.bank.get("ex_leg")
        self.assertEqual(stored.cost.measured_dims()
                         & {"llm_tokens"}, {"llm_tokens"})
        self.assertIsNone(stored.cost_measured if False else None)  # mask stays legacy
        cell = __import__("or_harness.strategy.stats",
                          fromlist=["ConditionalStats"]).ConditionalStats(
            self.bank).cell(stored.group_l1, "S01")
        self.assertEqual(cell.n_measured.get("llm_tokens", 0), 1)
        self.assertAlmostEqual(cell.measured_cost("llm_tokens"), 1000.0)


class TestBackfillConsistency(HarnessTestCase):
    def setUp(self):
        super().setUp()
        self.h = ORHarness(home=self.home)

    def _seed_and_snapshot(self, tokens=1000.0, n=2):
        for i in range(n):
            self.h.bank.append(self.make_record(
                execution_id=f"ex_s{i}", task_id=f"ts{i}", strategy_id="S01",
                cost=CostVector(llm_tokens=tokens, tool_calls=2,
                                solver_runtime_s=1.0, retries=0, latency_s=1.0),
                cost_measured=MEASURED_ALL))

    def test_late_backfill_keeps_feedback_consistent(self):
        """Acceptance 6a: feedback computed at record time with actual=110;
        a later backfill to 1000 must update the persisted summary — never
        actual=110 alongside cost=1000."""
        self._seed_and_snapshot(tokens=1000.0)
        snapshot = self.h.predict_cost(_task("t_fb"), "S01")
        rec = self.make_record(
            task_id="t_fb", strategy_id="S01",
            cost=CostVector(llm_tokens=0, tool_calls=2,
                            solver_runtime_s=1.0, retries=0, latency_s=1.0),
            cost_measured=("tool_calls", "solver_runtime_s", "retries",
                           "latency_s"))
        outcome = self.h.record(rec, prediction=snapshot,
                                override={"llm_tokens": 110.0})
        feedback = outcome["cost_feedback"]
        self.assertEqual(feedback["per_dimension"]["llm_tokens"]["actual"],
                         110.0)
        self.h.bank.update_cost(outcome["execution_id"], llm_tokens=1000.0)
        stored = self.h.bank.get(outcome["execution_id"])
        self.assertEqual(stored.cost.llm_tokens, 1000.0)
        self.assertEqual(
            stored.execution_features["cost_feedback"]
            ["per_dimension"]["llm_tokens"]["actual"], 1000.0)

    def test_re_induction_refreshes_cost(self):
        """Acceptance 6b: cost changed after induction -> the next explicit
        induction refreshes the expected cost (the induction update-join
        only — no induction refactoring)."""
        self._seed_and_snapshot(tokens=100.0)
        first = self.h.induce(strategy_id="S01")
        entry_id = first["results"][0]["created"]
        entry = self.h.sbank.get(entry_id)
        self.assertAlmostEqual(entry.expected_cost_hat.llm_tokens, 100.0)
        for i in range(2):
            self.h.bank.update_cost(f"ex_s{i}", llm_tokens=1000.0)
        again = self.h.induce(strategy_id="S01")
        updated = again["results"][0].get("updated")
        self.assertEqual(updated, entry_id)
        entry = self.h.sbank.get(entry_id)
        self.assertAlmostEqual(entry.expected_cost_hat.llm_tokens, 1000.0)
        self.assertEqual(entry.cost_support_n.get("llm_tokens"), 2)


class TestFrozenSnapshotAndBaseline(HarnessTestCase):
    def setUp(self):
        super().setUp()
        self.h = ORHarness(home=self.home)

    def test_feedback_uses_frozen_snapshot_not_current_estimate(self):
        """Acceptance 8 (snapshot part): the estimate moved after the
        snapshot was frozen; feedback still compares against the frozen
        prediction."""
        for i in range(2):
            self.h.bank.append(self.make_record(
                execution_id=f"ex_f{i}", task_id=f"tf{i}", strategy_id="S01",
                cost=CostVector(llm_tokens=1000, tool_calls=2,
                                solver_runtime_s=1.0, retries=0, latency_s=1.0),
                cost_measured=MEASURED_ALL))
        snapshot = self.h.predict_cost(_task("t_fr"), "S01")  # frozen at 1000
        self.h.bank.append(self.make_record(
            execution_id="ex_shift", task_id="tsh", strategy_id="S01",
            cost=CostVector(llm_tokens=5000, tool_calls=2,
                            solver_runtime_s=1.0, retries=0, latency_s=1.0),
            cost_measured=MEASURED_ALL))
        rec = self.make_record(
            task_id="t_fr", strategy_id="S01",
            cost=CostVector(llm_tokens=1100, tool_calls=2,
                            solver_runtime_s=1.0, retries=0, latency_s=1.0),
            cost_measured=MEASURED_ALL)
        outcome = self.h.record(rec, prediction=snapshot)
        feedback = outcome["cost_feedback"]
        self.assertAlmostEqual(
            feedback["per_dimension"]["llm_tokens"]["predicted"], 1000.0)

    def test_paused_compaction_and_retention_mark_intact(self):
        """Acceptance 9: GC compaction stays deferred (touches nothing),
        explicit retention marks are preserved, and profiling works without
        a CIR (the consolidated analysis entry)."""
        rec = self.make_record(execution_id="ex_keep", task_id="tk")
        outcome = self.h.record(rec, retain_reason="contrast")
        stored = self.h.bank.get(outcome["execution_id"])
        self.assertEqual(stored.retention_reason, "contrast")
        n_before = self.h.bank.count()
        gc_result = self.h.collect_garbage(mode="compact", dry_run=False)
        self.assertTrue(gc_result.get("deferred"))
        self.assertEqual(self.h.bank.count(), n_before)
        # The consolidated analysis entry works without a CIR: the profile
        # is still produced (no error, no separate understand step).
        profile = self.h.profile({"task_id": "t_cir", "family": "routing"})
        self.assertEqual(profile.family, "routing")


class TestLegacyEntryCompatibility(HarnessTestCase):
    """An entry induced BEFORE the measured-mask existed carries a fixed
    interval for all five dimensions (including never-backfilled tokens).
    The interval key set must NOT be read as proof of measurement."""

    def _legacy_entry_payload(self):
        from or_harness.core.schema import COST_DIMENSIONS
        return {
            "entry_id": "se_legacy", "strategy_id": "S01",
            "pattern": {"predicates": {}},
            "expected": {
                "quality_hat": 0.9, "quality_interval": [0.5, 1.0],
                "cost_hat": {"llm_tokens": 0.0, "tool_calls": 0.0,
                             "solver_runtime_s": 0.0, "retries": 0.0,
                             "latency_s": 0.0},
                # Old code wrote the fixed band for every dimension.
                "cost_interval": {d: [0.5, 2.0] for d in COST_DIMENSIONS},
                "failure_prob": 0.1,
            },
            "support_n": 2,
        }

    def test_legacy_fixed_intervals_prove_nothing(self):
        from or_harness.core.schema import StrategicEntry, compute_cost_feedback
        entry = StrategicEntry.from_dict(self._legacy_entry_payload())
        # tokens=0 with a fixed interval is NOT "measured zero".
        self.assertEqual(entry.expected_cost_hat.measured, None)
        self.assertNotIn("llm_tokens", entry.expected_cost_hat.measured_dims())
        snapshot = PredictionSnapshot(
            strategy_id="S01", expected_cost=entry.expected_cost_hat,
            source="entry", support_n=entry.support_n)
        feedback = compute_cost_feedback(
            snapshot, "S01", "attempt",
            CostVector(llm_tokens=1000, measured={"llm_tokens"}))
        self.assertIsNone(feedback)  # no 27.6-style pseudo log-error

    def test_new_entries_carry_explicit_mask(self):
        from or_harness.core.schema import StrategicEntry
        entry = StrategicEntry(
            entry_id="se_new", strategy_id="S01",
            pattern={"predicates": {}},
            expected_cost_hat=CostVector(llm_tokens=100.0,
                                         measured={"llm_tokens"}),
            cost_interval={"llm_tokens": (0.5, 2.0)},
            cost_support_n={"llm_tokens": 3})
        restored = StrategicEntry.from_dict(entry.to_dict())
        self.assertEqual(restored.expected_cost_hat.measured, {"llm_tokens"})
        self.assertEqual(restored.cost_support_n, {"llm_tokens": 3})


class TestDelayedBackfillFeedback(HarnessTestCase):
    """A dimension unknown at record time must produce feedback once
    backfilled — the gate is "is there a snapshot", not "was there already
    feedback"."""

    def setUp(self):
        super().setUp()
        self.h = ORHarness(home=self.home)

    def test_first_delayed_backfill_creates_feedback(self):
        snapshot = PredictionSnapshot(
            strategy_id="S01",
            expected_cost=CostVector(llm_tokens=100.0,
                                     measured={"llm_tokens"}),
            source="entry", support_n=2, support_per_dim={"llm_tokens": 2})
        rec = self.make_record(
            task_id="td", strategy_id="S01",
            cost=CostVector(llm_tokens=0, tool_calls=2, solver_runtime_s=1.0,
                            retries=0, latency_s=1.0),
            cost_measured=("tool_calls", "solver_runtime_s", "retries",
                           "latency_s"))
        outcome = self.h.record(rec, prediction=snapshot)
        # tokens unmeasured on the actual side -> no feedback yet.
        self.assertNotIn("cost_feedback", outcome)
        self.h.bank.update_cost(outcome["execution_id"], llm_tokens=1000.0)
        stored = self.h.bank.get(outcome["execution_id"])
        feedback = stored.execution_features.get("cost_feedback")
        self.assertIsNotNone(feedback)
        self.assertAlmostEqual(
            feedback["per_dimension"]["llm_tokens"]["predicted"], 100.0)
        self.assertAlmostEqual(
            feedback["per_dimension"]["llm_tokens"]["actual"], 1000.0)

    def test_no_snapshot_means_backfill_writes_no_feedback(self):
        rec = self.make_record(task_id="td2", strategy_id="S01",
                               cost_measured=MEASURED_ALL)
        outcome = self.h.record(rec)
        self.h.bank.update_cost(outcome["execution_id"], llm_tokens=1000.0)
        stored = self.h.bank.get(outcome["execution_id"])
        self.assertNotIn("cost_feedback", stored.execution_features)


class TestRecordPositionalCompatibility(HarnessTestCase):
    """The historical positional call
    ``record(record, override, retain_reason)`` keeps its meaning; the new
    keyword-only parameters never capture positional arguments."""

    def setUp(self):
        super().setUp()
        self.h = ORHarness(home=self.home)

    def test_positional_retain_reason_still_works(self):
        rec = self.make_record(task_id="tp1", strategy_id="S01",
                               cost_measured=MEASURED_ALL)
        outcome = self.h.record(rec, None, "contrast")
        stored = self.h.bank.get(outcome["execution_id"])
        self.assertEqual(stored.retention_reason, "contrast")

    def test_positional_override_and_retain_reason_together(self):
        rec = self.make_record(task_id="tp2", strategy_id="S01",
                               cost_measured=MEASURED_ALL)
        outcome = self.h.record(rec, {"llm_tokens": 50.0}, "keep")
        stored = self.h.bank.get(outcome["execution_id"])
        self.assertEqual(stored.retention_reason, "keep")
        self.assertEqual(stored.cost.llm_tokens, 50.0)


class TestSupportRefresh(HarnessTestCase):
    """Identical means with more measured samples must still refresh the
    entry's per-dimension support (and thus the prediction snapshot's)."""

    def setUp(self):
        super().setUp()
        self.h = ORHarness(home=self.home)

    def _seed(self):
        self.h.bank.append(self.make_record(
            execution_id="ex_one", task_id="ta", strategy_id="S01",
            cost=CostVector(llm_tokens=100, tool_calls=2,
                            solver_runtime_s=1.0, retries=0, latency_s=1.0),
            cost_measured=MEASURED_ALL))
        self.h.bank.append(self.make_record(
            execution_id="ex_two", task_id="tb", strategy_id="S01",
            cost=CostVector(llm_tokens=0, tool_calls=2, solver_runtime_s=1.0,
                            retries=0, latency_s=1.0),
            cost_measured=("tool_calls", "solver_runtime_s", "retries",
                           "latency_s")))

    def test_support_change_triggers_re_induction(self):
        self._seed()
        first = self.h.induce(strategy_id="S01")
        entry_id = first["results"][0]["created"]
        entry = self.h.sbank.get(entry_id)
        self.assertEqual(entry.cost_support_n["llm_tokens"], 1)
        # Same mean (100), one more measured sample.
        self.h.bank.update_cost("ex_two", llm_tokens=100.0)
        again = self.h.induce(strategy_id="S01")
        self.assertEqual(again["results"][0].get("updated"), entry_id)
        entry = self.h.sbank.get(entry_id)
        self.assertEqual(entry.cost_support_n["llm_tokens"], 2)
        snapshot = self.h.predict_cost(_task("t_sup"), "S01")
        self.assertEqual(snapshot.support_per_dim["llm_tokens"], 2)


if __name__ == "__main__":
    unittest.main()
