"""Induction tests: consolidation, honest intervals, evidence-derived
applicability, independent-evidence gate, restatement refusal, cold-archive
veto, offline revision, rebuild."""
import unittest

from helpers import HarnessTestCase

from or_harness.api import ORHarness
from or_harness.core.schema import (StrategicEntry, evidence_predicates,
                                    min_interval_width)
from or_harness.strategy.experience_bank import ExperienceBank
from or_harness.strategy.induction import InductionEngine
from or_harness.strategy.stats import ConditionalStats
from or_harness.strategy.strategic_bank import StrategicBank


class InductionCase(HarnessTestCase):
    def setUp(self):
        super().setUp()
        self.bank = ExperienceBank(self.store)
        self.sbank = StrategicBank(self.store)
        self.stats = ConditionalStats(self.bank)
        self.engine = InductionEngine(self.stats, self.sbank)

    def seed(self, strategy_id, gaps, family="routing", task_prefix="s",
             **coupling):
        ids = []
        for i, gap in enumerate(gaps):
            profile = self.make_profile(problem_id=f"{task_prefix}{i}",
                                        family=family, **coupling)
            rec = self.make_record(execution_id=f"{task_prefix}_{strategy_id}_{i}",
                                   task_id=f"{task_prefix}{i}",
                                   strategy_id=strategy_id, profile=profile, gap=gap)
            self.bank.append(rec)
            ids.append(rec.execution_id)
        return ids


class TestInduce(InductionCase):
    def test_creates_candidate_entry(self):
        ids = self.seed("S01", [0.05, 0.10, 0.08])
        result = self.engine.induce(self.make_profile("q"), "S01")
        self.assertIsNotNone(result.get("created"))
        entry = self.sbank.get(result["created"])
        self.assertEqual(entry.status, "candidate")  # never born validated
        self.assertEqual(entry.predicates["family"], "routing")
        self.assertAlmostEqual(entry.expected_quality_hat, (0.95 + 0.90 + 0.92) / 3, places=3)
        self.assertEqual(sorted(entry.provenance), sorted(ids))

    def test_honest_interval_for_small_n(self):
        self.seed("S01", [0.0, 0.0])
        result = self.engine.induce(self.make_profile("q"), "S01")
        entry = self.sbank.get(result["created"])
        lo, hi = entry.quality_interval
        self.assertGreaterEqual(hi - lo, min_interval_width(2))
        # n=2 may not claim [0.95, 1.0]-style narrow intervals
        self.assertLessEqual(lo, 0.5)

    def test_single_sample_refused(self):
        self.seed("S01", [0.05])
        result = self.engine.induce(self.make_profile("q"), "S01")
        self.assertIsNone(result.get("created"))
        self.assertIn("fewer than 2", result["skipped"])

    def test_restatement_refused(self):
        self.seed("S01", [0.05, 0.10, 0.08])
        first = self.engine.induce(self.make_profile("q"), "S01")
        again = self.engine.induce(self.make_profile("q2"), "S01")
        self.assertIsNone(again.get("created"))
        self.assertIn("restatement", again["skipped"])
        self.assertEqual(again["entry_id"], first["created"])

    def test_update_when_evidence_changes(self):
        self.seed("S01", [0.05, 0.10])
        first = self.engine.induce(self.make_profile("q"), "S01")
        self.seed("S01", [0.60, 0.65], task_prefix="late")
        result = self.engine.induce(self.make_profile("q2"), "S01")
        self.assertEqual(result.get("updated"), first["created"])
        entry = self.sbank.get(first["created"])
        self.assertLess(entry.expected_quality_hat, 0.8)

    def test_dry_run_creates_nothing(self):
        self.seed("S01", [0.05, 0.10])
        result = self.engine.induce(self.make_profile("q"), "S01", dry_run=True)
        self.assertIn("would_create", result)
        self.assertEqual(self.sbank.count(), 0)


class TestIndependentEvidence(HarnessTestCase):
    """A claim needs >=2 distinct tasks: repeating one task is repetition, not
    reproduction. The gate guards CREATION only — an existing claim is still
    refreshed by any new evidence, and the evidence itself is never discarded
    (it stays available to recall as conditional statistics)."""

    def setUp(self):
        super().setUp()
        self.h = ORHarness(home=self.home)
        self.addCleanup(self.h.close)
        self._seq = 0

    def _record(self, task_id, gaps, *, strategy="S01", **coupling):
        for gap in gaps:
            self._seq += 1
            self.h.bank.append(self.make_record(
                execution_id=f"ex_{task_id}_{self._seq}", task_id=task_id,
                strategy_id=strategy, gap=gap,
                profile=self.make_profile(problem_id=task_id, **coupling)))

    def test_same_task_repeated_is_not_independent_evidence(self):
        self._record("lonely", [0.0, 0.0, 0.0])
        result = self.h.induce(strategy_id="S01")["results"][0]
        self.assertIsNone(result.get("created"))
        self.assertIn("needs independent evidence", result["skipped"])
        self.assertEqual(result["verification"],
                         {"tasks": ["lonely"], "required_tasks": 2})
        self.assertEqual(self.h.sbank.count(), 0)

    def test_gate_applies_to_dry_run(self):
        self._record("lonely", [0.0, 0.0])
        result = self.h.induce(strategy_id="S01", dry_run=True)["results"][0]
        self.assertNotIn("would_create", result)
        self.assertIn("needs independent evidence", result["skipped"])

    def test_refused_evidence_is_still_recallable(self):
        self._record("lonely", [0.0, 0.0, 0.0])
        self.h.induce(strategy_id="S01")
        recs = self.h.selector.recall(self.make_profile(problem_id="q"), top=5)
        s01 = next(r for r in recs if r.strategy.strategy_id == "S01")
        self.assertEqual(s01.evidence, "conditional_stats")
        self.assertEqual(len(s01.evidence_refs), 3)

    def test_two_tasks_create_the_claim(self):
        self._record("ta", [0.0, 0.0])
        self._record("tb", [0.0, 0.0])
        result = self.h.induce(strategy_id="S01")["results"][0]
        self.assertIsNotNone(result.get("created"))
        self.assertNotIn("verification", result)

    def test_repetition_still_refreshes_an_existing_claim(self):
        self._record("ta", [0.0])
        self._record("tb", [0.0])
        created = self.h.induce(strategy_id="S01")["results"][0]["created"]
        self._record("ta", [0.0])          # same task again — refresh, not create
        result = self.h.induce(strategy_id="S01")["results"][0]
        self.assertEqual(result.get("updated"), created)
        self.assertNotIn("skipped", result)
        self.assertEqual(self.h.sbank.get(created).support_n, 3)


class TestColdArchiveVeto(InductionCase):
    def test_veto_reports_before_the_admission_gate(self):
        """With a card standing, a single-task evidence set must hear
        "vetoed" (the real blocker), not "collect a second task" — that path
        cannot succeed while the card is there."""
        self.seed("S01", [0.05, 0.10], task_prefix="one")  # a single task
        profile = self.make_profile("q")
        predicates = evidence_predicates(self.stats.evidence(profile, "S01"))
        self.sbank.add(StrategicEntry(
            entry_id="se_hand", strategy_id="S01",
            pattern={"predicates": predicates},
            expected_quality_hat=0.9, quality_interval=[0.5, 1.0],
            status="suspect"))
        self.sbank.retire("se_hand", reason="did not reproduce")
        result = self.engine.induce(profile, "S01")
        self.assertIn("cold-archive veto", result["skipped"])
        self.assertNotIn("verification", result)

    def test_veto_blocks_and_force_lifts_it(self):
        self.seed("S01", [0.05, 0.10])
        created = self.engine.induce(self.make_profile("q"), "S01")["created"]
        entry = self.sbank.get(created)
        entry.status = "suspect"
        self.sbank.update(entry)
        card = self.sbank.retire(created, reason="bad generalization")
        # Same evidence, same pattern -> vetoed.
        blocked = self.engine.induce(self.make_profile("q2"), "S01")
        self.assertIsNone(blocked.get("created"))
        self.assertEqual(blocked["vetoed"]["pattern_hash"], card.pattern_hash)
        # Force LIFTS the veto (the card is removed), so the harness's
        # environment-drift judgment is made once — the next plain induce
        # must not be vetoed again.
        forced = self.engine.induce(self.make_profile("q2"), "S01", force=True)
        self.assertIsNotNone(forced.get("created"))
        self.assertEqual(self.sbank.cold_archive(), [])
        again = self.engine.induce(self.make_profile("q3"), "S01")
        self.assertNotIn("vetoed", again)


class TestStructuralCells(InductionCase):
    """A claim's applicability is the structural CELL its evidence occupies
    (family + the rc/tc/rx interval). Structurally different regions of one
    family are separate evidence sets — pooling them once averaged a Q=1.0
    region and a Q=0.1 region into a single claim predicting 0.55. There are
    no levels and no widening command."""

    def test_predicates_are_the_cell(self):
        self.seed("S01", [0.05, 0.10], task_prefix="a", resource_coupling=0.80)
        self.seed("S01", [0.05, 0.10], task_prefix="b", resource_coupling=0.94)
        entry = self.sbank.get(self.engine.induce(self.make_profile("q"), "S01")["created"])
        self.assertEqual(entry.predicates["family"], "routing")
        self.assertEqual(entry.predicates["resource_coupling"], [0.75, 1.0])
        self.assertTrue(entry.matches(self.make_profile(problem_id="p80",
                                                        resource_coupling=0.80)))
        self.assertFalse(entry.matches(self.make_profile(problem_id="p40",
                                                         resource_coupling=0.40)))

    def test_opposite_regions_never_merge(self):
        """The reproduced defect: one strategy scoring 1.0 in a low-coupling
        region and 0.1 in a high-coupling region used to produce ONE claim
        covering both and predicting 0.55."""
        self.seed("S01", [0.0, 0.0], task_prefix="lo", resource_coupling=0.10)
        self.seed("S01", [0.9, 0.9], task_prefix="hi", resource_coupling=0.90)

        low = self.engine.induce(self.make_profile("q", resource_coupling=0.10),
                                 "S01")
        high = self.engine.induce(self.make_profile("q", resource_coupling=0.90),
                                  "S01")
        self.assertIsNotNone(low.get("created"))
        self.assertIsNotNone(high.get("created"))
        self.assertNotEqual(low["created"], high["created"])
        lo_entry = self.sbank.get(low["created"])
        hi_entry = self.sbank.get(high["created"])
        self.assertEqual(lo_entry.predicates["resource_coupling"], [0.0, 0.25])
        self.assertEqual(hi_entry.predicates["resource_coupling"], [0.75, 1.0])
        self.assertAlmostEqual(lo_entry.expected_quality_hat, 1.0, places=3)
        self.assertAlmostEqual(hi_entry.expected_quality_hat, 0.1, places=3)
        # Neither claim answers for the other's structure.
        self.assertFalse(lo_entry.matches(self.make_profile(problem_id="h",
                                                           resource_coupling=0.90)))
        self.assertFalse(hi_entry.matches(self.make_profile(problem_id="l",
                                                           resource_coupling=0.10)))

    def test_scattered_evidence_forms_no_claim(self):
        """Accepted trade-off of the cell rule: evidence scattered across
        cells may be too thin to form a claim. Statistics stay visible and NO
        automatic cell-merging invents knowledge."""
        for i, rc in enumerate([0.10, 0.40, 0.60, 0.90]):
            self.seed("S01", [0.05], task_prefix=f"sc{i}", resource_coupling=rc)
        result = self.engine.induce(self.make_profile("q", resource_coupling=0.10),
                                    "S01")
        self.assertIsNone(result.get("created"))
        self.assertIn("fewer than 2", result["skipped"])
        self.assertEqual(self.sbank.count(), 0)
        # The facts are still there to be counted.
        self.assertEqual(
            self.stats.evidence(self.make_profile("q", resource_coupling=0.10),
                                "S01")[0].execution_id, "sc0_S01_0")

    def test_claim_stays_inside_its_family(self):
        """Evidence in one family says nothing about another: cross-family
        transfer is a harness judgment, not an automatic claim."""
        self.seed("S01", [0.05, 0.10], family="routing", task_prefix="a")
        self.seed("S01", [0.05, 0.10], family="packing", task_prefix="b")
        self.engine.induce(self.make_profile("q"), "S01")
        for entry in self.sbank.list():
            other = ("packing" if entry.predicates["family"] == "routing"
                     else "routing")
            self.assertFalse(entry.matches(
                self.make_profile(problem_id="x", family=other)))


class TestApplicabilityNotes(InductionCase):
    """Notes are harness-written free text: stored verbatim, shown by
    ``inspect``, never scored and never "verified" — the framework cannot
    check a sentence, so it does not pretend to."""

    def test_notes_are_stored_verbatim(self):
        self.seed("S01", [0.05, 0.10])
        note = "only trust this when the shared resource is the bottleneck"
        result = self.engine.induce(self.make_profile("q"), "S01", notes=[note])
        entry = self.sbank.get(result["created"])
        self.assertEqual(entry.applicability, [note])

    def test_blank_notes_ignored(self):
        self.seed("S01", [0.05, 0.10])
        result = self.engine.induce(self.make_profile("q"), "S01",
                                    notes=["   ", ""])
        entry = self.sbank.get(result["created"])
        self.assertEqual(entry.applicability, [])

    def test_notes_append_on_refresh(self):
        self.seed("S01", [0.05, 0.10])
        first = self.engine.induce(self.make_profile("q"), "S01", notes=["one"])
        result = self.engine.induce(self.make_profile("q2"), "S01", notes=["two"])
        self.assertEqual(result.get("updated"), first["created"])
        self.assertEqual(self.sbank.get(first["created"]).applicability,
                         ["one", "two"])


class TestPeerRelations(InductionCase):
    """Induction reads RELATIONS, not only one cell's means.

    Peer evidence is the reason a claim is worth committing — another
    strategy performing differently in the same cell, or the same strategy
    performing differently in a neighbouring cell. It is read ONLY to phrase
    the claim: it never enters the target's statistics, never satisfies the
    admission gate, and never creates an entry on its own.
    """

    def test_contrast_is_written_to_risk_conditions(self):
        self.seed("S01", [0.05, 0.05], task_prefix="a")   # meanQ 0.95
        self.seed("S04", [0.60, 0.60], task_prefix="b")   # meanQ 0.40
        peer = {"S04": self.stats.evidence(self.make_profile("q"), "S04")}
        result = self.engine.induce(self.make_profile("q"), "S01",
                                    peer_evidence=peer)
        entry = self.sbank.get(result["created"])
        self.assertEqual(len(entry.risk_conditions), 1)
        self.assertIn("S04", entry.risk_conditions[0])
        self.assertIn("contrast vs", entry.risk_conditions[0])
        self.assertEqual(result["peer_relations"], entry.risk_conditions)

    def test_peer_evidence_never_enters_the_statistics(self):
        """The claim's own numbers are unchanged by a peer: a relation is
        applicability text, not a second statistic."""
        self.seed("S01", [0.05, 0.05], task_prefix="a")
        self.seed("S04", [0.60, 0.60], task_prefix="b")
        profile = self.make_profile("q")
        peer = {"S04": self.stats.evidence(profile, "S04")}
        result = self.engine.induce(profile, "S01", peer_evidence=peer)
        entry = self.sbank.get(result["created"])
        self.assertAlmostEqual(entry.expected_quality_hat, 0.95, places=6)
        self.assertEqual(entry.support_n, 2)
        self.assertEqual(entry.provenance,
                         [r.execution_id for r in self.stats.evidence(profile,
                                                                      "S01")])

    def test_peer_evidence_never_creates_an_entry(self):
        """A contrast with NO target evidence is not a claim: the engine
        refuses on its own gate, peer or not."""
        self.seed("S04", [0.60, 0.60], task_prefix="b")
        peer = {"S04": self.stats.evidence(self.make_profile("q"), "S04")}
        result = self.engine.induce(self.make_profile("q"), "S01",
                                    peer_evidence=peer)
        self.assertIsNone(result.get("created"))
        self.assertEqual(self.sbank.count(), 0)

    def test_incomplete_peer_cost_is_withheld_from_the_relation(self):
        """The peer's cost is quoted only where BOTH sides measured it on
        every record — a partial mean would make the ratio meaningless."""
        from or_harness.core.schema import CostVector
        self.seed("S01", [0.05, 0.05], task_prefix="a")
        for i in range(2):
            self.bank.append(self.make_record(
                execution_id=f"b_S04_{i}", task_id=f"b{i}", strategy_id="S04",
                profile=self.make_profile(problem_id=f"b{i}", family="routing"),
                gap=0.60,
                cost=CostVector(llm_tokens=500, solver_runtime_s=1.0),
                cost_measured=("llm_tokens",)))  # latency/runtime unmeasured
        peer = {"S04": self.stats.evidence(self.make_profile("q"), "S04")}
        result = self.engine.induce(self.make_profile("q"), "S01",
                                    peer_evidence=peer)
        entry = self.sbank.get(result["created"])
        self.assertIn("contrast vs", entry.risk_conditions[0])
        self.assertNotIn("solver_runtime_s", entry.risk_conditions[0])

    def test_no_peer_evidence_leaves_the_entry_unchanged(self):
        """A plain induction is byte-for-byte what it was: no relation lines,
        no new result key."""
        self.seed("S01", [0.05, 0.10])
        result = self.engine.induce(self.make_profile("q"), "S01")
        entry = self.sbank.get(result["created"])
        self.assertEqual(entry.risk_conditions, [])
        self.assertNotIn("peer_relations", result)

    def test_reversal_peer_is_phrased_as_a_boundary(self):
        """A neighbouring cell with opposite behaviour is written as the
        boundary it is — the reader must see WHERE the advantage stops."""
        for i in range(2):
            self.bank.append(self.make_record(
                execution_id=f"low_{i}", task_id=f"lowt{i}", strategy_id="S01",
                profile=self.make_profile(problem_id=f"lowt{i}",
                                          resource_coupling=0.10), gap=0.05))
        self.seed("S01", [0.95, 0.95], task_prefix="high", resource_coupling=0.90)
        profile = self.make_profile("q", resource_coupling=0.90)
        peer_key = [k for k in
                    self.stats.cells_in_family("S01", "routing")
                    if "0.00,0.25" in k][0]
        peer = {peer_key: [r for r in self.bank.all()
                           if r.source == "executed"
                           and r.strategy_id == "S01"
                           and "0.00,0.25" in self._cell_of(r)]}
        result = self.engine.induce(profile, "S01", peer_evidence=peer)
        entry = self.sbank.get(result["created"])
        self.assertTrue(entry.risk_conditions)
        self.assertIn("contrast vs", entry.risk_conditions[0])

    @staticmethod
    def _cell_of(record):
        from or_harness.core.schema import group_key
        return group_key(record.profile_snapshot)


class TestRebuild(InductionCase):
    """Re-induction from currently retained evidence. NOT exact
    reconstruction: the re-induced bank may legitimately differ from the
    previous one (induction logic and evidence sets evolve)."""

    def test_rebuild_regenerates_entries(self):
        self.seed("S01", [0.05, 0.10], task_prefix="a")
        self.seed("S04", [0.02, 0.04], task_prefix="b")
        plan = self.engine.rebuild(dry_run=True)
        self.assertEqual(plan["would_rebuild"], 2)
        result = self.engine.rebuild()
        self.assertEqual(result["rebuilt"], 2)
        self.assertEqual(self.sbank.count(), 2)

    def test_rebuild_preserves_cold_archive(self):
        self.seed("S01", [0.05, 0.10], task_prefix="a")
        created = self.engine.induce(self.make_profile("q"), "S01")["created"]
        self.sbank.retire(created, reason="x")
        self.engine.rebuild()
        self.assertEqual(len(self.sbank.cold_archive()), 1)


class TestOfflineRevision(HarnessTestCase):
    """Online recording only accumulates evidence; every knowledge change
    (promotion, demotion, scope tightening, dormancy wakeup) happens in the
    explicit offline `induce` call, replayed from the frozen checks written
    on the facts."""

    def setUp(self):
        super().setUp()
        self.h = ORHarness(home=self.home)
        self.addCleanup(self.h.close)

    def _entry(self, strategy="S01", gaps=(0.0, 0.0), tasks=("a", "b")):
        """Two independent tasks -> entry (honest interval floors at width 0.5)."""
        for i, gap in enumerate(gaps):
            self.h.bank.append(self.make_record(
                execution_id=f"ex_seed_{strategy}_{i}", task_id=tasks[i],
                strategy_id=strategy, gap=gap))
        result = self.h.induce(strategy_id=strategy)
        return result["results"][0]["created"]

    def _record(self, execution_id, task_id, gap, *, strategy="S01",
                family="routing", scope="attempt", **coupling):
        rec = self.make_record(
            execution_id=execution_id, task_id=task_id, strategy_id=strategy,
            gap=gap, profile=self.make_profile(problem_id=task_id,
                                               family=family, **coupling))
        rec.measurement_scope = scope
        return self.h.record(rec)

    # -- online: evidence only -------------------------------------------------

    def test_frozen_check_is_persisted_with_the_fact(self):
        entry_id = self._entry()
        entry = self.h.sbank.get(entry_id)
        out = self._record("ex_frozen", "ft1", gap=0.9)
        self.assertEqual([c["entry_id"] for c in out["prediction_checks"]],
                         [entry_id])
        self.assertEqual(out["prediction_checks"][0]["hit"], False)
        # The check carries the interval that was in force at that moment.
        self.assertEqual(out["prediction_checks"][0]["interval"],
                         list(entry.quality_interval))
        stored = self.h.bank.get("ex_frozen")
        self.assertEqual(stored.execution_features["quality_feedback"],
                         out["prediction_checks"])

    def test_recording_never_changes_the_entry(self):
        entry_id = self._entry()
        for i in range(3):
            self._record(f"ex_m{i}", f"mt{i}", gap=0.9)
        entry = self.h.sbank.get(entry_id)
        self.assertEqual(entry.status, "candidate")
        self.assertEqual(entry.prediction_track.n_predictions, 0)
        self.assertEqual(entry.prediction_track.consecutive_misses, 0)

    def test_other_strategy_never_audits_the_entry(self):
        """A never audits B: S06's execution says nothing about S01's entry."""
        entry_id = self._entry()
        out = self._record("ex_s06", "ot1", gap=0.9, strategy="S06")
        self.assertEqual(out["prediction_checks"], [])
        entry = self.h.sbank.get(entry_id)
        self.assertEqual((entry.status, entry.prediction_track.n_predictions),
                         ("candidate", 0))

    def test_task_scope_record_writes_no_check(self):
        """A task-scope total never audits attempt-scope knowledge."""
        entry_id = self._entry()
        out = self._record("ex_ts", "ts1", gap=0.9, scope="task")
        self.assertEqual(out["prediction_checks"], [])
        stored = self.h.bank.get("ex_ts")
        self.assertNotIn("quality_feedback", stored.execution_features)
        self.assertEqual(self.h.sbank.get(entry_id).prediction_track.n_predictions, 0)

    # -- offline: revision -----------------------------------------------------

    def test_content_misses_demote_at_revise(self):
        entry_id = self._entry()
        for i in range(3):
            out = self._record(f"ex_c{i}", f"ct{i}", gap=0.9)
            self.assertEqual([c["hit"] for c in out["prediction_checks"]], [False])
        report = self.h.induction.revise("S01")
        item = next(r for r in report if r["entry_id"] == entry_id)
        self.assertEqual(item["forward"]["consecutive_misses"], 3)
        self.assertIn("demoted:->suspect", item["transitions"])
        self.assertEqual(self.h.sbank.get(entry_id).status, "suspect")

    def test_hits_promote_at_revise(self):
        entry_id = self._entry()
        # Admission first: forward calibration can never promote a claim that
        # was not verified.
        entry = self.h.sbank.get(entry_id)
        entry.verification = {"state": "verified", "claim": "S01 holds here",
                              "conclusion": "check passed"}
        self.h.sbank.update(entry)
        for i in range(5):
            out = self._record(f"ex_h{i}", f"ht{i}", gap=0.0)
            self.assertEqual([c["hit"] for c in out["prediction_checks"]], [True])
        self.assertEqual(self.h.sbank.get(entry_id).status, "candidate")
        report = self.h.induction.revise("S01")
        item = next(r for r in report if r["entry_id"] == entry_id)
        self.assertIn("promoted:candidate->validated", item["transitions"])
        self.assertEqual(self.h.sbank.get(entry_id).status, "validated")

    def test_hits_never_promote_an_unverified_claim(self):
        """The reproduced defect: five frozen hits promoted a candidate that
        had never passed admission verification."""
        entry_id = self._entry()
        for i in range(5):
            self._record(f"ex_np{i}", f"npt{i}", gap=0.0)
        report = self.h.induction.revise("S01")
        item = next(r for r in report if r["entry_id"] == entry_id)
        self.assertNotIn("promoted:candidate->validated", item["transitions"])
        self.assertEqual(self.h.sbank.get(entry_id).status, "candidate")

    def test_out_of_range_evidence_is_not_checked(self):
        """A claim only answers for the structure it was induced from: a
        record outside its ranges matches nothing and changes nothing."""
        entry_id = self._entry()          # evidence sits at rc=0.9
        out = self._record("ex_far", "far1", gap=0.9, resource_coupling=0.2)
        self.assertEqual(out["prediction_checks"], [])
        entry = self.h.sbank.get(entry_id)
        self.assertEqual((entry.status, entry.prediction_track.n_predictions),
                         ("candidate", 0))

    def test_other_family_is_not_checked(self):
        """Claims are family-scoped; another family's execution is not a check
        on this claim (and not a counterexample to it either)."""
        entry_id = self._entry()
        out = self._record("ex_fam", "f1", gap=0.9, family="packing")
        self.assertEqual(out["prediction_checks"], [])
        self.assertEqual(self.h.sbank.get(entry_id).prediction_track.n_predictions, 0)
        self.assertEqual([r["entry_id"] for r in self.h.induction.revise("S01")], [])

    def test_dormant_wakes_on_new_evidence(self):
        entry_id = self._entry()
        entry = self.h.sbank.get(entry_id)
        entry.status = "dormant"
        entry.created_at = 1.0          # long before the new evidence
        self.h.sbank.update(entry)
        self._record("ex_wake", "wt1", gap=0.0)
        self.assertEqual(self.h.sbank.get(entry_id).status, "dormant")  # online
        self.h.induction.revise("S01")
        self.assertEqual(self.h.sbank.get(entry_id).status, "candidate")

    def test_revise_dry_run_writes_nothing(self):
        entry_id = self._entry()
        for i in range(3):
            self._record(f"ex_d{i}", f"dt{i}", gap=0.9)
        report = self.h.induction.revise("S01", dry_run=True)
        item = next(r for r in report if r["entry_id"] == entry_id)
        self.assertEqual(item["forward"]["consecutive_misses"], 3)
        entry = self.h.sbank.get(entry_id)
        self.assertEqual(entry.status, "candidate")
        self.assertEqual(entry.prediction_track.consecutive_misses, 0)


class TestCostClaimCompleteness(InductionCase):
    """A cost claim is COMPLETE-OR-SILENT.

    A dimension measured on only some supporting records must not be
    published: a mean over a subset is a partial observation, and downstream
    it feeds strategy selection and world-model prediction as if it were the
    whole truth."""

    def _seed_with_missing(self, missing_dims, n=3):
        """Seed n records of one strategy; the last one leaves
        ``missing_dims`` unmeasured."""
        from or_harness.core.schema import COST_DIMENSIONS, CostVector
        ids = []
        for i in range(n):
            profile = self.make_profile(problem_id=f"c{i}", family="routing")
            measured = set(COST_DIMENSIONS)
            if i == n - 1:
                measured -= set(missing_dims)
            cost = CostVector(
                llm_tokens=1000.0, tool_calls=2.0, solver_runtime_s=1.0,
                retries=0.0, latency_s=0.5, measured=measured)
            rec = self.make_record(
                execution_id=f"c_{i}", task_id=f"c{i}", strategy_id="S01",
                profile=profile, cost=cost)
            self.bank.append(rec)
            ids.append(rec.execution_id)
        return ids

    def test_partial_dimension_is_withheld(self):
        self._seed_with_missing({"llm_tokens"})
        out = self.engine.induce(self.make_profile(family="routing"), "S01")
        self.assertIsNotNone(out["created"])
        entry = self.sbank.get(out["created"])
        # llm_tokens was measured on 2 of 3 records: no claim.
        self.assertNotIn("llm_tokens", entry.expected_cost_hat.measured_dims())
        self.assertNotIn("llm_tokens", entry.cost_interval)
        # The complete dimensions ARE published — the gate is per-dimension,
        # not all-or-nothing.
        self.assertIn("solver_runtime_s", entry.expected_cost_hat.measured_dims())
        self.assertIn("tool_calls", entry.expected_cost_hat.measured_dims())
        self.assertEqual(entry.cost_support_n["llm_tokens"], 2)

    def test_withheld_report_names_the_dimension_and_the_fix(self):
        self._seed_with_missing({"llm_tokens", "tool_calls"})
        out = self.engine.induce(self.make_profile(family="routing"), "S01")
        withheld = out["cost_claim_withheld"]
        self.assertEqual(set(withheld["dimensions"]),
                         {"llm_tokens", "tool_calls"})
        self.assertEqual(withheld["dimensions"]["llm_tokens"],
                         {"n_measured": 2, "n": 3})
        self.assertEqual(len(withheld["supporting_executions"]), 3)
        self.assertIn("--override", withheld["note"])

    def test_no_withheld_report_when_everything_is_complete(self):
        self._seed_with_missing(set())
        out = self.engine.induce(self.make_profile(family="routing"), "S01")
        self.assertNotIn("cost_claim_withheld", out)

    def test_backfilling_restores_the_claim(self):
        from or_harness.core.schema import COST_DIMENSIONS
        ids = self._seed_with_missing({"llm_tokens"})
        out = self.engine.induce(self.make_profile(family="routing"), "S01")
        entry_id = out["created"]
        self.assertNotIn("llm_tokens",
                         self.sbank.get(entry_id).expected_cost_hat.measured_dims())
        # The fix is a backfill of the ONE incomplete record, not a re-run.
        self.bank.update_cost(ids[-1], llm_tokens=1234.0)
        self.engine.induce(self.make_profile(family="routing"), "S01")
        entry = self.sbank.get(entry_id)
        self.assertIn("llm_tokens", entry.expected_cost_hat.measured_dims())
        self.assertIn("llm_tokens", entry.cost_interval)
        # 1000, 1000, 1234 -> the mean now includes the backfilled value.
        self.assertAlmostEqual(entry.expected_cost_hat.llm_tokens,
                               3234.0 / 3.0, places=4)
        self.assertEqual(len(entry.expected_cost_hat.measured_dims()),
                         len(COST_DIMENSIONS))

    def test_incomplete_dimension_reads_unknown_not_partial_mean(self):
        """The withheld dimension must be UNKNOWN downstream, never the
        subset mean presented as a measurement."""
        from or_harness.core.schema import CostVector
        self._seed_with_missing({"llm_tokens"})
        out = self.engine.induce(self.make_profile(family="routing"), "S01")
        entry = self.sbank.get(out["created"])
        # 2 of 3 records had 1000 tokens: the subset mean would be 1000.0.
        self.assertEqual(entry.expected_cost_hat.llm_tokens, 0.0)
        self.assertNotIn("llm_tokens", entry.expected_cost_hat.measured_dims())
        self.assertEqual(entry.expected_cost_hat.measured_dims(),
                         set(entry.cost_interval))


if __name__ == "__main__":
    unittest.main()
