"""Knowledge CLAIMS: ONE claim per entry, verified and recalled as knowledge.

The unit of knowledge is the ENTRY, and one entry is one claim. This suite
covers the claim's generation (from cross-task evidence, with no host lookup
and no relation sub-structure), its assertion semantics, the publication gate
(>= 2 independent tasks), revision (a substantive change invalidates the old
verdict), transfer to a future task's recall, and the rule that two
independent claims never inherit each other's verification.
"""
import unittest

from helpers import HarnessTestCase

from or_harness.api import ORHarness
from or_harness.core.schema import (CLAIM_MIN_TASKS, StrategicEntry,
                                    claim_scope_tasks, validate_claim)
from or_harness.strategy.experience_bank import ExperienceBank
from or_harness.strategy.induction import InductionEngine
from or_harness.strategy.stats import ConditionalStats
from or_harness.strategy.strategic_bank import StrategicBank


class ClaimCase(HarnessTestCase):
    def setUp(self):
        super().setUp()
        self.bank = ExperienceBank(self.store)
        self.sbank = StrategicBank(self.store)
        self.stats = ConditionalStats(self.bank)
        self.engine = InductionEngine(self.stats, self.sbank)
        self.h = ORHarness(home=self.home)
        self.addCleanup(self.h.close)

    def profile(self, problem_id, tc=0.7, family="scheduling"):
        return self.make_profile(problem_id=problem_id, family=family,
                                 temporal_coupling=tc, resource_coupling=0.2,
                                 route_complexity=0.1)

    def exec_record(self, eid, task_id, status="optimal", gap=0.0,
                    feasible=True, strategy="S01", tc=0.7, measured=None):
        return self.make_record(
            execution_id=eid, task_id=task_id, strategy_id=strategy,
            profile=self.profile(task_id, tc=tc), status=status, gap=gap,
            feasible=feasible, cost_measured=measured)

    def seed_cross_period(self):
        """The acceptance example's evidence: three tasks, one strategy tag,
        differing cross-period handling."""
        for rec in [
            self.exec_record("ex_t1d", "T1", status="infeasible",
                             feasible=False),
            self.exec_record("ex_t1p", "T1"),
            self.exec_record("ex_t2d", "T2", status="feasible", gap=0.40),
            self.exec_record("ex_t2p", "T2", status="optimal", gap=0.05),
            self.exec_record("ex_t3p", "T3"),
        ]:
            self.bank.append(rec)

    def cross_period_claim(self, extra_evidence=()):
        evidence = [
            {"execution_id": "ex_t1d", "role": "dropped"},
            {"execution_id": "ex_t1p", "role": "preserved"},
            {"execution_id": "ex_t2d", "role": "dropped"},
            {"execution_id": "ex_t2p", "role": "preserved"},
            {"execution_id": "ex_t3p", "role": "preserved"},
        ]
        evidence = list(evidence) + list(extra_evidence)
        return {
            "subject": "principle:cross_period_state",
            "claim": ("时间耦合>=0.5 的调度任务上，时间分解必须保留跨期衔接状态；"
                      "丢弃它导致不可行或显著质量损失"),
            "conditions": {"predicates": {"family": "scheduling",
                                          "temporal_coupling": [0.5, 1.0]}},
            "evidence": evidence,
            "check": {"assertions": [
                {"kind": "probe", "roles": ["preserved"],
                 "path": "quality.feasible", "equals": True},
                {"kind": "comparison", "metric": "quality",
                 "roles_a": ["preserved"], "roles_b": ["dropped"],
                 "direction": "higher", "min_gap": 0.2,
                 "mode": "paired", "aggregation": "all"}]},
        }

    def verify_payload(self, claim):
        return {"purpose": "relation",
                "check": {"assertions": claim["check"]["assertions"]}}

    def statistical_entry(self, strategy_id, predicates=None):
        """A statistical (claim-less) entry, built directly.

        Induction no longer writes a strategic entry from a cell's means —
        knowledge is only written from an agent-formed strategy — but the
        StrategicBank still holds statistical entries (from a migration or a
        harness) and they must not merge with a claim entry. This builds one
        without going through the removed statistical path."""
        return StrategicEntry(
            entry_id=StrategicEntry.new_id(),
            strategy_id=strategy_id,
            pattern={"predicates": dict(
                predicates or {"family": "scheduling",
                               "temporal_coupling": [0.5, 1.0]})},
            expected_quality_hat=0.5,
            quality_interval=(0.0, 1.0),
            support_n=2,
        )


class TestClaimGeneration(ClaimCase):
    """Generation: cross-task evidence forms a NEW knowledge object, with no
    recommended/comparison strategy and no pre-existing statistical entry."""

    def test_claim_only_entry_is_created_without_any_statistical_entry(self):
        self.seed_cross_period()
        self.assertEqual(self.sbank.count(), 0)   # nothing pre-exists
        claim = self.cross_period_claim()
        out = self.engine.submit_relation(
            claim, verify=self.verify_payload(claim))
        self.assertIsNotNone(out["saved"])
        entry = self.sbank.get(out["saved"])
        # The claim does not belong to a catalog strategy: its subject names
        # the principle, it carries NO statistical claim, and its stated
        # claim lives directly on the entry.
        self.assertEqual(entry.strategy_id, "principle:cross_period_state")
        self.assertEqual(entry.support_n, 0)
        self.assertTrue(entry.is_claim_only)
        self.assertIsNotNone(entry.claim)
        self.assertIn("跨期", entry.claim["text"])

    def test_evidence_identity_is_derived_not_submitted(self):
        """tasks/family/strategy ids come from the recorded facts: the caller
        never submits a second, contradictory identity."""
        self.seed_cross_period()
        claim = self.cross_period_claim()
        # The caller supplies NO tasks / family / strategy_ids.
        self.assertNotIn("tasks", claim)
        self.assertNotIn("family", claim)
        out = self.engine.submit_relation(claim)
        stored = self.sbank.get(out["saved"]).claim
        self.assertEqual(stored["tasks"], ["T1", "T2", "T3"])
        self.assertEqual(stored["family"], "scheduling")
        self.assertEqual(stored["strategy_ids"], ["S01"])

    def test_unknown_execution_is_refused(self):
        self.seed_cross_period()
        claim = self.cross_period_claim()
        claim["evidence"].append({"execution_id": "ex_missing",
                                  "role": "preserved"})
        out = self.engine.submit_relation(claim)
        self.assertIsNone(out.get("saved"))
        self.assertIn("unknown execution", out["skipped"])
        self.assertEqual(self.sbank.count(), 0)

    def test_role_is_required_on_every_reference(self):
        with self.assertRaises(ValueError) as ctx:
            validate_claim({"text": "c",
                            "evidence": [{"execution_id": "ex_x"}]})
        self.assertIn("role", str(ctx.exception))

    def test_claim_text_is_required(self):
        with self.assertRaises(ValueError):
            validate_claim({"evidence": [{"execution_id": "ex_x",
                                          "role": "a"}]})

    def test_statistical_induction_is_untouched(self):
        """Submitting a claim must not perform statistical induction: the
        peer evidence never enters the target's statistics."""
        self.seed_cross_period()
        claim = self.cross_period_claim()
        self.engine.submit_relation(claim)
        # No statistical entry was created for the evidence's strategy.
        self.assertEqual(self.sbank.count(), 1)
        entry = self.sbank.list()[0]
        self.assertEqual(entry.support_n, 0)
        self.assertEqual(entry.expected_quality_hat, 0.0)


class TestClaimAssertions(ClaimCase):
    """Verification checks the DECLARED assertions over the referenced
    evidence — not a natural-language sentence, and not a per-kind template."""

    def test_probe_and_paired_comparison_verify(self):
        self.seed_cross_period()
        claim = self.cross_period_claim()
        out = self.engine.submit_relation(
            claim, verify=self.verify_payload(claim))
        entry = self.sbank.get(out["saved"])
        self.assertEqual(entry.verification_state, "verified")
        scope = entry.verification["scope"]
        self.assertEqual(scope["tasks"], ["T1", "T2", "T3"])
        self.assertEqual(scope["assertions_checked"], [0, 1])
        self.assertEqual(scope["assertions_unchecked"], [])

    def test_paired_assertion_ignores_records_without_a_counterpart(self):
        """T3 has no 'dropped' side: it must NOT be mixed into the paired
        statistic — it is recorded as unpaired in the scope."""
        self.seed_cross_period()
        claim = self.cross_period_claim()
        out = self.engine.submit_relation(
            claim, verify=self.verify_payload(claim))
        checks = self.sbank.get(out["saved"]).verification["checks"]
        paired = [c for c in checks
                  if c.get("check") == "assertion_paired_scope"]
        self.assertEqual(paired[0]["n_pairs"], 2)
        self.assertIn("ex_t3p", paired[0]["unpaired_execution_ids"])

    def test_counterexample_refutes_a_paired_all_assertion(self):
        """A new comparable pair that violates the declared direction/gap
        refutes the claim — 'all pairs' means every pair."""
        self.seed_cross_period()
        claim = self.cross_period_claim()
        self.engine.submit_relation(claim, verify=self.verify_payload(claim))
        self.bank.append(self.exec_record("ex_t9d", "T9", status="feasible",
                                          gap=0.02))
        self.bank.append(self.exec_record("ex_t9p", "T9", status="feasible",
                                          gap=0.03))
        claim_v2 = self.cross_period_claim(
            extra_evidence=[{"execution_id": "ex_t9d", "role": "dropped"},
                            {"execution_id": "ex_t9p", "role": "preserved"}])
        out = self.engine.submit_relation(
            claim_v2, verify=self.verify_payload(claim_v2))
        entry = self.sbank.get(out["saved"])
        self.assertEqual(entry.verification_state, "refuted")
        self.assertFalse(entry.is_published)

    def test_mean_aggregation_tolerates_one_negative_pair(self):
        """With ``aggregation=mean`` a single unfavourable pair does not
        refute the claim: the batch mean is what is asserted."""
        self.seed_cross_period()
        self.bank.append(self.exec_record("ex_t9d", "T9", status="feasible",
                                          gap=0.02))
        self.bank.append(self.exec_record("ex_t9p", "T9", status="feasible",
                                          gap=0.03))
        claim = self.cross_period_claim(
            extra_evidence=[{"execution_id": "ex_t9d", "role": "dropped"},
                            {"execution_id": "ex_t9p", "role": "preserved"}])
        claim["check"]["assertions"] = [
            {"kind": "comparison", "metric": "quality",
             "roles_a": ["preserved"], "roles_b": ["dropped"],
             "direction": "higher", "min_gap": 0.2,
             "mode": "paired", "aggregation": "mean"}]
        out = self.engine.submit_relation(claim,
                                          verify=self.verify_payload(claim))
        entry = self.sbank.get(out["saved"])
        # (T1: 1.0-0.0, T2: 0.95-0.60, T9: 0.97-0.98) mean = (1.0+0.35-0.01)/3
        self.assertEqual(entry.verification_state, "verified")

    def test_unmeasured_metric_is_insufficient_not_refuted(self):
        """A metric no record measured can never decide a claim: 'we could
        not measure it' is insufficient, never a refutation."""
        for rec in [
            self.exec_record("ex_t1d", "T1", status="infeasible",
                             feasible=False),
            self.exec_record("ex_t1p", "T1"),
            self.exec_record("ex_t2d", "T2", status="feasible", gap=0.40),
            self.exec_record("ex_t2p", "T2", status="optimal", gap=0.05),
        ]:
            rec.cost.measured = {"llm_tokens"}   # tool_calls never measured
            self.bank.append(rec)
        claim = self.cross_period_claim()
        claim["evidence"] = [
            {"execution_id": "ex_t1d", "role": "dropped"},
            {"execution_id": "ex_t1p", "role": "preserved"},
            {"execution_id": "ex_t2d", "role": "dropped"},
            {"execution_id": "ex_t2p", "role": "preserved"}]
        claim["check"]["assertions"] = [
            {"kind": "comparison", "metric": "cost:tool_calls",
             "roles_a": ["preserved"], "roles_b": ["dropped"],
             "direction": "lower", "min_gap": 0.1, "mode": "group"}]
        out = self.engine.submit_relation(claim,
                                          verify=self.verify_payload(claim))
        entry = self.sbank.get(out["saved"])
        self.assertEqual(entry.verification_state, "insufficient_evidence")

    def test_no_assertion_is_insufficient(self):
        """A claim with nothing declared checkable stays unverified: the
        framework computes only what is computable."""
        self.seed_cross_period()
        claim = self.cross_period_claim()
        claim["check"] = {"assertions": []}
        out = self.engine.submit_relation(claim,
                                          verify=self.verify_payload(claim))
        entry = self.sbank.get(out["saved"])
        self.assertEqual(entry.verification_state, "insufficient_evidence")

    def test_only_declared_parts_are_covered(self):
        """A passing verdict records WHICH checks ran and states what is NOT
        covered, so 'verified' is never read as a proof of the whole prose."""
        self.seed_cross_period()
        claim = self.cross_period_claim()
        claim["claim"] = "质量更高且 token 更低"
        out = self.engine.submit_relation(claim,
                                          verify=self.verify_payload(claim))
        block = self.sbank.get(out["saved"]).verification
        declared = block["scope"]["assertions_declared"]
        self.assertEqual(declared, ["probe", "comparison"])
        self.assertNotIn("cost", [str(d) for d in declared])
        self.assertIn("ONLY the declared checks", block["not_covered"])
        self.assertEqual(len(block["checked"]["execution_ids"]), 5)


class TestClaimPublication(ClaimCase):
    """Publication is the cross-task independence gate, applied to the
    claim's own evidence — a single-task fact is saved but not published."""

    def test_verified_claim_with_two_tasks_is_published(self):
        self.seed_cross_period()
        claim = self.cross_period_claim()
        out = self.engine.submit_relation(
            claim, verify=self.verify_payload(claim))
        self.assertTrue(out["publication"]["published"])
        self.assertEqual(out["publication"]["distinct_tasks"], 3)

    def test_single_task_claim_is_saved_but_not_published(self):
        self.seed_cross_period()
        claim = {
            "subject": "principle:one_task_only",
            "claim": "只在一个任务上观察到的修复",
            "evidence": [{"execution_id": "ex_t2d", "role": "before"},
                         {"execution_id": "ex_t2p", "role": "after"}],
            "check": {"assertions": [
                {"kind": "comparison", "metric": "quality",
                 "roles_a": ["after"], "roles_b": ["before"],
                 "direction": "higher", "min_gap": 0.2, "mode": "paired"}]},
        }
        out = self.engine.submit_relation(claim,
                                          verify=self.verify_payload(claim))
        # Saved, and the single-task fact itself verified...
        self.assertIsNotNone(out["saved"])
        entry = self.sbank.get(out["saved"])
        self.assertEqual(entry.verification_state, "verified")
        # ...but NOT published as transferable knowledge.
        self.assertFalse(out["publication"]["published"])
        self.assertTrue(any("task" in r
                            for r in out["publication"]["reasons"]))
        self.assertLess(len(claim_scope_tasks(entry)), CLAIM_MIN_TASKS)

    def test_unverified_claim_is_not_published(self):
        self.seed_cross_period()
        claim = self.cross_period_claim()
        # No verdict AND no embedded check: nothing computable to verify.
        claim.pop("check", None)
        out = self.engine.submit_relation(claim)   # no verify, no check
        self.assertIsNotNone(out["saved"])
        self.assertFalse(out["publication"]["published"])
        self.assertFalse(self.sbank.get(out["saved"]).is_published)

    def test_embedded_relation_check_is_honoured(self):
        """A ``check`` block INSIDE the relation is read, not silently
        ignored: the two spellings of a declared check are equivalent."""
        self.seed_cross_period()
        claim = self.cross_period_claim()
        out = self.engine.submit_relation(claim)   # embedded check only
        entry = self.sbank.get(out["saved"])
        self.assertEqual(entry.verification_state, "verified")
        self.assertTrue(out["publication"]["published"])
        self.assertEqual(out["check_note"]["check_source"],
                         "embedded_relation_check")

    def test_verify_arg_wins_over_an_embedded_check(self):
        """When BOTH are supplied the standalone ``--verify`` wins and the
        override is reported, never swallowed."""
        self.seed_cross_period()
        claim = self.cross_period_claim()
        payload = self.verify_payload(claim)
        # A deliberately WRONG embedded check that would refute if used.
        claim["check"] = {"assertions": [
            {"kind": "probe", "roles": ["preserved"],
             "path": "quality.feasible", "equals": False}]}
        out = self.engine.submit_relation(claim, verify=payload)
        self.assertEqual(out["check_note"]["check_source"], "verify_arg")
        self.assertIn("overridden", out["check_note"]["note"])
        # The standalone (correct) verdict won.
        self.assertEqual(self.sbank.get(out["saved"]).verification_state,
                         "verified")


class TestClaimRevision(ClaimCase):
    """Revision: a substantive change invalidates the old verdict; a fresh
    verification replaces it; an identical resubmission keeps it."""

    def test_substantive_change_without_reverification_marks_stale(self):
        self.seed_cross_period()
        claim = self.cross_period_claim()
        out = self.engine.submit_relation(
            claim, verify=self.verify_payload(claim))
        entry_id = out["saved"]
        self.assertTrue(out["publication"]["published"])
        # Same identity (subject+kind+cell), CHANGED claim, NO fresh verdict
        # and NO embedded check (so nothing re-verifies the revised claim).
        revised = self.cross_period_claim()
        revised["claim"] = "修订后的主张：只保留跨期状态的一部分"
        revised.pop("check", None)
        self.engine.submit_relation(revised)
        entry = self.sbank.get(entry_id)
        self.assertEqual(entry.verification_state, "verified")
        self.assertTrue(entry.verification["stale_after_revision"])
        self.assertFalse(entry.is_published)

    def test_fresh_verification_replaces_the_verdict(self):
        """A newly supplied verdict WINS: it must never be overwritten by
        the stale marker of the previous one."""
        self.seed_cross_period()
        claim = self.cross_period_claim()
        self.engine.submit_relation(claim, verify=self.verify_payload(claim))
        revised = self.cross_period_claim()
        revised["claim"] = "修订后的主张"
        out = self.engine.submit_relation(
            revised, verify=self.verify_payload(revised))
        entry = self.sbank.get(out["saved"])
        self.assertEqual(entry.verification_state, "verified")
        self.assertFalse(entry.verification.get("stale_after_revision"))

    def test_identical_resubmission_keeps_the_verdict(self):
        self.seed_cross_period()
        claim = self.cross_period_claim()
        out = self.engine.submit_relation(
            claim, verify=self.verify_payload(claim))
        self.assertEqual(self.sbank.get(out["saved"]).verification_state,
                         "verified")
        # Re-submit the SAME claim with no verification: the verdict stays.
        resubmit = self.cross_period_claim()
        resubmit.pop("check", None)
        out2 = self.engine.submit_relation(resubmit)
        entry = self.sbank.get(out2["saved"])
        self.assertEqual(entry.verification_state, "verified")
        self.assertFalse(entry.verification.get("stale_after_revision"))

    def test_resubmission_revises_rather_than_duplicates(self):
        """Same identity -> same entry: a revision must not append a second
        copy of the same knowledge object."""
        self.seed_cross_period()
        claim = self.cross_period_claim()
        self.engine.submit_relation(claim, verify=self.verify_payload(claim))
        self.engine.submit_relation(self.cross_period_claim())
        self.assertEqual(self.sbank.count(), 1)

    def test_explicit_target_entry_id_revises_that_entry(self):
        """A revision can name the entry to change directly, instead of
        re-deriving it from subject + cell + kind."""
        self.seed_cross_period()
        claim = self.cross_period_claim()
        first = self.engine.submit_relation(claim,
                                            verify=self.verify_payload(claim))
        revised = dict(self.cross_period_claim(),
                       claim="revised text entirely", target_entry_id=first["saved"])
        out = self.engine.submit_relation(revised)
        self.assertEqual(out["saved"], first["saved"])
        self.assertEqual(self.sbank.count(), 1)

    def test_unknown_target_entry_id_is_refused(self):
        """An explicit target that does not exist is refused, never silently
        turned into a different edit or a new entry."""
        self.seed_cross_period()
        claim = dict(self.cross_period_claim(),
                     target_entry_id="se_does_not_exist")
        out = self.engine.submit_relation(claim)
        self.assertIsNone(out.get("saved"))
        self.assertIn("unknown target_entry_id", out["skipped"])
        self.assertEqual(self.sbank.count(), 0)

    def test_two_claims_under_one_subject_stay_independent(self):
        """#8: two independent claims must NOT share an entry, and a
        verification on one must never leak to the other."""
        self.seed_cross_period()
        first = self.cross_period_claim()
        first["kind"] = "intervention_recovery"
        out1 = self.engine.submit_relation(
            first, verify=self.verify_payload(first))
        # A DIFFERENT kind under the same subject is a second claim, and it
        # carries NO verdict of its own (no verify, no embedded check).
        other = self.cross_period_claim()
        other["kind"] = "structural_reproduction"
        other["claim"] = "另一条独立主张"
        other.pop("check", None)
        out2 = self.engine.submit_relation(other)
        self.assertNotEqual(out1["saved"], out2["saved"])
        self.assertEqual(self.sbank.count(), 2)
        # The second was never verified: it must not inherit the first's
        # verdict.
        independent = self.sbank.get(out2["saved"])
        self.assertEqual(independent.verification_state, "unverified")
        self.assertFalse(independent.is_published)

    def test_claim_and_statistical_claim_do_not_merge(self):
        """A statistical entry and a claim entry under the same strategy id
        are DIFFERENT knowledge objects: the claim must not attach to the
        statistical entry or inherit its verdict."""
        self.seed_cross_period()
        stats_entry = self.statistical_entry("S01")
        self.sbank.add(stats_entry)
        claim = self.cross_period_claim()
        claim["subject"] = "S01"
        out = self.engine.submit_relation(
            claim, verify=self.verify_payload(claim))
        self.assertNotEqual(out["saved"], stats_entry.entry_id)
        self.assertEqual(self.sbank.count(), 2)
        self.assertIsNone(self.sbank.get(stats_entry.entry_id).claim)


class TestClaimRecall(ClaimCase):
    """Transfer: a verified claim reaches a future task's recall through the
    ORDINARY entry path (no special section)."""

    def task(self, task_id="NEW1"):
        return {"task_id": task_id,
                "description": "scheduling with cross-period state",
                "family": "scheduling",
                "annotations": {"coupling": {"resource_coupling": 0.2,
                                             "temporal_coupling": 0.7,
                                             "route_complexity": 0.1}}}

    def test_verified_claim_reaches_recall(self):
        self.seed_cross_period()
        claim = self.cross_period_claim()
        self.h.induce(relations=[claim],
                      verify=self.verify_payload(claim))
        recalled = self.h.recall(self.task())
        recs = recalled["recommendations"]
        self.assertTrue(recs)
        item = next(r for r in recs
                    if r["strategy_id"] == "principle:cross_period_state")
        self.assertEqual(item["evidence"], "strategic_entry")
        self.assertEqual(item["knowledge"]["verification_state"], "verified")
        self.assertIsNotNone(item["knowledge"]["claim"])
        self.assertIn("跨期", item["knowledge"]["claim"]["text"])

    def test_claim_survives_beside_an_unpublished_statistical_entry(self):
        """A statistical entry for the same strategy is unverified; the
        verified claim is a SEPARATE entry and must still be recalled."""
        self.seed_cross_period()
        self.sbank.add(self.statistical_entry("S01"))
        claim = self.cross_period_claim()
        claim["subject"] = "S01"
        self.engine.submit_relation(claim, verify=self.verify_payload(claim))
        # The claim is a separate entry, so the S01 recommendation now shows
        # the verified claim as its own knowledge object.
        recalled = self.h.recall(self.task())
        self.assertTrue(any(r["strategy_id"] == "S01"
                            and r["knowledge"]["claim"] is not None
                            for r in recalled["recommendations"]))

    def test_refuted_claim_is_not_offered_as_knowledge(self):
        self.seed_cross_period()
        claim = self.cross_period_claim()
        self.h.induce(relations=[claim], verify=self.verify_payload(claim))
        self.assertTrue(self.h.recall(self.task())["recommendations"])
        # A counterexample refutes it.
        self.bank.append(self.exec_record("ex_t9d", "T9", status="feasible",
                                          gap=0.02))
        self.bank.append(self.exec_record("ex_t9p", "T9", status="feasible",
                                          gap=0.03))
        claim_v2 = self.cross_period_claim(
            extra_evidence=[{"execution_id": "ex_t9d", "role": "dropped"},
                            {"execution_id": "ex_t9p", "role": "preserved"}])
        self.h.induce(relations=[claim_v2],
                      verify=self.verify_payload(claim_v2))
        # The refuted claim no longer appears in recommendations...
        recs = self.h.recall(self.task())["recommendations"]
        self.assertFalse(any(r["strategy_id"] == "principle:cross_period_state"
                             for r in recs))
        # ...but the offline view still shows it, with its state.
        offline = self.h.recall(self.task(), include_unverified=True)
        held = offline["held_claims"]
        self.assertEqual(held[0]["verification_state"], "refuted")

    def test_newer_evidence_since_verification_is_reported(self):
        self.seed_cross_period()
        claim = self.cross_period_claim()
        self.h.induce(relations=[claim], verify=self.verify_payload(claim))
        self.bank.append(self.exec_record("ex_late", "TLATE", tc=0.9))
        offline = self.h.recall(self.task(), include_unverified=True)
        # A published claim reaches recommendations; the held-claims section
        # is where the newer-evidence annotation is most visible for a
        # claim-only entry that is not (yet) admitted.
        entry = next(e for e in self.h.sbank.list()
                     if e.claim is not None)
        self.assertGreaterEqual(
            self.h._newer_evidence_count({"entry_id": entry.entry_id}), 1)


class TestClaimDryRunAndVeto(ClaimCase):
    """A rehearsal writes nothing, and a retired pattern cannot resurrect."""

    def test_dry_run_writes_nothing(self):
        self.seed_cross_period()
        claim = self.cross_period_claim()
        out = self.engine.submit_relation(
            claim, dry_run=True, verify=self.verify_payload(claim))
        self.assertIsNone(out["saved"])
        self.assertIn("would_create", out)
        self.assertEqual(self.sbank.count(), 0)

    def test_cold_archive_veto_blocks_then_force_lifts(self):
        self.seed_cross_period()
        claim = self.cross_period_claim()
        first = self.engine.submit_relation(claim)
        entry = self.sbank.get(first["saved"])
        entry.status = "suspect"
        self.sbank.update(entry)
        card = self.sbank.retire(entry.entry_id, reason="did not reproduce")
        # Same subject + same conditions -> the same pattern is vetoed.
        blocked = self.engine.submit_relation(self.cross_period_claim())
        self.assertIsNone(blocked["saved"])
        self.assertEqual(blocked["vetoed"]["pattern_hash"], card.pattern_hash)
        forced = self.engine.submit_relation(self.cross_period_claim(),
                                             force=True)
        self.assertIsNotNone(forced["saved"])
        self.assertEqual(self.sbank.cold_archive(), [])


class TestClaimRoundTrip(ClaimCase):
    """Serialization: the claim survives the storage boundary intact."""

    def test_claim_round_trips(self):
        self.seed_cross_period()
        claim = self.cross_period_claim()
        out = self.engine.submit_relation(
            claim, verify=self.verify_payload(claim))
        entry = self.sbank.get(out["saved"])
        again = StrategicEntry.from_dict(entry.to_dict())
        self.assertIsNotNone(again.claim)
        self.assertEqual(again.claim["text"], entry.claim["text"])
        self.assertEqual(again.claim["evidence"], entry.claim["evidence"])
        self.assertEqual(again.verification_state, "verified")
        self.assertTrue(again.is_claim_only)

    def test_legacy_entry_with_relations_key_does_not_crash(self):
        """A legacy payload that still carries a ``relations`` list loads
        with the field simply absent (the claim is None) — read compatibility
        only, never re-interpreted as a claim."""
        entry = StrategicEntry(entry_id="se_legacy", strategy_id="S01",
                               pattern={"predicates": {"family": "routing"}})
        payload = entry.to_dict()
        payload["relations"] = [{"relation_id": "rel_x", "claim": "old"}]
        loaded = StrategicEntry.from_dict(payload)
        self.assertIsNone(loaded.claim)
        self.assertFalse(loaded.is_claim_only)


if __name__ == "__main__":
    unittest.main()
