"""Relation claims: structured cross-task strategic knowledge.

The point of this suite is that a claim is GENERATED, CHECKED and TRANSFERRED
— not merely that a field exists. It covers the plan's acceptance example
(no recommended/comparison strategy, no pre-existing statistical entry), the
publication gate, the assertion semantics (paired per-assertion, counterexample
handling), qualification of relation vs statistical knowledge, and revision.
"""
import unittest

from helpers import HarnessTestCase

from or_harness.api import ORHarness
from or_harness.core.schema import (StrategicEntry, published_relations,
                                    relation_is_published, relation_state,
                                    validate_relation)
from or_harness.strategy.experience_bank import ExperienceBank
from or_harness.strategy.induction import InductionEngine
from or_harness.strategy.stats import ConditionalStats
from or_harness.strategy.strategic_bank import StrategicBank


class RelationCase(HarnessTestCase):
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

    def cross_period_relation(self, extra_evidence=()):
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

    def verify_payload(self, relation):
        return {"purpose": "relation",
                "check": {"assertions": relation["check"]["assertions"]}}


class TestRelationGeneration(RelationCase):
    """Generation: cross-task evidence forms a NEW knowledge object without
    any recommended/comparison strategy and without a pre-existing
    statistical entry."""

    def test_relation_only_entry_is_created_without_any_statistical_entry(self):
        self.seed_cross_period()
        self.assertEqual(self.sbank.count(), 0)   # nothing pre-exists
        relation = self.cross_period_relation()
        out = self.engine.submit_relation(
            relation, verify=self.verify_payload(relation))
        self.assertIsNotNone(out["saved"])
        entry = self.sbank.get(out["saved"])
        # The claim does not belong to a catalog strategy: its subject names
        # the principle, and it carries NO statistical claim.
        self.assertEqual(entry.strategy_id, "principle:cross_period_state")
        self.assertEqual(entry.support_n, 0)
        self.assertTrue(entry.is_relation_only)
        self.assertEqual(len(entry.relations), 1)

    def test_evidence_identity_is_derived_not_submitted(self):
        """tasks/family/strategy ids come from the recorded facts: the caller
        never submits a second, contradictory identity."""
        self.seed_cross_period()
        relation = self.cross_period_relation()
        # The caller supplies NO tasks / family / strategy_ids.
        self.assertNotIn("tasks", relation)
        self.assertNotIn("family", relation)
        out = self.engine.submit_relation(relation)
        stored = self.sbank.get(out["saved"]).relations[0]
        self.assertEqual(stored["tasks"], ["T1", "T2", "T3"])
        self.assertEqual(stored["family"], "scheduling")
        self.assertEqual(stored["strategy_ids"], ["S01"])

    def test_unknown_execution_is_refused(self):
        self.seed_cross_period()
        relation = self.cross_period_relation()
        relation["evidence"].append({"execution_id": "ex_missing",
                                     "role": "preserved"})
        out = self.engine.submit_relation(relation)
        self.assertIsNone(out.get("saved"))
        self.assertIn("unknown execution", out["skipped"])
        self.assertEqual(self.sbank.count(), 0)

    def test_role_is_required_on_every_reference(self):
        with self.assertRaises(ValueError) as ctx:
            validate_relation({"claim": "c",
                               "evidence": [{"execution_id": "ex_x"}]})
        self.assertIn("role", str(ctx.exception))

    def test_claim_is_required(self):
        with self.assertRaises(ValueError):
            validate_relation({"evidence": [{"execution_id": "ex_x",
                                             "role": "a"}]})

    def test_statistical_induction_is_untouched(self):
        """Submitting a relation must not perform statistical induction: the
        peer evidence never enters the target's statistics."""
        self.seed_cross_period()
        relation = self.cross_period_relation()
        self.engine.submit_relation(relation)
        # No statistical entry was created for the evidence's strategy.
        self.assertEqual(self.sbank.count(), 1)
        entry = self.sbank.list()[0]
        self.assertEqual(entry.support_n, 0)
        self.assertEqual(entry.expected_quality_hat, 0.0)


class TestRelationAssertions(RelationCase):
    """Verification checks the DECLARED assertions over the referenced
    evidence — not a natural-language sentence, and not a per-kind template."""

    def test_probe_and_paired_comparison_verify(self):
        self.seed_cross_period()
        relation = self.cross_period_relation()
        out = self.engine.submit_relation(
            relation, verify=self.verify_payload(relation))
        stored = self.sbank.get(out["saved"]).relations[0]
        self.assertEqual(relation_state(stored), "verified")
        scope = stored["verification"]["scope"]
        self.assertEqual(scope["tasks"], ["T1", "T2", "T3"])
        self.assertEqual(scope["assertions_checked"], [0, 1])
        self.assertEqual(scope["assertions_unchecked"], [])

    def test_paired_assertion_ignores_records_without_a_counterpart(self):
        """T3 has no 'dropped' side: it must NOT be mixed into the paired
        statistic — it is recorded as unpaired in the scope."""
        self.seed_cross_period()
        relation = self.cross_period_relation()
        out = self.engine.submit_relation(
            relation, verify=self.verify_payload(relation))
        stored = self.sbank.get(out["saved"]).relations[0]
        paired = [c for c in stored["verification"]["checks"]
                  if c.get("check") == "assertion_paired_scope"]
        self.assertEqual(paired[0]["n_pairs"], 2)
        self.assertIn("ex_t3p", paired[0]["unpaired_execution_ids"])

    def test_counterexample_refutes_a_paired_all_assertion(self):
        """A new comparable pair that violates the declared direction/gap
        refutes the claim — 'all pairs' means every pair."""
        self.seed_cross_period()
        relation = self.cross_period_relation()
        self.engine.submit_relation(relation,
                                    verify=self.verify_payload(relation))
        self.bank.append(self.exec_record("ex_t9d", "T9", status="feasible",
                                          gap=0.02))
        self.bank.append(self.exec_record("ex_t9p", "T9", status="feasible",
                                          gap=0.03))
        relation_v2 = self.cross_period_relation(
            extra_evidence=[{"execution_id": "ex_t9d", "role": "dropped"},
                            {"execution_id": "ex_t9p", "role": "preserved"}])
        out = self.engine.submit_relation(
            relation_v2, verify=self.verify_payload(relation_v2))
        stored = self.sbank.get(out["saved"]).relations[0]
        self.assertEqual(relation_state(stored), "refuted")
        self.assertFalse(relation_is_published(stored))

    def test_mean_aggregation_tolerates_one_negative_pair(self):
        """With ``aggregation=mean`` a single unfavourable pair does not
        refute the claim: the batch mean is what is asserted."""
        self.seed_cross_period()
        self.bank.append(self.exec_record("ex_t9d", "T9", status="feasible",
                                          gap=0.02))
        self.bank.append(self.exec_record("ex_t9p", "T9", status="feasible",
                                          gap=0.03))
        relation = self.cross_period_relation(
            extra_evidence=[{"execution_id": "ex_t9d", "role": "dropped"},
                            {"execution_id": "ex_t9p", "role": "preserved"}])
        relation["check"]["assertions"] = [
            {"kind": "comparison", "metric": "quality",
             "roles_a": ["preserved"], "roles_b": ["dropped"],
             "direction": "higher", "min_gap": 0.2,
             "mode": "paired", "aggregation": "mean"}]
        out = self.engine.submit_relation(
            relation, verify=self.verify_payload(relation))
        stored = self.sbank.get(out["saved"]).relations[0]
        # (T1: 1.0-0.0, T2: 0.95-0.60, T9: 0.97-0.98) mean = (1.0+0.35-0.01)/3
        self.assertEqual(relation_state(stored), "verified")

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
        relation = self.cross_period_relation(
            extra_evidence=[])
        relation["evidence"] = [
            {"execution_id": "ex_t1d", "role": "dropped"},
            {"execution_id": "ex_t1p", "role": "preserved"},
            {"execution_id": "ex_t2d", "role": "dropped"},
            {"execution_id": "ex_t2p", "role": "preserved"}]
        relation["check"]["assertions"] = [
            {"kind": "comparison", "metric": "cost:tool_calls",
             "roles_a": ["preserved"], "roles_b": ["dropped"],
             "direction": "lower", "min_gap": 0.1, "mode": "group"}]
        out = self.engine.submit_relation(
            relation, verify=self.verify_payload(relation))
        stored = self.sbank.get(out["saved"]).relations[0]
        self.assertEqual(relation_state(stored), "insufficient_evidence")

    def test_no_assertion_is_insufficient(self):
        """A claim with nothing declared checkable stays unverified: the
        framework computes only what is computable."""
        self.seed_cross_period()
        relation = self.cross_period_relation()
        relation["check"] = {"assertions": []}
        out = self.engine.submit_relation(
            relation, verify=self.verify_payload(relation))
        stored = self.sbank.get(out["saved"]).relations[0]
        self.assertEqual(relation_state(stored), "insufficient_evidence")

    def test_only_declared_parts_are_covered(self):
        """A multi-part claim that only declares the quality assertion
        records the cost part as unchecked — it does not get a blanket
        'verified'."""
        self.seed_cross_period()
        relation = self.cross_period_relation()
        relation["claim"] = "质量更高且 token 更低"
        out = self.engine.submit_relation(
            relation, verify=self.verify_payload(relation))
        stored = self.sbank.get(out["saved"]).relations[0]
        declared = stored["verification"]["scope"]["assertions_declared"]
        self.assertEqual(declared, ["probe", "comparison"])
        self.assertNotIn("cost", [str(d) for d in declared])


class TestRelationPublication(RelationCase):
    """Publication is the cross-task independence gate, applied to the
    RELATION's own evidence — a single-task fact is saved but not published."""

    def test_verified_relation_with_two_tasks_is_published(self):
        self.seed_cross_period()
        relation = self.cross_period_relation()
        out = self.engine.submit_relation(
            relation, verify=self.verify_payload(relation))
        self.assertTrue(out["publication"]["published"])
        self.assertEqual(out["publication"]["distinct_tasks"], 3)

    def test_single_task_relation_is_saved_but_not_published(self):
        self.seed_cross_period()
        relation = {
            "subject": "principle:one_task_only",
            "claim": "只在一个任务上观察到的修复",
            "evidence": [{"execution_id": "ex_t2d", "role": "before"},
                         {"execution_id": "ex_t2p", "role": "after"}],
            "check": {"assertions": [
                {"kind": "comparison", "metric": "quality",
                 "roles_a": ["after"], "roles_b": ["before"],
                 "direction": "higher", "min_gap": 0.2, "mode": "paired"}]},
        }
        out = self.engine.submit_relation(
            relation, verify=self.verify_payload(relation))
        # Saved, and the single-task fact itself verified...
        self.assertIsNotNone(out["saved"])
        stored = self.sbank.get(out["saved"]).relations[0]
        self.assertEqual(relation_state(stored), "verified")
        # ...but NOT published as transferable knowledge.
        self.assertFalse(out["publication"]["published"])
        self.assertTrue(any("task" in r
                            for r in out["publication"]["reasons"]))
        entry = self.sbank.get(out["saved"])
        self.assertEqual(published_relations(entry), [])

    def test_unverified_relation_is_not_published(self):
        self.seed_cross_period()
        relation = self.cross_period_relation()
        out = self.engine.submit_relation(relation)   # no verify
        self.assertIsNotNone(out["saved"])
        self.assertFalse(out["publication"]["published"])
        entry = self.sbank.get(out["saved"])
        self.assertEqual(published_relations(entry), [])


class TestRelationRevision(RelationCase):
    """Revision: a substantive change invalidates the old verdict; a fresh
    verification replaces it; a refuted verdict is not 'stale'."""

    def test_substantive_change_without_reverification_marks_stale(self):
        self.seed_cross_period()
        relation = self.cross_period_relation()
        out = self.engine.submit_relation(
            relation, verify=self.verify_payload(relation))
        entry_id = out["saved"]
        self.assertTrue(out["publication"]["published"])
        # Same identity (subject+kind), CHANGED claim, NO fresh verification.
        revised = self.cross_period_relation()
        revised["claim"] = "修订后的主张：只保留跨期状态的一部分"
        out2 = self.engine.submit_relation(revised)
        stored = self.sbank.get(entry_id).relations[0]
        self.assertEqual(relation_state(stored), "verified")
        self.assertTrue(stored["verification"]["stale_after_revision"])
        self.assertFalse(relation_is_published(stored))

    def test_fresh_verification_replaces_the_verdict(self):
        """A newly supplied verdict WINS: it must never be overwritten by
        the stale marker of the previous one."""
        self.seed_cross_period()
        relation = self.cross_period_relation()
        self.engine.submit_relation(relation,
                                    verify=self.verify_payload(relation))
        revised = self.cross_period_relation()
        revised["claim"] = "修订后的主张"
        out = self.engine.submit_relation(
            revised, verify=self.verify_payload(revised))
        stored = self.sbank.get(out["saved"]).relations[0]
        self.assertEqual(relation_state(stored), "verified")
        self.assertFalse(stored["verification"].get("stale_after_revision"))

    def test_identical_resubmission_keeps_the_verdict(self):
        self.seed_cross_period()
        relation = self.cross_period_relation()
        out = self.engine.submit_relation(
            relation, verify=self.verify_payload(relation))
        stored = self.sbank.get(out["saved"]).relations[0]
        self.assertEqual(relation_state(stored), "verified")
        # Re-submit the SAME relation with no verification: the verdict stays.
        out2 = self.engine.submit_relation(self.cross_period_relation())
        stored2 = self.sbank.get(out2["saved"]).relations[0]
        self.assertEqual(relation_state(stored2), "verified")
        self.assertFalse(stored2["verification"].get("stale_after_revision"))

    def test_resubmission_revises_rather_than_duplicates(self):
        """Same subject+kind -> same relation identity: a revision must not
        append a second copy of the same knowledge object."""
        self.seed_cross_period()
        relation = self.cross_period_relation()
        self.engine.submit_relation(relation,
                                    verify=self.verify_payload(relation))
        self.engine.submit_relation(self.cross_period_relation())
        entry = self.sbank.list()[0]
        self.assertEqual(len(entry.relations), 1)

    def test_distinct_kinds_coexist_under_one_subject(self):
        self.seed_cross_period()
        relation = self.cross_period_relation()
        relation["kind"] = "intervention_recovery"
        self.engine.submit_relation(relation,
                                    verify=self.verify_payload(relation))
        other = self.cross_period_relation()
        other["kind"] = "structural_reproduction"
        self.engine.submit_relation(other)
        entry = self.sbank.list()[0]
        self.assertEqual(len(entry.relations), 2)


class TestRelationRecall(RelationCase):
    """Transfer: the relation must actually reach a future task's recall,
    including when the host entry's STATISTICAL claim is unpublished."""

    def task(self, task_id="NEW1"):
        return {"task_id": task_id,
                "description": "scheduling with cross-period state",
                "family": "scheduling",
                "annotations": {"coupling": {"resource_coupling": 0.2,
                                             "temporal_coupling": 0.7,
                                             "route_complexity": 0.1}}}

    def test_verified_relation_reaches_recall(self):
        self.seed_cross_period()
        relation = self.cross_period_relation()
        self.h.induce(relations=[relation],
                      verify=self.verify_payload(relation))
        recalled = self.h.recall(self.task())
        knowledge = recalled["knowledge"]
        self.assertTrue(knowledge)
        item = knowledge[0]
        self.assertEqual(item["verification_state"], "verified")
        self.assertTrue(item["published"])
        self.assertIn("跨期", item["claim"])
        self.assertEqual(item["tasks"], ["T1", "T2", "T3"])
        self.assertIn("temporal_coupling", item["conditions"]["predicates"])

    def test_relation_survives_an_unpublished_statistical_host(self):
        """The host entry's statistical claim is unverified (no --verify for
        it), yet the verified relation must still be recallable."""
        self.seed_cross_period()
        # Build a statistical entry for S01 in the same cell, unverified.
        self.engine.induce(self.profile("T1"), "S01")
        host = self.sbank.list(strategy_id="S01")[0]
        self.assertNotEqual(host.verification_state, "verified")
        relation = self.cross_period_relation()
        relation["subject"] = "S01"
        out = self.engine.submit_relation(
            relation, verify=self.verify_payload(relation))
        self.assertEqual(out["saved"], host.entry_id)
        recalled = self.h.recall(self.task())
        self.assertTrue(recalled["knowledge"],
                        "a verified relation must reach recall even when its "
                        "host's statistical claim is unpublished")

    def test_refuted_relation_is_not_offered_as_knowledge(self):
        self.seed_cross_period()
        relation = self.cross_period_relation()
        self.h.induce(relations=[relation],
                      verify=self.verify_payload(relation))
        self.assertTrue(self.h.recall(self.task())["knowledge"])
        # A counterexample refutes it.
        self.bank.append(self.exec_record("ex_t9d", "T9", status="feasible",
                                          gap=0.02))
        self.bank.append(self.exec_record("ex_t9p", "T9", status="feasible",
                                          gap=0.03))
        relation_v2 = self.cross_period_relation(
            extra_evidence=[{"execution_id": "ex_t9d", "role": "dropped"},
                            {"execution_id": "ex_t9p", "role": "preserved"}])
        self.h.induce(relations=[relation_v2],
                      verify=self.verify_payload(relation_v2))
        self.assertEqual(self.h.recall(self.task())["knowledge"], [])
        # The offline/inspection view still shows it, with its state.
        offline = self.h.recall(self.task(), include_unverified=True)
        self.assertEqual(offline["knowledge"][0]["verification_state"],
                         "refuted")

    def test_newer_evidence_since_verification_is_reported(self):
        self.seed_cross_period()
        relation = self.cross_period_relation()
        self.h.induce(relations=[relation],
                      verify=self.verify_payload(relation))
        self.bank.append(self.exec_record("ex_late", "TLATE", tc=0.9))
        item = self.h.recall(self.task())["knowledge"][0]
        self.assertGreaterEqual(item["newer_evidence_since_verification"], 1)

    def test_structural_hits_carry_relations_and_boundaries(self):
        """The structured channel must not drop what the knowledge says."""
        from or_harness.world_model.context import _structural_hits
        hits = _structural_hits([{
            "strategy_id": "S01", "evidence_refs": ["se_x"],
            "evidence": "strategic_entry", "expected": {},
            "confidence": 0.5, "basis": "b",
            "risk_warnings": ["a boundary"],
            "relations": [{"relation_id": "rel_1", "claim": "c"}],
        }])
        content = hits[0]["content"]
        self.assertEqual(content["risk_warnings"], ["a boundary"])
        self.assertEqual(content["relations"][0]["relation_id"], "rel_1")


class TestRelationDryRunAndVeto(RelationCase):
    """A rehearsal writes nothing, and a retired pattern cannot resurrect."""

    def test_dry_run_writes_nothing(self):
        self.seed_cross_period()
        relation = self.cross_period_relation()
        out = self.engine.submit_relation(
            relation, dry_run=True, verify=self.verify_payload(relation))
        self.assertIsNone(out["saved"])
        self.assertIn("would_create", out)
        self.assertEqual(self.sbank.count(), 0)

    def test_cold_archive_veto_blocks_then_force_lifts(self):
        self.seed_cross_period()
        relation = self.cross_period_relation()
        first = self.engine.submit_relation(relation)
        entry = self.sbank.get(first["saved"])
        entry.status = "suspect"
        self.sbank.update(entry)
        card = self.sbank.retire(entry.entry_id, reason="did not reproduce")
        # Same subject + same conditions -> the same pattern is vetoed.
        blocked = self.engine.submit_relation(self.cross_period_relation())
        self.assertIsNone(blocked["saved"])
        self.assertEqual(blocked["vetoed"]["pattern_hash"], card.pattern_hash)
        forced = self.engine.submit_relation(self.cross_period_relation(),
                                             force=True)
        self.assertIsNotNone(forced["saved"])
        self.assertEqual(self.sbank.cold_archive(), [])


class TestRelationRoundTrip(RelationCase):
    """Serialization: relations survive the storage boundary intact."""

    def test_relations_round_trip(self):
        self.seed_cross_period()
        relation = self.cross_period_relation()
        out = self.engine.submit_relation(
            relation, verify=self.verify_payload(relation))
        entry = self.sbank.get(out["saved"])
        again = StrategicEntry.from_dict(entry.to_dict())
        self.assertEqual(len(again.relations), 1)
        self.assertEqual(again.relations[0]["relation_id"],
                         entry.relations[0]["relation_id"])
        self.assertEqual(relation_state(again.relations[0]), "verified")
        self.assertTrue(again.is_relation_only)

    def test_legacy_entry_without_relations_reads_empty(self):
        entry = StrategicEntry(entry_id="se_legacy", strategy_id="S01",
                               pattern={"predicates": {"family": "routing"}})
        payload = entry.to_dict()
        payload.pop("relations", None)
        self.assertEqual(StrategicEntry.from_dict(payload).relations, [])
        self.assertFalse(StrategicEntry.from_dict(payload).is_relation_only)


if __name__ == "__main__":
    unittest.main()
