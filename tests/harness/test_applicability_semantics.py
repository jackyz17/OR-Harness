"""Stage 2: Strategic Knowledge semantics — machine check vs applicability.

The framework supplies deterministic structural checks and the rule's own
``when``; it never decides the semantic condition and never treats a recall
hit as adoption.
"""
from __future__ import annotations

import copy
import unittest

from or_harness.api import ORHarness
from or_harness.core.schema import ProblemProfile, StrategicEntry

from tests.harness.helpers import HarnessTestCase
from tests.harness.test_vector_recall import (
    REQ_A, VectorCase, _entry, _task,
)


def _profile(problem_id="t1", family="routing", **coupling):
    values = {"semantic_coupling": 0.8, "resource_coupling": 0.3,
              "temporal_coupling": 0.2, "route_complexity": 0.8}
    values.update(coupling)
    return ProblemProfile(
        problem_id=problem_id, family=family,
        scale_features={"n_vars": 100.0, "n_constraints": 50.0,
                        "n_int_vars": 100.0, "density": 0.01},
        **values)


def _rule_entry(entry_id, when, do, expect, *, predicates=None,
                strategy_id="S04", verified=True):
    return StrategicEntry(
        entry_id=entry_id, strategy_id=strategy_id,
        pattern={"predicates": copy.deepcopy(predicates or {})},
        quality_estimated=False, support_n=1,
        rule={
            "schema_version": "or-rule/2",
            "view_version": 1,
            "active_version": 1,
            "state": "active",
            "version_state": "active",
            "body": {"when": when, "do": do, "expect": expect,
                     "predicates": copy.deepcopy(predicates or {})},
        },
        verification=({"state": "verified", "claim": expect,
                       "conclusion": "declared check passed"}
                      if verified else {}),
        provenance=["ex_1"])


class TestMachineApplicabilityLayers(unittest.TestCase):
    def test_empty_predicates_are_not_unconditional_applicability(self):
        from or_harness.strategy.applicability import machine_applicability
        result = machine_applicability(_profile(), {})
        self.assertEqual(result["state"], "not_specified")
        self.assertFalse(result["machine_compatible"])
        self.assertTrue(result["requires_agent_assessment"])
        self.assertIn("NOT unconditional", result["reason"])
        self.assertIn("`when`", result["reason"])

    def test_supported_conditions_can_be_compatible(self):
        from or_harness.strategy.applicability import machine_applicability
        result = machine_applicability(
            _profile(resource_coupling=0.9),
            {"resource_coupling": [0.75, 1.0]})
        self.assertEqual(result["state"], "compatible")
        self.assertTrue(result["machine_compatible"])
        # A machine pass is NECESSARY, never the semantic verdict.
        self.assertEqual(result["state"] != "compatible", False)
        self.assertIn("NECESSARY", result["note"])

    def test_unknown_structural_value_stays_unknown(self):
        from or_harness.strategy.applicability import machine_applicability
        profile = ProblemProfile(problem_id="p", family="routing")
        result = machine_applicability(
            profile, {"resource_coupling": [0.75, 1.0]})
        self.assertEqual(result["state"], "unknown")
        self.assertFalse(result["machine_compatible"])
        self.assertIn("unmeasured", result["reason"])

    def test_measured_structural_conflict_has_a_reason(self):
        from or_harness.strategy.applicability import machine_applicability
        result = machine_applicability(
            _profile(resource_coupling=0.1),
            {"resource_coupling": [0.75, 1.0]})
        self.assertEqual(result["state"], "conflict")
        self.assertFalse(result["machine_compatible"])
        self.assertTrue(result["conflicts"])
        self.assertIn("OUTSIDE", result["reason"])

    def test_family_label_difference_is_not_a_conflict(self):
        from or_harness.strategy.applicability import machine_applicability
        result = machine_applicability(
            _profile(family="routing", resource_coupling=0.9),
            {"family": "scheduling", "resource_coupling": [0.75, 1.0]})
        self.assertEqual(result["state"], "compatible")
        self.assertTrue(result["machine_compatible"])
        self.assertEqual(result["family_label"]["relation"], "different")
        self.assertIn("LABELS differ", result["reason"])
        self.assertIn("mechanism", result["reason"])

    def test_semantic_predicate_keys_are_never_defaulted_to_compatible(self):
        from or_harness.strategy.applicability import machine_applicability
        result = machine_applicability(
            _profile(), {"integrality": "integer"})
        self.assertEqual(result["state"], "unknown")
        self.assertFalse(result["machine_compatible"])
        self.assertEqual(result["semantic_predicate_keys"], ["integrality"])
        self.assertIn("only an agent can judge", result["reason"])


class TestRuleBodyAndAdoption(unittest.TestCase):
    def test_rule_when_do_expect_are_read_as_recorded(self):
        from or_harness.strategy.applicability import rule_body
        entry = _rule_entry(
            "1", "When every quantity must be an integer and demand is "
                 "indivisible.", "Use an integer formulation.",
            "The solver returns an integral optimum.")
        body = rule_body(entry)
        self.assertEqual(body["source"], "rule")
        self.assertIn("integer", body["when"])
        self.assertIn("integer formulation", body["do"])
        self.assertIn("integral optimum", body["expect"])

    def test_a_hit_is_requires_assessment_and_not_adopted(self):
        from or_harness.strategy.applicability import (
            applicability_annotations,
        )
        entry = _rule_entry(
            "2", "When the quantities are continuous.", "Use the LP.",
            "A continuous optimum is returned.")
        annotations = applicability_annotations(_profile(), entry)
        self.assertEqual(annotations["machine_applicability"]["state"],
                         "not_specified")
        self.assertEqual(
            annotations["semantic_applicability"]["state"],
            "requires_agent_assessment")
        self.assertTrue(annotations["requires_agent_assessment"])
        self.assertEqual(annotations["adoption_state"], "not_adopted")
        self.assertIn("NOT adoption", annotations["adoption_note"])

    def test_integer_and_continuous_rules_have_opposite_when_but_no_auto_verdict(self):
        integer = _rule_entry(
            "3", "When the required quantities are integers.",
            "Build an integer model.", "The answer is integral.")
        continuous = _rule_entry(
            "4", "When the required quantities may be fractional.",
            "Build a continuous model.", "The answer is continuous.")
        from or_harness.strategy.applicability import (
            applicability_annotations,
        )
        for profile_family in ("production", "logistics"):
            profile = _profile(family=profile_family)
            int_annotations = applicability_annotations(profile, integer)
            cont_annotations = applicability_annotations(profile, continuous)
            # Both are surfaced by semantic retrieval, both carry different
            # `when` texts, and NEITHER is machine-declared applicable.
            self.assertEqual(
                int_annotations["machine_applicability"]["state"],
                "not_specified")
            self.assertEqual(
                cont_annotations["machine_applicability"]["state"],
                "not_specified")
            self.assertNotEqual(integer.rule["body"]["when"],
                                continuous.rule["body"]["when"])
            # The decision is the agent's: an integer task may select the
            # integer rule; a continuous task the continuous one.
            chosen = (integer if profile_family == "production"
                      else continuous)
            self.assertIn("integer" if profile_family == "production"
                          else "fractional", chosen.rule["body"]["when"])

    def test_machine_strict_matches_vs_discovery_gate(self):
        from or_harness.strategy.applicability import (
            machine_matched, possibly_applicable,
        )
        profile = _profile(resource_coupling=0.9)
        self.assertFalse(machine_matched(profile, {}))
        self.assertTrue(possibly_applicable(profile, {}))
        self.assertFalse(machine_matched(
            profile, {"resource_coupling": [0.0, 0.25]}))
        self.assertFalse(possibly_applicable(
            profile, {"resource_coupling": [0.0, 0.25]}))

    def test_published_lifecycle_is_independent_of_predicates(self):
        entry = _rule_entry(
            "5", "When a valid bound exists.", "Use the bound.", "Bound holds.")
        self.assertTrue(entry.is_published)
        self.assertEqual(entry.verification_state, "verified")
        from or_harness.strategy.applicability import rule_body
        self.assertEqual(rule_body(entry)["source"], "rule")
        # Empty predicates do NOT change publication/verification state.
        self.assertEqual(entry.predicates, {})


class TestCrossFamilyDiscovery(VectorCase):
    def test_cross_family_rule_is_discovered_and_marked_for_assessment(self):
        self.solve(_task("t_a", description=REQ_A))
        # Family label differs and the structural cell is compatible: the
        # label difference must NOT hide the rule (a shared mechanism may
        # transfer), and the hit must remain un-adopted/un-assessed.
        eid = self.h.sbank.add(_entry(
            predicates={"family": "scheduling",
                        "resource_coupling": [0.25, 0.50]}))
        self.h.index_sync.sync_entries([eid])
        out = self.h.recall(_task("t_q", description=REQ_A,
                                  resource_coupling=0.3), top=5)
        hit = next(k for k in out["vector_recall"]["strategic_knowledge"]
                   if k["entry_id"] == eid)
        self.assertEqual(hit["structural_match"], "compatible")
        self.assertTrue(hit["reusable"])
        self.assertTrue(hit["requires_agent_assessment"])
        self.assertEqual(hit["machine_applicability"]["state"],
                         "compatible")
        self.assertEqual(
            hit["machine_applicability"]["family_label"]["relation"],
            "different")
        self.assertIn("LABELS differ", hit["reason"])
        self.assertEqual(hit["semantic_applicability"]["state"],
                         "requires_agent_assessment")
        self.assertEqual(hit["adoption_state"], "not_adopted")
        self.assertIn(hit["verification_state"],
                      ("", "unverified", "verified"))


class TestStoredEntryReaderCompatibility(HarnessTestCase):
    def setUp(self):
        super().setUp()
        self.h = ORHarness(home=self.home)
        self.addCleanup(self.h.close)

    def test_legacy_entry_with_empty_predicates_reads_via_new_layers(self):
        """Historical predicates stay readable; an empty set is not a match."""
        entry = StrategicEntry(
            entry_id="9", strategy_id="S09",
            pattern={"predicates": {}},
            claim={"text": "A legacy claim with no structured predicates.",
                   "evidence": []},
            verification={"state": "verified", "claim": "c",
                          "conclusion": "checked"})
        eid = self.h.sbank.add(entry)
        loaded = self.h.sbank.get(eid)
        self.assertTrue(loaded.is_published)
        self.assertEqual(loaded.verification_state, "verified")
        self.assertEqual(loaded.predicates, {})
        self.assertFalse(loaded.matches(_profile()))
        from or_harness.strategy.applicability import (
            applicability_annotations,
        )
        annotations = applicability_annotations(_profile(), loaded)
        self.assertEqual(annotations["machine_applicability"]["state"],
                         "not_specified")
        self.assertEqual(annotations["rule_body"]["source"],
                         "legacy_claim")
        self.assertIn("legacy claim", annotations["rule_body"]["expect"])


if __name__ == "__main__":
    unittest.main()
