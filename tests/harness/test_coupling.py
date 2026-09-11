"""Tests for the Coupling-Aware Intermediate Representation (CIR).

Covers:
- schema round-trip (to_dict / from_dict / to_json)
- L1 format validation (empty names, duplicates, bad evidence)
- L2 referential integrity (dangling source/target/member/resource)
- domain-generality (unknown kinds accepted, no hard-coded enum)
- structural inference: co-occurrence → depends_on only (never semantic)
- semantic upgrade: capacity keyword + resource-kind entity → uses_resource
- shared_bottleneck derivation
- CIR ↔ model cross-check (missing decisions, relation not in model)
- understand() high-level entry (no coupling field, with coupling, with model)
"""
import unittest

from helpers import HarnessTestCase

from or_harness.core.coupling import (
    CouplingAwareIR,
    Entity,
    Decision,
    ConstraintRef,
    Relation,
    CouplingGroup,
    CouplingIssue,
    validate_cir,
    infer_structural_relations,
    derive_coupling_groups,
    render_modeling_guidance,
    cross_check_cir_model,
    understand,
)
from or_harness.profiling.model_syntax import parse_model


class TestCIRSchema(HarnessTestCase):
    """Schema round-trip and domain-generality."""

    def test_round_trip_empty(self):
        cir = CouplingAwareIR()
        d = cir.to_dict()
        cir2 = CouplingAwareIR.from_dict(d)
        self.assertEqual(cir2.to_dict(), d)

    def test_round_trip_populated(self):
        cir = CouplingAwareIR(
            entities=[Entity(name="R1", kind="resource")],
            decisions=[Decision(name="x", kind="production", indexes=["i", "t"])],
            constraints=[ConstraintRef(id="C1", kind="capacity", expr="sum(x) <= cap")],
            relations=[Relation(source="x", target="R1", type="uses_resource",
                                evidence="semantic", detail="test")],
            coupling_groups=[CouplingGroup(type="shared_bottleneck",
                                           members=["x"], resource="R1",
                                           implication="impl")],
        )
        d = cir.to_dict()
        cir2 = CouplingAwareIR.from_dict(d)
        self.assertEqual(cir2.to_dict(), d)

    def test_unknown_kind_accepted(self):
        """Domain-generality: any kind string is accepted, never rejected."""
        cir = CouplingAwareIR(
            entities=[Entity(name="E1", kind="completely_unknown_domain_thing")],
            decisions=[Decision(name="d1", kind="exotic_decision_type")],
        )
        validate_cir(cir)
        self.assertTrue(cir.passed)

    def test_to_json(self):
        cir = CouplingAwareIR(entities=[Entity(name="E1")])
        s = cir.to_json()
        self.assertIn("E1", s)


class TestCIRValidation(HarnessTestCase):
    """L1 format + L2 referential integrity."""

    def test_empty_entity_name(self):
        cir = CouplingAwareIR(entities=[Entity(name="")])
        validate_cir(cir)
        self.assertFalse(cir.passed)
        self.assertEqual(cir.issues[0].layer, "L1")
        self.assertEqual(cir.issues[0].code, "empty_name")

    def test_duplicate_name_across_types(self):
        cir = CouplingAwareIR(
            entities=[Entity(name="x")],
            decisions=[Decision(name="x")],
        )
        validate_cir(cir)
        self.assertFalse(cir.passed)
        self.assertTrue(any(i.code == "duplicate_name" for i in cir.issues))

    def test_bad_evidence_level(self):
        cir = CouplingAwareIR(
            entities=[Entity(name="A"), Entity(name="B")],
            relations=[Relation(source="A", target="B", evidence="invented")],
        )
        validate_cir(cir)
        self.assertFalse(cir.passed)
        self.assertTrue(any(i.code == "bad_evidence" for i in cir.issues))

    def test_dangling_source(self):
        cir = CouplingAwareIR(
            entities=[Entity(name="A")],
            relations=[Relation(source="A", target="nonexistent")],
        )
        validate_cir(cir)
        self.assertFalse(cir.passed)
        self.assertTrue(any(i.code == "dangling_target" for i in cir.issues))

    def test_dangling_group_member(self):
        cir = CouplingAwareIR(
            entities=[Entity(name="A")],
            coupling_groups=[CouplingGroup(type="shared_bottleneck",
                                           members=["ghost"], resource="A")],
        )
        validate_cir(cir)
        self.assertFalse(cir.passed)
        self.assertTrue(any(i.code == "dangling_member" for i in cir.issues))

    def test_dangling_group_resource(self):
        cir = CouplingAwareIR(
            entities=[Entity(name="A")],
            coupling_groups=[CouplingGroup(type="shared_bottleneck",
                                           members=["A"], resource="ghost")],
        )
        validate_cir(cir)
        self.assertFalse(cir.passed)
        self.assertTrue(any(i.code == "dangling_resource" for i in cir.issues))

    def test_valid_cir_passes(self):
        cir = CouplingAwareIR(
            entities=[Entity(name="R1", kind="resource"), Entity(name="R2")],
            decisions=[Decision(name="x"), Decision(name="y")],
            constraints=[ConstraintRef(id="C1")],
            relations=[Relation(source="x", target="R1", type="uses_resource",
                                evidence="semantic"),
                       Relation(source="y", target="R2", type="uses_resource",
                                evidence="semantic")],
        )
        validate_cir(cir)
        self.assertTrue(cir.passed)


class TestStructuralInference(HarnessTestCase):
    """Co-occurrence → depends_on only; semantic upgrade requires semantics."""

    def test_cooccurrence_produces_only_structural_edges(self):
        """Two variables co-occur in a constraint → depends_on, evidence=structural."""
        model_text = """
SETS:
 i in Items = {a, b}
PARAMETERS:
 limit[i]
VARIABLES:
 x[i] continuous
 y[i] continuous
OBJECTIVE:
 maximize sum(i, x[i])
CONSTRAINTS:
 C1: sum(i, x[i] + y[i]) <= limit[i]
"""
        model = parse_model(model_text)
        cir = CouplingAwareIR()
        infer_structural_relations(cir, model)
        # x and y co-occur in C1 → at least one depends_on edge
        dep_edges = [r for r in cir.relations if r.type == "depends_on"]
        self.assertTrue(len(dep_edges) >= 1)
        for r in dep_edges:
            self.assertEqual(r.evidence, "structural")

    def test_no_semantic_upgrade_without_entity_semantics(self):
        """Co-occurrence alone must NOT produce uses_resource."""
        model_text = """
SETS:
 i in Items = {a, b}
PARAMETERS:
 limit[i]
VARIABLES:
 x[i] continuous
 y[i] continuous
OBJECTIVE:
 maximize sum(i, x[i])
CONSTRAINTS:
 C1: sum(i, x[i] + y[i]) <= limit[i]
"""
        model = parse_model(model_text)
        # CIR has no entities → no semantic upgrade possible
        cir = CouplingAwareIR()
        infer_structural_relations(cir, model)
        uses_resource = [r for r in cir.relations if r.type == "uses_resource"]
        self.assertEqual(uses_resource, [])

    def test_semantic_upgrade_with_resource_entity(self):
        """Structural edge + capacity constraint + resource-kind entity → uses_resource."""
        cir = CouplingAwareIR(
            entities=[Entity(name="cap", kind="resource")],
            decisions=[Decision(name="x")],
            constraints=[ConstraintRef(id="C1", kind="capacity",
                                       expr="sum(x) <= cap_limit")],
            relations=[Relation(source="x", target="cap", type="depends_on",
                                evidence="structural")],
        )
        infer_structural_relations(cir, parsed_model=None)
        upgraded = [r for r in cir.relations if r.type == "uses_resource"]
        self.assertEqual(len(upgraded), 1)
        self.assertEqual(upgraded[0].evidence, "semantic")

    def test_no_upgrade_when_entity_not_resource_kind(self):
        """Structural edge + capacity constraint + non-resource entity → no upgrade."""
        cir = CouplingAwareIR(
            entities=[Entity(name="cap", kind="product")],
            decisions=[Decision(name="x")],
            constraints=[ConstraintRef(id="C1", kind="capacity",
                                       expr="sum(x) <= cap_limit")],
            relations=[Relation(source="x", target="cap", type="depends_on",
                                evidence="structural")],
        )
        infer_structural_relations(cir, parsed_model=None)
        upgraded = [r for r in cir.relations if r.type == "uses_resource"]
        self.assertEqual(upgraded, [])

    def test_inference_idempotent(self):
        """Calling infer twice does not duplicate edges."""
        model_text = """
SETS:
 i in Items = {a, b}
PARAMETERS:
 limit[i]
VARIABLES:
 x[i] continuous
 y[i] continuous
OBJECTIVE:
 maximize sum(i, x[i])
CONSTRAINTS:
 C1: sum(i, x[i] + y[i]) <= limit[i]
"""
        model = parse_model(model_text)
        cir = CouplingAwareIR()
        infer_structural_relations(cir, model)
        n1 = len(cir.relations)
        infer_structural_relations(cir, model)
        self.assertEqual(len(cir.relations), n1)


class TestCouplingGroupDerivation(HarnessTestCase):
    """shared_bottleneck detection from uses_resource edges."""

    def test_shared_bottleneck_detected(self):
        cir = CouplingAwareIR(
            entities=[Entity(name="R1", kind="resource")],
            decisions=[Decision(name="x"), Decision(name="y"), Decision(name="z")],
            relations=[
                Relation(source="x", target="R1", type="uses_resource",
                         evidence="semantic"),
                Relation(source="y", target="R1", type="uses_resource",
                         evidence="semantic"),
                Relation(source="z", target="R1", type="uses_resource",
                         evidence="semantic"),
            ],
        )
        derive_coupling_groups(cir)
        groups = [g for g in cir.coupling_groups if g.type == "shared_bottleneck"]
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0].resource, "R1")
        self.assertEqual(set(groups[0].members), {"x", "y", "z"})
        self.assertIn("aggregate", groups[0].implication.lower())

    def test_no_bottleneck_with_single_consumer(self):
        cir = CouplingAwareIR(
            entities=[Entity(name="R1", kind="resource")],
            decisions=[Decision(name="x")],
            relations=[Relation(source="x", target="R1", type="uses_resource",
                                evidence="semantic")],
        )
        derive_coupling_groups(cir)
        groups = [g for g in cir.coupling_groups if g.type == "shared_bottleneck"]
        self.assertEqual(groups, [])

    def test_agent_declared_group_preserved(self):
        """Agent-declared coupling groups (e.g. route_convergence) are preserved."""
        cir = CouplingAwareIR(
            entities=[Entity(name="stage1")],
            decisions=[Decision(name="r1"), Decision(name="r2")],
            coupling_groups=[CouplingGroup(
                type="route_convergence", members=["r1", "r2"],
                resource="stage1", implication="custom")],
        )
        derive_coupling_groups(cir)
        groups = [g for g in cir.coupling_groups
                  if g.type == "route_convergence"]
        self.assertEqual(len(groups), 1)

    def test_guidance_rendered(self):
        cir = CouplingAwareIR(
            entities=[Entity(name="R1", kind="resource")],
            decisions=[Decision(name="x"), Decision(name="y")],
            relations=[
                Relation(source="x", target="R1", type="uses_resource",
                         evidence="semantic"),
                Relation(source="y", target="R1", type="uses_resource",
                         evidence="semantic"),
            ],
        )
        derive_coupling_groups(cir)
        guidance = render_modeling_guidance(cir)
        self.assertEqual(len(guidance), 1)
        self.assertEqual(guidance[0]["type"], "shared_bottleneck")
        self.assertIn("implication", guidance[0])
        self.assertIn("R1", guidance[0]["resource"])


class TestCrossCheck(HarnessTestCase):
    """CIR ↔ model consistency checks."""

    def test_missing_decision_in_model(self):
        model_text = """
SETS:
 i in Items = {a}
PARAMETERS:
 limit[i]
VARIABLES:
 x[i] continuous
OBJECTIVE:
 maximize sum(i, x[i])
CONSTRAINTS:
 C1: sum(i, x[i]) <= limit[i]
"""
        model = parse_model(model_text)
        cir = CouplingAwareIR(
            decisions=[Decision(name="x"), Decision(name="ghost_var")],
        )
        warnings = cross_check_cir_model(cir, model)
        codes = [w["code"] for w in warnings]
        self.assertIn("cir_decision_not_in_model", codes)

    def test_missing_decision_in_cir(self):
        model_text = """
SETS:
 i in Items = {a}
PARAMETERS:
 limit[i]
VARIABLES:
 x[i] continuous
 y[i] continuous
OBJECTIVE:
 maximize sum(i, x[i])
CONSTRAINTS:
 C1: sum(i, x[i] + y[i]) <= limit[i]
"""
        model = parse_model(model_text)
        cir = CouplingAwareIR(
            decisions=[Decision(name="x")],
        )
        warnings = cross_check_cir_model(cir, model)
        codes = [w["code"] for w in warnings]
        self.assertIn("model_var_not_in_cir", codes)

    def test_relation_not_in_model(self):
        """CIR declares a uses_resource edge that has no co-occurrence in model."""
        model_text = """
SETS:
 i in Items = {a}
PARAMETERS:
 limit[i]
VARIABLES:
 x[i] continuous
 y[i] continuous
OBJECTIVE:
 maximize sum(i, x[i])
CONSTRAINTS:
 C1: sum(i, x[i]) <= limit[i]
"""
        model = parse_model(model_text)
        cir = CouplingAwareIR(
            entities=[Entity(name="R1", kind="resource")],
            decisions=[Decision(name="x"), Decision(name="y")],
            relations=[Relation(source="x", target="y", type="uses_resource",
                                evidence="semantic")],
        )
        warnings = cross_check_cir_model(cir, model)
        codes = [w["code"] for w in warnings]
        self.assertIn("relation_not_in_model", codes)

    def test_no_warnings_when_consistent(self):
        model_text = """
SETS:
 i in Items = {a}
PARAMETERS:
 limit[i]
VARIABLES:
 x[i] continuous
 y[i] continuous
OBJECTIVE:
 maximize sum(i, x[i])
CONSTRAINTS:
 C1: sum(i, x[i] + y[i]) <= limit[i]
"""
        model = parse_model(model_text)
        cir = CouplingAwareIR(
            decisions=[Decision(name="x"), Decision(name="y")],
        )
        warnings = cross_check_cir_model(cir, model)
        self.assertEqual(warnings, [])

    def test_no_warnings_without_model(self):
        cir = CouplingAwareIR(decisions=[Decision(name="x")])
        warnings = cross_check_cir_model(cir, None)
        self.assertEqual(warnings, [])


class TestUnderstand(HarnessTestCase):
    """High-level understand() entry point."""

    def test_no_coupling_field(self):
        result = understand({"task_id": "t1", "family": "routing"})
        self.assertIsNone(result["cir"])
        self.assertIn("message", result)

    def test_with_coupling_field(self):
        task = {
            "task_id": "t1", "family": "scheduling",
            "coupling": {
                "entities": [
                    {"name": "R1", "kind": "resource"},
                ],
                "decisions": [
                    {"name": "x", "kind": "production"},
                    {"name": "y", "kind": "production"},
                ],
                "constraints": [
                    {"id": "C1", "kind": "capacity", "expr": "sum(x, y) <= cap_R1"},
                ],
                "relations": [
                    {"source": "x", "target": "R1", "type": "depends_on",
                     "evidence": "structural"},
                    {"source": "y", "target": "R1", "type": "depends_on",
                     "evidence": "structural"},
                ],
            },
        }
        result = understand(task)
        cir = result["cir"]
        self.assertIsNotNone(cir)
        self.assertEqual(len(cir["entities"]), 1)
        # Structural edges should be upgraded to uses_resource because:
        # - target entity R1 has kind="resource" (resource-like)
        # - constraint C1 has "cap" keyword (capacity semantics)
        uses = [r for r in cir["relations"] if r["type"] == "uses_resource"]
        self.assertEqual(len(uses), 2)
        # shared_bottleneck should be derived
        groups = [g for g in cir["coupling_groups"]
                  if g["type"] == "shared_bottleneck"]
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["resource"], "R1")
        # Guidance should be rendered
        guidance = result["modeling_guidance"]
        self.assertEqual(len(guidance), 1)
        self.assertIn("aggregate", guidance[0]["implication"].lower())

    def test_with_model_cross_check(self):
        task = {
            "task_id": "t1", "family": "routing",
            "coupling": {
                "entities": [{"name": "R1", "kind": "resource"}],
                "decisions": [{"name": "x"}, {"name": "ghost_var"}],
            },
            "model": """
SETS:
 i in Items = {a}
PARAMETERS:
 limit[i]
VARIABLES:
 x[i] continuous
OBJECTIVE:
 maximize sum(i, x[i])
CONSTRAINTS:
 C1: sum(i, x[i]) <= limit[i]
""",
        }
        result = understand(task)
        warnings = result["cir_warnings"]
        codes = [w["code"] for w in warnings]
        self.assertIn("cir_decision_not_in_model", codes)

    def test_bad_coupling_data(self):
        result = understand({"task_id": "t1", "family": "f",
                             "coupling": "not_a_dict"})
        self.assertIsNone(result["cir"])


if __name__ == "__main__":
    unittest.main()
