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
- the merged `orx profile` analysis output (CIR + guidance + profile)
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
    coupling_from_cir,
)
from or_harness.profiling.model_syntax import parse_model
from or_harness.api import ORHarness


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

    def test_no_upgrade_with_demand_keyword_only(self):
        """'demand' is a consumption requirement, not capacity evidence."""
        cir = CouplingAwareIR(
            entities=[Entity(name="R1", kind="resource")],
            decisions=[Decision(name="x")],
            constraints=[ConstraintRef(id="C1", kind="other",
                                       expr="sum(x) >= demand_R1")],
            relations=[Relation(source="x", target="R1", type="depends_on",
                                evidence="structural")],
        )
        infer_structural_relations(cir, parsed_model=None)
        upgraded = [r for r in cir.relations if r.type == "uses_resource"]
        self.assertEqual(upgraded, [])

    def test_no_upgrade_when_constraint_not_mentioning_resource(self):
        """Capacity keyword + source present, but the constraint never
        references the target resource → no upgrade (the constraint is not
        evidence that THIS resource is coupled)."""
        cir = CouplingAwareIR(
            entities=[Entity(name="R1", kind="resource"),
                      Entity(name="R2", kind="resource")],
            decisions=[Decision(name="x")],
            constraints=[ConstraintRef(id="C1", kind="capacity",
                                       expr="sum(x) <= limit_of_R2")],
            relations=[Relation(source="x", target="R1", type="depends_on",
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
        """CIR declares a uses_resource edge between TWO DECISIONS that never
        co-occur in the model → flagged (both endpoints are variables)."""
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

    def test_decision_to_resource_relation_not_flagged(self):
        """The NORMAL uses_resource shape is decision→resource entity; the
        resource is not a model variable, so no variable-variable
        co-occurrence pair exists — and none should be demanded."""
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
            entities=[Entity(name="R1", kind="resource")],
            decisions=[Decision(name="x")],
            relations=[Relation(source="x", target="R1", type="uses_resource",
                                evidence="semantic")],
        )
        warnings = cross_check_cir_model(cir, model)
        self.assertEqual(warnings, [])

    def test_relation_source_unconstrained(self):
        """Decision→resource edge whose source appears in NO model constraint
        → weak check flags that the model may be missing the coupling."""
        model_text = """
SETS:
 i in Items = {a}
PARAMETERS:
 p
VARIABLES:
 x[i] continuous
OBJECTIVE:
 maximize sum(i, x[i])
"""
        model = parse_model(model_text)
        cir = CouplingAwareIR(
            entities=[Entity(name="R1", kind="resource")],
            decisions=[Decision(name="x")],
            relations=[Relation(source="x", target="R1", type="uses_resource",
                                evidence="semantic")],
        )
        warnings = cross_check_cir_model(cir, model)
        codes = [w["code"] for w in warnings]
        self.assertIn("relation_source_unconstrained", codes)

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


class TestCouplingFromCIR(HarnessTestCase):
    """Scalar ProblemSignature derivation from CIR structure."""

    def test_resource_coupling_from_relations(self):
        cir = CouplingAwareIR(
            entities=[Entity(name="R1", kind="resource")],
            decisions=[Decision(name=f"x{i}") for i in range(4)],
            relations=[
                Relation(source=f"x{i}", target="R1", type="uses_resource",
                         evidence="semantic") for i in range(3)
            ],
        )
        dims = coupling_from_cir(cir)
        self.assertEqual(dims["resource_coupling"], 0.75)
        self.assertEqual(dims["temporal_coupling"], 0.0)
        self.assertIsNone(dims["semantic_coupling"])

    def test_temporal_and_route_from_indexes(self):
        cir = CouplingAwareIR(
            decisions=[Decision(name="x", indexes=["i", "t"]),
                       Decision(name="y", indexes=["arc"]),
                       Decision(name="z", indexes=["j"])],
        )
        dims = coupling_from_cir(cir)
        self.assertAlmostEqual(dims["temporal_coupling"], 1 / 3, places=4)
        self.assertAlmostEqual(dims["route_complexity"], 1 / 3, places=4)

    def test_single_letter_index_matches_by_equality_only(self):
        """Index 'route' must not count as temporal via the letter 't'."""
        cir = CouplingAwareIR(
            decisions=[Decision(name="x", indexes=["route"])],
        )
        dims = coupling_from_cir(cir)
        self.assertEqual(dims["temporal_coupling"], 0.0)
        self.assertEqual(dims["route_complexity"], 1.0)

    def test_empty_decisions_all_none(self):
        cir = CouplingAwareIR()
        dims = coupling_from_cir(cir)
        self.assertIsNone(dims["resource_coupling"])
        self.assertIsNone(dims["temporal_coupling"])
        self.assertIsNone(dims["route_complexity"])


class TestProfileWiring(HarnessTestCase):
    """CIR participates in the retrieval signature: recall/execute derive
    scalar dims from the task's coupling field (not a parallel model-only
    line)."""

    def test_profile_derives_from_task_coupling_field(self):
        h = ORHarness(home=self.home)
        try:
            task = {
                "task_id": "t_wire", "family": "scheduling",
                "coupling": {
                    "entities": [{"name": "R1", "kind": "resource"}],
                    "decisions": [{"name": "x"}, {"name": "y"},
                                  {"name": "z"}, {"name": "w"}],
                    "relations": [
                        {"source": "x", "target": "R1", "type": "uses_resource",
                         "evidence": "semantic"},
                        {"source": "y", "target": "R1", "type": "uses_resource",
                         "evidence": "semantic"},
                    ],
                },
                "spec": {"n_vars": 10},
            }
            profile = h.profile(task)
            self.assertEqual(profile.resource_coupling, 0.5)
            report = profile.annotations["profiling"]
            self.assertEqual(report["origin"]["resource_coupling"], "cir")
        finally:
            h.close()

    def test_recall_uses_cir_signature(self):
        h = ORHarness(home=self.home)
        try:
            task = {
                "task_id": "t_wire2", "family": "scheduling",
                "coupling": {
                    "entities": [{"name": "R1", "kind": "resource"}],
                    "decisions": [{"name": "x"}, {"name": "y"}],
                    "relations": [
                        {"source": "x", "target": "R1", "type": "uses_resource",
                         "evidence": "semantic"},
                        {"source": "y", "target": "R1", "type": "uses_resource",
                         "evidence": "semantic"},
                    ],
                },
                "spec": {"n_vars": 10},
            }
            result = h.recall(task)
            self.assertEqual(result["profile"]["resource_coupling"], 1.0)
        finally:
            h.close()


class TestProfileAnalysisEntry(HarnessTestCase):
    """The merged `orx profile` analysis output: CIR + guidance + profile
    in one call (the understand entry was consolidated into profile)."""

    def _run_profile_cli(self, task):
        import json
        from or_harness.cli import main
        import io, contextlib, tempfile, os
        with tempfile.TemporaryDirectory() as home:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                code = main(["--home", home, "profile",
                             "--task", json.dumps(task)])
            out = json.loads(buf.getvalue())
        return code, out

    def test_no_coupling_field(self):
        code, out = self._run_profile_cli(
            {"task_id": "t1", "family": "routing"})
        self.assertEqual(code, 0)
        result = out["result"]
        self.assertIsNone(result["coupling"]["cir"])
        self.assertIn("message", result["coupling"])
        # The profile itself is still produced.
        self.assertEqual(result["profile"]["family"], "routing")

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
        code, out = self._run_profile_cli(task)
        self.assertEqual(code, 0)
        coupling = out["result"]["coupling"]
        cir = coupling["cir"]
        self.assertIsNotNone(cir)
        self.assertEqual(len(cir["entities"]), 1)
        uses = [r for r in cir["relations"] if r["type"] == "uses_resource"]
        self.assertEqual(len(uses), 2)
        groups = [g for g in cir["coupling_groups"]
                  if g["type"] == "shared_bottleneck"]
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["resource"], "R1")
        guidance = coupling["modeling_guidance"]
        self.assertEqual(len(guidance), 1)
        self.assertIn("aggregate", guidance[0]["implication"].lower())
        # CIR-derived coupling feeds the profile derivation.
        self.assertIsNotNone(out["result"]["profile"]["resource_coupling"])

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
        code, out = self._run_profile_cli(task)
        self.assertEqual(code, 0)
        codes = [w["code"] for w in out["result"]["coupling"]["cir_warnings"]]
        self.assertIn("cir_decision_not_in_model", codes)

    def test_bad_coupling_data(self):
        code, out = self._run_profile_cli(
            {"task_id": "t1", "family": "f", "coupling": "not_a_dict"})
        self.assertEqual(code, 0)
        self.assertIsNone(out["result"]["coupling"]["cir"])


if __name__ == "__main__":
    unittest.main()
