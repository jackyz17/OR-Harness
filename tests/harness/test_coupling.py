"""Tests for the Coupling-Aware Intermediate Representation (CIR).

Covers:
- schema round-trip (to_dict / from_dict / to_json)
- the shape gate: nested / typo / bad type / empty / scalar / non-object
- L1 format validation (empty names, duplicates, bad evidence)
- L2 referential integrity (dangling source/target/member/resource)
- domain-generality (unknown kinds accepted, no hard-coded enum)
- structural inference: co-occurrence → depends_on only (never semantic)
- semantic upgrade: capacity keyword + resource-kind entity → uses_resource
- shared_bottleneck derivation
- the merged `orx profile` analysis output (CIR + guidance + profile)
"""
import json
import os
import unittest

from helpers import HarnessTestCase

from or_harness.core.coupling import (
    CIRFormatError,
    CouplingAwareIR,
    Entity,
    Decision,
    ConstraintRef,
    Relation,
    CouplingGroup,
    CouplingIssue,
    cir_from_task,
    cir_shape_problems,
    coerce_cir,
    validate_cir,
    infer_structural_relations,
    derive_coupling_groups,
    render_modeling_guidance,
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

    def test_round_trip_empty_survives_a_strict_input_gate(self):
        """The internal deserializer must not reject what to_dict produces.

        ``allow_empty`` is False at the INPUT boundary and True at the
        deserializer: an empty CIR is a value ``to_dict`` can legitimately
        produce, so a round trip that enforced the input policy would break.
        """
        d = CouplingAwareIR().to_dict()
        self.assertEqual(CouplingAwareIR.from_dict(d).to_dict(), d)
        self.assertEqual(CouplingAwareIR.from_dict(d, allow_empty=True)
                         .to_dict(), d)

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


class TestCIRShapeGate(HarnessTestCase):
    """The six failure modes that used to parse into an EMPTY CIR silently.

    Before the gate, ``{"coupling": {"cir": {...}}}`` and
    ``{"coupling": {"entites": [...]}}`` both produced a zero-entity CIR with
    no error, so a malformed problem was indistinguishable from an uncoupled
    one and the whole episode ran in the ``[unknown]`` cell.
    """

    #: FM1 — the nesting mistake we actually hit.
    NESTED = {"cir": {"entities": [{"name": "M1"}]}}
    #: FM2 — a typo.
    TYPO = {"entites": [{"name": "M1"}]}
    #: FM3 — a dict where a list belongs.
    BAD_TYPE = {"entities": {"M1": {}}}
    #: FM4 — an empty CIR.
    EMPTY = {}
    #: FM5 — scalar coupling in the CIR slot.
    SCALAR = {"resource_coupling": 0.9}
    #: FM6 — not an object at all.
    NOT_OBJECT = "just a string"

    def _cause(self, payload, **kwargs):
        with self.assertRaises(CIRFormatError) as caught:
            CouplingAwareIR.from_dict(payload, allow_empty=False, **kwargs)
        return caught.exception

    def test_fm1_nested_is_named_with_a_repair_hint(self):
        exc = self._cause(self.NESTED)
        self.assertEqual(exc.kind, "unknown_keys")
        self.assertEqual(exc.ctx["unknown_keys"], ["cir"])
        self.assertIn("NESTED", exc.hint)
        self.assertIn("DIRECTLY", exc.hint)

    def test_fm2_typo_lists_the_allowed_keys(self):
        exc = self._cause(self.TYPO)
        self.assertEqual(exc.kind, "unknown_keys")
        self.assertEqual(exc.ctx["unknown_keys"], ["entites"])
        self.assertIn("entities", exc.hint)

    def test_fm3_bad_type_names_the_key(self):
        exc = self._cause(self.BAD_TYPE)
        self.assertEqual(exc.kind, "bad_type")
        self.assertEqual(exc.ctx["key"], "entities")
        self.assertIn("list", exc.detail)

    def test_fm4_empty_cir_is_refused(self):
        exc = self._cause(self.EMPTY)
        self.assertEqual(exc.kind, "empty_cir")

    def test_fm4_empty_cir_allowed_when_asked(self):
        cir = CouplingAwareIR.from_dict(self.EMPTY, allow_empty=True)
        self.assertEqual(cir.entities, [])

    def test_fm5_scalar_is_redirected_to_annotations(self):
        exc = self._cause(self.SCALAR)
        self.assertEqual(exc.kind, "unknown_keys")
        self.assertIn("annotations.coupling", exc.hint)

    def test_fm6_non_object(self):
        exc = self._cause(self.NOT_OBJECT)
        self.assertEqual(exc.kind, "not_object")

    def test_error_str_is_actionable_without_the_object(self):
        """A caller that only logs the message still gets the repair hint."""
        message = str(self._cause(self.NESTED))
        self.assertIn("Unknown CIR key(s)", message)
        self.assertIn("NESTED", message)

    def test_structural_problems_are_never_downgraded(self):
        """OR_CIR_STRICT=0 downgrades POLICY, never STRUCTURE: a payload that
        cannot be deserialized has no lenient reading."""
        self._cause(self.NOT_OBJECT, strict=False)
        self._cause(self.BAD_TYPE, strict=False)

    def test_legacy_extra_keys_downgrade_to_lints(self):
        problems = cir_shape_problems(self.NESTED, strict=False)
        self.assertTrue(problems)
        self.assertTrue(all(p["severity"] == "lint" for p in problems))
        # The payload now loads (as an empty CIR) instead of raising.
        cir = CouplingAwareIR.from_dict(self.NESTED, strict=False,
                                       allow_empty=True)
        self.assertEqual(cir.entities, [])

    def test_shape_problems_never_raise(self):
        for payload in (self.NESTED, self.TYPO, self.BAD_TYPE, self.EMPTY,
                        self.SCALAR, self.NOT_OBJECT):
            self.assertIsInstance(cir_shape_problems(payload), list)

    def test_strict_env_var_downgrades_the_policy(self):
        self.addCleanup(os.environ.pop, "OR_CIR_STRICT", None)
        os.environ["OR_CIR_STRICT"] = "0"
        cir = CouplingAwareIR.from_dict(self.TYPO, allow_empty=True)
        self.assertEqual(cir.entities, [])

    def test_cir_from_task_distinguishes_absent_from_malformed(self):
        # Absent: a normal state.
        self.assertIsNone(cir_from_task({"task_id": "t1"}))
        self.assertIsNone(cir_from_task({"task_id": "t1", "coupling": None}))
        # Present and well-formed.
        cir = cir_from_task({"coupling": {"entities": [{"name": "E1"}]}})
        self.assertEqual(len(cir.entities), 1)
        # Present and malformed: an error, NOT 'no CIR'.
        with self.assertRaises(CIRFormatError):
            cir_from_task({"coupling": "not_a_dict"})

    def test_coerce_cir_normalizes_both_input_forms(self):
        raw = {"entities": [{"name": "E1"}], "decisions": [{"name": "x"}]}
        parsed = CouplingAwareIR.from_dict(raw)
        self.assertIs(coerce_cir(parsed), parsed)     # already parsed
        self.assertIsNone(coerce_cir(None))
        self.assertEqual(coerce_cir(raw).to_dict(), parsed.to_dict())
        with self.assertRaises(CIRFormatError):
            coerce_cir(self.SCALAR)


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

    def _run_profile_cli(self, task, extra=None):
        import json
        from or_harness.cli import main
        import io, contextlib, tempfile
        with tempfile.TemporaryDirectory() as home:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                code = main(["--home", home, "profile",
                             "--task", json.dumps(task), *(extra or [])])
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
        """The model is a POST-strategy artifact: it is verified and its
        coupling is reported as a DIAGNOSTIC, never used as the key."""
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
        derivation = out["result"]["derivation"]
        # The model was verified and its own reading is reported...
        self.assertIn("model_verification", derivation)
        self.assertIn("model_coupling", derivation)
        # ...but the KEY comes from the CIR, and no cross-check warning
        # channel exists any more.
        self.assertEqual(derivation["resource_coupling"]["origin"], "cir")
        self.assertNotIn("cir_warnings", out["result"]["coupling"])

    def test_bad_coupling_data_is_refused(self):
        """A non-object coupling field is a precondition failure (exit 2),
        not a silent 'no CIR'."""
        code, out = self._run_profile_cli(
            {"task_id": "t1", "family": "f", "coupling": "not_a_dict"})
        self.assertEqual(code, 2)
        error = out["result"]["error"]
        self.assertEqual(error["kind"], "cir_format")
        self.assertEqual(error["cause"], "not_object")

    def test_nested_cir_is_named_and_refused(self):
        """The failure we actually hit: the CIR nested under 'cir'."""
        code, out = self._run_profile_cli(
            {"task_id": "t1", "family": "f",
             "coupling": {"cir": {"entities": [{"name": "M1"}]}}})
        self.assertEqual(code, 2)
        error = out["result"]["error"]
        self.assertEqual(error["cause"], "unknown_keys")
        self.assertEqual(error["unknown_keys"], ["cir"])
        self.assertIn("NESTED", error["hint"])

    def test_scalar_coupling_in_the_cir_slot_is_redirected(self):
        code, out = self._run_profile_cli(
            {"task_id": "t1", "family": "f",
             "coupling": {"resource_coupling": 0.9}})
        self.assertEqual(code, 2)
        error = out["result"]["error"]
        self.assertEqual(error["cause"], "unknown_keys")
        self.assertIn("annotations.coupling", error["hint"])

    def test_empty_cir_is_refused_unless_allowed(self):
        task = {"task_id": "t1", "family": "f", "coupling": {}}
        code, out = self._run_profile_cli(task)
        self.assertEqual(code, 2)
        self.assertEqual(out["result"]["error"]["cause"], "empty_cir")
        code, out = self._run_profile_cli(task, extra=["--allow-empty-cir"])
        self.assertEqual(code, 0)
        health = out["result"]["coupling"]["health"]
        self.assertTrue(health["present"])
        self.assertFalse(health["parsed"])
        self.assertFalse(health["contributes_scalars"])

    def test_health_reports_a_cir_that_contributes(self):
        task = {
            "task_id": "t1", "family": "scheduling",
            "coupling": {
                "entities": [{"name": "R1", "kind": "resource"}],
                "decisions": [{"name": "x"}, {"name": "y"}],
                "relations": [
                    {"source": "x", "target": "R1",
                     "type": "uses_resource", "evidence": "semantic"},
                    {"source": "y", "target": "R1",
                     "type": "uses_resource", "evidence": "semantic"},
                ],
            },
        }
        code, out = self._run_profile_cli(task)
        self.assertEqual(code, 0)
        health = out["result"]["coupling"]["health"]
        self.assertTrue(health["parsed"])
        self.assertTrue(health["contributes_scalars"])
        self.assertEqual(health["decisions"], 2)

    def test_no_coupling_field_reports_no_cir_health(self):
        code, out = self._run_profile_cli(
            {"task_id": "t1", "family": "routing"})
        self.assertEqual(code, 0)
        health = out["result"]["coupling"]["health"]
        self.assertFalse(health["present"])
        self.assertIn("no CIR supplied", health["note"])


if __name__ == "__main__":
    unittest.main()
