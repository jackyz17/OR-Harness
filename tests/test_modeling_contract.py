"""Tests for the three-stage modeling contract with verify-before-codegen (D17)."""

import asyncio
import unittest

from or_experience_bank.modeling.modeling_contract import (
    FormatValidator,
    GamsStyleSyntax,
    ModelingGate,
    SemanticValidator,
    StructuralValidator,
    parse_modeling_output,
)
from or_experience_bank.core.modeling_schemas import StructuralSignature


VALID_MODEL = """
<think>
Multi-period inventory: introduce state I[i,t], balance equation couples periods.
</think>
<model>
SETS
  i in Products
  t in Periods

PARAMETERS
  demand[i,t]
  capacity[t]
  c[i,t]
  h[i,t]

VARIABLES
  x[i,t] >= 0, continuous
  I[i,t] >= 0, continuous

OBJECTIVE
  minimize sum_{i,t} c[i,t]*x[i,t] + h[i,t]*I[i,t]

CONSTRAINTS
  C1: I[i,t] = I[i,t-1] + x[i,t] - demand[i,t]   forall i,t
  C2: sum_i x[i,t] <= capacity[t]                forall t
</model>
"""


class ParseOutputTest(unittest.TestCase):
    def test_extracts_think_and_model(self):
        parsed = parse_modeling_output(VALID_MODEL)
        self.assertIn("inventory", parsed["think"])
        self.assertIn("VARIABLES", parsed["model"])

    def test_missing_tag_returns_none(self):
        parsed = parse_modeling_output("<think>only think</think>")
        self.assertIsNone(parsed["model"])


class FormatValidatorTest(unittest.TestCase):
    def test_valid_passes(self):
        parsed = parse_modeling_output(VALID_MODEL)
        report = FormatValidator().validate(parsed["think"], parsed["model"])
        self.assertTrue(report.passed, [i.to_dict() for i in report.issues])

    def test_missing_model_tag(self):
        report = FormatValidator().validate("think", None)
        self.assertFalse(report.passed)
        self.assertEqual(report.issues[0].type, "missing_tag")

    def test_missing_block(self):
        parsed = parse_modeling_output(VALID_MODEL)
        model = parsed["model"].replace("OBJECTIVE", "OBJ_REMOVED")
        report = FormatValidator().validate(parsed["think"], model)
        self.assertFalse(report.passed)
        types = {i.detail for i in report.issues}
        self.assertTrue(any("OBJECTIVE" in detail for detail in types))


class StructuralValidatorTest(unittest.TestCase):
    def test_valid_passes(self):
        parsed = parse_modeling_output(VALID_MODEL)
        report = StructuralValidator().validate(parsed["model"])
        undefined = [i for i in report.issues if i.type == "undefined_symbol"]
        self.assertFalse(undefined, [i.to_dict() for i in undefined])

    def test_undefined_symbol_flagged(self):
        parsed = parse_modeling_output(VALID_MODEL)
        model = parsed["model"] + "\n  C3: y[i,t] <= 5   forall i,t\n"
        report = StructuralValidator().validate(model)
        self.assertFalse(report.passed)
        self.assertTrue(any("y" in i.detail for i in report.issues))

    def test_index_vars_allowed_when_sets_declared(self):
        parsed = parse_modeling_output(VALID_MODEL)
        # i, t appear as indices; they must NOT be flagged (sets declared).
        report = StructuralValidator().validate(parsed["model"])
        flagged = {i.detail for i in report.issues if i.type == "undefined_symbol"}
        self.assertFalse(any("'i'" in d or "'t'" in d for d in flagged))

    def test_signature_mismatch_binary(self):
        parsed = parse_modeling_output(VALID_MODEL)
        model = parsed["model"].replace("x[i,t] >= 0, continuous", "x[i,t] binary")
        signature = StructuralSignature(decision=["continuous_flow"])
        report = StructuralValidator().validate(model, signature)
        self.assertTrue(any(i.type == "signature_mismatch" for i in report.issues))

    def test_signature_consistent_multi_index(self):
        parsed = parse_modeling_output(VALID_MODEL)
        signature = StructuralSignature(decision=["continuous_flow", "multi_index_2d"])
        report = StructuralValidator().validate(parsed["model"], signature)
        self.assertFalse(any(i.type == "signature_mismatch" for i in report.issues))


class SemanticValidatorTest(unittest.TestCase):
    def test_no_llm_is_noop(self):
        report = asyncio.run(SemanticValidator(None).validate("problem", "model"))
        self.assertTrue(report.passed)

    def test_llm_reports_missing_constraint(self):
        class _FakeLLM:
            async def generate_object(self, prompt):
                return [{"type": "missing_constraint", "detail": "no capacity limit"}]

        report = asyncio.run(SemanticValidator(_FakeLLM()).validate("problem", "model"))
        self.assertFalse(report.passed)
        self.assertEqual(report.issues[0].layer, "semantic")

    def test_llm_clean_model_passes(self):
        class _FakeLLM:
            async def generate_object(self, prompt):
                return []

        report = asyncio.run(SemanticValidator(_FakeLLM()).validate("problem", "model"))
        self.assertTrue(report.passed)


class ModelingGateTest(unittest.TestCase):
    def test_full_valid_passes_static(self):
        report = ModelingGate().check_static("inventory problem", VALID_MODEL)
        self.assertTrue(report.passed, [i.to_dict() for i in report.issues])

    def test_format_failure_short_circuits(self):
        report = ModelingGate().check_static("problem", "<think>x</think>")
        self.assertFalse(report.passed)

    def test_structural_failure_blocks_codegen(self):
        bad = VALID_MODEL.replace("</model>", "  C9: zzz <= 1\n</model>")
        report = ModelingGate().check_static("problem", bad)
        self.assertFalse(report.passed)

    def test_full_check_with_semantic(self):
        class _FakeLLM:
            async def generate_object(self, prompt):
                return []

        gate = ModelingGate(semantic_validator=SemanticValidator(_FakeLLM()))
        report = asyncio.run(gate.check("problem", VALID_MODEL))
        self.assertTrue(report.passed, [i.to_dict() for i in report.issues])


class GamsStyleSyntaxTest(unittest.TestCase):
    def test_index_dim_counted(self):
        syntax = GamsStyleSyntax()
        parsed = syntax.split_blocks(parse_modeling_output(VALID_MODEL)["model"])
        declared = syntax.declared_symbols(parsed)
        self.assertEqual(declared["x"]["index_dim"], 2)
        self.assertEqual(declared["capacity"]["index_dim"], 1)
        self.assertEqual(declared["i"]["kind"], "set")

    def test_vtype_parsed(self):
        syntax = GamsStyleSyntax()
        parsed = syntax.split_blocks(parse_modeling_output(VALID_MODEL)["model"])
        declared = syntax.declared_symbols(parsed)
        self.assertEqual(declared["x"]["vtype"], "continuous")

    def test_set_members_registered(self):
        syntax = GamsStyleSyntax()
        parsed = syntax.split_blocks("SETS\n  i in Animals = {cow, sheep, chicken}")
        declared = syntax.declared_symbols(parsed)
        self.assertEqual(declared["cow"]["kind"], "set_member")
        self.assertEqual(declared["chicken"]["kind"], "set_member")

    def test_literal_index_resolves_to_member(self):
        model = (
            "SETS\n  i in Animals = {cow, sheep, chicken}\n"
            "PARAMETERS\n  manure[i]\n  cap\n"
            "VARIABLES\n  x[i] >= 0, continuous\n"
            "OBJECTIVE\n  maximize sum_i manure[i] * x[i]\n"
            "CONSTRAINTS\n  C1: x['cow'] <= cap\n"
        )
        report = StructuralValidator().validate(model)
        undefined = [i for i in report.issues if i.type == "undefined_symbol"]
        self.assertFalse(undefined, [i.to_dict() for i in undefined])


if __name__ == "__main__":
    unittest.main()
